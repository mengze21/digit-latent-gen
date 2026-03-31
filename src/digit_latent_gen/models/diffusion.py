from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim < 1:
            raise ValueError("Embedding dimension must be at least 2.")

        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time.float()[:, None] * frequencies[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class LatentResidualBlock(nn.Module):
    """Residual MLP block used by the latent denoiser."""

    def __init__(
        self,
        hidden_dim: int,
        time_emb_dim: int,
        class_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.class_proj = (
            nn.Linear(class_emb_dim, hidden_dim) if class_emb_dim is not None else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        class_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = h + self.time_proj(self.act(time_emb))
        if self.class_proj is not None and class_emb is not None:
            h = h + self.class_proj(self.act(class_emb))
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


class LatentDenoiser(nn.Module):
    """Conditional denoiser for latent diffusion."""

    def __init__(
        self,
        latent_dim: int,
        time_emb_dim: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        num_classes: int = 10,
        class_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim or max(4 * latent_dim, 128)
        self.num_layers = num_layers

        class_emb_dim = class_emb_dim or time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.input_proj = nn.Linear(latent_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LatentResidualBlock(
                    hidden_dim=self.hidden_dim,
                    time_emb_dim=time_emb_dim,
                    class_emb_dim=class_emb_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"LatentDenoiser expects 2D latent tensors shaped [batch, latent_dim], got {tuple(x.shape)}."
            )

        time_emb = self.time_mlp(time)
        class_emb = self.class_emb(class_labels) if class_labels is not None else None

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, time_emb, class_emb)

        h = self.out_norm(h)
        noise_pred = self.out_proj(h)

        if latent_z is not None:
            # Optional auxiliary conditioning. Keep it lightweight and additive so the
            # model remains a true latent diffusion model even when this is unused.
            if latent_z.ndim != 2:
                raise ValueError(
                    f"latent_z must be a 2D tensor when provided, got {tuple(latent_z.shape)}."
                )
            if latent_z.shape[0] != x.shape[0]:
                raise ValueError("latent_z batch size must match x batch size.")

        return noise_pred


class DiffusionModel(nn.Module):
    """Conditional latent diffusion model for digit generation."""

    def __init__(
        self,
        latent_dim: int = 32,
        time_steps: int = 1000,
        time_emb_dim: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        num_classes: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim or max(4 * latent_dim, 128)
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = torch.device(device)

        self.denoiser = LatentDenoiser(
            latent_dim=latent_dim,
            time_emb_dim=time_emb_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        betas = torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(
        self, values: torch.Tensor, timesteps: torch.Tensor, x_shape: Sequence[int]
    ) -> torch.Tensor:
        gathered = values.gather(0, timesteps)
        view_shape = (timesteps.shape[0],) + (1,) * (len(x_shape) - 1)
        return gathered.view(view_shape)

    @property
    def betas_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.betas)

    @property
    def sqrt_alphas_cumprod_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.sqrt_alphas_cumprod)

    @property
    def sqrt_one_minus_alphas_cumprod_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.sqrt_one_minus_alphas_cumprod)

    @property
    def sqrt_recip_alphas_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.sqrt_recip_alphas)

    @property
    def posterior_variance_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.posterior_variance)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process in latent space."""
        if x_start.ndim != 2:
            raise ValueError(
                f"Latent diffusion expects x_start shaped [batch, latent_dim], got {tuple(x_start.shape)}."
            )

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod_tensor, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod_tensor, t, x_start.shape
        )

        x_noisy = (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return x_noisy, noise

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.denoiser(x_t, t, class_labels=class_labels, latent_z=latent_z)

    def compute_loss(
        self,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the standard diffusion noise-prediction objective."""
        if x_start.ndim != 2:
            raise ValueError(
                f"Latent diffusion expects x_start shaped [batch, latent_dim], got {tuple(x_start.shape)}."
            )

        if t is None:
            t = torch.randint(
                0, self.time_steps, (x_start.shape[0],), device=x_start.device
            ).long()
        else:
            t = t.to(device=x_start.device, dtype=torch.long)

        if class_labels is not None:
            class_labels = class_labels.to(device=x_start.device, dtype=torch.long)
        if latent_z is not None:
            latent_z = latent_z.to(device=x_start.device, dtype=x_start.dtype)

        if noise is not None and noise.shape != x_start.shape:
            raise ValueError("noise must have the same shape as x_start.")

        x_noisy, noise = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.predict_noise(
            x_noisy, t, class_labels=class_labels, latent_z=latent_z
        )
        return F.mse_loss(predicted_noise, noise)

    def forward(
        self,
        x_start: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_loss(x_start, class_labels=class_labels, latent_z=latent_z)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One reverse diffusion step."""
        if noise is None:
            noise = torch.randn_like(x_t)

        predicted_noise = self.predict_noise(
            x_t, t, class_labels=class_labels, latent_z=latent_z
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_tensor, t, x_t.shape)
        betas_t = self._extract(self.betas_tensor, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod_tensor, t, x_t.shape
        )

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = self._extract(
            self.posterior_variance_tensor, t, x_t.shape
        )
        nonzero_mask = (t != 0).float().view((t.shape[0],) + (1,) * (x_t.ndim - 1))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(
        self,
        shape: Sequence[int],
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Run the full reverse process and return latent samples."""
        sample_device = (
            torch.device(device) if device is not None else self.betas_tensor.device
        )
        x = torch.randn(*shape, device=sample_device)
        if class_labels is not None:
            class_labels = class_labels.to(device=sample_device, dtype=torch.long)
        if latent_z is not None:
            latent_z = latent_z.to(device=sample_device, dtype=x.dtype)

        for timestep in reversed(range(self.time_steps)):
            t = torch.full(
                (shape[0],), timestep, device=sample_device, dtype=torch.long
            )
            step_noise = torch.randn_like(x) if timestep > 0 else torch.zeros_like(x)
            x = self.p_sample(
                x, t, class_labels=class_labels, latent_z=latent_z, noise=step_noise
            )

        return x

    def generate(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        latent_z: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Generate latent samples for downstream VAE decoding."""
        if class_labels is None:
            class_labels = torch.zeros(
                num_samples, dtype=torch.long, device=self.betas_tensor.device
            )
        elif class_labels.ndim == 0:
            class_labels = class_labels.repeat(num_samples)

        if class_labels.shape[0] != num_samples:
            raise ValueError("class_labels batch size must match num_samples.")

        if latent_z is not None and latent_z.shape[0] != num_samples:
            raise ValueError("latent_z batch size must match num_samples.")

        return self.p_sample_loop(
            shape=(num_samples, self.latent_dim),
            class_labels=class_labels.to(self.betas_tensor.device),
            latent_z=(
                latent_z.to(self.betas_tensor.device) if latent_z is not None else None
            ),
            device=device,
        )


# Backward-compatible aliases for older imports.
Block = LatentResidualBlock
UNet = LatentDenoiser
