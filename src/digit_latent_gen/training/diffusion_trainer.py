from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm


class DiffusionTrainer:
    """Train a diffusion model with conditional labels and optional latent conditioning.

    The trainer is intentionally batch-format agnostic so it can be used for:
    - image diffusion: batches of ``(images, labels)``
    - latent diffusion: batches of ``(latents, labels, latent_z)`` or dict batches

    The diffusion model is expected to return a scalar loss when called as:
    ``model(x_start, class_labels=labels, latent_z=latent_z)``.
    """

    def __init__(
        self,
        model,
        train_loader,
        device,
        learning_rate: float = 1e-4,
        scheduler_config: Optional[Mapping[str, Any]] = None,
        grad_clip_norm: Optional[float] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = self._build_scheduler(scheduler_config)

        self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        """Move the model and its diffusion schedule tensors onto the trainer device."""
        self.model.to(self.device)

        if hasattr(self.model, "device"):
            self.model.device = self.device

        # DiffusionModel stores its noise schedule as plain tensors rather than buffers.
        # Keep them aligned with the trainer device if they are present.
        for attr_name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
        ):
            value = getattr(self.model, attr_name, None)
            if torch.is_tensor(value):
                setattr(self.model, attr_name, value.to(self.device))

    def _build_scheduler(self, scheduler_config: Optional[Mapping[str, Any]]):
        if not scheduler_config:
            return None

        scheduler_type = str(scheduler_config.get("type", "")).lower()
        if not scheduler_type:
            return None

        if scheduler_type != "cosine":
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. Only 'cosine' is supported."
            )

        t_max = scheduler_config.get("t_max")
        if t_max is None:
            raise ValueError("Cosine scheduler requires 't_max' in scheduler_config.")

        eta_min = scheduler_config.get("eta_min", 0.0)
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=eta_min
        )

    def get_current_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _unpack_batch(
        self, batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Support tuple- and dict-shaped batches.

        Accepted tuple formats:
        - (data, labels)
        - (data, labels, latent_z)

        Accepted dict keys:
        - data/images/x
        - labels/class_labels/y
        - latent_z/latents/z
        """
        if isinstance(batch, Mapping):
            data = batch.get("data")
            if data is None:
                data = batch.get("images", batch.get("x"))

            labels = batch.get("labels")
            if labels is None:
                labels = batch.get("class_labels", batch.get("y"))

            latent_z = batch.get("latent_z")
            if latent_z is None:
                latent_z = batch.get("latents", batch.get("z"))

            if data is None:
                raise ValueError("Batch dictionary must contain a data/images/x entry.")
            return data, labels, latent_z

        if not isinstance(batch, tuple):
            batch = tuple(batch)

        if len(batch) == 2:
            data, labels = batch
            latent_z = None
        elif len(batch) == 3:
            data, labels, latent_z = batch
        else:
            raise ValueError(
                "Expected a batch with 2 or 3 items: (data, labels) or (data, labels, latent_z)."
            )

        return data, labels, latent_z

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute the diffusion loss for one batch."""
        data, labels, latent_z = self._unpack_batch(batch)

        data = data.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        latent_z = latent_z.to(self.device) if latent_z is not None else None

        if labels is not None and labels.dtype != torch.long:
            labels = labels.long()

        if latent_z is not None and latent_z.dtype != data.dtype:
            latent_z = latent_z.to(dtype=data.dtype)

        if labels is None:
            return self.model(data, class_labels=None, latent_z=latent_z)
        return self.model(data, class_labels=labels, latent_z=latent_z)

    def train_epoch(self, step_scheduler: bool = True):
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in tqdm(self.train_loader, desc="Training diffusion"):
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise ValueError("Diffusion model must return a scalar loss tensor.")

            if torch.isnan(loss):
                print("NaN detected in diffusion loss!")
                break

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        if self.scheduler is not None and step_scheduler:
            self.scheduler.step()

        avg_loss = total_loss / max(total_batches, 1)
        return {
            "avg_loss": avg_loss,
            "num_batches": total_batches,
        }

    def train(self, num_epochs: int, epoch_end_callback=None):
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch()
            if epoch_end_callback is not None:
                epoch_end_callback(epoch, metrics, self.get_current_learning_rate())
