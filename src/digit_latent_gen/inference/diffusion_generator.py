from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digit_latent_gen.models.diffusion import DiffusionModel
from digit_latent_gen.models.vae import VAE


LabelInput = Union[int, Sequence[int], torch.Tensor]


class DiffusionGenerator:
    """Generate digit images with a latent diffusion model plus a VAE decoder."""

    def __init__(
        self,
        latent_dim: int,
        label: LabelInput,
        num_classes: int,
        vae_checkpoint_path: Union[str, Path],
        diffusion_checkpoint_path: Union[str, Path],
        output_dir: Union[str, Path],
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        diffusion_config: Optional[dict] = None,
    ):
        self.latent_dim = latent_dim
        self.label = label
        self.num_classes = num_classes
        self.vae_checkpoint_path = Path(vae_checkpoint_path)
        self.diffusion_checkpoint_path = Path(diffusion_checkpoint_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.diffusion_config = diffusion_config or {}

        self._vae: Optional[VAE] = None
        self._diffusion: Optional[DiffusionModel] = None

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

    def _load_compatible_state_dict(self, model, checkpoint) -> None:
        state_dict = self._extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a valid state dict.")

        model_state = model.state_dict()
        cleaned_state = {}
        for key, value in state_dict.items():
            cleaned_key = key.removeprefix("module.")
            if cleaned_key in model_state and model_state[cleaned_key].shape == value.shape:
                cleaned_state[cleaned_key] = value

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint partially matched the current model architecture. "
                "Please verify that the checkpoint belongs to the expected VAE or diffusion model."
            )

    def _load_vae(self) -> VAE:
        if self._vae is not None:
            return self._vae

        if not self.vae_checkpoint_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {self.vae_checkpoint_path}")

        model = VAE(latent_dim=self.latent_dim, num_classes=self.num_classes).to(self.device)
        checkpoint = torch.load(self.vae_checkpoint_path, map_location=self.device)
        self._load_compatible_state_dict(model, checkpoint)
        model.eval()
        self._vae = model
        return model

    def _load_diffusion(self) -> DiffusionModel:
        if self._diffusion is not None:
            return self._diffusion

        if not self.diffusion_checkpoint_path.exists():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {self.diffusion_checkpoint_path}")

        model = DiffusionModel(
            latent_dim=self.latent_dim,
            time_steps=int(self.diffusion_config.get("time_steps", 1000)),
            hidden_dim=self.diffusion_config.get("hidden_dim"),
            num_layers=int(self.diffusion_config.get("num_layers", 4)),
            num_classes=self.num_classes,
            device=self.device,
        ).to(self.device)
        checkpoint = torch.load(self.diffusion_checkpoint_path, map_location=self.device)
        self._load_compatible_state_dict(model, checkpoint)
        model.eval()
        self._diffusion = model
        return model

    def _normalize_labels(self, label: Optional[LabelInput], num_samples: int) -> torch.Tensor:
        label_value = self.label if label is None else label

        if isinstance(label_value, torch.Tensor):
            labels = label_value.to(device=self.device, dtype=torch.long)
            if labels.ndim == 0:
                labels = labels.repeat(num_samples)
        elif isinstance(label_value, Iterable) and not isinstance(label_value, (str, bytes)):
            labels = torch.tensor(list(label_value), device=self.device, dtype=torch.long)
        else:
            labels = torch.full((num_samples,), int(label_value), device=self.device, dtype=torch.long)

        if labels.numel() == 1 and num_samples > 1:
            labels = labels.repeat(num_samples)

        if labels.numel() != num_samples:
            raise ValueError(
                f"Number of labels ({labels.numel()}) must match num_samples ({num_samples})."
            )

        return labels

    def _sample_latents(self, num_samples: int, labels: torch.Tensor) -> torch.Tensor:
        diffusion = self._load_diffusion()
        with torch.no_grad():
            return diffusion.generate(num_samples=num_samples, class_labels=labels, device=self.device)

    def generate(self, num_samples: Optional[int] = None, label: Optional[LabelInput] = None) -> torch.Tensor:
        vae = self._load_vae()
        self._load_diffusion()

        label_value = self.label if label is None else label
        if num_samples is None:
            if isinstance(label_value, torch.Tensor) and label_value.ndim > 0:
                sample_count = int(label_value.numel())
            elif isinstance(label_value, Iterable) and not isinstance(label_value, (str, bytes, int)):
                label_value = list(label_value)
                sample_count = len(label_value)
            else:
                sample_count = self.batch_size
        else:
            sample_count = num_samples

        labels = self._normalize_labels(label_value, sample_count)
        latents = self._sample_latents(sample_count, labels)

        with torch.no_grad():
            images = vae.decoder(latents, labels)

        return images.detach().cpu()

    def save_generated_images(
        self,
        images: torch.Tensor,
        output_path: Optional[Union[str, Path]] = None,
        max_images: int = 16,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_path) if output_path is not None else self.output_dir / f"diffusion_label_{self.label}.png"

        num_images = min(int(images.size(0)), max_images)
        cols = min(4, num_images)
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]
            ax.axis("off")
            if idx < num_images:
                ax.imshow(images[idx].squeeze().numpy(), cmap="gray")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def generate_and_save(
        self,
        num_samples: Optional[int] = None,
        label: Optional[LabelInput] = None,
        output_path: Optional[Union[str, Path]] = None,
        max_images: int = 16,
    ) -> Path:
        images = self.generate(num_samples=num_samples, label=label)
        return self.save_generated_images(images, output_path=output_path, max_images=max_images)
