from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digit_latent_gen.models.vae import VAE


LabelInput = Union[int, Sequence[int], torch.Tensor]


class Generator:
    """Generate digit images from a trained conditional VAE.

    The current model is label-conditioned, so generation from a digit label
    works by sampling a latent vector from N(0, I) and passing the latent
    sample together with the requested label(s) to the decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        label: LabelInput,
        num_classes: int,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        self.latent_dim = latent_dim
        self.label = label
        self.num_classes = num_classes
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._model: Optional[VAE] = None

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

    def _load_model(self) -> VAE:
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        model = VAE(latent_dim=self.latent_dim, num_classes=self.num_classes).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(self._extract_state_dict(checkpoint))
        model.eval()
        self._model = model
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

    def _sample_latent(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, self.latent_dim, device=self.device)

    def generate(self, num_samples: Optional[int] = None, label: Optional[LabelInput] = None) -> torch.Tensor:
        """Generate digit images conditioned on the provided label(s)."""
        model = self._load_model()
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
        latent = self._sample_latent(sample_count)

        with torch.no_grad():
            images = model.decoder(latent, labels)

        return images.detach().cpu()

    def save_generated_images(
        self,
        images: torch.Tensor,
        output_path: Optional[Union[str, Path]] = None,
        max_images: int = 16,
    ) -> Path:
        """Save a generated image grid to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_path) if output_path is not None else self.output_dir / f"generated_label_{self.label}.png"

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
