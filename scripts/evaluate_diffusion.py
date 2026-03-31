import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from digit_latent_gen.common.utils import get_mnist_dataset
from digit_latent_gen.models.diffusion import DiffusionModel
from digit_latent_gen.models.vae import VAE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "diffusion_config.yaml"
LOGGER = logging.getLogger(__name__)
CONFIG_KEY_WIDTH = 28


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained latent diffusion checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        type=Path,
        default=None,
        help="Override VAE checkpoint path.",
    )
    parser.add_argument(
        "--diffusion-checkpoint-path",
        type=Path,
        default=None,
        help="Override diffusion checkpoint path.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override evaluation batch size."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader worker count.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device: cpu, cuda, or mps."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override evaluation output directory.",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="Override log directory."
    )
    parser.add_argument(
        "--save-generated-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable saving generated sample grids.",
    )
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_device(requested_device=None):
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8"),
        ],
        force=True,
    )


def log_config_section(title, config_dict):
    LOGGER.info("%s", title)
    for key, value in config_dict.items():
        LOGGER.info("  %-*s : %s", CONFIG_KEY_WIDTH, key, value)


def resolve_runtime_config(config, args):
    model_config = config["model"].copy()
    evaluation_config = config.get("evaluation", {}).copy()
    paths_config = config.get("paths", {}).copy()
    diffusion_config = config.get("diffusion", {}).copy()

    if args.vae_checkpoint_path is not None:
        evaluation_config["vae_checkpoint_path"] = str(args.vae_checkpoint_path)
    if args.diffusion_checkpoint_path is not None:
        evaluation_config["diffusion_checkpoint_path"] = str(
            args.diffusion_checkpoint_path
        )
    if args.batch_size is not None:
        evaluation_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        evaluation_config["num_workers"] = args.num_workers
    if args.device is not None:
        evaluation_config["device"] = args.device
    if args.output_dir is not None:
        evaluation_config["output_dir"] = str(args.output_dir)
    if args.log_dir is not None:
        paths_config["log_dir"] = str(args.log_dir)
    if args.save_generated_samples is not None:
        evaluation_config["save_generated_samples"] = args.save_generated_samples

    return {
        "model": model_config,
        "evaluation": evaluation_config,
        "diffusion": diffusion_config,
        "paths": {
            "data_dir": resolve_path(paths_config.get("data_dir", "data")),
            "output_dir": resolve_path(
                evaluation_config.get(
                    "output_dir", paths_config.get("output_dir", "outputs")
                )
            ),
            "log_dir": resolve_path(paths_config.get("log_dir", "outputs/logs")),
        },
    }


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def save_generated_grid(images: torch.Tensor, labels: torch.Tensor, output_path: Path):
    num_images = min(16, images.size(0))
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

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
            ax.imshow(images[idx].detach().cpu().squeeze().numpy(), cmap="gray")
            ax.set_title(f"Label: {labels[idx].item()}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_latent_test_loader(vae_model, test_dataset, batch_size, num_workers, device):
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    latent_batches = []
    label_batches = []

    vae_model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mu, _ = vae_model.encoder(inputs, labels)
            latent_batches.append(mu.cpu())
            label_batches.append(labels.cpu())

    latents = torch.cat(latent_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return DataLoader(
        TensorDataset(latents, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    runtime_config = resolve_runtime_config(config, args)

    model_config = runtime_config["model"]
    evaluation_config = runtime_config["evaluation"]
    diffusion_config = runtime_config["diffusion"]
    paths_config = runtime_config["paths"]

    batch_size = evaluation_config.get("batch_size", 64)
    num_workers = evaluation_config.get("num_workers", 0)
    device = get_device(evaluation_config.get("device"))
    save_generated_samples = evaluation_config.get("save_generated_samples", True)
    num_generated_samples = evaluation_config.get("num_generated_samples", 16)

    vae_checkpoint_path = resolve_path(
        evaluation_config.get("vae_checkpoint_path", "checkpoints/latest.pt")
    )
    diffusion_checkpoint_path = resolve_path(
        evaluation_config.get(
            "diffusion_checkpoint_path", "checkpoints/latest_diffusion.pt"
        )
    )
    output_dir = paths_config["output_dir"] / "diffusion_evaluation"
    log_dir = paths_config["log_dir"]
    log_filename = evaluation_config.get("log_filename", "evaluate_diffusion.log")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename
    setup_logging(log_file_path)

    LOGGER.info("Starting latent diffusion evaluation")
    LOGGER.info("Using device: %s", device)
    log_config_section("Model configuration", model_config)
    log_config_section("Evaluation configuration", evaluation_config)
    log_config_section(
        "Path configuration",
        {
            "data_dir": str(paths_config["data_dir"]),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "vae_checkpoint_path": str(vae_checkpoint_path),
            "diffusion_checkpoint_path": str(diffusion_checkpoint_path),
        },
    )

    if not vae_checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")
    if not diffusion_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {diffusion_checkpoint_path}"
        )

    test_dataset = get_mnist_dataset(
        train=False, root_dir=str(paths_config["data_dir"])
    )

    vae_model = VAE(
        latent_dim=model_config["latent_dim"],
        num_classes=model_config["num_classes"],
    ).to(device)
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    vae_model.load_state_dict(extract_model_state_dict(vae_checkpoint))
    vae_model.eval()

    diffusion_model = DiffusionModel(
        latent_dim=model_config["latent_dim"],
        time_steps=diffusion_config.get("time_steps", 1000),
        hidden_dim=diffusion_config.get("hidden_dim"),
        num_layers=diffusion_config.get("num_layers", 4),
        num_classes=model_config["num_classes"],
        device=device,
    ).to(device)
    diffusion_checkpoint = torch.load(diffusion_checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(extract_model_state_dict(diffusion_checkpoint))
    diffusion_model.eval()

    latent_test_loader = build_latent_test_loader(
        vae_model=vae_model,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    total_loss = 0.0
    total_samples = 0

    LOGGER.info("Starting evaluation on latent test set...")
    with torch.no_grad():
        for batch_idx, (latents, labels) in enumerate(latent_test_loader):
            latents = latents.to(device)
            labels = labels.to(device)
            loss = diffusion_model(latents, class_labels=labels)

            batch_size_actual = latents.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            if (batch_idx + 1) % 10 == 0:
                LOGGER.info("Processed %d batches...", batch_idx + 1)

    avg_loss = total_loss / max(total_samples, 1)
    LOGGER.info("Evaluation completed")
    LOGGER.info("Evaluation results")
    LOGGER.info("  %-*s : %.4f", CONFIG_KEY_WIDTH, "avg_diffusion_loss", avg_loss)
    LOGGER.info("  %-*s : %d", CONFIG_KEY_WIDTH, "num_test_samples", total_samples)

    if save_generated_samples:
        generated_labels = (
            torch.arange(num_generated_samples, device=device)
            % model_config["num_classes"]
        )
        generated_latents = diffusion_model.generate(
            num_samples=num_generated_samples,
            class_labels=generated_labels,
        )
        generated_images = vae_model.decoder(generated_latents, generated_labels)
        sample_path = output_dir / "generated_samples.png"
        save_generated_grid(generated_images, generated_labels.cpu(), sample_path)
        LOGGER.info("Saved generated samples to %s", sample_path)

    LOGGER.info("Latent diffusion evaluation finished successfully")


if __name__ == "__main__":
    main()
