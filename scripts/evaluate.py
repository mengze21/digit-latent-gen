import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from digit_latent_gen.common.utils import get_mnist_dataset
from digit_latent_gen.models.vae import VAE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config.yaml"
LOGGER = logging.getLogger(__name__)
CONFIG_KEY_WIDTH = 24


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VAE checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override evaluation output directory.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Override log directory.")
    parser.add_argument(
        "--save-reconstructions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable reconstruction image saving.",
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
    testing_config = config.get("testing", {}).copy()
    paths_config = config.get("paths", {}).copy()

    if args.batch_size is not None:
        testing_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        testing_config["num_workers"] = args.num_workers
    if args.device is not None:
        testing_config["device"] = args.device
    if args.save_reconstructions is not None:
        testing_config["save_reconstructions"] = args.save_reconstructions
    if args.checkpoint_path is not None:
        testing_config["checkpoint_path"] = str(args.checkpoint_path)
    if args.output_dir is not None:
        testing_config["output_dir"] = str(args.output_dir)
    if args.log_dir is not None:
        paths_config["log_dir"] = str(args.log_dir)

    runtime_config = {
        "model": model_config,
        "testing": testing_config,
        "paths": {
            "data_dir": resolve_path(paths_config.get("data_dir", "data")),
            "output_dir": resolve_path(
                testing_config.get("output_dir", paths_config.get("output_dir", "outputs"))
            ),
            "log_dir": resolve_path(paths_config.get("log_dir", "outputs/logs")),
        },
    }
    return runtime_config


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def compute_vae_loss(inputs, reconstructions, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(reconstructions, inputs, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


def save_reconstruction_grid(inputs, reconstructions, labels, output_path):
    num_images = min(8, inputs.size(0))
    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))

    for idx in range(num_images):
        axes[0, idx].imshow(inputs[idx].detach().cpu().squeeze().numpy(), cmap="gray")
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"Label: {labels[idx].item()}")

        axes[1, idx].imshow(reconstructions[idx].detach().cpu().squeeze().numpy(), cmap="gray")
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    runtime_config = resolve_runtime_config(config, args)

    model_config = runtime_config["model"]
    testing_config = runtime_config["testing"]
    paths_config = runtime_config["paths"]

    checkpoint_path = resolve_path(testing_config.get("checkpoint_path", "checkpoints/latest.pt"))
    output_dir = paths_config["output_dir"] / "evaluation"
    log_dir = paths_config["log_dir"]
    log_filename = testing_config.get("log_filename", "evaluate.log")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename
    setup_logging(log_file_path)

    batch_size = testing_config.get("batch_size", 64)
    num_workers = testing_config.get("num_workers", 0)
    save_reconstructions = testing_config.get("save_reconstructions", True)
    device = get_device(testing_config.get("device"))

    LOGGER.info("Starting VAE evaluation")
    LOGGER.info("Using device: %s", device)
    log_config_section("Model configuration", model_config)
    log_config_section("Testing configuration", testing_config)
    log_config_section(
        "Path configuration",
        {
            "data_dir": str(paths_config["data_dir"]),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "checkpoint_path": str(checkpoint_path),
        },
    )
    LOGGER.info("Logging to %s", log_file_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dataset = get_mnist_dataset(train=False, root_dir=str(paths_config["data_dir"]))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = VAE(
        latent_dim=model_config["latent_dim"],
        num_classes=model_config["num_classes"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0
    sample_batch = None
    sample_labels = None
    sample_reconstructions = None

    LOGGER.info("Starting evaluation on test set...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            reconstructions, mu, logvar = model(inputs, labels)
            loss, recon_loss, kl_loss = compute_vae_loss(inputs, reconstructions, mu, logvar)

            batch_size_actual = inputs.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch_size_actual

            if batch_idx == 0:
                sample_batch = inputs.cpu()
                sample_labels = labels.cpu()
                sample_reconstructions = reconstructions.cpu()
            
            if (batch_idx + 1) % 10 == 0:
                LOGGER.info("Processed %d batches...", batch_idx + 1)

    avg_total_loss = total_loss / total_samples
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples

    LOGGER.info("Evaluation completed")
    LOGGER.info("Evaluation results")
    LOGGER.info("  %-*s : %.4f", CONFIG_KEY_WIDTH, "avg_total_loss", avg_total_loss)
    LOGGER.info("  %-*s : %.4f", CONFIG_KEY_WIDTH, "avg_recon_loss", avg_recon_loss)
    LOGGER.info("  %-*s : %.4f", CONFIG_KEY_WIDTH, "avg_kl_loss", avg_kl_loss)
    LOGGER.info("  %-*s : %d", CONFIG_KEY_WIDTH, "num_test_samples", total_samples)

    if save_reconstructions and sample_batch is not None:
        reconstruction_path = output_dir / "test_reconstructions.png"
        save_reconstruction_grid(sample_batch, sample_reconstructions, sample_labels, reconstruction_path)
        LOGGER.info("Saved reconstructions to %s", reconstruction_path)
    
    LOGGER.info("Evaluation finished successfully")


if __name__ == "__main__":
    main()
