import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from digit_latent_gen.common.utils import get_mnist_dataset
from digit_latent_gen.models.vae import VAE
from digit_latent_gen.models.diffusion import DiffusionModel
from digit_latent_gen.training.diffusion_trainer import DiffusionTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "diffusion_config.yaml"
LOGGER = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the latent diffusion model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override diffusion batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Override log directory.")
    parser.add_argument("--time-steps", type=int, default=None, help="Override diffusion time steps.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimensionality.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override denoiser hidden size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override denoiser depth.")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Optional gradient clipping norm for diffusion training.",
    )
    return parser.parse_args()


def get_device(requested_device=None):
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_runtime_config(config, args):
    model_config = config["model"].copy()
    training_config = config["training"].copy()
    diffusion_config = config.get("diffusion", {}).copy()
    paths_config = config.get("paths", {}).copy()

    if args.batch_size is not None:
        diffusion_config["batch_size"] = args.batch_size
    if args.epochs is not None:
        diffusion_config["epochs"] = args.epochs
    if args.learning_rate is not None:
        diffusion_config["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        diffusion_config["num_workers"] = args.num_workers
    if args.device is not None:
        diffusion_config["device"] = args.device
    if args.data_dir is not None:
        paths_config["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        paths_config["output_dir"] = str(args.output_dir)
    if args.checkpoint_dir is not None:
        paths_config["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.log_dir is not None:
        paths_config["log_dir"] = str(args.log_dir)
    if args.time_steps is not None:
        diffusion_config["time_steps"] = args.time_steps
    if args.latent_dim is not None:
        diffusion_config["latent_dim"] = args.latent_dim
    if args.hidden_dim is not None:
        diffusion_config["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        diffusion_config["num_layers"] = args.num_layers
    if args.grad_clip_norm is not None:
        diffusion_config["grad_clip_norm"] = args.grad_clip_norm

    return {
        "model": model_config,
        "training": training_config,
        "diffusion": diffusion_config,
        "paths": {
            "data_dir": resolve_path(paths_config.get("data_dir", "data")),
            "output_dir": resolve_path(paths_config.get("output_dir", "outputs")),
            "checkpoint_dir": resolve_path(paths_config.get("checkpoint_dir", "checkpoints")),
            "log_dir": resolve_path(paths_config.get("log_dir", "outputs/logs")),
        },
    }


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8"),
        ],
    )


def log_epoch_metrics(epoch, metrics, current_lr, num_epochs):
    LOGGER.info(
        "Epoch %s/%s | avg_loss: %.4f | lr: %.6f | batches: %s",
        epoch,
        num_epochs,
        metrics["avg_loss"],
        current_lr,
        metrics["num_batches"],
    )


def extract_latent_dataset(vae_model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    latent_batches = []
    label_batches = []

    vae_model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            mu, _ = vae_model.encoder(data, labels)
            latent_batches.append(mu.cpu())
            label_batches.append(labels.cpu())

    latents = torch.cat(latent_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return TensorDataset(latents, labels)


def main():
    args = parse_args()
    config = load_config(args.config)
    runtime_config = resolve_runtime_config(config, args)

    model_config = runtime_config["model"]
    diffusion_config = runtime_config["diffusion"]
    paths_config = runtime_config["paths"]

    latent_dim = diffusion_config.get("latent_dim", model_config["latent_dim"])
    num_classes = model_config["num_classes"]
    batch_size = diffusion_config.get("batch_size", 32)
    num_epochs = diffusion_config.get("epochs", 10)
    learning_rate = diffusion_config.get("learning_rate", 1e-4)
    num_workers = diffusion_config.get("num_workers", 0)
    time_steps = diffusion_config.get("time_steps", 1000)
    hidden_dim = diffusion_config.get("hidden_dim")
    num_layers = diffusion_config.get("num_layers", 4)
    grad_clip_norm = diffusion_config.get("grad_clip_norm")
    checkpoint_name = diffusion_config.get("checkpoint_name", "diffusion_model.pt")
    latest_checkpoint_name = diffusion_config.get("latest_checkpoint_name", "latest_diffusion.pt")
    log_filename = diffusion_config.get("log_filename", "diffusion_train.log")
    vae_checkpoint_path = resolve_path(diffusion_config.get("vae_checkpoint_path", "checkpoints/latest.pt"))
    data_dir = paths_config["data_dir"]
    checkpoint_dir = paths_config["checkpoint_dir"]
    log_dir = paths_config["log_dir"]
    device = get_device(diffusion_config.get("device"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / log_filename)

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Model configuration: %s", model_config)
    LOGGER.info("Diffusion configuration: %s", diffusion_config)
    LOGGER.info("Path configuration: %s", {name: str(path) for name, path in paths_config.items()})

    if not vae_checkpoint_path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_checkpoint_path}. Train the VAE first before latent diffusion."
        )

    base_dataset = get_mnist_dataset(train=True, root_dir=str(data_dir))

    vae_model = VAE(
        latent_dim=model_config["latent_dim"],
        num_classes=num_classes,
    ).to(device)
    vae_state = torch.load(vae_checkpoint_path, map_location=device)
    if isinstance(vae_state, dict) and "model_state_dict" in vae_state:
        vae_state = vae_state["model_state_dict"]
    vae_model.load_state_dict(vae_state)
    latent_dataset = extract_latent_dataset(
        vae_model=vae_model,
        dataset=base_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    train_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = DiffusionModel(
        latent_dim=latent_dim,
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        device=device,
    ).to(device)
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        scheduler_config=diffusion_config.get("lr_scheduler"),
        grad_clip_norm=grad_clip_norm,
    )

    LOGGER.info("Starting diffusion training for %s epochs", num_epochs)

    def on_epoch_end(epoch, metrics, current_lr):
        log_epoch_metrics(epoch, metrics, current_lr, num_epochs)

    trainer.train(num_epochs, epoch_end_callback=on_epoch_end)

    LOGGER.info("Diffusion training completed")

    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info("Model saved to %s", checkpoint_path)

    latest_path = checkpoint_dir / latest_checkpoint_name
    torch.save(model.state_dict(), latest_path)
    LOGGER.info("Latest checkpoint saved to %s", latest_path)


if __name__ == "__main__":
    main()
