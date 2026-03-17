import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from digit_latent_gen.models.vae import VAE
from digit_latent_gen.training.trainer import Trainer
from digit_latent_gen.common.utils import get_mnist_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config.yaml"
LOGGER = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VAE model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Override log directory.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dataloader shuffling.",
    )
    return parser.parse_args()


def get_device(requested_device=None):
    """Return the requested device if available, otherwise choose the best available backend."""
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
    train_config = config["training"].copy()
    paths_config = config.get("paths", {}).copy()

    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        train_config["num_workers"] = args.num_workers
    if args.shuffle is not None:
        train_config["shuffle"] = args.shuffle
    if args.device is not None:
        train_config["device"] = args.device
    if args.data_dir is not None:
        paths_config["data_dir"] = str(args.data_dir)
    if args.output_dir is not None:
        paths_config["output_dir"] = str(args.output_dir)
    if args.checkpoint_dir is not None:
        paths_config["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.log_dir is not None:
        paths_config["log_dir"] = str(args.log_dir)

    runtime_config = {
        "model": model_config,
        "training": train_config,
        "paths": {
            "data_dir": resolve_path(paths_config.get("data_dir", "data")),
            "output_dir": resolve_path(paths_config.get("output_dir", "outputs")),
            "checkpoint_dir": resolve_path(paths_config.get("checkpoint_dir", "checkpoints")),
            "log_dir": resolve_path(paths_config.get("log_dir", "outputs/logs")),
        },
    }
    return runtime_config


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8"),
        ],
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    runtime_config = resolve_runtime_config(config, args)
    model_config = runtime_config["model"]
    train_config = runtime_config["training"]
    paths_config = runtime_config["paths"]

    latent_dim = model_config["latent_dim"]
    num_classes = model_config["num_classes"]
    batch_size = train_config["batch_size"]
    num_epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]
    kl_weight = train_config.get("kl_weight", 1.0)
    shuffle = train_config.get("shuffle", True)
    num_workers = train_config.get("num_workers", 0)
    checkpoint_name = train_config.get("checkpoint_name", "vae_model.pt")
    latest_checkpoint_name = train_config.get("latest_checkpoint_name", "latest.pt")
    log_filename = train_config.get("log_filename", "train.log")
    data_dir = paths_config["data_dir"]
    output_dir = paths_config["output_dir"]
    checkpoint_dir = paths_config["checkpoint_dir"]
    log_dir = paths_config["log_dir"]
    device = get_device(train_config.get("device"))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename
    setup_logging(log_file_path)

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Model configuration: %s", model_config)
    LOGGER.info("Training configuration: %s", train_config)
    LOGGER.info(
        "Path configuration: %s",
        {name: str(path) for name, path in paths_config.items()},
    )
    LOGGER.info("Logging to %s", log_file_path)

    train_dataset = get_mnist_dataset(train=True, root_dir=str(data_dir))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    model = VAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
    )

    LOGGER.info("Starting training for %s epochs", num_epochs)
    trainer.train(num_epochs)
    LOGGER.info("Training completed")

    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info("Model saved to %s", checkpoint_path)

    latest_path = checkpoint_dir / latest_checkpoint_name
    torch.save(model.state_dict(), latest_path)
    LOGGER.info("Latest checkpoint saved to %s", latest_path)


if __name__ == "__main__":
    main()
