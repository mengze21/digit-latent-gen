import argparse
import logging
import os
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
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
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


def resolve_training_config(config, args):
    model_config = config["model"].copy()
    train_config = config["training"].copy()

    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        train_config["num_workers"] = args.num_workers
    if args.checkpoint_dir is not None:
        train_config["checkpoint_dir"] = args.checkpoint_dir
    if args.shuffle is not None:
        train_config["shuffle"] = args.shuffle
    if args.device is not None:
        train_config["device"] = args.device

    return model_config, train_config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)
    model_config, train_config = resolve_training_config(config, args)
    data_config = config.get("data", {})

    latent_dim = model_config["latent_dim"]
    num_classes = model_config["num_classes"]
    data_root_dir = data_config.get("root_dir", "data")
    batch_size = train_config["batch_size"]
    num_epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]
    shuffle = train_config.get("shuffle", True)
    num_workers = train_config.get("num_workers", 0)
    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
    device = get_device(train_config.get("device"))

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Model configuration: %s", model_config)
    LOGGER.info("Data configuration: %s", data_config)
    LOGGER.info("Training configuration: %s", train_config)

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = get_mnist_dataset(train=True, root_dir=data_root_dir)
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
    )

    LOGGER.info("Starting training for %s epochs", num_epochs)
    trainer.train(num_epochs)
    LOGGER.info("Training completed")

    checkpoint_path = os.path.join(checkpoint_dir, "vae_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info("Model saved to %s", checkpoint_path)

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(model.state_dict(), latest_path)
    LOGGER.info("Latest checkpoint saved to %s", latest_path)


if __name__ == "__main__":
    main()
