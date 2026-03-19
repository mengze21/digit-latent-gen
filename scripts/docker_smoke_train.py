import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from digit_latent_gen.models.vae import VAE
from digit_latent_gen.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a 1-epoch Docker smoke test for VAE training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for the smoke test.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of synthetic samples to generate.")
    return parser.parse_args()


def build_synthetic_loader(model_config, batch_size, num_samples):
    data = torch.randn(
        num_samples,
        model_config["in_channels"],
        model_config["height"],
        model_config["width"],
    )
    labels = torch.randint(0, model_config["num_classes"], (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    paths_config = config.get("paths", {})

    checkpoint_dir = resolve_path(paths_config.get("checkpoint_dir", "checkpoints"))
    output_dir = resolve_path(paths_config.get("output_dir", "outputs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    scheduler_config = training_config.get("lr_scheduler")

    train_loader = build_synthetic_loader(model_config, batch_size, args.num_samples)
    model = VAE(
        latent_dim=model_config["latent_dim"],
        num_classes=model_config["num_classes"],
    ).to(device)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        scheduler_config=scheduler_config,
    )

    print(f"Running Docker smoke training on {device} for 1 epoch")
    metrics = trainer.train_epoch()
    print(
        "Smoke training loss: "
        f"{metrics['avg_loss']:.4f} "
        f"(recon={metrics['avg_recon_loss']:.4f}, kl={metrics['avg_kl_loss']:.4f})"
    )

    checkpoint_path = checkpoint_dir / "docker_smoke_vae.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    loaded_model = VAE(
        latent_dim=model_config["latent_dim"],
        num_classes=model_config["num_classes"],
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    sample_batch, sample_labels = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    sample_labels = sample_labels.to(device)
    with torch.no_grad():
        recon, mu, logvar = loaded_model(sample_batch, sample_labels)

    assert checkpoint_path.exists(), "Checkpoint file was not created."
    assert recon.shape == sample_batch.shape, "Loaded checkpoint produced invalid reconstruction shape."
    assert mu.shape[1] == model_config["latent_dim"], "Loaded checkpoint produced invalid latent mean shape."
    assert logvar.shape[1] == model_config["latent_dim"], "Loaded checkpoint produced invalid latent logvar shape."

    print("Checkpoint reload succeeded")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
