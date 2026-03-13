"""Smoke tests for VAE forward pass, trainer integration, and visualization."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from digit_latent_gen.models.vae import VAE
from digit_latent_gen.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for smoke testing."""
    num_samples = 64
    data = torch.randn(num_samples, 1, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(data, labels)


@pytest.fixture
def test_device():
    """Use CPU for stable smoke tests."""
    return torch.device("cpu")


@pytest.fixture
def vae_model():
    """Create a VAE model for smoke tests."""
    return VAE(latent_dim=32, num_classes=10)


def test_forward_pass(vae_model, synthetic_dataset, test_device):
    """Smoke test a forward pass through the VAE."""
    vae_model.to(test_device)
    vae_model.eval()

    data, labels = synthetic_dataset[0:4]
    data = data.to(test_device)
    labels = labels.to(test_device)

    with torch.no_grad():
        recon, mu, logvar = vae_model(data, labels)

    assert recon.shape == data.shape
    assert mu.shape == (4, vae_model.latent_dim)
    assert logvar.shape == (4, vae_model.latent_dim)
    assert not torch.isnan(recon).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(logvar).any()


def test_trainer_integration(vae_model, synthetic_dataset, test_device):
    """Smoke test one training epoch through the Trainer."""
    train_loader = DataLoader(synthetic_dataset, batch_size=8, shuffle=True)
    trainer = Trainer(
        model=vae_model,
        train_loader=train_loader,
        device=test_device,
        learning_rate=1e-3,
    )

    avg_loss = trainer.train_epoch()
    assert isinstance(avg_loss, float)
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))


def test_visualization_generation(vae_model, synthetic_dataset, test_device):
    """Smoke test visualization generation in the project outputs directory."""
    vae_model.to(test_device)
    vae_model.eval()

    data, labels = synthetic_dataset[0:4]
    data = data.to(test_device)
    labels = labels.to(test_device)

    with torch.no_grad():
        recon, _, _ = vae_model(data, labels)

    output_dir = ROOT / CONFIG["testing"]["output_dir"] / "smoke_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualization output directory: {output_dir}")

    fig1, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axes[0, i].imshow(data[i].detach().cpu().squeeze().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Label: {labels[i].item()}")
        axes[1, i].imshow(recon[i].detach().cpu().squeeze().numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.suptitle("VAE Reconstructions", fontsize=14)
    plt.tight_layout()

    recon_path = output_dir / "reconstructions.png"
    plt.savefig(recon_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    all_data, all_labels = synthetic_dataset[:]
    all_data = all_data.to(test_device)
    all_labels = all_labels.to(test_device)
    with torch.no_grad():
        mu, logvar = vae_model.encoder(all_data, all_labels)
        z = vae_model.reparameterize(mu, logvar)

    fig2, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        z[:, 0].detach().cpu().numpy(),
        z[:, 1].detach().cpu().numpy(),
        c=all_labels.detach().cpu().numpy(),
        cmap="tab10",
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Digit Class")
    ax.set_xlabel("Latent Dimension 0")
    ax.set_ylabel("Latent Dimension 1")
    ax.set_title("Latent Space Visualization")
    ax.grid(True, alpha=0.3)

    latent_path = output_dir / "latent_space.png"
    plt.savefig(latent_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    assert recon_path.exists()
    assert latent_path.exists()
