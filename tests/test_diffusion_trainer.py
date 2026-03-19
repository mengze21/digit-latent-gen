"""Smoke tests for the latent diffusion trainer."""
from torch.utils.data import DataLoader, TensorDataset

import pytest
import torch

from digit_latent_gen.models.diffusion import DiffusionModel
from digit_latent_gen.training.diffusion_trainer import DiffusionTrainer


@pytest.fixture
def synthetic_diffusion_dataset():
    num_samples = 12
    data = torch.randn(num_samples, 32)
    labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(data, labels)


@pytest.fixture
def test_device():
    return torch.device("cpu")


def test_diffusion_trainer_one_epoch(synthetic_diffusion_dataset, test_device):
    model = DiffusionModel(
        time_steps=8,
        latent_dim=32,
        num_classes=10,
        device=str(test_device),
    )
    train_loader = DataLoader(synthetic_diffusion_dataset, batch_size=4, shuffle=True)
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        device=test_device,
        learning_rate=1e-4,
        scheduler_config={"type": "cosine", "t_max": 2, "eta_min": 0.0},
    )

    initial_lr = trainer.get_current_learning_rate()
    metrics = trainer.train_epoch()
    updated_lr = trainer.get_current_learning_rate()

    assert isinstance(metrics, dict)
    assert metrics["avg_loss"] > 0
    assert metrics["num_batches"] == 3
    assert not torch.isnan(torch.tensor(metrics["avg_loss"]))
    assert updated_lr == pytest.approx(initial_lr * 0.5)

    model.eval()
    with torch.no_grad():
        samples = model.generate(
            num_samples=4,
            class_labels=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )

    assert samples.shape == (4, 32)
    assert not torch.isnan(samples).any()
