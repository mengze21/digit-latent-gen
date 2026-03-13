# Digit Latent Generation

This project implements Variational Autoencoder (VAE) for digit generation and latent space exploration.

## Overview

This project provides implementations of:
- **Variational Autoencoder (VAE)** - For learning latent representations of MNIST digits
- **Latent space visualization and manipulation**
## Features

- MNIST dataset support with automatic download
- Conditional VAE with class labels
- Training and inference pipelines
- Checkpoint saving and loading
- Utility functions for dataset statistics computation

## Project Structure
# Digit Latent Generation

This project implements Variational Autoencoder (VAE) for digit generation and latent space exploration.

## Overview

This project provides implementations of:
- **Variational Autoencoder (VAE)** - For learning latent representations of MNIST digits
- **Latent space visualization and manipulation**

## Features

- MNIST dataset support with automatic download
- Conditional VAE with class labels
- Training and inference pipelines
- Checkpoint saving and loading
- Utility functions for dataset statistics computation

## Project Structure

```
digit-latent-gen/
├── configs/
│   └── config.yaml          # Configuration file for VAE
├── src/
│   └── digit_latent_gen/
│       ├── models/          # Model implementations
│       │   └── vae.py       # VAE model implementation
│       ├── training/        # Training pipeline
│       ├── inference/       # Inference pipeline
│       ├── common/          # Common utilities
│       │   └── utils.py     # Dataset utilities
│       └── data/            # Data loading modules
├── scripts/
│   ├── train.py             # Training script for VAE
│   └── show_vae_structure.py # VAE structure visualization
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
└── pyproject.toml          # Project configuration
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd digit-latent-gen
```

2. Install core dependencies first, especially PyTorch and torchvision:
```bash
pip install "torch==2.7.0" "torchvision==0.22.0"
```

If you are using Apple Silicon, install a PyTorch build that supports your macOS environment and MPS setup before running the editable install. The current `conda mps` environment uses `torch==2.7.0` and `torchvision==0.22.0`. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the recommended command for your setup.

3. Install the project in editable mode:
```bash
pip install -e .
```

## Usage

### Train VAE Model

```bash
python scripts/train.py
```

This will:
# MNIST Digit Latent Space Generator

A PyTorch-based Variational Autoencoder (VAE) for generating and exploring MNIST digit latent spaces.

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+

### Installation
- Train a VAE model with parameters from `configs/config.yaml`
- Save checkpoints to `checkpoints/` directory

You can also override training parameters from the command line. For example:

```bash
python scripts/train.py --batch-size 64 --epochs 20 --learning-rate 0.001
```

### Configuration

Edit `configs/config.yaml` to customize training parameters:

```yaml
model:
  in_channels: 1           # Grayscale images
  height: 32               # Image height
  width: 32                # Image width  
  latent_dim: 32           # Latent space dimension
  num_classes: 10          # MNIST digits (0-9)

training:
  batch_size: 64           # Training batch size
  epochs: 20               # Number of training epochs
  learning_rate: 0.001     # Learning rate
  weight_decay: 0.0        # L2 regularization
  kl_weight: 1.0           # KL divergence weight
  device: "cpu"            # "cpu" or "cuda"
  checkpoint_dir: "checkpoints"  # Directory for saving models
```

### Code Examples

#### Load and Train VAE

```python
import torch
from torch.utils.data import DataLoader
from digit_latent_gen.models.vae import VAE
from digit_latent_gen.training.trainer import Trainer
from digit_latent_gen.common.utils import get_mnist_dataset

# Load dataset
train_dataset = get_mnist_dataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
model = VAE(latent_dim=32)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    learning_rate=0.001
)

# Train model
trainer.train(num_epochs=20)
```

#### Compute Dataset Statistics

```python
from digit_latent_gen.common.utils import get_mnist_dataset, compute_dataset_stats

# Get dataset
dataset = get_mnist_dataset(train=True)

# Compute mean and std
mean, std = compute_dataset_stats(dataset, batch_size=128)
print(f"Dataset mean: {mean}, std: {std}")
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Structure

- `src/digit_latent_gen/models/vae.py`: VAE model implementation with Encoder and Decoder
- `src/digit_latent_gen/training/trainer.py`: Training loop and loss computation
- `src/digit_latent_gen/common/utils.py`: Utility functions for dataset handling
- `scripts/train.py`: Main training script

## License

MIT License
