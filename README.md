# Digit Latent Generation

This project implements a conditional Variational Autoencoder (VAE) for MNIST digit generation and latent space exploration.

## Features

- MNIST dataset support with automatic download
- Conditional VAE with class labels
- Training, evaluation, and generation pipelines
- Checkpoint saving and loading
- Streamlit UI for label-conditioned digit generation

## Project Structure

```text
digit-latent-gen/
├── configs/
│   ├── vae_config.yaml
│   ├── diffusion_config.yaml
│   └── generate_config.yaml
├── scripts/
│   ├── train_vae.py
│   ├── train_diffusion.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_diffusion.py
│   ├── generate.py
│   ├── generate_vae.py
│   ├── generate_diffusion.py
│   └── show_vae_structure.py
├── src/digit_latent_gen/
│   ├── models/vae.py
│   ├── training/trainer.py
│   ├── inference/generator.py
│   └── common/utils.py
├── streamlit_app.py
├── tests/
└── pyproject.toml
```

## Installation

Install PyTorch first, then install the project in editable mode.

```bash
pip install "torch==2.7.0" "torchvision==0.22.0"
pip install -e .
```

If you are using Apple Silicon, use a PyTorch build compatible with your MPS environment.

## Training

Train the VAE with the default configuration:

```bash
python scripts/train_vae.py
```

You can override selected options from the command line:

```bash
python scripts/train_vae.py --batch-size 64 --epochs 20 --learning-rate 0.001
```

To enable the configured learning-rate scheduler explicitly:

```bash
python scripts/train_vae.py --use-scheduler
```

The VAE training configuration lives in `configs/vae_config.yaml`.
The generation and deployment defaults live in `configs/generate_config.yaml`, separate from the training config.

Train the latent diffusion model after the VAE has been trained:

```bash
python scripts/train_diffusion.py
```

The latent diffusion training configuration lives in `configs/diffusion_config.yaml` and expects a VAE checkpoint at `vae_checkpoint_path`.

## Generation

Generate digit images from a trained checkpoint with a digit label from `0` to `9`:

```bash
python scripts/generate_vae.py --label 7 --num-samples 4
```

Useful options:

- `--checkpoint-path`: override the checkpoint location
- `--output-path`: save the generated grid to a specific file
- `--device`: force `cpu`, `cuda`, or `mps`

By default, generation uses `checkpoints/latest.pt`.
The VAE generation script reads `configs/generate_config.yaml` by default.

Generate digit images from a trained latent diffusion checkpoint:

```bash
python scripts/generate_diffusion.py --label 7 --num-samples 4
```

The latent diffusion generation script reads `configs/diffusion_config.yaml` by default.

## Streamlit App

Launch the browser UI:

```bash
streamlit run streamlit_app.py
```

The app lets you:

- switch between `VAE` and `Latent Diffusion`
- choose a digit label from `0` to `9`
- select the number of images to generate
- choose the relevant checkpoint path(s) and device
- view the generated digit images directly in the browser

The app reads `configs/generate_config.yaml` for the VAE mode and `configs/diffusion_config.yaml` for the latent diffusion mode by default.

## Evaluation

Evaluate a trained checkpoint on MNIST test data:

```bash
python scripts/evaluate.py
```

Evaluate a trained latent diffusion checkpoint after the VAE is available:

```bash
python scripts/evaluate_diffusion.py
```

## Configuration

Edit `configs/vae_config.yaml` to customize VAE training and evaluation behavior.

```yaml
model:
  in_channels: 1
  height: 32
  width: 32
  latent_dim: 32
  num_classes: 10

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  kl_weight: 0.1
  use_scheduler: false
  lr_scheduler:
    type: "cosine"
    t_max: 10
    eta_min: 0.0001
```

Edit `configs/diffusion_config.yaml` to customize latent diffusion training defaults.

Edit `configs/generate_config.yaml` to customize inference and deployment defaults.

```yaml
model:
  latent_dim: 32
  num_classes: 10

generation:
  checkpoint_path: "checkpoints/latest.pt"
  output_dir: "outputs/generated"
  default_num_samples: 4
  default_label: 0
  default_device: "auto"
```

## Development

Run tests with:

```bash
python -m pytest tests/
```

## License

MIT License
