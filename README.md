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
│   ├── config.yaml
│   └── generate_config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
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
python scripts/train.py
```

You can override selected options from the command line:

```bash
python scripts/train.py --batch-size 64 --epochs 20 --learning-rate 0.001
```

To enable the configured learning-rate scheduler explicitly:

```bash
python scripts/train.py --use-scheduler
```

The default configuration keeps scheduler disabled, but the `lr_scheduler` block remains in `configs/config.yaml` so it can be turned on later.
The generation and deployment defaults live in `configs/generate_config.yaml`, separate from the training config.

## Generation

Generate digit images from a trained checkpoint with a digit label from `0` to `9`:

```bash
python scripts/generate.py --label 7 --num-samples 4
```

Useful options:

- `--checkpoint-path`: override the checkpoint location
- `--output-path`: save the generated grid to a specific file
- `--device`: force `cpu`, `cuda`, or `mps`

By default, generation uses `checkpoints/latest.pt`.
The generation script reads `configs/generate_config.yaml` by default.

## Streamlit App

Launch the browser UI:

```bash
streamlit run streamlit_app.py
```

The app lets you:

- choose a digit label from `0` to `9`
- select the number of images to generate
- choose the checkpoint path and device
- view the generated digit images directly in the browser
The app also reads `configs/generate_config.yaml` by default.

## Evaluation

Evaluate a trained checkpoint on MNIST test data:

```bash
python scripts/evaluate.py
```

## Configuration

Edit `configs/config.yaml` to customize training behavior.

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
