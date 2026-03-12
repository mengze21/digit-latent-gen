import argparse
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
try:
    from digit_latent_gen.models.vae import Decoder, Encoder, VAE
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import 'digit_latent_gen'. Install the project into the environment first, "
        "for example with 'pip install -e .', then rerun this script."
    ) from exc


def _format_shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return [_format_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: _format_shape(item) for key, item in value.items()}
    return type(value).__name__


def _collect_layer_shapes(model, *inputs):
    rows = []
    hooks = []

    for name, module in model.named_modules():
        if not name:
            continue
        hooks.append(
            module.register_forward_hook(
                lambda _, module_inputs, output, layer_name=name, layer_type=module.__class__.__name__: rows.append(
                    (layer_name, layer_type, _format_shape(module_inputs), _format_shape(output))
                )
            )
        )

    try:
        model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return rows


def _print_summary(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Layer Name':<20} {'Type':<20} {'Input Shape':<28} Output Shape")
    for layer_name, layer_type, input_shape, output_shape in rows:
        print(f"{layer_name:<20} {layer_type:<20} {str(input_shape):<28} {output_shape}")


def main():
    parser = argparse.ArgumentParser(description="Display VAE layer names and tensor shapes.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dummy inputs.")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model_config = config["model"]
    in_channels = model_config["in_channels"]
    height = model_config["height"]
    width = model_config["width"]
    latent_dim = model_config["latent_dim"]
    num_classes = model_config["num_classes"]

    x = torch.randn(args.batch_size, in_channels, height, width)
    labels = torch.arange(args.batch_size, dtype=torch.long) % num_classes
    z = torch.randn(args.batch_size, latent_dim)

    encoder = Encoder(latent_dim=latent_dim, num_classes=num_classes)
    decoder = Decoder(latent_dim=latent_dim, num_classes=num_classes)
    vae = VAE(latent_dim=latent_dim, num_classes=num_classes)

    _print_summary("Encoder", _collect_layer_shapes(encoder, x, labels))
    _print_summary("Decoder", _collect_layer_shapes(decoder, z, labels))
    _print_summary("VAE", _collect_layer_shapes(vae, x, labels))


if __name__ == "__main__":
    main()
