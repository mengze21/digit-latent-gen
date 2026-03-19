import argparse
from pathlib import Path

import torch
import yaml

from digit_latent_gen.inference import VAEGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "generate_config.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_device(requested_device=None):
    if requested_device is not None and requested_device != "auto":
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate digit images from a trained conditional VAE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--label", type=int, required=True, help="Digit label to generate, from 0 to 9.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional explicit output image path.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    generation_config = config.get("generation", {})

    checkpoint_path = args.checkpoint_path or resolve_path(
        generation_config.get("checkpoint_path", "checkpoints/latest.pt")
    )
    output_dir = args.output_dir or resolve_path(generation_config.get("output_dir", "outputs/generated"))
    device = get_device(args.device or generation_config.get("default_device"))

    generator = VAEGenerator(
        latent_dim=model_config["latent_dim"],
        label=args.label,
        num_classes=model_config["num_classes"],
        model_path=checkpoint_path,
        output_dir=output_dir,
        batch_size=args.num_samples,
        device=device,
    )

    images = generator.generate(num_samples=args.num_samples, label=args.label)
    save_path = generator.save_generated_images(
        images,
        output_path=args.output_path,
        max_images=max(1, args.num_samples),
    )

    print(f"Generated {images.size(0)} image(s) for label {args.label}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
