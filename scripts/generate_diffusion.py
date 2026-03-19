import argparse
from pathlib import Path

import torch
import yaml

from digit_latent_gen.inference import DiffusionGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "diffusion_config.yaml"


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
    parser = argparse.ArgumentParser(
        description="Generate digit images from a trained latent diffusion model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--label", type=int, default=None, help="Digit label to generate, from 0 to 9.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--vae-checkpoint-path", type=Path, default=None, help="Override VAE checkpoint path.")
    parser.add_argument(
        "--diffusion-checkpoint-path",
        type=Path,
        default=None,
        help="Override diffusion checkpoint path.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional explicit output image path.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu, cuda, or mps.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    generation_config = config.get("generation", {})
    diffusion_config = config.get("diffusion", {})

    label = args.label if args.label is not None else generation_config.get("default_label", 0)
    num_samples = args.num_samples if args.num_samples is not None else generation_config.get("default_num_samples", 1)
    vae_checkpoint_path = args.vae_checkpoint_path or resolve_path(
        generation_config.get("vae_checkpoint_path", "checkpoints/latest.pt")
    )
    diffusion_checkpoint_path = args.diffusion_checkpoint_path or resolve_path(
        generation_config.get("diffusion_checkpoint_path", "checkpoints/latest_diffusion.pt")
    )
    output_dir = args.output_dir or resolve_path(generation_config.get("output_dir", "outputs/generated"))
    device = get_device(args.device or generation_config.get("default_device"))

    generator = DiffusionGenerator(
        latent_dim=model_config["latent_dim"],
        label=label,
        num_classes=model_config["num_classes"],
        vae_checkpoint_path=vae_checkpoint_path,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        output_dir=output_dir,
        batch_size=num_samples,
        device=device,
        diffusion_config=diffusion_config,
    )

    images = generator.generate(num_samples=num_samples, label=label)
    save_path = generator.save_generated_images(
        images,
        output_path=args.output_path,
        max_images=max(1, num_samples),
    )

    print(f"Generated {images.size(0)} image(s) for label {label}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
