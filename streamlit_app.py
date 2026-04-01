from pathlib import Path
from typing import Any, Optional

import streamlit as st
import torch
import yaml

from digit_latent_gen.inference import DiffusionGenerator, VAEGenerator

GeneratorType = VAEGenerator | DiffusionGenerator

ROOT = Path(__file__).resolve().parent
VAE_CONFIG_PATH = ROOT / "configs" / "generate_config.yaml"
DIFFUSION_CONFIG_PATH = ROOT / "configs" / "diffusion_config.yaml"


st.set_page_config(
    page_title="Digit Latent Generator",
    page_icon="7",
    layout="centered",
)


@st.cache_data
def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def render_generated_images(images: torch.Tensor) -> None:
    num_images = int(images.size(0))
    cols = st.columns(min(4, max(1, num_images)))
    for idx, image in enumerate(images):
        with cols[idx % len(cols)]:
            st.image(
                image.squeeze().numpy(),
                clamp=True,
                use_container_width=True,
                caption=f"Sample {idx + 1}",
            )


def render_comparison(
    primary_title: str,
    primary_images: torch.Tensor,
    secondary_title: str,
    secondary_images: torch.Tensor,
) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader(primary_title)
        render_generated_images(primary_images)
    with right:
        st.subheader(secondary_title)
        render_generated_images(secondary_images)


st.title("Digit Latent Generator")
st.caption("在 VAE 和 Latent Diffusion 两种生成模式之间切换。")

vae_config = load_config(VAE_CONFIG_PATH)
diffusion_config = load_config(DIFFUSION_CONFIG_PATH)

mode_options = ["VAE", "Latent Diffusion"]

with st.sidebar:
    st.header("Generation Settings")
    mode = st.selectbox("Mode", mode_options, index=0)

    if mode == "VAE":
        model_config = vae_config["model"]
        generation_config = vae_config.get("generation", {})
        label_default = generation_config.get("default_label", 0)
        num_samples_default = generation_config.get("default_num_samples", 4)
        checkpoint_default = resolve_path(
            generation_config.get("checkpoint_path", "checkpoints/vae_model.pt")
        )
        output_default = resolve_path(
            generation_config.get("output_dir", "outputs/generated")
        )
        device_default = generation_config.get("default_device", "auto")
    else:
        model_config = diffusion_config["model"]
        generation_config = diffusion_config.get("generation", {})
        label_default = generation_config.get("default_label", 0)
        num_samples_default = generation_config.get("default_num_samples", 4)
        vae_checkpoint_default = resolve_path(
            generation_config.get("vae_checkpoint_path", "checkpoints/vae_model.pt")
        )
        diffusion_checkpoint_default = resolve_path(
            generation_config.get(
                "diffusion_checkpoint_path", "checkpoints/diffusion_model.pt"
            )
        )
        output_default = resolve_path(
            generation_config.get("output_dir", "outputs/generated")
        )
        device_default = generation_config.get("default_device", "auto")

    label = st.selectbox(
        "Digit label", list(range(model_config["num_classes"])), index=label_default
    )
    num_samples = st.slider(
        "Number of images",
        min_value=1,
        max_value=16,
        value=num_samples_default,
    )

    if mode == "VAE":
        checkpoint_path = st.text_input(
            "VAE checkpoint path",
            value=str(checkpoint_default),
        )
    else:
        vae_checkpoint_path = st.text_input(
            "VAE checkpoint path",
            value=str(vae_checkpoint_default),
        )
        diffusion_checkpoint_path = st.text_input(
            "Diffusion checkpoint path",
            value=str(diffusion_checkpoint_default),
        )

    output_dir = st.text_input(
        "Output directory",
        value=str(output_default),
    )

    device_options = ["auto", "cpu", "mps", "cuda"]
    device = st.selectbox(
        "Device",
        device_options,
        index=(
            device_options.index(device_default)
            if device_default in device_options
            else 0
        ),
    )
    compare_outputs = st.checkbox("Keep both outputs for comparison", value=True)
    generate_clicked = st.button("Generate")

resolved_device = get_device() if device == "auto" else torch.device(device)

with st.sidebar:
    st.markdown(
        f"""
        <div style="
            font-size: 1.15rem;
            font-weight: 800;
            padding: 0.7rem 0.9rem;
            border-radius: 0.7rem;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.28);
            margin-top: 0.5rem;
            line-height: 1.2;
        ">
            Using device: <code>{resolved_device.type}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

if generate_clicked:
    generator: GeneratorType
    comparison_generator: Optional[GeneratorType] = None
    comparison_images: Optional[torch.Tensor] = None

    if mode == "VAE":
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            st.error(f"Checkpoint not found: {checkpoint}")
            st.stop()

        generator = VAEGenerator(
            latent_dim=model_config["latent_dim"],
            label=label,
            num_classes=model_config["num_classes"],
            model_path=checkpoint,
            output_dir=output_dir,
            batch_size=num_samples,
            device=resolved_device,
        )

        with st.spinner("Generating images..."):
            images = generator.generate(num_samples=num_samples, label=label)

        if compare_outputs:
            vae_checkpoint = checkpoint
            diffusion_checkpoint = Path(
                diffusion_config.get("generation", {}).get(
                    "diffusion_checkpoint_path", "checkpoints/diffusion_model.pt"
                )
            )
            if diffusion_checkpoint.exists():
                comparison_generator = DiffusionGenerator(
                    latent_dim=model_config["latent_dim"],
                    label=label,
                    num_classes=model_config["num_classes"],
                    vae_checkpoint_path=vae_checkpoint,
                    diffusion_checkpoint_path=diffusion_checkpoint,
                    output_dir=output_dir,
                    batch_size=num_samples,
                    device=resolved_device,
                    diffusion_config=diffusion_config.get("training", {}),
                )
                with st.spinner("Generating diffusion comparison..."):
                    comparison_images = comparison_generator.generate(
                        num_samples=num_samples, label=label
                    )

    else:
        vae_checkpoint = Path(vae_checkpoint_path)
        diffusion_checkpoint = Path(diffusion_checkpoint_path)
        if not vae_checkpoint.exists():
            st.error(f"VAE checkpoint not found: {vae_checkpoint}")
            st.stop()
        if not diffusion_checkpoint.exists():
            st.error(f"Diffusion checkpoint not found: {diffusion_checkpoint}")
            st.stop()

        generator = DiffusionGenerator(
            latent_dim=model_config["latent_dim"],
            label=label,
            num_classes=model_config["num_classes"],
            vae_checkpoint_path=vae_checkpoint,
            diffusion_checkpoint_path=diffusion_checkpoint,
            output_dir=output_dir,
            batch_size=num_samples,
            device=resolved_device,
            diffusion_config=diffusion_config.get("training", {}),
        )

        with st.spinner("Generating images with latent diffusion..."):
            images = generator.generate(num_samples=num_samples, label=label)

        if compare_outputs:
            comparison_generator = VAEGenerator(
                latent_dim=model_config["latent_dim"],
                label=label,
                num_classes=model_config["num_classes"],
                model_path=vae_checkpoint,
                output_dir=output_dir,
                batch_size=num_samples,
                device=resolved_device,
            )
            with st.spinner("Generating VAE comparison..."):
                comparison_images = comparison_generator.generate(
                    num_samples=num_samples, label=label
                )

    st.success(f"Generated {images.size(0)} image(s) for label {label}.")
    if compare_outputs and comparison_images is not None:
        if mode == "VAE":
            render_comparison(
                "VAE output", images, "Latent Diffusion output", comparison_images
            )
        else:
            render_comparison(
                "Latent Diffusion output", images, "VAE output", comparison_images
            )
    else:
        render_generated_images(images)

    save_path = generator.save_generated_images(images, max_images=num_samples)
    st.info(f"Saved preview to {save_path}")
    if compare_outputs and comparison_images is not None:
        if comparison_generator is None:
            raise RuntimeError(
                "comparison_generator must exist when comparison_images are present."
            )
        comparison_save_path = comparison_generator.save_generated_images(
            comparison_images, max_images=num_samples
        )
        st.info(f"Saved comparison preview to {comparison_save_path}")
    elif compare_outputs:
        st.warning(
            "Comparison output was skipped because the alternate checkpoint was not found."
        )
else:
    st.info("从侧边栏选择模式、数字并点击 Generate。")
