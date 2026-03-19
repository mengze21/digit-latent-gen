from pathlib import Path

import streamlit as st
import torch
import yaml

from digit_latent_gen.inference import Generator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "generate_config.yaml"


st.set_page_config(
    page_title="Digit Latent Generator",
    page_icon="7",
    layout="centered",
)


@st.cache_data
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def render_generated_images(images):
    cols = st.columns(min(4, images.size(0)))
    for idx, image in enumerate(images):
        with cols[idx % len(cols)]:
            st.image(image.squeeze().numpy(), clamp=True, use_container_width=True, caption=f"Sample {idx + 1}")


st.title("Digit Latent Generator")
st.caption("输入 0 到 9 的数字标签，生成对应的手写数字图像。")

config = load_config(DEFAULT_CONFIG_PATH)
model_config = config["model"]
generation_config = config.get("generation", {})

with st.sidebar:
    st.header("Generation Settings")
    label = st.selectbox("Digit label", list(range(10)), index=generation_config.get("default_label", 0))
    num_samples = st.slider(
        "Number of images",
        min_value=1,
        max_value=16,
        value=generation_config.get("default_num_samples", 4),
    )
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=str(resolve_path(generation_config.get("checkpoint_path", "checkpoints/latest.pt"))),
    )
    output_dir = st.text_input(
        "Output directory",
        value=str(resolve_path(generation_config.get("output_dir", "outputs/generated"))),
    )
    device_options = ["auto", "cpu", "mps", "cuda"]
    default_device = generation_config.get("default_device", "auto")
    device = st.selectbox(
        "Device",
        device_options,
        index=device_options.index(default_device) if default_device in device_options else 0,
    )
    generate_clicked = st.button("Generate")

resolved_device = get_device() if device == "auto" else torch.device(device)

if generate_clicked:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        st.stop()

    generator = Generator(
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

    st.success(f"Generated {images.size(0)} image(s) for label {label}.")
    render_generated_images(images)

    save_path = generator.save_generated_images(images, max_images=num_samples)
    st.info(f"Saved preview to {save_path}")
else:
    st.info("从侧边栏选择数字并点击 Generate。")
