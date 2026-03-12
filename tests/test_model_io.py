from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

MODEL_CONFIG = CONFIG["model"]

from digit_latent_gen.models.vae import Decoder, Encoder, VAE


IN_CHANNELS = MODEL_CONFIG["in_channels"]
LATENT_DIM = MODEL_CONFIG["latent_dim"]
NUM_CLASSES = MODEL_CONFIG["num_classes"]
BATCH_SIZE = 2
HEIGHT = MODEL_CONFIG["height"]
WIDTH = MODEL_CONFIG["width"]


def _dummy_inputs():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    labels = torch.tensor([1, 7], dtype=torch.long)
    return x, labels


def _capture_module_outputs(model, module_names, *inputs):
    captured = {}
    hooks = []

    for name, module in model.named_modules():
        if name in module_names:
            hooks.append(module.register_forward_hook(lambda _, __, output, key=name: captured.setdefault(key, output)))

    try:
        model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return captured


def test_encoder_layer_shapes():
    x, labels = _dummy_inputs()
    model = Encoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)

    captured = _capture_module_outputs(model, {"label_emb", "enc_conv1", "enc_conv2", "fc_mu", "fc_logvar"}, x, labels)
    mu, logvar = model(x, labels)

    # Add detailed error messages
    assert captured["label_emb"].shape == (BATCH_SIZE, NUM_CLASSES), \
        f"label_emb shape mismatch: expected {(BATCH_SIZE, NUM_CLASSES)}, got {captured['label_emb'].shape}"
    
    assert captured["enc_conv1"].shape == (BATCH_SIZE, 32, 16, 16), \
        f"enc_conv1 shape mismatch: expected {(BATCH_SIZE, 32, 16, 16)}, got {captured['enc_conv1'].shape}"
    
    assert captured["enc_conv2"].shape == (BATCH_SIZE, 64, 8, 8), \
        f"enc_conv2 shape mismatch: expected {(BATCH_SIZE, 64, 8, 8)}, got {captured['enc_conv2'].shape}"
    
    assert captured["fc_mu"].shape == (BATCH_SIZE, LATENT_DIM), \
        f"fc_mu shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {captured['fc_mu'].shape}"
    
    assert captured["fc_logvar"].shape == (BATCH_SIZE, LATENT_DIM), \
        f"fc_logvar shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {captured['fc_logvar'].shape}"
    
    assert mu.shape == (BATCH_SIZE, LATENT_DIM), \
        f"mu shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {mu.shape}"
    
    assert logvar.shape == (BATCH_SIZE, LATENT_DIM), \
        f"logvar shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {logvar.shape}"


def test_decoder_layer_shapes():
    _, labels = _dummy_inputs()
    z = torch.randn(BATCH_SIZE, LATENT_DIM)
    model = Decoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)

    captured = _capture_module_outputs(model, {"label_emb", "fc", "dec_conv1", "dec_conv2"}, z, labels)
    x_recon = model(z, labels)

    # Add detailed error messages
    assert captured["label_emb"].shape == (BATCH_SIZE, NUM_CLASSES), \
        f"label_emb shape mismatch: expected {(BATCH_SIZE, NUM_CLASSES)}, got {captured['label_emb'].shape}"
    
    assert captured["fc"].shape == (BATCH_SIZE, 64 * 8 * 8), \
        f"fc shape mismatch: expected {(BATCH_SIZE, 64 * 8 * 8)}, got {captured['fc'].shape}"
    
    assert captured["dec_conv1"].shape == (BATCH_SIZE, 32, 16, 16), \
        f"dec_conv1 shape mismatch: expected {(BATCH_SIZE, 32, 16, 16)}, got {captured['dec_conv1'].shape}"
    
    assert captured["dec_conv2"].shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), \
        f"dec_conv2 shape mismatch: expected {(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)}, got {captured['dec_conv2'].shape}"
    
    assert x_recon.shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), \
        f"x_recon shape mismatch: expected {(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)}, got {x_recon.shape}"


def test_vae_end_to_end_shapes():
    x, labels = _dummy_inputs()
    model = VAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)

    x_recon, mu, logvar = model(x, labels)

    # Add detailed error messages
    assert x_recon.shape == x.shape, \
        f"x_recon shape mismatch: expected {x.shape}, got {x_recon.shape}"
    
    assert mu.shape == (BATCH_SIZE, LATENT_DIM), \
        f"mu shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {mu.shape}"
    
    assert logvar.shape == (BATCH_SIZE, LATENT_DIM), \
        f"logvar shape mismatch: expected {(BATCH_SIZE, LATENT_DIM)}, got {logvar.shape}"
