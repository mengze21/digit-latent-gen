import torch
import sys
sys.path.insert(0, 'src')

from digit_latent_gen.models.vae import VAE  # Assume VAE is the main model class; please confirm the actual class name

def test_vae_forward_shape():
    # Create dummy input: batch_size=2, channels=3, height=64, width=64
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 64, 64)

    # Initialize model (adjust parameters according to your VAE __init__)
    model = VAE(input_channels=3, latent_dim=128)

    # Forward pass
    output, mu, logvar = model(input_tensor)

    # Assertions: output shape should match input
    assert output.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output.shape}"
    assert mu.shape == (batch_size, 128), f"Expected mu shape (2, 128), got {mu.shape}"
    assert logvar.shape == (batch_size, 128), f"Expected logvar shape (2, 128), got {logvar.shape}"
