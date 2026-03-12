import torch
import pytest

def test_model_evaluation(model, dataloader, loss_fn, device):
    """Test model evaluation loop."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle VAE output format (x_recon, mu, logvar) vs classification output
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                # VAE model - use reconstruction for loss calculation
                x_recon = outputs[0]
                loss = loss_fn(x_recon, images)
            else:
                # Classification model
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            total += labels.size(0)
    
    # Calculate and return accuracy (for classification models)
    if correct > 0 or total > 0:
        accuracy = 100 * correct / total
        return accuracy
    return None


def test_vae_forward_pass():
    """Test VAE model forward pass."""
    from digit_latent_gen.models.vae import VAE
    
    model = VAE(latent_dim=32, num_classes=10)
    model.eval()
    
    # Create dummy inputs
    x = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([1, 7], dtype=torch.long)
    
    # Forward pass
    x_recon, mu, logvar = model(x, labels)
    
    # Assert output shapes
    assert x_recon.shape == x.shape, f"Reconstruction shape mismatch: {x_recon.shape} vs {x.shape}"
    assert mu.shape == (2, 32), f"Mu shape mismatch: {mu.shape}"
    assert logvar.shape == (2, 32), f"Logvar shape mismatch: {logvar.shape}"


def test_encoder_forward_pass():
    """Test Encoder model forward pass."""
    from digit_latent_gen.models.vae import Encoder
    
    model = Encoder(latent_dim=32, num_classes=10)
    model.eval()
    
    # Create dummy inputs
    x = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([1, 7], dtype=torch.long)
    
    # Forward pass
    mu, logvar = model(x, labels)
    
    # Assert output shapes
    assert mu.shape == (2, 32), f"Mu shape mismatch: {mu.shape}"
    assert logvar.shape == (2, 32), f"Logvar shape mismatch: {logvar.shape}"


def test_decoder_forward_pass():
    """Test Decoder model forward pass."""
    from digit_latent_gen.models.vae import Decoder
    
    model = Decoder(latent_dim=32, num_classes=10)
    model.eval()
    
    # Create dummy inputs
    z = torch.randn(2, 32)
    labels = torch.tensor([1, 7], dtype=torch.long)
    
    # Forward pass
    x_recon = model(z, labels)
    
    # Assert output shape
    assert x_recon.shape == (2, 1, 28, 28), f"Reconstruction shape mismatch: {x_recon.shape}"
