import torch
from torch.utils.data import DataLoader
from digit_latent_gen.models.vae import VAE
from digit_latent_gen.training.trainer import Trainer
from digit_latent_gen.common.utils import get_mnist_dataset


def main():
    # Configuration
    latent_dim = 32
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset = get_mnist_dataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VAE(latent_dim=latent_dim).to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(num_epochs)
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), "checkpoints/vae_model.pt")
    print("Model saved to checkpoints/vae_model.pt")


if __name__ == "__main__":
    main()
