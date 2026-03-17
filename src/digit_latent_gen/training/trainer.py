import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, device, learning_rate=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data, labels)
            
            # Compute loss
            loss = self.compute_loss(data, recon_batch, mu, logvar)

            if torch.isnan(loss):
                print("NaN detected!")
                print("mu max:", mu.max(), "min:", mu.min())
                print("logvar max:", logvar.max(), "min:", logvar.min())
                break
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            self.optimizer.step()
        
        return total_loss / len(self.train_loader.dataset)
    
    def compute_loss(self, data, recon, mu, logvar):
        """Compute reconstruction loss and KL divergence separately."""
        # Reconstruction loss (per sample)
        recon_loss = self.loss_fn(recon, data)
        
        # KL divergence (per sample)
        # See Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
