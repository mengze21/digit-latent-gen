import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        learning_rate=1e-3,
        kl_weight=1.0,
        scheduler_config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.kl_weight = kl_weight
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = self._build_scheduler(scheduler_config)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def _build_scheduler(self, scheduler_config):
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get("type", "").lower()
        if not scheduler_type:
            return None

        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Only 'cosine' is supported.")

        t_max = scheduler_config.get("t_max")
        if t_max is None:
            raise ValueError("Cosine scheduler requires 't_max' in scheduler_config.")
        eta_min = scheduler_config.get("eta_min", 0.0)
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, step_scheduler=True):
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

        if self.scheduler is not None and step_scheduler:
            self.scheduler.step()

        return total_loss / len(self.train_loader.dataset)
    
    def compute_loss(self, data, recon, mu, logvar):
        """Compute reconstruction loss and KL divergence separately."""
        recon_loss = self.loss_fn(recon, data)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.kl_weight * kl_loss
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch()
            current_lr = self.get_current_learning_rate()
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
