import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic block for U-Net."""
    def __init__(self, in_ch, out_ch, time_emb_dim=None, num_classes=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # Time embedding
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_mlp = None
            
        # Class embedding
        if num_classes is not None:
            self.class_mlp = nn.Linear(num_classes, out_ch)
        else:
            self.class_mlp = None
            
        self.act = nn.SiLU()
        
        # Residual connection
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(self.act(time_emb))[:, :, None, None]
        if self.class_mlp is not None and class_emb is not None:
            h = h + self.class_mlp(self.act(class_emb))[:, :, None, None]
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(self.act(time_emb))[:, :, None, None]
        if self.class_mlp is not None and class_emb is not None:
            h = h + self.class_mlp(self.act(class_emb))[:, :, None, None]
        h = self.act(h)
        
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion model."""
    def __init__(self, img_channels=1, base_ch=32, 
                 time_emb_dim=256, num_classes=10, latent_dim=32):
        super().__init__()
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Class embeddings
        self.class_emb = nn.Embedding(num_classes, num_classes)
        
        # Initial projection
        self.conv0 = nn.Conv2d(img_channels, base_ch, 3, padding=1)
        
        # Down samples
        self.down1 = Block(base_ch, base_ch * 2, time_emb_dim, num_classes)
        self.down2 = Block(base_ch * 2, base_ch * 4, time_emb_dim, num_classes)
        self.down3 = Block(base_ch * 4, base_ch * 8, time_emb_dim, num_classes)
        
        self.down_sample1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)
        self.down_sample2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)
        self.down_sample3 = nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)
        
        # Bottleneck
        self.bot1 = Block(base_ch * 8, base_ch * 8, time_emb_dim, num_classes)
        self.bot2 = Block(base_ch * 8, base_ch * 8, time_emb_dim, num_classes)
        
        # Up samples
        self.up_sample1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1)
        self.up1 = Block(base_ch * 8, base_ch * 4, time_emb_dim, num_classes)
        
        self.up_sample2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.up2 = Block(base_ch * 4, base_ch * 2, time_emb_dim, num_classes)
        
        self.up_sample3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.up3 = Block(base_ch * 2, base_ch, time_emb_dim, num_classes)
        
        # Output
        self.conv_out = nn.Conv2d(base_ch, img_channels, 1)
        
        # Latent projection (for latent diffusion)
        if latent_dim is not None:
            self.latent_proj = nn.Linear(latent_dim, time_emb_dim)
        else:
            self.latent_proj = None

    def forward(self, x, time, class_labels=None, latent_z=None):
        # Embeddings
        time_emb = self.time_mlp(time)
        
        # Class embeddings
        class_emb = None
        if class_labels is not None:
            class_emb = self.class_emb(class_labels)
        
        # Latent embeddings
        if latent_z is not None and self.latent_proj is not None:
            latent_emb = self.latent_proj(latent_z)
            time_emb = time_emb + latent_emb
        
        # Initial conv
        x = self.conv0(x)
        
        # Down blocks
        x1 = self.down1(x, time_emb, class_emb)
        x = self.down_sample1(x1)
        
        x2 = self.down2(x, time_emb, class_emb)
        x = self.down_sample2(x2)
        
        x3 = self.down3(x, time_emb, class_emb)
        x = self.down_sample3(x3)
        
        # Bottleneck
        x = self.bot1(x, time_emb, class_emb)
        x = self.bot2(x, time_emb, class_emb)
        
        # Up blocks
        x = self.up_sample1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x, time_emb, class_emb)
        
        x = self.up_sample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, time_emb, class_emb)
        
        x = self.up_sample3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x, time_emb, class_emb)
        
        # Output
        out = self.conv_out(x)
        
        return out


class DiffusionModel(nn.Module):
    """Diffusion model for image generation."""
    def __init__(self, 
                 img_channels=1,
                 img_size=32,
                 base_ch=32,
                 time_steps=1000,
                 latent_dim=32,
                 num_classes=10,
                 device='cpu'):
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.base_ch = base_ch
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        
        # U-Net model
        self.model = UNet(
            img_channels=img_channels,
            base_ch=base_ch,
            time_emb_dim=256,
            num_classes=num_classes,
            latent_dim=latent_dim
        )
        
        # Noise schedule
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Compute beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, time_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Computed quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_sample(self, x_t, t, class_labels=None, latent_z=None, noise=None):
        """Sample from reverse diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_t)
        
        # Predict noise
        noise_pred = self.model(x_t, t, class_labels, latent_z)
        
        # Compute mean
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        # Sample
        variance = self.posterior_variance[t]
        std = torch.sqrt(variance)
        
        return mean + std * noise

    def p_sample_loop(self, shape, class_labels=None, latent_z=None):
        """Full reverse diffusion process."""
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.time_steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, class_labels, latent_z)
            
        return img

    def p_sample_loop_conditional(self, shape, class_labels, latent_z=None):
        """Generate images with class labels and optional latent conditioning."""
        return self.p_sample_loop(shape, class_labels, latent_z)

    def compute_loss(self, x_start, t, class_labels=None, latent_z=None):
        """Compute training loss."""
        noise = torch.randn_like(x_start)
        x_noisy, _ = self.q_sample(x_start, t, noise)
        
        noise_pred = self.model(x_noisy, t, class_labels, latent_z)
        
        return F.mse_loss(noise_pred, noise)

    def forward(self, x_start, class_labels=None, latent_z=None):
        """Forward pass for training."""
        batch_size = x_start.size(0)
        t = torch.randint(0, self.time_steps, (batch_size,), device=self.device).long()
        
        return self.compute_loss(x_start, t, class_labels, latent_z)

    def generate(self, num_samples, class_labels=None, latent_z=None):
        """Generate new images."""
        shape = (num_samples, self.img_channels, self.img_size, self.img_size)
        
        if class_labels is None:
            class_labels = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        
        return self.p_sample_loop_conditional(shape, class_labels, latent_z)