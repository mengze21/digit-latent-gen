import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder module for VAE.
    Args:
        latent_dim: dimension of the latent representation
    """

    def __init__(self, latent_dim, num_classes=10):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # input image: [B, 1, 32, 32]
        # label map:   [B, 10, 32, 32]
        # concat ->    [B, 11, 32, 32]

        # label embedding: [B] -> [B, num_classes]
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.enc_conv1 = nn.Conv2d(
            1 + num_classes, 32, kernel_size=4, stride=2, padding=1
        )  # [B, 32, 16, 16]
        self.enc_conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # [B, 64, 8, 8]

        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x, labels):
        # x: [B, 1, 32, 32]
        # labels: [B] -> embedded labels [B, num_classes]
        label_emb = self.label_emb(labels)  # [B, num_classes]
        label_map = label_emb.unsqueeze(2).unsqueeze(3)  # [B, num_classes, 1, 1]
        label_map = label_map.expand(
            -1, -1, x.size(2), x.size(3)
        )  # [B, num_classes, 32, 32]

        x_in = torch.cat([x, label_map], dim=1)  # [B, 1 + num_classes, 32, 32]
        h = nn.ReLU()(self.enc_conv1(x_in))  # [B, 32, 16, 16]
        h = nn.ReLU()(self.enc_conv2(h))  # [B, 64, 8, 8]

        h = h.flatten(1)  # [B, 64 * 8 * 8]
        mu = self.fc_mu(h)  # [B, latent_dim]
        logvar = self.fc_logvar(h)  # [B, latent_dim]

        return mu, logvar


class Decoder(nn.Module):
    """Decoder module for VAE.
    Args:
        latent_dim: dimension of the latent representation
    """

    def __init__(self, latent_dim, num_classes=10):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # latent vector: [B, latent_dim]
        # label embedding: [B, num_classes]
        # concat -> [B, latent_dim + num_classes]
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Linear(latent_dim + num_classes, 64 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # [B, 32, 16, 16]
        self.dec_conv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )  # [B, 1, 32, 32]

    def forward(self, z, labels):
        # z: [B, latent_dim]
        # labels: [B]
        label_emb = self.label_emb(labels)  # [B, num_classes]
        z = torch.cat([z, label_emb], dim=1)
        x = F.relu(self.fc(z))
        x = x.reshape(-1, 64, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = torch.sigmoid(self.dec_conv2(x))  # Output in [0, 1]
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, labels)
        return x_recon, mu, logvar
