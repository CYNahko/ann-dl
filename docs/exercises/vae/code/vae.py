import torch
import torch.nn.functional as F
from torch import nn

class VAE(nn.Module):
    def __init__(self, d=16):
        super().__init__()

        # Encoder: 784 -> 512 -> 256
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Duas cabeças: 256 -> d
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)

        # Decoder: d -> 256 -> 512 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    batch_size = x.size(0)
    x = x.view(-1, 784)

    # Reconstrução: BCE somada sobre os pixels, média sobre as imagens
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / batch_size

    # KL em forma fechada: 1/2 * soma(mu^2 + sigma^2 - 1 - log sigma^2)
    kl = 0.5 * torch.sum(mu**2 + logvar.exp() - 1 - logvar) / batch_size

    return recon_loss + beta * kl, recon_loss, kl


def evaluate(model, loader):
    model.eval()
    recon_total, kl_total, n = 0.0, 0.0, 0

    with torch.no_grad():
        for x, _ in loader:
            recon, mu, logvar = model(x)
            _, recon_loss, kl = vae_loss(recon, x, mu, logvar)

            recon_total += recon_loss.item() * x.size(0)
            kl_total += kl.item() * x.size(0)
            n += x.size(0)

    return recon_total / n, kl_total / n


def train_vae(model, train_loader, val_loader, epochs=20, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}

    for epoch in range(epochs):
        model.train()
        recon_total, kl_total, n = 0.0, 0.0, 0

        for x, _ in train_loader:
            recon, mu, logvar = model(x)
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_total += recon_loss.item() * x.size(0)
            kl_total += kl.item() * x.size(0)
            n += x.size(0)

        val_recon, val_kl = evaluate(model, val_loader)

        history["train_recon"].append(recon_total / n)
        history["train_kl"].append(kl_total / n)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"Época {epoch + 1}: "
            f"treino rec = {recon_total / n:.2f}, KL = {kl_total / n:.2f} | "
            f"validação rec = {val_recon:.2f}, KL = {val_kl:.2f}"
        )

    return history
