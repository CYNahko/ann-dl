import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import datasets, transforms
from pathlib import Path
from vae import VAE, evaluate, train_vae

data_dir = Path(__file__).resolve().parents[4] / "data"   # sempre ann-dl/data
figures = Path(__file__).resolve().parents[1] / "figures"

#A
train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
train, val = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42))

#B
torch.manual_seed(42)   # o enunciado pede antes de construir cada modelo
model = VAE(d=16)

#C
train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128)
test_loader = DataLoader(test, batch_size=128)

history = train_vae(model, train_loader, val_loader, epochs=20, learning_rate=1e-3)

# O teste só é usado aqui, para os números finais
test_recon, test_kl = evaluate(model, test_loader)
print("Teste (d = 16)")
print(f"Reconstrução: {test_recon:.2f} nats")
print(f"KL: {test_kl:.2f} nats")
print(f"ELBO negativo: {test_recon + test_kl:.2f} nats")
print(f"KL de validação: época 1 = {history['val_kl'][0]:.2f}, época 20 = {history['val_kl'][-1]:.2f}")

# Guarda o modelo e os números; o Exercício 2 usa os dois
torch.save(
    {"state": model.state_dict(), "test_recon": test_recon, "test_kl": test_kl},
    data_dir / "vae_d16.pt"
)

## Figura 1
epochs = np.arange(1, 21)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(epochs, history["train_recon"], label="Treino")
axes[0].plot(epochs, history["val_recon"], label="Validação")
axes[0].set_title("Reconstrução por época")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Reconstrução (nats por imagem)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history["train_kl"], label="Treino")
axes[1].plot(epochs, history["val_kl"], label="Validação")
axes[1].set_title("KL por época")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("KL (nats por imagem)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Figura 1 — Termos da perda por época (d = 16)")
plt.savefig(figures / "fig1_perdas.png", dpi=300, bbox_inches="tight")
plt.show()

#D
model.eval()

## Figura 2
labels = test.targets.numpy()
first_of_each = [int(np.where(labels == digit)[0][0]) for digit in range(10)]
originals = torch.stack([test[i][0] for i in first_of_each])

with torch.no_grad():
    mu, _ = model.encode(originals)
    reconstructions = model.decode(mu)   # decodifica mu, sem sortear z

fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for digit in range(10):
    axes[0, digit].imshow(originals[digit].squeeze(), cmap="gray")
    axes[1, digit].imshow(reconstructions[digit].view(28, 28), cmap="gray")
    axes[0, digit].set_title(str(digit))
    axes[0, digit].axis("off")
    axes[1, digit].axis("off")

fig.text(0.1, 0.68, "Original", ha="right", va="center")
fig.text(0.1, 0.28, "Reconstrução", ha="right", va="center")
fig.suptitle("Figura 2 — Imagens de teste e reconstruções a partir de μ (d = 16)")
plt.savefig(figures / "fig2_reconstrucoes.png", dpi=300, bbox_inches="tight")
plt.show()

## Figura 3
torch.manual_seed(0)
z = torch.randn(64, 16)

with torch.no_grad():
    samples = model.decode(z)

fig, axes = plt.subplots(8, 8, figsize=(7, 7.4))
for ax, sample in zip(axes.ravel(), samples):
    ax.imshow(sample.view(28, 28), cmap="gray")
    ax.axis("off")

fig.suptitle("Figura 3 — Amostras decodificadas de z ~ N(0, I) (d = 16)")
plt.savefig(figures / "fig3_amostras.png", dpi=300, bbox_inches="tight")
plt.show()
