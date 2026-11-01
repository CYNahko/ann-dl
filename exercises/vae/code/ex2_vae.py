import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import datasets, transforms
from pathlib import Path
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from vae import VAE, evaluate, train_vae

# Rode ex1_vae.py antes: este script carrega o modelo d = 16 salvo por ele
data_dir = Path(__file__).resolve().parents[4] / "data"   # sempre ann-dl/data
figures = Path(__file__).resolve().parents[1] / "figures"

train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
train, val = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128)
test_loader = DataLoader(test, batch_size=128)

#A
torch.manual_seed(42)   # o enunciado pede antes de construir cada modelo
model_2d = VAE(d=2)
train_vae(model_2d, train_loader, val_loader, epochs=20, learning_rate=1e-3)

checkpoint = torch.load(data_dir / "vae_d16.pt")
model_16d = VAE(d=16)
model_16d.load_state_dict(checkpoint["state"])
model_16d.eval()

test_recon, test_kl = evaluate(model_2d, test_loader)
print("Teste")
print(f"d = 2:  reconstrução = {test_recon:.2f} nats, KL = {test_kl:.2f} nats")
print(f"d = 16: reconstrução = {checkpoint['test_recon']:.2f} nats, KL = {checkpoint['test_kl']:.2f} nats")

#B
mus = []
with torch.no_grad():
    for x, _ in test_loader:
        mus.append(model_2d.encode(x)[0])
mus = torch.cat(mus).numpy()
labels = test.targets.numpy()

## Figura 4
plt.figure(figsize=(9, 7))
for digit in range(10):
    points = mus[labels == digit]
    plt.scatter(points[:, 0], points[:, 1], s=4, alpha=0.6,
                color=plt.get_cmap("tab10")(digit), label=str(digit))
plt.xlabel("$\\mu_1$")
plt.ylabel("$\\mu_2$")
plt.title("Figura 4 — Média latente das 10 000 imagens de teste (d = 2)")
plt.legend(title="Dígito", markerscale=3, loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(alpha=0.2)
plt.savefig(figures / "fig4_latente.png", dpi=300, bbox_inches="tight")
plt.show()

## Sobreposição: de que dígito são os 20 vizinhos mais próximos de cada ponto
neighbors = NearestNeighbors(n_neighbors=21).fit(mus).kneighbors(mus, return_distance=False)[:, 1:]
mixing = np.zeros((10, 10), dtype=int)
for digit, row in zip(labels, labels[neighbors]):
    mixing[digit] += np.bincount(row, minlength=10)
pairs = mixing + mixing.T
np.fill_diagonal(pairs, 0)
order = np.argsort(pairs[np.triu_indices(10, 1)])[::-1][:5]
rows, cols = np.triu_indices(10, 1)
print("Pares de dígitos mais misturados entre vizinhos:")
for k in order:
    print(f"{rows[k]} e {cols[k]}: {pairs[rows[k], cols[k]]}")

#C
q = np.linspace(0.05, 0.95, 15)
z_values = norm.ppf(q)
canvas = np.zeros((15 * 28, 15 * 28))

with torch.no_grad():
    for row, z2 in enumerate(z_values[::-1]):   # z2 cresce para cima, como na Figura 4
        for col, z1 in enumerate(z_values):
            z = torch.tensor([[z1, z2]], dtype=torch.float32)
            image = model_2d.decode(z).view(28, 28).numpy()
            canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = image

## Figura 5
ticks = np.arange(15) * 28 + 14
plt.figure(figsize=(9, 9))
plt.imshow(canvas, cmap="gray")
plt.xticks(ticks, [f"{v:.2f}" for v in z_values], rotation=90)
plt.yticks(ticks, [f"{v:.2f}" for v in z_values[::-1]])
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("Figura 5 — Plano latente decodificado na grade de $\\Phi^{-1}$ (d = 2)")
plt.savefig(figures / "fig5_plano.png", dpi=300, bbox_inches="tight")
plt.show()

#D
index_a, index_b = 3, 2   # dígitos 0 e 1 do conjunto de teste
x_a, digit_a = test[index_a]
x_b, digit_b = test[index_b]
print(f"Interpolação: índice {index_a} (dígito {digit_a}) e índice {index_b} (dígito {digit_b})")

ts = np.linspace(0, 1, 10)

with torch.no_grad():
    mu_a, _ = model_16d.encode(x_a)
    mu_b, _ = model_16d.encode(x_b)

    pixel_row = [(1 - t) * x_a.view(28, 28) + t * x_b.view(28, 28) for t in ts]
    latent_row = [model_16d.decode((1 - t) * mu_a + t * mu_b).view(28, 28) for t in ts]

## Figura 6
fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for col, t in enumerate(ts):
    axes[0, col].imshow(pixel_row[col], cmap="gray")
    axes[1, col].imshow(latent_row[col], cmap="gray")
    axes[0, col].set_title(f"t = {t:.2f}", fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].axis("off")

fig.text(0.1, 0.68, "Pixels", ha="right", va="center")
fig.text(0.1, 0.28, "Latente", ha="right", va="center")
fig.suptitle("Figura 6 — Interpolação nos pixels e no espaço latente (d = 16)")
plt.savefig(figures / "fig6_interpolacao.png", dpi=300, bbox_inches="tight")
plt.show()
