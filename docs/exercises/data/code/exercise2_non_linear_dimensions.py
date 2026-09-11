import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

rng = np.random.default_rng(seed=42)
figures = Path(__file__).resolve().parents[1] / "figures"

mean_a = np.array([0,0,0,0,0])
mean_b = np.array([1.5,1.5,1.5,1.5,1.5])

cov_matrix_a = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0]
])

cov_matrix_b = np.array([
    [ 1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7,  1.5, 0.4, 0.0, 0.0],
    [ 0.2,  0.4, 1.5, 0.6, 0.0],
    [ 0.0,  0.0, 0.6, 1.5, 0.3],
    [ 0.0,  0.0, 0.0, 0.3, 1.5]
])

n = 500
# A
dataset_a = rng.multivariate_normal(mean=mean_a, cov=cov_matrix_a, size=n)
dataset_b = rng.multivariate_normal(mean=mean_b, cov=cov_matrix_b, size=n)
# B
v_c = rng.normal(size=(500, 5))
sizes_c = np.linalg.norm(v_c, axis=1, keepdims=True)
u_c = v_c / sizes_c
v_d = rng.normal(size=(500, 5))
u_d = v_d / np.linalg.norm(v_d, axis=1, keepdims=True)
radius_c = rng.normal(
    loc=2.0,
    scale=0.4,
    size=(500, 1)
)
radius_d = rng.normal(
    loc=5.0,
    scale=0.4,
    size=(500, 1)
)
dataset_c = radius_c * u_c
dataset_d = radius_d * u_d

dataset_1 = np.vstack([dataset_a, dataset_b])
pca_1 = PCA(n_components=2)
dataset_1_2d = pca_1.fit_transform(dataset_1)

a_2d = dataset_1_2d[:500]
b_2d = dataset_1_2d[500:]

dataset_2 = np.vstack([dataset_c, dataset_d])

pca_2 = PCA(n_components=2)
dataset_2_2d = pca_2.fit_transform(dataset_2)

c_2d = dataset_2_2d[:500]
d_2d = dataset_2_2d[500:]

# Figura 4
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    a_2d[:, 0],
    a_2d[:, 1],
    alpha=0.6,
    label="Classe A"
)

axes[0].scatter(
    b_2d[:, 0],
    b_2d[:, 1],
    alpha=0.6,
    label="Classe B"
)

axes[0].set_title("Dataset I após PCA")
axes[0].set_xlabel("Primeiro componente")
axes[0].set_ylabel("Segundo componente")
axes[0].legend()


axes[1].scatter(
    c_2d[:, 0],
    c_2d[:, 1],
    alpha=0.6,
    label="Classe C"
)

axes[1].scatter(
    d_2d[:, 0],
    d_2d[:, 1],
    alpha=0.6,
    label="Classe D"
)

axes[1].set_title("Dataset II após PCA")
axes[1].set_xlabel("Primeiro componente")
axes[1].set_ylabel("Segundo componente")
axes[1].legend()

fig.suptitle("Projeção dos datasets com PCA")
plt.tight_layout()
fig.savefig(figures / 'datasets_pca.png',  dpi=300, bbox_inches='tight')
plt.show()

#Variancia explicada
variance_1 = pca_1.explained_variance_ratio_
variance_2 = pca_2.explained_variance_ratio_

print("Dataset I:")
print(f"PC1: {pca_1.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca_1.explained_variance_ratio_[1]:.2%}")
print(f"Total: {pca_1.explained_variance_ratio_.sum():.2%}")

print("\nDataset II:")
print(f"PC1: {pca_2.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca_2.explained_variance_ratio_[1]:.2%}")
print(f"Total: {pca_2.explained_variance_ratio_.sum():.2%}")

#Distancia entre os centros
center_a = np.mean(dataset_a, axis=0)
center_b = np.mean(dataset_b, axis=0)

distance_centers_1 = np.linalg.norm(center_a - center_b)

print(f"Distância A-B: {distance_centers_1:.4f}")

center_c = np.mean(dataset_c, axis=0)
center_d = np.mean(dataset_d, axis=0)

distance_centers_2 = np.linalg.norm(center_c - center_d)

print(f"Distância C-D: {distance_centers_2:.4f}")

radius_a = np.linalg.norm(dataset_a, axis=1)
radius_b = np.linalg.norm(dataset_b, axis=1)

radius_c = np.linalg.norm(dataset_c, axis=1)
radius_d = np.linalg.norm(dataset_d, axis=1)

#Figura 5
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(radius_a, bins=30, alpha=0.6, label="Classe A")
axes[0].hist(radius_b, bins=30, alpha=0.6, label="Classe B")
axes[0].set_title("Dataset I")
axes[0].set_xlabel("Raio")
axes[0].set_ylabel("Quantidade de pontos")
axes[0].legend
axes[0].legend()

axes[1].hist(radius_c, bins=30, alpha=0.6, label="Classe C")
axes[1].hist(radius_d, bins=30, alpha=0.6, label="Classe D")
axes[1].set_title("Dataset II")
axes[1].set_xlabel("Raio")
axes[1].set_ylabel("Quantidade de pontos")
axes[1].legend()

fig.suptitle("Distribuição dos raios")
plt.tight_layout()
fig.savefig(figures / 'radius_distributions.png', dpi=300, bbox_inches='tight')
plt.show()