import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from mlp import MLP
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
figures = Path(__file__).resolve().parents[1] / "figures"

#A
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)      # classe 0: 1 grupo | classe 1: 2 grupos

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

## Figura 1
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", alpha=0.6, label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", alpha=0.6, label="Classe 1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Figura 1 — Dispersão dos 1000 pontos")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(figures / "fig1_dispersao.png", dpi=300, bbox_inches="tight")
plt.show()

#B
baseline = LogisticRegression()
baseline.fit(X_train, y_train)

baseline_acc = baseline.score(X_test, y_test)

print("Linha de base (regressão logística)")
print(f"Pesos: {baseline.coef_[0]}")
print(f"Bias: {baseline.intercept_[0]:.4f}")
print(f"Acurácia no teste: {baseline_acc:.4f}")

#C
model = MLP(
    layer_sizes=[2, 8, 1],
    rng=rng,
    learning_rate=0.5,
    epochs=2000,
    output="sigmoid"
)
model.fit(X_train, y_train)

mlp_acc = np.mean(model.predict(X_test) == y_test)

print("\nMLP")
print("Arquitetura: 2-8-1 (tanh na oculta, sigmoide na saída)")
print("Taxa de aprendizado: 0.5 | Épocas: 2000")
print(f"Perda inicial: {model.loss_history[0]:.4f}")
print(f"Perda final: {model.loss_history[-1]:.4f}")
print(f"Acurácia no teste: {mlp_acc:.4f}")

## Figura 2
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, label="Perda de treino")
plt.xlabel("Época")
plt.ylabel("Entropia cruzada binária")
plt.title("Figura 2 — Perda de treino por época")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(figures / "fig2_perda_binario.png", dpi=300, bbox_inches="tight")
plt.show()

## Figura 3
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)
grid = np.column_stack([xx1.ravel(), xx2.ravel()])
regions = model.predict(grid).reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, regions, levels=[-0.5, 0.5, 1.5], colors=["#9ecae1", "#fcbba1"], alpha=0.6)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", edgecolors="k", label="Classe 0 (teste)")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", edgecolors="k", label="Classe 1 (teste)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Figura 3 — Fronteira de decisão do MLP")
plt.legend()
plt.savefig(figures / "fig3_fronteira.png", dpi=300, bbox_inches="tight")
plt.show()
