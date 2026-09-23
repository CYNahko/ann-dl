import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from perceptron_class import Perceptron

rng = np.random.default_rng(seed=42)
figures = Path(__file__).resolve().parents[1] / "figures"

# A
means = np.array([[1.5, 1.5], [5,5]])
covariance = np.array([
    [0.5, 0],
    [0, 0.5]
])
n = 1000
class0 = rng.multivariate_normal(mean=means[0], cov=covariance, size=n)
class1 = rng.multivariate_normal(mean=means[1], cov=covariance, size=n)

## Figura 1
plt.figure(figsize=(8,6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Classe 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Classe 1')
plt.title("Figura 1 - Gráfico de dispersão dos 2000 pontos")
plt.legend()
plt.savefig(figures / 'dispersion_grafic.png', dpi=300, bbox_inches='tight')
plt.show()

# B

x = np.vstack([class0, class1])
y = np.concatenate([
    np.zeros(n, dtype=int),
    np.ones(n, dtype=int)
])

perceptron = Perceptron(
    rng=rng,
    learning_rate=0.01,
    max_epochs=100
)

perceptron.fit(x, y)

print("Pesos:", perceptron.weights)
print("Bias:", perceptron.bias)
print("Acurácia final:", perceptron.accuracy_history[-1])
print("Épocas executadas:", len(perceptron.accuracy_history))

# Resultados
print("\nResultados finais")
print("w final:", perceptron.weights)
print("b final:", perceptron.bias)
print("Número de épocas:", len(perceptron.accuracy_history))
print("Acurácia final:", perceptron.accuracy_history[-1])

# Figura 2

y_pred = perceptron.predict(x)

misclassified = y_pred != y

plt.figure(figsize=(8, 6))

plt.scatter(
    x[y == 0, 0],
    x[y == 0, 1],
    label="Classe 0",
    alpha=0.5
)

plt.scatter(
    x[y == 1, 0],
    x[y == 1, 1],
    label="Classe 1",
    alpha=0.5
)

plt.scatter(
    x[misclassified, 0],
    x[misclassified, 1],
    facecolors="none",
    edgecolors="red",
    linewidths=2,
    s=100,
    label="Mal classificados"
)

x1 = np.linspace(
    x[:, 0].min(),
    x[:, 0].max(),
    200
)

w1, w2 = perceptron.weights
b = perceptron.bias

x2 = -(w1 * x1 + b) / w2

plt.plot(
    x1,
    x2,
    color="black",
    linewidth=2,
    label="Fronteira de decisão"
)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Figura 2 — Fronteira de decisão do perceptron")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(figures / 'decision_frontier_perceptron.png', dpi=300, bbox_inches='tight')
plt.show()

#Figura 3
epochs = np.arange(
    1,
    len(perceptron.accuracy_history) + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    perceptron.accuracy_history,
    marker="o"
)

plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Figura 3 — Acurácia por época")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.savefig(figures / 'accuracy_per_epoch.png', dpi=300, bbox_inches='tight')
plt.show()

# D - 2
perceptron_001 = Perceptron(
    rng=rng,
    learning_rate=0.01,
    max_epochs=100
)

perceptron_1 = Perceptron(
    rng=rng,
    learning_rate=1.0,
    max_epochs=100
)

perceptron_001.fit(x, y)
perceptron_1.fit(x, y)

direction_001 = (
    perceptron_001.weights
    / np.linalg.norm(perceptron_001.weights)
)

direction_1 = (
    perceptron_1.weights
    / np.linalg.norm(perceptron_1.weights)
)

print("\nη = 0.01")
print("Épocas:", len(perceptron_001.accuracy_history))
print("Acurácia:", perceptron_001.accuracy_history[-1])
print("Pesos:", perceptron_001.weights)
print("Direção de w:", direction_001)

print("\nη = 1.0")
print("Épocas:", len(perceptron_1.accuracy_history))
print("Acurácia:", perceptron_1.accuracy_history[-1])
print("Pesos:", perceptron_1.weights)
print("Direção de w:", direction_1)