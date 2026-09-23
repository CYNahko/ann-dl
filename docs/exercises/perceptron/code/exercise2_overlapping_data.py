import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from perceptron_class import Perceptron

rng = np.random.default_rng(seed=42)
figures = Path(__file__).resolve().parents[1] / "figures"

# A 
means = np.array([[3, 3], [4,4]])
covariance = np.array([
    [1.5, 0],
    [0, 1.5]
])
n = 1000
class0 = rng.multivariate_normal(mean=means[0], cov=covariance, size=n)
class1 = rng.multivariate_normal(mean=means[1], cov=covariance, size=n)

## Figura 4
plt.figure(figsize=(8,6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Classe 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Classe 1')
plt.title("Figura 4 - Gráfico de dispersão dos 2000 pontos")
plt.legend()
plt.savefig(figures / 'dispersion_grafic_2.png', dpi=300, bbox_inches='tight')
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

final_predictions = perceptron.predict(x)
final_accuracy = np.mean(final_predictions == y)

print("Pesos finais:", perceptron.weights)
print("Bias final:", perceptron.bias)
print("Acurácia final:", final_accuracy)

# Pocket
print("Pesos do pocket:", perceptron.best_weights)
print("Bias do pocket:", perceptron.best_bias)
print("Acurácia do pocket:", perceptron.best_accuracy)
print("Melhor época:", perceptron.best_epoch)

# Previsão usando parâmetros específicos
def predict_with_parameters(X, weights, bias):
    z = X @ weights + bias
    return (z >= 0).astype(int)


final_predictions = predict_with_parameters(
    x,
    perceptron.weights,
    perceptron.bias
)

pocket_predictions = predict_with_parameters(
    x,
    perceptron.best_weights,
    perceptron.best_bias
)

final_misclassified = final_predictions != y
pocket_misclassified = pocket_predictions != y

x1 = np.linspace(
    x[:, 0].min(),
    x[:, 0].max(),
    200
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    sharex=True,
    sharey=True
)

models = [
    (
        perceptron.weights,
        perceptron.bias,
        final_misclassified,
        "Pesos finais"
    ),
    (
        perceptron.best_weights,
        perceptron.best_bias,
        pocket_misclassified,
        "Pesos do pocket"
    )
]

for ax, (weights, bias, misclassified, title) in zip(
    axes,
    models
):
    ax.scatter(
        x[y == 0, 0],
        x[y == 0, 1],
        alpha=0.4,
        label="Classe 0"
    )

    ax.scatter(
        x[y == 1, 0],
        x[y == 1, 1],
        alpha=0.4,
        label="Classe 1"
    )

    ax.scatter(
        x[misclassified, 0],
        x[misclassified, 1],
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        s=70,
        label="Mal classificados"
    )

    w1, w2 = weights
    x2 = -(w1 * x1 + bias) / w2

    ax.plot(
        x1,
        x2,
        color="black",
        linewidth=2,
        label="Fronteira"
    )

    accuracy = np.mean(
        predict_with_parameters(x, weights, bias) == y
    )

    ax.set_title(
        f"{title} — acurácia: {accuracy:.4f}"
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.2)
    ax.legend()

fig.suptitle(
    "Figura 5 — Fronteiras final e pocket",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    figures / "final_and_pocket_frontiers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Figura 6
epochs = np.arange(
    1,
    len(perceptron.accuracy_history) + 1
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    perceptron.accuracy_history,
    label="Pesos atuais",
    linewidth=1.5,
    alpha=0.8
)

plt.plot(
    epochs,
    perceptron.pocket_accuracy_history,
    label="Melhor até agora (pocket)",
    linewidth=2.5
)

plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Figura 6 — Acurácia atual e acurácia do pocket")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()

plt.savefig(
    figures / "current_and_pocket_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()