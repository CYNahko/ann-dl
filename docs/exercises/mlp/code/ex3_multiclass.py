import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

rng = np.random.default_rng(42)
figures = Path(__file__).resolve().parents[1] / "figures"

#A
X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=2, class_sep=1.0,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#B
model = MLP(
    layer_sizes=[4, 16, 3],
    rng=rng,
    learning_rate=0.5,
    epochs=2000,
    output="softmax"
)

#C
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mlp_acc = np.mean(y_pred == y_test)

print("MLP")
print("Arquitetura: 4-16-3 (tanh na oculta, softmax na saída)")
print("Taxa de aprendizado: 0.5 | Épocas: 2000")
print(f"Perda inicial: {model.loss_history[0]:.4f}")
print(f"Perda final: {model.loss_history[-1]:.4f}")
print(f"Acurácia no teste: {mlp_acc:.4f}")

## Figura 4
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, label="Perda de treino")
plt.xlabel("Época")
plt.ylabel("Entropia cruzada categórica")
plt.title("Figura 4 — Perda de treino por época (multiclasse)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(figures / "fig4_perda_multiclasse.png", dpi=300, bbox_inches="tight")
plt.show()

## Figura 5
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusão:\n", cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Número de amostras")
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks(range(3), ["Classe 0", "Classe 1", "Classe 2"])
plt.yticks(range(3), ["Classe 0", "Classe 1", "Classe 2"])
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.title("Figura 5 — Matriz de confusão (teste)")
plt.savefig(figures / "fig5_confusao.png", dpi=300, bbox_inches="tight")
plt.show()

#D
## Par mais confundido
errors = cm + cm.T
np.fill_diagonal(errors, 0)
i, j = np.unravel_index(errors.argmax(), errors.shape)
print(f"\nPar mais confundido: Classe {i} e Classe {j} ({errors[i, j]} erros)")

## Linha de base linear
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)
print(f"Linha de base (regressão logística): {baseline_acc:.4f}")
print(f"Ganho do MLP: {mlp_acc - baseline_acc:+.4f}")

## Centros aproximados dos 2 grupos de cada classe
from sklearn.cluster import KMeans
print("\nCentros aproximados dos grupos:")
for k in range(3):
    centers = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X[y == k]).cluster_centers_
    print(f"Classe {k}:\n{np.round(centers, 2)}")
