import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MLP

figures = Path(__file__).resolve().parents[1] / "figures"

# Mesmos dados e mesma divisão do Exercício 3
X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=2, class_sep=1.0,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Só muda a profundidade; taxa e épocas iguais às do Exercício 3
architectures = {
    "Ex. 3 (4-16-3)": [4, 16, 3],
    "Ex. 4 (4-16-16-3)": [4, 16, 16, 3],
}
seeds = [42, 43, 44]
results = {}

#A e B
for name, sizes in architectures.items():
    accs, final_losses, curves = [], [], []

    for seed in seeds:
        model = MLP(
            layer_sizes=sizes,
            rng=np.random.default_rng(seed),
            learning_rate=0.5,
            epochs=2000,
            output="softmax"
        )
        model.fit(X_train, y_train)

        accs.append(np.mean(model.predict(X_test) == y_test))
        final_losses.append(model.loss_history[-1])
        curves.append(model.loss_history)

    n_params = sum(n_in * n_out + n_out for n_in, n_out in zip(sizes[:-1], sizes[1:]))

    results[name] = {
        "hidden": len(sizes) - 2,
        "params": n_params,
        "accs": accs,
        "loss": np.mean(final_losses),
        "acc": np.mean(accs),
        "std": np.std(accs),
        "curve": np.mean(curves, axis=0),
    }

    print(f"\n{name}")
    print(f"Acurácias por semente {seeds}: {np.round(accs, 4)}")
    print(f"Perdas finais por semente: {np.round(final_losses, 4)}")

## Tabela comparativa
print("\nRede | Camadas ocultas | Parâmetros | Perda final (média) | Acurácia teste (média ± dp)")
for name, r in results.items():
    print(f"{name} | {r['hidden']} | {r['params']} | {r['loss']:.4f} | {r['acc']:.4f} ± {r['std']:.4f}")

diff = results["Ex. 4 (4-16-16-3)"]["acc"] - results["Ex. 3 (4-16-3)"]["acc"]
print(f"\nDiferença de acurácia (Ex. 4 − Ex. 3): {diff:+.4f}")

## Figura 6
plt.figure(figsize=(8, 5))
for name, r in results.items():
    plt.plot(r["curve"], label=name)
plt.xlabel("Época")
plt.ylabel("Entropia cruzada categórica")
plt.title("Figura 6 — Perda de treino média (3 sementes)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(figures / "fig6_comparacao.png", dpi=300, bbox_inches="tight")
plt.show()
