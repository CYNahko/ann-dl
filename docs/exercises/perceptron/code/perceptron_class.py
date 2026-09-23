import numpy as np

class Perceptron:
    def __init__(self, rng, learning_rate=0.01, max_epochs=100):
        self.rng = rng
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def step(self, z):
        return 1 if z >= 0 else 0

    def predict_sample(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.step(z)

    def predict(self, X):
        return np.array([
            self.predict_sample(x)
            for x in X
        ])

    def fit(self, X, y):
        self.weights = self.rng.normal(
            0,
            0.01,
            size=X.shape[1]
        )

        self.bias = 0.0

        self.accuracy_history = []
        self.updates_history = []
        self.pocket_accuracy_history = []

        # Estado inicial do pocket
        initial_predictions = self.predict(X)

        self.best_accuracy = np.mean(
            initial_predictions == y
        )

        self.best_weights = self.weights.copy()
        self.best_bias = self.bias
        self.best_epoch = 0

        for epoch in range(self.max_epochs):
            updates = 0

            for x_i, y_i in zip(X, y):
                y_pred = self.predict_sample(x_i)
                error = y_i - y_pred

                if error != 0:
                    # Regra original do perceptron
                    self.weights += (
                        self.learning_rate * error * x_i
                    )

                    self.bias += (
                        self.learning_rate * error
                    )

                    updates += 1

                    predictions = self.predict(X)
                    accuracy = np.mean(predictions == y)

                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_weights = self.weights.copy()
                        self.best_bias = self.bias
                        self.best_epoch = epoch + 1

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)

            self.accuracy_history.append(accuracy)
            self.updates_history.append(updates)
            self.pocket_accuracy_history.append(
                self.best_accuracy
            )

            print(
                f"Época {epoch + 1}: "
                f"acurácia = {accuracy:.4f}, "
                f"pocket = {self.best_accuracy:.4f}, "
                f"atualizações = {updates}"
            )

            if updates == 0:
                print("Treinamento convergiu.")
                break

        return self