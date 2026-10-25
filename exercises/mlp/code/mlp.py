import numpy as np

rng = np.random.default_rng(42)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1-a)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, layer_sizes, rng, learning_rate=0.1, epochs=1000, output="sigmoid"):
        # layer_sizes = [entrada, oculta_1, ..., saída], ex.: [2, 8, 1]
        self.layer_sizes = layer_sizes
        self.rng = rng
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output = output

        # Um W e um b para cada par de camadas consecutivas
        self.weights = []
        self.biases = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Pesos pequenos e aleatórios; escala 1/sqrt(n_in) evita saturar o tanh
            W = self.rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out))
            b = np.zeros(n_out)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        # Guarda a saída de cada camada; a volta precisa delas
        self.activations = [X]
        a = X

        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b

            if layer == len(self.weights) - 1:
                # Última camada: sigmoide (binário) ou softmax (multiclasse)
                a = sigmoid(z) if self.output == "sigmoid" else softmax(z)
            else:
                a = tanh(z)

            self.activations.append(a)

        return a

    def loss(self, y_pred, y):
        # Evita log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)

        if self.output == "sigmoid":
            # Entropia cruzada binária
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # Entropia cruzada categórica (y em one-hot)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))

    def backward(self, y):
        n = y.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Sigmoide + BCE e softmax + CCE dão o mesmo gradiente na saída: y_pred - y
        delta = (self.activations[-1] - y) / n

        # Percorre as camadas de trás para frente
        for layer in reversed(range(len(self.weights))):
            a_prev = self.activations[layer]

            grads_W[layer] = a_prev.T @ delta
            grads_b[layer] = delta.sum(axis=0)

            if layer > 0:
                # Passa a culpa para a camada anterior e pelo tanh dela
                delta = (delta @ self.weights[layer].T) * tanh_derivative(a_prev)

        return grads_W, grads_b

    def update(self, grads_W, grads_b):
        for layer in range(len(self.weights)):
            self.weights[layer] -= self.learning_rate * grads_W[layer]
            self.biases[layer] -= self.learning_rate * grads_b[layer]

    def prepare_targets(self, y):
        # Binário: coluna (n, 1); multiclasse: one-hot (n, n_classes)
        if self.output == "sigmoid":
            return y.reshape(-1, 1).astype(float)

        n_classes = self.layer_sizes[-1]
        return np.eye(n_classes)[y]

    def fit(self, X, y):
        y_target = self.prepare_targets(y)
        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = self.forward(X)
            self.loss_history.append(self.loss(y_pred, y_target))

            grads_W, grads_b = self.backward(y_target)
            self.update(grads_W, grads_b)

        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        proba = self.predict_proba(X)

        if self.output == "sigmoid":
            return (proba[:, 0] >= 0.5).astype(int)

        return np.argmax(proba, axis=1)
