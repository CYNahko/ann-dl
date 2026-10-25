import numpy as np

#Valores do enunciado
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, -.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3   

# Item a
def forward(W1, b1, W2, b2):
    z1 = W1 @ x + b1         # mistura de cada ajudante
    h1 = np.tanh(z1)         # limitador dos ajudantes
    u2 = W2 @ h1 + b2        # mistura do chef
    y_hat = np.tanh(u2)      # limitador do chef = previsão
    L = (y - y_hat) ** 2     # reclamação
    return z1, h1, u2, y_hat, L


np.set_printoptions(precision=4, suppress=True)

z1, h1, u2, y_hat, L = forward(W1, b1, W2, b2)
print("A. Passagem direta")
print("z1    =", z1)
print("h1    =", h1)
print(f"u2    = {u2:.4f}")
print(f"y_hat = {y_hat:.4f}")
print(f"L     = {L:.4f}")

# Item b
dL_dyhat = -2 * (y - y_hat)              # culpa no prato
dL_du2 = dL_dyhat * (1 - y_hat ** 2)     # passa pelo limitador do chef
dW2 = dL_du2 * h1                        # culpa dos pesos do chef
db2 = dL_du2                             # culpa da pitada do chef
dL_dh1 = W2 * dL_du2                     # chef passa a culpa pros ajudantes
dL_dz1 = dL_dh1 * (1 - h1 ** 2)          # passa pelo limitador dos ajudantes
dW1 = np.outer(dL_dz1, x)                # culpa dos pesos dos ajudantes
db1 = dL_dz1                             # culpa das pitadas dos ajudantes

print("\nB. Passagem reversa")
print(f"dL/dy_hat = {dL_dyhat:.4f}")
print(f"dL/du2    = {dL_du2:.4f}")
print("dL/dW2    =", dW2)
print(f"dL/db2    = {db2:.4f}")
print("dL/dh1    =", dL_dh1)
print("dL/dz1    =", dL_dz1)
print("dL/dW1    =\n", dW1)
print("dL/db1    =", db1)

#Item c
W1_new = W1 - eta * dW1
b1_new = b1 - eta * db1
W2_new = W2 - eta * dW2
b2_new = b2 - eta * db2

print("\nC. Parâmetros atualizados")
print("W1 =\n", W1_new)
print("b1 =", b1_new)
print("W2 =", W2_new)
print(f"b2 = {b2_new:.4f}")

_, _, _, y_hat_new, L_new = forward(W1_new, b1_new, W2_new, b2_new)
print(f"\nSegunda passagem: y_hat = {y_hat_new:.4f}, L = {L_new:.4f} (antes: {L:.4f})")
