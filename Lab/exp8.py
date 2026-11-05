
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# Network: 2 input neurons -> 4 hidden neurons -> 1 output neuron
W1 = np.random.randn(2, 4) * 0.8
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.8
b2 = np.zeros((1, 1))

learning_rate = 1.0
epochs = 15000
loss_history = []

for epoch in range(epochs):
    # Forward propagation
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)

    # Binary cross-entropy loss
    eps = 1e-8
    loss = -np.mean(y * np.log(output + eps) +
                    (1 - y) * np.log(1 - output + eps))
    loss_history.append(loss)

    # Backpropagation
    m = X.shape[0]
    dz2 = (output - y) / m
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Final predictions
final_output = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
print("XOR predictions")
print("Input       Target   Predicted probability   Class")
for inputs, target, probability in zip(X, y, final_output):
    print(f"{inputs.astype(int)}       {int(target[0])}          "
          f"{probability[0]:.4f}              {int(probability[0] >= 0.5)}")

plt.plot(loss_history)
plt.title("Training Loss: Two-Layer Neural Network")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.grid(True)
plt.tight_layout()
plt.show()
