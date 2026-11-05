
import numpy as np
import matplotlib.pyplot as plt

# AND truth table: two inputs and one target output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=int)

def step(z):
    """Binary step activation function."""
    return (z >= 0).astype(int)

weights = np.zeros(2)
bias = 0.0
learning_rate = 0.1
epochs = 20
errors_per_epoch = []

# Perceptron training
for epoch in range(epochs):
    errors = 0
    for inputs, target in zip(X, y):
        z = np.dot(inputs, weights) + bias
        prediction = int(step(z))
        error = target - prediction
        weights += learning_rate * error * inputs
        bias += learning_rate * error
        errors += int(error != 0)
    errors_per_epoch.append(errors)

print("Learned weights:", weights)
print("Learned bias:", bias)
print("\nAND Gate Predictions")
print("Input 1 | Input 2 | Target | Prediction")
for inputs, target in zip(X, y):
    prediction = int(step(np.dot(inputs, weights) + bias))
    print(f"   {int(inputs[0])}    |    {int(inputs[1])}    |   {target}    |     {prediction}")

plt.plot(range(1, epochs + 1), errors_per_epoch, marker="o")
plt.title("Perceptron Learning: Errors per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of errors")
plt.grid(True)
plt.tight_layout()
plt.show()
