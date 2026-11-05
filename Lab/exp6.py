
import numpy as np
import matplotlib.pyplot as plt

# Minimize f(w) = (w - 3)^2; gradient = 2(w - 3)
def objective(w):
    return (w - 3.0) ** 2

def gradient(w):
    return 2.0 * (w - 3.0)

steps = 60
learning_rate = 0.1
initial_w = -5.0

# Batch Gradient Descent
w_gd = initial_w
gd_history = []
for _ in range(steps):
    w_gd -= learning_rate * gradient(w_gd)
    gd_history.append(objective(w_gd))

# SGD: one noisy gradient estimate per step (simulated)
rng = np.random.default_rng(7)
w_sgd = initial_w
sgd_history = []
for _ in range(steps):
    noisy_grad = gradient(w_sgd) + rng.normal(0, 1.5)
    w_sgd -= learning_rate * noisy_grad
    sgd_history.append(objective(w_sgd))

# Momentum optimizer
w_mom = initial_w
velocity = 0.0
beta = 0.9
momentum_history = []
for _ in range(steps):
    g = gradient(w_mom)
    velocity = beta * velocity + g
    w_mom -= learning_rate * velocity
    momentum_history.append(objective(w_mom))

# Adam optimizer
w_adam = initial_w
m = 0.0
v = 0.0
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
adam_history = []
for t in range(1, steps + 1):
    g = gradient(w_adam)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w_adam -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    adam_history.append(objective(w_adam))

print("Final parameter values (target is w=3):")
print(f"Gradient Descent: {w_gd:.4f}")
print(f"SGD (simulated noisy gradient): {w_sgd:.4f}")
print(f"Momentum: {w_mom:.4f}")
print(f"Adam: {w_adam:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(gd_history, label="Gradient Descent")
plt.plot(sgd_history, label="SGD")
plt.plot(momentum_history, label="Momentum")
plt.plot(adam_history, label="Adam")
plt.xlabel("Iteration")
plt.ylabel("Objective value (loss)")
plt.title("Optimizer Convergence Comparison")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
