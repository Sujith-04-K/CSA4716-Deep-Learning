
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Reproducible nonlinear dataset
rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 40).reshape(-1, 1)
y = 0.5 * X.ravel() ** 2 + X.ravel() + rng.normal(0, 1.2, 40)

# Train/test split by alternating observations for a simple demonstration
X_train, y_train = X[::2], y[::2]
X_test, y_test = X[1::2], y[1::2]

degrees = [1, 2, 15]
x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    curve = model.predict(x_plot)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    ax.scatter(X_train, y_train, label="Training data")
    ax.scatter(X_test, y_test, marker="x", label="Test data")
    ax.plot(x_plot, curve, label=f"Degree {degree} model")
    ax.set_title(f"Degree {degree}\nTrain MSE={train_mse:.2f}, Test MSE={test_mse:.2f}")
    ax.set_xlabel("X")
    ax.grid(True)
    ax.legend(fontsize=8)

axes[0].set_ylabel("y")
fig.suptitle("Underfitting, Suitable Fit and Overfitting")
plt.tight_layout()
plt.show()

print("Interpretation:")
print("- Degree 1 may underfit because the model is too simple.")
print("- Degree 2 can capture the quadratic pattern.")
print("- Degree 15 may overfit by following noise in the training data.")
print("Compare training and test MSE to assess generalization.")
