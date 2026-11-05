
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Generate a noisy binary classification dataset
n = 1600
X = np.random.uniform(-2.5, 2.5, (n, 2)).astype("float32")
boundary = np.sin(2.0 * X[:, 0]) + 0.4 * X[:, 0]
noise = np.random.normal(0, 0.45, n)
y = ((X[:, 1] > boundary + noise).astype("float32"))

indices = np.random.permutation(n)
X, y = X[indices], y[indices]
train_end, val_end = 1000, 1300
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

def build_model(kind):
    layers = [tf.keras.layers.Input(shape=(2,))]
    if kind == "l2":
        regularizer = tf.keras.regularizers.l2(0.01)
        layers += [
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)
        ]
    elif kind == "dropout":
        layers += [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3)
        ]
    else:  # baseline and early-stopping models
        layers += [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu")
        ]
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

results = {}
histories = {}
for kind in ["baseline", "l2", "dropout", "early_stopping"]:
    print(f"\nTraining: {kind}")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_model(kind)
    callbacks = []
    if kind == "early_stopping":
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, batch_size=32, verbose=0, callbacks=callbacks
    )
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    results[kind] = accuracy
    histories[kind] = history
    print(f"Test accuracy: {accuracy * 100:.2f}% | Test loss: {loss:.4f}")

print("\nTest accuracy comparison")
for name, accuracy in results.items():
    print(f"{name:15s}: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 5))
for name, history in histories.items():
    plt.plot(history.history["val_loss"], label=name)
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(list(results.keys()), [value * 100 for value in results.values()])
plt.title("Test Accuracy: Regularization Techniques")
plt.xlabel("Model")
plt.ylabel("Test accuracy (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
