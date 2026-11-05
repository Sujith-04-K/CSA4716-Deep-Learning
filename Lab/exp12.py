
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and add channel dimension
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

# Use subsets for a quicker practical demonstration
x_train, y_train = x_train[:12000], y_train[:12000]
x_test, y_test = x_test[:3000], y_test[:3000]

def build_cnn(filters, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

experiments = [
    ("Small CNN, LR=0.001", 16, 0.001),
    ("Larger CNN, LR=0.001", 32, 0.001),
    ("Small CNN, LR=0.0003", 16, 0.0003)
]

results = []
for name, filters, lr in experiments:
    print(f"\nTraining {name}")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_cnn(filters, lr)
    history = model.fit(
        x_train, y_train, epochs=3, batch_size=64,
        validation_split=0.1, verbose=1
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    results.append((name, test_accuracy, test_loss))
    print(f"Test accuracy: {test_accuracy * 100:.2f}% | Test loss: {test_loss:.4f}")

print("\nCNN comparison results")
for name, accuracy, loss in results:
    print(f"{name:25s} Accuracy: {accuracy * 100:6.2f}%  Loss: {loss:.4f}")

names = [item[0] for item in results]
accuracies = [item[1] * 100 for item in results]

plt.figure(figsize=(9, 5))
plt.bar(names, accuracies)
plt.title("CNN Architecture and Learning Rate Comparison")
plt.xlabel("Experiment configuration")
plt.ylabel("Test accuracy (%)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()
