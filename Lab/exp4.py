
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# True and predicted probability distributions
p = np.array([0.7, 0.2, 0.1])  # True distribution
q = np.array([0.6, 0.3, 0.1])  # Predicted distribution

# Entropy: uncertainty in the true distribution
H_p = entropy(p, base=2)

# Cross-entropy: average coding cost using q for data from p
cross_entropy = -np.sum(p * np.log2(q))

# KL divergence: difference between cross-entropy and entropy
kl_divergence = entropy(p, q, base=2)

print("True distribution (p):", p)
print("Predicted distribution (q):", q)
print(f"Entropy H(p): {H_p:.4f} bits")
print(f"Cross-Entropy H(p, q): {cross_entropy:.4f} bits")
print(f"KL Divergence D_KL(p || q): {kl_divergence:.4f} bits")
print(f"Check: Cross-Entropy - Entropy = {cross_entropy - H_p:.4f} bits")

# Plot distributions
labels = ["Class 1", "Class 2", "Class 3"]
positions = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(positions - width / 2, p, width, label="True distribution")
plt.bar(positions + width / 2, q, width, label="Predicted distribution")
plt.xticks(positions, labels)
plt.ylabel("Probability")
plt.title("True vs Predicted Probability Distributions")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
