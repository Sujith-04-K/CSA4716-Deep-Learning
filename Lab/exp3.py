
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# Normal distribution
x = np.linspace(-4, 4, 500)
normal_pdf = norm.pdf(x, loc=0, scale=1)

# Binomial distribution: 10 trials, probability of success = 0.5
k = np.arange(0, 11)
binomial_pmf = binom.pmf(k, n=10, p=0.5)

print("Normal distribution: mean =", norm.mean(), ", standard deviation =", norm.std())
print("Binomial distribution: n=10, p=0.5")
print("Probabilities for 0 to 10 successes:\n", np.round(binomial_pmf, 4))

# Bayes' theorem example: disease testing
p_disease = 0.01                 # Prior probability
p_positive_given_disease = 0.95 # Sensitivity
p_positive_given_no_disease = 0.05

p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_no_disease * (1 - p_disease))
p_disease_given_positive = (
    p_positive_given_disease * p_disease / p_positive
)

print("\nBayes' theorem example:")
print(f"P(Disease | Positive) = {p_disease_given_positive:.4f}")
print(f"Posterior probability = {p_disease_given_positive * 100:.2f}%")

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(x, normal_pdf)
axes[0].set_title("Standard Normal Distribution")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Probability Density")
axes[0].grid(True)

axes[1].bar(k, binomial_pmf)
axes[1].set_title("Binomial Distribution (n=10, p=0.5)")
axes[1].set_xlabel("Number of successes")
axes[1].set_ylabel("Probability")
axes[1].grid(axis="y")
plt.tight_layout()
plt.show()
