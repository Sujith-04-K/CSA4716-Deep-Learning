
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "Study_Hours": [1, 2, 3, 4, 5, 6],
    "Exam_Score": [50, 55, 65, 70, 75, 85]
})

print("Dataset:\n", data)
print("\nMean:\n", data.mean())
print("\nVariance:\n", data.var())
print("\nCovariance matrix:\n", data.cov())
print("\nCorrelation matrix:\n", data.corr())

# Eigenvalue and eigenvector analysis of covariance matrix
cov_matrix = data.cov().to_numpy()
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)

# Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(data["Study_Hours"], data["Exam_Score"], s=70)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize covariance matrix
plt.figure(figsize=(5, 4))
plt.imshow(cov_matrix, cmap="coolwarm")
plt.colorbar(label="Covariance")
plt.xticks([0, 1], data.columns)
plt.yticks([0, 1], data.columns)
plt.title("Covariance Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cov_matrix[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.show()
