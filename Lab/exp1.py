
import numpy as np
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([2, 3])
v2 = np.array([4, 1])

print("Vector 1:", v1)
print("Vector 2:", v2)
print("Vector addition:", v1 + v2)
print("Vector subtraction:", v1 - v2)
print("Dot product:", np.dot(v1, v2))

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\nMatrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix addition:\n", A + B)
print("Matrix multiplication:\n", np.dot(A, B))
print("Transpose of A:\n", A.T)
print("Determinant of A:", np.linalg.det(A))
print("Inverse of A:\n", np.linalg.inv(A))

# Plot the two vectors
plt.figure(figsize=(6, 5))
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
           label="Vector 1", color="blue")
plt.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
           label="Vector 2", color="red")
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vector Visualization")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
