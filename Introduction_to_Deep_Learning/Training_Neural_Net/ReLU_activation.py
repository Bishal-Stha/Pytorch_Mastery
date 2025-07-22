import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create input values
arr = torch.arange(-9, 10, 1, dtype=torch.float)

# Define ReLU function
def relu(X: torch.Tensor) -> torch.Tensor:
    return torch.maximum(X, torch.tensor(0.0))

# Apply ReLU
output = relu(arr)

# Plot
plt.subplot(1,2,1)
plt.plot(arr, output)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# Leaky ReLU
leaky_relu = nn.LeakyReLU(
    negative_slope =0.05
)
output2 = leaky_relu(arr)
# Plot
plt.subplot(1,2,2)
plt.plot(arr, output2)
plt.title("LeakyReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.tight_layout()
plt.show()