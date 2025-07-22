# Fully Connected layer and Linear Layer is essentially same thing in context of Neural Netoworks.

import torch.nn as nn
import torch
# Input Neurons: features
# Output Neurons: classes

input_tensor = torch.tensor([
    [0.4524, 0.2482, -0.5329]
    ])

linear_layer = nn.Linear(
    in_features=3,
    out_features=2
)

output = linear_layer(input_tensor)
print(output)
print("Weight: ",linear_layer.weight)
print("Bias: ",linear_layer.bias)
