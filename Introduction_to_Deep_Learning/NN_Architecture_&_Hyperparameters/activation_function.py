"""
Activation Function adds Non-Linearity to the network.
Sigmoid -> Binary Classification
Softmax -> Multi-class Classification
"""

import torch
import torch.nn as nn

input_tensor = torch.tensor([[6]])
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)
print(output)

model = nn.Sequential(
    nn.Linear(6,4), # First Linear Layer
    nn.Linear(4,1), # Second Linear Layer
    nn.Sigmoid() # Sigmoid activation function
)

inp_tensor = torch.tensor([[4.3, 6.1,2.3]])

probabilities = nn.Softmax(dim=-1) # softmax is applied to the input tensor's last dimension.
output_tensor = probabilities(inp_tensor)
print(output_tensor)