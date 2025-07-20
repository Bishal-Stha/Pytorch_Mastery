import torch
import torch.nn as nn

# stacking layers using nn.Sequential
n_features, n_classes = 16, 2
model = nn.Sequential(
    nn.Linear(n_features,8),
    nn.Linear(8,4),
    nn.Linear(4,n_classes)
)

# More hidden layers = more parameters = higher model capacity
modelX = nn.Sequential(
    nn.Linear(8,4),
    nn.Linear(4,2)
)
"""
First layer has 4 neurons, each neuron has 8+1 parameters.
9*4 = 36
Second layer has 2 neurons, each neurons has 4+1 parameters. 5*2 = 10
In total 46 learnable parameters.
"""

total = 0
for param in modelX.parameters():
    total += param.numel()
print("Learnable parameters in modelX: ",total)