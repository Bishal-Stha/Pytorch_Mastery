import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Dropout(p=0.5))

features = torch.randn((1,8))
print(model(features))

## Weight Decay
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)

## Data Augmentation: Generally used in Computer vision to deal with imbalanced dataset. makes copies of images but with different orientations, angles and other different ways.