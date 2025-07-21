"""
While calculating Loss Function, we provide model prediction and ground truth to the loss function and it returns a float value.
"""
#One hot encoding. Converts an integer y to a tensor of zeros and ones.
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

print(F.one_hot(torch.tensor(0),num_classes=3))
print(F.one_hot(torch.tensor(1),num_classes=3))
print(F.one_hot(torch.tensor(2),num_classes=3))

"""
Loss Function:
1. For Classification:
a) Cross Entropy Loss
b) BCELogitLoss
2. For Regression:
a) MAE (L1Loss)
b) MSE (L2Loss)
"""
score = torch.tensor([-5.2, 4.6, 0.8])
one_hot_target = torch.tensor([1,0,0])

criterion = CrossEntropyLoss()
print(criterion(score.double(),one_hot_target.double()))