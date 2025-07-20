import torch
import torch.nn as nn

input_data = torch.rand(5,6)

model = nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,1),
    nn.Sigmoid()
)

output = model(input_data)
print("Output of model 1 (binary classification):", output, output.shape)

n_classes =3

model2 = nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,n_classes),
    nn.Softmax(dim=-1)
)

output2 = model2(input_data)
print("Output of model 2 (multi-class classification):", output2, output2.shape)

reg_model = nn.Sequential(
    nn.Linear(6,4),
    nn.Linear(4,1)
)

reg_out = reg_model(input_data)
print("Ouptut of reg_model (Regression):",reg_out, reg_out.shape)