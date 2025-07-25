# conv_net.py
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Dynamically compute flatten size
        dummy_input = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
            flatten_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Linear(flatten_size, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
