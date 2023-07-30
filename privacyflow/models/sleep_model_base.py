import torch
import torch.nn as nn


class SleepModelBase(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(SleepModelBase, self).__init__()
        self.input_layer = nn.Linear(input_size, 50)
        self.hidden_layer1 = nn.Linear(50, 50)
        self.hidden_layer2 = nn.Linear(50, 50)
        self.hidden_layer3 = nn.Linear(50,50)
        self.output_layer = nn.Linear(50, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.relu(self.hidden_layer3(x))
        return self.output_layer(x)