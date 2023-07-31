import torch
import torch.nn as nn


class HeartModelBase(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.hidden_layer2 = nn.Linear(256, 100)
        self.output_layer = nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        return self.output_layer(x)


class HeartModelSmall(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)


class HeartModelLarge(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden_layer3 = nn.Linear(256, 100)
        self.output_layer = nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.relu(self.hidden_layer3(x))
        return self.output_layer(x)

class HeartMIModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        return self.output_layer(x)