import torch
import torch.nn as nn

class MIMetaClassifierSmall(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.batch_norm0 = nn.BatchNorm1d(256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        return torch.sigmoid(self.output_layer(x))


class MIMetaClassifierMedium(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 512)
        self.batch_norm0 = nn.BatchNorm1d(512)
        self.hidden_layer1 = nn.Linear(512, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.hidden_layer2 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.hidden_layer3 = nn.Linear(512, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        x = torch.relu(self.batch_norm3(self.hidden_layer3(x)))
        return torch.sigmoid(self.output_layer(x))

class MIMetaClassifierLarge(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 1024)
        self.batch_norm0 = nn.BatchNorm1d(1024)
        self.hidden_layer1 = nn.Linear(1024, 1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.hidden_layer2 = nn.Linear(1024, 1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.hidden_layer3 = nn.Linear(1024, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        x = torch.relu(self.batch_norm3(self.hidden_layer3(x)))
        return torch.sigmoid(self.output_layer(x))

class MIMetaClassifierDeep(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.batch_norm0 = nn.BatchNorm1d(256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.hidden_layer3 = nn.Linear(256, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.hidden_layer4 = nn.Linear(256, 256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.hidden_layer5 = nn.Linear(256, 128)
        self.batch_norm5 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        x = torch.relu(self.batch_norm3(self.hidden_layer3(x)))
        x = torch.relu(self.batch_norm4(self.hidden_layer4(x)))
        x = torch.relu(self.batch_norm5(self.hidden_layer5(x)))
        return torch.sigmoid(self.output_layer(x))

class MIMetaClassifierDeepLarge(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 1024)
        self.batch_norm0 = nn.BatchNorm1d(1024)
        self.hidden_layer1 = nn.Linear(1024, 1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.hidden_layer2 = nn.Linear(1024, 1024)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.hidden_layer3 = nn.Linear(1024, 1024)
        self.batch_norm3 = nn.BatchNorm1d(1024)
        self.hidden_layer4 = nn.Linear(1024, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024)
        self.hidden_layer5 = nn.Linear(1024, 256)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        x = torch.relu(self.batch_norm3(self.hidden_layer3(x)))
        x = torch.relu(self.batch_norm4(self.hidden_layer4(x)))
        x = torch.relu(self.batch_norm5(self.hidden_layer5(x)))
        return torch.sigmoid(self.output_layer(x))