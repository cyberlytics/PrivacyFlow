import torch
import torch.nn as nn
import torchvision.models as tv_models


def get_FaceModelBase(output_size: int):
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, output_size),
        nn.Sigmoid()
    )
    return model


class FaceMIModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.batch_norm0 = nn.BatchNorm1d(256)
        self.hidden_layer1 = nn.Linear(256, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        return torch.sigmoid(self.output_layer(x))


class FaceMIModelLarge(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 512)
        self.batch_norm0 = nn.BatchNorm1d(512)
        self.hidden_layer1 = nn.Linear(512, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.hidden_layer2 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.hidden_layer3 = nn.Linear(512, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.batch_norm0(self.input_layer(x)))
        x = torch.relu(self.batch_norm1(self.hidden_layer1(x)))
        x = torch.relu(self.batch_norm2(self.hidden_layer2(x)))
        x = torch.relu(self.batch_norm3(self.hidden_layer3(x)))
        return torch.sigmoid(self.output_layer(x))
