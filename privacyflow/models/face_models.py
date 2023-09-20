import torch
import torch.nn as nn
import torchvision.models as tv_models

from privacyflow.configs import path_configs

def get_FaceModelDenseNet(output_size: int):
    model = tv_models.densenet121(weights=tv_models.DenseNet121_Weights, memory_efficient=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, output_size),
        nn.Sigmoid()
    )
    return model


def get_FaceModelResNet(output_size: int, pretrained:bool=True) -> nn.Module:
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, output_size),
        nn.Sigmoid()
    )
    return model


def get_FaceVisionTransformer(output_size: int, pretrained:bool=True) -> nn.Module:
    weights = tv_models.ViT_B_16_Weights if pretrained else None
    model = tv_models.vit_b_16(weights=weights)
    model.heads = nn.Sequential(
        nn.Linear(model.hidden_dim, output_size),
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

def load_saved_model(model_type:str = "cnn", alt_path:str= None):
    if model_type == "cnn":
        model = get_FaceModelResNet(40)
        path = path_configs.FACE_CNN_MODEL if not alt_path else alt_path

    elif model_type == "transformer":
        model = get_FaceVisionTransformer(40)
        path = path_configs.FACE_VIT_MODEL if not alt_path else alt_path

    elif model_type == "dense":
        model = get_FaceModelDenseNet(40)
        path = path_configs.FACE_DENSE_MODEL if not alt_path else alt_path

    else:
        return None

    model.load_state_dict(torch.load(path))
    return model