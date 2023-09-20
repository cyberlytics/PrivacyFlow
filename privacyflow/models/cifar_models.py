import torch
import torch.nn as nn


class CifarCNNModel(nn.Module):
    def __init__(self, output_size: int = 10, use_log_softmax:bool=True):
        super().__init__()
        self.conv1 = self._create_conv_block(in_channels=3, out_channels=64, pooling_layer=True)
        self.res1 = self._create_res_block(num_channels=64)
        self.conv2 = self._create_conv_block(in_channels=64, out_channels=128, pooling_layer=True)
        self.res2 = self._create_res_block(num_channels=128)
        self.conv3 = self._create_conv_block(in_channels=128, out_channels=256, pooling_layer=True)
        self.res3 = self._create_res_block(num_channels=256)
        self.conv4 = self._create_conv_block(in_channels=256, out_channels=512, pooling_layer=False)
        self.output_layer = nn.Linear(in_features=512, out_features=output_size)
        self.use_log_softmax = use_log_softmax

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.conv3(x)
        x = self.res3(x) + x
        x = self.conv4(x)
        x = nn.AvgPool2d(kernel_size=4, padding=0)(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.output_layer(x)
        if self.use_log_softmax:
            return torch.log_softmax(x, dim=-1)
        return x
    def _create_conv_block(self,
                           in_channels: int,
                           out_channels: int,
                           batch_norm_layer: bool = True,
                           pooling_layer: bool = False):
        block_layers = []
        block_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        if batch_norm_layer:
            block_layers.append(nn.BatchNorm2d(out_channels))
        block_layers.append(nn.ReLU())
        if pooling_layer:
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*block_layers)

    def _create_res_block(self, num_channels: int):
        return nn.Sequential(
            self._create_conv_block(in_channels=num_channels, out_channels=num_channels),
            self._create_conv_block(in_channels=num_channels, out_channels=num_channels)
        )
