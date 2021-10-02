
import typing as tp
import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SincNet import SincConv


class CNNBlock(nn.Module):
    def __init__(self, seq_len: int, conv_type: tp.Union[tp.Type[nn.Conv1d], tp.Type[SincConv]],
                 in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 3, dropout_p: float = 0.0):
        super().__init__()
        conv_block = conv_type(in_channels, out_channels,
                               kernel_size, padding=(kernel_size - 1) // 2)
        pooling = nn.MaxPool1d(pool_size)
        ln = nn.LayerNorm(seq_len // pool_size)
        lrelu = nn.LeakyReLU()
        dropout = nn.Dropout(p=dropout_p)

        self.net = nn.Sequential(conv_block, pooling, ln, lrelu, dropout)

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float = 0.0):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        bn = nn.BatchNorm1d(out_features, momentum=0.05)
        lrelu = nn.LeakyReLU()
        dropout = nn.Dropout(p=dropout_p)

        self.net = nn.Sequential(linear, bn, lrelu, dropout)

    def forward(self, x):
        return self.net(x)
