import typing as tp
import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def init_params_mel(self, out_channels):
        def to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        return to_hz(np.linspace(to_mel(self.min_hz),
                                 to_mel(self.sample_rate / 2 -
                                        self.min_hz - self.min_band),
                                 out_channels + 1))

    def __init__(self, in_channels, out_channels, kernel_size, padding, sample_rate=16000, min_hz=50, min_band=50):
        super().__init__()
        self.sample_rate, self.min_hz, self.min_band = sample_rate, min_hz, min_band
        self.padding = padding
        hz = self.init_params_mel(out_channels)
        self.hz_left = nn.Parameter(
            torch.unsqueeze(torch.Tensor(hz[:-1]), dim=1))
        self.hz_band = nn.Parameter(
            torch.unsqueeze(torch.Tensor(np.diff(hz)), dim=1))
        self.window = torch.hann_window(kernel_size)[:kernel_size // 2]
        self.n = 2 * np.pi * \
            torch.unsqueeze(torch.arange(-(kernel_size // 2), 0),
                            dim=0) / sample_rate

    def forward(self, wav):
        self.window, self.n = self.window.to(wav.device), self.n.to(wav.device)

        low = self.min_hz + torch.abs(self.hz_left)
        high = low + self.min_band + self.hz_band
        band = (high - low)[:, 0]

        f_low = torch.matmul(low, self.n)
        f_high = torch.matmul(high, self.n)

        band_pass_left = 2 * (torch.sin(f_high) -
                              torch.sin(f_low)) / self.n * self.window
        band_pass_center = 2 * torch.unsqueeze(band, dim=1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat(
            (band_pass_left, band_pass_center, band_pass_right), dim=1) / band_pass_center

        filters = torch.unsqueeze(band_pass, dim=1)
        return F.conv1d(wav, filters, padding=self.padding)


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


class SincNet(nn.Module):
    def __init__(self, chunk_len, n_classes, conv_type: str):
        super().__init__()
        if conv_type not in ['cnn', 'sinc']:
            raise ValueError(
                "Only two types of first convolution are supported, cnn and sinc.")
        if conv_type == 'cnn':
            first_block = nn.Conv1d
        else:
            first_block = SincConv
        ln1 = nn.LayerNorm(chunk_len)
        cnn_blocks = nn.Sequential(CNNBlock(chunk_len, first_block, 1, 80, 251),
                                   CNNBlock(chunk_len // 3,
                                            nn.Conv1d, 80, 60, 5),
                                   CNNBlock(chunk_len // 9, nn.Conv1d, 60, 60, 5))
        flatten = nn.Flatten(start_dim=1)
        ln2 = nn.LayerNorm(chunk_len // 27 * 60)
        mlp_blocks = nn.Sequential(MLPBlock(chunk_len // 27 * 60, 2048),
                                   MLPBlock(2048, 2048),
                                   MLPBlock(2048, 2048))
        self.backbone = nn.Sequential(
            ln1, cnn_blocks, flatten, ln2, mlp_blocks)
        self.classification_head = nn.Linear(2048, n_classes)

    def forward(self, wavs):
        return self.classification_head(self.backbone(wavs))

    def compute_d_vectors(self, wavs):
        return self.backbone(wavs)


class myResnet(nn.Module):
    def __init__(self, pretrained=True, channel_output=462):
        super(myResnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, channel_output, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class MS_SincResNet(nn.Module):
    def __init__(self, sample_rate=16000, chunk_len=3200, n_classes=462):
        super(MS_SincResNet, self).__init__()
        self.layerNorm = nn.LayerNorm([1, chunk_len])
        kernel1 = 251
        out_channels = 80
        self.sincNet1 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.sincNet2 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.sincNet3 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.resnet = myResnet(pretrained=True)

    def forward(self, x):
        x = self.layerNorm(x)

        feat1 = self.sincNet1(x)
        feat2 = self.sincNet2(x)
        feat3 = self.sincNet3(x)

        x = torch.cat((feat1.unsqueeze_(dim=1),
                       feat2.unsqueeze_(dim=1),
                       feat3.unsqueeze_(dim=1)), dim=1)
        x = self.resnet(x)
        return x
