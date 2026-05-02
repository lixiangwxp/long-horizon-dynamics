import torch.nn as nn

from .mlp import MLP


class TCN(nn.Module):
    def __init__(
        self,
        input_size,
        encoder_sizes,
        history_len,
        decoder_sizes,
        output_size,
        kernel_size,
        dropout,
        **kwargs,
    ):
        super().__init__()
        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        self.decoder = MLP(
            encoder_sizes[-1] / history_len,
            history_len,
            decoder_sizes,
            output_size,
            dropout,
        )

    def forward(self, x, args=None):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return self.decoder(x[:, :, -1:])


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for idx in range(num_levels):
            dilation_size = 2**idx
            in_channels = num_inputs if idx == 0 else num_channels[idx - 1]
            out_channels = num_channels[idx]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.LeakyReLU(0.2)
        self.init_weights()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, : -self.chomp_size].contiguous()
