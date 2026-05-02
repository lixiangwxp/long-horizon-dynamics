import torch.nn as nn

from .mlp import MLP


class TCN(nn.Module):
    """TCN 编码器 + MLP 解码器。

    输入 x 的常见形状是 [B, history_len, input_size]。
    TCN 需要 Conv1d 格式 [B, input_size, history_len]，所以 forward 里会先 permute。
    最终只取最后一个历史时刻的 TCN 特征，输出一步预测 delta。
    """

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
        # TemporalConvNet 沿时间维做因果卷积，输出通道数为 encoder_sizes[-1]。
        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        # 原始 TCN 只把最后一个时间步的编码结果交给 MLP，所以 history_len 仍传入 MLP 保持接口兼容。
        self.decoder = MLP(
            encoder_sizes[-1] / history_len,
            history_len,
            decoder_sizes,
            output_size,
            dropout,
        )

    def forward(self, x, args=None):
        # [B, H, F] -> [B, F, H]，适配 nn.Conv1d 的输入布局。
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        # 只取最后一个历史时刻的编码特征，形状 [B, C, 1]。
        return self.decoder(x[:, :, -1:])


class TemporalConvNet(nn.Module):
    """由多个 dilation 不断增大的 TemporalBlock 组成的 TCN 主体。"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for idx in range(num_levels):
            # dilation 逐层翻倍，用较少层数覆盖更长历史。
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
        # 输入输出都保持 [B, C, H] 形式，时间长度由 Chomp1d 裁回原长度。
        return self.network(x)


class TemporalBlock(nn.Module):
    """TCN 的残差卷积块：两层 dilated Conv1d + 残差连接。"""

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
        # padding 会让卷积看到足够历史，后面的 Chomp1d 再裁掉多余的未来端。
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

        # 两个卷积子层串起来构成主分支。
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
        # 如果输入输出通道数不同，用 1x1 卷积把 residual 投影到同一维度。
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
        # 残差连接让深层 TCN 更容易训练。
        return self.relu(out + residual)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


class Chomp1d(nn.Module):
    """裁掉右侧 padding，避免卷积输出比输入时间维更长。"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, : -self.chomp_size].contiguous()
