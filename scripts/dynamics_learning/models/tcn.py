import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .mlp import MLP

  

# class TCN(nn.Module):
#     def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size, kernel_size, dropout, **kwargs):
#         super(TCN, self).__init__()
#         self.encoder = TemporalConvNet(input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout)
#         self.decoder = MLP(encoder_sizes[-1], history_len, decoder_sizes, output_size, dropout)
    

#     def forward(self, x,  args=None):
#         x = x.permute(0, 2, 1)  # Transpose input to (batch_size, num_features, history_length)
#         y = self.encoder(x)
#         y = self.decoder(y) # Take the output of the last time step
#         return y
    
class TCN(nn.Module):
    """TCN 模型入口。

    输入 x 的常见 shape 是 [batch_size, history_length, input_size]。
    输出 shape 是 [batch_size, output_size]，表示根据一段历史预测下一步目标。
    """
    def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size, kernel_size, dropout, **kwargs):
        super(TCN, self).__init__()
        
        # encoder 用一串时间卷积块提取历史序列里的时间特征。
        self.encoder = TemporalConvNet(input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout)
        # decoder 接收最后一个时间点的 encoder 特征，再映射成最终预测值。
        # 这里传入 encoder_sizes[-1] / history_len，是为了配合 MLP 内部 flatten 后得到 encoder_sizes[-1] 维输入。
        self.decoder = MLP(encoder_sizes[-1] / history_len, history_len, decoder_sizes, output_size, dropout)
        self.linear = nn.Linear(encoder_sizes[-1], output_size)
    

    def forward(self, x,  args=None):
        # 原始输入是 [batch, history, features]，Conv1d 需要 [batch, features, history]。
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        # 只取最后一个历史时刻的编码结果，用它代表整段 history 的总结信息。
        x = self.decoder(x[:, :, -1:])
        # x = self.linear(x[:, :, -1])
        return x
    
class TemporalConvNet(nn.Module):
    """按层堆叠 TemporalBlock，通道数由 num_channels 逐层指定。"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # dilation 按 1, 2, 4, ... 增大，让后面的卷积层能看到更长的历史范围。
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # padding 先补齐时间长度，后面再用 Chomp1d 把多出来的右侧部分裁掉。
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 输入/输出都保持 [batch, channels, history]，便于多个时间卷积块串联。
        return self.network(x)
    

#----------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    """一个 TCN 残差块：两层 dilated Conv1d + 激活 + dropout + residual。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0):
        super(TemporalBlock, self).__init__()
        # 第一层扩张卷积：在时间维上看一段历史窗口。
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        # 第二层扩张卷积：继续提取时间特征。
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入通道数和输出通道数不同，先用 1x1 卷积把 residual 的维度对齐。
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.LeakyReLU(0.2)

        self.init_weights()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # residual connection：让卷积块学习对输入的修正，训练更稳定。
        return self.relu(out + res)
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

class Chomp1d(nn.Module):
    """裁掉 Conv1d padding 多出来的右侧时间步，保持输出时间长度不变。"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Chomp 只裁掉时间维右侧 padding；前面的历史信息保持因果卷积语义。
        return x[:, :, :-self.chomp_size].contiguous()
