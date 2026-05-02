import torch
import torch.nn as nn

from .mlp import MLP


class GRU(nn.Module):
    """GRU 编码器 + MLP 解码器。

    输入 x 通常为 [B, history_len, input_size]。
    GRU 沿时间维编码历史窗口，再按 encoder_output 选择 hidden、最后输出或完整序列。
    """

    def __init__(
        self,
        input_size,
        encoder_sizes,
        num_layers,
        history_len,
        decoder_sizes,
        output_size,
        dropout,
        encoder_output,
        **kwargs,
    ):
        super().__init__()
        # batch_first=True 让 GRU 直接接收 [B, H, F]，和 Dataset 输出保持一致。
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=encoder_sizes[0],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # MLP 会把输入展平；这里通过 decoder_input 和 history_len 组合出最终展平宽度。
        if encoder_output == "hidden":
            decoder_input = encoder_sizes[0] / history_len
        elif encoder_output == "output":
            decoder_input = encoder_sizes[0] / history_len
        else:
            decoder_input = encoder_sizes[0]

        self.decoder = MLP(
            decoder_input, history_len, decoder_sizes, output_size, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_output = encoder_output
        self.num_layers = num_layers
        self.hidden_size = encoder_sizes[0]
        self.memory = None

    def forward(self, x, init_memory):
        # init_memory=True 时从零状态开始；否则复用上一轮 forward 保存的 GRU hidden state。
        memory = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
        x, memory = self.encoder(x, memory)
        # GRU 返回更新后的 hidden state，这里会保存下来供下一次 init_memory=False 使用。
        self.memory = memory

        if self.encoder_output == "hidden":
            # 取最后一层 hidden state，形状 [B, hidden_size]。
            x_encoder = memory[-1]
        elif self.encoder_output == "output":
            # 取最后一个时间步的输出，形状 [B, hidden_size]。
            x_encoder = x[:, -1, :]
        else:
            # 保留完整时间序列输出，形状 [B, H, hidden_size]。
            x_encoder = x

        x_encoder = self.dropout(x_encoder)
        return self.decoder(x_encoder)

    def init_memory(self, batch_size, device):
        # GRU 的 memory 只有 hidden state，形状 [num_layers, B, hidden_size]。
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
