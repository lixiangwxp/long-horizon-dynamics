import torch.nn as nn


class MLP(nn.Module):
    """通用展平 MLP。

    输入可以是 [B, F] 或 [B, H, F]；forward 会先把除 batch 外的维度展平，
    再经过若干 Linear + GELU + Dropout，最后输出 [B, output_size]。
    """

    def __init__(
        self, input_size, history_len, decoder_sizes, output_size, dropout, **kwargs
    ):
        super().__init__()
        # 这里的 input_size 是单个时间步的特征宽度，history_len 用来估计展平后的总输入维度。
        self.model = self.make(
            int(input_size * history_len), decoder_sizes, output_size, dropout
        )

    def make(self, input_size, decoder_sizes, output_size, dropout):
        # 第一层把展平后的历史窗口映射到 decoder_sizes[0]。
        layers = [
            nn.Linear(input_size, decoder_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        # 中间层按 decoder_sizes 顺序堆叠。
        for idx in range(len(decoder_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(decoder_sizes[idx], decoder_sizes[idx + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(decoder_sizes[-1], output_size))
        return nn.Sequential(*layers)

    def forward(self, x, args=None):
        # 保留 batch 维，把时间维和特征维合并成一个向量。
        x = x.reshape(x.shape[0], -1)
        return self.model(x)
