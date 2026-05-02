import torch
import torch.nn as nn

from .mlp import MLP
from .tcn import TemporalConvNet


class TCNLSTM(nn.Module):
    """TCN encoder + 初始化式 LSTM decoder + Bahdanau attention。

    输入 x 为 [B, H, input_size]。TCN 先把整段历史编码成 enc_seq [B, H, C]；
    enc_seq 的最后一步特征和时间均值用于初始化 LSTM decoder 的 h0/c0；
    attention 再从完整 enc_seq 中取 context，最后输出一步预测 delta [B, output_size]。
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
        num_layers=2,
        decoder_hidden_size=None,
        attention_size=None,
        use_recurrent_memory=True,
        **kwargs,
    ):
        super().__init__()
        if not encoder_sizes:
            raise ValueError("encoder_sizes must contain at least one channel size.")

        self.input_size = input_size
        self.history_len = history_len
        self.num_layers = num_layers
        self.tcn_channels = encoder_sizes[-1]
        self.decoder_hidden_size = (
            self.tcn_channels if decoder_hidden_size is None else decoder_hidden_size
        )
        self.attention_size = (
            self.decoder_hidden_size if attention_size is None else attention_size
        )
        self.use_recurrent_memory = use_recurrent_memory

        # TCN 期望输入布局为 [B, F, H]，forward 中会从 [B, H, F] permute 过去。
        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )

        # initializer 不展平全历史，只使用最后时刻编码和历史平均编码，控制参数量。
        initializer_input_size = 2 * self.tcn_channels
        initializer_hidden_size = max(self.tcn_channels, self.decoder_hidden_size)
        initializer_output_size = 2 * self.num_layers * self.decoder_hidden_size
        self.state_initializer = nn.Sequential(
            nn.Linear(initializer_input_size, initializer_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(initializer_hidden_size, initializer_output_size),
        )

        # Bahdanau-style additive attention: key 来自 TCN 序列，query 来自 LSTM 初始 hidden。
        self.attn_key = nn.Linear(self.tcn_channels, self.attention_size, bias=False)
        self.attn_query = nn.Linear(
            self.decoder_hidden_size, self.attention_size, bias=False
        )
        self.attn_score = nn.Linear(self.attention_size, 1, bias=False)

        # 把最后一帧原始输入投影到 decoder hidden 维度，再和 attention context 拼接。
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 单步 LSTM decoder：每次 forward 只消费一个 decoder_input，不在模型里做 rollout。
        self.decoder_rnn = nn.LSTM(
            input_size=self.decoder_hidden_size + self.tcn_channels,
            hidden_size=self.decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 输出 MLP 把 decoder hidden 映射成下一步 full-state residual delta。
        self.decoder = MLP(
            self.decoder_hidden_size, 1, decoder_sizes, output_size, dropout
        )
        self.memory = None

    def reset_memory(self):
        # 外部调试或切换序列时可以手动清空 decoder memory。
        self.memory = None

    def _init_decoder_state(self, enc_seq):
        # enc_seq: [B, H, C]。最后时刻代表当前状态，mean 汇总整段历史。
        init_feature = torch.cat([enc_seq[:, -1, :], enc_seq.mean(dim=1)], dim=-1)
        state = torch.tanh(self.state_initializer(init_feature))
        h0, c0 = torch.chunk(state, 2, dim=-1)

        batch_size = enc_seq.shape[0]
        # PyTorch LSTM 需要 [num_layers, B, hidden_size]。
        h0 = h0.reshape(
            batch_size, self.num_layers, self.decoder_hidden_size
        ).permute(1, 0, 2)
        c0 = c0.reshape(
            batch_size, self.num_layers, self.decoder_hidden_size
        ).permute(1, 0, 2)
        return h0.contiguous(), c0.contiguous()

    def _memory_is_usable(self, batch_size, device, dtype):
        # batch/device/dtype 不匹配时不能复用旧 memory，否则 LSTM 会报 shape 或 device 错。
        if self.memory is None:
            return False

        h, c = self.memory
        return (
            h.shape == (self.num_layers, batch_size, self.decoder_hidden_size)
            and c.shape == (self.num_layers, batch_size, self.decoder_hidden_size)
            and h.device == device
            and c.device == device
            and h.dtype == dtype
            and c.dtype == dtype
        )

    def _attend(self, enc_seq, query):
        # scores: [B, H]，每个历史时刻一个注意力分数。
        scores = self.attn_score(
            torch.tanh(self.attn_key(enc_seq) + self.attn_query(query).unsqueeze(1))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        # context: [B, C]，是 TCN 编码序列按注意力权重的加权和。
        context = torch.bmm(weights.unsqueeze(1), enc_seq).squeeze(1)
        return context, weights

    def forward(self, x, init_memory=True, return_attention=False):
        if x.ndim != 3:
            raise ValueError(
                f"TCNLSTM expected input shape [B, H, F], got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"TCNLSTM expected input_size={self.input_size}, "
                f"got {x.shape[-1]}."
            )

        batch_size = x.shape[0]
        # [B, H, F] -> [B, F, H] -> TCN -> [B, C, H] -> [B, H, C]。
        enc_input = x.permute(0, 2, 1).contiguous()
        enc_seq = self.encoder(enc_input).permute(0, 2, 1).contiguous()

        # init_memory=True 时从当前 enc_seq 初始化；否则尽量复用上一轮 decoder memory。
        if (
            init_memory
            or not self.use_recurrent_memory
            or not self._memory_is_usable(batch_size, x.device, x.dtype)
        ):
            state = self._init_decoder_state(enc_seq)
        else:
            state = self.memory

        # 用 decoder 初始 hidden 的最后一层作为 attention query。
        query = state[0][-1]
        context, attn_weights = self._attend(enc_seq, query)

        # decoder 输入只使用最后一帧原始输入和当前历史的 attention context。
        x_last = x[:, -1, :]
        projected_x_last = self.input_proj(x_last)
        decoder_input = torch.cat([projected_x_last, context], dim=-1).unsqueeze(1)

        # 保存 LSTM 返回的新 hidden/cell state，供 init_memory=False 的下一次调用使用。
        decoder_out, self.memory = self.decoder_rnn(decoder_input, state)
        y = self.decoder(decoder_out[:, -1, :])

        if return_attention:
            return y, attn_weights
        return y
