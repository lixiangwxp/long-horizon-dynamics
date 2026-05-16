import torch
import torch.nn as nn

from .mlp import MLP
from .tcn import TemporalConvNet


class TCNLSTM(nn.Module):
    """TCN anchor + recurrent latent context + local attitude correction。

    输入 x 为 [B, H, input_size]。TCN 先把整段历史编码成 enc_seq [B, H, C]；
    raw-history LSTM 从原始 [x,u] 中提取 latent context；TCN anchor 预测主
    residual，LSTM/attention decoder 只预测由 latent context 调制的小修正。
    velocity-only residual 是 checkpoint-safe 的小尺度速度阻尼分支，只写
    y[:, 3:6]，用于吸收 actuator lag / aero drag 的慢变量误差。
    GRU context bridge、output-space residual 和 low-rank coupled residual 来自
    attitude checkpoint 之后的失败候选，当前只保留 key 兼容；forward 回到
    H10 attitude anchor 的 context + attitude correction 路径做稳态延续。
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
        self.anchor_history_len = min(history_len, 10)
        self.adaptive_history_context = bool(
            kwargs.get("adaptive_history_context", False)
        )
        self.adaptive_history_short_window = int(
            kwargs.get("adaptive_history_short_window", 10)
        )
        self.adaptive_history_mid_window = int(
            kwargs.get("adaptive_history_mid_window", 25)
        )
        self.side_history_scale_init = float(
            kwargs.get("tcnlstm_side_history_scale_init", 0.05)
        )
        self.side_history_selector_prior = kwargs.get(
            "tcnlstm_side_history_selector_prior", "uniform"
        )
        self.history_context_dim = int(kwargs.get("history_context_dim", 0) or 0)
        self.adaptive_history_stats = {}

        # TCN 期望输入布局为 [B, F, H]，forward 中会从 [B, H, F] permute 过去。
        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        temporal_refiner_heads = 4 if self.tcn_channels % 4 == 0 else 1
        self.temporal_refiner_attn_norm = nn.LayerNorm(self.tcn_channels)
        self.temporal_refiner_attn = nn.MultiheadAttention(
            self.tcn_channels,
            num_heads=temporal_refiner_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_refiner_ffn_norm = nn.LayerNorm(self.tcn_channels)
        self.temporal_refiner_ffn = nn.Sequential(
            nn.Linear(self.tcn_channels, 2 * self.tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.tcn_channels, self.tcn_channels),
        )
        self.temporal_refiner_out = nn.Linear(self.tcn_channels, self.tcn_channels)
        nn.init.zeros_(self.temporal_refiner_out.weight)
        nn.init.zeros_(self.temporal_refiner_out.bias)
        self.temporal_refiner_scale = nn.Parameter(torch.tensor(0.1))
        self.history_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.history_norm = nn.LayerNorm(self.decoder_hidden_size)
        self.gru_context = nn.GRU(
            input_size=self.tcn_channels,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru_context_norm = nn.LayerNorm(self.decoder_hidden_size)
        self.lag_observer = TemporalConvNet(
            2 * input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        self.lag_observer_norm = nn.LayerNorm(self.tcn_channels)
        self.long_history_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.long_history_norm = nn.LayerNorm(self.decoder_hidden_size)

        # initializer 使用 TCN 局部特征和 raw-history latent context 初始化 decoder。
        initializer_input_size = 2 * self.tcn_channels + self.decoder_hidden_size
        initializer_hidden_size = max(self.tcn_channels, self.decoder_hidden_size)
        initializer_output_size = 2 * self.num_layers * self.decoder_hidden_size
        self.state_initializer = nn.Sequential(
            nn.Linear(initializer_input_size, initializer_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(initializer_hidden_size, initializer_output_size),
        )
        decoder_state_residual_input_size = (
            3 * self.tcn_channels + self.decoder_hidden_size
        )
        self.decoder_state_residual_norm = nn.LayerNorm(
            decoder_state_residual_input_size
        )
        self.decoder_state_residual = nn.Sequential(
            nn.Linear(decoder_state_residual_input_size, initializer_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(initializer_hidden_size, initializer_output_size),
        )
        nn.init.zeros_(self.decoder_state_residual[-1].weight)
        nn.init.zeros_(self.decoder_state_residual[-1].bias)
        self.decoder_state_residual_scale = nn.Parameter(torch.tensor(0.05))

        # Bahdanau-style additive attention: key 来自 TCN 序列，query 来自 LSTM 初始 hidden。
        self.attn_key = nn.Linear(self.tcn_channels, self.attention_size, bias=False)
        self.attn_query = nn.Linear(
            self.decoder_hidden_size, self.attention_size, bias=False
        )
        self.attn_score = nn.Linear(self.attention_size, 1, bias=False)
        self.attitude_attn_key = nn.Linear(
            self.tcn_channels, self.attention_size, bias=False
        )
        self.attitude_attn_query = nn.Linear(
            self.decoder_hidden_size + input_size, self.attention_size, bias=False
        )
        self.attitude_attn_score = nn.Linear(self.attention_size, 1, bias=False)
        self.null_context = nn.Parameter(torch.zeros(self.tcn_channels))
        self.context_gate = nn.Linear(
            self.decoder_hidden_size + self.tcn_channels, 1
        )
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, 2.0)

        self.encoder_norm = nn.LayerNorm(self.tcn_channels)
        self.decoder_input_norm = nn.LayerNorm(
            self.decoder_hidden_size + self.tcn_channels
        )
        self.decoder_output_norm = nn.LayerNorm(self.decoder_hidden_size)
        self.shortcut_scale = nn.Parameter(torch.tensor(0.25))
        self.output_gain = nn.Parameter(torch.ones(output_size))
        # Keep the TCN anchor under the same name as the standalone TCN decoder so
        # TCN checkpoints initialize this path directly.
        self.decoder = MLP(
            self.tcn_channels, 1, decoder_sizes, output_size, dropout
        )
        self.base_delta_scale = nn.Parameter(torch.tensor(1.0))
        self.context_residual = nn.Sequential(
            nn.Linear(
                self.tcn_channels + self.decoder_hidden_size,
                self.decoder_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )
        self.context_modulator = nn.Sequential(
            nn.Linear(
                2 * self.decoder_hidden_size + self.tcn_channels,
                self.decoder_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            nn.Sigmoid(),
        )
        self.context_residual_scale = nn.Parameter(torch.tensor(0.5))
        gru_bridge_input_size = 3 * self.tcn_channels + 2 * self.decoder_hidden_size
        self.gru_context_bridge_norm = nn.LayerNorm(gru_bridge_input_size)
        self.gru_context_bridge_gate = nn.Sequential(
            nn.Linear(gru_bridge_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.tcn_channels),
            nn.Sigmoid(),
        )
        self.gru_context_bridge = nn.Sequential(
            nn.Linear(gru_bridge_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.tcn_channels),
        )
        nn.init.zeros_(self.gru_context_bridge[-1].weight)
        nn.init.zeros_(self.gru_context_bridge[-1].bias)
        self.gru_context_bridge_scale = nn.Parameter(torch.tensor(0.05))
        self.correction_gate = nn.Sequential(
            nn.Linear(
                2 * self.tcn_channels + 2 * self.decoder_hidden_size,
                self.decoder_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, output_size),
            nn.Sigmoid(),
        )
        self.context_delta_scale = nn.Parameter(torch.tensor(0.1))
        attitude_input_size = 2 * self.tcn_channels + 3 * self.decoder_hidden_size
        self.attitude_input_norm = nn.LayerNorm(attitude_input_size)
        self.attitude_gate = nn.Sequential(
            nn.Linear(attitude_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.attitude_decoder = nn.Sequential(
            nn.Linear(attitude_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.attitude_decoder[-1].weight)
        nn.init.zeros_(self.attitude_decoder[-1].bias)
        self.attitude_delta_scale = nn.Parameter(torch.tensor(0.05))
        attitude_fine_input_size = attitude_input_size + 6
        self.attitude_fine_norm = nn.LayerNorm(attitude_fine_input_size)
        self.attitude_fine_gate = nn.Sequential(
            nn.Linear(attitude_fine_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.attitude_fine_decoder = nn.Sequential(
            nn.Linear(attitude_fine_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.attitude_fine_decoder[-1].weight)
        nn.init.zeros_(self.attitude_fine_decoder[-1].bias)
        self.attitude_fine_delta_scale = nn.Parameter(torch.tensor(0.02))
        attitude_output_residual_input_size = attitude_input_size + 6
        self.attitude_output_residual_norm = nn.LayerNorm(
            attitude_output_residual_input_size
        )
        self.attitude_output_residual_gate = nn.Sequential(
            nn.Linear(attitude_output_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.attitude_output_residual_v2 = nn.Sequential(
            nn.Linear(attitude_output_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.attitude_output_residual_v2[-1].weight)
        nn.init.zeros_(self.attitude_output_residual_v2[-1].bias)
        self.attitude_output_residual_scale = nn.Parameter(torch.tensor(0.005))
        coupled_residual_input_size = attitude_input_size + output_size
        coupled_residual_rank = min(8, output_size)
        self.coupled_residual_norm = nn.LayerNorm(coupled_residual_input_size)
        self.coupled_residual_gate = nn.Sequential(
            nn.Linear(coupled_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, output_size),
            nn.Sigmoid(),
        )
        self.coupled_residual_coeff = nn.Sequential(
            nn.Linear(coupled_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, coupled_residual_rank),
        )
        self.coupled_residual_basis = nn.Linear(
            coupled_residual_rank, output_size, bias=False
        )
        nn.init.zeros_(self.coupled_residual_basis.weight)
        self.coupled_residual_scale = nn.Parameter(torch.tensor(0.01))
        long_delta_input_size = 3 * self.decoder_hidden_size + self.tcn_channels + 6
        self.long_delta_norm = nn.LayerNorm(long_delta_input_size)
        self.long_delta_gate = nn.Sequential(
            nn.Linear(long_delta_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.long_delta_decoder = nn.Sequential(
            nn.Linear(long_delta_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.long_delta_decoder[-1].weight)
        nn.init.zeros_(self.long_delta_decoder[-1].bias)
        self.long_delta_scale = nn.Parameter(torch.tensor(0.03))
        lag_context_input_size = 2 * self.tcn_channels + self.decoder_hidden_size
        self.lag_context_norm = nn.LayerNorm(lag_context_input_size)
        self.lag_context_proj = nn.Sequential(
            nn.Linear(lag_context_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )
        nn.init.zeros_(self.lag_context_proj[-1].weight)
        nn.init.zeros_(self.lag_context_proj[-1].bias)
        self.lag_context_scale = nn.Parameter(torch.tensor(0.05))
        velocity_residual_input_size = attitude_input_size + 6
        self.velocity_residual_norm = nn.LayerNorm(velocity_residual_input_size)
        self.velocity_residual_gate = nn.Sequential(
            nn.Linear(velocity_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
            nn.Sigmoid(),
        )
        self.velocity_residual_decoder = nn.Sequential(
            nn.Linear(velocity_residual_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
        )
        nn.init.zeros_(self.velocity_residual_decoder[-1].weight)
        nn.init.zeros_(self.velocity_residual_decoder[-1].bias)
        self.velocity_residual_scale = nn.Parameter(torch.tensor(0.005))

        if self.adaptive_history_context:
            side_token_size = input_size + 12 + self.history_context_dim
            self.tcnlstm_side_history_input_norm = nn.LayerNorm(side_token_size)
            self.tcnlstm_side_history_encoder = nn.Sequential(
                nn.Linear(side_token_size, self.decoder_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                nn.GELU(),
            )
            self.tcnlstm_side_history_null = nn.Parameter(
                torch.zeros(self.decoder_hidden_size)
            )
            selector_input_size = 4 * self.decoder_hidden_size
            self.tcnlstm_side_history_selector = nn.Sequential(
                nn.LayerNorm(selector_input_size),
                nn.Linear(selector_input_size, self.decoder_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_hidden_size, 4),
            )
            side_residual_input_size = (
                3 * self.decoder_hidden_size + self.tcn_channels
            )
            self.tcnlstm_side_history_residual_norm = nn.LayerNorm(
                side_residual_input_size
            )
            self.tcnlstm_side_history_reliability = nn.Sequential(
                nn.Linear(side_residual_input_size, self.decoder_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                nn.Sigmoid(),
            )
            self.tcnlstm_side_history_residual = nn.Sequential(
                nn.Linear(side_residual_input_size, self.decoder_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            )
            nn.init.zeros_(self.tcnlstm_side_history_residual[-1].weight)
            nn.init.zeros_(self.tcnlstm_side_history_residual[-1].bias)
            selector_out = self.tcnlstm_side_history_selector[-1]
            if self.side_history_selector_prior == "null_short":
                selector_out.bias.data.copy_(
                    torch.tensor([1.5, 1.0, -1.0, -2.0])
                )
            self.tcnlstm_side_history_scale = nn.Parameter(
                torch.tensor(self.side_history_scale_init)
            )

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
        # Context correction starts as a no-op and learns residual improvements on
        # top of the checkpoint-compatible TCN anchor.
        self.context_decoder = MLP(
            self.decoder_hidden_size, 1, decoder_sizes, output_size, dropout
        )
        nn.init.zeros_(self.context_decoder.model[-1].weight)
        nn.init.zeros_(self.context_decoder.model[-1].bias)
        self.memory = None

    def reset_memory(self):
        # 外部调试或切换序列时可以手动清空 decoder memory。
        self.memory = None

    def _refine_temporal_context(self, enc_seq):
        refiner_input = self.temporal_refiner_attn_norm(enc_seq)
        attn_out, _ = self.temporal_refiner_attn(
            refiner_input, refiner_input, refiner_input, need_weights=False
        )
        ffn_input = self.temporal_refiner_ffn_norm(enc_seq + attn_out)
        refine_delta = self.temporal_refiner_out(
            self.temporal_refiner_ffn(ffn_input)
        )
        return enc_seq + self.temporal_refiner_scale * refine_delta

    def _init_decoder_state(self, enc_seq, history_context, x_last):
        # enc_seq 的最后时刻代表当前窗口，mean 和 raw-history context 汇总隐状态。
        init_feature = torch.cat(
            [enc_seq[:, -1, :], enc_seq.mean(dim=1), history_context],
            dim=-1,
        )
        base_state = torch.tanh(self.state_initializer(init_feature))
        h0, c0 = torch.chunk(base_state, 2, dim=-1)

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

    def _attend_attitude(self, enc_seq, decoder_feature, x_last):
        query = self.attitude_attn_query(torch.cat([decoder_feature, x_last], dim=-1))
        scores = self.attitude_attn_score(
            torch.tanh(self.attitude_attn_key(enc_seq) + query.unsqueeze(1))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_seq).squeeze(1)
        return context

    def _quat_normalize(self, q):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _quat_log(self, q):
        q = self._quat_normalize(q)
        q = torch.where(q[..., :1] < 0.0, -q, q)
        vec = q[..., 1:]
        vec_norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vec_norm, q[..., :1].clamp_min(1e-12))
        return angle * vec / vec_norm.clamp_min(1e-12)

    def _geometric_motion_delta(self, x):
        delta = x.new_zeros(x.shape[0], x.shape[1], 12)
        delta[:, 1:, 0:3] = x[:, 1:, 0:3] - x[:, :-1, 0:3]
        delta[:, 1:, 3:6] = x[:, 1:, 3:6] - x[:, :-1, 3:6]
        q_prev = self._quat_normalize(x[:, :-1, 6:10])
        q_next = self._quat_normalize(x[:, 1:, 6:10])
        q_prev_inv = torch.cat([q_prev[..., :1], -q_prev[..., 1:]], dim=-1)
        delta[:, 1:, 6:9] = self._quat_log(self._quat_multiply(q_prev_inv, q_next))
        delta[:, 1:, 9:12] = x[:, 1:, 10:13] - x[:, :-1, 10:13]
        return delta

    def _tail_mean(self, encoded, window):
        window = min(max(int(window), 1), encoded.shape[1])
        return encoded[:, -window:, :].mean(dim=1)

    def _side_history_residual(
        self, x, context_hist, history_context, context, decoder_feature
    ):
        if not self.adaptive_history_context:
            return torch.zeros_like(history_context)

        motion_delta = self._geometric_motion_delta(x)
        token_parts = [x, motion_delta]
        if self.history_context_dim > 0:
            if context_hist is None:
                context_hist = x.new_zeros(
                    x.shape[0], x.shape[1], self.history_context_dim
                )
            token_parts.append(context_hist)
        side_tokens = torch.cat(token_parts, dim=-1)
        side_tokens = self.tcnlstm_side_history_input_norm(side_tokens)
        encoded = self.tcnlstm_side_history_encoder(side_tokens)

        null_context = self.tcnlstm_side_history_null.unsqueeze(0).expand(
            x.shape[0], -1
        )
        short_context = self._tail_mean(encoded, self.adaptive_history_short_window)
        mid_context = self._tail_mean(encoded, self.adaptive_history_mid_window)
        full_context = encoded.mean(dim=1)
        selector_contexts = torch.stack(
            [null_context, short_context, mid_context, full_context], dim=1
        )
        selector_feature = selector_contexts.flatten(start_dim=1)
        selector_weights = torch.softmax(
            self.tcnlstm_side_history_selector(selector_feature), dim=-1
        )
        side_context = torch.sum(selector_weights.unsqueeze(-1) * selector_contexts, dim=1)

        residual_input = self.tcnlstm_side_history_residual_norm(
            torch.cat([side_context, history_context, context, decoder_feature], dim=-1)
        )
        reliability = self.tcnlstm_side_history_reliability(residual_input)
        residual = self.tcnlstm_side_history_residual(residual_input)
        side_delta = self.tcnlstm_side_history_scale * reliability * residual

        weights = selector_weights.detach()
        reliability_detached = reliability.detach()
        self.adaptive_history_stats = {
            "null_weight": weights[:, 0].mean(),
            "short_weight": weights[:, 1].mean(),
            "mid_weight": weights[:, 2].mean(),
            "full_weight": weights[:, 3].mean(),
            "gate_saturation": (weights.max(dim=-1).values > 0.7).float().mean(),
            "reliability_mean": reliability_detached.mean(),
            "reliability_std": reliability_detached.std(unbiased=False),
            "reliability_saturation": (
                (reliability_detached < 0.05) | (reliability_detached > 0.95)
            )
            .float()
            .mean(),
            "side_residual_norm": side_delta.detach().norm(dim=-1).mean(),
        }
        return side_delta

    def forward(self, x, init_memory=True, return_attention=False, context_hist=None):
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
        x_anchor = x[:, -self.anchor_history_len :, :]
        enc_input = x_anchor.permute(0, 2, 1).contiguous()
        anchor_seq = self.encoder(enc_input).permute(0, 2, 1).contiguous()
        enc_seq = self.encoder_norm(anchor_seq)
        enc_seq = self._refine_temporal_context(enc_seq)
        _, (history_hidden, _) = self.history_lstm(x_anchor)
        history_context = self.history_norm(history_hidden[-1])
        x_last = x[:, -1, :]

        # init_memory=True 时从当前 enc_seq 初始化；否则尽量复用上一轮 decoder memory。
        if (
            init_memory
            or not self.use_recurrent_memory
            or not self._memory_is_usable(batch_size, x.device, x.dtype)
        ):
            state = self._init_decoder_state(enc_seq, history_context, x_last)
        else:
            state = self.memory

        # 用 decoder 初始 hidden 的最后一层作为 attention query。
        query = state[0][-1]
        context, attn_weights = self._attend(enc_seq, query)
        context_gate = torch.sigmoid(
            self.context_gate(torch.cat([query, context], dim=-1))
        )
        null_context = self.null_context.unsqueeze(0).expand_as(context)
        context = context_gate * context + (1.0 - context_gate) * null_context

        # decoder 输入只使用最后一帧原始输入和当前历史的 attention context。
        projected_x_last = self.input_proj(x_last)
        base_feature = anchor_seq[:, -1, :]
        base_delta = self.decoder(base_feature)
        decoder_input = torch.cat([projected_x_last, context], dim=-1).unsqueeze(1)
        decoder_input = self.decoder_input_norm(decoder_input)

        # 保存 LSTM 返回的新 hidden/cell state，供 init_memory=False 的下一次调用使用。
        decoder_out, self.memory = self.decoder_rnn(decoder_input, state)
        decoder_feature = decoder_out[:, -1, :]
        context_residual = self.context_residual(
            torch.cat([context, history_context], dim=-1)
        )
        context_gain = self.context_modulator(
            torch.cat([decoder_feature, context, history_context], dim=-1)
        )
        side_history_delta = self._side_history_residual(
            x, context_hist, history_context, context, decoder_feature
        )
        head_input = (
            decoder_feature
            + self.shortcut_scale * projected_x_last
            + self.context_residual_scale * context_gain * context_residual
            + side_history_delta
        )
        head_input = self.decoder_output_norm(head_input)
        context_delta = self.context_decoder(head_input)
        correction_gate = self.correction_gate(
            torch.cat([base_feature, context, history_context, decoder_feature], dim=-1)
        )
        attitude_context = self._attend_attitude(enc_seq, decoder_feature, x_last)
        attitude_input = self.attitude_input_norm(
            torch.cat(
                [
                    decoder_feature,
                    projected_x_last,
                    history_context,
                    context,
                    attitude_context,
                ],
                dim=-1,
            )
        )
        attitude_delta = torch.zeros_like(context_delta)
        attitude_delta[:, 6:12] = (
            self.attitude_delta_scale
            * self.attitude_gate(attitude_input)
            * self.attitude_decoder(attitude_input)
        )
        velocity_input = self.velocity_residual_norm(
            torch.cat([attitude_input, base_delta[:, :6]], dim=-1)
        )
        velocity_delta = torch.zeros_like(context_delta)
        velocity_delta[:, 3:6] = (
            self.velocity_residual_scale
            * self.velocity_residual_gate(velocity_input)
            * self.velocity_residual_decoder(velocity_input)
        )
        y = (
            self.base_delta_scale * base_delta
            + self.context_delta_scale * correction_gate * context_delta
            + attitude_delta
            + velocity_delta
        ) * self.output_gain

        if return_attention:
            return y, attn_weights
        return y
