import torch
import torch.nn as nn

from .mlp import MLP
from .tcn import TemporalConvNet


class GRUTCN(nn.Module):
    """TCN encoder + GRU decoder with latent motion-state context.

    The base TCN path preserves checkpoint behavior. Failed motion-diff and
    output-observer candidates are kept for key compatibility. The active
    raw-token branch reads ordered raw history with a lightweight Transformer
    side path and injects zero-init output residuals. For history-expanded
    runs, the calibrated anchor keeps using the recent H20 window while a
    zero-init adaptive branch learns how much short/mid/full history to use.
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
        self.anchor_history_len = min(history_len, 20)
        self.sampling_frequency = kwargs.get("sampling_frequency", 100)
        self.multi_step_delta_vomega = kwargs.get("multi_step_delta_vomega", False)
        self.multi_step_kinematic_update = kwargs.get(
            "multi_step_kinematic_update", False
        )
        self.num_layers = num_layers
        self.tcn_channels = encoder_sizes[-1]
        self.decoder_hidden_size = (
            self.tcn_channels if decoder_hidden_size is None else decoder_hidden_size
        )
        self.attention_size = (
            self.decoder_hidden_size if attention_size is None else attention_size
        )
        self.use_recurrent_memory = use_recurrent_memory

        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        latent_se_hidden = max(16, self.tcn_channels // 4)
        self.latent_se_norm = nn.LayerNorm(self.tcn_channels)
        self.latent_se_gate = nn.Sequential(
            nn.Linear(self.tcn_channels, latent_se_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_se_hidden, self.tcn_channels),
        )
        nn.init.zeros_(self.latent_se_gate[-1].weight)
        nn.init.zeros_(self.latent_se_gate[-1].bias)
        self.latent_se_scale = nn.Parameter(torch.tensor(0.02))
        self.motion_encoder = TemporalConvNet(
            2 * input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )
        self.motion_encoder_norm = nn.LayerNorm(self.tcn_channels)
        motion_fusion_size = 3 * self.tcn_channels
        self.motion_fusion_norm = nn.LayerNorm(motion_fusion_size)
        self.motion_fusion = nn.Sequential(
            nn.Linear(motion_fusion_size, self.tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.tcn_channels, self.tcn_channels),
        )
        nn.init.zeros_(self.motion_fusion[-1].weight)
        nn.init.zeros_(self.motion_fusion[-1].bias)
        self.motion_fusion_scale = nn.Parameter(torch.tensor(0.05))
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
        self.context_gru = nn.GRU(
            input_size=self.tcn_channels,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.history_norm = nn.LayerNorm(self.decoder_hidden_size)

        self.state_initializer = nn.Sequential(
            nn.Linear(
                2 * self.tcn_channels + self.decoder_hidden_size,
                self.decoder_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.decoder_hidden_size,
                self.num_layers * self.decoder_hidden_size,
            ),
            nn.Tanh(),
        )
        self.attn_key = nn.Linear(self.tcn_channels, self.attention_size, bias=False)
        self.attn_query = nn.Linear(
            self.decoder_hidden_size, self.attention_size, bias=False
        )
        self.attn_score = nn.Linear(self.attention_size, 1, bias=False)
        self.null_context = nn.Parameter(torch.zeros(self.tcn_channels))
        self.context_gate = nn.Linear(
            self.decoder_hidden_size + self.tcn_channels, 1
        )
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, 2.0)
        dual_context_size = 3 * self.tcn_channels + self.decoder_hidden_size
        self.dual_context_norm = nn.LayerNorm(dual_context_size)
        self.dual_context_mlp = nn.Sequential(
            nn.Linear(dual_context_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.tcn_channels),
        )
        nn.init.zeros_(self.dual_context_mlp[-1].weight)
        nn.init.zeros_(self.dual_context_mlp[-1].bias)
        self.dual_context_scale = nn.Parameter(torch.tensor(0.1))

        self.encoder_norm = nn.LayerNorm(self.tcn_channels)
        self.decoder_input_norm = nn.LayerNorm(
            self.decoder_hidden_size + self.tcn_channels
        )
        self.decoder_output_norm = nn.LayerNorm(self.decoder_hidden_size)
        self.shortcut_scale = nn.Parameter(torch.tensor(0.25))
        self.output_gain = nn.Parameter(torch.ones(output_size))
        self.base_decoder = MLP(
            self.tcn_channels, 1, decoder_sizes, output_size, dropout
        )
        self.tcn_anchor_proj = nn.Sequential(
            nn.Linear(self.tcn_channels, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(
                self.tcn_channels + 3 * self.decoder_hidden_size,
                self.decoder_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )
        self.context_fusion_scale = nn.Parameter(torch.tensor(0.5))
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
        self.memory_refresh_gate = nn.Linear(
            self.tcn_channels + self.decoder_hidden_size, self.num_layers
        )
        nn.init.zeros_(self.memory_refresh_gate.weight)
        nn.init.constant_(self.memory_refresh_gate.bias, 0.85)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.raw_history_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        raw_context_size = 3 * self.decoder_hidden_size
        self.raw_history_norm = nn.LayerNorm(raw_context_size)
        self.raw_history_mlp = nn.Sequential(
            nn.Linear(raw_context_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )
        nn.init.zeros_(self.raw_history_mlp[-1].weight)
        nn.init.zeros_(self.raw_history_mlp[-1].bias)
        self.raw_history_scale = nn.Parameter(torch.tensor(0.1))
        raw_token_size = self.decoder_hidden_size
        raw_token_heads = 4 if raw_token_size % 4 == 0 else 1
        self.raw_token_input_norm = nn.LayerNorm(2 * input_size)
        self.raw_token_proj = nn.Linear(2 * input_size, raw_token_size)
        self.raw_token_pos = nn.Parameter(
            torch.zeros(1, self.anchor_history_len, raw_token_size)
        )
        self.raw_token_adaptive_pos = nn.Parameter(
            torch.zeros(1, history_len, raw_token_size)
        )
        raw_token_layer = nn.TransformerEncoderLayer(
            d_model=raw_token_size,
            nhead=raw_token_heads,
            dim_feedforward=2 * raw_token_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.raw_token_encoder = nn.TransformerEncoder(
            raw_token_layer, num_layers=2
        )
        raw_token_query_size = 3 * self.decoder_hidden_size
        self.raw_token_query_norm = nn.LayerNorm(raw_token_query_size)
        self.raw_token_query = nn.Linear(raw_token_query_size, raw_token_size)
        self.raw_token_score = nn.Linear(raw_token_size, 1, bias=False)
        self.raw_token_context_norm = nn.LayerNorm(raw_token_size)
        raw_token_adaptive_input_size = 5 * raw_token_size
        self.raw_token_adaptive_gate_norm = nn.LayerNorm(
            raw_token_adaptive_input_size
        )
        self.raw_token_adaptive_gate = nn.Sequential(
            nn.Linear(raw_token_adaptive_input_size, raw_token_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(raw_token_size, 3),
        )
        self.raw_token_adaptive_context_norm = nn.LayerNorm(
            raw_token_adaptive_input_size
        )
        self.raw_token_adaptive_context_delta = nn.Sequential(
            nn.Linear(raw_token_adaptive_input_size, raw_token_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(raw_token_size, raw_token_size),
        )
        nn.init.zeros_(self.raw_token_adaptive_context_delta[-1].weight)
        nn.init.zeros_(self.raw_token_adaptive_context_delta[-1].bias)
        self.raw_token_adaptive_context_scale = nn.Parameter(torch.tensor(0.02))
        raw_token_head_input_size = (
            raw_token_size
            + 2 * self.tcn_channels
            + 3 * self.decoder_hidden_size
        )
        self.raw_token_head_norm = nn.LayerNorm(raw_token_head_input_size)
        self.raw_token_head_delta = nn.Sequential(
            nn.Linear(raw_token_head_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )
        nn.init.zeros_(self.raw_token_head_delta[-1].weight)
        nn.init.zeros_(self.raw_token_head_delta[-1].bias)
        self.raw_token_head_scale = nn.Parameter(torch.tensor(0.02))
        raw_token_observer_base = (
            raw_token_size
            + 2 * self.tcn_channels
            + 3 * self.decoder_hidden_size
            + input_size
        )
        raw_token_velocity_input_size = raw_token_observer_base + 3
        self.raw_token_velocity_norm = nn.LayerNorm(
            raw_token_velocity_input_size
        )
        self.raw_token_velocity_gate = nn.Sequential(
            nn.Linear(raw_token_velocity_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
            nn.Sigmoid(),
        )
        self.raw_token_velocity_decoder = nn.Sequential(
            nn.Linear(raw_token_velocity_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
        )
        nn.init.zeros_(self.raw_token_velocity_decoder[-1].weight)
        nn.init.zeros_(self.raw_token_velocity_decoder[-1].bias)
        self.raw_token_velocity_scale = nn.Parameter(torch.tensor(0.003))
        raw_token_attitude_input_size = raw_token_observer_base + 6
        self.raw_token_attitude_norm = nn.LayerNorm(
            raw_token_attitude_input_size
        )
        self.raw_token_attitude_gate = nn.Sequential(
            nn.Linear(raw_token_attitude_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.raw_token_attitude_decoder = nn.Sequential(
            nn.Linear(raw_token_attitude_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.raw_token_attitude_decoder[-1].weight)
        nn.init.zeros_(self.raw_token_attitude_decoder[-1].bias)
        self.raw_token_attitude_scale = nn.Parameter(torch.tensor(0.003))
        multi_step_vomega_input_size = (
            raw_token_size
            + 2 * self.tcn_channels
            + 3 * self.decoder_hidden_size
            + input_size
            + 6
        )
        self.multi_step_vomega_norm = nn.LayerNorm(multi_step_vomega_input_size)
        self.multi_step_vomega_gate = nn.Sequential(
            nn.Linear(multi_step_vomega_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.multi_step_vomega_decoder = nn.Sequential(
            nn.Linear(multi_step_vomega_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.multi_step_vomega_decoder[-1].weight)
        nn.init.zeros_(self.multi_step_vomega_decoder[-1].bias)
        self.multi_step_vomega_scale = nn.Parameter(torch.tensor(0.02))
        self.multi_step_kinematic_scale = nn.Parameter(torch.tensor(0.0))
        velocity_observer_input_size = (
            2 * self.tcn_channels + 2 * self.decoder_hidden_size + input_size + 3
        )
        self.velocity_observer_norm = nn.LayerNorm(
            velocity_observer_input_size
        )
        self.velocity_observer_gate = nn.Sequential(
            nn.Linear(velocity_observer_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
            nn.Sigmoid(),
        )
        self.velocity_observer_decoder = nn.Sequential(
            nn.Linear(velocity_observer_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 3),
        )
        nn.init.zeros_(self.velocity_observer_decoder[-1].weight)
        nn.init.zeros_(self.velocity_observer_decoder[-1].bias)
        self.velocity_observer_scale = nn.Parameter(torch.tensor(0.003))
        attitude_observer_input_size = (
            2 * self.tcn_channels
            + 3 * self.decoder_hidden_size
            + input_size
            + 6
        )
        self.attitude_observer_norm = nn.LayerNorm(attitude_observer_input_size)
        self.attitude_observer_gate = nn.Sequential(
            nn.Linear(attitude_observer_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
            nn.Sigmoid(),
        )
        self.attitude_observer_decoder = nn.Sequential(
            nn.Linear(attitude_observer_input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_hidden_size, 6),
        )
        nn.init.zeros_(self.attitude_observer_decoder[-1].weight)
        nn.init.zeros_(self.attitude_observer_decoder[-1].bias)
        self.attitude_observer_scale = nn.Parameter(torch.tensor(0.003))

        self.decoder_rnn = nn.GRU(
            input_size=self.decoder_hidden_size + self.tcn_channels,
            hidden_size=self.decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = MLP(
            self.decoder_hidden_size, 1, decoder_sizes, output_size, dropout
        )
        nn.init.zeros_(self.decoder.model[-1].weight)
        nn.init.zeros_(self.decoder.model[-1].bias)
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

    def _apply_latent_se(self, enc_seq):
        channel_delta = torch.tanh(
            self.latent_se_gate(self.latent_se_norm(enc_seq.mean(dim=1)))
        ).unsqueeze(1)
        return enc_seq + self.latent_se_scale * enc_seq * channel_delta

    def _encode(self, x):
        x_anchor = x[:, -self.anchor_history_len :, :]
        enc_input = x_anchor.permute(0, 2, 1).contiguous()
        enc_seq = self.encoder(enc_input).permute(0, 2, 1).contiguous()
        enc_seq = self.encoder_norm(enc_seq)
        enc_seq = self._apply_latent_se(enc_seq)
        enc_seq = self._refine_temporal_context(enc_seq)
        _, history_state = self.context_gru(enc_seq)
        history_context = self.history_norm(history_state[-1])
        return enc_seq, history_context

    def _encode_raw_tokens(self, x, position):
        dx = torch.zeros_like(x)
        dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        raw_tokens = torch.cat([x, dx], dim=-1)
        raw_tokens = self.raw_token_proj(self.raw_token_input_norm(raw_tokens))
        raw_tokens = raw_tokens + position[:, : x.shape[1], :]
        raw_tokens = self.raw_token_encoder(raw_tokens)
        return raw_tokens, dx[:, -1, :]

    def _window_mean(self, tokens, window_size):
        window_size = min(window_size, tokens.shape[1])
        return tokens[:, -window_size:, :].mean(dim=1)

    def _apply_raw_adaptive_context(self, raw_context, full_tokens, query):
        short_context = self._window_mean(full_tokens, 10)
        mid_context = self._window_mean(full_tokens, 25)
        full_context = full_tokens.mean(dim=1)
        gate_input = torch.cat(
            [raw_context, query, short_context, mid_context, full_context],
            dim=-1,
        )
        scale_weights = torch.softmax(
            self.raw_token_adaptive_gate(
                self.raw_token_adaptive_gate_norm(gate_input)
            ),
            dim=-1,
        )
        scale_contexts = torch.stack(
            [short_context, mid_context, full_context],
            dim=1,
        )
        adaptive_context = torch.sum(
            scale_weights.unsqueeze(-1) * scale_contexts,
            dim=1,
        )
        delta_input = self.raw_token_adaptive_context_norm(
            torch.cat(
                [raw_context, adaptive_context, short_context, mid_context, full_context],
                dim=-1,
            )
        )
        return (
            raw_context
            + self.raw_token_adaptive_context_scale
            * self.raw_token_adaptive_context_delta(delta_input)
        )

    def _raw_token_context(self, x, history_context, decoder_feature, projected_x_last):
        x_anchor = x[:, -self.anchor_history_len :, :]
        raw_tokens, _ = self._encode_raw_tokens(x_anchor, self.raw_token_pos)
        query_input = self.raw_token_query_norm(
            torch.cat([history_context, decoder_feature, projected_x_last], dim=-1)
        )
        query = self.raw_token_query(query_input)
        weights = torch.softmax(
            self.raw_token_score(torch.tanh(raw_tokens + query.unsqueeze(1))).squeeze(-1),
            dim=1,
        )
        raw_context = torch.bmm(weights.unsqueeze(1), raw_tokens).squeeze(1)
        raw_context = self.raw_token_context_norm(raw_context)
        full_tokens, dx_last = self._encode_raw_tokens(x, self.raw_token_adaptive_pos)
        raw_context = self._apply_raw_adaptive_context(raw_context, full_tokens, query)
        return raw_context, dx_last

    def _apply_multi_step_delta_vomega(self, x_last, y, context_features):
        (
            raw_token_context,
            base_feature,
            context,
            history_context,
            decoder_feature,
            projected_x_last,
            raw_token_dx_last,
        ) = context_features
        multi_step_input = self.multi_step_vomega_norm(
            torch.cat(
                [
                    raw_token_context,
                    base_feature,
                    context,
                    history_context,
                    decoder_feature,
                    projected_x_last,
                    raw_token_dx_last,
                    y[:, 3:6],
                    y[:, 9:12],
                ],
                dim=-1,
            )
        )
        multi_step_delta = (
            self.multi_step_vomega_scale
            * self.multi_step_vomega_gate(multi_step_input)
            * self.multi_step_vomega_decoder(multi_step_input)
        )
        delta_v = y[:, 3:6] + multi_step_delta[:, 0:3]
        delta_omega = y[:, 9:12] + multi_step_delta[:, 3:6]

        if self.multi_step_kinematic_update:
            dt = x_last.new_tensor(1.0 / float(self.sampling_frequency))
            kin_delta_p = dt * (x_last[:, 3:6] + 0.5 * delta_v)
            kin_dtheta = dt * (x_last[:, 10:13] + 0.5 * delta_omega)
            y = y.clone()
            y[:, 0:3] = y[:, 0:3] + self.multi_step_kinematic_scale * (
                kin_delta_p - y[:, 0:3]
            )
            y[:, 6:9] = y[:, 6:9] + self.multi_step_kinematic_scale * (
                kin_dtheta - y[:, 6:9]
            )

        y = y.clone()
        y[:, 3:6] = delta_v
        y[:, 9:12] = delta_omega
        return y

    def _init_hidden(self, enc_seq, history_context):
        init_feature = torch.cat(
            [enc_seq[:, -1, :], enc_seq.mean(dim=1), history_context],
            dim=-1,
        )
        hidden = self.state_initializer(init_feature)
        batch_size = enc_seq.shape[0]
        hidden = hidden.reshape(
            batch_size, self.num_layers, self.decoder_hidden_size
        ).permute(1, 0, 2)
        return hidden.contiguous()

    def _blend_memory(self, fresh_state, enc_seq, history_context):
        refresh_feature = torch.cat([enc_seq[:, -1, :], history_context], dim=-1)
        refresh = torch.sigmoid(self.memory_refresh_gate(refresh_feature))
        refresh = refresh.permute(1, 0).unsqueeze(-1)
        return (refresh * fresh_state + (1.0 - refresh) * self.memory).contiguous()

    def _attend(self, enc_seq, query):
        keys = self.attn_key(enc_seq)
        q = self.attn_query(query).unsqueeze(1)
        scores = self.attn_score(torch.tanh(keys + q)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_seq).squeeze(1)
        return context, weights

    def _memory_matches(self, batch_size):
        return self.memory is not None and self.memory.shape[1] == batch_size

    def reset_memory(self):
        self.memory = None

    def forward(self, x, init_memory=True, return_attention=False):
        if x.ndim != 3:
            raise ValueError(
                f"GRUTCN expected input shape [B, H, F], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.history_len:
            raise ValueError(
                f"GRUTCN expected history_len={self.history_len}, got {x.shape[1]}."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"GRUTCN expected input_size={self.input_size}, got {x.shape[-1]}."
            )

        batch_size = x.shape[0]
        enc_seq, history_context = self._encode(x)
        fresh_state = self._init_hidden(enc_seq, history_context)

        if (
            init_memory
            or not self.use_recurrent_memory
            or not self._memory_matches(batch_size)
            or self.memory.device != x.device
            or self.memory.dtype != x.dtype
        ):
            state = fresh_state
        else:
            state = self._blend_memory(fresh_state, enc_seq, history_context)

        query = state[-1]
        context, attn_weights = self._attend(enc_seq, query)
        context_gate = torch.sigmoid(
            self.context_gate(torch.cat([query, context], dim=-1))
        )
        null_context = self.null_context.unsqueeze(0).expand_as(context)
        context = context_gate * context + (1.0 - context_gate) * null_context
        dual_context = torch.cat(
            [context, history_context, enc_seq[:, -1, :], enc_seq.mean(dim=1)],
            dim=-1,
        )
        context = context + self.dual_context_scale * self.dual_context_mlp(
            self.dual_context_norm(dual_context)
        )

        x_last = x[:, -1, :]
        projected_x_last = self.input_proj(x_last)
        x_anchor = x[:, -self.anchor_history_len :, :]
        raw_history = self.raw_history_proj(x_anchor).mean(dim=1)
        base_feature = enc_seq[:, -1, :]
        base_delta = self.base_decoder(base_feature)
        tcn_anchor = self.tcn_anchor_proj(base_feature)
        decoder_input = torch.cat([projected_x_last, context], dim=-1).unsqueeze(1)
        decoder_input = self.decoder_input_norm(decoder_input)

        decoder_out, self.memory = self.decoder_rnn(decoder_input, state)
        decoder_feature = decoder_out[:, -1, :]
        raw_token_context, raw_token_dx_last = self._raw_token_context(
            x, history_context, decoder_feature, projected_x_last
        )
        latent_feature = self.context_fusion(
            torch.cat(
                [decoder_feature, context, history_context, projected_x_last],
                dim=-1,
            )
        )
        raw_residual = self.raw_history_mlp(
            self.raw_history_norm(
                torch.cat([raw_history, projected_x_last, history_context], dim=-1)
            )
        )
        head_input = (
            decoder_feature
            + self.shortcut_scale * projected_x_last
            + tcn_anchor
            + self.context_fusion_scale * latent_feature
            + self.raw_history_scale * raw_residual
        )
        raw_token_head_input = self.raw_token_head_norm(
            torch.cat(
                [
                    raw_token_context,
                    base_feature,
                    context,
                    history_context,
                    decoder_feature,
                    projected_x_last,
                ],
                dim=-1,
            )
        )
        head_input = (
            head_input
            + self.raw_token_head_scale
            * self.raw_token_head_delta(raw_token_head_input)
        )
        head_input = self.decoder_output_norm(head_input)
        context_delta = self.decoder(head_input)
        correction_gate = self.correction_gate(
            torch.cat([base_feature, context, history_context, decoder_feature], dim=-1)
        )
        y = (
            base_delta
            + self.context_delta_scale * correction_gate * context_delta
        ) * self.output_gain
        dx_last = x[:, -1, :] - x[:, -2, :]
        velocity_input = self.velocity_observer_norm(
            torch.cat(
                [
                    base_feature,
                    context,
                    history_context,
                    projected_x_last,
                    dx_last,
                    base_delta[:, 3:6],
                ],
                dim=-1,
            )
        )
        velocity_delta = torch.zeros_like(y)
        velocity_delta[:, 3:6] = (
            self.velocity_observer_scale
            * self.velocity_observer_gate(velocity_input)
            * self.velocity_observer_decoder(velocity_input)
        )
        y = y + velocity_delta
        attitude_input = self.attitude_observer_norm(
            torch.cat(
                [
                    base_feature,
                    context,
                    history_context,
                    projected_x_last,
                    decoder_feature,
                    dx_last,
                    base_delta[:, 6:12],
                ],
                dim=-1,
            )
        )
        attitude_delta = torch.zeros_like(y)
        attitude_delta[:, 6:12] = (
            self.attitude_observer_scale
            * self.attitude_observer_gate(attitude_input)
            * self.attitude_observer_decoder(attitude_input)
        )
        y = y + attitude_delta
        raw_token_common = [
            raw_token_context,
            base_feature,
            context,
            history_context,
            decoder_feature,
            projected_x_last,
            raw_token_dx_last,
        ]
        raw_token_velocity_input = self.raw_token_velocity_norm(
            torch.cat(raw_token_common + [y[:, 3:6]], dim=-1)
        )
        raw_token_delta = torch.zeros_like(y)
        raw_token_delta[:, 3:6] = (
            self.raw_token_velocity_scale
            * self.raw_token_velocity_gate(raw_token_velocity_input)
            * self.raw_token_velocity_decoder(raw_token_velocity_input)
        )
        raw_token_attitude_input = self.raw_token_attitude_norm(
            torch.cat(raw_token_common + [y[:, 6:12]], dim=-1)
        )
        raw_token_delta[:, 6:12] = (
            self.raw_token_attitude_scale
            * self.raw_token_attitude_gate(raw_token_attitude_input)
            * self.raw_token_attitude_decoder(raw_token_attitude_input)
        )
        y = y + raw_token_delta

        if self.multi_step_delta_vomega:
            y = self._apply_multi_step_delta_vomega(
                x_last,
                y,
                (
                    raw_token_context,
                    base_feature,
                    context,
                    history_context,
                    decoder_feature,
                    projected_x_last,
                    raw_token_dx_last,
                ),
            )

        if return_attention:
            return y, attn_weights
        return y
