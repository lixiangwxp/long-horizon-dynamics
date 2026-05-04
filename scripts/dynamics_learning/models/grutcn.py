import torch
import torch.nn as nn

from .mlp import MLP
from .tcn import TemporalConvNet


class GRUTCN(nn.Module):
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

        self.encoder = TemporalConvNet(
            input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout
        )

        self.state_initializer = nn.Sequential(
            nn.Linear(2 * self.tcn_channels, self.decoder_hidden_size),
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

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
        self.memory = None

    def _encode(self, x):
        enc_input = x.permute(0, 2, 1).contiguous()
        return self.encoder(enc_input).permute(0, 2, 1).contiguous()

    def _init_hidden(self, enc_seq):
        init_feature = torch.cat([enc_seq[:, -1, :], enc_seq.mean(dim=1)], dim=-1)
        hidden = self.state_initializer(init_feature)
        batch_size = enc_seq.shape[0]
        hidden = hidden.reshape(
            batch_size, self.num_layers, self.decoder_hidden_size
        ).permute(1, 0, 2)
        return hidden.contiguous()

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
        enc_seq = self._encode(x)

        if (
            init_memory
            or not self.use_recurrent_memory
            or not self._memory_matches(batch_size)
            or self.memory.device != x.device
            or self.memory.dtype != x.dtype
        ):
            state = self._init_hidden(enc_seq)
        else:
            state = self.memory

        query = state[-1]
        context, attn_weights = self._attend(enc_seq, query)

        x_last = x[:, -1, :]
        projected_x_last = self.input_proj(x_last)
        decoder_input = torch.cat([projected_x_last, context], dim=-1).unsqueeze(1)

        decoder_out, self.memory = self.decoder_rnn(decoder_input, state)
        y = self.decoder(decoder_out[:, -1, :])

        if return_attention:
            return y, attn_weights
        return y
