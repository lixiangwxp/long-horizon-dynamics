import torch
import torch.nn as nn

from .mlp import MLP


class GRU(nn.Module):
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
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=encoder_sizes[0],
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

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
        memory = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
        x, memory = self.encoder(x, memory)
        self.memory = memory

        if self.encoder_output == "hidden":
            x_encoder = memory[-1]
        elif self.encoder_output == "output":
            x_encoder = x[:, -1, :]
        else:
            x_encoder = x

        x_encoder = self.dropout(x_encoder)
        return self.decoder(x_encoder)

    def init_memory(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
