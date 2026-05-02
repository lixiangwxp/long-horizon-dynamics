import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size, history_len, decoder_sizes, output_size, dropout, **kwargs
    ):
        super().__init__()
        self.model = self.make(
            int(input_size * history_len), decoder_sizes, output_size, dropout
        )

    def make(self, input_size, decoder_sizes, output_size, dropout):
        layers = [
            nn.Linear(input_size, decoder_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
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
        x = x.reshape(x.shape[0], -1)
        return self.model(x)
