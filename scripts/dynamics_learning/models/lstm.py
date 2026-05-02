import torch
import torch.nn as nn
from torch.autograd import Variable
from .mlp import MLP

class LSTM(nn.Module):
  """LSTM 编码器 + MLP 解码器。

  输入 x 通常为 [batch, history_len, input_size]。LSTM 先沿时间维编码历史，
  再根据 encoder_output 选择隐藏状态、最后时间步输出或完整序列交给 MLP 解码。
  """
  def __init__(self, input_size, encoder_sizes, num_layers, history_len, decoder_sizes, output_size, dropout,
               encoder_output, **kwargs):
    super(LSTM, self).__init__()
    # batch_first=True 使 LSTM 接收 [batch, time, features]，与数据窗口的常见布局一致。
    self.encoder = nn.LSTM(input_size=input_size, hidden_size=encoder_sizes[0],
                           num_layers=num_layers, batch_first=True, dropout=dropout)
    decoder_input = history_len if encoder_output == 'output' else num_layers
    # decoder_input 是传给 MLP 的“单个时间步/状态”的特征宽度；MLP 内部还会乘 history_len 后展平。
    if encoder_output == 'hidden':
      decoder_input = 2 * (encoder_sizes[0] / history_len)
      #让MLP的输入维度与encoder输出的特征维度匹配。 MLP 会自动乘以 history_len 来展平时间维。
      #2 * (H / T) * T= 2H
    elif encoder_output == 'output':
      decoder_input = encoder_sizes[0] / history_len
    else:
      decoder_input = encoder_sizes[0]

    self.decoder = MLP(decoder_input, history_len, decoder_sizes, output_size, dropout)
    self.encoder_output = encoder_output
    self.dropout = nn.Dropout(dropout)
    self.num_layers = num_layers
    self.hidden_size = encoder_sizes[0]
    self.memory = None

  def forward(self, x, init_memory):
    """根据 init_memory 决定重置记忆或复用上一批的 LSTM 状态。"""
    h = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
    x, _ = self.encoder(x, h)
    #PyTorch 的 nn.LSTM 返回的是：output, (h_n, c_n) = lstm(input, (h_0, c_0))
    #但这里写成了 x, _ = ...，把更新后的 (h_n, c_n) 丢掉了。然后 self.memory = h 存回去的是输入进去的旧状态，不是 LSTM 跑完后的新状态。
    self.memory = h
    # x_encoder = torch.cat([h[0][-1], h[1][-1]], dim=1) if self.encoder_output == 'hidden' else x[:, -1, :]
    if self.encoder_output == 'hidden':
      #LSTM 的 h 是(h_n, c_n)，h_n.shape = [num_layers, B, H]，c_n.shape = [num_layers, B, H]
      # 使用最后一层 hidden state 与 cell state 拼接，形状近似 [batch, 2 * hidden_size]。
      x_encoder = torch.cat([h[0][-1], h[1][-1]], dim=1)

    elif self.encoder_output == 'output':
      # 只取最后一个时间步的 LSTM 输出，形状 [batch, hidden_size]。[B, T, H]
      x_encoder = x[:, -1, :]#[B, H]

    else:
      # 保留完整时间序列输出，形状 [batch, history_len, hidden_size]，交给 MLP 展平。[B, T, H]
      x_encoder = x

    x_encoder = self.dropout(x_encoder)
    x = self.decoder(x_encoder)
    return x

  def init_memory(self, batch_size, device):
    # LSTM 记忆由 hidden state 和 cell state 组成，形状均为 [num_layers, batch, hidden_size]。
    return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device))
