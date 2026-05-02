import torch
import torch.nn as nn
from torch.autograd import Variable
from .mlp import MLP

class GRU(nn.Module):
  """GRU 编码器 + MLP 解码器。

  输入 x 通常为 [batch, history_len, input_size]。GRU 沿时间维读取历史，
  forward 再按 encoder_output 选择隐藏状态、最后时间步输出或完整序列作为解码输入。
  """
  def __init__(self, input_size, encoder_sizes, num_layers, history_len, decoder_sizes, output_size, dropout,
               encoder_output, **kwargs):
    super(GRU, self).__init__()
    # batch_first=True 表示输入/输出的时间序列布局为 [batch, time, features]。
    self.encoder = nn.GRU(input_size=input_size, hidden_size=encoder_sizes[0],
                          num_layers=num_layers, batch_first=True, dropout=dropout)
    decoder_input = history_len if encoder_output == 'output' else num_layers
    # decoder_input 需要与后面传入 MLP 的张量形状配合，MLP 会在 forward 中展平除 batch 外的维度。
    if encoder_output == 'hidden':
      decoder_input = 2 * (encoder_sizes[0] / history_len)
    elif encoder_output == 'output':
      decoder_input = encoder_sizes[0] / history_len
    
    else:
      decoder_input = encoder_sizes[0]

    self.decoder = MLP(decoder_input, history_len, decoder_sizes, output_size, dropout)
    self.dropout = nn.Dropout(dropout)
    self.encoder_output = encoder_output
    self.num_layers = num_layers
    self.hidden_size = encoder_sizes[0]
    self.memory = None

  def forward(self, x, init_memory):
    """根据 init_memory 决定初始化 GRU hidden state，或延续 self.memory。"""
    h = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
    x, h = self.encoder(x, h)
    # GRU 的记忆只有 hidden state，形状为 [num_layers, batch, hidden_size]。
    self.memory = h


    if self.encoder_output == 'hidden':
      # 当前实现拼接 h[0][-1] 与 h[1][-1]，可理解为取前两层在最后 batch 索引处的状态片段。
      x_encoder = torch.cat([h[0][-1], h[1][-1]], dim=1)

    elif self.encoder_output == 'output':
      # 只保留最后一个历史时间步的输出，形状 [batch, hidden_size]。
      x_encoder = x[:, -1, :]

    else:
      # 保留完整 GRU 输出序列，形状 [batch, history_len, hidden_size]，由 MLP 负责展平。
      x_encoder = x
      
    x_encoder = self.dropout(x_encoder)
    x = self.decoder(x_encoder)
    
    return x

  def init_memory(self, batch_size, device):
    # 初始 hidden state 全零，设备与输入 x 保持一致。
    return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device)
  
if __name__=='__main__':
    input_size = 19 #number of features
    encoder_sizes = 256
    num_layers = 2
    history_len = 10
    dropout = 0.2
    output_type = 'hidden'
    encoder_dim = 256

    model = GRU(input_size, encoder_dim, encoder_sizes, num_layers, history_len, dropout, output_type)

    x = torch.randn(16384, 10, 19)

    y = model(x)

    print(y.shape)
