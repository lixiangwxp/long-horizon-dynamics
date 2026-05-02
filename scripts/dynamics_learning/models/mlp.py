import torch
import torch.nn as nn

class MLP(nn.Module):
  """把一段历史特征展平后送入全连接解码器。

  期望输入通常是 [batch, history_len, input_size] 或已经等价展开的张量；
  forward 中会保留 batch 维，把其余维度合并成 [batch, input_size * history_len]。
  """
  def __init__(self, input_size, history_len, decoder_sizes, output_size, dropout, **kwargs):
    super(MLP, self).__init__()
    # input_size 与 history_len 相乘后，对应 forward 展平后的单样本特征维度。
    self.model = self.make(int(input_size * history_len), decoder_sizes, output_size, dropout)

  def make(self, input_size, decoder_sizes, output_size, dropout):
    """按 decoder_sizes 组装 Linear/GELU/Dropout 堆叠，最后映射到 output_size。"""
    layers = []
    layers.append(nn.Linear(input_size, decoder_sizes[0]))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))
    for i in range(len(decoder_sizes) - 1):
      layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
      layers.append(nn.GELU())
      layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(decoder_sizes[-1], output_size))
    return nn.Sequential(*layers)

  def forward(self, x, args=None):
    # 保留 batch 维，将历史窗口和特征维合并，输出为 [batch, output_size]。
    x = x.reshape(x.shape[0], -1)
    x = self.model(x)
    return x
