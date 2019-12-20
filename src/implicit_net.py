"""Implementation of implicit networks architecture.
"""
import torch
import torch.nn as nn


class ImNet(nn.Module):
  """ImNet layer pytorch implementation.
  """

  def __init__(self, dim=3, in_features=128, out_features=1, nf=128,
               activation=torch.nn.LeakyReLU):
    """Initialization.

    Args:
      dim: int, dimension of input points.
      in_features: int, length of input features (i.e., latent code).
      out_features: number of output features.
      nf: int, width of the second to last layer.
      activation: tf activation op.
      name: str, name of the layer.
    """
    super(ImNet, self).__init__()
    self.dim = dim
    self.in_features = in_features
    self.dimz = dim + in_features
    self.out_features = out_features
    self.nf = nf
    self.activ = activation()
    self.fc0 = nn.Linear(self.dimz, nf*16)
    self.fc1 = nn.Linear(nf*16 + self.dimz, nf*8)
    self.fc2 = nn.Linear(nf*8 + self.dimz, nf*4)
    self.fc3 = nn.Linear(nf*4 + self.dimz, nf*2)
    self.fc4 = nn.Linear(nf*2 + self.dimz, nf*1)
    self.fc5 = nn.Linear(nf*1, out_features)
    self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
    self.fc = nn.ModuleList(self.fc)

  def forward(self, x):
    """Forward method.

    Args:
      x: `[batch_size, dim+in_features]` tensor, inputs to decode.
    Returns:
      x_: output through this layer.
    """
    x_ = x
    for dense in self.fc[:4]:
        x_ = self.activ(dense(x_))
        x_ = torch.cat([x_, x], dim=-1)
    x_ = self.activ(self.fc4(x_))
    x_ = self.fc5(x_)
    return x_
