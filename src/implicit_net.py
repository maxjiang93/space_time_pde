"""Implementation of implicit networks architecture.
"""
import torch
import torch.nn as nn
# pylint: disable=too-many-instance-attributes, too-many-arguments, invalid-name


class ImNet(nn.Module):
    """ImNet layer pytorch implementation.
    """

    def __init__(self, dim=3, in_features=32, out_features=4, nf=32,
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
          output through this layer of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)
        return x_tmp
