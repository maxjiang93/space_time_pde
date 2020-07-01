import torch
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

        
NONLINEARITIES = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "swish": Swish,
    "leakyrelu": nn.LeakyReLU,
}
