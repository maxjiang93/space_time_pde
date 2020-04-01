"""3D U-Net with residual blocks.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=invalid-name, too-many-instance-attributes, arguments-differ, too-many-arguments


class ResBlock3D(nn.Module):
    """3D convolutional Residue Block. Maintains same resolution.
    """

    def __init__(self, in_channels, neck_channels, out_channels, final_relu=True):
        """Initialization.

        Args:
          in_channels: int, number of input channels.
          neck_channels: int, number of channels in bottleneck layer.
          out_channels: int, number of output channels.
          final_relu: bool, add relu to the last layer.
        """
        super(ResBlock3D, self).__init__()
        self.in_channels = in_channels
        self.neck_channels = neck_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, neck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(neck_channels, neck_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(neck_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(num_features=neck_channels)
        self.bn2 = nn.BatchNorm3d(num_features=neck_channels)
        self.bn3 = nn.BatchNorm3d(num_features=out_channels)

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.final_relu = final_relu

    def forward(self, x):  # pylint:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += self.shortcut(identity)
        if self.final_relu:
            x = F.relu(x)

        return x


class UNet3d(nn.Module):  # pylint: disable=too-many-instance-attributes
    """UNet that consumes even dimension grid and outputs odd dimension grid.
    """

    def __init__(self, in_features=4, out_features=32, igres=(4, 32, 32), ogres=None,
                 nf=16, mf=512):
        """initialize 3D UNet.

        Args:
          in_features: int, number of input features.
          out_features: int, number of output features.
          igres: tuple, input grid resolution in each dimension. each dimension must be integer
          powers of 2.
          ogres: tuple, output grid resolution in each dimension. each dimension must be integer
          powers of 2. #NOTE for now must be same as igres or must be 2^k multipliers of igres.
          nf: int, number of base feature layers.
          mf: int, a cap for max number of feature layers throughout the network.
        """
        super(UNet3d, self).__init__()
        self.igres = igres

        self.nf = nf
        self.mf = mf
        self.in_features = in_features
        self.out_features = out_features

        # for now ogres must be igres, else not implemented
        if ogres is None:
            self.ogres = self.igres
        else:
            self.ogres = ogres
        
        # assert integer multipliers of igres
        mul = np.array(self.ogres) / np.array(self.igres)
        fac = np.log2(mul)
        if not np.allclose(fac%1, np.zeros_like(fac)):
            raise ValueError("ogres must be 2^k times greater than igres where k >= 0. "
                             "Instead igres: {}, ogres: {}".format(igres, ogres))
        if not np.all(fac>=0):
            raise ValueError("ogres must be greater or equal to igres. "
                             "Instead igres: {}, ogres: {}".format(igres, ogres))
        self.exp_fac = fac.astype(np.int32)
        if not np.allclose(self.exp_fac, np.zeros_like(self.exp_fac)):
            self.expand = True
        else:
            self.expand = False

        # assert dimensions acceptable
        if isinstance(self.igres, int):
            self.igres = tuple([self.igres] * 3)
        if isinstance(self.ogres, int):
            self.ogres = tuple([self.ogres] * 3)

        self._check_grid_res()
        self.li = math.log(np.max(np.array(self.igres)), 2)  # input layers
        self.lo = math.log(np.max(np.array(self.ogres)), 2)  # output layers
        assert self.li % 1 == 0
        assert self.lo % 1 == 0
        self.li = int(self.li)  # number of input levels
        self.lo = int(self.lo)  # number of output levels

        self._create_layers()

    def _check_grid_res(self):
        # check type
        if not (hasattr(self.igres, '__len__') and hasattr(self.ogres, '__len__')):
            raise TypeError('igres and ogres must be tuples for grid dimensions')
        # check size
        if not (len(self.igres) == 3 and len(self.ogres) == 3):
            raise ValueError('igres and ogres must have len = 3, however detected to be'
                             '{} and {}'.format(len(self.igres), len(self.ogres)))
        # check powers of 2
        for d in list(self.igres) + list(self.ogres):
            if not (math.log(d, 2) % 1 == 0 and np.issubdtype(type(d), np.integer)):
                raise ValueError('dimensions in igres and ogres must be  integer powers of 2.'
                                 'instead they are {} and {}.'.format(self.igres, self.ogres))

    def _create_layers(self):
        # num. features in downward path
        nfeat_down_out = [self.nf*(2**(i+1)) for i in range(self.li)]

        # cap the maximum number of feature layers
        nfeat_down_out = [n if n <= self.mf else self.mf for n in nfeat_down_out]
        nfeat_down_in = [self.nf] + nfeat_down_out[:-1]

        # num. features in upward path
        # self.nfeat_up = nfeat_down_out[::-1][:self.lo]
        nfeat_up_in = [int(n*2) for n in nfeat_down_in[::-1][:-1]]
        nfeat_up_out = nfeat_down_in[::-1][1:]

        self.conv_in = ResBlock3D(self.in_features, self.nf, self.nf)
        self.conv_out = ResBlock3D(nfeat_down_in[0]*2, nfeat_down_in[0]*2, self.out_features,
                                   final_relu=False)
        self.conv_mid = ResBlock3D(nfeat_down_out[-1], nfeat_down_out[-2], nfeat_down_out[-2])
        self.down_modules = [ResBlock3D(n_in, int(n/2), n) for n_in, n in zip(nfeat_down_in,
                                                                              nfeat_down_out)]
        self.up_modules = [ResBlock3D(n_in, n, n) for n_in, n in zip(nfeat_up_in,
                                                                     nfeat_up_out)]
        self.down_pools = []
        self.up_interps = []

        prev_layer_dims = np.array(self.igres)
        for _ in range(len(nfeat_down_out)):
            pool_kernel_size, next_layer_dims = self._get_pool_kernel_size(prev_layer_dims)
            pool_layer = nn.MaxPool3d(pool_kernel_size)
            # use the reverse op in the upward branch
            upsamp_layer = nn.Upsample(scale_factor=tuple(pool_kernel_size))
            self.down_pools.append(pool_layer)
            self.up_interps = [upsamp_layer] + self.up_interps  # add to front
            prev_layer_dims = next_layer_dims
            
        # create expansion modules
        if self.expand:
            n_exp = np.max(self.exp_fac)
#             self.exp_modules = [ResBlock3D(2*self.nf, self.nf, self.nf)]
#             self.exp_modules = self.exp_modules + [ResBlock3D(self.nf, self.nf, self.nf) for _ in range(n_exp-1)]
            self.exp_modules = [ResBlock3D(2*self.nf, 2*self.nf, 2*self.nf) for _ in range(n_exp)]
            self.exp_interps = []
            for _ in range(n_exp):
                exp_kernel_size, self.exp_fac = self._get_exp_kernel_size(self.exp_fac)
                self.exp_interps.append(nn.Upsample(scale_factor=tuple(exp_kernel_size)))
            self.exp_interps = nn.ModuleList(self.exp_interps)
            self.exp_modules = nn.ModuleList(self.exp_modules)
            
        self.down_modules = nn.ModuleList(self.down_modules)
        self.up_modules = nn.ModuleList(self.up_modules)
        self.down_pools = nn.ModuleList(self.down_pools)
        self.up_interps = nn.ModuleList(self.up_interps)

    @staticmethod
    def _get_pool_kernel_size(prev_layer_dims):
        if np.all(prev_layer_dims == np.min(prev_layer_dims)):
            next_layer_dims = (prev_layer_dims/2).astype(np.int)
            pool_kernel_size = [2, 2, 2]
        else:
            min_dim = np.min(prev_layer_dims)
            pool_kernel_size = [1 if d == min_dim else 2 for d in prev_layer_dims]
            next_layer_dims = [int(d/k) for d, k in zip(prev_layer_dims, pool_kernel_size)]
            next_layer_dims = np.array(next_layer_dims)

        return pool_kernel_size, next_layer_dims

    @staticmethod
    def _get_exp_kernel_size(prev_exp_fac):
        """Get expansion kernel size."""
        next_exp_fac = np.clip(prev_exp_fac - 1, 0, None)
        exp_kernel_size = prev_exp_fac - next_exp_fac + 1
        return exp_kernel_size, next_exp_fac
        
    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch, in_features, igres[0], igres[1], igres[2]]` tensor, input voxel grid.
        Returns:
          `[batch, out_features, ogres[0], ogres[1], ogres[2]]` tensor, output voxel grid.
        """
        x = self.conv_in(x)
        x_dns = [x]
        for mod, pool_op in zip(self.down_modules, self.down_pools):
            x = pool_op(mod(x_dns[-1]))
            x_dns.append(x)

        x = x_dns.pop(-1)
        upsamp_op = self.up_interps[0]
        x = self.conv_mid(upsamp_op(x))

        for mod, upsamp_op in zip(self.up_modules, self.up_interps[1:]):
            x = torch.cat([x, x_dns.pop(-1)], dim=1)
            x = mod(x)
            x = upsamp_op(x)

        x = torch.cat([x, x_dns.pop(-1)], dim=1)

        if self.expand:
            for mod, upsamp_op in zip(self.exp_modules, self.exp_interps):
                x = mod(x)
                x = upsamp_op(x)
            
        x = self.conv_out(x)

        return x


class Encoder3d(nn.Module):  # pylint: disable=too-many-instance-attributes
    """3D convolutional encoder that consumes even dimension grid and outputs a vector.
    """

    def __init__(self, in_features=4, out_features=32, igres=(4, 32, 32), nf=16, mf=512):
        """initialize 3D convolutional encoder.

        Args:
          in_features: int, number of input features.
          out_features: int, number of output features.
          igres: tuple, input grid resolution in each dimension. each dimension must be integer
          powers of 2.
          powers of 2. #NOTE for now must be same as igres or must be 2^k multipliers of igres.
          nf: int, number of base feature layers.
          mf: int, a cap for max number of feature layers throughout the network.
        """
        super(Encoder3d, self).__init__()
        self.igres = igres
        self.nf = nf
        self.mf = mf
        self.in_features = in_features
        self.out_features = out_features

        # assert dimensions acceptable
        if isinstance(self.igres, int):
            self.igres = tuple([self.igres] * 3)

        self._check_grid_res()
        self.li = math.log(np.max(np.array(self.igres)), 2)  # input layers
        assert self.li % 1 == 0
        self.li = int(self.li)  # number of input levels

        self._create_layers()

    def _check_grid_res(self):
        # check type
        if not (hasattr(self.igres, '__len__')):
            raise TypeError('igres must be tuples for grid dimensions')
        # check size
        if not (len(self.igres) == 3):
            raise ValueError('igres must have len = 3, however detected to be'
                             '{}'.format(len(self.igres)))
        # check powers of 2
        for d in list(self.igres):
            if not (math.log(d, 2) % 1 == 0 and np.issubdtype(type(d), np.integer)):
                raise ValueError('dimensions in igres must be integer powers of 2.'
                                 'instead it is {}.'.format(self.igres))

    def _create_layers(self):
        # num. features in downward path
        nfeat_down_out = [self.nf*(2**(i+1)) for i in range(self.li)]

        # cap the maximum number of feature layers
        nfeat_down_out = [n if n <= self.mf else self.mf for n in nfeat_down_out]
        nfeat_down_in = [self.nf] + nfeat_down_out[:-1]

        self.conv_in = ResBlock3D(self.in_features, self.nf, self.nf)
        self.conv_out = nn.Conv3d(nfeat_down_out[-1], self.out_features, kernel_size=1, stride=1)
        
        self.down_modules = [ResBlock3D(n_in, int(n/2), n) for n_in, n in zip(nfeat_down_in,
                                                                              nfeat_down_out)]
        self.down_pools = []

        prev_layer_dims = np.array(self.igres)
        for _ in range(len(nfeat_down_out)):
            pool_kernel_size, next_layer_dims = self._get_pool_kernel_size(prev_layer_dims)
            pool_layer = nn.MaxPool3d(pool_kernel_size)
            self.down_pools.append(pool_layer)
            prev_layer_dims = next_layer_dims
            
        self.down_modules = nn.ModuleList(self.down_modules)
        self.down_pools = nn.ModuleList(self.down_pools)

    @staticmethod
    def _get_pool_kernel_size(prev_layer_dims):
        if np.all(prev_layer_dims == np.min(prev_layer_dims)):
            next_layer_dims = (prev_layer_dims/2).astype(np.int)
            pool_kernel_size = [2, 2, 2]
        else:
            min_dim = np.min(prev_layer_dims)
            pool_kernel_size = [1 if d == min_dim else 2 for d in prev_layer_dims]
            next_layer_dims = [int(d/k) for d, k in zip(prev_layer_dims, pool_kernel_size)]
            next_layer_dims = np.array(next_layer_dims)

        return pool_kernel_size, next_layer_dims

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch, in_features, igres[0], igres[1], igres[2]]` tensor, input voxel grid.
        Returns:
          `[batch, out_features]` tensor, output feature vectors.
        """
        x = self.conv_in(x)
        for mod, pool_op in zip(self.down_modules, self.down_pools):
            x = pool_op(mod(x))
            
        x = self.conv_out(x)
        x = x.view([x.shape[0], x.shape[1]])  # squeeze out all spatial dimensions

        return x
    

if __name__ == '__main__':
#     unet = UNet3d(out_features=4, nf=8, igres=(4, 16, 16), ogres=(16, 128, 128))
    x_samp = torch.rand(4, 4, 4, 16, 16)  # [batch, in_features, rest, resx, resy]
#     y = unet(x_samp)
    encoder = Encoder3d(in_features=4, out_features=32, nf=16, igres=(4, 16, 16))
    y = encoder(x_samp)
    print(y.shape)