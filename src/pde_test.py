"""Unit Test for regular_nd_grid_interpolation."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import unittest
import torch
import pde
import numpy as np
from parameterized import parameterized


def generate_test_data_heat_eqn():
    in_vars = 'x, y, t'
    out_vars = 'u, v'
    eqn_strs = ['dif(u, t) - (dif(dif(u, x), x) + dif(dif(u, y), y))',
                'dif(v, t) - (dif(dif(v, x), x) + dif(dif(v, y), y))']
    eqn_names = ['diffusion_u', 'diffusion_v']

    # arbitrary forward function, where inpt[0], inpt[1], inpt[2] correspond to x, y, t
    def fwd_fn(inpt):
        u = inpt[..., 0:1]**2 + 3*inpt[..., 1:2]**2*inpt[..., 2:3] + inpt[..., 0:1]*inpt[..., 2:3]
        v = inpt[..., 0:1]**2 + 3*inpt[..., 1:2]**2*inpt[..., 2:3] + inpt[..., 0:1]*inpt[..., 2:3]
        return torch.cat([u, v], axis=-1)

    # input tensor
    inpt = torch.tensor([[1., 2., 3.]])
    x, y, t = inpt[..., 0:1], inpt[..., 1:2], inpt[..., 2:3]
    g = x + 3*y**2 - 2 - 6*t
    expected_grads = {eqn_names[0]: g, eqn_names[1]: g}
    expected_val = fwd_fn(inpt)

    return in_vars, out_vars, eqn_strs, eqn_names, fwd_fn, inpt, expected_grads, expected_val

class PDELayerTest(unittest.TestCase):
    """Unit test for pde layer"""

    @parameterized.expand((
        generate_test_data_heat_eqn(),
        ))
    def test_pde_layer(self, in_vars, out_vars, eqn_strs, eqn_names,
                       fwd_fn, inpt, expected_grads, expected_val):
        """unit test for pde layer."""

        pdel = pde.PDELayer(in_vars=in_vars, out_vars=out_vars)
        for eqn_str, eqn_name in zip(eqn_strs, eqn_names):
            pdel.add_equation(eqn_str, eqn_name)
        pdel.update_forward_method(fwd_fn)
        val, grads = pdel(inpt)
        np.testing.assert_allclose(val.detach().numpy(), expected_val.detach().numpy(), atol=1e-4)
        for eqn_name in eqn_names:
            np.testing.assert_allclose(grads[eqn_name].detach().numpy(),
                                       expected_grads[eqn_name].detach().numpy())


if __name__ == '__main__':
    unittest.main()