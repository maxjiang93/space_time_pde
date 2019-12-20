"""Unit Test for local_implicit_grid."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import unittest
import numpy as np
import torch
from parameterized import parameterized
import local_implicit_grid as lig
import implicit_net


class LocalImplicitGridTest(unittest.TestCase):
    """Unit test for local_implicit_grid"""

    @parameterized.expand((
        [8, 512, 3, 32, 3, 16, 16, 0., 1.],  # 3-dimensional coords test case
        [8, 512, 4, 32, 3, 16, 16, 0., 1.],  # 4-dimensional coords test case
        ))
    def test_query_local_implicit_grid(self, batch_size, npts, n_dim, n_in, n_out, n_filter,
                                       latent_grid_size, xmin, xmax):
        """unit test."""
        query_pts = torch.rand(batch_size, npts, n_dim)
        model = implicit_net.ImNet(dim=n_dim, in_features=n_in, out_features=n_out, nf=n_filter)
        latent_grid = torch.rand(batch_size, *([latent_grid_size] * n_dim), n_in)
        # [b, n1, ..., nd, c]
        # import pdb; pdb.set_trace()
        out = lig.query_local_implicit_grid(model, latent_grid, query_pts, xmin, xmax)

        np.testing.assert_allclose(out.shape, [batch_size, npts, n_out], atol=1e-4)

if __name__ == '__main__':
    unittest.main()