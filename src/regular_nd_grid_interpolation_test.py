"""Unit Test for regular_nd_grid_interpolation."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import unittest
import numpy as np
import torch
from parameterized import parameterized
import regular_nd_grid_interpolation as rgi


# test data for 1d case
GRID_DATA_1D = torch.arange(11).float().unsqueeze(0).unsqueeze(-1)
POINTS_1D = torch.rand(100).unsqueeze(0).unsqueeze(-1)
GT_1D = POINTS_1D * 10.

# test data for 2d case
X, Y = torch.meshgrid(torch.arange(11), torch.arange(11))
GRID_DATA_2D = torch.stack([X, Y], dim=-1).unsqueeze(0)
POINTS_2D = torch.rand(100, 2).unsqueeze(0)
GT_2D = POINTS_2D * 10.

# test data for 3d case
X, Y, Z = torch.meshgrid(torch.arange(11), torch.arange(11), torch.arange(11))
GRID_DATA_3D = torch.stack([X, Y, Z], dim=-1).unsqueeze(0)
POINTS_3D = torch.rand(100, 3).unsqueeze(0)
GT_3D = POINTS_3D * 10.

class RegularNDGridInterpolationTest(unittest.TestCase):
    """Unit test for regular_nd_grid_interpolation"""

    @parameterized.expand((
        [GRID_DATA_1D, POINTS_1D, 0., 1., GT_1D],
        [GRID_DATA_2D, POINTS_2D, 0., 1., GT_2D],
        [GRID_DATA_3D, POINTS_3D, 0., 1., GT_3D],
        ))
    def test_regular_grid_interpolation(self, grid, query_pts, xmin, xmax, ground_truth):
        """unit test."""
        out = rgi.regular_nd_grid_interpolation(grid, query_pts, xmin, xmax)
        np.testing.assert_allclose(out.numpy(), ground_truth.numpy(), atol=1e-4)

if __name__ == '__main__':
    unittest.main()
