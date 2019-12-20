"""Unit Test for implicit_net."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import unittest
import numpy as np
import torch
from parameterized import parameterized
import implicit_net


class ImplicitNetTest(unittest.TestCase):
    """Unit test for implicit_net"""

    @parameterized.expand((
        [32, 2048, 4, 3, 32, 16],
        ))
    def test_imnet(self, batch_size, npts, n_in, n_out, n_chan, n_filter):
        """unit test."""
        input_coords = torch.rand(batch_size, npts, n_in)
        input_chan = torch.rand(batch_size, npts, n_chan)
        inputs = torch.cat([input_coords, input_chan], axis=-1)
        model = implicit_net.ImNet(dim=n_in, in_features=n_chan, out_features=n_out, nf=n_filter)
        out = model(inputs)

        np.testing.assert_allclose(out.shape, [batch_size, npts, n_out], atol=1e-4)

if __name__ == '__main__':
    unittest.main()