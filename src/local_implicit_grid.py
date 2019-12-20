"""Local implicit grid query function."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import torch
import torch.nn as nn
import regular_nd_grid_interpolation as rgi


def query_local_implicit_grid(model, latent_grid, query_pts, xmin, xmax):
    """Function for querying local implicit grid.

    The latent feature grid can be of aribtrary physical dimensions. query_pts are query points
    representing the coordinates at which the grid is queried. xmin and xmax are the bounds of
    the grid. E.g. for a 2 dimensional grid,

                   *--------*--------* xmax
                   |        |    x   |
                   |        |        |
                   |     x  |        |
                   |        |   x    |
                   *--------*--------*
                   |        |        |
                   |        |        |
                   | x      |        |
                   |        |      x |
              xmin *--------*--------*
    In the schematic above, for the latent grid, n1=n2=2, num_pts=5, etc.

    Args:
        model: nn.Module instance, model for decoding local latents. Must accept input of length
        d+c.
        latent_grid: tensor of shape [b, n1, n2, ..., nd, c] where b is the batch size, n1, ..., nd
        are the spatial resolution in each dimension, c is the number of latent channels.
        query_pts: tensor of shape [b, num_pts, d] where num_pts is the number of query points, d is
        the dimension of the query points. The query points must fall within xmin and xmax, or else
        will be clipped to xmin and xmax.
        xmin: float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the lower left corner of the grid.
        xmax:float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the upper right corner of the
        grid.
    Returns:
        query_vals: tensor of shape [b, num_pts, o], queried values at query points, where o is the
        number output channels from the model.
    """
    corner_values, weights, x_relative = rgi.regular_nd_grid_interpolation_coefficients(
        latent_grid, query_pts, xmin, xmax)
    concat_features = torch.cat([x_relative, corner_values], axis=-1)  # [b, num_points, 2**d,  d+c]
    input_shape = concat_features.shape

    # flatten and feed through model
    output = model(concat_features.reshape([-1, input_shape[-1]]))

    # reshape output
    output = output.reshape([input_shape[0], input_shape[1], input_shape[2], -1])  # [b, p, 2**d, o]

    # interpolate the output values
    output = torch.sum(output * weights.unsqueeze(-1), axis=-2)  # [b, p, o]

    return output


