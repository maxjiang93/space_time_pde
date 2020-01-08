"""Regular multi-linear ND Grid Interpolation."""

# pylint: disable=no-member, too-many-locals, not-callable

import torch
import numpy as np


def clip_tensor(input_tensor, xmin, xmax):
    """Clip tensor per by per column bounds."""
    return torch.max(torch.min(input_tensor, xmax), xmin)


def regular_nd_grid_interpolation_coefficients(grid, query_pts, xmin=0., xmax=1.):
    """Batched regular n-dimensional grid interpolation (return interp coefficents).

    Args:
        grid: tensor of shape (batch, *self.size, in_features)
        query_pts: tensor of shape (batch, num_points, dim) within range (xmin, xmax)
        xmin: float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the lower left corner of the grid.
        xmax:float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the upper right corner of the
        grid.
    Returns:
        corner_values: tensor of shape (batch, num_points, 2**dim, in_features), value at the cell
        corners for the each cell that query_pts falls into.
        weights: tensor of shape (batch, num_points, 2**dim), interpolation weights of the corners.
        x_relative: tensor of shape (batch, num_points, 2**dim, dim), relative coordinate of the
        query points with respect to each corner. Normalized between (-1, 1).

    """

    # dimension
    device = grid.device
    dim = len(grid.shape) - 2
    size = torch.tensor(grid.shape[1:-1]).float().to(device)

    # convert xmin and xmax
    if isinstance(xmin, (int, float)) or isinstance(xmax, (int, float)):
        xmin = float(xmin) * torch.ones([dim], dtype=torch.float32, device=grid.device)
        xmax = float(xmax) * torch.ones([dim], dtype=torch.float32, device=grid.device)
    elif isinstance(xmin, (list, tuple, np.ndarray)) or isinstance(xmax, (list, tuple, np.ndarray)):
        xmin = torch.tensor(xmin).to(grid.device)
        xmax = torch.tensor(xmax).to(grid.device)

    # clip query_pts
    eps = 1e-6 * (xmax - xmin)
    query_pts = clip_tensor(query_pts, xmin+eps, xmax-eps)

    cubesize = (xmax - xmin) / (size - 1)
    ind0 = torch.floor(query_pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = ind0 + 1
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    ind_b = (torch.arange(grid.shape[0])
             .expand(ind_n.shape[1], ind_n.shape[2], grid.shape[0])
             .permute(2, 0, 1))  # (batch, num_points, 2**dim)

    # latent code on neighbor nodes
    unpack_ind_n = tuple([ind_b] + [ind_n[..., i] for i in range(ind_n.shape[-1])])
    corner_values = grid[unpack_ind_n] # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.float() * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.float() + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2, 3, 0, 1)   # (batch, num_points, 2**dim, dim)
    dxyz_ = torch.abs(query_pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    x_relative = (query_pts.unsqueeze(-2) - pos) / cubesize # (batch, num_points, 2**dim, dim)

    return corner_values, weights, x_relative


def regular_nd_grid_interpolation(grid, query_pts, xmin=0., xmax=1.):
    """Batched regular n-dimensional grid interpolation.

    Performs n-dimensional multi-linear interpolation. I.e., linear interpolation for 1D grid,
    bilinear interpolation for 2D grid, trilinear interpolation for 3D grid, etc. The values are on
    the grid points, whereas the interpolation values are queries at query points.
    Args:
        grid: tensor of shape (batch, *self.size, in_features)
        query_pts: tensor of shape (batch, num_points, dim) within range (xmin, xmax)
        xmin: float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the lower left corner of the grid.
        xmax:float or tuple of floats or tensor. If float, automatically broadcast to the
        corresponding dimensions. Reference spatial coordinate of the upper right corner of the
        grid.
    Returns:
        query_vals: tensor of shape (batch, num_points, in_features), values at query points.

    """

    corner_values, weights, _ = regular_nd_grid_interpolation_coefficients(
        grid, query_pts, xmin, xmax)
    interp_val = torch.sum(corner_values * weights.unsqueeze(-1), axis=-2)

    return interp_val
