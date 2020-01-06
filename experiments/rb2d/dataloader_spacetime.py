"""RB2 Experiment Dataloader"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class RB2DataLoader(Dataset):  # pylint: disable=too-many-instance-attributes
    """Pytorch Dataset instance for loading Rayleigh Bernard 2D dataset.

    Loads a 2d space + time cubic cutout from the whole simulation.
    """
    def __init__(self, data_path="./", data_filename="rb2d_ra1e6_s42.npz",  # pylint: disable=too-many-arguments
                 nx=128, ny=128, nt=16, n_samp_pts_per_crop=1024, interp_method='linear',
                 downsamp_xy=4, downsamp_t=4):
        """

        Initialize DataSet
        Args:
          data_path: str, path to the dataset folder, default="./"
          data_filename: str, name of the dataset file, default="rb2d_ra1e6_s42"
          nx: int, number of 'pixels' in x dimension for high res dataset.
          ny: int, number of 'pixels' in y dimension for high res dataset.
          nt: int, number of timesteps in time for high res dataset.
          n_samp_pts_per_crop: int, number of sample points to return per crop.
          interp_method: str, interpolation method. choice of 'linear/nearest'
          downsamp_xy: int, downsampling factor for the spatial dimensions.
          downsamp_t: int, downsampling factor for the temporal dimension.
        """
        self.data_path = data_path
        self.data_filename = data_filename
        self.nx_hres = nx
        self.ny_hres = ny
        self.nt_hres = nt
        self.nx_lres = int(nx/downsamp_xy)
        self.ny_lres = int(ny/downsamp_xy)
        self.nt_lres = int(nt/downsamp_t)
        self.n_samp_pts_per_crop = n_samp_pts_per_crop
        self.interp_method = interp_method
        self.downsamp_xy = downsamp_xy
        self.downsamp_t = downsamp_t

        # concatenating pressure, temperature, x-velocity, and z-velocity as a 4 channel array: pbuw
        # shape: (4, 200, 512, 128)
        npdata = np.load(self.data_path + self.data_filename)
        self.data = np.stack([npdata['p'], npdata['b'], npdata['u'], npdata['w']], axis=0)
        nc_data, nt_data, nx_data, ny_data = self.data.shape

        # assert nx, ny, nt are viable
        if (nx > nx_data) or (ny > ny_data) or (nt > nt_data):
            raise ValueError('Resolution in each spatial temporal dimension x ({}), y({}), t({})'
                             'must not exceed dataset limits x ({}) y ({}) t ({})'.format(
                                 nx, ny, nt, nx_data, ny_data, nt_data))
        if (nt % downsamp_t != 0) or (nx % downsamp_xy != 0) or (ny % downsamp_xy != 0):
            raise ValueError('nx, ny and nt must be divisible by downsamp factor.')

        self.nx_start_range = np.arange(0, nx_data-nx+1)
        self.ny_start_range = np.arange(0, ny_data-ny+1)
        self.nt_start_range = np.arange(0, nt_data-nt+1)
        self.rand_grid = np.stack(np.meshgrid(self.nx_start_range,
                                              self.ny_start_range,
                                              self.nt_start_range, indexing='ij'), axis=-1)
        # (xaug, yaug, taug, 3)
        self.rand_start_id = self.rand_grid.reshape([-1, 3])
        self.scale_hres = np.array([self.nt_hres, self.nx_hres, self.ny_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nt_lres, self.nx_lres, self.ny_lres], dtype=np.int32)

        # compute channel-wise mean and std
        self._mean = np.mean(self.data, axis=(1, 2, 3))
        self._std = np.std(self.data, axis=(1, 2, 3))

    def __len__(self):
        return self.rand_start_id.shape[0]

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the crop to return. must be smaller than len(self).

        Returns:
          space_time_crop_lres: array of shape [nt_lres, nx_lres, ny_lres]
          point_coord: array of shape [n_samp_pts_per_crop, 3], where 3 are the t, x, y dims.
          point_value: array of shape [n_samp_pts_per_crop, 4], where 4 are the phys channels pbuw.
        """
        x_id, y_id, t_id = self.rand_start_id[idx]
        space_time_crop_hres = self.data[:,
                                         t_id:t_id+self.nt_hres,
                                         x_id:x_id+self.nx_hres,
                                         y_id:y_id+self.ny_hres]
        # space_time_crop default shape: (c, nt, nx, ny)

        # create low res grid from hires space time crop
        interp = RegularGridInterpolator(
            (np.arange(self.nt_hres), np.arange(self.nx_hres), np.arange(self.ny_hres)),
            values=np.transpose(space_time_crop_hres, axes=(1, 2, 3, 0)), method=self.interp_method)
        lres_coord = np.stack(np.meshgrid(np.linspace(0, self.nt_hres-1, self.nt_lres),
                                          np.linspace(0, self.nx_hres-1, self.nx_lres),
                                          np.linspace(0, self.ny_hres-1, self.ny_lres),
                                          indexing='ij'), axis=-1)
        space_time_crop_lres = interp(lres_coord)

        # create random point samples within space time crop
        point_coord = np.random.rand(self.n_samp_pts_per_crop, 3) * (self.scale_hres - 1)
        point_value = interp(point_coord)
        point_coord = point_coord / (self.scale_hres - 1) * (self.scale_lres - 1)

        return (space_time_crop_lres.astype(np.float32),
                point_coord.astype(np.float32),
                point_value.astype(np.float32))

    @property
    def channel_mean(self):
        """channel-wise mean of dataset."""
        return self._mean

    @property
    def channel_std(self):
        """channel-wise mean of dataset."""
        return self._std



if __name__ == '__main__':
    ### example for using the data loader
    data_loader = RB2DataLoader(nt=4, n_samp_pts_per_crop=10000, downsamp_t=2)
    # lres_crop, point_coord, point_value = data_loader[61234]
    # import matplotlib.pyplot as plt
    # plt.scatter(point_coord[:, 1], point_coord[:, 2], c=point_value[:, 0])
    # plt.colorbar()
    # plt.show()
    # plt.imshow(lres_crop[0, :, :, 0].T, origin='lower'); plt.show()
    # plt.imshow(lres_crop[1, :, :, 0].T, origin='lower'); plt.show()

    data_batches = torch.utils.data.DataLoader(
        data_loader, batch_size=16, shuffle=True, num_workers=1)

    for batch_idx, (lowres_input_batch, point_coords, point_values) in enumerate(data_batches):
        print("Reading batch #{}:\t with lowres inputs of size {}, sample coord of size {}, sampe val of size {}"
              .format(batch_idx+1, list(lowres_input_batch.shape),  list(point_coords.shape), list(point_values.shape)))
        if batch_idx > 16:
            break
