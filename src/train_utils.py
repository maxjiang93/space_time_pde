"""Utility tools for training the model.
"""
import logging
import os
import shutil
import torch
import numpy as np
from matplotlib import cm, colors

# pylint: disable=too-many-arguments


def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    """Save checkpoint.

    Args:
        state: dict, containing state of the model to save.
        is_best: bool, indicate whether this is the best model so far.
        epoch: int, epoch number.
        output_folder: str, path to output folder.
        filename: str, the name to save the model as.
        logger: logger object to log progress.
    """
    if epoch > 1:
        os.remove(output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')


def snapshot_files(list_of_filenames, log_dir):
    """Snapshot list of files in current run state to the log directory.

    Args:
        list_of_filenames: list of str.
        log_dir: str, log directory to save code snapshots.
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    for filename in list_of_filenames:
        shutil.copy2(filename, os.path.join(log_dir, filename))


def get_logger(log_dir, name='train', level=logging.DEBUG, log_file_name='log.txt'):
    """Get a logger that writes a log file in log_dir.

    Args:
        log_dir: str, log directory to save logs.
        name: str, name of the logger instance.
        level: logging level.
        log_file_name: str, name of the log file to output.
    Returns:
        a logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(log_dir, os.path.basename(log_file_name)))
    logger.addHandler(file_handler)
    return logger


def colorize_scalar_tensors(x, vmin=None, vmax=None, cmap='viridis', out_channel='rgb'):
    """Colorize scalar field tensors.

    Args:
        x: torch tensor of shape [H, W].
        vmin: float, min value to normalize the colors to.
        vmax: float, max value to normalize the colors to.
        cmap: str or Colormap instance, the colormap used to map normalized data values to RGBA
        colors.
        out_channel: str, 'rgb' or 'rgba'.
    Returns:
        y: torch tensor of shape [H, W, 3(or 4 if out_channel=='rgbd')], mapped colors.
    """
    if vmin or vmax:
        normalizer = colors.Normalize(vmin, vmax)
    else:
        normalizer = None
    assert out_channel in ["rgb", "rgba"]

    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    x_ = x.detach().cpu().numpy()

    y_ = mapper.to_rgba(x_)[..., :len(out_channel)].astype(x_.dtype)
    y = torch.tensor(y_, device=x.device)


    return y


def batch_colorize_scalar_tensors(x, vmin=None, vmax=None, cmap='viridis', out_channel='rgb'):
    """Colorize scalar field tensors.

    Args:
        x: torch tensor of shape [N, H, W].
        vmin: float, or array of length N. min value to normalize the colors to.
        vmax: float, or array of length N. max value to normalize the colors to.
        cmap: str or Colormap instance, the colormap used to map normalized data values to RGBA
        colors.
        out_channel: str, 'rgb' or 'rgba'.
    Returns:
        y: torch tensor of shape [N, H, W, 3(or 4 if out_channel=='rgbd')]
    """
    def broadcast_limits(v):
        if v:
            if not isinstance(v, np.array):
                v = np.array(v)
            v = np.broadcast_to(v, x.shape[0])
        return v
    vmin = broadcast_limits(vmin)
    vmax = broadcast_limits(vmax)
    y = torch.zeros(list(x.shape)+[len(out_channel)], device=x.device)
    for idx in range(x.shape[0]):
        y[idx] = colorize_scalar_tensors(x[idx])

    return y