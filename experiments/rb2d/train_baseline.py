"""Training script UNet baseline for RB2 experiment.
"""
import argparse
import json
import os
from glob import glob
import numpy as np
from collections import defaultdict
np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# import our modules
import sys
sys.path.append("../../src")
import train_utils as utils
from unet3d import UNet3d
from local_implicit_grid import query_local_implicit_grid
from physics import get_rb2_pde_layer
import dataloader_spacetime as loader
# pylint: disable=no-member


def loss_functional(loss_type):
    """Get loss function given function type names."""
    if loss_type == 'l1':
        return F.l1_loss
    if loss_type == 'l2':
        return F.mse_loss
    # else (loss_type == 'huber')
    return F.smooth_l1_loss


def train(args, model, train_loader, epoch, global_step, device, logger, writer, optimizer):
    """Training function."""
    model.train()
    tot_loss = 0
    count = 0
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(train_loader):
        # send tensors to device
        data_tensors = [t.to(device) for t in data_tensors]
        hres_grid, input_grid, _, _ = data_tensors
        optimizer.zero_grad()
        pred_grid = model(input_grid)  # [batch, C, T, X, Y]

        # function value regression loss
        loss = loss_func(pred_grid, hres_grid)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(model.module.parameters(), args.clip_grad)

        optimizer.step()

        tot_loss += loss.item()
        count += input_grid.size()[0]
        if batch_idx % args.log_interval == 0:
            # logger log
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t".format(
                    epoch, batch_idx * len(input_grid), len(train_loader) * len(input_grid),
                    100. * batch_idx / len(train_loader), loss.item()))
            # tensorboard log
            writer.add_scalar('train/loss', loss, global_step=int(global_step))

        global_step += 1
    tot_loss /= count
    return tot_loss


def eval(args, model, eval_loader, epoch, global_step, device, logger, writer, optimizer):
    """Eval function. Used for evaluating entire slices and comparing to GT."""
    model.eval()
    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    for data_tensors in eval_loader:
        # only need the first batch
        break
    # send tensors to device
    data_tensors = [t.to(device) for t in data_tensors]
    hres_grid, lres_grid, _, _ = data_tensors
    pred_grid = model(lres_grid)  # [batch, C, T, Z, X]
    nb, nc, nt, nz, nx = hres_grid.shape

    # log the imgs sample-by-sample
    for samp_id in range(nb):
        for key in phys_channels:
            field = pred_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
            # add predicted slices
            images = utils.batch_colorize_scalar_tensors(field)  # [nt, nz, nx, 3]

            writer.add_images('sample_{}/{}/predicted'.format(samp_id, key), images,
                dataformats='NHWC', global_step=int(global_step))
            # add ground truth slices (only for phys channels)
            gt_fields = hres_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
            gt_images = utils.batch_colorize_scalar_tensors(gt_fields)  # [nt, nz, nx, 3]

            writer.add_images('sample_{}/{}/ground_truth'.format(samp_id, key), gt_images,
                              dataformats='NHWC', global_step=int(global_step))


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=3000, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="R",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_folder", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--train_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--eval_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log_dir", type=str, required=True, help="log directory for run")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("--nt", default=16, type=int, help="resolution of high res crop in t.")
    parser.add_argument("--nx", default=128, type=int, help="resolution of high res crop in x.")
    parser.add_argument("--nz", default=128, type=int, help="resolution of high res crop in y.")
    parser.add_argument("--downsamp_t", default=4, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--downsamp_xz", default=8, type=int,
                        help="down sampling factor in x and y for low resolution crop.")
    parser.add_argument("--dt", default=0.25, type=float,
                        help="time difference between consecutive timesteps in the simulation.")
    parser.add_argument("--dz", default=1./128, type=float,
                        help="physical width in z per 1 pixel.")
    parser.add_argument("--dx", default=1./128, type=float,
                        help="physical width in x per 1 pixel.")
    parser.add_argument("--lat_dims", default=32, type=int, help="number of latent dimensions.")
    parser.add_argument("--unet_nf", default=16, type=int,
                        help="number of base number of feature layers in unet.")
    parser.add_argument("--unet_mf", default=256, type=int,
                        help="a cap for max number of feature layers throughout the unet.")
    parser.add_argument("--imnet_nf", default=32, type=int,
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--reg_loss_type", default="l1", type=str,
                        choices=["l1", "l2", "huber"],
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--num_log_images", default=2, type=int, help="number of images to log.")
    parser.add_argument("--normalize_channels", dest='normalize_channels', action='store_true')
    parser.add_argument("--no_normalize_channels", dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")
    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    # adjust batch size based on the number of gpus available
    args.batch_size = int(torch.cuda.device_count()) * args.batch_size_per_gpu

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.train_data,
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=1,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, normalize_hres=args.normalize_channels,
        return_hres=True, lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )
    evalset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.eval_data,
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=1,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, normalize_hres=args.normalize_channels,
        return_hres=True, lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )

    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(evalset, replacement=True, num_samples=args.num_log_images)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, **kwargs)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             sampler=eval_sampler, **kwargs)

    # setup model
    model = UNet3d(in_features=4, out_features=4,
                   igres=trainset.scale_lres, ogres=trainset.scale_hres,
                   nf=args.unet_nf, mf=args.unet_mf)

    if args.optim == "sgd":
        optimizer = optim.SGD(list(model.parameters()), lr=args.lr)
    else:
        optimizer = optim.Adam(list(model.parameters()), lr=args.lr)

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        model.load_state_dict(resume_dict["model_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model = nn.DataParallel(model)
    model.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info("{}(unet) paramerters in total".format(model_param_count(model)))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        loss = train(args, model, train_loader, epoch, global_step, device, logger, writer,
                     optimizer)
        eval(args, model, eval_loader, epoch, global_step, device, logger, writer, optimizer)
        if args.lr_scheduler:
            scheduler.step(loss)
        if loss < tracked_stats:
            tracked_stats = loss
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
        }, is_best, epoch, checkpoint_path, "_"+"unet", logger)

if __name__ == "__main__":
    main()
