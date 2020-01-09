"""Training script for RB2 experiment.
"""
import argparse
import os
from glob import glob
import numpy as np
np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# import my modules
import sys
sys.path.append("../../src")
import train_utils as utils
from unet3d import UNet3d
from implicit_net import ImNet
from pde import PDELayer
from local_implicit_grid import query_local_implicit_grid
import dataloader_spacetime as loader

# pylint: disable=no-member


def get_rb2_pde_layer(prandtl=1., rayleigh=1e6):
    """Get PDE layer corresponding to the RB2 govening equations."""
    # constants
    P = (rayleigh * prandtl)**(-1/2)
    R = (rayleigh / prandtl)**(-1/2)
    # set up variables and equations
    in_vars = 't, x, z'
    out_vars = 'p, b, u, w'
    eqn_strs = [
        f'dif(b,t)-{P}*(dif(dif(b,x),x)+dif(dif(b,z),z))             +(u*dif(b,x)+w*dif(b,z))',
        f'dif(u,t)-{R}*(dif(dif(u,x),x)+dif(dif(u,z),z))+dif(p,x)    +(u*dif(u,x)+w*dif(u,z))',
        f'dif(w,t)-{R}*(dif(dif(w,x),x)+dif(dif(w,z),z))+dif(p,z)-b  +(u*dif(w,x)+w*dif(w,z))',
    ]
    # a name/identifier for the equations
    eqn_names = ['transport_eqn_b', 'transport_eqn_u', 'transport_eqn_w']

    # initialize the pde layer
    pde_layer = PDELayer(in_vars=in_vars, out_vars=out_vars)

    for eqn_str, eqn_name in zip(eqn_strs, eqn_names):  # add equations
        pde_layer.add_equation(eqn_str, eqn_name)

    return pde_layer  # NOTE: forward method has not yet been updated.


def loss_functional(loss_type):
    """Get loss function given function type names."""
    if loss_type == 'l1':
        return F.l1_loss
    if loss_type == 'l2':
        return F.mse_loss
    # else (loss_type == 'huber')
    return F.smooth_l1_loss


def train(args, unet, imnet, train_loader, epoch, global_step, device,
          logger, writer, optimizer, pde_layer):
    """Training function."""
    unet.train()
    imnet.train()
    tot_loss = 0
    count = 0
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(train_loader):
        # send tensors to device
        data_tensors = [t.to(device) for t in data_tensors]
        input_grid, point_coord, point_value = data_tensors
        optimizer.zero_grad()
        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True)

        # function value regression loss
        reg_loss = loss_func(pred_value, point_value)

        # pde residue loss
        pde_loss = torch.sum(torch.stack([d for d in residue_dict.values()], dim=0))
        loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        count += input_grid.size()[0]
        if batch_idx % args.log_interval == 0:
            # logger log
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                "Loss Reg: {:.6f}\tLoss Pde: {:.6f}".format(
                    epoch, batch_idx * len(input_grid), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    args.alpha_reg * reg_loss, args.alpha_pde * pde_loss))
            # tensorboard log
            writer.add_scalar('train/reg_loss_unweighted', reg_loss, global_step=int(global_step))
            writer.add_scalar('train/pde_loss_unweighted', pde_loss, global_step=int(global_step))
            writer.add_scalar('train/sum_loss', loss, global_step=int(global_step))
            writer.add_scalars('train/losses_weighted',
                               {"reg_loss": args.alpha_reg * reg_loss,
                                "pde_loss": args.alpha_pde * pde_loss,
                                "sum_loss": loss}, global_step=int(global_step))

        global_step += 1
    tot_loss /= count
    return tot_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--batch_size", type=int, default=32, metavar="N",
                        help="input batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=2048, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 2048)")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="R",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_folder", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log_dir", type=str, default="log",
                        help="log directory for run")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("--train_stats_freq", default=0, type=int,
                        help="frequency for printing training set stats. 0 for never.")
    parser.add_argument("--nt", default=16, type=int, help="resolution of high res crop in t.")
    parser.add_argument("--nx", default=128, type=int, help="resolution of high res crop in x.")
    parser.add_argument("--ny", default=128, type=int, help="resolution of high res crop in y.")
    parser.add_argument("--downsamp_t", default=4, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--downsamp_xy", default=4, type=int,
                        help="down sampling factor in x and y for low resolution crop.")
    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per crop.")
    parser.add_argument("--lat_dims", default=32, type=int, help="number of latent dimensions.")
    parser.add_argument("--unet_nf", default=32, type=int,
                        help="number of base number of feature layers in unet.")
    parser.add_argument("--unet_mf", default=512, type=int,
                        help="a cap for max number of feature layers throughout the unet.")
    parser.add_argument("--imnet_nf", default=32, type=int,
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--reg_loss_type", default="huber", type=str,
                        choices=["l1", "l2", "huber"],
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--alpha_reg", default=1., type=float, help="weight of regression loss.")
    parser.add_argument("--alpha_pde", default=1., type=float, help="weight of pde residue loss.")
    parser.add_argument("--num_log_images", default=8, type=int, help="number of images to log.")


    args = parser.parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename="rb2d_ra1e6_s42.npz",
        nx=args.nx, ny=args.ny, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        interp_method='linear', downsamp_xy=args.downsamp_xy, downsamp_t=args.downsamp_t,
        normalize_output=False, return_hres=False)
    evalset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename="rb2d_ra1e6_s42.npz",
        nx=args.nx, ny=args.ny, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        interp_method='linear', downsamp_xy=args.downsamp_xy, downsamp_t=args.downsamp_t,
        normalize_output=False, return_hres=True)

    train_sampler = RandomSampler(trainset, replacement=False, num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(evalset, replacement=False, num_samples=args.num_log_images)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             sampler=eval_sampler)

    # setup model
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=trainset.scale_lres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf)
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    if args.optim == "sgd":
        optimizer = optim.SGD(all_model_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_model_params, lr=args.lr)

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        unet.load(resume_dict["unet_state_dict"])
        imnet.load(resume_dict["imnet_state_dict"])
        optimizer.load(resume_dict["optim_state_dict"])

    unet = nn.DataParallel(unet)
    unet.to(device)
    imnet = nn.DataParallel(imnet)
    imnet.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info("{}(unet) + {}(imnet) paramerters in total".format(model_param_count(unet),
                                                                   model_param_count(imnet)))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    # get pdelayer for the RB2 equations
    pde_layer = get_rb2_pde_layer()

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        loss = train(args, unet, imnet, train_loader, epoch, global_step, device, logger, writer, optimizer, pde_layer)
        if loss < tracked_stats:
            tracked_stats = loss
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint({
            "epoch": epoch,
            "unet_state_dict": unet.modules.state_dict(),
            "imnet_state_dict": imnet.modules.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
        }, is_best, epoch, checkpoint_path, "_pdenet", logger)

if __name__ == "__main__":
    main()
