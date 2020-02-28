"""Training script for RB2 experiment.
"""
import argparse
import json
import os
from glob import glob
import numpy as np
from collections import defaultdict
import time
import datetime
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
from implicit_net import ImNet
from local_implicit_grid import query_local_implicit_grid
import dataloader_spacetime as loader
from physics import get_rb2_pde_layer

# FOR DISTRIBUTED:
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as TDDP
try:
    from apex.parallel import DistributedDataParallel as ADDP
    from apex import amp
    HASAPEX = True
except Exception as e:
    import_error = e
    HASAPEX = False
    

def setup(rank, world_size, offset=0):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355+offset)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

# pylint: disable=no-member



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
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        pde_loss = loss_func(pde_tensors, torch.zeros_like(pde_tensors))
        loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        if args.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(unet.module.parameters(), args.clip_grad)
        torch.nn.utils.clip_grad_value_(imnet.module.parameters(), args.clip_grad)

        optimizer.step()

        tot_loss += loss.item()
        count += input_grid.size()[0]
        if batch_idx % args.log_interval == 0:
            if args.rank == 0:
                # logger log
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                    "Loss Reg: {:.6f}\tLoss Pde: {:.6f}".format(
                        epoch, batch_idx * len(input_grid) * args.nprocs, 
                        len(train_loader) * len(input_grid) * args.nprocs,
                        100. * batch_idx / len(train_loader), loss.item(),
                        args.alpha_reg * reg_loss, args.alpha_pde * pde_loss))
                # tensorboard log
                writer.add_scalar('train/reg_loss_unweighted', 
                                  reg_loss, global_step=int(global_step))
                writer.add_scalar('train/pde_loss_unweighted', 
                                  pde_loss, global_step=int(global_step))
                writer.add_scalar('train/sum_loss', loss, global_step=int(global_step))
                writer.add_scalars('train/losses_weighted',
                                   {"reg_loss": args.alpha_reg * reg_loss,
                                    "pde_loss": args.alpha_pde * pde_loss,
                                    "sum_loss": loss}, global_step=int(global_step))

        global_step += 1
    tot_loss /= count
    return tot_loss


def eval(args, unet, imnet, eval_loader, epoch, global_step, device,
         logger, writer, optimizer, pde_layer):
    """Eval function. Used for evaluating entire slices and comparing to GT."""
    unet.eval()
    imnet.eval()
    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    for data_tensors in eval_loader:
        # only need the first batch
        break
    # send tensors to device
    data_tensors = [t.to(device) for t in data_tensors]
    hres_grid, lres_grid, _, _ = data_tensors
    latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
    nb, nc, nt, nz, nx = hres_grid.shape

    # permute such that C is the last channel for local implicit grid query
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

    # define lambda function for pde_layer
    fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(fwd_fn)

    # layout query points for the desired slices
    eps = 1e-6
    t_seq = torch.linspace(eps, 1-eps, nt)[::int(nt/8)]  # temporal sequences
    z_seq = torch.linspace(eps, 1-eps, nz)  # z sequences
    x_seq = torch.linspace(eps, 1-eps, nx)  # x sequences

    query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
    query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
    n_query = query_coord.shape[0]

    res_dict = defaultdict(list)

    n_iters = int(np.ceil(n_query/args.pseudo_batch_size))

    for idx in range(n_iters):
        sid = idx * args.pseudo_batch_size
        eid = min(sid+args.pseudo_batch_size, n_query)
        query_coord_batch = query_coord[sid:eid]
        query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

        pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        pred_value = pred_value.detach()
        for key in residue_dict.keys():
            residue_dict[key] = residue_dict[key].detach()
        for name, chan_id in zip(phys_channels, range(4)):
            res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
        for name, val in residue_dict.items():
            res_dict[name].append(val[..., 0])   # [b, pb]

    for key in res_dict.keys():
        res_dict[key] = (torch.cat(res_dict[key], axis=1)
                         .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))

    # log the imgs sample-by-sample
    if args.rank == 0:
        for samp_id in range(nb):
            for key in res_dict.keys():
                field = res_dict[key][samp_id]  # [nt, nz, nx]
                # add predicted slices
                images = utils.batch_colorize_scalar_tensors(field)  # [nt, nz, nx, 3]

                writer.add_images('sample_{}/{}/predicted'.format(samp_id, key), images,
                    dataformats='NHWC', global_step=int(global_step))
                # add ground truth slices (only for phys channels)
                if key in phys_channels:
                    gt_fields = hres_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
                    gt_images = utils.batch_colorize_scalar_tensors(gt_fields)  # [nt, nz, nx, 3]

                    writer.add_images('sample_{}/{}/ground_truth'.format(samp_id, key), gt_images,
                        dataformats='NHWC', global_step=int(global_step))


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--nprocs", type=int, default=2, metavar="N",
                        help="number of parallel gpu processes (default: 2)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=3360, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 3360)")
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
    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per crop.")
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
    parser.add_argument("--alpha_reg", default=1., type=float, help="weight of regression loss.")
    parser.add_argument("--alpha_pde", default=1., type=float, help="weight of pde residue loss.")
    parser.add_argument("--num_log_images", default=2, type=int, help="number of images to log.")
    parser.add_argument("--pseudo_batch_size", default=1024, type=int,
                        help="size of pseudo batch during eval.")
    parser.add_argument("--normalize_channels", dest='normalize_channels', action='store_true')
    parser.add_argument("--no_normalize_channels", dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--use_apex", dest='use_apex', action='store_true')
    parser.add_argument("--no_use_apex", dest='use_apex', action='store_false')
    parser.set_defaults(use_apex=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")
    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))
    parser.add_argument("--apex_optim_level", default="O0", type=str, 
                        help=("Apex Optimization Level. O0: FP32 training, "
                              "O1: Conservative Mixed Precision, O2: Fast Mixed Precision, O3: FP16 training."))
    parser.add_argument("--output_timing", default="", type=str, 
                        help="output time for this run to file. Useful for running scaling experiments.")
    parser.add_argument("--comm_backend", default="gloo", type=str,
                        help="communication backend for distributed computing. choices: 'gloo', 'mpi', 'nccl'.")

    args = parser.parse_args()
    return args


def main_ddp(rank, world_size, args):
    offset = int(args.apex_optim_level[1])*world_size+rank
    setup(rank, world_size, offset=offset)

    args.rank = rank
    if args.use_apex and (not HASAPEX):
        if rank == 0:
            print(import_error)
            warnings.warn(
                "Failed to import Apex. Falling back to PyTorch DistributedDataParallel.",
                ImportError)
        args.use_apex = False
    DDP = ADDP if args.use_apex else TDDP
    
#     n_per_rank = torch.cuda.device_count() // world_size
#     device_ids = list(range(rank * n_per_rank, (rank + 1) * n_per_rank))
    device_ids = [args.rank]
    torch.cuda.set_device(args.rank)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    device = torch.device(device_ids[0])
    # no need to adjust batch size. batch size = batch_size_per_gpu
    args.batch_size = args.batch_size_per_gpu

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    if args.rank == 0: logger.info("%s", repr(args))
        
    logger.info(f"[Rank] {rank:2d} [Cuda IDs] {device_ids}")

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed((args.seed+1) * rank)
    np.random.seed((args.seed+1) * rank)

    # create dataloaders
    trainset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename="rb2d_ra1e6_s42.npz",
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=False,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )
    evalset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename="rb2d_ra1e6_s42.npz",
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=True,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )

    nsamp_per_proc = args.pseudo_epoch_size // args.nprocs
    train_sampler = RandomSampler(trainset, replacement=True, num_samples=nsamp_per_proc)
    eval_sampler = RandomSampler(evalset, replacement=True, num_samples=args.num_log_images)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, **kwargs)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             sampler=eval_sampler, **kwargs)

    # setup model
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=trainset.scale_lres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf)
    
    if args.resume:        
        # configure map_location properly
        rank0_devices = [x - rank * len(device_ids) for x in device_ids]
        device_pairs = zip(rank0_devices, device_ids)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}

        resume_dict = torch.load(args.resume, map_location=map_location)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        unet.load_state_dict(resume_dict["unet_state_dict"])
        imnet.load_state_dict(resume_dict["imnet_state_dict"])
    
    unet.to(device)
    imnet.to(device)
    
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    if args.optim == "sgd":
        optimizer = optim.SGD(all_model_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_model_params, lr=args.lr)
    
    if args.use_apex:
        (unet, imnet), optimizer = amp.initialize([unet, imnet], optimizer, opt_level=args.apex_optim_level)
    
    if args.use_apex:
        unet = DDP(unet)
        imnet = DDP(imnet)
    else:
        unet = DDP(unet, device_ids=device_ids)
        imnet = DDP(imnet, device_ids=device_ids)
        
    if args.resume:
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf


    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    if args.rank == 0: 
        logger.info("{}(unet) + {}(imnet) paramerters in total".format(
            model_param_count(unet), model_param_count(imnet)))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = trainset.channel_mean
        std = trainset.channel_std
    else:
        mean = std = None
    pde_layer = get_rb2_pde_layer(mean=mean, std=std,
        t_crop=args.nt*0.125, z_crop=args.nz*(1./128), x_crop=args.nx*(1./128))

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        t0 = time.time()
        loss = train(args, unet, imnet, train_loader, epoch, global_step, device, logger, writer,
                     optimizer, pde_layer)
        t1 = time.time()
        eval(args, unet, imnet, eval_loader, epoch, global_step, device, logger, writer, 
             optimizer, pde_layer)
        t2 = time.time()
        if args.lr_scheduler:
            scheduler.step(loss)
        if loss < tracked_stats:
            tracked_stats = loss
            is_best = True
        else:
            is_best = False
        if args.rank == 0:
            utils.save_checkpoint({
                "epoch": epoch,
                "unet_state_dict": unet.module.state_dict(),
                "imnet_state_dict": imnet.module.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "tracked_stats": tracked_stats,
                "global_step": global_step,
            }, is_best, epoch, checkpoint_path, "_pdenet", logger)
        t3 = time.time()
        if args.rank == 0:
            logger.info(f"Total time per epoch: {datetime.timedelta(seconds=t3-t0)} ({t3-t0:.2f} secs)")
            logger.info(f"Train time per epoch: {datetime.timedelta(seconds=t1-t0)} ({t1-t0:.2f} secs)")
            logger.info(f"Eval  time per epoch: {datetime.timedelta(seconds=t2-t1)} ({t2-t1:.2f} secs)")
            if epoch == 1 and args.output_timing:
                if not os.path.exists(args.output_timing):
                    newfile = True
                else:
                    newfile = False
                with open(args.output_timing, "a") as fh:
                    if newfile:
                        fh.write("num_gpu,opt_level,total_time_per_epoch,train_time_per_epoch,eval_time_per_epoch\n")
                    fh.write(("{num_gpu},{opt_level},{tot_time},{train_time},{eval_time}\n"
                              .format(num_gpu=args.nprocs, opt_level=args.apex_optim_level, 
                                      tot_time=t3-t0, train_time=t1-t0, eval_time=t2-t1)))
        
    cleanup()
    
def main():
    args = get_args()

    ndevices = torch.cuda.device_count()
    if args.nprocs > ndevices:
        raise RuntimeError("Number of processes ({}) is greater than available GPU devices ({}). "
                           "Rerun with --nprocs=N where N <= {}.".format(args.nprocs, ndevices, ndevices))    

    mp.spawn(main_ddp,
            args=(args.nprocs, args),
            nprocs=args.nprocs,
            join=True)

if __name__ == "__main__":
    main()
