import argparse
from collections import defaultdict
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import shutil
import os
# import our modules
import sys
sys.path.append("../../src")
from unet3d import UNet3d
from implicit_net import ImNet
from pde import PDELayer
from nonlinearities import NONLINEARITIES
from local_implicit_grid import query_local_implicit_grid
import dataloader_spacetime as loader
from physics import get_rb2_pde_layer
from torch_flow_stats import *

def evaluate_feat_grid(pde_layer, latent_grid, t_seq, z_seq, x_seq, mins, maxs, pseudo_batch_size):
    """Evaluate latent feature grid at fixed intervals.

    Args:
        pde_layer: PDELayer instance where fwd_fn has been defined.
        latent_grid: latent feature grid of shape [batch, T, Z, X, C]
        t_seq: flat torch array of t-coordinates to evaluate
        z_seq: flat torch array of z-coordinates to evaluate
        x_seq: flat torch array of x-coordinates to evaluate
        mins: flat torch array of len 3 for min coords of t, z, x
        maxs: flat torch array of len 3 for max coords of t, z, x
        pseudo_batch_size, int, size of pseudo batch during eval
    Returns:
        res_dict: result dict.
    """
    device = latent_grid.device
    nb = latent_grid.shape[0]
    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))

    query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]

    nt, nz, nx, _ = query_coord.shape
    query_coord = query_coord.reshape([-1, 3]).to(device)
    n_query  = query_coord.shape[0]

    res_dict = defaultdict(list)

    n_iters = int(np.ceil(n_query/pseudo_batch_size))

    for idx in tqdm(range(n_iters)):
        sid = idx * pseudo_batch_size
        eid = min(sid+pseudo_batch_size, n_query)
        query_coord_batch = query_coord[sid:eid]
        query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

        pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        pred_value = pred_value.detach().cpu().numpy()
        for key in residue_dict.keys():
            residue_dict[key] = residue_dict[key].detach().cpu().numpy()
        for name, chan_id in zip(phys_channels, range(4)):
            res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
        for name, val in residue_dict.items():
            res_dict[name].append(val[..., 0])   # [b, pb]

    for key in res_dict.keys():
        res_dict[key] = (np.concatenate(res_dict[key], axis=1)
                         .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))[0]
    return res_dict


def frames_to_video(frames_pattern, save_video_to, frame_rate=10, keep_frames=False):
    """Create video from frames.

    frames_pattern: str, glob pattern of frames.
    save_video_to: str, path to save video to.
    keep_frames: bool, whether to keep frames after generating video.
    """
    cmd = ("ffmpeg -framerate {frame_rate} -pattern_type glob -i '{frames_pattern}' "
           "-c:v libx264 -r 30 -pix_fmt yuv420p {save_video_to}"
           .format(frame_rate=frame_rate, frames_pattern=frames_pattern,
                   save_video_to=save_video_to))
    os.system(cmd)
    # print
    print("Saving videos to {}".format(save_video_to))
    # delete frames if keep_frames is not needed
    if not keep_frames:
        frames_dir = os.path.dirname(frames_pattern)
        shutil.rmtree(frames_dir)


def calculate_flow_stats(pred, hres, visc=0.0001):
    data = pred
    uw = np.transpose(data[2:4,:,:,1:1+128], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file = open("REPORT___FlowStats_Pred_vs_GroundTruth.txt", "w")

    file.write("***** Pred Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n\n\n\n\n".format(s[8]))

    data = hres
    uw = np.transpose(data[2:4,:,:,1:1+128], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file.write("***** Ground Truth Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n".format(s[8]))


def export_video(args, res_dict, hres, lres, dataset):
    """Export inference result as a video.
    """
    phys_channels = ["p", "b", "u", "w"]
    if dataset:
        # hres = dataset.denormalize_grid(hres.copy())
        lres = dataset.denormalize_grid(lres.copy())
        pred = np.stack([res_dict[key] for key in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)
        calculate_flow_stats(pred, hres)       # Warning: only works with pytorch > v1.3 and CUDA >= v10.1
        # np.savez_compressed(args.save_path+'highres_lowres_pred', hres=lres, lres=lres, pred=pred)

    os.makedirs(args.save_path, exist_ok=True)
    # enumerate through physical channels first

    for idx, name in enumerate(phys_channels):
        frames_dir = os.path.join(args.save_path, f'frames_{name}')
        os.makedirs(frames_dir, exist_ok=True)
        hres_frames = hres[idx]
        lres_frames = lres[idx]
        pred_frames = pred[idx]

        # loop over each timestep in pred_frames
        max_val = np.max(hres_frames)
        min_val = np.min(hres_frames)

        for pid in range(pred_frames.shape[0]):
            hid = int(np.round(pid / (pred_frames.shape[0] - 1) * (hres_frames.shape[0] - 1)))
            lid = int(np.round(pid / (pred_frames.shape[0] - 1) * (lres_frames.shape[0] - 1)))

            fig, axes = plt.subplots(3, figsize=(10, 10))#, 1, sharex=True)
            # high res ground truth
            im0 = axes[0].imshow(hres_frames[hid], cmap='RdBu',interpolation='spline16')
            axes[0].set_title(f'{name} channel, high res ground truth.')
            im0.set_clim(min_val, max_val)
            # low res input
            im1 = axes[1].imshow(lres_frames[lid], cmap='RdBu',interpolation='none')
            axes[1].set_title(f'{name} channel, low  res ground truth.')
            im1.set_clim(min_val, max_val)
            # prediction
            im2 = axes[2].imshow(pred_frames[pid], cmap='RdBu',interpolation='spline16')
            axes[2].set_title(f'{name} channel, predicted values.')
            im2.set_clim(min_val, max_val)
            # add shared colorbar
            cbaxes = fig.add_axes([0.1, 0, .82, 0.05])
            fig.colorbar(im2, orientation="horizontal", pad=0, cax=cbaxes)
            frame_name = 'frame_{:03d}.png'.format(pid)
            fig.savefig(os.path.join(frames_dir, frame_name))

        # stitch frames into video (using ffmpeg)
        frames_to_video(
            frames_pattern=os.path.join(frames_dir, "*.png"),
            save_video_to=os.path.join(args.save_path, f"video_{name}.mp4"),
            frame_rate=args.frame_rate, keep_frames=args.keep_frames)



def model_inference(args, lres, pde_layer):
    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct model
    print(f"Loading model parameters from {args.ckpt}...")
    igres = (int(args.nt/args.downsamp_t),
             int(args.nz/args.downsamp_xz),
             int(args.nx/args.downsamp_xz),)
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=igres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])

    # load model params
    resume_dict = torch.load(args.ckpt)
    unet.load_state_dict(resume_dict["unet_state_dict"])
    imnet.load_state_dict(resume_dict["imnet_state_dict"])

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    # evaluate
    latent_grid = unet(torch.tensor(lres, dtype=torch.float32)[None].to(device))
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

    # create evaluation grid
    t_max = float(192/args.nt)
    z_max = 1
    x_max = 4

    # layout query points for the desired slices
    eps = 1e-6
    t_seq = torch.linspace(eps, t_max-eps, args.eval_tres)  # temporal sequences
    z_seq = torch.linspace(eps, z_max-eps, args.eval_zres)  # z sequences
    x_seq = torch.linspace(eps, x_max-eps, args.eval_xres)  # x sequences

    mins = torch.zeros(3, dtype=torch.float32, device=device)
    maxs = torch.tensor([t_max, z_max, x_max], dtype=torch.float32, device=device)

    # define lambda function for pde_layer
    fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, mins, maxs)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(fwd_fn)

    res_dict = evaluate_feat_grid(pde_layer, latent_grid, t_seq, z_seq, x_seq, mins, maxs,
                                  args.eval_pseudo_batch_size)

    return res_dict



def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--eval_xres", type=int, default=512, metavar="X",
                        help="x resolution during evaluation (default: 512)")
    parser.add_argument("--eval_zres", type=int, default=128, metavar="Z",
                        help="z resolution during evaluation (default: 128)")
    parser.add_argument("--eval_tres", type=int, default=192, metavar="T",
                        help="t resolution during evaluation (default: 192)")
    parser.add_argument('--ckpt', type=str, required=True, help="path to checkpoint")
    parser.add_argument("--save_path", type=str, default='eval')
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--lres_interp", type=str, default='linear',
                        help="str, interpolation scheme for generating low res. choices of 'linear', 'nearest'")
    parser.add_argument("--lres_filter", type=str, default='none',
                        help=" str, filter to apply on original high-res image before \
                        interpolation. choices of 'none', 'gaussian', 'uniform', 'median', 'maximum'")
    parser.add_argument("--frame_rate", type=int, default=10, metavar="N",
                        help="frame rate for output video (default: 10)")
    parser.add_argument("--keep_frames", dest='keep_frames', action='store_true')
    parser.add_argument("--no_keep_frames", dest='keep_frames', action='store_false')
    parser.add_argument("--eval_pseudo_batch_size", type=int, default=10000,
                        help="psudo batch size for querying the grid. set to a smaller"
                             " value if OOM error occurs")
    parser.add_argument('--rayleigh', type=float, required=True,
                        help='Simulation Rayleigh number.')
    parser.add_argument('--prandtl', type=float, required=True,
                        help='Simulation Prandtl number.')
    parser.set_defaults(keep_frames=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    param_file = os.path.join(os.path.dirname(args.ckpt), "params.json")
    with open(param_file, 'r') as fh:
        args.__dict__.update(json.load(fh))

    print(args)
    # prepare dataset
    dataset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.eval_dataset,
        nx=512, nz=128, nt=192, n_samp_pts_per_crop=1,
        lres_interp=args.lres_interp, lres_filter=args.lres_filter, downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=True)

    # extract data
    hres, lres, _, _ = dataset[0]

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = dataset.channel_mean
        std = dataset.channel_std
    else:
        mean = std = None
    pde_layer = get_rb2_pde_layer(mean=mean, std=std, prandtl=args.prandtl, rayleigh=args.rayleigh)
    # pde_layer = get_rb2_pde_layer(mean=mean, std=std)

    # evaluate model for getting high res spatial temporal sequence
    res_dict = model_inference(args, lres, pde_layer)

    # save video
    export_video(args, res_dict, hres, lres, dataset)

if __name__ == '__main__':
    main()
