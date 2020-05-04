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
import dataloader_spacetime as loader
from torch_flow_stats import *


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
        if args.baseline_no == 2:
            pred = np.stack([res_dict[key] for key in phys_channels], axis=0)
            pred = dataset.denormalize_grid(pred)
        else:
            pred = res_dict
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



def model_inference(args, lres):
    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct model
    print(f"Loading model parameters from {args.ckpt}...")
    igres = (int(args.nt/args.downsamp_t),
             int(args.nz/args.downsamp_xz),
             int(args.nx/args.downsamp_xz),)
    model = UNet3d(in_features=4, out_features=4,
                   igres=igres, ogres=(args.nt, args.nz, args.nx),
                   nf=args.unet_nf, mf=args.unet_mf)

    # load model params
    resume_dict = torch.load(args.ckpt)
    model.load_state_dict(resume_dict["model_state_dict"])

    model.to(device)
    model.eval()
    all_model_params = list(model.parameters())

    # evaluate
    input_tensor = torch.tensor(lres, dtype=torch.float32)[None].to(device)
    nt = input_tensor.shape[2]
    half_nt = int(nt/2)
    with torch.no_grad():
        # split in half and then merge due to memory constraint
        pred_grid_0 = model(input_tensor[:, :, :half_nt]).detach()  # [batch, C, T, Z, X]
        pred_grid_1 = model(input_tensor[:, :, half_nt:]).detach()  # [batch, C, T, Z, X]
        pred_grid = torch.cat([pred_grid_0, pred_grid_1], dim=2)
    pred_grid = pred_grid[0]

    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    res_dict = {}
    for key in phys_channels:
        res_dict[key] = pred_grid[phys2id[key]].detach().cpu().numpy()

    return res_dict



def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Evaluate baseline model performance on superresoluton.")
    parser.add_argument("--eval_xres", type=int, default=512, metavar="X",
                        help="x resolution during evaluation (default: 512)")
    parser.add_argument("--eval_zres", type=int, default=128, metavar="Z",
                        help="z resolution during evaluation (default: 128)")
    parser.add_argument("--eval_tres", type=int, default=192, metavar="T",
                        help="t resolution during evaluation (default: 192)")
    parser.add_argument('--ckpt', type=str, required=True, help="path to checkpoint")
    parser.add_argument('--baseline_no', type=int, required=True, help="baseline model number - choices (int): 1 or 2")
    parser.add_argument("--save_path", type=str, default='eval')
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--lres_interp", type=str, default='linear',
                        help="str, interpolation scheme for generating low res. choices of 'linear', 'nearest'")
    parser.add_argument("--lres_filter", type=str, default='gaussian',
                        help=" str, filter to apply on original high-res image before \
                        interpolation. choices of 'none', 'gaussian', 'uniform', 'median', 'maximum'")
    parser.add_argument("--frame_rate", type=int, default=10, metavar="N",
                        help="frame rate for output video (default: 10)")
    parser.add_argument("--keep_frames", dest='keep_frames', action='store_true')
    parser.add_argument("--no_keep_frames", dest='keep_frames', action='store_false')
    parser.set_defaults(keep_frames=False)

    args = parser.parse_args()
    return args

def get_highres_pred_modelFree_trilinear_interp(lres_data):
    
    stats_all_lres = []
    tl, zl, xl = np.linspace(0, 50, 48), np.linspace(0, 1, 16), np.linspace(0, 4, 64)
    th, zh, xh = np.linspace(0, 50, 192), np.linspace(0, 1, 128), np.linspace(0, 4, 512)
    highres_pred = []
    for d in range(4): #["p", "b", "u", "w"]
        grid_interp_func = RegularGridInterpolator((tl, zl, xl), lres_data[d, :, :, :])
        query_points = [[i, j, k] for k in xh for j in zh for i in th]
        highres_pred.append(np.reshape(grid_interp_func(query_points), [192, 128, 512]))
    highres_pred = np.stack(highres_pred)

    return highres_pred


    
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

    if args.baseline_no == 2:
        # evaluate model for getting high res spatial temporal sequence - Baseline II - Unet
        res_dict = model_inference(args, lres)
    else:
        # get high res spatial temporal sequence using model-free trilinear grid interpolation - Baseline I
        res_dict = get_highres_pred_modelFree_trilinear_interp(lres)

    # save video
    export_video(args, res_dict, hres, lres, dataset)

if __name__ == '__main__':
    main()
