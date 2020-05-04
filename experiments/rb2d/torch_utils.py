import torch
import torch.nn as nn
from torch_flow_stats import compute_all_stats
import logging, os
import torchvision.utils as vutils
import shutil
from scipy.stats import ks_2samp



def ks_stats(dist1, dist2):
    """
    compute ks statistic based on two samples
    :param dist1: shape (n1, dim)
    :param dist2: shape (n2, dim)
    """
    assert(dist1.shape[1] == dist2.shape[1])
    dim = dist1.shape[1]
    dist1_ = dist1.detach().cpu().numpy()
    dist2_ = dist2.detach().cpu().numpy()
    return torch.tensor([ks_2samp(dist1_[:, i], dist2_[:, i])[0] for i in range(dim)]).to(dist1.device)

def initialize_logger(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(logdir, "log.txt"))
    logger.addHandler(fh)
    return logger

def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    new_file = os.path.join(output_folder, filename + '_%03d' % (epoch) + '.pth.tar')
    torch.save(state, new_file)
    if epoch > 0:
        prev_file = os.path.join(output_folder, filename + '_%03d' % (epoch-1) + '.pth.tar')
        if os.path.exists(prev_file) and (epoch-1)%10 != 0:
            os.remove(prev_file)
    if is_best:
        logger.info("Saving new best model")
        best_file = os.path.join(output_folder, filename + '_best.pth.tar')
        shutil.copyfile(new_file, best_file)

def dcgan_tblogging_scalar(sample_dist, ksstats, datastatsmean, res, errG, errD, step, writer):
    # names for variables in sample_dist
    names = ["tkenergy", "dissipation", "rmsvelocity", "tmscale", "tsreynolds", "ktimescale", "klenscale", "intscale", "eddytime"]

    # scalars for sample dist
    s_mean = torch.mean(sample_dist, dim=0)
    for i, name in enumerate(names):
        writer.add_scalar("meanval/{}({})".format(name, datastatsmean[i].item()), s_mean[i].item(), step)
        writer.add_scalar("ksstats/{}".format(name), ksstats[i].item(), step)

    # simple scalar plot for other training stats
    writer.add_scalar("train/residue", res, step)
    writer.add_scalar("train/errD", errD, step)
    writer.add_scalar("train/errG", errG, step)

def dcgan_tblogging_image(real, fake, step, writer):
    r = vutils.make_grid(real[..., 64], nrow=4, normalize=True, scale_each=True)
    f = vutils.make_grid(fake[..., 64], nrow=4, normalize=True, scale_each=True)
    writer.add_image('real', r, step)
    writer.add_image('fake', f, step)