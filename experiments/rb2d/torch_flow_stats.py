import torch
import numpy as np
import time
from torchsearchsorted import searchsorted
from torch_spec_operator import img, rfftfreqs, fftfreqs, spec_grad, pad_rfft3, pad_irfft3


##################### UTILITIES ###########################


def energy_spectrum(vel):
    """
    Compute energy spectrum given a velocity field
    :param vel: tensor of shape (N, 3, res, res, res)
    :return spec: tensor of shape(N, res/2)
    :return k: tensor of shape (res/2,), frequencies corresponding to spec
    """
    device = vel.device
    res = vel.shape[-2:]

    assert(res[0] == res[1])
    r = res[0]
    k_end = int(r/2)
    vel_ = pad_rfft3(vel, onesided=False) # (N, 3, res, res, res, 2)
    uu_ = (torch.norm(vel_, dim=-1) / r**3)**2
    e_ = torch.sum(uu_, dim=1)  # (N, res, res, res)
    k = fftfreqs(res).to(device) # (3, res, res, res)
    rad = torch.norm(k, dim=0) # (res, res, res)
    k_bin = torch.arange(k_end, device=device).float()+1
    bins = torch.zeros(k_end+1).to(device)
    bins[1:-1] = (k_bin[1:]+k_bin[:-1])/2
    bins[-1] = k_bin[-1]
    bins = bins.unsqueeze(0)
    bins[1:] += 1e-3
    inds = searchsorted(bins, rad.flatten().unsqueeze(0)).squeeze().int()
    # bincount = torch.histc(inds.cpu(), bins=bins.shape[1]+1).to(device)
    bincount = torch.bincount(inds)
    asort = torch.argsort(inds.squeeze())
    sorted_e_ = e_.view(e_.shape[0], -1)[:, asort]
    csum_e_ = torch.cumsum(sorted_e_, dim=1)
    binloc = torch.cumsum(bincount, dim=0).long()-1
    spec_ = csum_e_[:,binloc[1:]] - csum_e_[:,binloc[:-1]]
    spec_ = spec_[:, :-1]
    spec_ = spec_ * 2 * np.pi * (k_bin.float()**2) / bincount[1:-1].float()
    return spec_, k_bin


##################### COMPUTE STATS ###########################

def tkenergy(vel, avg=True):
    """
    compute total kinetic energy inside system
    :param vel: tensor of shape (N, 3, res, res, res)
    """

    tke = 0.5 * (vel[:, 0] **2 + vel[:, 1] **2)

    if avg:
        return torch.mean(tke, dim=(1, 2))     # N, 1
    else:
        return tke                             # N, res, res

def dissipation(vel, viscosity=0.000185, avg=True):
    """
    compute total energy dissipation inside system
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    vel_ = pad_rfft3(vel) # (N, 3, res, res, res/2+1, 2)
    # print("vel_.shape",vel_.shape)
    grad_ = spec_grad(vel_)   # (N, 3, 3, res, res, res/2+1, 2)
    # print("grad_.shape", grad_.shape)
    grad_t_ = grad_.transpose(1, 2)
    strain_ = 0.5 * (grad_ + grad_t_) # (N, 3, 3, res, res, res/2+1, 2)
    # print("strain_.shape", strain_.shape)
    strain = pad_irfft3(strain_) # (N, 3, 3, res, res, res)
    # print("strain.shape", strain.shape)
    diss = 2 * viscosity * torch.sum(strain**2, dim=(1, 2)) # (N, res, res, res)
    # print("diss.shape",diss.shape)
    if avg:
        return torch.mean(diss, dim=(1, 2))
    else:
        return diss

def rmsvelocity(vel, avg=True):
    """
    compute RMS velocity
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    rmsv = (tkenergy(vel, avg=False) * (2/3))**(1/2)
    # print("rmsv.shape",rmsv.shape)
    if avg:
        return torch.mean(rmsv, dim=(1, 2))
    else:
        return rmsv

def tmscale(vel, viscosity=0.000185, avg=True):
    """
    compute Taylor Micro Scale
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    rmsv = rmsvelocity(vel, avg=False)
    diss = dissipation(vel, avg=False, viscosity=viscosity)
    lambd = (15*viscosity*(rmsv**2)/diss)**(1/2)
    if avg:
        return torch.mean(lambd, dim=(1, 2))
    else:
        return lambd

def tsreynolds(vel, viscosity=0.000185, avg=True):
    """
    compute Taylor-scale Reynolds number
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    rmsv = rmsvelocity(vel, avg=False)
    lambd = tmscale(vel, viscosity=viscosity, avg=False)
    rey = rmsv * lambd / viscosity
    if avg:
        return torch.mean(rey, dim=(1, 2))
    else:
        return rey

def ktimescale(vel, viscosity=0.000185, avg=True):
    """
    compute Kolmogorov time scale
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    diss = dissipation(vel, viscosity, False)
    tau = (viscosity/diss)**(1/2)
    if avg:
        return torch.mean(tau, dim=(1, 2))
    else:
        return tau

def klenscale(vel, viscosity=0.000185, avg=True):
    """
    compute Kolmogorov length scale
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    diss = dissipation(vel, viscosity, False)
    eta = viscosity**(3/4) * diss**(-1/4)
    if avg:
        return torch.mean(eta, dim=(1, 2))
    else:
        return eta

def intscale(vel, avg=True):
    """
    compute integral scale
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    spec, k = energy_spectrum(vel)
    rmsv = rmsvelocity(vel, avg=False)

    c1 = np.pi/(2*rmsv**2)
    c2 = torch.sum(spec / k, dim=1)
    L = c1*(c2.unsqueeze(1).unsqueeze(1))
    if avg:
        return torch.mean(L, dim=(1, 2))
    else:
        return L

def eddytime(vel, avg=True):
    """
    compute large eddy turnover time
    :param vel: tensor of shape (N, 3, res, res, res)
    """
    L = intscale(vel)
    rmsv = rmsvelocity(vel, avg=False)
    # print(L.shape, rmsv.shape)
    TL = L.unsqueeze(1).unsqueeze(1) / rmsv
    if avg:
        return torch.mean(TL, dim=(1, 2))
    else:
        return TL

def compute_all_stats(vel, viscosity=0.000185, description=False):
    """
    compute all statistics
    :param vel: tensor of shape (N, 3, res, res, res)
    """

    spec, k = energy_spectrum(vel)

    tk = tkenergy(vel, avg=True)
    dis = dissipation(vel, avg=True)
    rms = (tk * (2 / 3)) ** (1 / 2)

    tkenergy_ = torch.mean(tk)
    dissipation_ = torch.mean(dis)
    rmsvelocity_ = torch.mean(rms)

    tm = (15 * viscosity * (rms ** 2) / dis) ** (1 / 2)
    tmscale_ = torch.mean(tm)

    tsreynolds_ = torch.mean(rms * tm / viscosity)

    ktimescale_ = torch.mean(torch.sqrt(viscosity / dis))
    klenscale_ = torch.mean(viscosity ** (3 / 4) * dis ** (-1 / 4))

    intscale = np.pi / (2 * rms ** 2) * torch.sum(spec / k, dim=1)
    intscale_ = torch.mean(intscale)

    eddytime_ = torch.mean(intscale / rms)

    stats = [tkenergy_, dissipation_, rmsvelocity_, tmscale_, tsreynolds_, ktimescale_, klenscale_, intscale_, eddytime_]
    stats = torch.stack(stats, dim=-1)
    descript = ["total kinetic energy", "total energy dissipation", "RMS velocity", "Taylor Micro Scale", "Taylor-scale Reynolds number", "Kolmogorov time scale", "Kolmogorov length scale", "integral scale", "large eddy turnover time"]
    if description:
        return stats, descript
    else:
        return stats


def test():
    dir = "./data/rb2d_ra1e6_s42.npz"
    npdata = np.load(dir)
    data = np.stack([npdata['p'], npdata['b'], npdata['u'], npdata['w']], axis=0)
    uw = np.transpose(data[2:, :, :, :], (1, 0, 2, 3))[:, :, :128, :]    #(2, N, 512, 128) -> (N, 2, 128, 128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    print(uw.shape)
    t0 = time.time()
    stats = compute_all_stats(uw[10:,:,:,:], viscosity=0.0001, description=False)
    t1 = time.time()

    s = [stats[..., i].item() for i in range(stats.shape[0])]
    print("********** Compute time for stats **********")
    print("{0:5.3f} s".format(t1-t0))
    print("************* Flow Statistics *************")
    print("Total Kinetic Energy     : {}".format(s[0]))
    print("Dissipation              : {}".format(s[1]))
    print("Rms velocity             : {}".format(s[2]))
    print("Taylor Micro. Scale      : {}".format(s[3]))
    print("Taylor-scale Reynolds    : {}".format(s[4]))
    print("Kolmogorov time sclae    : {}".format(s[5]))
    print("Kolmogorov length sclae  : {}".format(s[6]))
    print("Integral scale           : {}".format(s[7]))
    print("Large eddy turnover time : {}".format(s[8]))

if __name__ == '__main__':
    test()

