import torch
import numpy as np

def pad_rfft3(f, onesided=True):
    """
    padded batch real fft
    :param f: tensor of shape [..., res0, res1, res2]
    """
    n0, n1, n2 = f.shape[-3:]
    h0, h1, h2 = int(n0/2), int(n1/2), int(n2/2)

    F2 = torch.rfft(f, signal_ndim=1, onesided=onesided) # [..., res0, res1, res2/2+1, 2]
    F2[..., h2, :] = 0

    F1 = torch.fft(F2.transpose(-3,-2), signal_ndim=1)
    F1[..., h1,:] = 0
    F1 = F1.transpose(-2,-3)

    F0 = torch.fft(F1.transpose(-4,-2), signal_ndim=1)
    F0[..., h0,:] = 0
    F0 = F0.transpose(-2,-4)
    return F0

def pad_irfft3(F):
    """
    padded batch inverse real fft
    :param f: tensor of shape [..., res0, res1, res2/2+1, 2]
    """
    res = F.shape[-3]
    f0 = torch.ifft(F.transpose(-4,-2), signal_ndim=1).transpose(-2,-4)
    f1 = torch.ifft(f0.transpose(-3,-2), signal_ndim=1).transpose(-2,-3)
    f2 = torch.irfft(f1, signal_ndim=1, signal_sizes=[res]) # [..., res0, res1, res2]
    return f2

def pad_fft2(f):
    """
    padded batch real fft
    :param f: tensor of shape [..., res0, res1]
    """
    n0, n1 = f.shape[-2:]
    h0, h1 = int(n0/2), int(n1/2)
    # turn f into complex signal
    f = torch.stack((f, torch.zeros_like(f)), dim=-1) # [..., res0, res1, 2]

    F1 = torch.fft(f, signal_ndim=1) # [..., res0, res1, 2]
    F1[..., h1,:] = 0 # [..., res0, res1, 2]

    F0 = torch.fft(F1.transpose(-3,-2), signal_ndim=1)
    F0[..., h0,:] = 0
    F0 = F0.transpose(-2,-3)
    return F0

def pad_ifft2(F):
    """
    padded batch inverse real fft
    :param f: tensor of shape [..., res0, res1, res2/2+1, 2]
    """
    f0 = torch.ifft(F.transpose(-3,-2), signal_ndim=1).transpose(-2,-3)
    f1 = torch.ifft(f0, signal_ndim=1)
    return f2

def rfftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return: frequency tensor of shape [dim, res, res, res/2+1]
    """
    # print("res",res)
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=0)

    # print("omega.shape",omega.shape)
    return omega

def fftfreqs(res, dtype=torch.float32):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return: frequency tensor of shape [dim, res, res, res]
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=0)

    return omega

def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res

def reconstruct(uv, w_):
    """
    reconstruct entire field from uv and mean w in z
    :param uv: tensor of shape (batch, 2, res0, res1, res2)
    :param w_: tensor of shape (batch, 1, res0, res1)
    """
    # reconstruct field
    res = uv.shape[-3]
    UV = pad_rfft3(uv) # [batch, 2, res0, res1, res2/2+1, 2]
    U = UV[:, 0:1] # [batch, 1, res0, res1, res2/2+1, 2]
    V = UV[:, 1:2] # [batch, 1, res0, res1, res2/2+1, 2]
    W0 = pad_fft2(w_) # [batch, 1, res0, res1, 2]
    K = rfftfreqs([res]*3, dtype=uv.dtype).to(uv.device) # [3, res0, res1, res2/2+1]
    K[2,:,:,0] += 1 # avoid division by 0
    K = K.unsqueeze(-1) # [3, res0, res1, res2/2+1, 1]
    W = - (K[0] * U + K[1] * V) / K[2]  # [batch, 1, res0, res1, res2/2+1, 2]
    W[..., 0, :] = W0
    w = pad_irfft3(W) # [batch, 1, res0, res1, res2]
    uvw = torch.cat((uv, w), dim=1) # [batch, 3, res0, res1, res2]

    K = rfftfreqs([res]*3, dtype=uv.dtype).to(uv.device).unsqueeze(-1) # [3, res0, res1, res2/2+1]
    F = torch.cat((U,V,W),dim=1)
    Div = -img(K*F)
    print(Div[0,2,:,:,0,0])
    print(Div[0,2,:,:,0,1])

    return uvw

def spec_grad(S, dim=[0, 1]):
    """
    Compute spectral gradient of scalar field (or Jacobian of vector field)
    *Note: Assumes same size in all dims
    :parap S:    scalar field of shape [batch, res, res, res/2+1, 2]
              or vector field of shape [batch, dim, res, res, res/2+1, 2]

    """
    assert(len(S.shape) in [5, 6])
    assert(isinstance(dim, list) or isinstance(dim, tuple))
    is_scalar = True if len(S.shape) == 4 else False
    # print("is_scalar",is_scalar)
    # print("S.shape",S.shape)
    res = S.shape[-3]
    K = rfftfreqs([res]*2, dtype=S.dtype).unsqueeze(-1).to(S.device) # [dim, res, res, res/2+1, 1]
    # print("K.shape",K.shape)
    if is_scalar:
        return -img(K[dim] * S.unsqueeze(1))
    else:
        return -img(K[dim].unsqueeze(1) * S.unsqueeze(1))

def spec_div(F):
    """
    Compute spectral divergence
    :param F: vector field tensor of shape (batch, dim, res, res, res/2+1, 2)
    """
    res = F.shape[2]
    K = rfftfreqs([res]*3, dtype=F.dtype).unsqueeze(-1).to(F.device)  # [dim, res, res, res/2+1, 1]
    Div = torch.sum(-img(K*F), dim=1) # [batch, res, res, res/2+1, 2]
    return Div

def spec_curl(F):
    """
    Compute spectral curl
    :param F: vector field tensor of shape (batch, dim, res, res, res/2+1, 2)
    :return cF: curl of F (batch, dim, res, res, res/2+1, 2)
    """
    U, V, W = F[:, 0:1], F[:, 1:2], F[:, 2:3]
    J = spec_grad(F) # [batch, dim, dim, res, res, res/2+1, 2]
    cF = J[:,[1,2,0],[2,0,1]] - J[:,[2,0,1],[1,2,0]]
    return cF

def phys_div(f):
    """
    Compute physical divergence
    :param f: vector field tensor of shape (batch, dim, res, res, res)
    """
    F = pad_rfft3(f) # [batch, dim, res, res, res/2+1, 2]
    Div = spec_div(F)
    div = pad_irfft3(Div) # [batch, res, res, res]
    return div

def phys_proj(f):
    """
    Project physical vector field to be solenoidal
    :param F: vector field tensor of shape (batch, dim, res, res, res)
    :return sF: solenoidal field f (batch, dim, res, res, res/2+1, 2)
    """
    F = pad_rfft3(f)
    F_ = spec_proj(F)
    f_ = pad_irfft3(F_)
    return f_

def spec_proj(F):
    """
    Project spectal vector field to be solenoidal
    :param F: vector field tensor of shape (batch, dim, res, res, res/2+1, 2)
    :return sF: solenoidal field F (batch, dim, res, res, res/2+1, 2)
    """
    res = F.shape[2]
    divF = spec_div(F)          # [batch, res, res, res/2+1, 2]
    K = rfftfreqs([res]*3, dtype=F.dtype).unsqueeze(-1).to(F.device) # [dim, res, res, res/2+1, 1]
    Lap = -torch.sum(K**2, dim=0)    # [res, res, res/2+1, 1]
    Lap[0, 0, 0] = 1                 #  prevent division by 0
    Phi = divF / Lap                 # [batch, res, res, res/2+1, 2]
    Phi[:, 0, 0, 0] = 0              #  arbitrary gauge value
    sF = F - spec_grad(Phi)
    return sF