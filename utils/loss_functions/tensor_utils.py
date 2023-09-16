import numpy as np
import torch
from torch import nn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def mean_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
