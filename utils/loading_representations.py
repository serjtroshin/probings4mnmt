import numpy as np
import torch


def loading_rep(path):
    output = np.load(path, allow_pickle=True)
    return np.stack(map(lambda x: x.numpy(), list(output)))

def loading_target(path):
    output = np.load(path, allow_pickle=True)
    return output
