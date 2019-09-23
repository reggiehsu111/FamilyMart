"""
Return losses of all samples summed, NOT average.
"""
import torch.nn as nn


def mse_loss(output, target):
    return nn.L1Loss(reduction='sum')(output, target)
    #return nn.MSELoss(reduction='sum')(output, target)
