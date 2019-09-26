"""
Return metrics of all samples summed, NOT average.
"""
import torch


def err(output, target, mean, std):
    """
    Metric example. Return accuracy.
    """
    output = torch.clamp_min(torch.round(output * std + mean), min=0.0)
    # output = output * std + mean
    target = target * std + mean

    err = torch.sum(torch.abs(output - target)) / torch.sum(target)
    return err * output.size(0)
