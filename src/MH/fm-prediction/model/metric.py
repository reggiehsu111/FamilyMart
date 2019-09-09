"""
Return metrics of all samples summed, NOT average.
"""
import torch


def err(output, target, mean, std):
    """
    Metric example. Return accuracy.
    """
    output = torch.round(output * std + mean)
    # output = output * std + mean
    target = target * std + mean

    err = torch.sum(torch.abs(output - target)) / torch.sum(target)
    # print(torch.sum(torch.abs(output - target)))
    
    return err * output.size(0)
