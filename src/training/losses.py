# src/training/losses.py
import torch.nn.functional as F


def mse_motion_loss(pred, target):
    """
    pred, target: (B,T,J,3)
    """
    return F.mse_loss(pred, target)
