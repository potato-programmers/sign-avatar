# src/training/trainer.py
from typing import Dict

import torch
from torch.utils.data import DataLoader
from .losses import mse_motion_loss


def make_gloss_mask(gloss_ids, pad_id):
    """
    gloss_ids: (B, N)
    return: mask (B, N)  True = pad 위치
    """
    return gloss_ids.eq(pad_id)


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        gloss_ids = batch["gloss_ids"].to(device)      # (B,N)
        motion_gt = batch["motion"].to(device)         # (B,T,J,3)

        gloss_mask = make_gloss_mask(gloss_ids, pad_id)  # (B,N)

        motion_pred = model(gloss_ids, gloss_mask)     # (B,T,J,3)

        loss = mse_motion_loss(motion_pred, motion_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        gloss_ids = batch["gloss_ids"].to(device)
        motion_gt = batch["motion"].to(device)

        gloss_mask = make_gloss_mask(gloss_ids, pad_id)

        motion_pred = model(gloss_ids, gloss_mask)

        loss = mse_motion_loss(motion_pred, motion_gt)

        total_loss += float(loss.item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss}
