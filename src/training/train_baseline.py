# src/training/train_baseline.py
import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import create_experiment_dir

from src.data.vocab import GlossVocab
from src.data.dataset import Gloss2MotionDataset
from src.models.transformer_gloss2motion import Gloss2MotionTransformer
from .trainer import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_baseline.yaml",
        help="path to yaml config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(42)

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # ----- 경로/로깅 세팅 -----
    output_root = cfg["logging"]["output_dir"]
    exp_name = cfg["experiment_name"]
    exp_dir = create_experiment_dir(output_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- vocab / dataset -----
    vocab_path = cfg["data"]["gloss_vocab"]
    gloss_vocab = GlossVocab(vocab_path)
    vocab_size = len(gloss_vocab.gloss2id)

    target_T = cfg["data"]["target_T"]
    max_gloss_len = cfg["data"]["max_gloss_len"]

    train_meta = cfg["data"]["meta_train"]
    val_meta = cfg["data"]["meta_val"]

    train_dataset = Gloss2MotionDataset(
        meta_path=train_meta,
        gloss_vocab=gloss_vocab,
        target_T=target_T,
        max_gloss_len=max_gloss_len,
        use_3d=True,
    )

    val_dataset = Gloss2MotionDataset(
        meta_path=val_meta,
        gloss_vocab=gloss_vocab,
        target_T=target_T,
        max_gloss_len=max_gloss_len,
        use_3d=True,
    )

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ----- 모델 준비 -----
    model_cfg = cfg["model"]
    model = Gloss2MotionTransformer(
        gloss_vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_gloss_len=max_gloss_len,
        max_motion_len=target_T,
        num_joints=model_cfg["num_joints"],
        dropout=model_cfg["dropout"],
    )
    model.to(device)

    # ----- optimizer -----
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    pad_id = gloss_vocab.pad_id
    num_epochs = train_cfg["epochs"]

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Epoch {epoch}/{num_epochs} ====")

        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=pad_id,
        )
        print(f"Train loss: {train_stats['loss']:.6f}")

        val_stats = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            pad_id=pad_id,
        )
        print(f"Val   loss: {val_stats['loss']:.6f}")

        # ----- 체크포인트 저장 -----
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "config": cfg,
            },
            ckpt_path,
        )

        # best 모델 따로 저장
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ New best model saved to {best_path}")


if __name__ == "__main__":
    main()
