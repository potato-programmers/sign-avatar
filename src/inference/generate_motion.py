# src/inference/generate_motion.py
import argparse
import os
import json
import numpy as np
import torch

from src.utils.config import load_config
from src.data.vocab import GlossVocab
from src.models.transformer_gloss2motion import Gloss2MotionTransformer


def load_model_from_checkpoint(ckpt_path, gloss_vocab, config_path=None, device="cpu"):
    """
    ckpt_path:
      - train_baseline에서 저장한 epoch_XXX.pt (config 포함)
      - 또는 best.pt (model_state만 있는 경우)

    config_path:
      - best.pt처럼 config가 없을 수도 있으니, yaml 경로를 받을 수 있게.
    """
    if config_path is not None:
        cfg = load_config(config_path)
    else:
        # config가 ckpt 안에 있다고 가정하고 로딩
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "config" not in ckpt:
            raise ValueError("Config not found in checkpoint. Please provide --config.")
        cfg = ckpt["config"]

    vocab_size = len(gloss_vocab.gloss2id)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = Gloss2MotionTransformer(
        gloss_vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_gloss_len=data_cfg["max_gloss_len"],
        max_motion_len=data_cfg["target_T"],
        num_joints=model_cfg["num_joints"],
        dropout=model_cfg["dropout"],
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 두 가지 포맷 모두 지원
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg


def parse_gloss_text(gloss_text):
    """
    --gloss "왼쪽 어디 있다_의문" 같은 입력을
    공백 기준으로 나눈다고 가정.
    """
    gloss_text = gloss_text.strip()
    if not gloss_text:
        return []
    return gloss_text.split()


def generate_motion(model, gloss_vocab, gloss_seq, device, target_T):
    """
    gloss_seq: ["왼쪽", "어디", ...]
    return: np.ndarray (T, J, 3)
    """
    max_len = target_T  # 굳이 같을 필요는 없지만, config와 맞춰 쓰자.
    # 실제 max_gloss_len은 config에서 가져오는 게 더 안전
    # 여기서는 vocab.encode 호출 시 max_gloss_len을 별도로 줄 수도 있음.

    gloss_ids, gloss_len = gloss_vocab.encode(gloss_seq, max_len=gloss_vocab.max_len)
    gloss_ids = gloss_ids.unsqueeze(0).to(device)  # (1,N)
    gloss_mask = gloss_ids.eq(gloss_vocab.pad_id)  # (1,N)

    with torch.no_grad():
        motion = model(gloss_ids, gloss_mask)  # (1,T,J,3)

    motion = motion.squeeze(0).cpu().numpy()  # (T,J,3)
    return motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to checkpoint (epoch_xxx.pt or best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to train yaml (needed if checkpoint has no config)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="path to gloss_vocab.json",
    )
    parser.add_argument(
        "--gloss",
        type=str,
        required=True,
        help='gloss sequence as a string, e.g. "왼쪽 어디 있다_의문"',
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output .npy path to save motion (T,J,3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ----- vocab 로드 -----
    gloss_vocab = GlossVocab(args.vocab)

    # GlossVocab에 max_len 정보 추가해주는 게 편해서 여기서 설정
    # (없다면 default로 30 같은 값을 하드코딩해도 됨)
    gloss_vocab.max_len = 30  # config["data"]["max_gloss_len"]과 맞춰 쓰면 더 안전

    # ----- 모델 로드 -----
    model, cfg = load_model_from_checkpoint(
        ckpt_path=args.checkpoint,
        gloss_vocab=gloss_vocab,
        config_path=args.config,
        device=device,
    )

    target_T = cfg["data"]["target_T"]

    # ----- gloss 문자열 파싱 -----
    gloss_seq = parse_gloss_text(args.gloss)
    print("Input gloss_seq:", gloss_seq)

    # ----- 모션 생성 -----
    motion = generate_motion(
        model=model,
        gloss_vocab=gloss_vocab,
        gloss_seq=gloss_seq,
        device=device,
        target_T=target_T,
    )
    print("Generated motion shape:", motion.shape)  # (T,J,3)

    # ----- 저장 -----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, motion)
    print("Saved motion to", args.output)


if __name__ == "__main__":
    main()
