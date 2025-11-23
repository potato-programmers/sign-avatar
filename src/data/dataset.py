# src/data/dataset.py
import json
import torch
from torch.utils.data import Dataset

from .keypoints_io import load_openpose_sequence
from .keypoints_processing import normalize_keypoints_3d, resample_motion


class Gloss2MotionDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        gloss_vocab,
        target_T: int = 60,
        max_gloss_len: int = 30,
        use_3d: bool = True,
    ):
        """
        meta_path: build_metadata.py 로 만든 meta_{train,val}.json
        gloss_vocab: src/data/vocab.py 의 GlossVocab 인스턴스
        """
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.gloss_vocab = gloss_vocab
        self.target_T = target_T
        self.max_gloss_len = max_gloss_len
        self.use_3d = use_3d

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        info = self.meta[idx]

        gloss_seq = info["gloss_seq"]          # ["왼쪽", ...]
        kp_prefix = info["keypoints_prefix"]   # .../NIA_SL_XXX/NIA_SL_XXX
        # fps = info.get("fps", 30)            # 필요하면 나중에 사용

        # ---- gloss 인코딩 ----
        gloss_ids, gloss_len = self.gloss_vocab.encode(
            gloss_seq, max_len=self.max_gloss_len
        )  # (max_len,), int

        # ---- keypoints 로드 → (T, J, 3) ----
        pose_seq, lh_seq, rh_seq = load_openpose_sequence(
            kp_prefix, use_3d=self.use_3d
        )  # (T,25,3), (T,21,3), (T,21,3)

        motion = normalize_keypoints_3d(pose_seq, lh_seq, rh_seq)  # (T,J,3)
        motion = resample_motion(motion, self.target_T)            # (target_T,J,3)

        motion_tensor = torch.from_numpy(motion).float()
        motion_len = self.target_T  # v1에서는 고정 길이

        sample = {
            "gloss_ids": gloss_ids,                 # (max_gloss_len,)
            "gloss_len": torch.tensor(gloss_len),   # scalar
            "motion": motion_tensor,                # (T,J,3)
            "motion_len": torch.tensor(motion_len)  # scalar
        }
        return sample
