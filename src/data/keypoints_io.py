import json
import os
import glob
import numpy as np


def _reshape_keypoints(arr, dims_per_joint):
    """
    arr: flat list from JSON
    dims_per_joint: 3 for 2D, 4 for 3D
    return: (K, dims_per_joint)
    """
    if arr is None:
        return None
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    assert arr.size % dims_per_joint == 0, \
        f"Array length {arr.size} not divisible by {dims_per_joint}"
    return arr.reshape(-1, dims_per_joint)


def load_openpose_frame(path, use_3d=True):
    """
    한 프레임의 *_keypoints.json 을 읽어서
    pose / left_hand / right_hand 를 (K,3)로 반환.

    use_3d=True면 *_keypoints_3d에서 (x,y,z) 사용
    use_3d=False면 *_keypoints_2d에서 (x,y,conf) 사용 (z=conf 로 두거나 이후 무시)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", None)
    if people is None:
        raise ValueError(f"'people' key not found in {path}")

    # 네 예시는 people이 dict 였는데, OpenPose 기본은 list라 둘 다 처리
    if isinstance(people, list):
        if len(people) == 0:
            raise ValueError(f"No person detected in {path}")
        p = people[0]
    else:
        p = people

    if use_3d:
        pose_raw = p.get("pose_keypoints_3d", [])
        lh_raw = p.get("hand_left_keypoints_3d", [])
        rh_raw = p.get("hand_right_keypoints_3d", [])
        dims = 4  # x,y,z,score
        pose = _reshape_keypoints(pose_raw, dims)
        lh = _reshape_keypoints(lh_raw, dims)
        rh = _reshape_keypoints(rh_raw, dims)
        # score(마지막 열)은 버리고 (x,y,z)만 사용
        pose = pose[:, :3] if pose is not None else None
        lh = lh[:, :3] if lh is not None else None
        rh = rh[:, :3] if rh is not None else None
    else:
        pose_raw = p.get("pose_keypoints_2d", [])
        lh_raw = p.get("hand_left_keypoints_2d", [])
        rh_raw = p.get("hand_right_keypoints_2d", [])
        dims = 3  # x,y,conf
        pose = _reshape_keypoints(pose_raw, dims)
        lh = _reshape_keypoints(lh_raw, dims)
        rh = _reshape_keypoints(rh_raw, dims)
        # 여기서는 (x,y,conf)를 그대로 두고 이후 normalize 단계에서 처리

    return pose, lh, rh


def load_openpose_sequence(prefix_path, use_3d=True):
    """
    prefix_path 예시:
      "/.../NIA_SL_WORD0001_REAL01_D/NIA_SL_WORD0001_REAL01_D"

    실제 파일들은:
      NIA_SL_WORD0001_REAL01_D_000000000000_keypoints.json
      NIA_SL_WORD0001_REAL01_D_000000000001_keypoints.json
      ...

    를 가정하고, 모든 프레임을 시간 순서대로 읽어서
    pose_seq, lh_seq, rh_seq 를 (T,K,3) 로 반환.
    """
    pattern = prefix_path + "_*_keypoints.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No keypoint json files found for pattern: {pattern}")

    pose_list, lh_list, rh_list = [], [], []

    for p in paths:
        pose, lh, rh = load_openpose_frame(p, use_3d=use_3d)
        pose_list.append(pose)
        lh_list.append(lh)
        rh_list.append(rh)

    # 길이가 다를 수 있으니 첫 프레임 기준으로 K를 체크
    pose_seq = np.stack(pose_list, axis=0) if pose_list[0] is not None else None
    lh_seq = np.stack(lh_list, axis=0) if lh_list[0] is not None else None
    rh_seq = np.stack(rh_list, axis=0) if rh_list[0] is not None else None

    return pose_seq, lh_seq, rh_seq
