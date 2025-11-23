# src/inference/export_tracks.py
# TODO - 이거 아바타쪽이랑 연동 확인 필요
import argparse
import json
import os
import numpy as np

# J 개수와 동일한 bone 이름 리스트 (keypoints_processing에서 정의한 순서와 맞춰야 함)
# 예시: 7 (upper body) + 21 (right hand) + 21 (left hand) = 49
BONE_NAMES = [
    "neck",
    "rightShoulder",
    "rightElbow",
    "rightWrist",
    "leftShoulder",
    "leftElbow",
    "leftWrist",
    # 오른손 21개 (이름은 엔진에서 쓰는 이름에 맞춰 바꿔도 됨)
] + [f"r_hand_{i}" for i in range(21)] + [f"l_hand_{i}" for i in range(21)]


def motion_to_tracks(motion, fps=30.0, name=""):
    """
    motion: np.ndarray (T, J, 3)
    """
    T, J, C = motion.shape
    assert J == len(BONE_NAMES), f"num joints mismatch: motion J={J}, len(BONE_NAMES)={len(BONE_NAMES)}"

    duration = T / fps
    times = [t / fps for t in range(T)]

    tracks = []
    for j_idx, bone_name in enumerate(BONE_NAMES):
        xyz = motion[:, j_idx, :]  # (T,3)
        values_flat = xyz.reshape(-1).tolist()
        tracks.append({
            "bone": bone_name,
            "type": "vector",
            "times": times,
            "values": values_flat
        })

    return {
        "name": name,
        "duration": duration,
        "tracks": tracks
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="path to .npy file containing motion (T,J,3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output json path",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="name field in tracks json (e.g. gloss or sentence)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="fps for time axis",
    )
    args = parser.parse_args()

    motion = np.load(args.motion_file)  # (T,J,3)
    print("Loaded motion:", motion.shape)

    tracks = motion_to_tracks(motion, fps=args.fps, name=args.name)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=2)

    print("Saved tracks json to", args.output)


if __name__ == "__main__":
    main()
