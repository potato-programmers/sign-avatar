# scripts/build_metadata.py
import os
import json
import argparse
import glob

from pathlib import Path


def load_morpheme_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metaData", {})
    video_name = meta.get("name")               # "NIA_SL_SEN0001_REAL17_D.mp4"
    video_id = os.path.splitext(video_name)[0]  # "NIA_SL_SEN0001_REAL17_D"

    segments = []
    for item in data.get("data", []):
        start = float(item["start"])
        end = float(item["end"])
        attrs = item.get("attributes", [])
        gloss = attrs[0]["name"] if attrs else None

        segments.append({
            "start": start,
            "end": end,
            "gloss": gloss
        })

    # 시간순 정렬
    segments = sorted(segments, key=lambda x: x["start"])

    gloss_seq = [s["gloss"] for s in segments if s["gloss"] is not None]

    return {
        "video_id": video_id,
        "video_name": video_name,
        "segments": segments,
        "gloss_seq": gloss_seq
    }


def find_keypoints_prefix(video_id, keypoints_root):
    """
    keypoints_root 아래에서
    {video_id}_*_keypoints.json 을 찾아 prefix 부분만 반환.

    예:
      /.../NIA_SL_SEN0001_REAL17_D/NIA_SL_SEN0001_REAL17_D_000000000000_keypoints.json
      -> prefix = /.../NIA_SL_SEN0001_REAL17_D/NIA_SL_SEN0001_REAL17_D
    """
    pattern = os.path.join(keypoints_root, "**", f"{video_id}_*_keypoints.json")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None

    # 첫 번째 매치를 기준으로 prefix 추출
    sample = candidates[0]
    # ".../NIA_SL_SEN0001_REAL17_D_000000000000_keypoints.json"
    base = os.path.basename(sample)
    # "NIA_SL_SEN0001_REAL17_D_000000000000_keypoints.json"
    stem = base.replace("_keypoints.json", "")
    # "NIA_SL_SEN0001_REAL17_D_000000000000"
    prefix_without_frame = stem.rsplit("_", 1)[0]
    # "NIA_SL_SEN0001_REAL17_D"

    # 디렉토리 경로
    dir_path = os.path.dirname(sample)
    prefix_path = os.path.join(dir_path, prefix_without_frame)

    return prefix_path


def build_metadata(morpheme_root, keypoints_root, output_path, fps=30):
    morpheme_files = sorted(glob.glob(os.path.join(morpheme_root, "**", "*_morpheme.json"),
                                      recursive=True))
    print(f"Found {len(morpheme_files)} morpheme files.")

    meta_list = []

    for m_path in morpheme_files:
        info = load_morpheme_file(m_path)
        video_id = info["video_id"]

        kp_prefix = find_keypoints_prefix(video_id, keypoints_root)
        if kp_prefix is None:
            print(f"[WARN] keypoints not found for {video_id}, skip.")
            continue

        item = {
            "id": video_id,
            "video_name": info["video_name"],
            "gloss_seq": info["gloss_seq"],
            "segments": info["segments"],
            "keypoints_prefix": kp_prefix,
            "fps": fps
        }
        meta_list.append(item)

    print(f"Built metadata for {len(meta_list)} samples.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--morpheme-root", type=str, required=True,
                        help="root dir of *_morpheme.json files")
    parser.add_argument("--keypoints-root", type=str, required=True,
                        help="root dir containing *_keypoints.json files")
    parser.add_argument("--output", type=str, required=True,
                        help="output meta json path")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()
    build_metadata(
        morpheme_root=args.morpheme_root,
        keypoints_root=args.keypoints_root,
        output_path=args.output,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
