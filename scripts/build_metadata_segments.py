# scripts/build_metadata_segments.py
import os
import json
import glob
import argparse

from scripts.build_metadata import load_morpheme_file, find_keypoints_prefix  # 재사용


def build_metadata_segments(morpheme_root, keypoints_root, output_path, fps=30):
    morpheme_files = sorted(
        glob.glob(os.path.join(morpheme_root, "**", "*_morpheme.json"), recursive=True)
    )
    print(f"Found {len(morpheme_files)} morpheme files.")

    meta_list = []
    for m_path in morpheme_files:
        info = load_morpheme_file(m_path)  # {"video_id", "video_name", "segments", "gloss_seq"}
        video_id = info["video_id"]
        segments = info["segments"]

        kp_prefix = find_keypoints_prefix(video_id, keypoints_root)
        if kp_prefix is None:
            print(f"[WARN] keypoints not found for {video_id}, skip.")
            continue

        for idx, seg in enumerate(segments):
            gloss = seg["gloss"]
            start = seg["start"]
            end = seg["end"]

            # frame index 계산 (간단히 floor/ceil 사용)
            start_frame = int(start * fps)
            end_frame = int(end * fps)

            item = {
                "id": f"{video_id}_seg{idx:03d}",
                "video_id": video_id,
                "video_name": info["video_name"],
                "gloss_seq": [gloss],   # 이 segment만 단일 gloss로 학습
                "segment": {
                    "start": start,
                    "end": end,
                    "start_frame": start_frame,
                    "end_frame": end_frame
                },
                "keypoints_prefix": kp_prefix,
                "fps": fps
            }
            meta_list.append(item)

    print(f"Built segment-level metadata for {len(meta_list)} samples.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--morpheme-root", type=str, required=True)
    parser.add_argument("--keypoints-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    build_metadata_segments(
        morpheme_root=args.morpheme_root,
        keypoints_root=args.keypoints_root,
        output_path=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
