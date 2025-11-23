# scripts/build_vocab.py
import os
import json
import glob
import argparse
from collections import Counter


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_gloss_from_meta(meta_path):
    """하나의 meta_*.json에서 gloss들을 전부 모아서 리스트로 반환."""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_gloss = []
    for item in data:
        # build_metadata.py에서 "gloss_seq": ["왼쪽", ...] 형태로 저장했다고 가정
        gloss_seq = item.get("gloss_seq", [])
        all_gloss.extend(gloss_seq)

    return all_gloss


def build_vocab(meta_paths, min_freq=1, top_k=None):
    """
    meta_paths: meta_train.json, meta_val.json 등 경로 리스트
    min_freq : 이 빈도 이상인 gloss만 vocab에 포함
    top_k    : 상위 top_k개만 사용 (None이면 제한 없음)
    """
    counter = Counter()

    for path in meta_paths:
        glosses = load_gloss_from_meta(path)
        counter.update(glosses)

    # 빈도수 기준 필터링
    items = [(g, f) for g, f in counter.items() if f >= min_freq]

    # 빈도가 높은 순으로 정렬
    items.sort(key=lambda x: (-x[1], x[0]))

    if top_k is not None:
        items = items[:top_k]

    gloss_list = [g for g, _ in items]

    # special tokens 먼저 배치
    vocab = {}
    idx = 0
    for tok in SPECIAL_TOKENS:
        vocab[tok] = idx
        idx += 1

    for g in gloss_list:
        if g in vocab:
            continue
        vocab[g] = idx
        idx += 1

    return vocab


def save_vocab(vocab, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meta",
        type=str,
        nargs="+",
        required=True,
        help="meta json paths (e.g. meta_train.json meta_val.json)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="output gloss_vocab.json path",
    )
    p.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="minimum frequency for a gloss to be included",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="max number of gloss tokens (excluding special tokens)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    vocab = build_vocab(
        meta_paths=args.meta,
        min_freq=args.min_freq,
        top_k=args.top_k,
    )
    print(f"Built vocab with {len(vocab)} tokens (including special tokens).")
    save_vocab(vocab, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
