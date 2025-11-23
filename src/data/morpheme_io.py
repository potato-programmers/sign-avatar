import json
import os


def load_morpheme_file(path):
    """
    NIA_SL_SEN0001_REAL17_D_morpheme.json 하나를 읽어서

    return:
      {
        "video_name": "NIA_SL_SEN0001_REAL17_D.mp4",
        "segments": [
          {
            "start": 2.052,
            "end": 2.963,
            "gloss": "왼쪽"
          },
          ...
        ]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metaData", {})
    video_name = meta.get("name")  # "NIA_SL_SEN0001_REAL17_D.mp4"

    segments = []
    for item in data.get("data", []):
        start = float(item["start"])
        end = float(item["end"])
        attrs = item.get("attributes", [])
        # 일단 가장 단순하게 첫 번째 attribute.name 을 gloss로 사용
        gloss_name = attrs[0]["name"] if attrs else None

        segments.append({
            "start": start,
            "end": end,
            "gloss": gloss_name
        })

    return {
        "video_name": video_name,
        "segments": segments
    }


def morpheme_to_gloss_sequence(morpheme_dict):
    """
    위 load_morpheme_file 결과를 받아서
    그냥 gloss 시퀀스 리스트만 뽑기 (시간 순서대로).
    """
    segs = sorted(morpheme_dict["segments"], key=lambda x: x["start"])
    gloss_seq = [s["gloss"] for s in segs if s["gloss"] is not None]
    return gloss_seq
