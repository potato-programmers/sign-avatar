import json
import numpy as np

BONE_NAMES = [
    "neck",
    "rightShoulder",
    "rightElbow",
    "rightWrist",
    # ...
]

def motion_to_tracks(motion, fps=30, name=""):
    """
    motion: (T,J,3) numpy
    """
    T, J, C = motion.shape
    duration = T / fps
    times = [t / fps for t in range(T)]

    tracks = []
    for j, bone in enumerate(BONE_NAMES):
        xyz = motion[:, j, :]   # (T,3)
        values_flat = xyz.reshape(-1).tolist()
        tracks.append({
            "bone": bone,
            "type": "vector",
            "times": times,
            "values": values_flat
        })

    return {
        "name": name,
        "duration": duration,
        "tracks": tracks
    }

def save_tracks_json(tracks_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tracks_dict, f, ensure_ascii=False, indent=2)
