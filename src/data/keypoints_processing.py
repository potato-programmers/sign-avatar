# src/data/keypoints_processing.py
import numpy as np

# OpenPose BODY_25 인덱스 (참고용)
# 0: nose, 1: neck, 2: R-shoulder, 3: R-elbow, 4: R-wrist
# 5: L-shoulder, 6: L-elbow, 7: L-wrist, ...
BODY_NECK = 1
BODY_R_SHOULDER = 2
BODY_R_ELBOW = 3
BODY_R_WRIST = 4
BODY_L_SHOULDER = 5
BODY_L_ELBOW = 6
BODY_L_WRIST = 7


def normalize_keypoints_3d(pose_seq, lh_seq, rh_seq):
    """
    pose_seq : (T, 25, 3)  - (x,y,z)
    lh_seq   : (T, 21, 3)
    rh_seq   : (T, 21, 3)

    return   : motion (T, J, 3)
               J = 7 (upper body) + 21 (right hand) + 21 (left hand) = 49 (예시)
    """
    T = pose_seq.shape[0]

    motion_list = []

    for t in range(T):
        pose = pose_seq[t]   # (25,3)
        lh = lh_seq[t]       # (21,3)
        rh = rh_seq[t]       # (21,3)

        # --- 중심/스케일 계산 ---
        neck = pose[BODY_NECK]          # (3,)
        r_sh = pose[BODY_R_SHOULDER]
        l_sh = pose[BODY_L_SHOULDER]

        cx, cy, cz = neck
        shoulder_dist = np.linalg.norm(r_sh - l_sh) + 1e-6

        def norm(pt):
            x, y, z = pt
            nx = (x - cx) / shoulder_dist
            ny = (y - cy) / shoulder_dist
            nz = (z - cz) / shoulder_dist
            return np.array([nx, ny, nz], dtype=np.float32)

        joints_t = []

        # --- upper body joint 순서 (원하는 대로 바꿔도 됨) ---
        body_indices = [
            BODY_NECK,
            BODY_R_SHOULDER, BODY_R_ELBOW, BODY_R_WRIST,
            BODY_L_SHOULDER, BODY_L_ELBOW, BODY_L_WRIST,
        ]
        for idx in body_indices:
            joints_t.append(norm(pose[idx]))

        # --- 오른손 21개 ---
        for j in range(lh_seq.shape[1]):  # 사실 21로 고정
            joints_t.append(norm(rh[j]))

        # --- 왼손 21개 ---
        for j in range(lh_seq.shape[1]):
            joints_t.append(norm(lh[j]))

        motion_list.append(np.stack(joints_t, axis=0))  # (J,3)

    motion = np.stack(motion_list, axis=0)  # (T, J, 3)
    return motion


def resample_motion(motion, target_T):
    """
    motion  : (T, J, 3)  (numpy)
    target_T: 원하는 프레임 수 (예: 60)

    return  : (target_T, J, 3)
    """
    T, J, C = motion.shape
    if T == target_T:
        return motion

    old_ts = np.linspace(0.0, 1.0, T)
    new_ts = np.linspace(0.0, 1.0, target_T)

    new_motion = np.zeros((target_T, J, C), dtype=np.float32)

    for j in range(J):
        for c in range(C):
            new_motion[:, j, c] = np.interp(new_ts, old_ts, motion[:, j, c])

    return new_motion
