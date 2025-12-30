import json
import numpy as np
import cv2

def load_stereo_json(json_path: str):
    with open(json_path, "r") as f:
        j = json.load(f)

    K1 = np.array(j["K1"], dtype=np.float64)
    D1 = np.array(j["D1"][0], dtype=np.float64).reshape(-1, 1)   # (5,1)

    K2 = np.array(j["K2"], dtype=np.float64)
    D2 = np.array(j["D2"][0], dtype=np.float64).reshape(-1, 1)

    R = np.array(j["R"], dtype=np.float64)
    T = np.array(j["T"], dtype=np.float64).reshape(3, 1)         # 单位：mm

    w, h = j["image_size"]
    return K1, D1, K2, D2, R, T, (w, h)

def build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=0):
    w, h = size

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2,
        (w, h),
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, (w, h), cv2.CV_16SC2
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, (w, h), cv2.CV_16SC2
    )
    return map1x, map1y, map2x, map2y, Q

def rectify_pair(left_bgr, right_bgr, maps):
    map1x, map1y, map2x, map2y, _ = maps
    left_rect  = cv2.remap(left_bgr,  map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect

def rectify(left, right, json_path: str):
    K1, D1, K2, D2, R, T, size = load_stereo_json(json_path)
    maps = build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=0)
    left_rect, rigth_rect = rectify_pair(left, right, maps)
    return left_rect,rigth_rect

def seperation(frame:np.ndarray):
    _, W = frame.shape[:2]
    mid = W // 2
    left_half  = frame[:, :mid]
    right_half = frame[:, mid:]
    return left_half,right_half