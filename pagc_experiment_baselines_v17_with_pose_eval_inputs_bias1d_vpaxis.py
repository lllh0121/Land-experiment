import cv2
import json
import numpy as np
import itertools
import math
from itertools import permutations
import os
import re
import glob
import argparse
import gc
import sys
import roi
from typing import Dict, Any, Optional, List, Tuple

# 独立的“几何一致椭圆边缘选择”模块
try:
    from ellipse_edge_geometry_selector import (
        choose_measurement_representation as choose_measurement_representation_geometry,
        GeometryEdgeSelectorConfig,
    )
except Exception:
    choose_measurement_representation_geometry = None
    GeometryEdgeSelectorConfig = None

try:
    from physical_randomized_contour_selector_v3 import select_randomized_physical_contour
except Exception:
    select_randomized_physical_contour = None

try:
    import pose_from_lights_no_chess_prior as light_only_pose
except Exception:
    light_only_pose = None

try:
    import light_only_iterative_pose as light_only_pose_v2
except Exception:
    light_only_pose_v2 = None

# ==========================
# 1. 路径配置（按需修改）
# ==========================
chess_img_path = os.path.join(os.path.dirname(__file__), "data", "images", "undist_d1p2_y0p0_p0p2_chess_001.jpg")
cam_intr_path = os.path.join(os.path.dirname(__file__), "config", "camera_rectified_intrinsics.json")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "legacy")

pattern_size = (11, 8)
square_size = 20.0

circle_centers_w = np.array([
    [-25, 275, 0.0], [375, 275, 0.0],
    [-25, 675, 0.0], [375, 675, 0.0]
], dtype=np.float64)

ROI_PAD = 10


def _robust_rescale(arr, valid_mask=None, q_low=2.0, q_high=98.0):
    arr = np.asarray(arr, dtype=np.float32)
    if valid_mask is None:
        vals = arr.reshape(-1)
    else:
        vals = arr[valid_mask > 0].reshape(-1)
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ellipse_contour_residual(ellipse, contour_pts):
    if ellipse is None or contour_pts is None or len(contour_pts) < 6:
        return np.inf
    pts = np.asarray(contour_pts, dtype=np.float64).reshape(-1, 2)
    (cx, cy), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, 1e-6)
    b = max(float(ma) * 0.5, 1e-6)
    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    resid = np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0)
    return float(np.median(resid))


def _auto_sample_count_from_ellipse(ellipse):
    (_, _), (MA, ma), _ = ellipse
    a = max(float(MA) * 0.5, 1.0)
    b = max(float(ma) * 0.5, 1.0)
    perim = np.pi * (3.0 * (a + b) - np.sqrt(max(1e-6, (3.0 * a + b) * (a + 3.0 * b))))
    n = int(np.clip(np.ceil(perim / 2.0), 72, 240))
    return n


def _bilinear_at(gray, x, y):
    h, w = gray.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = float(gray[y0, x0])
    v10 = float(gray[y0, x1])
    v01 = float(gray[y1, x0])
    v11 = float(gray[y1, x1])
    return (
        (1.0 - dx) * (1.0 - dy) * v00 +
        dx * (1.0 - dy) * v10 +
        (1.0 - dx) * dy * v01 +
        dx * dy * v11
    )


def _smooth_profile_1d(profile):
    prof = np.asarray(profile, dtype=np.float32).reshape(1, -1)
    k = int(max(3, 2 * (len(profile) // 40) + 1))
    if k % 2 == 0:
        k += 1
    sm = cv2.GaussianBlur(prof, (k, 1), 0.0).reshape(-1)
    return sm.astype(np.float64)


def _auto_search_half_len(ellipse, roi_gray, base_pt, normal):
    (_, _), (MA, ma), _ = ellipse
    minor_r = max(2.0, 0.25 * min(float(MA), float(ma)))
    probe = int(np.clip(round(minor_r * 0.40), 5, 24))
    ds = np.linspace(-probe, probe, 2 * probe + 1)
    vals = np.array([_bilinear_at(roi_gray, base_pt[0] + d * normal[0], base_pt[1] + d * normal[1]) for d in ds], dtype=np.float64)
    vals = _smooth_profile_1d(vals)
    g = np.gradient(vals)
    mag = np.abs(g)
    if mag.size < 3:
        return max(5, probe // 2)
    peak_idx = int(np.argmax(mag))
    peak = float(mag[peak_idx])
    support = np.where(mag >= 0.35 * max(peak, 1e-6))[0]
    if support.size >= 2:
        width = float(ds[support[-1]] - ds[support[0]])
    else:
        width = 2.0
    L = int(np.clip(np.ceil(1.8 * max(2.0, width)), 5, 24))
    return L



def _fit_edge_from_profile(gray_roi, base_pt, normal, search_half_len):
    step = 0.5
    ds = np.arange(-float(search_half_len), float(search_half_len) + step, step, dtype=np.float64)
    prof = np.array([
        _bilinear_at(gray_roi, base_pt[0] + d * normal[0], base_pt[1] + d * normal[1])
        for d in ds
    ], dtype=np.float64)
    prof_sm = _smooth_profile_1d(prof)
    if prof_sm.size < 9:
        return None

    center_idx = int(np.argmin(np.abs(ds)))
    inner_vals = prof_sm[:center_idx + 1]
    outer_vals = prof_sm[center_idx:]
    if inner_vals.size < 4 or outer_vals.size < 4:
        return None

    inner_mean = float(np.mean(inner_vals))
    outer_mean = float(np.mean(outer_vals))
    if inner_mean >= outer_mean:
        ds_work = ds[center_idx:].copy()
        prof_work = prof_sm[center_idx:].copy()
        sign_flip = 1.0
    else:
        ds_work = (-ds[:center_idx + 1][::-1]).copy()
        prof_work = prof_sm[:center_idx + 1][::-1].copy()
        sign_flip = -1.0

    if prof_work.size < 9:
        return None

    grad_work = np.gradient(prof_work, ds_work)
    grad_noise = float(np.median(np.abs(grad_work - np.median(grad_work))) + 1e-9)
    inner_len = max(4, min(len(prof_work), max(6, len(prof_work) // 5)))
    outer_len = max(4, min(len(prof_work), max(6, len(prof_work) // 5)))
    peak_val = float(np.percentile(prof_work[:inner_len], 92.0))
    bg_val = float(np.percentile(prof_work[-outer_len:], 20.0))
    dyn = float(max(peak_val - bg_val, np.max(prof_work) - np.min(prof_work)))
    if dyn < 4.0:
        return None

    plateau_val_thr = float(peak_val - max(2.5, 0.10 * dyn))
    plateau_grad_thr = float(max(2.8 * grad_noise, 0.015 * dyn))
    plateau_end = 0
    non_plateau_run = 0
    for i in range(len(prof_work) - 2):
        is_plateau = (prof_work[i] >= plateau_val_thr) and (abs(grad_work[i]) <= plateau_grad_thr)
        if is_plateau:
            plateau_end = i
            non_plateau_run = 0
        else:
            non_plateau_run += 1
            if non_plateau_run >= 2:
                break
    plateau_end = int(np.clip(plateau_end, 0, len(prof_work) - 4))

    guard = max(1, int(np.ceil(0.08 * len(prof_work))))
    start_idx = int(np.clip(plateau_end + guard, 1, len(prof_work) - 4))
    candidate_idx = np.arange(start_idx, len(prof_work) - 2, dtype=np.int32)
    if candidate_idx.size < 3:
        return None

    neg_grad_thr = float(max(2.2 * grad_noise, 0.018 * dyn))
    strong_mask = (-grad_work[candidate_idx]) >= neg_grad_thr
    if not np.any(strong_mask):
        strong_mask = grad_work[candidate_idx] < 0.0
    candidate_idx = candidate_idx[strong_mask]
    if candidate_idx.size == 0:
        return None

    level = np.clip((prof_work[candidate_idx] - bg_val) / (dyn + 1e-9), 0.0, 1.0)
    target_level = 0.52
    level_score = np.exp(-((level - target_level) ** 2) / (2.0 * (0.18 ** 2)))
    dist_score = np.exp(-0.085 * np.maximum(ds_work[candidate_idx] - ds_work[start_idx], 0.0))
    monotonic_score = []
    for idx in candidate_idx:
        left = max(start_idx, idx - 2)
        right = min(len(prof_work) - 1, idx + 3)
        seg = np.diff(prof_work[left:right])
        mono = float(np.mean(seg <= max(0.0, 0.01 * dyn))) if seg.size else 0.0
        monotonic_score.append(mono)
    monotonic_score = np.asarray(monotonic_score, dtype=np.float64)
    grad_score = np.maximum(-grad_work[candidate_idx], 0.0)
    score = grad_score * level_score * dist_score * np.clip(0.55 + 0.65 * monotonic_score, 0.1, 1.5)
    if score.size == 0 or not np.any(np.isfinite(score)):
        return None

    peak_local = int(np.argmax(score))
    peak_idx = int(candidate_idx[peak_local])
    peak_mag = float(-grad_work[peak_idx])
    if peak_mag < neg_grad_thr:
        return None

    if 1 <= peak_idx < len(ds_work) - 1:
        y1, y2, y3 = score[max(0, peak_local - 1)], score[peak_local], score[min(len(score) - 1, peak_local + 1)]
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-12:
            frac = np.clip((y1 - y3) / (2.0 * denom), -1.0, 1.0)
        else:
            frac = 0.0
    else:
        frac = 0.0

    d_sub_work = ds_work[peak_idx] + frac * step
    d_sub = sign_flip * d_sub_work
    edge_pt = base_pt + d_sub * normal

    left_idx = max(start_idx, peak_idx - 4)
    right_idx = min(len(prof_work) - 1, peak_idx + 4)
    contrast = float(abs(np.mean(prof_work[left_idx:peak_idx + 1]) - np.mean(prof_work[peak_idx:right_idx + 1])))
    local_grad = grad_work[max(start_idx, peak_idx - 2):min(len(grad_work), peak_idx + 3)]
    local_curv = float(np.var(local_grad) + 1e-9)
    sigma_pos = float(np.clip((1.0 / np.sqrt(local_curv)) * (1.0 + 1.0 / max(contrast, 1.0)), 0.12, 2.75))

    plateau_span = max(1, peak_idx)
    plateau_ratio = float(np.mean((prof_work[:plateau_span] >= plateau_val_thr) & (np.abs(grad_work[:plateau_span]) <= plateau_grad_thr))) if plateau_span > 0 else 0.0
    fit_quality = float((score[peak_local] * max(contrast, 1.0)) / max(sigma_pos, 1e-6))
    edge_level = float(np.clip((prof_work[peak_idx] - bg_val) / (dyn + 1e-9), 0.0, 1.0))

    if plateau_ratio > 0.92 and edge_level > 0.72:
        return None

    return {
        "point": edge_pt.astype(np.float64),
        "sigma": sigma_pos,
        "peak_grad": peak_mag,
        "contrast": contrast,
        "plateau_ratio": plateau_ratio,
        "fit_quality": fit_quality,
        "search_half_len": int(search_half_len),
        "used_profile_direction": "forward" if sign_flip > 0 else "reverse",
        "edge_level": edge_level,
        "edge_offset": float(d_sub_work),
        "plateau_end_offset": float(ds_work[min(plateau_end, len(ds_work) - 1)]),
    }

def _plane_basis_from_normal(n_c):
    n = np.asarray(n_c, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    anchor = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(n, anchor)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-12)
    return n, e1, e2




def extract_hardcore_contours(img, num_lights=4):
    """重写版：
    1) 不再依赖固定阈值 215 / 99.55 分位；
    2) 在 ROI 内融合亮度、绿色占比、局部对比度；
    3) 用 Otsu 自适应二值化；
    4) 通过区域统计排序，而不是单纯面积前 num_lights。
    """
    if img is None:
        return []
    if img.ndim != 3 or img.shape[2] != 3:
        return []

    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    valid_mask = (b + g + r) > 0.0
    if np.count_nonzero(valid_mask) < 16:
        return []

    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    local_contrast = np.clip(gray - blur, 0.0, None)
    green_ratio = g / (r + g + b + 1.0)

    gray_n = _robust_rescale(gray, valid_mask)
    ratio_n = _robust_rescale(green_ratio, valid_mask)
    contrast_n = _robust_rescale(local_contrast, valid_mask)

    score = 0.58 * gray_n + 0.24 * ratio_n + 0.18 * contrast_n
    score_u8 = np.zeros_like(gray, dtype=np.uint8)
    score_u8[valid_mask] = np.clip(np.round(score[valid_mask] * 255.0), 0, 255).astype(np.uint8)

    vals = score_u8[valid_mask]
    if vals.size < 8:
        return []
    otsu_thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw = cv2.threshold(score_u8, int(otsu_thr), 255, cv2.THRESH_BINARY)

    ksz = int(np.clip(round(np.sqrt(np.count_nonzero(valid_mask)) / 180.0), 3, 7))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    valid_area = float(np.count_nonzero(valid_mask))
    min_area = max(8.0, 2.0e-5 * valid_area)
    candidates = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or len(c) < 5:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 3 or h < 3:
            continue

        region_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [c], -1, 255, thickness=-1)
        pix = score[region_mask > 0]
        gray_pix = gray[region_mask > 0]
        if pix.size == 0:
            continue
        mean_score = float(np.mean(pix))
        peak_score = float(np.max(pix))
        mean_gray = float(np.mean(gray_pix))
        solidity = area / max(1.0, float(w * h))
        shape_penalty = abs(np.log((w + 1.0) / (h + 1.0)))
        rank_score = (
            0.42 * mean_score +
            0.18 * peak_score +
            0.20 * np.clip(area / max(1.0, valid_area * 0.02), 0.0, 1.0) +
            0.12 * np.clip(mean_gray / 255.0, 0.0, 1.0) +
            0.08 * np.clip(solidity, 0.0, 1.0)
        ) - 0.06 * shape_penalty
        candidates.append((rank_score, area, c))

    if not candidates:
        return []
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, _, c in candidates[:num_lights]]

def fit_ellipse_from_contour(contour):
    if contour is None or contour.shape[0] < 6:
        return None
    try:
        return cv2.fitEllipse(contour)
    except cv2.error:
        return None


def sample_ellipse_points(ellipse, num_samples=220):
    (cx, cy), (MA, ma), angle_deg = ellipse
    a, b = MA * 0.5, ma * 0.5
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    ts = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    x = cx + a * np.cos(ts) * c - b * np.sin(ts) * s
    y = cy + a * np.cos(ts) * s + b * np.sin(ts) * c
    return np.column_stack([x, y])


def crop_local(gray, ellipse, pad=10):
    (cx, cy), (MA, ma), _ = ellipse
    r = int(0.5 * max(MA, ma) + pad)
    x0 = max(int(cx) - r, 0)
    y0 = max(int(cy) - r, 0)
    x1 = min(int(cx) + r, gray.shape[1] - 1)
    y1 = min(int(cy) + r, gray.shape[0] - 1)
    roi_crop = gray[y0:y1 + 1, x0:x1 + 1]
    e_local = ((cx - x0, cy - y0), (MA, ma), ellipse[2])
    return roi_crop, (x0, y0), e_local


def robust_trimmed_ellipse_center(pts, old_center, trim_ratio=0.18):
    if pts is None or len(pts) < 12:
        return np.array(old_center, dtype=np.float64)
    try:
        e1 = cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return np.array(old_center, dtype=np.float64)

    (cx, cy), (MA, ma), ang = e1
    a, b = max(MA * 0.5, 1e-6), max(ma * 0.5, 1e-6)
    th = np.deg2rad(ang)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy])) @ R
    resid = np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0)
    k = max(12, int((1.0 - trim_ratio) * len(resid)))
    keep = np.argpartition(resid, k - 1)[:k]
    pts2 = pts[keep]
    try:
        e2 = cv2.fitEllipse(pts2.astype(np.float32).reshape(-1, 1, 2))
        return np.array(e2[0], dtype=np.float64)
    except cv2.error:
        return np.array(e1[0], dtype=np.float64)


def _fit_ellipse_safe_from_pts(pts):
    if pts is None or len(pts) < 6:
        return None
    try:
        return cv2.fitEllipse(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None



def fit_core_ellipse_from_multilevel(gray, contour, ellipse):
    """重写版：
    在目标轮廓内部基于强度分位点生成 level set，联合考虑中心稳定性、拟合残差、
    饱和覆盖率、平顶核心比例、边界梯度与外扩速率选择边界，降低强曝光下选中饱和核或 bloom 外扩轮廓的概率。
    """
    if ellipse is None or contour is None:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "invalid_input"}

    roi_gray, (ox, oy), _ = crop_local(gray, ellipse, pad=max(ROI_PAD, 14))
    if roi_gray.size == 0:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "empty_roi"}

    contour_local = contour.reshape(-1, 2).astype(np.int32).copy()
    contour_local[:, 0] -= ox
    contour_local[:, 1] -= oy

    mask = np.zeros(roi_gray.shape, dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [contour_local.reshape(-1, 1, 2)], 255)
    except Exception:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "mask_fail"}

    vals = roi_gray[mask > 0]
    if vals.size < 24:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "too_few_pixels"}

    q_low = float(np.percentile(vals, 50.0))
    q_high = float(np.percentile(vals, 98.5))
    sat_thr = float(np.percentile(vals, 99.5))
    dyn = float(q_high - q_low)
    if dyn < 6.0:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "low_dynamic_range"}

    raw_center = np.array(ellipse[0], dtype=np.float64)
    raw_axes = np.array(ellipse[1], dtype=np.float64)
    raw_area = max(1.0, float(cv2.contourArea(contour)))
    global_sat_ratio = float(np.mean(vals >= sat_thr)) if vals.size else 0.0

    levels = np.unique(np.round(np.linspace(q_low + 0.10 * dyn, q_high - 0.04 * dyn, 11)).astype(np.int32))
    if levels.size < 3:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "too_few_levels"}

    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    grad_ref = float(np.percentile(gmag[mask > 0], 90.0) + 1e-9)

    candidates = []
    ksz = int(np.clip(round(min(raw_axes) / 16.0), 3, 7))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    for lvl_idx, thr in enumerate(levels):
        bw = np.zeros_like(roi_gray, dtype=np.uint8)
        bw[(roi_gray >= thr) & (mask > 0)] = 255
        if np.count_nonzero(bw) < 12:
            continue
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue

        best_local = None
        best_local_score = np.inf
        for cc in cs:
            if len(cc) < 6:
                continue
            area = float(cv2.contourArea(cc))
            if area < 10.0:
                continue
            e_try = fit_ellipse_from_contour(cc)
            if e_try is None:
                continue
            ctr = np.array([e_try[0][0] + ox, e_try[0][1] + oy], dtype=np.float64)
            axes = np.array(e_try[1], dtype=np.float64)
            drift = float(np.linalg.norm(ctr - raw_center))
            axis_delta = float(np.linalg.norm((axes - raw_axes) / (raw_axes + 1e-6)))
            area_ratio = area / raw_area
            resid = _ellipse_contour_residual(e_try, cc.reshape(-1, 2).astype(np.float64))

            cc_mask = np.zeros_like(roi_gray, dtype=np.uint8)
            cv2.drawContours(cc_mask, [cc], -1, 255, thickness=-1)
            cc_vals = roi_gray[cc_mask > 0]
            if cc_vals.size == 0:
                continue
            sat_ratio = float(np.mean(cc_vals >= sat_thr))
            plateau_core_ratio = float(np.mean(cc_vals >= (q_high - 0.08 * dyn)))
            halo_low_ratio = float(np.mean(cc_vals <= (q_low + 0.18 * dyn)))
            level_norm = float(np.clip((thr - q_low) / (dyn + 1e-9), 0.0, 1.0))
            level_penalty = abs(level_norm - 0.70)
            cc_pts = cc.reshape(-1, 2).astype(np.int32)
            cc_pts[:, 0] = np.clip(cc_pts[:, 0], 0, roi_gray.shape[1] - 1)
            cc_pts[:, 1] = np.clip(cc_pts[:, 1], 0, roi_gray.shape[0] - 1)
            boundary_grad = float(np.mean(gmag[cc_pts[:, 1], cc_pts[:, 0]]) / grad_ref)
            boundary_grad_penalty = float(np.clip(1.0 - boundary_grad, 0.0, 1.5))

            local_score = (
                drift +
                3.6 * resid +
                0.70 * axis_delta +
                0.30 * abs(np.log(max(area_ratio, 1e-6))) +
                0.95 * sat_ratio +
                0.95 * plateau_core_ratio +
                0.70 * halo_low_ratio +
                0.55 * level_penalty +
                0.50 * boundary_grad_penalty
            )
            if local_score < best_local_score:
                best_local_score = local_score
                best_local = (cc, e_try, area, ctr, axes, drift, axis_delta, resid, sat_ratio, plateau_core_ratio, halo_low_ratio, boundary_grad, level_norm)

        if best_local is None:
            continue
        cc, e_try, area, ctr, axes, drift, axis_delta, resid, sat_ratio, plateau_core_ratio, halo_low_ratio, boundary_grad, level_norm = best_local
        candidates.append({
            "level_idx": int(lvl_idx),
            "thr": int(thr),
            "contour_local": cc.copy(),
            "ellipse_local": e_try,
            "area": float(area),
            "center": ctr,
            "axes": axes,
            "drift": float(drift),
            "axis_delta": float(axis_delta),
            "resid": float(resid),
            "sat_ratio": float(sat_ratio),
            "plateau_core_ratio": float(plateau_core_ratio),
            "halo_low_ratio": float(halo_low_ratio),
            "boundary_grad": float(boundary_grad),
            "level_norm": float(level_norm),
        })

    if len(candidates) < 2:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "no_stable_candidate"}

    candidates.sort(key=lambda d: d["level_idx"])
    area_gains = []
    for i, cand in enumerate(candidates):
        prev_c = candidates[i - 1] if i > 0 else None
        next_c = candidates[i + 1] if i + 1 < len(candidates) else None
        stability = 0.0
        support = 0
        for nb in (prev_c, next_c):
            if nb is None:
                continue
            stability += float(np.linalg.norm(cand["center"] - nb["center"]))
            stability += 0.55 * float(np.linalg.norm((cand["axes"] - nb["axes"]) / (raw_axes + 1e-6)))
            stability += 2.4 * abs(cand["resid"] - nb["resid"])
            support += 1
        if support > 0:
            stability /= support
        else:
            stability = cand["drift"] + 2.4 * cand["resid"]

        outward_growth = 0.0
        if next_c is not None:
            outward_growth = max(0.0, np.log(max(next_c["area"], 1e-6) / max(cand["area"], 1e-6)))
        inward_growth = 0.0
        if prev_c is not None:
            inward_growth = max(0.0, np.log(max(cand["area"], 1e-6) / max(prev_c["area"], 1e-6)))
        bloom_expansion = float(outward_growth + 0.35 * inward_growth)
        area_gains.append(bloom_expansion)
        cand["stability"] = float(stability)
        cand["bloom_expansion"] = bloom_expansion

    expansion_arr = np.asarray(area_gains, dtype=np.float64)
    exp_norm = normalize_scores(expansion_arr) if expansion_arr.size else np.zeros((len(candidates),), dtype=np.float64)
    for cand, exp_penalty in zip(candidates, exp_norm.tolist()):
        high_exp_guard = float(1.0 + 6.0 * global_sat_ratio)
        cand["global_score"] = float(
            0.34 * cand["stability"] +
            0.18 * cand["drift"] +
            0.14 * cand["resid"] +
            0.12 * cand["plateau_core_ratio"] * high_exp_guard +
            0.08 * cand["halo_low_ratio"] * high_exp_guard +
            0.06 * exp_penalty +
            0.05 * abs(cand["level_norm"] - 0.70) -
            0.03 * cand["boundary_grad"]
        )

    chosen = min(candidates, key=lambda d: d["global_score"])
    cc_img = chosen["contour_local"].reshape(-1, 2).astype(np.float64) + np.array([ox, oy], dtype=np.float64)
    e_try = chosen["ellipse_local"]
    e_img = ((e_try[0][0] + ox, e_try[0][1] + oy), e_try[1], e_try[2])

    dbg = {
        "enabled": True,
        "used": True,
        "reason": "stable_multilevel_contour_with_exposure_guard_v2",
        "num_levels": int(levels.size),
        "num_candidates": int(len(candidates)),
        "chosen_thr": int(chosen["thr"]),
        "chosen_level_idx": int(chosen["level_idx"]),
        "chosen_drift": float(chosen["drift"]),
        "chosen_resid": float(chosen["resid"]),
        "chosen_stability": float(chosen["stability"]),
        "chosen_sat_ratio": float(chosen["sat_ratio"]),
        "chosen_plateau_core_ratio": float(chosen["plateau_core_ratio"]),
        "chosen_halo_low_ratio": float(chosen["halo_low_ratio"]),
        "chosen_boundary_grad": float(chosen["boundary_grad"]),
        "chosen_bloom_expansion": float(chosen["bloom_expansion"]),
        "chosen_level_norm": float(chosen["level_norm"]),
        "global_saturation_ratio": float(global_sat_ratio),
        "local_saturation_threshold": sat_thr,
    }
    return e_img, cc_img.reshape(-1, 1, 2).astype(np.int32), dbg

def choose_measurement_representation(
    gray,
    contour,
    ellipse,
    K=None,
    plane_normal_camera=None,
    plane_distance=None,
    selector_config=None,
):
    """统一的测量表示选择入口。

    优先调用独立模块 ellipse_edge_geometry_selector.py 中的
    几何一致候选边缘选择器；如果模块未导入成功，则退回到
    本文件原有的 fit_core_ellipse_from_multilevel(...) 逻辑。
    """
    if choose_measurement_representation_geometry is not None:
        try:
            cfg = selector_config if selector_config is not None else (
                GeometryEdgeSelectorConfig() if GeometryEdgeSelectorConfig is not None else None
            )
            return choose_measurement_representation_geometry(
                gray=gray,
                contour=contour,
                ellipse=ellipse,
                K=K,
                plane_normal_camera=plane_normal_camera,
                plane_distance=plane_distance,
                config=cfg,
            )
        except Exception as e:
            # 出现异常时退回原有方案，保证主程序仍可运行
            work_ellipse, work_contour, rep_dbg = fit_core_ellipse_from_multilevel(gray, contour, ellipse)
            rep_dbg = dict(rep_dbg) if isinstance(rep_dbg, dict) else {}
            rep_dbg["selector_fallback"] = True
            rep_dbg["selector_fallback_reason"] = f"geometry_selector_exception: {type(e).__name__}: {e}"
            return work_ellipse, work_contour, rep_dbg

    work_ellipse, work_contour, rep_dbg = fit_core_ellipse_from_multilevel(gray, contour, ellipse)
    rep_dbg = dict(rep_dbg) if isinstance(rep_dbg, dict) else {}
    rep_dbg["selector_fallback"] = True
    rep_dbg["selector_fallback_reason"] = "geometry_selector_module_unavailable"
    return work_ellipse, work_contour, rep_dbg




def project_pixels_to_plane_filtered(points_uv, K, n_c, d_c, den_eps=0.0, keep_ratio=0.0, proj_ctx=None):
    """重写版：
    - 不再依赖固定 den_eps / keep_ratio；
    - 根据当前样本射线与平面的交角分布自动计算病态过滤阈值；
    - 不做硬裁前 30%，而是输出所有稳定点与连续权重。
    返回接口仍为：pts3d, weights, dbg。
    """
    points_uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    if points_uv.shape[0] < 6:
        return None, None, {"valid_ratio": 0.0, "num_valid": 0, "num_total": int(points_uv.shape[0])}

    if proj_ctx is None:
        proj_ctx = _prepare_projection_context(K, n_c, d_c)
    Kinv = np.asarray(proj_ctx["Kinv"], dtype=np.float64).reshape(3, 3)
    n_vec = np.asarray(proj_ctx["n_c"], dtype=np.float64).reshape(3)
    d_val = float(proj_ctx["d_c"])

    pts_h = np.column_stack([points_uv, np.ones((points_uv.shape[0], 1), dtype=np.float64)]).T
    dirs_c = Kinv @ pts_h
    den = (n_vec.reshape(1, 3) @ dirs_c).reshape(-1)
    den_abs = np.abs(den)

    q10 = float(np.percentile(den_abs, 10.0))
    q25 = float(np.percentile(den_abs, 25.0))
    auto_den_eps = max(1e-6, 0.5 * q10 + 0.15 * q25)
    if den_eps is not None and float(den_eps) > 0:
        auto_den_eps = float(min(max(auto_den_eps, 0.5 * float(den_eps)), 1.5 * float(den_eps)))

    valid_mask = den_abs >= auto_den_eps
    if np.count_nonzero(valid_mask) < 8:
        valid_mask = den_abs >= max(1e-6, 0.65 * auto_den_eps)
    if np.count_nonzero(valid_mask) < 6:
        return None, None, {
            "valid_ratio": 0.0,
            "num_valid": int(np.count_nonzero(valid_mask)),
            "num_total": int(points_uv.shape[0]),
            "auto_den_eps": float(auto_den_eps),
        }

    dirs_valid = dirs_c[:, valid_mask]
    den_valid = den[valid_mask]
    s_vals = -d_val / den_valid
    pts3d = (dirs_valid * s_vals).T

    med_s = float(np.median(s_vals))
    mad_s = float(np.median(np.abs(s_vals - med_s)) + 1e-9)
    depth_score = 1.0 / (1.0 + ((s_vals - med_s) / (2.5 * mad_s + 1e-9)) ** 2)
    angle_score = den_abs[valid_mask] / (np.max(den_abs[valid_mask]) + 1e-9)
    weights = np.sqrt(np.clip(depth_score, 1e-6, None) * np.clip(angle_score, 1e-6, None))

    keep = weights >= max(0.15, 0.35 * float(np.median(weights)))
    if np.count_nonzero(keep) >= 6:
        pts3d = pts3d[keep]
        weights = weights[keep]

    dbg = {
        "valid_ratio": float(np.count_nonzero(valid_mask) / max(1, points_uv.shape[0])),
        "num_valid": int(np.count_nonzero(valid_mask)),
        "num_total": int(points_uv.shape[0]),
        "num_after_depth_filter": int(len(pts3d)),
        "auto_den_eps": float(auto_den_eps),
        "depth_median": float(med_s),
        "depth_mad": float(mad_s),
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_std": float(np.std(weights)) if len(weights) else 0.0,
    }
    return pts3d, weights.astype(np.float64), dbg


def project_center_to_image(center_3d, K):
    x = K @ center_3d
    return np.array([x[0] / x[2], x[1] / x[2]], dtype=np.float64)


def gray_centroid_in_contour(gray, contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.uint8)
    contour_local = contour.copy()
    contour_local[:, :, 0] -= x
    contour_local[:, :, 1] -= y
    cv2.drawContours(mask, [contour_local], -1, 255, thickness=-1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    wgt = roi[ys, xs].astype(np.float64) + 1.0
    cx = np.sum(xs * wgt) / np.sum(wgt)
    cy = np.sum(ys * wgt) / np.sum(wgt)
    return np.array([x + cx, y + cy], dtype=np.float64)


def _auto_miss_penalty(preds, refs):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    refs = np.asarray(refs, dtype=np.float64).reshape(-1, 2)
    if len(preds) == 0 or len(refs) == 0:
        return 50.0
    dist = np.linalg.norm(preds[:, None, :] - refs[None, :, :], axis=2).reshape(-1)
    if dist.size == 0:
        return 50.0
    return float(max(8.0, np.percentile(dist, 90.0)))


def match_points_hungarian(preds, refs, miss_penalty=None):
    """小规模精确匹配器。当前任务最多 4 点，直接枚举 permutation 求 padded cost 最优解。"""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    refs = np.asarray(refs, dtype=np.float64).reshape(-1, 2)
    n_pred, n_ref = len(preds), len(refs)
    n = max(n_pred, n_ref)
    if n == 0:
        penalty = float(miss_penalty) if miss_penalty is not None else 0.0
        return {
            "assignment": [],
            "matched_pairs": [],
            "pair_costs": [],
            "eval_costs": [],
            "mean_cost": np.nan,
            "median_cost": np.nan,
            "miss_penalty": penalty,
            "num_missed": 0,
            "num_false": 0,
        }

    if miss_penalty is None:
        miss_penalty = _auto_miss_penalty(preds, refs)
    miss_penalty = float(miss_penalty)

    cost = np.full((n, n), miss_penalty, dtype=np.float64)
    if n_pred > 0 and n_ref > 0:
        dist = np.linalg.norm(preds[:, None, :] - refs[None, :, :], axis=2)
        cost[:n_pred, :n_ref] = dist

    best_cost = np.inf
    best_perm = None
    for perm in permutations(range(n), n):
        c = 0.0
        for i in range(n):
            c += cost[i, perm[i]]
        if c < best_cost:
            best_cost = c
            best_perm = list(perm)

    assignment = []
    matched_pairs = []
    pair_costs = []
    matched_real_refs = set()
    num_false = 0
    for pred_idx in range(n_pred):
        ref_idx = best_perm[pred_idx]
        if ref_idx < n_ref:
            c = float(cost[pred_idx, ref_idx])
            assignment.append((pred_idx, ref_idx, c))
            matched_pairs.append((pred_idx, ref_idx))
            pair_costs.append(c)
            matched_real_refs.add(ref_idx)
        else:
            assignment.append((pred_idx, None, miss_penalty))
            pair_costs.append(miss_penalty)
            num_false += 1

    num_missed = max(0, n_ref - len(matched_real_refs))
    eval_costs = pair_costs + [miss_penalty] * num_missed

    return {
        "assignment": assignment,
        "matched_pairs": matched_pairs,
        "pair_costs": pair_costs,
        "eval_costs": eval_costs,
        "mean_cost": float(np.mean(eval_costs)) if eval_costs else np.nan,
        "median_cost": float(np.median(eval_costs)) if eval_costs else np.nan,
        "miss_penalty": miss_penalty,
        "num_missed": int(num_missed),
        "num_false": int(num_false),
    }


def evaluate_predictions_with_miss_penalty(preds, refs, name="", miss_penalty=None, verbose=True):
    match = match_points_hungarian(preds, refs, miss_penalty=miss_penalty)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    refs = np.asarray(refs, dtype=np.float64).reshape(-1, 2)

    aligned_refs = []
    perm = []
    pair_costs = []
    for pred_idx, ref_idx, c in match["assignment"]:
        if pred_idx is None or pred_idx >= len(preds):
            continue
        if ref_idx is None or ref_idx >= len(refs):
            aligned_refs.append([np.nan, np.nan])
            perm.append(None)
            pair_costs.append(float(match["miss_penalty"]))
        else:
            aligned_refs.append(refs[ref_idx].tolist())
            perm.append(int(ref_idx))
            pair_costs.append(float(c))

    aligned_refs = np.asarray(aligned_refs, dtype=np.float64) if aligned_refs else None
    errs = np.asarray(pair_costs, dtype=np.float64) if pair_costs else None
    mean_err = float(match["mean_cost"])

    if verbose:
        miss = match["num_missed"]
        false = match["num_false"]
        print(f"[{name}] 平均代价: {mean_err:.3f} px | 中位数: {match['median_cost']:.3f} px | 漏检: {miss} | 误检: {false} | 明细: {np.round(errs, 3) if errs is not None else []}")

    return mean_err, aligned_refs, perm, match


def find_best_matching_permutation(preds, trues):
    match = match_points_hungarian(preds, trues, miss_penalty=None)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    perm = [None] * len(preds)
    for pred_idx, ref_idx, _ in match["assignment"]:
        if pred_idx is not None and pred_idx < len(preds):
            perm[pred_idx] = ref_idx
    return perm



def estimate_ellipse_center_covariance(ellipse, contour=None):
    if ellipse is None:
        return np.eye(2, dtype=np.float64) * 25.0
    area = max(1.0, ellipse_area_from_fit(ellipse))
    ecc = max(1.0, ellipse_ecc_ratio_from_fit(ellipse))
    n_pts = int(len(contour.reshape(-1, 2))) if contour is not None else 24
    resid = _ellipse_contour_residual(ellipse, contour.reshape(-1, 2).astype(np.float64)) if contour is not None and len(contour) >= 6 else 0.12

    (_, _), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, 1.0)
    b = max(float(ma) * 0.5, 1.0)
    sigma_major = max(0.20, a * (resid + 0.02) / np.sqrt(max(n_pts, 6)))
    sigma_minor = max(0.20, b * (resid + 0.02) / np.sqrt(max(n_pts, 6)))
    sigma_major *= np.sqrt(ecc)

    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    cov_local = np.diag([sigma_major ** 2, sigma_minor ** 2])
    cov = R @ cov_local @ R.T
    cov += 1e-9 * np.eye(2)
    return cov.astype(np.float64)


def estimate_geometry_center_covariance(point_dbg, ellipse=None):
    if point_dbg is None:
        return np.eye(2, dtype=np.float64) * 36.0
    fit_resid = point_dbg.get("fit_residual_median", point_dbg.get("fit_residual_mean", None))
    valid_ratio = float(point_dbg.get("valid_ratio", 0.0) or 0.0)
    used_points = int(point_dbg.get("fit_used_points", point_dbg.get("num_after_depth_filter", 0)) or 0)
    cov2d = point_dbg.get("fit_center_cov_2d", None)
    if cov2d is not None:
        cov = np.asarray(cov2d, dtype=np.float64)
        if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
            scale = 1.0 / max(valid_ratio, 0.10)
            return cov * scale

    area = max(1.0, ellipse_area_from_fit(ellipse)) if ellipse is not None else 400.0
    resid = float(fit_resid) if fit_resid is not None else 4.0
    sigma = max(0.35, resid * np.sqrt(1.0 / max(valid_ratio, 0.10)) * np.sqrt(18.0 / max(used_points, 6)))
    sigma *= np.clip(600.0 / area, 0.65, 2.5) ** 0.25
    return np.eye(2, dtype=np.float64) * (sigma ** 2)


def fuse_center_estimates(ellipse_center, ellipse_cov, geo_center=None, geo_cov=None, pose_center=None, pose_cov=None):
    terms = []
    labels = []
    if ellipse_center is not None and ellipse_cov is not None:
        terms.append((np.asarray(ellipse_center, dtype=np.float64).reshape(2), np.asarray(ellipse_cov, dtype=np.float64).reshape(2, 2)))
        labels.append("ellipse")
    if geo_center is not None and geo_cov is not None:
        terms.append((np.asarray(geo_center, dtype=np.float64).reshape(2), np.asarray(geo_cov, dtype=np.float64).reshape(2, 2)))
        labels.append("geometry")
    if pose_center is not None and pose_cov is not None:
        terms.append((np.asarray(pose_center, dtype=np.float64).reshape(2), np.asarray(pose_cov, dtype=np.float64).reshape(2, 2)))
        labels.append("pose")

    if not terms:
        return None, None, {"used_terms": []}

    P = np.zeros((2, 2), dtype=np.float64)
    y = np.zeros((2,), dtype=np.float64)
    used_terms = []
    source_precision_trace = {}
    for label, (mu, cov) in zip(labels, terms):
        cov = 0.5 * (cov + cov.T)
        cov = cov + 1e-9 * np.eye(2)
        try:
            prec = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        P += prec
        y += prec @ mu
        used_terms.append(label)
        source_precision_trace[label] = float(np.trace(prec))

    if not used_terms:
        return None, None, {"used_terms": []}

    cov_f = np.linalg.inv(P + 1e-9 * np.eye(2))
    mu_f = cov_f @ y
    dbg = {
        "used_terms": used_terms,
        "cov_trace": float(np.trace(cov_f)),
        "cov_det": float(np.linalg.det(cov_f)),
        "source_precision_trace": source_precision_trace,
    }
    return mu_f.astype(np.float64), cov_f.astype(np.float64), dbg



def solve_pose_chessboard_only(gray_chess, K):
    ok, corners = cv2.findChessboardCorners(gray_chess, pattern_size)
    if not ok or corners is None:
        raise RuntimeError("棋盘格角点检测失败")
    corners_sub = cv2.cornerSubPix(
        gray_chess, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    ok, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, np.zeros(5))
    if not ok:
        raise RuntimeError("初始 solvePnP 失败")
    return objp, corners_sub, rvec, tvec


def compute_camera_depths(R, t, points_w):
    pts_c = (R @ points_w.T + t.reshape(3, 1)).T
    return pts_c[:, 2].astype(np.float64)






def ellipse_area_from_fit(ellipse):
    if ellipse is None:
        return 0.0
    (_, _), (MA, ma), _ = ellipse
    return float(np.pi * 0.5 * MA * 0.5 * ma)


def ellipse_ecc_ratio_from_fit(ellipse):
    if ellipse is None:
        return 1.0
    (_, _), (MA, ma), _ = ellipse
    mn = max(1e-9, min(MA, ma))
    mx = max(MA, ma)
    return float(mx / mn)


def normalize_scores(x, invert=False):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < 1e-12:
        out = np.ones_like(x) if invert else np.zeros_like(x)
    else:
        out = (x - xmin) / (xmax - xmin + 1e-12)
        if invert:
            out = 1.0 - out
    return np.clip(out, 0.0, 1.0)




def estimate_pose_reference_uncertainty(objp, corners_sub, rvec, tvec, K):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))
    proj = proj.reshape(-1, 2).astype(np.float64)
    corners = np.asarray(corners_sub, dtype=np.float64).reshape(-1, 2)
    if len(proj) == 0 or len(corners) == 0:
        return {
            "reprojection_errors_px": [],
            "reprojection_rms_px": None,
            "reprojection_median_px": None,
            "reprojection_p90_px": None,
            "pose_sigma_px": 1.5,
        }
    errs = np.linalg.norm(proj - corners, axis=1)
    rms = float(np.sqrt(np.mean(errs ** 2)))
    med = float(np.median(errs))
    p90 = float(np.percentile(errs, 90.0))
    sigma = float(np.clip(max(rms, 0.5 * (med + p90)), 0.15, 3.0))
    return {
        "reprojection_errors_px": errs.astype(np.float64).tolist(),
        "reprojection_rms_px": rms,
        "reprojection_median_px": med,
        "reprojection_p90_px": p90,
        "pose_sigma_px": sigma,
    }


def _finite_point_errors(points, aligned_points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) if points is not None else np.zeros((0, 2), dtype=np.float64)
    ali = np.asarray(aligned_points, dtype=np.float64).reshape(-1, 2) if aligned_points is not None else np.zeros((0, 2), dtype=np.float64)
    if len(pts) == 0 or len(ali) == 0:
        return []
    n = min(len(pts), len(ali))
    errs = []
    for i in range(n):
        if np.all(np.isfinite(pts[i])) and np.all(np.isfinite(ali[i])):
            errs.append(float(np.linalg.norm(pts[i] - ali[i])))
        else:
            errs.append(None)
    return errs


def _error_stats_from_match(match):
    if not isinstance(match, dict):
        return {"mean_cost": None, "median_cost": None, "p90_cost": None, "p95_cost": None, "num_missed": None, "num_false": None}
    eval_costs = np.asarray(match.get("eval_costs", []), dtype=np.float64)
    finite = eval_costs[np.isfinite(eval_costs)]
    if finite.size == 0:
        return {
            "mean_cost": None,
            "median_cost": None,
            "p90_cost": None,
            "p95_cost": None,
            "num_missed": int(match.get("num_missed", 0)),
            "num_false": int(match.get("num_false", 0)),
            "miss_penalty": float(match.get("miss_penalty", 0.0)) if match.get("miss_penalty", None) is not None else None,
        }
    return {
        "mean_cost": float(np.mean(finite)),
        "median_cost": float(np.median(finite)),
        "p90_cost": float(np.percentile(finite, 90.0)),
        "p95_cost": float(np.percentile(finite, 95.0)),
        "num_missed": int(match.get("num_missed", 0)),
        "num_false": int(match.get("num_false", 0)),
        "miss_penalty": float(match.get("miss_penalty", 0.0)) if match.get("miss_penalty", None) is not None else None,
    }


def _method_result_package(points, aligned_points, perm, match):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) if points is not None else np.zeros((0, 2), dtype=np.float64)
    ali = np.asarray(aligned_points, dtype=np.float64).reshape(-1, 2) if aligned_points is not None else np.zeros((0, 2), dtype=np.float64)
    errs = _finite_point_errors(pts, ali)
    stats = _error_stats_from_match(match)
    return {
        "points": pts.tolist(),
        "aligned_points": ali.tolist() if aligned_points is not None else [],
        "matching_order": perm if perm is not None else [],
        "errors_per_point": errs,
        **stats,
        "assignment": match.get("assignment", []) if isinstance(match, dict) else [],
        "matched_pairs": match.get("matched_pairs", []) if isinstance(match, dict) else [],
        "pair_costs": match.get("pair_costs", []) if isinstance(match, dict) else [],
        "complete_4of4": bool(len(pts) == 4 and int(stats["num_missed"] or 0) == 0 and int(stats["num_false"] or 0) == 0),
    }



def _as_finite_point_list(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    out = []
    for p in arr:
        if np.all(np.isfinite(p)):
            out.append([float(p[0]), float(p[1])])
    return out


def _as_finite_point_or_none(point):
    try:
        p = np.asarray(point, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(p)):
            return [float(p[0]), float(p[1])]
    except Exception:
        pass
    return None


def _pose_eval_method_display_name(raw_method):
    mapping = {
        "bounding_box_centroid": "BoundingBox",
        "gray_centroid": "Gray",
        "weighted_gray_centroid": "WeightedGray",
        "contour_circle_fitting": "Circle",
        "ellipse_fitting": "DirectEllipse",
        "step4_work_ellipse": "Step4WorkEllipse",
        "ellipse_backprojection_correction": "EllipseBackProj",
        "homography_rectified_circle": "HomographyRectified",
        "geometry_correction": "PCCL",
        "geometry_diagnostic_copy": "GeometryDiagnostic",
    }
    return mapping.get(str(raw_method), str(raw_method))


def _build_pose_eval_inputs_for_downstream_pose(results):
    """Build compact per-image data needed by downstream pose-estimation evaluation.

    The full per-image *_eval.json already contains detailed measurements, but
    batch_summary.json previously stored only mean errors. This function extracts
    the minimum complete pose-evaluation inputs and stores them into each batch
    summary item:
        - method-wise 2D center observations with lamp_id correspondence
        - reference physical-center projections
        - 3D lamp center coordinates
        - chessboard-assisted reference pose
        - camera matrix and plane information

    The lamp_id is obtained from the matching_order produced by the existing
    Hungarian matching evaluation, so 2D image centers and 3D physical lamp
    points can be paired directly for PnP/IPPE/Homography pose estimation.
    """
    if not isinstance(results, dict):
        return {}

    img_info = results.get("image_info", {}) or {}
    input_calib = results.get("input_and_calibration", {}) or {}
    camera_pose = input_calib.get("camera_pose", {}) or {}
    method_results = results.get("method_results", {}) or {}

    object_points = (
        input_calib.get("ground_truth_points_3d_world")
        or results.get("ground_truth_points_3d_world")
        or []
    )
    reference_centers = (
        input_calib.get("pose_reference_points_2d_image")
        or results.get("pose_reference_points_2d_image")
        or results.get("ground_truth_points_2d_image")
        or []
    )

    # Camera matrix is available through the runtime projection context in pose_ctx.
    # It is stored here to make the batch summary self-contained enough for
    # later CSV extraction and pose evaluation.
    K = None
    try:
        proj_ctx = None
        # The projection context is not stored in the full results by default.
        # Therefore K may be absent here; the standalone camera intrinsics file
        # remains the primary source. This field is kept for compatibility.
        proj_ctx = input_calib.get("projection_context", None)
        if isinstance(proj_ctx, dict) and "K" in proj_ctx:
            K = np.asarray(proj_ctx["K"], dtype=np.float64).reshape(3, 3).tolist()
    except Exception:
        K = None

    center_observations = []
    for raw_method, mr in method_results.items():
        if not isinstance(mr, dict):
            continue
        pts = _as_finite_point_list(mr.get("points", []))
        matching_order = mr.get("matching_order", [])
        errors = mr.get("errors_per_point", [])
        complete = bool(mr.get("complete_4of4", False))

        for det_idx, uv in enumerate(pts):
            lamp_id = None
            if isinstance(matching_order, list) and det_idx < len(matching_order):
                try:
                    if matching_order[det_idx] is not None:
                        lamp_id = int(matching_order[det_idx])
                except Exception:
                    lamp_id = None

            err = None
            if isinstance(errors, list) and det_idx < len(errors):
                try:
                    if errors[det_idx] is not None:
                        err = float(errors[det_idx])
                except Exception:
                    err = None

            center_observations.append({
                "method": _pose_eval_method_display_name(raw_method),
                "raw_method": str(raw_method),
                "detected_index": int(det_idx),
                "lamp_id": lamp_id,
                "u": float(uv[0]),
                "v": float(uv[1]),
                "point_error_px": err,
                "complete_4of4": complete,
            })

    reference_center_rows = []
    for lamp_id, uv in enumerate(_as_finite_point_list(reference_centers)):
        reference_center_rows.append({
            "lamp_id": int(lamp_id),
            "u": float(uv[0]),
            "v": float(uv[1]),
        })

    object_point_rows = []
    try:
        obj_arr = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        for lamp_id, X in enumerate(obj_arr):
            if np.all(np.isfinite(X)):
                object_point_rows.append({
                    "lamp_id": int(lamp_id),
                    "X": float(X[0]),
                    "Y": float(X[1]),
                    "Z": float(X[2]),
                })
    except Exception:
        object_point_rows = []

    reference_pose = {
        "rvec": camera_pose.get("rvec_initial"),
        "tvec": camera_pose.get("tvec_initial"),
        "R": camera_pose.get("R_matrix"),
        "t": camera_pose.get("t_vector"),
        "plane_normal_camera": camera_pose.get("plane_normal_camera"),
        "plane_distance": camera_pose.get("plane_distance"),
        "reference_type": camera_pose.get("pose_source", "light_only"),
    }

    return {
        "schema_version": "pose_eval_inputs_v1",
        "image_id": img_info.get("light_image"),
        "chessboard_image": img_info.get("chessboard_image"),
        "center_observations": center_observations,
        "reference_centers": reference_center_rows,
        "object_points_3d": object_point_rows,
        "reference_pose": reference_pose,
        "camera_matrix": K,
        "notes": (
            "lamp_id is the matched reference lamp index from the existing "
            "center-localization evaluation. Use center_observations grouped "
            "by method and lamp_id as 2D-3D correspondences for downstream "
            "pose estimation."
        ),
    }


def _covariance_to_ellipse(cov, nsig=2.4477):
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    axes = (float(nsig * np.sqrt(vals[0])), float(nsig * np.sqrt(vals[1])))
    return axes, angle


def _draw_covariance_ellipse(vis, center, cov, color, thickness=1):
    if center is None or cov is None:
        return
    center = np.asarray(center, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(center)):
        return
    params = _covariance_to_ellipse(cov)
    if params is None:
        return
    (a, b), ang = params
    if not np.isfinite(a) or not np.isfinite(b):
        return
    a = int(max(1, round(a)))
    b = int(max(1, round(b)))
    cv2.ellipse(
        vis,
        (int(round(center[0])), int(round(center[1]))),
        (a, b),
        ang,
        0.0,
        360.0,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def save_results_to_json(results_dict, file_path):
    def to_py(v):
        if isinstance(v, np.ndarray):
            return [to_py(x) for x in v.tolist()]
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return fv if np.isfinite(fv) else None
        if isinstance(v, tuple):
            return [to_py(x) for x in v]
        if isinstance(v, set):
            return [to_py(x) for x in sorted(v)]
        if isinstance(v, dict):
            return {str(k): to_py(val) for k, val in v.items()}
        if isinstance(v, list):
            return [to_py(x) for x in v]
        if isinstance(v, (bool, int, str)) or v is None:
            return v
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        return str(v)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(to_py(results_dict), f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已保存到: {file_path}")


def parse_pose_from_name(name):
    if not name:
        return {"group_prefix": None, "distance": None, "y_offset": None, "pitch": None, "exp_index": None}
    base = os.path.basename(name)
    m = re.search(r"(undist_d[-\dp]+_y[-\dp]+_p[-\dp]+)", base)
    group_prefix = m.group(1) if m else None
    pose = {"group_prefix": group_prefix, "distance": None, "y_offset": None, "pitch": None, "exp_index": None}
    if group_prefix is not None:
        m2 = re.search(r"d([-\dp]+)_y([-\dp]+)_p([-\dp]+)", group_prefix)
        if m2:
            pose["distance"] = float(m2.group(1).replace("p", "."))
            pose["y_offset"] = float(m2.group(2).replace("p", "."))
            pose["pitch"] = float(m2.group(3).replace("p", "."))
    m3 = re.search(r"exp_(\d+)", base)
    if m3:
        pose["exp_index"] = int(m3.group(1))
    return pose


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _stats_from_list(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def summarize_applicability_features(light_img_path, point_debug_pass1, priority_point_debug, conf_scores_pass1,
                                     risk_scores_pass1, ellipse_err, geo_err, geo_bias_err, detected_points):
    pose_info = parse_pose_from_name(light_img_path)
    conf_arr = np.asarray(conf_scores_pass1, dtype=np.float64).reshape(-1) if len(conf_scores_pass1) else np.zeros((0,), dtype=np.float64)
    risk_arr = np.asarray(risk_scores_pass1, dtype=np.float64).reshape(-1) if len(risk_scores_pass1) else np.zeros((0,), dtype=np.float64)

    valid_ratios = []
    fit_residuals = []
    fit_used_points = []
    ellipse_areas = []
    ecc_ratios = []
    brightness_means = []
    core_used_flags = []
    core_stability = []
    cov_traces = []
    edge_sigmas = []
    saturation_ratios = []
    plateau_ratios = []
    arc_coverages = []

    for dbg in point_debug_pass1:
        if not isinstance(dbg, dict):
            continue
        valid_ratios.append(_safe_float(dbg.get("valid_ratio")))
        fit_residuals.append(_safe_float(dbg.get("fit_residual_median", dbg.get("fit_residual_mean"))))
        fit_used_points.append(_safe_float(dbg.get("fit_used_points")))
        meas_mode = dbg.get("measurement_mode", {}) if isinstance(dbg.get("measurement_mode", {}), dict) else {}
        ellipse_areas.append(_safe_float(meas_mode.get("ellipse_area")))
        ecc_ratios.append(_safe_float(meas_mode.get("ecc_ratio")))
        brightness_means.append(_safe_float(dbg.get("brightness_mean", meas_mode.get("brightness_mean"))))
        edge_sigmas.append(_safe_float(dbg.get("edge_sigma_mean")))
        cov_traces.append(_safe_float(dbg.get("fit_center_cov_trace")))
        saturation_ratios.append(_safe_float(dbg.get("saturation_ratio")))
        plateau_ratios.append(_safe_float(dbg.get("edge_plateau_ratio_mean")))
        arc_coverages.append(_safe_float(dbg.get("effective_arc_coverage")))
        rep = dbg.get("representation", {}) if isinstance(dbg.get("representation", {}), dict) else {}
        core_used_flags.append(1.0 if bool(rep.get("used", False)) else 0.0)
        core_stability.append(_safe_float(rep.get("chosen_stability", rep.get("stability_drift", rep.get("global_center_spread")))))

    valid_stats = _stats_from_list(valid_ratios)
    fit_stats = _stats_from_list(fit_residuals)
    fit_used_stats = _stats_from_list(fit_used_points)
    area_stats = _stats_from_list(ellipse_areas)
    ecc_stats = _stats_from_list(ecc_ratios)
    bright_stats = _stats_from_list(brightness_means)
    core_stability_stats = _stats_from_list(core_stability)
    cov_trace_stats = _stats_from_list(cov_traces)
    edge_sigma_stats = _stats_from_list(edge_sigmas)
    sat_ratio_stats = _stats_from_list(saturation_ratios)
    plateau_ratio_stats = _stats_from_list(plateau_ratios)
    arc_coverage_stats = _stats_from_list(arc_coverages)

    conf_stats = _stats_from_list(conf_arr.tolist())
    risk_stats = _stats_from_list(risk_arr.tolist())

    geo_gain = None if ellipse_err is None or geo_err is None else float(ellipse_err - geo_err)
    geo_bias_gain = None if geo_err is None or geo_bias_err is None else float(geo_err - geo_bias_err)

    return {
        "group_prefix": pose_info.get("group_prefix"),
        "distance": pose_info.get("distance"),
        "y_offset": pose_info.get("y_offset"),
        "pitch": pose_info.get("pitch"),
        "exp_index": pose_info.get("exp_index"),
        "geo_gain": geo_gain,
        "geo_bias_gain": geo_bias_gain,
        "geo_effective": bool(geo_gain is not None and geo_gain > 0.0),
        "geo_strong_effective": bool(geo_gain is not None and geo_gain > 0.5),
        "detected_points": int(detected_points),
        "priority_strategy": priority_point_debug.get("strategy") if isinstance(priority_point_debug, dict) else None,
        "risk_mean": risk_stats["mean"],
        "risk_min": risk_stats["min"],
        "risk_max": risk_stats["max"],
        "confidence_mean": conf_stats["mean"],
        "confidence_min": conf_stats["min"],
        "confidence_max": conf_stats["max"],
        "bad_point_count": int(np.sum(risk_arr > 0.80)) if len(risk_arr) else 0,
        "good_point_count": int(np.sum(conf_arr > 0.55)) if len(conf_arr) else 0,
        "valid_ratio_mean": valid_stats["mean"],
        "valid_ratio_min": valid_stats["min"],
        "valid_ratio_max": valid_stats["max"],
        "fit_residual_mean": fit_stats["mean"],
        "fit_residual_min": fit_stats["min"],
        "fit_residual_max": fit_stats["max"],
        "fit_used_points_mean": fit_used_stats["mean"],
        "fit_used_points_min": fit_used_stats["min"],
        "fit_used_points_max": fit_used_stats["max"],
        "ellipse_area_mean": area_stats["mean"],
        "ellipse_area_min": area_stats["min"],
        "ellipse_area_max": area_stats["max"],
        "ecc_ratio_mean": ecc_stats["mean"],
        "ecc_ratio_min": ecc_stats["min"],
        "ecc_ratio_max": ecc_stats["max"],
        "brightness_mean_mean": bright_stats["mean"],
        "brightness_mean_min": bright_stats["min"],
        "brightness_mean_max": bright_stats["max"],
        "core_used_count": int(sum(1 for v in core_used_flags if v > 0.5)),
        "core_used_ratio": float(np.mean(core_used_flags)) if core_used_flags else None,
        "core_stability_mean": core_stability_stats["mean"],
        "center_cov_trace_mean": cov_trace_stats["mean"],
        "center_cov_trace_min": cov_trace_stats["min"],
        "center_cov_trace_max": cov_trace_stats["max"],
        "edge_sigma_mean": edge_sigma_stats["mean"],
        "edge_sigma_min": edge_sigma_stats["min"],
        "edge_sigma_max": edge_sigma_stats["max"],
        "saturation_ratio_mean": sat_ratio_stats["mean"],
        "saturation_ratio_min": sat_ratio_stats["min"],
        "saturation_ratio_max": sat_ratio_stats["max"],
        "edge_plateau_ratio_mean": plateau_ratio_stats["mean"],
        "edge_plateau_ratio_min": plateau_ratio_stats["min"],
        "edge_plateau_ratio_max": plateau_ratio_stats["max"],
        "effective_arc_coverage_mean": arc_coverage_stats["mean"],
        "effective_arc_coverage_min": arc_coverage_stats["min"],
        "effective_arc_coverage_max": arc_coverage_stats["max"],
    }


def _points_array(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    arr = arr.reshape(-1, 2)
    finite = np.all(np.isfinite(arr), axis=1)
    return arr[finite]




def load_inputs(chess_img_path, light_img_path):
    chess_img = cv2.imread(chess_img_path)
    light_img = cv2.imread(light_img_path)
    if chess_img is None or light_img is None:
        raise RuntimeError(f"图像读取失败: chess={chess_img_path}, light={light_img_path}")
    return {
        "chess_img": chess_img,
        "light_img": light_img,
        "gray_chess": cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY),
        "gray_light": cv2.cvtColor(light_img, cv2.COLOR_BGR2GRAY),
    }



def prepare_manual_roi(inputs_ctx, roi_points=None):
    out = dict(inputs_ctx)
    if roi_points is not None:
        out["light_img_proc"] = roi.apply_roi_mask(out["light_img"], roi_points)
        out["gray_light_proc"] = cv2.cvtColor(out["light_img_proc"], cv2.COLOR_BGR2GRAY)
    else:
        out["light_img_proc"] = out["light_img"]
        out["gray_light_proc"] = out["gray_light"]
    out["roi_points"] = roi_points
    return out






def estimate_pose_from_chessboard(gray_chess, K):
    objp, corners_sub, rvec1, tvec1 = solve_pose_chessboard_only(gray_chess, K)
    R1, _ = cv2.Rodrigues(rvec1)
    t1 = tvec1.reshape(3)

    v1 = circle_centers_w[1] - circle_centers_w[0]
    v2 = circle_centers_w[2] - circle_centers_w[0]
    n_w = np.cross(v1, v2)
    n_w /= np.linalg.norm(n_w)
    d_w = -np.dot(n_w, circle_centers_w[0])

    n_c1 = R1 @ n_w
    n_c1 /= np.linalg.norm(n_c1)
    d_c1 = d_w - n_c1 @ t1

    proj_pose_final = np.array([
        (K @ (R1 @ Xw + t1))[:2] / (K @ (R1 @ Xw + t1))[2] for Xw in circle_centers_w
    ], dtype=np.float64)

    pose_ref_debug = estimate_pose_reference_uncertainty(objp, corners_sub, rvec1, tvec1, K)
    pose_sigma_px = float(pose_ref_debug.get("pose_sigma_px", 1.5))
    pose_reference_covariances = [
        (np.eye(2, dtype=np.float64) * (pose_sigma_px ** 2)).tolist()
        for _ in range(len(proj_pose_final))
    ]

    return {
        "objp": objp,
        "corners_sub": corners_sub,
        "rvec1": rvec1,
        "tvec1": tvec1,
        "R1": R1,
        "t1": t1,
        "n_c1": n_c1,
        "d_c1": float(d_c1),
        "proj_pose_final": proj_pose_final,
        "proj_true_eval": proj_pose_final.copy(),
        "pose_reference_eval": proj_pose_final.copy(),
        "pose_ref_debug": pose_ref_debug,
        "pose_sigma_px": pose_sigma_px,
        "pose_reference_covariances": pose_reference_covariances,
    }


def _reference_label_from_pose_ctx(pose_ctx):
    source = str((pose_ctx or {}).get("pose_source", "light_only")).lower()
    if source.startswith("light_only"):
        return "light_only_layout_radius_pose_internal"
    return "pose_reference_from_chessboard"


def _roi_mask_from_points(image_shape, roi_points):
    if roi_points is None:
        return None
    pts = np.asarray(roi_points, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def estimate_pose_from_light_layout(light_img, K, roi_points=None):
    """Estimate target pose only from the light image and known lamp geometry."""
    roi_mask = _roi_mask_from_points(light_img.shape, roi_points)
    if light_only_pose_v2 is not None:
        pose_ctx, result = light_only_pose_v2.estimate_pose_ctx_from_light_only(
            light_img,
            K,
            light_centers_mm=circle_centers_w,
            light_radius_mm=LIGHT_RADIUS_MM,
            roi_mask=roi_mask,
            max_candidates=8,
            lm_iterations=16,
        )
        pose_ctx = dict(pose_ctx)
        pose_ctx.setdefault("objp", None)
        pose_ctx.setdefault("corners_sub", None)
        pose_ctx["pose_source"] = "light_only_iterative_layout_radius"
        pose_ctx["pose_ref_debug"] = {
            **(pose_ctx.get("pose_ref_debug", {}) or {}),
            "frontend_upgrade": "candidate_quality_prior_plus_pose_guided_normal_profile_edges",
            "result_metrics": light_only_pose_v2.to_jsonable(result.get("metrics", {})),
        }
        return pose_ctx

    if light_only_pose is None:
        raise RuntimeError("pose_from_lights_no_chess_prior.py 无法导入，不能使用 light_only 位姿源")
    if light_img is None:
        raise RuntimeError("light_only 位姿估计需要有效的光源图像")

    candidates = light_only_pose.detect_light_candidates(light_img, roi_mask=roi_mask, max_candidates=14)
    init, used_candidates = light_only_pose.initialize_pose_from_candidates(
        candidates, K, top_k=min(14, len(candidates))
    )
    used_candidates = light_only_pose.enrich_candidates_with_pose_guided_boundaries(
        light_img, used_candidates, init["assignments"], K, init["rvec"], init["tvec"]
    )
    rvec, tvec, refine_debug = light_only_pose.lm_refine_pose_on_circle_edges(
        init["rvec"], init["tvec"], K, init["assignments"], used_candidates
    )
    metrics = light_only_pose.final_pose_metrics(rvec, tvec, K, init["assignments"], used_candidates)

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R1, _ = cv2.Rodrigues(rvec)
    t1 = tvec.reshape(3)
    n_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_c1 = R1 @ n_w
    n_c1 /= max(np.linalg.norm(n_c1), 1e-12)
    d_c1 = -float(np.dot(n_c1, t1))
    proj_pose_final, _ = cv2.projectPoints(circle_centers_w, rvec, tvec, K, np.zeros(5))
    proj_pose_final = proj_pose_final.reshape(-1, 2).astype(np.float64)

    reproj_med = metrics.get("median_center_reprojection_px")
    if reproj_med is None or not np.isfinite(reproj_med):
        reproj_med = 2.0
    pose_sigma_px = float(np.clip(max(0.75, 0.75 * float(reproj_med)), 0.5, 8.0))
    pose_reference_covariances = [
        (np.eye(2, dtype=np.float64) * (pose_sigma_px ** 2)).tolist()
        for _ in range(len(proj_pose_final))
    ]

    pose_ref_debug = {
        "pose_source": "light_only",
        "logic_contract": {
            "uses_chessboard_prior": False,
            "uses_external_pose_prior": False,
            "pose_source": "estimated_from_light_image_candidates_known_layout_and_known_radius",
            "allowed_inputs": [
                "light image",
                "camera intrinsics",
                "known 3D lamp layout",
                "known lamp radius",
                "optional ROI mask",
            ],
        },
        "initial_pose": {
            "rvec": np.asarray(init["rvec"], dtype=np.float64).reshape(3).tolist(),
            "tvec": np.asarray(init["tvec"], dtype=np.float64).reshape(3).tolist(),
            "score": float(init["score"]),
            "solver_flag": int(init["solver_flag"]),
            "debug": init.get("debug", {}),
        },
        "refine_debug": refine_debug,
        "metrics": light_only_pose.to_jsonable(metrics),
        "assignment": [
            {
                "object_index": int(obj_idx),
                "candidate_index": int(cand_idx),
                "candidate_center_px": np.asarray(used_candidates[cand_idx]["center"], dtype=np.float64).tolist(),
            }
            for obj_idx, cand_idx in init["assignments"]
        ],
        "num_candidates": int(len(used_candidates)),
        "pose_sigma_px": pose_sigma_px,
    }

    return {
        "objp": None,
        "corners_sub": None,
        "rvec1": rvec,
        "tvec1": tvec,
        "R1": R1,
        "t1": t1,
        "n_c1": n_c1,
        "d_c1": float(d_c1),
        "proj_pose_final": proj_pose_final,
        "proj_true_eval": proj_pose_final.copy(),
        "pose_reference_eval": proj_pose_final.copy(),
        "pose_ref_debug": pose_ref_debug,
        "pose_sigma_px": pose_sigma_px,
        "pose_reference_covariances": pose_reference_covariances,
        "pose_source": "light_only",
    }
def _prepare_projection_context(K, n_c, d_c):
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n_c = np.asarray(n_c, dtype=np.float64).reshape(3)
    return {
        "K": K,
        "Kinv": np.linalg.inv(K),
        "n_c": n_c,
        "d_c": float(d_c),
        "plane_basis": _plane_basis_from_normal(n_c),
    }


def _project_single_pixel_to_plane(uv, proj_ctx):
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    Kinv = np.asarray(proj_ctx['Kinv'], dtype=np.float64).reshape(3, 3)
    n_c = np.asarray(proj_ctx['n_c'], dtype=np.float64).reshape(3)
    d_c = float(proj_ctx['d_c'])
    ray = Kinv @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    den = float(np.dot(n_c, ray))
    if not np.isfinite(den) or abs(den) < 1e-12:
        return None
    s = -d_c / den
    X = ray * s
    return X.astype(np.float64)


def _radius_consistent_point_on_normal(base_pt, normal, fit, center_cam, proj_ctx, target_radius_mm):
    if fit is None:
        return None, {'enabled': False, 'reason': 'no_fit'}
    center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_cam)):
        return None, {'enabled': False, 'reason': 'invalid_center_cam'}
    base_pt = np.asarray(base_pt, dtype=np.float64).reshape(2)
    normal = np.asarray(normal, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(base_pt)) or not np.all(np.isfinite(normal)):
        return None, {'enabled': False, 'reason': 'invalid_base_or_normal'}
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        return None, {'enabled': False, 'reason': 'zero_normal'}
    normal = normal / nrm

    sign_flip = 1.0 if str(fit.get('used_profile_direction', 'forward')) == 'forward' else -1.0
    edge_offset = sign_flip * float(fit.get('edge_offset', 0.0))
    plateau_offset = sign_flip * float(fit.get('plateau_end_offset', 0.0))
    lo = min(edge_offset, plateau_offset)
    hi = max(edge_offset, plateau_offset)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, {'enabled': False, 'reason': 'nonfinite_interval'}
    if hi - lo < 1e-6:
        return None, {'enabled': False, 'reason': 'degenerate_interval'}

    def g(d):
        uv = base_pt + d * normal
        X = _project_single_pixel_to_plane(uv, proj_ctx)
        if X is None or not np.all(np.isfinite(X)):
            return np.nan
        return float(np.linalg.norm(X - center_cam) - float(target_radius_mm))

    ds = np.linspace(lo, hi, 17, dtype=np.float64)
    gs = np.array([g(float(d)) for d in ds], dtype=np.float64)
    finite = np.isfinite(gs)
    if np.count_nonzero(finite) < 2:
        return None, {'enabled': False, 'reason': 'projection_fail'}

    ds_f = ds[finite]
    gs_f = gs[finite]
    best_idx = int(np.argmin(np.abs(gs_f)))
    best_d = float(ds_f[best_idx])
    best_err = float(abs(gs_f[best_idx]))

    bracket = None
    for i in range(len(ds_f) - 1):
        if gs_f[i] == 0.0:
            best_d = float(ds_f[i])
            best_err = 0.0
            bracket = None
            break
        if gs_f[i] * gs_f[i + 1] < 0.0:
            bracket = (float(ds_f[i]), float(ds_f[i + 1]))
            break

    if bracket is not None:
        a, b = bracket
        ga, gb = g(a), g(b)
        for _ in range(24):
            m = 0.5 * (a + b)
            gm = g(m)
            if not np.isfinite(gm):
                break
            if abs(gm) < best_err:
                best_d = float(m)
                best_err = float(abs(gm))
            if ga * gm <= 0.0:
                b, gb = m, gm
            else:
                a, ga = m, gm

    uv_best = base_pt + best_d * normal
    edge_uv = base_pt + edge_offset * normal
    corr_px = float(np.linalg.norm(uv_best - edge_uv))
    return uv_best.astype(np.float64), {
        'enabled': True,
        'best_offset': float(best_d),
        'edge_offset_signed': float(edge_offset),
        'plateau_offset_signed': float(plateau_offset),
        'radius_error_mm': float(best_err),
        'correction_px': float(corr_px),
        'interval': [float(lo), float(hi)],
    }


def _build_pose_runtime_cache(pose_ctx, K, circle_centers_w):
    R1 = np.asarray(pose_ctx['R1'], dtype=np.float64)
    t1 = np.asarray(pose_ctx['t1'], dtype=np.float64).reshape(3)
    proj_ctx = _prepare_projection_context(K, pose_ctx['n_c1'], pose_ctx['d_c1'])
    expected_img = np.asarray(pose_ctx['proj_pose_final'], dtype=np.float64).reshape(-1, 2)
    circle_centers = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)
    expected_cam = (R1 @ circle_centers.T + t1.reshape(3, 1)).T
    expected_ellipses = [
        _project_world_circle_to_image_ellipse(np.asarray(Xw, dtype=np.float64), LIGHT_RADIUS_MM, proj_ctx['K'], R1, t1)
        for Xw in circle_centers
    ]
    pose_ctx['runtime_cache'] = {
        'projection_ctx': proj_ctx,
        'expected_img': expected_img,
        'expected_cam': expected_cam,
        'expected_ellipses': expected_ellipses,
        'R1': R1,
        't1': t1,
    }
    return pose_ctx['runtime_cache']


def _contour_pixel_stats_local(gray, contour):
    contour = _normalize_contour_for_opencv(contour)
    if contour.size == 0:
        return {'bbox': (0, 0, 0, 0), 'pixels': np.zeros((0,), dtype=np.float64)}
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return {'bbox': (int(x), int(y), int(w), int(h)), 'pixels': np.zeros((0,), dtype=np.float64)}
    roi_gray = gray[y:y + h, x:x + w]
    contour_local = contour.copy()
    contour_local[:, :, 0] -= x
    contour_local[:, :, 1] -= y
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour_local], -1, 255, thickness=-1)
    pix = roi_gray[mask > 0].astype(np.float64)
    return {'bbox': (int(x), int(y), int(w), int(h)), 'pixels': pix}



def detect_light_candidates(light_img_proc, max_contours=4):
    return extract_hardcore_contours(light_img_proc, num_lights=max_contours)











def pack_candidate_result(contour, measurement, edge_set, geo_est, fused, extra_baselines=None):
    fused_center = None if fused is None else np.asarray(fused.get("center", [np.nan, np.nan]), dtype=np.float64)
    extra_baselines = dict(extra_baselines or {})
    antipodal_est = estimate_antipodal_edge_midpoint_center(edge_set, measurement)
    return {
        "contour": contour,
        "bbox_center": np.asarray(measurement.get("bbox_center", [np.nan, np.nan]), dtype=np.float64),
        "gray_center": np.asarray(measurement.get("gray_center", [np.nan, np.nan]), dtype=np.float64),
        "weighted_gray_center": np.asarray(measurement.get("weighted_gray_center", [np.nan, np.nan]), dtype=np.float64),
        "contour_circle_center": np.asarray(measurement.get("contour_circle_center", [np.nan, np.nan]), dtype=np.float64),
        "direct_ellipse_center": np.asarray(measurement.get("direct_ellipse_center", [np.nan, np.nan]), dtype=np.float64),
        "ellipse_center": np.asarray(measurement.get("ellipse_center", [np.nan, np.nan]), dtype=np.float64),
        "ellipse_backproj_center": np.asarray(extra_baselines.get("ellipse_backproj_center", [np.nan, np.nan]), dtype=np.float64),
        "homography_rectified_center": np.asarray(extra_baselines.get("homography_rectified_center", [np.nan, np.nan]), dtype=np.float64),
        "antipodal_edge_center": np.asarray(antipodal_est.get("center", [np.nan, np.nan]), dtype=np.float64),
        "geo_center": np.asarray(geo_est.get("center", [np.nan, np.nan]), dtype=np.float64),
        "fused_center": fused_center,
        "refined_pts": edge_set.get("points"),
        "ellipse_cache": measurement.get("work_ellipse"),
        "point_debug": geo_est.get("point_debug", {}),
        "fusion_debug": None if fused is None else fused.get("debug", {}),
        "baseline_debug": extra_baselines.get("baseline_debug", {}),
        "antipodal_edge_debug": antipodal_est.get("debug", {}),
        "measurement": measurement,
        "edge_set": edge_set,
        "geo_est": geo_est,
        "fused_est": fused,
    }



def fit_step4_work_ellipse_result(selected_item, proj_pose_final=None):
    """对 Step 4 输出的工作轮廓做椭圆拟合，作为消融实验分支。

    该分支对应：物理轮廓假设搜索后的工作轮廓 -> 图像平面椭圆中心。
    不经过 Step 5 亚像素边缘提取，也不经过 Step 6 反投影圆拟合。
    """
    contour = selected_item.get("work_contour", None)
    ellipse = selected_item.get("work_ellipse", None)

    if ellipse is None and contour is not None and len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(np.asarray(contour, dtype=np.float32))
        except Exception:
            ellipse = None

    out = {
        "available": False,
        "source": "step4_work_contour",
        "ellipse": None,
        "center_xy": None,
        "axes": None,
        "angle_deg": None,
        "fit_residual": None,
        "err_to_pose_ref_px": None,
        "matched_pose_index": None,
    }

    if ellipse is None:
        return out

    (cx, cy), (MA, ma), ang = ellipse
    center = np.array([float(cx), float(cy)], dtype=np.float64)

    out["available"] = True
    out["ellipse"] = {
        "center": center.tolist(),
        "axes": [float(MA), float(ma)],
        "angle_deg": float(ang),
    }
    out["center_xy"] = center.tolist()
    out["axes"] = [float(MA), float(ma)]
    out["angle_deg"] = float(ang)

    if contour is not None and len(contour) >= 6:
        try:
            out["fit_residual"] = float(_ellipse_contour_residual(ellipse, contour))
        except Exception:
            out["fit_residual"] = None

    if proj_pose_final is not None:
        proj_pose_final = np.asarray(proj_pose_final, dtype=np.float64).reshape(-1, 2)
        if proj_pose_final.size > 0:
            d = np.linalg.norm(proj_pose_final - center[None, :], axis=1)
            idx = int(np.argmin(d))
            out["matched_pose_index"] = idx
            out["err_to_pose_ref_px"] = float(d[idx])

    return out




def attach_uncertainty_fusion(measurements, proj_pose_final=None, pose_sigma_px=None):
    """保留函数名以兼容主链。
    当前将 estimate_geometric_center 输出的“主方法结果”和“诊断原始几何结果”同时保留：
    - item["geo_center"]: 论文主方法输出（带保守回退/融合）
    - item["fused_center"]: 诊断用原始几何中心（未保守回退时）
    """
    for item in measurements:
        geo_est = item.get("geo_est", {}) or {}
        geo_ctr = np.asarray(item.get("geo_center", item["measurement"].get("ellipse_center", [np.nan, np.nan])), dtype=np.float64)
        geo_cov = np.asarray(geo_est.get("cov", np.eye(2, dtype=np.float64) * 36.0), dtype=np.float64)
        diag_ctr = np.asarray(geo_est.get("diagnostic_center", geo_ctr), dtype=np.float64)
        diag_cov = np.asarray(geo_est.get("diagnostic_cov", geo_cov), dtype=np.float64)
        fusion_debug = geo_est.get("fusion_debug", {}) or {}

        item["fused_est"] = {
            "center": diag_ctr.copy(),
            "cov": diag_cov.copy(),
            "debug": {
                "mode": "geometry_output_vs_raw_diagnostic",
                "pose_reference_index": None,
                "pose_center": None,
                "pose_cov": None,
                "ellipse_center": np.asarray(item["measurement"].get("ellipse_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
                "ellipse_cov": np.asarray(item["measurement"].get("ellipse_cov", np.eye(2) * 25.0), dtype=np.float64).tolist(),
                "geometry_center": geo_ctr.tolist(),
                "geometry_cov": geo_cov.tolist(),
                "diagnostic_center": diag_ctr.tolist(),
                "diagnostic_cov": diag_cov.tolist(),
                "fused_center": diag_ctr.tolist(),
                "fused_cov": diag_cov.tolist(),
                "fusion": {
                    "used_terms": ["geometry_output", "raw_geometry_diagnostic"],
                    "cov_trace": float(np.trace(diag_cov)),
                    "cov_det": float(np.linalg.det(diag_cov)),
                    "source_precision_trace": {},
                },
                **fusion_debug,
            },
        }
        item["fused_center"] = diag_ctr.copy()
        item["fusion_debug"] = item["fused_est"]["debug"]
    return measurements, [None] * len(measurements)
def _rank_badness(arr, high_bad=True):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) <= 1:
        return out
    vals = arr[finite]
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(vals))
    out[finite] = ranks if high_bad else (1.0 - ranks)
    return out






def evaluate_method_suite(light_img_path, pred_bbox, pred_gray_centroid, pred_ellipse,
                          pred_step4_work_ellipse, pred_geo_pass1, pred_geo_bias_pass1, proj_true_eval,
                          pred_weighted_gray=None, pred_circle=None,
                          pred_ellipse_backproj=None, pred_homography_rectified=None,
                          pred_layout_constrained_pnp=None,
                          pred_antipodal_edge=None,
                          pred_joint_layout_edge=None,
                          pred_final_edge_layout=None,
                          pred_physical_edge_circle=None,
                          reference_label="pose_reference_from_chessboard"):
    print("\n" + "=" * 70)
    print(f"处理图像: {os.path.basename(light_img_path)}")
    print(f"多方法定位误差全对比（评估参考: {reference_label}）")
    print("=" * 70)
    bbox_err, bbox_aligned_true, bbox_perm, bbox_match = evaluate_predictions_with_miss_penalty(pred_bbox, proj_true_eval, "边界框中心法  ", miss_penalty=None, verbose=False)
    centroid_err, centroid_aligned_true, centroid_perm, centroid_match = evaluate_predictions_with_miss_penalty(pred_gray_centroid, proj_true_eval, "灰度质心法    ", miss_penalty=None, verbose=True)
    wcent_err, wcent_aligned_true, wcent_perm, wcent_match = evaluate_predictions_with_miss_penalty(pred_weighted_gray if pred_weighted_gray is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "加权灰度质心  ", miss_penalty=None, verbose=True)
    circle_err, circle_aligned_true, circle_perm, circle_match = evaluate_predictions_with_miss_penalty(pred_circle if pred_circle is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "轮廓圆拟合    ", miss_penalty=None, verbose=True)
    ellipse_err, ellipse_aligned_true, ellipse_perm, ellipse_match = evaluate_predictions_with_miss_penalty(pred_ellipse, proj_true_eval, "直接椭圆拟合  ", miss_penalty=None, verbose=True)
    step4_ellipse_err, step4_ellipse_aligned_true, step4_ellipse_perm, step4_ellipse_match = evaluate_predictions_with_miss_penalty(pred_step4_work_ellipse, proj_true_eval, "选中边界中心  ", miss_penalty=None, verbose=False)
    print("-" * 70)
    ebc_err, ebc_aligned_true, ebc_perm, ebc_match = evaluate_predictions_with_miss_penalty(pred_ellipse_backproj if pred_ellipse_backproj is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "椭圆反投影校正", miss_penalty=None, verbose=True)
    hpr_err, hpr_aligned_true, hpr_perm, hpr_match = evaluate_predictions_with_miss_penalty(pred_homography_rectified if pred_homography_rectified is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "平面整圆恢复  ", miss_penalty=None, verbose=True)
    lcp_err, lcp_aligned_true, lcp_perm, lcp_match = evaluate_predictions_with_miss_penalty(pred_layout_constrained_pnp if pred_layout_constrained_pnp is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "布局约束PnP   ", miss_penalty=None, verbose=True)
    anti_err, anti_aligned_true, anti_perm, anti_match = evaluate_predictions_with_miss_penalty(pred_antipodal_edge if pred_antipodal_edge is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "对偶边缘校正  ", miss_penalty=None, verbose=True)
    joint_err, joint_aligned_true, joint_perm, joint_match = evaluate_predictions_with_miss_penalty(pred_joint_layout_edge if pred_joint_layout_edge is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "四灯联合边缘  ", miss_penalty=None, verbose=True)
    final_edge_err, final_edge_aligned_true, final_edge_perm, final_edge_match = evaluate_predictions_with_miss_penalty(pred_final_edge_layout if pred_final_edge_layout is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "最终边缘布局  ", miss_penalty=None, verbose=True)
    physical_edge_err, physical_edge_aligned_true, physical_edge_perm, physical_edge_match = evaluate_predictions_with_miss_penalty(pred_physical_edge_circle if pred_physical_edge_circle is not None else np.empty((0, 2), dtype=np.float64), proj_true_eval, "物理边缘圆约束", miss_penalty=None, verbose=True)
    pass1_err, pass1_aligned_true, pass1_perm, pass1_match = evaluate_predictions_with_miss_penalty(pred_geo_pass1, proj_true_eval, "几何校正法    ", miss_penalty=None, verbose=True)
    pass3a_err, pass3a_aligned_true, pass3a_perm, pass3a_match = evaluate_predictions_with_miss_penalty(pred_geo_bias_pass1, proj_true_eval, "几何诊断输出  ", miss_penalty=None, verbose=False)
    print("=" * 70)

    return {
        "bbox": {"err": bbox_err, "aligned": bbox_aligned_true, "perm": bbox_perm, "match": bbox_match},
        "centroid": {"err": centroid_err, "aligned": centroid_aligned_true, "perm": centroid_perm, "match": centroid_match},
        "weighted_centroid": {"err": wcent_err, "aligned": wcent_aligned_true, "perm": wcent_perm, "match": wcent_match},
        "circle": {"err": circle_err, "aligned": circle_aligned_true, "perm": circle_perm, "match": circle_match},
        "ellipse": {"err": ellipse_err, "aligned": ellipse_aligned_true, "perm": ellipse_perm, "match": ellipse_match},
        "step4_work_ellipse": {"err": step4_ellipse_err, "aligned": step4_ellipse_aligned_true, "perm": step4_ellipse_perm, "match": step4_ellipse_match},
        "ellipse_backproj": {"err": ebc_err, "aligned": ebc_aligned_true, "perm": ebc_perm, "match": ebc_match},
        "homography_rectified": {"err": hpr_err, "aligned": hpr_aligned_true, "perm": hpr_perm, "match": hpr_match},
        "layout_constrained_pnp": {"err": lcp_err, "aligned": lcp_aligned_true, "perm": lcp_perm, "match": lcp_match},
        "antipodal_edge": {"err": anti_err, "aligned": anti_aligned_true, "perm": anti_perm, "match": anti_match},
        "joint_layout_edge": {"err": joint_err, "aligned": joint_aligned_true, "perm": joint_perm, "match": joint_match},
        "final_edge_layout": {"err": final_edge_err, "aligned": final_edge_aligned_true, "perm": final_edge_perm, "match": final_edge_match},
        "physical_edge_circle": {"err": physical_edge_err, "aligned": physical_edge_aligned_true, "perm": physical_edge_perm, "match": physical_edge_match},
        "geo": {"err": pass1_err, "aligned": pass1_aligned_true, "perm": pass1_perm, "match": pass1_match},
        "fusion": {"err": pass3a_err, "aligned": pass3a_aligned_true, "perm": pass3a_perm, "match": pass3a_match},
    }

def assemble_single_image_results(chess_img_path, light_img_path, save_dir, roi_points, max_contours,
                                  pose_ctx, measurement_ctx, eval_ctx):
    reference_label = _reference_label_from_pose_ctx(pose_ctx)
    contours = measurement_ctx["contours"]
    pred_bbox = measurement_ctx["pred_bbox"]
    pred_gray_centroid = measurement_ctx["pred_gray_centroid"]
    pred_weighted_gray = measurement_ctx.get("pred_weighted_gray", np.empty((0, 2), dtype=np.float64))
    pred_circle = measurement_ctx.get("pred_circle", np.empty((0, 2), dtype=np.float64))
    pred_ellipse = measurement_ctx["pred_ellipse"]
    pred_step4_work_ellipse = measurement_ctx.get("pred_step4_work_ellipse", np.empty((0, 2), dtype=np.float64))
    pred_ellipse_backproj = measurement_ctx.get("pred_ellipse_backproj", np.empty((0, 2), dtype=np.float64))
    pred_homography_rectified = measurement_ctx.get("pred_homography_rectified", np.empty((0, 2), dtype=np.float64))
    pred_layout_constrained_pnp = measurement_ctx.get("pred_layout_constrained_pnp", np.empty((0, 2), dtype=np.float64))
    pred_antipodal_edge = measurement_ctx.get("pred_antipodal_edge", np.empty((0, 2), dtype=np.float64))
    pred_joint_layout_edge = measurement_ctx.get("pred_joint_layout_edge", np.empty((0, 2), dtype=np.float64))
    pred_final_edge_layout = measurement_ctx.get("pred_final_edge_layout", np.empty((0, 2), dtype=np.float64))
    pred_physical_edge_circle = measurement_ctx.get("pred_physical_edge_circle", np.empty((0, 2), dtype=np.float64))
    pred_geo_pass1 = measurement_ctx["pred_geo_pass1"]
    pred_geo_bias_pass1 = measurement_ctx["pred_geo_bias_pass1"]
    point_debug_pass1 = measurement_ctx["point_debug_pass1"]
    measurements = measurement_ctx["measurements"]
    refined_pts_cache = measurement_ctx["refined_pts_cache"]
    ellipse_cache = measurement_ctx["ellipse_cache"]
    conf_scores_pass1 = measurement_ctx["conf_scores_pass1"]
    risk_scores_pass1 = measurement_ctx["risk_scores_pass1"]
    priority_point_debug = measurement_ctx["priority_point_debug"]
    conf_debug_pass1 = measurement_ctx["conf_debug_pass1"]
    pose_bias_debug_pass1 = measurement_ctx["pose_bias_debug_pass1"]
    fusion_debug_pass1 = measurement_ctx["fusion_debug_pass1"]
    uncertainty_debug_pass1 = measurement_ctx["uncertainty_debug_pass1"]

    bbox = eval_ctx["bbox"]
    centroid = eval_ctx["centroid"]
    weighted_centroid_eval = eval_ctx.get("weighted_centroid", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    circle_eval = eval_ctx.get("circle", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    ellipse_eval = eval_ctx["ellipse"]
    step4_work_ellipse_eval = eval_ctx.get("step4_work_ellipse", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    ellipse_backproj_eval = eval_ctx.get("ellipse_backproj", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    homography_rectified_eval = eval_ctx.get("homography_rectified", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    layout_constrained_pnp_eval = eval_ctx.get("layout_constrained_pnp", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    antipodal_edge_eval = eval_ctx.get("antipodal_edge", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    joint_layout_edge_eval = eval_ctx.get("joint_layout_edge", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    final_edge_layout_eval = eval_ctx.get("final_edge_layout", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    physical_edge_circle_eval = eval_ctx.get("physical_edge_circle", {"err": np.nan, "aligned": None, "perm": None, "match": {}})
    geo = eval_ctx["geo"]
    fusion_eval = eval_ctx["fusion"]

    bbox_err = bbox["err"]
    centroid_err = centroid["err"]
    weighted_centroid_err = weighted_centroid_eval["err"]
    circle_err = circle_eval["err"]
    ellipse_err = ellipse_eval["err"]
    step4_work_ellipse_err = step4_work_ellipse_eval["err"]
    ellipse_backproj_err = ellipse_backproj_eval["err"]
    homography_rectified_err = homography_rectified_eval["err"]
    layout_constrained_pnp_err = layout_constrained_pnp_eval["err"]
    antipodal_edge_err = antipodal_edge_eval["err"]
    joint_layout_edge_err = joint_layout_edge_eval["err"]
    final_edge_layout_err = final_edge_layout_eval["err"]
    physical_edge_circle_err = physical_edge_circle_eval["err"]
    pass1_err = geo["err"]
    pass3a_err = fusion_eval["err"]

    applicability_features = summarize_applicability_features(
        light_img_path, point_debug_pass1, priority_point_debug, conf_scores_pass1, risk_scores_pass1,
        float(ellipse_err) if not np.isnan(ellipse_err) else None,
        float(pass1_err) if not np.isnan(pass1_err) else None,
        float(pass3a_err) if not np.isnan(pass3a_err) else None,
        len(contours),
    )

    candidate_detection = []
    per_point_measurements = []
    for i, contour in enumerate(contours):
        dbg = point_debug_pass1[i] if i < len(point_debug_pass1) else {}
        fusion_dbg = fusion_debug_pass1[i] if i < len(fusion_debug_pass1) else {}
        m = measurements[i] if i < len(measurements) else {}
        x, y, w, h = cv2.boundingRect(contour)
        candidate_detection.append({
            "detected_index": int(i),
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "contour_area": float(cv2.contourArea(contour)),
            "brightness_mean": dbg.get("brightness_mean"),
            "representation": dbg.get("representation"),
        })
        per_point_measurements.append({
            "detected_index": int(i),
            "bbox_center": np.asarray(m.get("bbox_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "gray_center": np.asarray(m.get("gray_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "weighted_gray_center": np.asarray(m.get("weighted_gray_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "contour_circle_center": np.asarray(m.get("contour_circle_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "ellipse_center": np.asarray(m.get("direct_ellipse_center", m.get("ellipse_center", [np.nan, np.nan])), dtype=np.float64).tolist(),
            "selected_boundary_center": np.asarray((m.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "step4_work_ellipse_center": np.asarray((m.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "ellipse_backproj_center": np.asarray(m.get("ellipse_backproj_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "homography_rectified_center": np.asarray(m.get("homography_rectified_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "layout_constrained_pnp_center": np.asarray(m.get("layout_constrained_pnp_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "antipodal_edge_center": np.asarray(m.get("antipodal_edge_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "joint_layout_edge_center": np.asarray(m.get("joint_layout_edge_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "joint_layout_edge_raw_center": np.asarray(m.get("joint_layout_edge_raw_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "joint_layout_edge_local_center": np.asarray(m.get("joint_layout_edge_local_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "joint_layout_edge_layout_anchor": np.asarray(m.get("joint_layout_edge_layout_anchor", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "final_edge_layout_center": np.asarray(m.get("final_edge_layout_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "physical_edge_circle_center": np.asarray(m.get("physical_edge_circle_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "physical_edge_circle_raw_center": np.asarray(m.get("physical_edge_circle_raw_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "geometry_center": np.asarray(m.get("geo_center", [np.nan, np.nan]), dtype=np.float64).tolist(),
            "assigned_layout_index": (dbg.get("representation") or {}).get("assigned_layout_index") if isinstance(dbg.get("representation"), dict) else None,
            "pose_reference_index": fusion_dbg.get("pose_reference_index"),
            "pose_reference_center": fusion_dbg.get("pose_center"),
            "uncertainty_fusion_center": fusion_dbg.get("fused_center"),
            "ellipse_cov": fusion_dbg.get("ellipse_cov"),
            "geometry_cov": fusion_dbg.get("geometry_cov"),
            "pose_cov": fusion_dbg.get("pose_cov"),
            "uncertainty_fusion_cov": fusion_dbg.get("fused_cov"),
            "edge_point_count": dbg.get("edge_point_count"),
            "edge_expected_count": dbg.get("edge_expected_count"),
            "edge_fit_success_ratio": dbg.get("edge_fit_success_ratio"),
            "edge_sigma_mean": dbg.get("edge_sigma_mean"),
            "edge_quality_mean": dbg.get("edge_quality_mean"),
            "edge_plateau_ratio_mean": dbg.get("edge_plateau_ratio_mean"),
            "effective_arc_coverage": dbg.get("effective_arc_coverage"),
            "geometry_cov_inflation": dbg.get("geometry_cov_inflation"),
            "saturation_ratio": dbg.get("saturation_ratio"),
            "valid_ratio": dbg.get("valid_ratio"),
            "fit_residual_median": dbg.get("fit_residual_median"),
            "fit_used_points": dbg.get("fit_used_points"),
            "refined_edge_points": (
                np.asarray(refined_pts_cache[i], dtype=np.float64).tolist()
                if os.environ.get("PAGC_SAVE_REFINED_EDGE_POINTS", PAGC_SAVE_REFINED_EDGE_POINTS_DEFAULT).strip().lower() in ("1", "true", "yes", "on")
                and i < len(refined_pts_cache)
                and refined_pts_cache[i] is not None
                else []
            ),
            "work_ellipse": {
                "center": list(ellipse_cache[i][0]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
                "axes": list(ellipse_cache[i][1]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
                "angle_deg": float(ellipse_cache[i][2]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
            },
            "selected_boundary": {
                "center": list(ellipse_cache[i][0]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
                "axes": list(ellipse_cache[i][1]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
                "angle_deg": float(ellipse_cache[i][2]) if i < len(ellipse_cache) and ellipse_cache[i] is not None else None,
            },
            "fusion_debug": fusion_dbg.get("fusion"),
            "step4_work_ellipse_ablation": m.get("step4_work_ellipse_ablation", {}),
            "layout_constrained_pnp_debug": m.get("layout_constrained_pnp_debug", {}),
            "antipodal_edge_debug": m.get("antipodal_edge_debug", {}),
            "joint_layout_edge_debug": m.get("joint_layout_edge_debug", {}),
            "joint_layout_edge_local_fit_debug": m.get("joint_layout_edge_local_fit_debug", {}),
            "final_edge_layout_debug": m.get("final_edge_layout_debug", {}),
            "physical_edge_circle_debug": m.get("physical_edge_circle_debug", {}),
        })

    edge_ratios = [dbg.get("edge_fit_success_ratio") for dbg in point_debug_pass1 if isinstance(dbg, dict) and dbg.get("edge_fit_success_ratio") is not None]
    fit_residuals = [dbg.get("fit_residual_median", dbg.get("fit_residual_mean")) for dbg in point_debug_pass1 if isinstance(dbg, dict) and dbg.get("fit_residual_median", dbg.get("fit_residual_mean")) is not None]
    uncertainty_traces = [dbg.get("fusion", {}).get("cov_trace") for dbg in fusion_debug_pass1 if isinstance(dbg, dict) and isinstance(dbg.get("fusion", {}), dict) and dbg.get("fusion", {}).get("cov_trace") is not None]

    results = {
        "image_info": {
            "chessboard_image": os.path.basename(chess_img_path) if chess_img_path else None,
            "light_image": os.path.basename(light_img_path),
            # "manual_roi": roi.roi_to_serializable(roi_points) if roi_points is not None else None,
        },
        "algorithm_params": {
            "ROI_PAD": ROI_PAD,
            "MAX_CONTOURS": max_contours,
            "detector": "adaptive_intensity_green_contrast_otsu",
            "measurement_contour": "stable_multilevel_contour",
            "edge_localizer": "normal_profile_subpixel_with_saturation_plateau_rejection",
            "center_fusion": "diagnostic_only_not_used_for_final_output",
            "evaluation_reference": reference_label,
            "pose_source": pose_ctx.get("pose_source", "light_only"),
        },
        "input_and_calibration": {
            "ground_truth_points_3d_world": circle_centers_w.tolist(),
            "pose_reference_points_2d_image": pose_ctx["pose_reference_eval"].tolist(),
            "pose_reference_covariances": pose_ctx.get("pose_reference_covariances", []),
            "pose_reference_debug": pose_ctx.get("pose_ref_debug"),
            "camera_pose": {
                "rvec_initial": pose_ctx["rvec1"].flatten().tolist(),
                "tvec_initial": pose_ctx["tvec1"].flatten().tolist(),
                "R_matrix": pose_ctx["R1"].tolist(),
                "t_vector": pose_ctx["t1"].tolist(),
                "plane_normal_camera": pose_ctx["n_c1"].tolist(),
                "plane_distance": float(pose_ctx["d_c1"]),
                "pose_source": pose_ctx.get("pose_source", "light_only"),
            },
        },
        "ground_truth_points_3d_world": circle_centers_w.tolist(),
        "pose_reference_points_2d_image": pose_ctx["pose_reference_eval"].tolist(),
        "ground_truth_points_2d_image": pose_ctx["proj_true_eval"].tolist(),
        "detected_points": len(contours),
        "candidate_detection": {
            "num_detected": int(len(contours)),
            "candidates": candidate_detection,
        },
        "per_point_measurements": per_point_measurements,
        "priority_point_strategy": priority_point_debug,
        "confidence_debug_pass1": conf_debug_pass1,
        "pose_bias_debug_pass1": pose_bias_debug_pass1,
        "point_debug_pass1": point_debug_pass1,
        "uncertainty_fusion": {
            "enabled": True,
            "strategy": "diagnostic_covariance_only",
            "fusion_debug_per_point": fusion_debug_pass1,
            "uncertainty_debug": uncertainty_debug_pass1,
        },
        "applicability_features": applicability_features,
        "method_results": {
            "bounding_box_centroid": _method_result_package(pred_bbox, bbox["aligned"], bbox["perm"], bbox["match"]),
            "gray_centroid": _method_result_package(pred_gray_centroid, centroid["aligned"], centroid["perm"], centroid["match"]),
            "weighted_gray_centroid": _method_result_package(pred_weighted_gray, weighted_centroid_eval["aligned"], weighted_centroid_eval["perm"], weighted_centroid_eval["match"]),
            "contour_circle_fitting": _method_result_package(pred_circle, circle_eval["aligned"], circle_eval["perm"], circle_eval["match"]),
            "ellipse_fitting": _method_result_package(pred_ellipse, ellipse_eval["aligned"], ellipse_eval["perm"], ellipse_eval["match"]),
            "selected_boundary_center": _method_result_package(pred_step4_work_ellipse, step4_work_ellipse_eval["aligned"], step4_work_ellipse_eval["perm"], step4_work_ellipse_eval["match"]),
            "step4_work_ellipse": _method_result_package(pred_step4_work_ellipse, step4_work_ellipse_eval["aligned"], step4_work_ellipse_eval["perm"], step4_work_ellipse_eval["match"]),
            "ellipse_backprojection_correction": _method_result_package(pred_ellipse_backproj, ellipse_backproj_eval["aligned"], ellipse_backproj_eval["perm"], ellipse_backproj_eval["match"]),
            "homography_rectified_circle": _method_result_package(pred_homography_rectified, homography_rectified_eval["aligned"], homography_rectified_eval["perm"], homography_rectified_eval["match"]),
            "layout_constrained_pnp": _method_result_package(pred_layout_constrained_pnp, layout_constrained_pnp_eval["aligned"], layout_constrained_pnp_eval["perm"], layout_constrained_pnp_eval["match"]),
            "antipodal_edge_correction": _method_result_package(pred_antipodal_edge, antipodal_edge_eval["aligned"], antipodal_edge_eval["perm"], antipodal_edge_eval["match"]),
            "joint_layout_edge_correction": _method_result_package(pred_joint_layout_edge, joint_layout_edge_eval["aligned"], joint_layout_edge_eval["perm"], joint_layout_edge_eval["match"]),
            "final_edge_layout_correction": _method_result_package(pred_final_edge_layout, final_edge_layout_eval["aligned"], final_edge_layout_eval["perm"], final_edge_layout_eval["match"]),
            "physical_edge_circle_correction": _method_result_package(pred_physical_edge_circle, physical_edge_circle_eval["aligned"], physical_edge_circle_eval["perm"], physical_edge_circle_eval["match"]),
            "geometry_correction": _method_result_package(pred_geo_pass1, geo["aligned"], geo["perm"], geo["match"]),
            "geometry_diagnostic_copy": _method_result_package(pred_geo_bias_pass1, fusion_eval["aligned"], fusion_eval["perm"], fusion_eval["match"]),
        },
        "evaluation": {
            "reference": reference_label,
            "complete_frame_success_4of4": bool(int(fusion_eval["match"].get("num_missed", 0)) == 0 and int(fusion_eval["match"].get("num_false", 0)) == 0 and len(pred_geo_bias_pass1) == 4),
            "methods": {
                "bounding_box_centroid": _error_stats_from_match(bbox["match"]),
                "gray_centroid": _error_stats_from_match(centroid["match"]),
                "weighted_gray_centroid": _error_stats_from_match(weighted_centroid_eval["match"]),
                "contour_circle_fitting": _error_stats_from_match(circle_eval["match"]),
                "ellipse_fitting": _error_stats_from_match(ellipse_eval["match"]),
                "selected_boundary_center": _error_stats_from_match(step4_work_ellipse_eval["match"]),
                "step4_work_ellipse": _error_stats_from_match(step4_work_ellipse_eval["match"]),
                "ellipse_backprojection_correction": _error_stats_from_match(ellipse_backproj_eval["match"]),
                "homography_rectified_circle": _error_stats_from_match(homography_rectified_eval["match"]),
                "layout_constrained_pnp": _error_stats_from_match(layout_constrained_pnp_eval["match"]),
                "antipodal_edge_correction": _error_stats_from_match(antipodal_edge_eval["match"]),
                "joint_layout_edge_correction": _error_stats_from_match(joint_layout_edge_eval["match"]),
                "final_edge_layout_correction": _error_stats_from_match(final_edge_layout_eval["match"]),
                "physical_edge_circle_correction": _error_stats_from_match(physical_edge_circle_eval["match"]),
                "geometry_correction": _error_stats_from_match(geo["match"]),
                "geometry_diagnostic_copy": _error_stats_from_match(fusion_eval["match"]),
            },
        },
        "debug_summary": {
            "edge_profile_fit_success_ratio_mean": float(np.mean(edge_ratios)) if edge_ratios else None,
            "edge_profile_fit_success_ratio_min": float(np.min(edge_ratios)) if edge_ratios else None,
            "edge_profile_fit_success_ratio_max": float(np.max(edge_ratios)) if edge_ratios else None,
            "plane_fit_residual_median_mean": float(np.mean(fit_residuals)) if fit_residuals else None,
            "plane_fit_residual_median_min": float(np.min(fit_residuals)) if fit_residuals else None,
            "plane_fit_residual_median_max": float(np.max(fit_residuals)) if fit_residuals else None,
            "uncertainty_trace_mean": float(np.mean(uncertainty_traces)) if uncertainty_traces else None,
            "uncertainty_trace_min": float(np.min(uncertainty_traces)) if uncertainty_traces else None,
            "uncertainty_trace_max": float(np.max(uncertainty_traces)) if uncertainty_traces else None,
        },
    }

    save_pose_eval_inputs = os.environ.get("PAGC_SAVE_POSE_EVAL_INPUTS", PAGC_SAVE_POSE_EVAL_INPUTS_DEFAULT).strip().lower() in ("1", "true", "yes", "on")
    pose_eval_inputs = _build_pose_eval_inputs_for_downstream_pose(results) if save_pose_eval_inputs else None
    if pose_eval_inputs is not None:
        results["pose_eval_inputs"] = pose_eval_inputs

    summary = {
        "light_image": os.path.basename(light_img_path),
        "pose_eval_inputs": pose_eval_inputs,
        "detected_points": int(len(contours)),
        "bbox_mean_error": float(bbox_err) if not np.isnan(bbox_err) else None,
        "gray_mean_error": float(centroid_err) if not np.isnan(centroid_err) else None,
        "weighted_gray_mean_error": float(weighted_centroid_err) if not np.isnan(weighted_centroid_err) else None,
        "circle_mean_error": float(circle_err) if not np.isnan(circle_err) else None,
        "ellipse_mean_error": float(ellipse_err) if not np.isnan(ellipse_err) else None,
        "selected_boundary_mean_error": float(step4_work_ellipse_err) if not np.isnan(step4_work_ellipse_err) else None,
        "step4_work_ellipse_mean_error": float(step4_work_ellipse_err) if not np.isnan(step4_work_ellipse_err) else None,
        "ellipse_backproj_mean_error": float(ellipse_backproj_err) if not np.isnan(ellipse_backproj_err) else None,
        "homography_rectified_mean_error": float(homography_rectified_err) if not np.isnan(homography_rectified_err) else None,
        "layout_constrained_pnp_mean_error": float(layout_constrained_pnp_err) if not np.isnan(layout_constrained_pnp_err) else None,
        "antipodal_edge_mean_error": float(antipodal_edge_err) if not np.isnan(antipodal_edge_err) else None,
        "joint_layout_edge_mean_error": float(joint_layout_edge_err) if not np.isnan(joint_layout_edge_err) else None,
        "final_edge_layout_mean_error": float(final_edge_layout_err) if not np.isnan(final_edge_layout_err) else None,
        "physical_edge_circle_mean_error": float(physical_edge_circle_err) if not np.isnan(physical_edge_circle_err) else None,
        "geo_mean_error": float(pass1_err) if not np.isnan(pass1_err) else None,
        "geo_pose_bias_mean_error": None,
        "uncertainty_fusion_mean_error": None,
        "anchor_true_index": priority_point_debug.get("anchor_true_index"),
        "anchor_detected_index": priority_point_debug.get("anchor_detected_index"),
        "priority_detected_indices": priority_point_debug.get("priority_detected_indices", []),
        "risk_scores": risk_scores_pass1.tolist() if len(risk_scores_pass1) else [],
        "confidence": conf_scores_pass1.tolist() if len(conf_scores_pass1) else [],
        "risk_scores_pass1": risk_scores_pass1.tolist() if len(risk_scores_pass1) else [],
        "fusion_uncertainty_trace_mean": float(np.mean(uncertainty_traces)) if uncertainty_traces else None,
        "fusion_complete_4of4": None,
        **applicability_features,
    }

    return results, summary





def build_final_measurement_visualization(inputs_ctx, measurement_ctx, results=None, title_text=None, zoom_scale=5, crop_radius=42):
    vis = inputs_ctx["light_img"].copy()

    def _to_pts2(arr):
        if arr is None:
            return np.empty((0, 2), dtype=np.float64)
        pts = np.asarray(arr, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return pts.reshape(-1, 2)

    def _to_contour_pts(arr):
        if arr is None:
            return np.empty((0, 2), dtype=np.int32)
        pts = np.asarray(arr, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)
        else:
            pts = pts.reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        return pts

    def _draw_points(points, color, radius=2, target=None):
        if target is None:
            target = vis
        pts = _to_pts2(points)
        for p in pts:
            if np.all(np.isfinite(p)):
                cv2.circle(target, tuple(np.round(p).astype(int)), radius, color, -1, lineType=cv2.LINE_AA)

    def _draw_centers(points, color, prefix, marker_type, target=None):
        if target is None:
            target = vis
        pts = _to_pts2(points)
        for i, p in enumerate(pts):
            if not np.all(np.isfinite(p)):
                continue
            pp = tuple(np.round(p).astype(int))
            cv2.drawMarker(target, pp, color, markerType=marker_type, markerSize=12, thickness=2, line_type=cv2.LINE_AA)
            cv2.putText(target, f"{prefix}{i+1}", (pp[0] + 5, pp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    measurements = measurement_ctx.get("measurements", [])
    refined_pts_cache = measurement_ctx.get("refined_pts_cache", [])
    pred_ellipse = _to_pts2(measurement_ctx.get("pred_ellipse", np.empty((0, 2), dtype=np.float64)))
    pred_antipodal_edge = _to_pts2(measurement_ctx.get("pred_antipodal_edge", np.empty((0, 2), dtype=np.float64)))
    pred_joint_layout_edge = _to_pts2(measurement_ctx.get("pred_joint_layout_edge", np.empty((0, 2), dtype=np.float64)))
    pred_geo_pass1 = _to_pts2(measurement_ctx.get("pred_geo_pass1", np.empty((0, 2), dtype=np.float64)))

    ref_pts = None
    if isinstance(results, dict):
        ref_pts = results.get("ground_truth_points_2d_image", None)
        if ref_pts is None:
            ref_pts = results.get("pose_reference_points_2d_image", None)
    ref_pts = _to_pts2(ref_pts)

    # 1) 工作轮廓
    contour_centers = []
    contour_boxes = []
    for i, item in enumerate(measurements):
        meas = item.get("measurement", {}) if isinstance(item, dict) else {}
        work_contour = meas.get("work_contour", None)
        contour_pts = _to_contour_pts(work_contour)
        if len(contour_pts) >= 2:
            cv2.polylines(vis, [contour_pts.reshape(-1, 1, 2)], True, (0, 165, 255), 2, lineType=cv2.LINE_AA)
            cxy = np.mean(contour_pts.astype(np.float64), axis=0)
            contour_centers.append(cxy)
            x, y, w, h = cv2.boundingRect(contour_pts.reshape(-1, 1, 2))
            contour_boxes.append((x, y, w, h))
            cv2.putText(vis, f"W{i+1}", (int(round(cxy[0])) + 4, int(round(cxy[1])) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        else:
            contour_centers.append(None)
            contour_boxes.append(None)

    # 2) 亚像素边缘点集
    for pts in refined_pts_cache:
        _draw_points(pts, (0, 0, 255), radius=1)

    # 3) 直接椭圆中心
    _draw_centers(pred_ellipse, (255, 255, 0), "E", cv2.MARKER_TILTED_CROSS)

    # 4) 几何校正中心
    _draw_centers(pred_antipodal_edge, (0, 220, 255), "A", cv2.MARKER_STAR)
    _draw_centers(pred_joint_layout_edge, (255, 120, 0), "J", cv2.MARKER_TRIANGLE_UP)
    _draw_centers(pred_geo_pass1, (0, 255, 0), "G", cv2.MARKER_CROSS)

    # 5) 参考投影中心
    _draw_centers(ref_pts, (255, 0, 255), "R", cv2.MARKER_DIAMOND)

    # 图例
    legend_items = [
        ((0, 165, 255), "Work contour"), 
        ((0, 0, 255), "Subpixel edge set"),
        ((255, 255, 0), "Direct ellipse center"),
        ((0, 220, 255), "Antipodal edge center"),
        ((255, 120, 0), "Joint layout edge center"),
        ((0, 255, 0), "Geometry center"),
        ((255, 0, 255), "Reference projection center"),
    ]
    lx, ly = 12, 40
    for idx, (color, txt) in enumerate(legend_items):
        yy = ly + idx * 22
        cv2.line(vis, (lx, yy), (lx + 18, yy), color, 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (lx + 26, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    if title_text:
        cv2.putText(vis, str(title_text), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # ---------- 生成放大视图 ----------
    def _finite_point(pt):
        if pt is None:
            return None
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return None
        return arr[:2]

    def _pick_center(idx):
        candidates = []
        if idx < len(ref_pts):
            candidates.append(ref_pts[idx])
        if idx < len(pred_geo_pass1):
            candidates.append(pred_geo_pass1[idx])
        if idx < len(pred_ellipse):
            candidates.append(pred_ellipse[idx])
        if idx < len(contour_centers):
            candidates.append(contour_centers[idx])
        for c in candidates:
            c = _finite_point(c)
            if c is not None:
                return c
        return None

    n_lights = max(len(measurements), len(refined_pts_cache), len(pred_ellipse), len(pred_geo_pass1), len(ref_pts))
    if n_lights <= 0:
        return vis

    H, W = vis.shape[:2]
    zoom_tiles = []
    tile_size = int((2 * crop_radius + 1) * zoom_scale)
    margin = 10

    for i in range(n_lights):
        center = _pick_center(i)
        if center is None:
            continue
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = crop_radius
        if i < len(contour_boxes) and contour_boxes[i] is not None:
            _, _, bw, bh = contour_boxes[i]
            r = max(r, int(max(bw, bh) * 1.8))
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - (W - 1))
        pad_b = max(0, y2 - (H - 1))
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(W - 1, x2), min(H - 1, y2)
        patch = vis[sy1:sy2 + 1, sx1:sx2 + 1].copy()
        if any(v > 0 for v in [pad_l, pad_t, pad_r, pad_b]):
            patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # ensure square
        if patch.shape[0] != patch.shape[1]:
            m = max(patch.shape[0], patch.shape[1])
            pad_bottom = m - patch.shape[0]
            pad_right = m - patch.shape[1]
            patch = cv2.copyMakeBorder(patch, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        patch_zoom = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(vis, (max(0, x1), max(0, y1)), (min(W - 1, x2), min(H - 1, y2)), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(patch_zoom, (0, 0), (tile_size - 1, tile_size - 1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(patch_zoom, f"Light {i+1}  Zoom x{zoom_scale}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        zoom_tiles.append(patch_zoom)

    if not zoom_tiles:
        return vis

    cols = 2 if len(zoom_tiles) > 1 else 1
    rows = int(np.ceil(len(zoom_tiles) / cols))
    mosaic_h = rows * tile_size + (rows + 1) * margin
    mosaic_w = cols * tile_size + (cols + 1) * margin
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(zoom_tiles):
        rr = idx // cols
        cc = idx % cols
        y0 = margin + rr * (tile_size + margin)
        x0 = margin + cc * (tile_size + margin)
        mosaic[y0:y0 + tile_size, x0:x0 + tile_size] = tile

    cv2.putText(mosaic, "Zoomed light-source views for paper display", (margin, mosaic_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 将整图缩放到与 mosaic 接近宽度，组成上下拼接版
    target_w = max(mosaic_w, min(W, 1600))
    scale = target_w / float(W)
    full_h = int(round(H * scale))
    full_resized = cv2.resize(vis, (target_w, full_h), interpolation=cv2.INTER_AREA)
    if mosaic_w != target_w:
        pad_left = (target_w - mosaic_w) // 2
        pad_right = target_w - mosaic_w - pad_left
        mosaic = cv2.copyMakeBorder(mosaic, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    sep = np.full((12, target_w, 3), 32, dtype=np.uint8)
    composite = np.vstack([full_resized, sep, mosaic])
    return composite


def finalize_single_image_outputs(inputs_ctx, light_img_path, save_dir, show, roi_points,
                                  measurement_ctx, eval_ctx, results, final_measurement_vis=False):
    contours = measurement_ctx["contours"]
    pred_bbox = measurement_ctx["pred_bbox"]
    pred_gray_centroid = measurement_ctx["pred_gray_centroid"]
    pred_ellipse = measurement_ctx["pred_ellipse"]
    pred_geo_pass1 = measurement_ctx["pred_geo_pass1"]
    pred_geo_bias_pass1 = measurement_ctx["pred_geo_bias_pass1"]
    fusion_debug_pass1 = measurement_ctx["fusion_debug_pass1"]
    pass3a_aligned_true = eval_ctx["fusion"]["aligned"]

    # vis = build_visualization(
    #     inputs_ctx["light_img"],
    #     contours,
    #     pred_bbox,
    #     pred_gray_centroid,
    #     pred_ellipse,
    #     pred_geo_pass1,
    #     pred_geo_bias_pass1,
    #     pass3a_aligned_true,
    #     fusion_debug_list=fusion_debug_pass1,
    #     pose_reference_points=results.get("pose_reference_points_2d_image"),
    # )

    # if roi_points is not None:
    #     vis = roi.draw_roi(vis, roi_points)

    final_vis = None
    if final_measurement_vis:
        title_text = os.path.basename(light_img_path)
        final_vis = build_final_measurement_visualization(inputs_ctx, measurement_ctx, results=results, title_text=title_text)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(light_img_path))[0]

        # 先保存可视化图，再保存 JSON。
        # 这样即使 JSON 序列化因某个字段异常失败，也不会影响叠加图落盘。
        if final_vis is not None:
            vis_path = os.path.join(save_dir, f"{base_name}_final_measurement_vis.png")
            ok = cv2.imwrite(vis_path, final_vis)
            if ok:
                print(f"[INFO] 最终叠加可视化已保存到: {vis_path}")
            else:
                print(f"[WARN] 最终叠加可视化保存失败: {vis_path}")
        elif final_measurement_vis:
            print("[WARN] 已开启 --final_measurement_vis，但未生成叠加可视化图。")
        #保存单张
        # eval_file_path = os.path.join(save_dir, f"{base_name}_eval.json")
        # save_results_to_json(results, eval_file_path)

    if show and final_vis is not None:
        cv2.imshow("Final Measurement Sent To Geometry", final_vis)
        cv2.waitKey(0)

    return final_vis


def batch_process_by_chess(chess_img_path, cam_intr_path, save_dir, show=False, roi_points=None,
                           force_reselect_roi=False, final_measurement_vis=False, frontend_mode='legacy',
                           pose_source='light_only'):
    """批处理函数。chess_img_path 仅作为同组命名和 ROI 缓存入口，不参与算法位姿解算。"""
    with open(cam_intr_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)
    K = np.array(cam_data.get("new_K", cam_data.get("K")), dtype=np.float64)

    pose_source = str(pose_source)
    if pose_source != 'light_only':
        raise ValueError("主算法流程禁止使用棋盘格位姿；pose_source 必须为 light_only")

    if roi_points is None:
        print("[INFO] 加载ROI...")
        roi_points, roi_cache_path = _load_existing_or_select_roi(
            chess_img_path,
            save_dir,
            force_reselect=force_reselect_roi,
            allow_select=bool(force_reselect_roi),
        )
        print("[INFO] ROI加载流程完成")
    else:
        print(f"[INFO] 使用外部传入的ROI")
        roi_cache_path = None

    prefix, chess_idx, _ = roi.derive_group_info_from_chess(chess_img_path)

    def find_light_images_by_chess(chess_path):
        _, _, light_pattern = roi.derive_group_info_from_chess(chess_path)
        return sorted(glob.glob(light_pattern))

    light_files = find_light_images_by_chess(chess_img_path)
    if not light_files:
        raise RuntimeError("未找到与该棋盘格图匹配的一组光源图")

    print(f"[INFO] 棋盘图: {os.path.basename(chess_img_path)}")
    print(f"[INFO] 位姿来源: {pose_source}")
    print(f"[INFO] 匹配到 {len(light_files)} 张光源图")
    for p in light_files:
        print(f"  - {os.path.basename(p)}")

    batch_rows = []
    for idx, lf in enumerate(light_files, 1):
        print(f"\n[Batch] {idx}/{len(light_files)} -> {os.path.basename(lf)}")
        try:
            _, summary = process_single_image(
                chess_img_path,
                lf,
                K,
                save_dir=save_dir,
                show=show,
                roi_points=roi_points,
                final_measurement_vis=final_measurement_vis,
                frontend_mode=frontend_mode,
                pose_source=pose_source,
            )
            batch_rows.append(summary)
            summary = None
        except Exception as e:
            print(f"[ERROR] {os.path.basename(lf)} 处理失败: {e}")
            batch_rows.append({
                "light_image": os.path.basename(lf),
                "detected_points": None,
                "bbox_mean_error": None,
                "gray_mean_error": None,
                "weighted_gray_mean_error": None,
                "circle_mean_error": None,
                "ellipse_mean_error": None,
                "ellipse_backproj_mean_error": None,
                "homography_rectified_mean_error": None,
                "geo_mean_error": None,
                "geo_pose_bias_mean_error": None,
                "group_prefix": parse_pose_from_name(os.path.basename(lf)).get("group_prefix"),
                "distance": parse_pose_from_name(os.path.basename(lf)).get("distance"),
                "y_offset": parse_pose_from_name(os.path.basename(lf)).get("y_offset"),
                "pitch": parse_pose_from_name(os.path.basename(lf)).get("pitch"),
                "exp_index": parse_pose_from_name(os.path.basename(lf)).get("exp_index"),
                "geo_gain": None,
                "geo_bias_gain": None,
                "geo_effective": None,
                "geo_strong_effective": None,
                "error": str(e)
            })
        finally:
            try:
                if not show:
                    cv2.destroyAllWindows()
            except Exception:
                pass
            gc.collect()

    batch_result = {
        "group_prefix": prefix,
        "chess_index": chess_idx,
        "chessboard_image": os.path.basename(chess_img_path),
        "camera_params": os.path.basename(cam_intr_path),
        "pose_source": pose_source,
        "logic_note": "light_only mode uses the chess filename only for locating the image group; pose is estimated from each light image and known lamp geometry.",
        "num_light_images": len(light_files),
        "items": batch_rows
    }

    os.makedirs(save_dir, exist_ok=True)
    batch_json = os.path.join(save_dir, f"{prefix}_light_{chess_idx}_batch_summary.json")
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 批处理汇总已保存到: {batch_json}")
    return batch_result






def _joint_layout_make_geo_est(measurement, edge_set, joint_item, joint_debug):
    bbox_center = np.asarray(measurement.get("bbox_center", [np.nan, np.nan]), dtype=np.float64)
    ellipse_center = np.asarray(measurement.get("ellipse_center", bbox_center), dtype=np.float64)
    point_debug = dict(measurement.get("point_debug_base", {}))
    point_debug.update(edge_set.get("debug", {}))

    radial_mad = joint_item.get("radial_mad")
    center_dev = joint_item.get("center_deviation")
    common_radius = joint_item.get("common_radius_estimate")
    provisional_radius = joint_item.get("provisional_radius")
    fit_used_points = int(joint_item.get("point_count", 0))
    geo_center_img = np.asarray(joint_item.get("center_image", ellipse_center), dtype=np.float64)
    geo_cov = np.asarray(joint_item.get("cov_image", np.eye(2, dtype=np.float64) * 25.0), dtype=np.float64)

    point_debug["valid_ratio"] = 1.0 if np.all(np.isfinite(geo_center_img)) else 0.0
    point_debug["fit_residual_mean"] = None if radial_mad is None or not np.isfinite(radial_mad) else float(radial_mad)
    point_debug["fit_residual_median"] = None if radial_mad is None or not np.isfinite(radial_mad) else float(radial_mad)
    point_debug["fit_used_points"] = fit_used_points
    point_debug["circle_fit_rms"] = None if radial_mad is None or not np.isfinite(radial_mad) else float(radial_mad)
    point_debug["radius_cv"] = None
    if provisional_radius is not None and np.isfinite(provisional_radius) and common_radius is not None and np.isfinite(common_radius) and abs(common_radius) > 1e-12:
        point_debug["radius_cv"] = float(abs(provisional_radius - common_radius) / abs(common_radius))
    point_debug["layout_center_deviation"] = None if center_dev is None or not np.isfinite(center_dev) else float(center_dev)
    point_debug["common_radius_estimate"] = None if common_radius is None or not np.isfinite(common_radius) else float(common_radius)
    point_debug["provisional_radius"] = provisional_radius
    point_debug["joint_assignment_layout_index"] = int(joint_item.get("layout_index", -1))
    point_debug["joint_score_total"] = float(joint_debug.get("score", np.nan))
    point_debug["joint_best_second_gap"] = joint_debug.get("best_second_gap")
    point_debug["joint_confidence"] = float(joint_item.get("confidence", 0.0))
    point_debug["joint_risk"] = float(joint_item.get("risk", 1.0))
    point_debug["geometry_projected_center"] = geo_center_img.tolist()
    point_debug["geometry_output_center"] = geo_center_img.tolist()
    point_debug["geometry_gate"] = {
        "decision": "joint_planar_layout_projection",
        "blend_alpha": 1.0,
        "reliability": float(joint_item.get("confidence", 0.0)),
        "reference_center": ellipse_center.tolist(),
        "raw_geometry_center": geo_center_img.tolist(),
        "layout_index": int(joint_item.get("layout_index", -1)),
        "joint_score": float(joint_debug.get("score", np.nan)),
    }

    return {
        "center": geo_center_img,
        "cov": geo_cov,
        "point_debug": point_debug,
        "points_3d": None,
        "weights": None,
    }




def _normalize_contour_for_opencv(contour):
    c = np.asarray(contour)
    if c.size == 0:
        return c.astype(np.int32).reshape(0, 1, 2)
    if c.ndim == 2 and c.shape[1] == 2:
        c = c.reshape(-1, 1, 2)
    elif c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2:
        pass
    else:
        c = c.reshape(-1, 1, 2)
    return np.round(c).astype(np.int32)




def _solve_weighted_lstsq_safe(Aw, bw):
    Aw = np.asarray(Aw, dtype=np.float64)
    bw = np.asarray(bw, dtype=np.float64).reshape(-1)
    finite = np.all(np.isfinite(Aw), axis=1) & np.isfinite(bw)
    Aw = Aw[finite]
    bw = bw[finite]
    if Aw.shape[0] < max(3, Aw.shape[1]):
        return None, 'too_few_rows'
    try:
        s = np.linalg.svd(Aw, compute_uv=False)
        rank = int(np.count_nonzero(s > np.finfo(np.float64).eps * max(Aw.shape) * (s[0] if s.size else 1.0)))
    except np.linalg.LinAlgError:
        rank = 0
    if rank < Aw.shape[1]:
        return None, 'rank_deficient'
    try:
        sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        if np.all(np.isfinite(sol)):
            return sol.astype(np.float64), None
    except Exception:
        pass
    try:
        sol = np.linalg.pinv(Aw) @ bw
        if np.all(np.isfinite(sol)):
            return np.asarray(sol, dtype=np.float64), None
    except Exception:
        pass
    return None, 'svd_fail'


def fit_circle_on_plane_weighted_debug(points_3d, n_c, weights=None, iters=5, plane_basis=None):
    if points_3d is None:
        return None, {
            'fit_residual_mean': None,
            'fit_residual_median': None,
            'fit_radius': None,
            'fit_used_points': 0,
            'fit_center_cov_trace': None,
            'fit_status': 'no_input',
        }

    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if len(X) < 3:
        return None, {
            'fit_residual_mean': None,
            'fit_residual_median': None,
            'fit_radius': None,
            'fit_used_points': int(len(X)),
            'fit_center_cov_trace': None,
            'fit_status': 'too_few_points',
        }

    if weights is None:
        W = np.ones((len(X),), dtype=np.float64)
    else:
        W = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(W) != len(finite):
            W = np.ones((len(X),), dtype=np.float64)
        else:
            W = W[finite]
    W = np.clip(W, 1e-9, None)
    W = W / (np.mean(W) + 1e-12)

    if plane_basis is None:
        n, e1, e2 = _plane_basis_from_normal(n_c)
    else:
        n, e1, e2 = plane_basis
        n = np.asarray(n, dtype=np.float64).reshape(3)
        e1 = np.asarray(e1, dtype=np.float64).reshape(3)
        e2 = np.asarray(e2, dtype=np.float64).reshape(3)
    resid = None
    radius = np.nan
    uc = vc = 0.0
    fit_status = 'ok'

    for _ in range(max(2, int(iters))):
        X0 = np.average(X, axis=0, weights=W)
        diff = X - X0
        u = diff @ e1
        v = diff @ e2
        A = np.column_stack([u, v, np.ones_like(u)])
        b = -(u ** 2 + v ** 2)
        sw = np.sqrt(np.clip(W, 1e-12, None))
        Aw = A * sw[:, None]
        bw = b * sw
        sol, err = _solve_weighted_lstsq_safe(Aw, bw)
        if sol is None:
            fit_status = err or 'solve_fail'
            break
        Ac, Bc, Cc = sol
        uc = -0.5 * Ac
        vc = -0.5 * Bc
        val = uc * uc + vc * vc - Cc
        if not np.isfinite(val):
            fit_status = 'nonfinite_radius'
            break
        radius = float(np.sqrt(max(0.0, val)))
        r = np.sqrt((u - uc) ** 2 + (v - vc) ** 2)
        resid = r - radius
        if not np.all(np.isfinite(resid)):
            fit_status = 'nonfinite_residual'
            break
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-12
        robust = 1.0 / (1.0 + ((resid - med) / mad) ** 2)
        W = np.clip(W * robust, 1e-9, None)
        W = W / (np.mean(W) + 1e-12)
    else:
        X0 = np.average(X, axis=0, weights=W)

    if fit_status != 'ok' or resid is None or not np.isfinite(radius):
        center_3d = np.average(X, axis=0, weights=W)
        cov_uv = np.eye(2, dtype=np.float64) * 9.0
        dbg = {
            'fit_residual_mean': None,
            'fit_residual_median': None,
            'fit_radius': None if not np.isfinite(radius) else float(radius),
            'fit_used_points': int(len(X)),
            'fit_center_cov_trace': float(np.trace(cov_uv)),
            'fit_center_cov_2d': cov_uv.tolist(),
            'fit_status': fit_status,
        }
        return center_3d.astype(np.float64), dbg

    center_3d = X0 + uc * e1 + vc * e2
    try:
        J = np.column_stack([2.0 * (u - uc), 2.0 * (v - vc)])
        H = (J.T * W) @ J
        sigma2 = float(np.average((resid - np.average(resid, weights=W)) ** 2, weights=W))
        cov_uv = np.linalg.pinv(H + 1e-9 * np.eye(2)) * max(sigma2, 1e-9)
    except Exception:
        cov_uv = np.eye(2, dtype=np.float64) * 9.0
    dbg = {
        'fit_residual_mean': float(np.mean(np.abs(resid))),
        'fit_residual_median': float(np.median(np.abs(resid))),
        'fit_radius': float(radius),
        'fit_used_points': int(len(X)),
        'fit_center_cov_trace': float(np.trace(cov_uv)),
        'fit_center_cov_2d': cov_uv.tolist(),
        'fit_status': 'ok',
    }
    return center_3d.astype(np.float64), dbg


def fit_known_radius_circle_on_plane_weighted_debug(
    points_3d,
    n_c,
    target_radius_mm,
    weights=None,
    iters=12,
    plane_basis=None,
    center_prior_3d=None,
    center_prior_sigma_mm=None,
):
    if points_3d is None:
        return None, {
            'fit_residual_mean': None,
            'fit_residual_median': None,
            'fit_radius': None,
            'fit_target_radius': float(target_radius_mm),
            'fit_used_points': 0,
            'fit_center_cov_trace': None,
            'fit_status': 'no_input',
            'fit_model': 'known_radius_circle_on_plane',
        }

    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if len(X) < 3:
        return None, {
            'fit_residual_mean': None,
            'fit_residual_median': None,
            'fit_radius': None,
            'fit_target_radius': float(target_radius_mm),
            'fit_used_points': int(len(X)),
            'fit_center_cov_trace': None,
            'fit_status': 'too_few_points',
            'fit_model': 'known_radius_circle_on_plane',
        }

    if weights is None:
        W = np.ones((len(X),), dtype=np.float64)
    else:
        W = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(W) != len(finite):
            W = np.ones((len(X),), dtype=np.float64)
        else:
            W = W[finite]
    W = np.clip(W, 1e-9, None)
    W = W / (np.mean(W) + 1e-12)

    if plane_basis is None:
        n, e1, e2 = _plane_basis_from_normal(n_c)
    else:
        n, e1, e2 = plane_basis
        n = np.asarray(n, dtype=np.float64).reshape(3)
        e1 = np.asarray(e1, dtype=np.float64).reshape(3)
        e2 = np.asarray(e2, dtype=np.float64).reshape(3)

    X0 = np.average(X, axis=0, weights=W)
    Y = X - X0
    uv = np.column_stack([Y @ e1, Y @ e2]).astype(np.float64)
    target_radius = max(float(target_radius_mm), 1e-6)

    prior_uv = None
    if center_prior_3d is not None:
        prior3 = np.asarray(center_prior_3d, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(prior3)):
            prior_uv = np.array([(prior3 - X0) @ e1, (prior3 - X0) @ e2], dtype=np.float64)

    c = np.average(uv, axis=0, weights=W)
    free_center, free_dbg = fit_circle_on_plane_weighted_debug(X, n_c, weights=W, iters=4, plane_basis=(n, e1, e2))
    if free_center is not None and np.all(np.isfinite(free_center)):
        c = np.array([(free_center - X0) @ e1, (free_center - X0) @ e2], dtype=np.float64)
    if prior_uv is not None:
        c = prior_uv.copy()

    prior_sigma = float(center_prior_sigma_mm) if center_prior_sigma_mm is not None else max(4.0, 0.45 * target_radius)
    prior_sigma = max(prior_sigma, 1e-6)

    fit_status = 'ok'
    resid = None
    for _ in range(max(1, int(iters))):
        diff = uv - c.reshape(1, 2)
        r = np.linalg.norm(diff, axis=1)
        valid = r > 1e-9
        if np.count_nonzero(valid) < 3:
            fit_status = 'degenerate_radius_jacobian'
            break
        resid = r - target_radius
        J = np.zeros((len(uv), 2), dtype=np.float64)
        J[valid, :] = -diff[valid, :] / r[valid, None]
        sw = np.sqrt(np.clip(W, 1e-12, None))
        A = J * sw[:, None]
        b = -resid * sw
        if prior_uv is not None:
            A_prior = np.eye(2, dtype=np.float64) / prior_sigma
            b_prior = -(c - prior_uv) / prior_sigma
            A = np.vstack([A, A_prior])
            b = np.concatenate([b, b_prior])
        try:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            fit_status = 'solve_fail'
            break
        if not np.all(np.isfinite(delta)):
            fit_status = 'nonfinite_delta'
            break
        step = float(np.linalg.norm(delta))
        c = c + delta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-12
        robust = 1.0 / (1.0 + ((resid - med) / (2.5 * mad)) ** 2)
        W = np.clip(W * robust, 1e-9, None)
        W = W / (np.mean(W) + 1e-12)
        if step < 1e-6:
            break

    diff = uv - c.reshape(1, 2)
    r = np.linalg.norm(diff, axis=1)
    resid = r - target_radius
    if not np.all(np.isfinite(resid)):
        fit_status = 'nonfinite_residual'

    center_3d = X0 + float(c[0]) * e1 + float(c[1]) * e2
    try:
        valid = r > 1e-9
        J = np.zeros((len(uv), 2), dtype=np.float64)
        J[valid, :] = -diff[valid, :] / r[valid, None]
        H = (J.T * W) @ J
        sigma2 = float(np.average((resid - np.average(resid, weights=W)) ** 2, weights=W))
        cov_uv = np.linalg.pinv(H + 1e-9 * np.eye(2)) * max(sigma2, 1e-9)
    except Exception:
        cov_uv = np.eye(2, dtype=np.float64) * 9.0
    dbg = {
        'fit_residual_mean': float(np.mean(np.abs(resid))) if resid is not None else None,
        'fit_residual_median': float(np.median(np.abs(resid))) if resid is not None else None,
        'fit_radius': float(target_radius),
        'fit_free_radius_before_constraint': free_dbg.get('fit_radius') if isinstance(free_dbg, dict) else None,
        'fit_target_radius': float(target_radius),
        'fit_used_points': int(len(X)),
        'fit_center_cov_trace': float(np.trace(cov_uv)),
        'fit_center_cov_2d': cov_uv.tolist(),
        'fit_status': fit_status,
        'fit_model': 'known_radius_circle_on_plane',
        'fit_center_prior_used': bool(prior_uv is not None),
        'fit_center_prior_sigma_mm': float(prior_sigma),
    }
    return center_3d.astype(np.float64), dbg


_legacy_estimate_subpixel_edge_set = globals().get('estimate_subpixel_edge_set')

def _effective_arc_coverage_from_angles(angles):
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if ang.size < 2:
        return 0.0
    ang = np.sort(np.mod(ang, 2.0 * np.pi))
    gaps = np.diff(np.r_[ang, ang[0] + 2.0 * np.pi])
    if gaps.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.max(gaps) / (2.0 * np.pi), 0.0, 1.0))


def _angle_balance_from_angles(angles):
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if ang.size == 0:
        return 0.0
    z = np.exp(1j * ang)
    resultant = float(np.abs(np.mean(z)))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


def _ellipse_normal_from_point(ellipse, point_xy):
    if ellipse is None:
        return None
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    (cx, cy), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, np.finfo(np.float64).eps)
    b = max(float(ma) * 0.5, np.finfo(np.float64).eps)
    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (p - np.array([cx, cy], dtype=np.float64)) @ R
    n_local = np.array([q[0] / (a * a), q[1] / (b * b)], dtype=np.float64)
    n_img = R.T @ n_local
    n_norm = float(np.linalg.norm(n_img))
    if not np.isfinite(n_norm) or n_norm <= np.finfo(np.float64).eps:
        n_img = p - np.array([cx, cy], dtype=np.float64)
        n_norm = float(np.linalg.norm(n_img))
        if not np.isfinite(n_norm) or n_norm <= np.finfo(np.float64).eps:
            return None
    return (n_img / n_norm).astype(np.float64)


def estimate_antipodal_edge_midpoint_center(edge_set, measurement):
    pts = edge_set.get('points') if isinstance(edge_set, dict) else None
    work_ellipse = measurement.get('work_ellipse') if isinstance(measurement, dict) else None
    selected_boundary_center = None
    if work_ellipse is not None:
        try:
            selected_boundary_center = np.asarray(work_ellipse[0], dtype=np.float64).reshape(2)
        except Exception:
            selected_boundary_center = None
    fallback = np.asarray(
        selected_boundary_center if selected_boundary_center is not None
        else measurement.get('ellipse_center', measurement.get('bbox_center', [np.nan, np.nan]))
        if isinstance(measurement, dict) else [np.nan, np.nan],
        dtype=np.float64,
    ).reshape(2)
    debug = {
        'method': 'antipodal_subpixel_edge_midpoint_v1',
        'used': False,
        'reason': None,
        'pair_count': 0,
        'center_shift_from_selected_boundary_px': None,
        'midpoint_residual_median_px': None,
        'midpoint_residual_mad_px': None,
        'angular_tolerance_rad': None,
    }
    if pts is None or work_ellipse is None:
        debug['reason'] = 'missing_edge_points_or_selected_boundary'
        return {'center': fallback.copy(), 'debug': debug}

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]
    if len(pts) < 12:
        debug['reason'] = 'too_few_subpixel_edge_points'
        return {'center': fallback.copy(), 'debug': debug}

    edge_weights = edge_set.get('weights') if isinstance(edge_set, dict) else None
    if edge_weights is not None:
        edge_weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
        if edge_weights.size == valid.size:
            edge_weights = edge_weights[valid]
        else:
            edge_weights = np.ones((len(pts),), dtype=np.float64)
    else:
        edge_weights = np.ones((len(pts),), dtype=np.float64)
    edge_weights = np.clip(edge_weights, 1e-9, None)

    angles = _ellipse_param_angles(work_ellipse, pts)
    if angles.size != len(pts):
        debug['reason'] = 'angle_parametrization_failed'
        return {'center': fallback.copy(), 'debug': debug}
    angles = np.mod(angles, 2.0 * np.pi)
    order = np.argsort(angles)
    a0 = angles[order]
    p0 = pts[order]
    w0 = edge_weights[order]
    a_ext = np.concatenate([a0, a0 + 2.0 * np.pi])
    p_ext = np.vstack([p0, p0])
    w_ext = np.concatenate([w0, w0])
    tol = float(np.clip(4.0 * (2.0 * np.pi / max(len(pts), 1)), 0.10, 0.42))
    debug['angular_tolerance_rad'] = tol

    mids = []
    pair_weights = []
    angular_gaps = []
    for theta, p, wi in zip(a0, p0, w0):
        target = theta + np.pi
        k = int(np.searchsorted(a_ext, target))
        if k <= 0 or k >= len(a_ext):
            continue
        left = k - 1
        right = k
        gap_left = float(target - a_ext[left])
        gap_right = float(a_ext[right] - target)
        span = float(a_ext[right] - a_ext[left])
        if gap_left < -1e-9 or gap_right < -1e-9 or span <= 1e-9:
            continue
        if max(gap_left, gap_right) > tol:
            continue
        frac = float(np.clip((target - a_ext[left]) / span, 0.0, 1.0))
        pj = (1.0 - frac) * p_ext[left] + frac * p_ext[right]
        if not np.all(np.isfinite(pj)):
            continue
        angular_gap = max(gap_left, gap_right)
        angular_score = float(np.exp(-0.5 * (angular_gap / max(tol, 1e-9)) ** 2))
        wj = (1.0 - frac) * w_ext[left] + frac * w_ext[right]
        mids.append(0.5 * (p + pj))
        pair_weights.append(float(np.sqrt(max(wi * wj, 1e-18)) * angular_score))
        angular_gaps.append(float(angular_gap))

    if len(mids) < 6:
        debug['reason'] = 'too_few_antipodal_pairs'
        debug['pair_count'] = int(len(mids))
        return {'center': fallback.copy(), 'debug': debug}

    mids = np.asarray(mids, dtype=np.float64).reshape(-1, 2)
    pair_weights = np.clip(np.asarray(pair_weights, dtype=np.float64).reshape(-1), 1e-9, None)
    med = np.median(mids, axis=0)
    resid0 = np.linalg.norm(mids - med.reshape(1, 2), axis=1)
    mad = float(np.median(np.abs(resid0 - np.median(resid0))) + 1e-9)
    robust_scale = max(0.18, 3.0 * mad)
    robust_w = pair_weights * (1.0 + (resid0 / robust_scale) ** 2) ** -1
    if np.any(np.isfinite(robust_w)) and float(np.sum(robust_w)) > 1e-12:
        robust_w = robust_w / float(np.sum(robust_w))
        center = np.sum(mids * robust_w.reshape(-1, 1), axis=0)
    else:
        center = med

    midpoint_center = center.astype(np.float64)

    harmonic_center = None
    harmonic_debug = {'used': False, 'reason': 'not_attempted'}
    try:
        (cx, cy), (MA, ma), ang = work_ellipse
        a = max(float(MA) * 0.5, np.finfo(np.float64).eps)
        b = max(float(ma) * 0.5, np.finfo(np.float64).eps)
        th = np.deg2rad(float(ang))
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
        xn = q[:, 0] / a
        yn = q[:, 1] / b
        rn = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan2(yn, xn)
        radial_resid = rn - 1.0
        X = np.column_stack([np.cos(theta), np.sin(theta)])
        w_h = edge_weights / (np.mean(edge_weights) + 1e-12)
        med_r = float(np.median(radial_resid))
        mad_r = float(np.median(np.abs(radial_resid - med_r)) + 1e-9)
        robust_r = (1.0 + ((radial_resid - med_r) / max(3.0 * mad_r, 1e-4)) ** 2) ** -1
        w_h = np.clip(w_h * robust_r, 1e-9, None)
        A = X.T @ (w_h[:, None] * X)
        yv = X.T @ (w_h * radial_resid)
        coef = np.linalg.solve(A + np.eye(2, dtype=np.float64) * 1e-9, yv)
        delta_q = np.array([coef[0] * a, coef[1] * b], dtype=np.float64)
        delta_img = delta_q @ R.T
        harmonic_center = np.array([cx, cy], dtype=np.float64) + delta_img
        harmonic_debug = {
            'used': bool(np.all(np.isfinite(harmonic_center))),
            'reason': 'first_harmonic_radial_center_shift',
            'coef_cos': float(coef[0]),
            'coef_sin': float(coef[1]),
            'radial_residual_median': float(np.median(radial_resid)),
            'radial_residual_mad': float(mad_r),
            'harmonic_shift_px': float(np.linalg.norm(delta_img)),
        }
    except Exception as e:
        harmonic_center = None
        harmonic_debug = {'used': False, 'reason': f'exception:{type(e).__name__}'}

    if harmonic_center is not None and np.all(np.isfinite(harmonic_center)):
        try:
            axes = np.asarray(work_ellipse[1], dtype=np.float64).reshape(2)
            ecc_ratio = float(np.max(axes) / max(np.min(axes), 1e-9))
        except Exception:
            ecc_ratio = 1.0
        max_gain = float(globals().get('ANTIPODAL_EDGE_CORRECTION_MAX_GAIN', 0.25))
        gain_center = float(globals().get('ANTIPODAL_EDGE_ECC_GAIN_CENTER', 1.18))
        gain_width = max(float(globals().get('ANTIPODAL_EDGE_ECC_GAIN_WIDTH', 0.025)), 1e-6)
        gain = float(max_gain / (1.0 + np.exp(-(ecc_ratio - gain_center) / gain_width)))
        center = fallback + gain * (harmonic_center.astype(np.float64) - fallback)
        output_mode = 'damped_first_harmonic_antipodal_center'
    else:
        gain = 1.0
        center = midpoint_center
        output_mode = 'robust_midpoint_center'

    resid = np.linalg.norm(mids - midpoint_center.reshape(1, 2), axis=1)
    debug.update({
        'used': True,
        'reason': output_mode,
        'pair_count': int(len(mids)),
        'angular_gap_median_rad': float(np.median(np.asarray(angular_gaps, dtype=np.float64))) if angular_gaps else None,
        'midpoint_residual_median_px': float(np.median(resid)),
        'midpoint_residual_mad_px': float(np.median(np.abs(resid - np.median(resid))) + 1e-9),
        'midpoint_center': midpoint_center.tolist(),
        'correction_gain': float(gain),
        'selected_boundary_axis_ratio': float(ecc_ratio) if 'ecc_ratio' in locals() else None,
        'undamped_harmonic_center': None if harmonic_center is None or not np.all(np.isfinite(harmonic_center)) else harmonic_center.astype(np.float64).tolist(),
        'harmonic_debug': harmonic_debug,
        'center_shift_from_selected_boundary_px': float(np.linalg.norm(center - fallback)) if np.all(np.isfinite(fallback)) else None,
    })
    return {'center': center.astype(np.float64), 'debug': debug}








def build_candidate_measurements(measurements):
    conf_scores_pass1, risk_scores_pass1, uncertainty_debug_pass1 = _diagnostic_risk_scores(measurements)

    pred_bbox = _points_array([m['bbox_center'] for m in measurements])
    pred_gray_centroid = _points_array([m['gray_center'] for m in measurements])
    pred_weighted_gray = _points_array([m.get('weighted_gray_center', [np.nan, np.nan]) for m in measurements])
    pred_circle = _points_array([m.get('contour_circle_center', [np.nan, np.nan]) for m in measurements])
    pred_ellipse = _points_array([m.get('direct_ellipse_center', m.get('ellipse_center', [np.nan, np.nan])) for m in measurements])
    pred_step4_work_ellipse = _points_array([
        (m.get('step4_work_ellipse_ablation', {}) or {}).get('center_xy', [np.nan, np.nan])
        if (m.get('step4_work_ellipse_ablation', {}) or {}).get('available', False)
        else [np.nan, np.nan]
        for m in measurements
    ])
    pred_ellipse_backproj = _points_array([m.get('ellipse_backproj_center', [np.nan, np.nan]) for m in measurements])
    pred_homography_rectified = _points_array([m.get('homography_rectified_center', [np.nan, np.nan]) for m in measurements])
    pred_layout_constrained_pnp = _points_array([m.get('layout_constrained_pnp_center', [np.nan, np.nan]) for m in measurements])
    pred_antipodal_edge = _points_array([m.get('antipodal_edge_center', [np.nan, np.nan]) for m in measurements])
    pred_joint_layout_edge = _points_array([m.get('joint_layout_edge_center', [np.nan, np.nan]) for m in measurements])
    pred_final_edge_layout = _points_array([m.get('final_edge_layout_center', [np.nan, np.nan]) for m in measurements])
    pred_physical_edge_circle = _points_array([m.get('physical_edge_circle_center', [np.nan, np.nan]) for m in measurements])
    pred_geo_pass1 = _points_array([m['geo_center'] for m in measurements])
    pred_geo_bias_pass1 = _points_array([m.get('fused_center', m.get('geo_center')) for m in measurements])
    refined_pts_cache = [m.get('refined_pts') for m in measurements]
    ellipse_cache = [m.get('ellipse_cache') for m in measurements]
    point_debug_pass1 = [m.get('point_debug', {}) for m in measurements]
    fusion_debug_pass1 = [m.get('fusion_debug', {}) for m in measurements]
    contours = [m.get('contour') for m in measurements]

    gate_decisions = [m.get('point_debug', {}).get('geometry_gate', {}).get('decision', 'geometry_direct') for m in measurements]
    gate_counts = {}
    for d in gate_decisions:
        gate_counts[d] = gate_counts.get(d, 0) + 1
    strategy_name = 'joint_arc_radius_constrained_physical_solvability_gate_geometry_v16'

    priority_point_debug = {
        'strategy': strategy_name,
        'num_points': int(len(measurements)),
        'uncertainty_debug': uncertainty_debug_pass1,
        'geometry_gate_decisions': gate_decisions,
        'geometry_gate_counts': gate_counts,
    }
    conf_debug_pass1 = {
        'strategy': strategy_name,
        'uncertainty_debug': uncertainty_debug_pass1,
        'geometry_gate_decisions': gate_decisions,
        'geometry_gate_counts': gate_counts,
    }
    pose_bias_debug_pass1 = {'enabled': True, 'strategy': 'raw_geometry_diagnostic_output'}

    return {
        'measurements': measurements,
        'contours': contours,
        'pred_bbox': pred_bbox,
        'pred_gray_centroid': pred_gray_centroid,
        'pred_weighted_gray': pred_weighted_gray,
        'pred_circle': pred_circle,
        'pred_ellipse': pred_ellipse,
        'pred_step4_work_ellipse': pred_step4_work_ellipse,
        'pred_ellipse_backproj': pred_ellipse_backproj,
        'pred_homography_rectified': pred_homography_rectified,
        'pred_layout_constrained_pnp': pred_layout_constrained_pnp,
        'pred_antipodal_edge': pred_antipodal_edge,
        'pred_joint_layout_edge': pred_joint_layout_edge,
        'pred_final_edge_layout': pred_final_edge_layout,
        'pred_physical_edge_circle': pred_physical_edge_circle,
        'pred_geo_pass1': pred_geo_pass1,
        'pred_geo_bias_pass1': pred_geo_bias_pass1,
        'refined_pts_cache': refined_pts_cache,
        'ellipse_cache': ellipse_cache,
        'point_debug_pass1': point_debug_pass1,
        'fusion_debug_pass1': fusion_debug_pass1,
        'conf_scores_pass1': conf_scores_pass1,
        'risk_scores_pass1': risk_scores_pass1,
        'priority_point_debug': priority_point_debug,
        'conf_debug_pass1': conf_debug_pass1,
        'pose_bias_debug_pass1': pose_bias_debug_pass1,
        'uncertainty_debug_pass1': uncertainty_debug_pass1,
    }
def parse_args():
    parser = argparse.ArgumentParser(description="基于灯具平面布局、已知半径和亚像素边缘的批量处理流程")
    parser.add_argument("--chess", type=str, default=chess_img_path, help="棋盘格图路径，例如 xxx_chess_001.jpg")
    parser.add_argument("--cam", type=str, default=cam_intr_path, help="相机内参 json 路径")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="结果输出目录")
    parser.add_argument("--light", type=str, default=None, help="可选：单张光源图路径；不填则自动按命名规则批处理整组")
    parser.add_argument("--show", action="store_true", help="是否显示可视化窗口")
    parser.add_argument("--force_reselect_roi", action="store_true", help="是否强制重新选择ROI（忽略缓存）")
    parser.add_argument("--final_measurement_vis", action="store_true", help="是否可视化并保存最终送入几何校正的椭圆边缘")
    parser.add_argument("--frontend_mode", type=str, default="legacy", choices=["legacy", "randomized_physical"], help="前端边缘测量模式：legacy 为原流程；randomized_physical 为物理约束随机轮廓假设搜索")
    parser.add_argument("--pose_source", type=str, default="light_only", choices=["light_only"], help="位姿来源固定为 light_only；棋盘格只允许在独立评估脚本中建立 ground truth")
    parser.add_argument("--roi_json", type=str, default=None, help="可选：外部ROI JSON；light_only 模式下不会自动从棋盘图选择ROI")
    return parser.parse_args()


def _load_roi_points_json(path):
    if not path:
        return None
    data = roi.load_roi_from_json(path)
    if data and data.get("roi_points"):
        return np.asarray(data["roi_points"], dtype=np.int32)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    points = raw.get("roi_points", raw.get("points", raw))
    return np.asarray(points, dtype=np.int32).reshape(-1, 2)


def _find_existing_roi_cache(chess_img_path, save_dir=None):
    chess_stem = os.path.splitext(os.path.basename(chess_img_path))[0]
    target_name = f"{chess_stem}_roi.json"
    candidates = []
    if save_dir:
        candidates.append(os.path.join(save_dir, "roi_cache", target_name))

    workspace_root = os.getcwd()
    result_root = os.path.join(workspace_root, "photos", "zed", "result")
    candidates.extend(glob.glob(os.path.join(result_root, "*", "roi_cache", target_name)))
    candidates.extend(glob.glob(os.path.join(workspace_root, "**", "roi_cache", target_name), recursive=True))

    seen = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return None


def _load_existing_or_select_roi(chess_img_path, save_dir, force_reselect=False, allow_select=False):
    if force_reselect and allow_select:
        print("[ROI] 强制重新选择ROI")
        return roi.load_or_select_roi(chess_img_path, save_dir, force_reselect=True)

    cache_path = _find_existing_roi_cache(chess_img_path, save_dir=save_dir)
    if cache_path:
        data = roi.load_roi_from_json(cache_path)
        if data and data.get("roi_points"):
            print(f"[ROI] 从已有缓存加载ROI: {cache_path}")
            return np.asarray(data["roi_points"], dtype=np.int32), cache_path

    if allow_select:
        print("[ROI] 未找到已有缓存，开始手动选择ROI")
        return roi.load_or_select_roi(chess_img_path, save_dir, force_reselect=force_reselect)

    print("[ROI] 未找到已有ROI缓存；将使用整张光源图")
    return None, None


def main():
    args = parse_args()
    with open(args.cam, "r", encoding="utf-8") as f:
        cam_data = json.load(f)
    K = np.array(cam_data.get("new_K", cam_data.get("K")), dtype=np.float64)

    roi_points = _load_roi_points_json(args.roi_json) if args.roi_json else None

    if args.light:
        if roi_points is None:
            print("[INFO] 加载ROI...")
            roi_points, roi_cache_path = _load_existing_or_select_roi(
                args.chess,
                args.save_dir,
                force_reselect=args.force_reselect_roi,
                allow_select=bool(args.force_reselect_roi),
            )
            print("[INFO] ROI加载流程完成")
        process_single_image(
            args.chess,
            args.light,
            K,
            save_dir=args.save_dir,
            show=args.show,
            roi_points=roi_points,
            final_measurement_vis=args.final_measurement_vis,
            frontend_mode=args.frontend_mode,
            pose_source=args.pose_source,
        )
    else:
        batch_process_by_chess(
            args.chess,
            args.cam,
            args.save_dir,
            show=args.show,
            roi_points=roi_points,
            force_reselect_roi=args.force_reselect_roi,
            final_measurement_vis=args.final_measurement_vis,
            frontend_mode=args.frontend_mode,
            pose_source=args.pose_source,
        )





# ============================================================
# v5 覆盖：接入单灯稳定肩部边模块，并改写 arc weight 与风险
# ============================================================
from fast_normal_offset_edge_selector_v7_fast_visualized import build_selected_measurements_for_pipeline













# ============================================================
# v8 受限 Sampson 细化：前端有效弧段 -> 图像平面初始椭圆
# 只做受限细化，不收缩支撑点；支撑不足时回退到前端初值
# ============================================================

def _angle_density_weights(angles):
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    n = len(ang)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)
    order = np.argsort(np.mod(ang, 2.0 * np.pi))
    ang_s = np.mod(ang[order], 2.0 * np.pi)
    ext = np.r_[ang_s[-1] - 2.0 * np.pi, ang_s, ang_s[0] + 2.0 * np.pi]
    gaps_left = ang_s - ext[:-2]
    gaps_right = ext[2:] - ang_s
    w_sorted = 0.5 * (gaps_left + gaps_right)
    w_sorted = np.clip(w_sorted, np.finfo(np.float64).eps, None)
    w = np.zeros_like(w_sorted)
    w[order] = w_sorted
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float64)


def _canonical_ellipse_from_cv(ellipse):
    if ellipse is None:
        return None
    (cx, cy), (MA, ma), ang = ellipse
    a = 0.5 * float(MA)
    b = 0.5 * float(ma)
    phi = np.deg2rad(float(ang))
    if a < b:
        a, b = b, a
        phi += 0.5 * np.pi
    phi = float((phi + np.pi) % np.pi - 0.5 * np.pi)
    a = max(a, np.finfo(np.float64).eps)
    b = max(b, np.finfo(np.float64).eps)
    return np.array([float(cx), float(cy), a, b, phi], dtype=np.float64)


def _cv_ellipse_from_canonical(can):
    cx, cy, a, b, phi = [float(v) for v in can]
    return ((cx, cy), (2.0 * a, 2.0 * b), float(np.rad2deg(phi)))


def _theta_from_canonical(can):
    cx, cy, a, b, phi = [float(v) for v in can]
    tb = np.log(max(b, np.finfo(np.float64).eps))
    td = np.log(max(a - b, np.finfo(np.float64).eps))
    return np.array([cx, cy, tb, td, phi], dtype=np.float64)


def _canonical_from_theta(theta):
    cx, cy, tb, td, phi = [float(v) for v in theta]
    b = np.exp(tb)
    a = b + np.exp(td)
    phi = float((phi + np.pi) % np.pi - 0.5 * np.pi)
    return np.array([cx, cy, a, b, phi], dtype=np.float64)


def _conic_from_canonical(can):
    cx, cy, a, b, phi = [float(v) for v in can]
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    D = np.diag([1.0 / max(a * a, np.finfo(np.float64).eps), 1.0 / max(b * b, np.finfo(np.float64).eps)])
    Q = R @ D @ R.T
    A = float(Q[0, 0])
    B = float(2.0 * Q[0, 1])
    C = float(Q[1, 1])
    Dd = float(-2.0 * A * cx - B * cy)
    Ee = float(-B * cx - 2.0 * C * cy)
    F = float(A * cx * cx + B * cx * cy + C * cy * cy - 1.0)
    return np.array([A, B, C, Dd, Ee, F], dtype=np.float64)


def _sampson_signed_residual(points_xy, can):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    A, B, C, Dd, Ee, F = _conic_from_canonical(can)
    x = pts[:, 0]
    y = pts[:, 1]
    f = A * x * x + B * x * y + C * y * y + Dd * x + Ee * y + F
    gx = 2.0 * A * x + B * y + Dd
    gy = B * x + 2.0 * C * y + Ee
    den = np.sqrt(np.maximum(gx * gx + gy * gy, np.finfo(np.float64).eps))
    return (f / den).astype(np.float64)


def _robust_scale_mad(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad + np.sqrt(np.finfo(np.float64).eps))


def _jacobian_fd(points_xy, theta, weights):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    base = _sampson_signed_residual(pts, _canonical_from_theta(th))
    J = np.zeros((len(base), len(th)), dtype=np.float64)
    step0 = np.sqrt(np.finfo(np.float64).eps)
    for k in range(len(th)):
        h = step0 * (abs(th[k]) + 1.0)
        tp = th.copy(); tp[k] += h
        tm = th.copy(); tm[k] -= h
        rp = _sampson_signed_residual(pts, _canonical_from_theta(tp))
        rm = _sampson_signed_residual(pts, _canonical_from_theta(tm))
        J[:, k] = (rp - rm) / (2.0 * h)
    sw = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None))
    return base * sw, J * sw[:, None]


def _robust_contiguous_inliers(angles, weights):
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(ang)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if n <= 6:
        return np.ones((n,), dtype=bool)
    order = np.argsort(np.mod(ang, 2.0 * np.pi))
    ws = w[order]
    score_u8 = np.clip(np.round(ws / (np.max(ws) + 1e-12) * 255.0), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(score_u8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    keep_sorted = score_u8 >= thr
    if np.count_nonzero(keep_sorted) < 6:
        med = np.median(ws)
        keep_sorted = ws >= med
    if np.count_nonzero(keep_sorted) < 6:
        keep_sorted[:] = True
    doubled = np.r_[keep_sorted, keep_sorted]
    best_len = 0; best_start = 0; cur_len = 0; cur_start = 0
    for i, flag in enumerate(doubled):
        if flag:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len and cur_len <= n:
                best_len = cur_len; best_start = cur_start
        else:
            cur_len = 0
    best = np.zeros((n,), dtype=bool)
    if best_len == 0:
        best[:] = keep_sorted
    else:
        idx = (np.arange(best_start, best_start + best_len) % n).astype(int)
        best[idx] = True
    out = np.zeros((n,), dtype=bool)
    out[order] = best
    return out


def _support_observability(angles, weights):
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(ang)
    if n < 6:
        return {'sufficient': False, 'rank': 0, 'cond': np.inf, 'n_eff': float(n), 'coverage': 0.0}
    if len(w) != n:
        w = np.ones((n,), dtype=np.float64)
    w = np.clip(w, np.finfo(np.float64).eps, None)
    M = np.column_stack([np.ones((n,), dtype=np.float64), np.cos(ang), np.sin(ang), np.cos(2.0 * ang), np.sin(2.0 * ang)])
    Mw = M * np.sqrt(w / (np.mean(w) + 1e-12))[:, None]
    try:
        s = np.linalg.svd(Mw, compute_uv=False)
    except np.linalg.LinAlgError:
        return {'sufficient': False, 'rank': 0, 'cond': np.inf, 'n_eff': 0.0, 'coverage': 0.0}
    smax = float(s[0]) if s.size else 0.0
    tol = max(np.sqrt(np.finfo(np.float64).eps) * smax, np.finfo(np.float64).eps)
    rank = int(np.sum(s > tol))
    smin = float(s[-1]) if s.size else 0.0
    cond = float(smax / max(smin, np.finfo(np.float64).eps)) if smax > 0.0 else np.inf
    n_eff = float((np.sum(w) ** 2) / max(np.sum(w * w), np.finfo(np.float64).eps))
    coverage = _effective_arc_coverage_from_angles(ang) if n >= 2 else 0.0
    sufficient = bool(rank >= 5 and np.isfinite(cond) and cond <= 1.0 / np.sqrt(np.finfo(np.float64).eps) and n_eff >= 6.0)
    return {'sufficient': sufficient, 'rank': rank, 'cond': cond, 'n_eff': n_eff, 'coverage': float(coverage)}




# ============================================================
# v7 最终版：单一 process_single_image + v6 shoulder 模块主路径
# ============================================================

from fast_normal_offset_edge_selector_v7_fast_visualized import build_selected_measurements_for_pipeline


def _optimize_profile_edges_by_global_radius(profile_pairs, weights, proj_ctx, n_c, target_radius_mm=None):
    if proj_ctx is None or not profile_pairs:
        return None, {"used": False, "reason": "no_profile_pairs_or_projection_context"}
    if target_radius_mm is None:
        target_radius_mm = float(globals().get("LIGHT_RADIUS_MM", 15.0))
    edge_pts = []
    shoulder_pts = []
    for pair in profile_pairs:
        e = np.asarray(pair.get("edge_pt"), dtype=np.float64).reshape(2)
        s = np.asarray(pair.get("shoulder_pt"), dtype=np.float64).reshape(2)
        if np.all(np.isfinite(e)) and np.all(np.isfinite(s)):
            edge_pts.append(e)
            shoulder_pts.append(s)
    if len(edge_pts) < 6:
        return None, {"used": False, "reason": "too_few_profile_pairs", "num_pairs": int(len(edge_pts))}

    edge_pts = np.asarray(edge_pts, dtype=np.float64)
    shoulder_pts = np.asarray(shoulder_pts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(edge_pts):
        weights = np.ones((len(edge_pts),), dtype=np.float64)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / (np.mean(weights) + 1e-12)

    best = None
    alphas = np.linspace(-0.15, 1.15, 27, dtype=np.float64)
    for alpha in alphas:
        pts_uv = edge_pts + float(alpha) * (shoulder_pts - edge_pts)
        pts3d, proj_weights, dbg_proj = project_pixels_to_plane_filtered(
            pts_uv,
            proj_ctx["K"],
            proj_ctx["n_c"],
            proj_ctx["d_c"],
            den_eps=0.0,
            keep_ratio=0.0,
            proj_ctx=proj_ctx,
        )
        if pts3d is None or len(pts3d) < 6:
            continue
        w = np.asarray(proj_weights, dtype=np.float64).reshape(-1)
        if len(w) == len(weights):
            w = np.clip(w * weights, 1e-9, None)
        center3d, fit_dbg = fit_circle_on_plane_weighted_debug(
            pts3d,
            n_c,
            weights=w,
            iters=5,
            plane_basis=proj_ctx.get("plane_basis"),
        )
        if center3d is None or not isinstance(fit_dbg, dict):
            continue
        fit_radius = fit_dbg.get("fit_radius")
        fit_med = fit_dbg.get("fit_residual_median")
        if fit_radius is None or fit_med is None or not np.isfinite(fit_radius) or not np.isfinite(fit_med):
            continue
        radius_err = abs(float(fit_radius) - float(target_radius_mm))
        score = float(fit_med + 0.45 * radius_err)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "alpha": float(alpha),
                "points": pts_uv,
                "fit_radius": float(fit_radius),
                "fit_residual_median": float(fit_med),
                "radius_error_mm": float(radius_err),
                "projection_debug": dbg_proj,
            }

    if best is None:
        return None, {"used": False, "reason": "no_valid_alpha_solution", "num_pairs": int(len(edge_pts))}
    return best["points"].astype(np.float64), {
        "used": True,
        "method": "global_profile_alpha_radius_search",
        "alpha": best["alpha"],
        "score": best["score"],
        "fit_radius_mm": best["fit_radius"],
        "fit_residual_median_mm": best["fit_residual_median"],
        "radius_error_mm": best["radius_error_mm"],
        "num_pairs": int(len(edge_pts)),
    }


def estimate_subpixel_edge_set(gray_light_proc, measurement, proj_ctx=None):
    pre = measurement.get('point_debug_base', {}).get('fast_normal_offset_edge', {}) if isinstance(measurement.get('point_debug_base', {}), dict) else {}
    arc_points = np.asarray(pre.get('arc_points', []), dtype=np.float64).reshape(-1, 2) if pre else np.empty((0, 2), dtype=np.float64)
    arc_angles = np.asarray(pre.get('arc_angles', []), dtype=np.float64).reshape(-1) if pre else np.empty((0,), dtype=np.float64)
    arc_weights = np.asarray(pre.get('arc_weights', []), dtype=np.float64).reshape(-1) if pre else np.empty((0,), dtype=np.float64)
    rep = measurement.get('point_debug_base', {}).get('representation', {}) if isinstance(measurement.get('point_debug_base', {}), dict) else {}
    plateau_support_mean = float(rep.get('plateau_support_mean', 0.0) or 0.0)
    confidence_light = float(rep.get('confidence_light', 0.0) or 0.0)
    work_ellipse = measurement.get('work_ellipse')

    if arc_points.size == 0 or work_ellipse is None:
        return {
            'points': None,
            'weights': None,
            'sigmas': [],
            'qualities': [],
            'sample_angles': [],
            'expected_count': 0,
            'edge_point_count': 0,
            'edge_fit_success_ratio': 0.0,
            'edge_sigma_mean': None,
            'debug': {
                'edge_expected_count': 0,
                'edge_point_count': 0,
                'edge_fit_success_ratio': 0.0,
                'edge_sigma_mean': None,
                'edge_plateau_ratio_mean': None,
                'edge_quality_mean': None,
                'effective_arc_coverage': 0.0,
                'angle_balance': 0.0,
                'edge_source': 'none',
                'confidence_light': confidence_light,
                'edge_fallback_used': False,
                'edge_search_half_len_mean': None,
            },
        }

    if arc_weights.size != len(arc_points) or not np.all(np.isfinite(arc_weights)):
        arc_weights = np.ones((len(arc_points),), dtype=np.float64)
    arc_weights = np.clip(arc_weights, 1e-9, None)
    arc_weights = arc_weights / (np.mean(arc_weights) + 1e-12)

    if arc_angles.size != len(arc_points):
        arc_angles = _ellipse_param_angles(work_ellipse, arc_points)

    refined_pts = []
    refined_weights = []
    sigmas = []
    qualities = []
    sample_angles = []
    plateau_ratios = []
    edge_levels = []
    search_half_lens = []
    radius_consistency_errors = []
    radius_correction_pixels = []
    profile_pairs = []

    rep_center_cam = np.asarray(rep.get('assigned_pose_center_camera', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    use_global_radius_edge = os.environ.get("PAGC_EDGE_RADIUS_GLOBAL", "0").strip().lower() in ("1", "true", "yes", "on")
    use_radius_relocalization = (not use_global_radius_edge) and proj_ctx is not None and np.all(np.isfinite(rep_center_cam))

    for pt, ang, w in zip(arc_points, arc_angles, arc_weights):
        normal = _ellipse_normal_from_point(work_ellipse, pt)
        if normal is None or not np.all(np.isfinite(normal)):
            continue
        search_half_len = _auto_search_half_len(work_ellipse, gray_light_proc, pt, normal)
        fit = _fit_edge_from_profile(gray_light_proc, pt, normal, search_half_len)
        if fit is None:
            continue
        edge_pt = np.asarray(fit.get('point', [np.nan, np.nan]), dtype=np.float64)
        if edge_pt.shape != (2,) or not np.all(np.isfinite(edge_pt)):
            continue

        nrm = float(np.linalg.norm(normal))
        normal_unit = normal / max(nrm, 1e-12)
        sign_flip = 1.0 if str(fit.get('used_profile_direction', 'forward')) == 'forward' else -1.0
        edge_offset = sign_flip * float(fit.get('edge_offset', 0.0))
        shoulder_offset = sign_flip * float(fit.get('plateau_end_offset', fit.get('edge_offset', 0.0)))
        edge_uv_for_pair = np.asarray(pt, dtype=np.float64).reshape(2) + edge_offset * normal_unit
        shoulder_uv_for_pair = np.asarray(pt, dtype=np.float64).reshape(2) + shoulder_offset * normal_unit
        if np.all(np.isfinite(edge_uv_for_pair)) and np.all(np.isfinite(shoulder_uv_for_pair)):
            profile_pairs.append({
                "edge_pt": edge_uv_for_pair,
                "shoulder_pt": shoulder_uv_for_pair,
            })

        radius_dbg = {'enabled': False, 'reason': 'disabled'}
        if use_radius_relocalization:
            corrected_pt, radius_dbg = _radius_consistent_point_on_normal(
                base_pt=pt,
                normal=normal,
                fit=fit,
                center_cam=rep_center_cam,
                proj_ctx=proj_ctx,
                target_radius_mm=LIGHT_RADIUS_MM,
            )
            if corrected_pt is not None and np.all(np.isfinite(corrected_pt)):
                edge_pt = corrected_pt

        sigma = float(fit.get('sigma', 2.5))
        quality = float(fit.get('fit_quality', 0.0))
        radius_err = radius_dbg.get('radius_error_mm', np.nan)
        corr_px = radius_dbg.get('correction_px', 0.0)
        radius_consistency_errors.append(float(radius_err) if np.isfinite(radius_err) else np.nan)
        radius_correction_pixels.append(float(corr_px) if np.isfinite(corr_px) else 0.0)
        consistency_term = 1.0 / (1.0 + max(float(radius_err), 0.0)) if np.isfinite(radius_err) else 1.0
        refined_pts.append(edge_pt)
        refined_weights.append(float(max(w, 1e-9) * max(quality, 1e-9) * max(consistency_term, 1e-9)))
        sigmas.append(sigma)
        qualities.append(quality)
        sample_angles.append(float(ang))
        plateau_ratios.append(float(fit.get('plateau_ratio', plateau_support_mean)))
        edge_levels.append(float(fit.get('edge_level', np.nan)))
        search_half_lens.append(int(search_half_len))

    edge_expected_count = int(len(arc_points))
    edge_point_count = int(len(refined_pts))
    edge_fit_success_ratio = float(edge_point_count / max(edge_expected_count, 1))
    radius_err_finite = [float(v) for v in radius_consistency_errors if np.isfinite(v)]

    if edge_point_count < 6:
        return {
            'points': None,
            'weights': None,
            'sigmas': sigmas,
            'qualities': qualities,
            'sample_angles': sample_angles,
            'expected_count': edge_expected_count,
            'edge_point_count': edge_point_count,
            'edge_fit_success_ratio': edge_fit_success_ratio,
            'edge_sigma_mean': float(np.mean(sigmas)) if sigmas else None,
            'debug': {
                'edge_expected_count': edge_expected_count,
                'edge_point_count': edge_point_count,
                'edge_fit_success_ratio': edge_fit_success_ratio,
                'edge_sigma_mean': float(np.mean(sigmas)) if sigmas else None,
                'edge_quality_mean': float(np.mean(qualities)) if qualities else None,
                'edge_plateau_ratio_mean': float(np.mean(plateau_ratios)) if plateau_ratios else plateau_support_mean,
                'edge_level_mean': float(np.nanmean(edge_levels)) if edge_levels else None,
                'effective_arc_coverage': float(_effective_arc_coverage_from_angles(sample_angles)) if len(sample_angles) >= 2 else 0.0,
                'angle_balance': float(_angle_balance_from_angles(sample_angles)) if sample_angles else 0.0,
                'edge_fallback_used': True,
                'edge_search_half_len_mean': float(np.mean(search_half_lens)) if search_half_lens else None,
                'radius_consistency_error_mean': float(np.mean(radius_err_finite)) if radius_err_finite else None,
                'radius_consistency_correction_px_mean': float(np.mean(radius_correction_pixels)) if radius_correction_pixels else 0.0,
                'edge_source': 'joint_arc_profile_relocalization_insufficient',
                'confidence_light': confidence_light,
            },
        }

    pts_arr = np.asarray(refined_pts, dtype=np.float64).reshape(-1, 2)
    w_arr = np.asarray(refined_weights, dtype=np.float64).reshape(-1)
    w_arr = np.clip(w_arr, 1e-9, None)
    w_arr = w_arr / (np.mean(w_arr) + 1e-12)
    global_radius_debug = {"used": False, "reason": "disabled"}
    if use_global_radius_edge and proj_ctx is not None:
        opt_pts, global_radius_debug = _optimize_profile_edges_by_global_radius(
            profile_pairs,
            w_arr,
            proj_ctx,
            proj_ctx["n_c"],
            target_radius_mm=LIGHT_RADIUS_MM,
        )
        if opt_pts is not None and len(opt_pts) == len(pts_arr):
            pts_arr = opt_pts
    effective_arc_coverage = _effective_arc_coverage_from_angles(sample_angles) if len(sample_angles) >= 2 else 0.0
    angle_balance = _angle_balance_from_angles(sample_angles) if sample_angles else 0.0

    return {
        'points': pts_arr,
        'weights': w_arr,
        'sigmas': sigmas,
        'qualities': qualities,
        'sample_angles': list(sample_angles),
        'expected_count': edge_expected_count,
        'edge_point_count': edge_point_count,
        'edge_fit_success_ratio': edge_fit_success_ratio,
        'edge_sigma_mean': float(np.mean(sigmas)) if sigmas else None,
        'debug': {
            'edge_expected_count': edge_expected_count,
            'edge_point_count': edge_point_count,
            'edge_fit_success_ratio': edge_fit_success_ratio,
            'edge_sigma_mean': float(np.mean(sigmas)) if sigmas else None,
            'edge_quality_mean': float(np.mean(qualities)) if qualities else None,
            'edge_plateau_ratio_mean': float(np.mean(plateau_ratios)) if plateau_ratios else plateau_support_mean,
            'edge_level_mean': float(np.nanmean(edge_levels)) if edge_levels else None,
            'effective_arc_coverage': float(effective_arc_coverage),
            'angle_balance': float(angle_balance),
            'edge_fallback_used': False,
            'edge_search_half_len_mean': float(np.mean(search_half_lens)) if search_half_lens else None,
            'radius_consistency_error_mean': float(np.mean(radius_err_finite)) if radius_err_finite else None,
            'radius_consistency_correction_px_mean': float(np.mean(radius_correction_pixels)) if radius_correction_pixels else 0.0,
            'global_radius_edge': global_radius_debug,
            'edge_source': 'joint_arc_profile_global_radius_alpha' if global_radius_debug.get("used") else 'joint_arc_profile_relocalized_radius_consistent',
            'confidence_light': confidence_light,
        },
    }
# ============================================================
# v10 覆盖：高曝光保守模式 + 更快的受限 Sampson 初始椭圆细化
# ============================================================

def _support_subsample_by_angle(points_xy, angles, weights=None):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    n = len(pts)
    if n == 0 or len(ang) != n:
        return pts, ang, np.ones((n,), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights is None or len(np.asarray(weights).reshape(-1)) != n:
        w = np.ones((n,), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, np.finfo(np.float64).eps, None)
    order = np.argsort(np.mod(ang, 2.0 * np.pi))
    pts_s = pts[order]
    ang_s = np.mod(ang[order], 2.0 * np.pi)
    w_s = w[order]
    obs = _support_observability(ang_s, w_s)
    n_eff = max(obs.get('n_eff', float(n)), 1.0)
    m = int(np.clip(np.ceil(np.sqrt(n_eff) * np.log1p(n_eff)), 16, n))
    if m >= n:
        return pts_s, ang_s, w_s
    qs = (np.arange(m, dtype=np.float64) + 0.5) / float(m)
    idx = np.clip(np.round(qs * (n - 1)).astype(int), 0, n - 1)
    idx = np.unique(idx)
    return pts_s[idx], ang_s[idx], w_s[idx]


def _ellipse_geom_scale(ellipse):
    if ellipse is None:
        return 1.0
    (_, _), (MA, ma), _ = ellipse
    return max(np.finfo(np.float64).eps, 0.25 * min(float(MA), float(ma)))


def _ellipse_shape_stats(ellipse):
    if ellipse is None:
        return np.nan, np.nan
    return float(ellipse_area_from_fit(ellipse)), float(ellipse_ecc_ratio_from_fit(ellipse))


def _bloom_badness_from_stats(scale, saturation_ratio, plateau_support, coverage, sigma_mean):
    sat = float(np.clip(saturation_ratio, 0.0, 1.0)) if np.isfinite(saturation_ratio) else 0.0
    plat = float(np.clip(plateau_support, 0.0, 1.0)) if np.isfinite(plateau_support) else 0.0
    cov = float(np.clip(coverage, 0.0, 1.0)) if np.isfinite(coverage) else 0.0
    sigma_ratio = float(sigma_mean / max(scale, np.finfo(np.float64).eps)) if np.isfinite(sigma_mean) else 0.0
    sigma_bad = float(sigma_ratio / (1.0 + sigma_ratio))
    return float(np.clip(plat * np.sqrt(max(sat, 0.0)) * np.sqrt(max(cov, 0.0)) * (0.5 + 0.5 * sigma_bad), 0.0, 1.0))


def _fit_ellipse_sampson_constrained(points_xy, base_weights=None, init_ellipse=None, support_angles=None, exposure_badness=0.0):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 6:
        return None
    if base_weights is None or len(np.asarray(base_weights).reshape(-1)) != n:
        base_w = np.ones((n,), dtype=np.float64)
    else:
        base_w = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    base_w = np.clip(base_w, np.finfo(np.float64).eps, None)
    if support_angles is not None and len(np.asarray(support_angles).reshape(-1)) == n:
        base_w = base_w * _angle_density_weights(support_angles)
    base_w = base_w / (np.mean(base_w) + 1e-12)

    if init_ellipse is None:
        init_ellipse = _fit_ellipse_safe_from_pts(pts)
        if init_ellipse is None:
            init_ellipse = _fit_ellipse_best(pts)
        if init_ellipse is None:
            return None
    can0 = _canonical_ellipse_from_cv(init_ellipse)
    if can0 is None:
        return None
    theta0 = _theta_from_canonical(can0)
    theta = theta0.copy()

    if support_angles is not None and len(np.asarray(support_angles).reshape(-1)) == n:
        obs_all = _support_observability(support_angles, base_w)
        if not obs_all.get('sufficient', False):
            return {
                'ellipse': init_ellipse,
                'canonical': can0,
                'residuals': _sampson_signed_residual(pts, can0),
                'weights': base_w,
                'inlier_mask': np.ones((n,), dtype=bool),
                'accepted': False,
                'debug': {'mode': 'fallback', 'reason': 'support_insufficient_initial', 'obs_all': obs_all, 'obs_keep': obs_all, 'inlier_ratio': 1.0, 'inlier_count': int(n), 'sampson_init_median': float(np.median(np.abs(_sampson_signed_residual(pts, can0)))), 'sampson_final_median': float(np.median(np.abs(_sampson_signed_residual(pts, can0)))), 'irls_steps': 0}
            }
        pts_opt, ang_opt, w_opt = _support_subsample_by_angle(pts, support_angles, base_w)
    else:
        obs_all = {'sufficient': True, 'rank': 5, 'cond': 1.0, 'n_eff': float(n), 'coverage': np.nan}
        pts_opt, ang_opt, w_opt = pts, None, base_w

    score0 = float(np.median(np.abs(_sampson_signed_residual(pts, can0))))
    area0, ecc0 = _ellipse_shape_stats(init_ellipse)
    hist = []
    total_w = w_opt.copy()
    obs_cov = float(obs_all.get('coverage', np.nan)) if isinstance(obs_all, dict) else np.nan
    obs_neff = float(obs_all.get('n_eff', float(len(pts_opt)))) if isinstance(obs_all, dict) else float(len(pts_opt))
    support_factor = 0.0 if not np.isfinite(obs_cov) else float(np.clip(obs_cov, 0.0, 1.0))
    neff_factor = float(np.clip(np.log1p(max(obs_neff, 1.0)) / np.log(65.0), 0.25, 1.0))
    exposure_factor = float(np.clip(exposure_badness, 0.0, 1.0))
    fast_factor = float(np.clip(0.45 * support_factor + 0.35 * neff_factor + 0.20 * (1.0 - exposure_factor), 0.0, 1.0))
    max_outer = 1 + int(np.rint(2.0 * (1.0 - fast_factor)))
    max_outer = int(np.clip(max_outer, 1, 3))
    max_inner = 1 if fast_factor >= 0.5 else 2
    prior_gain = 1.0 + float(np.clip(exposure_badness, 0.0, 1.0))

    for _ in range(max_outer):
        can = _canonical_from_theta(theta)
        resid = _sampson_signed_residual(pts_opt, can)
        scale = _robust_scale_mad(resid)
        robust_w = 1.0 / (1.0 + (resid / scale) ** 2)
        total_w = np.clip(w_opt * robust_w, np.finfo(np.float64).eps, None)
        total_w = total_w / (np.mean(total_w) + 1e-12)
        rw, Jw = _jacobian_fd(pts_opt, theta, total_w)
        col = np.sqrt(np.sum(Jw * Jw, axis=0)) + np.sqrt(np.finfo(np.float64).eps)
        lam = prior_gain * float(np.median(np.abs(rw)) + np.sqrt(np.finfo(np.float64).eps))
        prior_r = np.sqrt(lam) * col * (theta - theta0)
        prior_J = np.sqrt(lam) * np.diag(col)
        A = np.vstack([Jw, prior_J])
        b = -np.r_[rw, prior_r]
        try:
            step, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        theta = theta + step
        hist.append(float(np.median(np.abs(resid))))
        if np.linalg.norm(step) <= np.sqrt(np.finfo(np.float64).eps) * (1.0 + np.linalg.norm(theta)):
            break

    can1 = _canonical_from_theta(theta)
    resid1_all = _sampson_signed_residual(pts, can1)
    score1 = float(np.median(np.abs(resid1_all)))
    ellipse1 = _cv_ellipse_from_canonical(can1)
    area1, ecc1 = _ellipse_shape_stats(ellipse1)
    shape_penalty = 0.0
    if np.isfinite(area0) and np.isfinite(area1) and area0 > 0.0 and area1 > 0.0:
        shape_penalty += float(abs(np.log(area1 / area0)))
    if np.isfinite(ecc0) and np.isfinite(ecc1) and ecc0 > 0.0 and ecc1 > 0.0:
        shape_penalty += float(abs(np.log(ecc1 / ecc0)))
    score1_pen = score1 + float(np.clip(exposure_badness, 0.0, 1.0)) * shape_penalty

    if support_angles is not None and len(np.asarray(support_angles).reshape(-1)) == n:
        keep = _robust_contiguous_inliers(support_angles, base_w)
        obs_keep = _support_observability(np.asarray(support_angles)[keep], base_w[keep]) if np.count_nonzero(keep) >= 6 else {'sufficient': False, 'rank': 0, 'cond': np.inf, 'n_eff': 0.0, 'coverage': 0.0}
    else:
        keep = np.ones((n,), dtype=bool)
        obs_keep = {'sufficient': True, 'rank': 5, 'cond': 1.0, 'n_eff': float(n), 'coverage': np.nan}

    accept = bool(obs_keep['sufficient'] and np.isfinite(score1_pen) and score1_pen <= score0)
    if accept:
        pts2 = pts[keep]
        if support_angles is not None and len(np.asarray(support_angles).reshape(-1)) == n:
            ang2 = np.asarray(support_angles, dtype=np.float64)[keep]
            w2 = base_w[keep]
            pts2, ang2, w2 = _support_subsample_by_angle(pts2, ang2, w2)
        else:
            w2 = base_w[keep]
        theta2 = theta.copy()
        for _ in range(max_inner):
            resid2 = _sampson_signed_residual(pts2, _canonical_from_theta(theta2))
            scale2 = _robust_scale_mad(resid2)
            robust_w2 = 1.0 / (1.0 + (resid2 / scale2) ** 2)
            ww2 = np.clip(w2 * robust_w2, np.finfo(np.float64).eps, None)
            ww2 = ww2 / (np.mean(ww2) + 1e-12)
            rw2, Jw2 = _jacobian_fd(pts2, theta2, ww2)
            col2 = np.sqrt(np.sum(Jw2 * Jw2, axis=0)) + np.sqrt(np.finfo(np.float64).eps)
            lam2 = prior_gain * float(np.median(np.abs(rw2)) + np.sqrt(np.finfo(np.float64).eps))
            prior_r2 = np.sqrt(lam2) * col2 * (theta2 - theta0)
            prior_J2 = np.sqrt(lam2) * np.diag(col2)
            A2 = np.vstack([Jw2, prior_J2])
            b2 = -np.r_[rw2, prior_r2]
            try:
                step2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step2)):
                break
            theta2 = theta2 + step2
            if np.linalg.norm(step2) <= np.sqrt(np.finfo(np.float64).eps) * (1.0 + np.linalg.norm(theta2)):
                break
        can_fin = _canonical_from_theta(theta2)
        ellipse = _cv_ellipse_from_canonical(can_fin)
        final_keep = keep
        mode = 'accepted'
        reason = 'sampson_constrained_refine_fast'
    else:
        can_fin = can0
        ellipse = init_ellipse
        final_keep = np.ones((n,), dtype=bool)
        mode = 'fallback'
        reason = 'support_insufficient_or_no_improvement'
    final_resid = _sampson_signed_residual(pts, can_fin)
    return {
        'ellipse': ellipse,
        'canonical': can_fin,
        'residuals': final_resid,
        'weights': base_w,
        'inlier_mask': final_keep,
        'accepted': accept,
        'debug': {
            'mode': mode,
            'reason': reason,
            'sampson_init_median': score0,
            'sampson_final_median': float(np.median(np.abs(final_resid))),
            'irls_steps': int(len(hist)),
            'obs_all': obs_all,
            'obs_keep': obs_keep,
            'inlier_ratio': float(np.mean(final_keep)),
            'inlier_count': int(np.count_nonzero(final_keep)),
            'exposure_badness': float(np.clip(exposure_badness, 0.0, 1.0)),
            'subsample_count': int(len(pts_opt)),
        }
    }


def build_candidate_measurement_from_preselected(selected_item, gray_light_proc):
    contour = _normalize_contour_for_opencv(selected_item["source_contour"])
    stats = _contour_pixel_stats_local(gray_light_proc, contour)
    x, y, w, h = stats['bbox']
    bbox_center = np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    gray_center = gray_centroid_in_contour(gray_light_proc, contour)
    weighted_gray_center = weighted_gray_centroid_in_contour(gray_light_proc, contour)
    contour_circle_center = contour_circle_center_from_contour(contour)
    contour_area = float(cv2.contourArea(contour))

    pix = stats['pixels']
    brightness_mean = float(np.mean(pix)) if pix.size else None
    brightness_peak = float(np.max(pix)) if pix.size else None
    saturation_level = float(np.percentile(pix, 99.5)) if pix.size else None
    saturation_ratio = float(np.mean(pix >= saturation_level)) if pix.size and saturation_level is not None else None

    source_ellipse = selected_item.get("source_ellipse")
    if source_ellipse is None:
        source_ellipse = fit_ellipse_from_contour(contour)
    init_ellipse = selected_item.get("work_ellipse")
    if init_ellipse is None:
        init_ellipse = source_ellipse
    work_contour = selected_item.get("work_contour")
    if work_contour is None:
        work_contour = contour
    work_contour = _normalize_contour_for_opencv(work_contour)

    arc_points = np.asarray(selected_item.get("arc_points", []), dtype=np.float64).reshape(-1, 2)
    arc_angles = np.asarray(selected_item.get("arc_angles", []), dtype=np.float64).reshape(-1)
    arc_weights = np.asarray(selected_item.get("arc_weights", []), dtype=np.float64).reshape(-1)
    if arc_weights.size != len(arc_points):
        arc_weights = np.ones((len(arc_points),), dtype=np.float64)

    rep_dbg = dict(selected_item.get("rep_dbg", {}))
    plateau_support_mean = float(rep_dbg.get('plateau_support_mean', 0.0) or 0.0)
    coverage = float(rep_dbg.get('effective_arc_coverage', 0.0) or 0.0)
    sigma_hint = np.nan
    if len(arc_points) >= 2:
        diffs = np.linalg.norm(np.diff(arc_points, axis=0), axis=1)
        sigma_hint = float(np.median(diffs)) if diffs.size else np.nan
    exposure_badness = _bloom_badness_from_stats(_ellipse_geom_scale(init_ellipse), float(saturation_ratio) if saturation_ratio is not None else np.nan, plateau_support_mean, coverage, sigma_hint)

    sampson_fit = None
    work_ellipse = init_ellipse
    if len(arc_points) >= 6 and init_ellipse is not None:
        quick_resid = float(_ellipse_contour_residual(init_ellipse, arc_points)) if len(arc_points) >= 6 else np.inf
        obs_cov = float(_support_observability(arc_angles, arc_weights).get('coverage', np.nan)) if len(arc_angles) == len(arc_points) and len(arc_points) >= 6 else np.nan
        run_sampson = not (np.isfinite(quick_resid) and np.isfinite(obs_cov) and quick_resid <= 0.25 and obs_cov >= 0.88 and exposure_badness <= 0.12)
        if run_sampson:
            sampson_fit = _fit_ellipse_sampson_constrained(
                arc_points,
                base_weights=arc_weights,
                init_ellipse=init_ellipse,
                support_angles=arc_angles if len(arc_angles) == len(arc_points) else None,
                exposure_badness=exposure_badness,
            )
        if sampson_fit is not None and sampson_fit.get('ellipse') is not None and sampson_fit.get('accepted', False):
            work_ellipse = sampson_fit['ellipse']

    direct_ellipse_center = np.asarray(source_ellipse[0], dtype=np.float64) if source_ellipse is not None else bbox_center.copy()

    if source_ellipse is None and work_ellipse is None:
        return {
            "contour": contour,
            "bbox_center": bbox_center,
            "gray_center": gray_center,
            "weighted_gray_center": weighted_gray_center,
            "contour_circle_center": contour_circle_center,
            "direct_ellipse_center": direct_ellipse_center,
            "source_ellipse": None,
            "work_ellipse": None,
            "work_contour": contour,
            "ellipse_center": bbox_center.copy(),
            "ellipse_cov": np.eye(2, dtype=np.float64) * 25.0,
            "point_debug_base": {
                "valid_ratio": 0.0,
                "fit_residual_mean": None,
                "fit_residual_median": None,
                "fit_used_points": 0,
                "brightness_mean": brightness_mean,
                "brightness_peak": brightness_peak,
                "saturation_level": saturation_level,
                "saturation_ratio": saturation_ratio,
                "edge_sigma_mean": None,
                "edge_point_count": 0,
                "edge_fit_success_ratio": 0.0,
                "contour_area": contour_area,
                "bbox": [int(x), int(y), int(w), int(h)],
                "representation": {"used": False, "reason": "ellipse_fail"},
                "source_ellipse": None,
                "work_ellipse": None,
                "measurement_mode": {"num_samples": 0, "search_half_len": 0, "den_eps": 0.0, "keep_ratio_3d": 0.0, "auto": True},
            },
        }
    if source_ellipse is None:
        source_ellipse = work_ellipse
    if work_ellipse is None:
        work_ellipse = source_ellipse

    pts_cont = contour.reshape(-1, 2).astype(np.float64)
    ell_center = robust_trimmed_ellipse_center(pts_cont, work_ellipse[0])

    rep_dbg["used"] = True
    rep_dbg["reason"] = rep_dbg.get("reason", "single_light_stable_shoulder_edge_v6")
    rep_dbg["selected_arc_points_count"] = int(len(arc_points))
    rep_dbg["selected_arc_angles_count"] = int(len(arc_angles))
    rep_dbg['exposure_badness'] = float(exposure_badness)
    if sampson_fit is not None:
        rep_dbg["sampson_refine"] = sampson_fit.get("debug", {})
        rep_dbg["sampson_refine_accepted"] = bool(sampson_fit.get("accepted", False))

    measurement = {
        "contour": contour,
        "bbox_center": bbox_center,
        "gray_center": gray_center,
        "weighted_gray_center": weighted_gray_center,
        "contour_circle_center": contour_circle_center,
        "direct_ellipse_center": direct_ellipse_center,
        "source_ellipse": source_ellipse,
        "work_ellipse": work_ellipse,
        "work_contour": work_contour,
        "ellipse_center": np.asarray(ell_center, dtype=np.float64),
        "ellipse_cov": estimate_ellipse_center_covariance(work_ellipse, work_contour),
        "point_debug_base": {
            "brightness_mean": brightness_mean,
            "brightness_peak": brightness_peak,
            "saturation_level": saturation_level,
            "saturation_ratio": saturation_ratio,
            "contour_area": contour_area,
            "bbox": [int(x), int(y), int(w), int(h)],
            "representation": rep_dbg,
            "source_ellipse": source_ellipse,
            "work_ellipse": work_ellipse,
            "measurement_mode": {
                "num_samples": int(_auto_sample_count_from_ellipse(work_ellipse)) if work_ellipse is not None else 0,
                "search_half_len": 0,
                "den_eps": 0.0,
                "keep_ratio_3d": 0.0,
                "auto": True,
                "ellipse_area": ellipse_area_from_fit(work_ellipse) if work_ellipse is not None else None,
                "ecc_ratio": ellipse_ecc_ratio_from_fit(work_ellipse) if work_ellipse is not None else None,
                "brightness_mean": brightness_mean,
                "saturation_ratio": saturation_ratio,
            },
            "fast_normal_offset_edge": {
                "arc_points": np.asarray(arc_points, dtype=np.float64).tolist(),
                "arc_angles": np.asarray(arc_angles, dtype=np.float64).tolist(),
                "arc_weights": np.asarray(arc_weights, dtype=np.float64).tolist(),
            },
        },
    }
    return measurement



def weighted_gray_centroid_in_contour(gray, contour, eps=1.0):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.uint8)
    contour_local = contour.copy()
    contour_local[:, :, 0] -= x
    contour_local[:, :, 1] -= y
    cv2.drawContours(mask, [contour_local], -1, 255, thickness=-1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    pix = roi[ys, xs].astype(np.float64) + 1.0
    init_center = gray_centroid_in_contour(gray, contour)
    gx = xs.astype(np.float64) + float(x)
    gy = ys.astype(np.float64) + float(y)
    dist = np.sqrt((gx - init_center[0]) ** 2 + (gy - init_center[1]) ** 2) + float(max(eps, 1e-6))
    wgt = pix / dist
    denom = float(np.sum(wgt))
    if not np.isfinite(denom) or denom <= 1e-12:
        return init_center.astype(np.float64)
    cx = float(np.sum(gx * wgt) / denom)
    cy = float(np.sum(gy * wgt) / denom)
    return np.array([cx, cy], dtype=np.float64)


def contour_circle_center_from_contour(contour):
    c = np.asarray(contour, dtype=np.float32)
    if c.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    try:
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        return np.array([float(cx), float(cy)], dtype=np.float64)
    except Exception:
        x, y, w, h = cv2.boundingRect(np.asarray(contour, dtype=np.int32))
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)


def _uniform_subsample_points(points_xy, max_points=240):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= max_points:
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=np.float64)
    idx = np.clip(np.round(idx).astype(np.int32), 0, pts.shape[0] - 1)
    idx = np.unique(idx)
    return pts[idx]


def _fit_min_enclosing_circle_on_plane(points_3d, plane_basis, weights=None):
    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    if X.shape[0] < 3:
        return None, {'fit_status': 'too_few_points', 'fit_used_points': int(X.shape[0])}
    if weights is None:
        W = np.ones((X.shape[0],), dtype=np.float64)
    else:
        W = np.asarray(weights, dtype=np.float64).reshape(-1)
        if W.size != X.shape[0]:
            W = np.ones((X.shape[0],), dtype=np.float64)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(W)
    X = X[valid]
    W = W[valid]
    if X.shape[0] < 3:
        return None, {'fit_status': 'too_few_valid_points', 'fit_used_points': int(X.shape[0])}
    W = np.clip(W, 1e-9, None)
    W = W / max(float(np.sum(W)), 1e-12)
    _, e1, e2 = plane_basis
    X0 = np.average(X, axis=0, weights=W)
    uv = np.column_stack([(X - X0) @ e1, (X - X0) @ e2]).astype(np.float32)
    try:
        (uc, vc), radius = cv2.minEnclosingCircle(uv.reshape(-1, 1, 2))
        center_3d = X0 + float(uc) * e1 + float(vc) * e2
        dbg = {
            'fit_status': 'ok',
            'fit_used_points': int(X.shape[0]),
            'fit_radius': float(radius),
            'fit_center_cov_trace': None,
        }
        return center_3d.astype(np.float64), dbg
    except Exception as e:
        return None, {'fit_status': f'min_enclosing_circle_fail:{type(e).__name__}', 'fit_used_points': int(X.shape[0])}


def compute_additional_baselines_for_measurement(measurement, gray_light_proc, proj_ctx=None):
    out = {
        'weighted_gray_center': np.asarray(measurement.get('weighted_gray_center', [np.nan, np.nan]), dtype=np.float64),
        'contour_circle_center': np.asarray(measurement.get('contour_circle_center', [np.nan, np.nan]), dtype=np.float64),
        'ellipse_backproj_center': np.array([np.nan, np.nan], dtype=np.float64),
        'homography_rectified_center': np.array([np.nan, np.nan], dtype=np.float64),
        'baseline_debug': {},
    }

    direct_ellipse = measurement.get('source_ellipse', None)
    plane_basis = None if proj_ctx is None else proj_ctx.get('plane_basis', None)

    if proj_ctx is not None and direct_ellipse is not None:
        try:
            num_samples = int(np.clip(_auto_sample_count_from_ellipse(direct_ellipse), 96, 240))
            ellipse_uv = sample_ellipse_points(direct_ellipse, num_samples=num_samples)
            pts3d, weights, proj_dbg = project_pixels_to_plane_filtered(
                ellipse_uv, proj_ctx['K'], proj_ctx['n_c'], proj_ctx['d_c'], proj_ctx=proj_ctx
            )
            center3d, fit_dbg = fit_circle_on_plane_weighted_debug(
                pts3d, proj_ctx['n_c'], weights=weights, plane_basis=plane_basis
            )
            if center3d is not None and np.all(np.isfinite(center3d)):
                out['ellipse_backproj_center'] = project_center_to_image(center3d, proj_ctx['K'])
            out['baseline_debug']['ellipse_backprojection'] = {
                'projection': proj_dbg,
                'fit': fit_dbg,
            }
        except Exception as e:
            out['baseline_debug']['ellipse_backprojection'] = {'fit_status': f'exception:{type(e).__name__}:{e}'}

    contour = measurement.get('contour', None)
    if proj_ctx is not None and contour is not None:
        try:
            contour_uv = _uniform_subsample_points(np.asarray(contour, dtype=np.float64).reshape(-1, 2), max_points=240)
            pts3d, weights, proj_dbg = project_pixels_to_plane_filtered(
                contour_uv, proj_ctx['K'], proj_ctx['n_c'], proj_ctx['d_c'], proj_ctx=proj_ctx
            )
            center3d, fit_dbg = _fit_min_enclosing_circle_on_plane(pts3d, plane_basis, weights=weights)
            if center3d is not None and np.all(np.isfinite(center3d)):
                out['homography_rectified_center'] = project_center_to_image(center3d, proj_ctx['K'])
            out['baseline_debug']['homography_rectified_circle'] = {
                'projection': proj_dbg,
                'fit': fit_dbg,
            }
        except Exception as e:
            out['baseline_debug']['homography_rectified_circle'] = {'fit_status': f'exception:{type(e).__name__}:{e}'}

    return out


def _weighted_plane_circle_support_metrics(points3d, weights, center3d, target_radius, plane_basis):
    pts3d = np.asarray(points3d, dtype=np.float64)
    if pts3d.ndim != 2 or pts3d.shape[0] < 6 or pts3d.shape[1] != 3:
        return {
            'valid': False,
            'reason': 'insufficient_points',
            'radius_rmse_norm': np.inf,
            'radius_bias_norm': np.inf,
            'condition_number': np.inf,
            'score': np.inf,
            'count': int(pts3d.shape[0]) if pts3d.ndim == 2 else 0,
        }
    C = np.asarray(center3d, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(C)):
        return {
            'valid': False,
            'reason': 'invalid_center',
            'radius_rmse_norm': np.inf,
            'radius_bias_norm': np.inf,
            'condition_number': np.inf,
            'score': np.inf,
            'count': int(pts3d.shape[0]),
        }
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != pts3d.shape[0]:
        w = np.ones((pts3d.shape[0],), dtype=np.float64)
    target_radius = float(target_radius)
    if not np.isfinite(target_radius) or target_radius <= np.finfo(np.float64).eps:
        return {
            'valid': False,
            'reason': 'invalid_radius',
            'radius_rmse_norm': np.inf,
            'radius_bias_norm': np.inf,
            'condition_number': np.inf,
            'score': np.inf,
            'count': int(pts3d.shape[0]),
        }

    diffs = pts3d - C.reshape(1, 3)
    radii = np.linalg.norm(diffs, axis=1)
    valid = np.all(np.isfinite(diffs), axis=1) & np.isfinite(radii) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(valid) < 6:
        return {
            'valid': False,
            'reason': 'insufficient_valid_points',
            'radius_rmse_norm': np.inf,
            'radius_bias_norm': np.inf,
            'condition_number': np.inf,
            'score': np.inf,
            'count': int(np.count_nonzero(valid)),
        }
    pts = pts3d[valid]
    diffs = diffs[valid]
    radii = radii[valid]
    wv = w[valid]
    wv = np.clip(wv, 1e-12, None)
    wsum = float(np.sum(wv))
    wv = wv / max(wsum, 1e-12)

    resid = radii - target_radius
    rmse = float(np.sqrt(np.sum(wv * resid * resid)))
    bias = float(abs(np.sum(wv * radii) - target_radius))
    radius_rmse_norm = float(rmse / target_radius)
    radius_bias_norm = float(bias / target_radius)

    _, e1, e2 = plane_basis
    u = diffs @ np.asarray(e1, dtype=np.float64).reshape(3)
    v = diffs @ np.asarray(e2, dtype=np.float64).reshape(3)
    plane_vec = np.stack([u, v], axis=1)
    plane_norm = np.linalg.norm(plane_vec, axis=1)
    good_dir = plane_norm > 1e-12
    if np.count_nonzero(good_dir) < 3:
        cond = np.inf
    else:
        dirs = plane_vec[good_dir] / plane_norm[good_dir, None]
        wd = wv[good_dir]
        wd = wd / max(float(np.sum(wd)), 1e-12)
        M = np.zeros((2, 2), dtype=np.float64)
        for di, wi in zip(dirs, wd):
            M += wi * np.outer(di, di)
        eigvals = np.linalg.eigvalsh(M)
        lmin = float(np.min(eigvals))
        lmax = float(np.max(eigvals))
        cond = np.inf if (not np.isfinite(lmin) or not np.isfinite(lmax) or lmin <= 1e-12) else float(lmax / lmin)

    score = np.inf if not np.isfinite(cond) else float((radius_rmse_norm + radius_bias_norm) * np.sqrt(max(cond, 1.0)))
    return {
        'valid': np.isfinite(score),
        'reason': 'ok' if np.isfinite(score) else 'ill_conditioned',
        'radius_rmse_norm': radius_rmse_norm,
        'radius_bias_norm': radius_bias_norm,
        'condition_number': cond,
        'score': score,
        'count': int(pts.shape[0]),
    }


def _physical_solvability_gate(points3d, weights, center_geom_3d, ellipse_center_uv, ellipse_cov, geo_cov, proj_ctx, target_radius_mm, saturation_ratio, plateau_support_mean):
    plane_basis = proj_ctx.get('plane_basis')
    geom_stats = _weighted_plane_circle_support_metrics(points3d, weights, center_geom_3d, target_radius_mm, plane_basis)
    ellipse_center_3d = _project_single_pixel_to_plane(ellipse_center_uv, proj_ctx)
    if ellipse_center_3d is None or not np.all(np.isfinite(ellipse_center_3d)):
        ellipse_stats = {
            'valid': False,
            'reason': 'ellipse_projection_fail',
            'radius_rmse_norm': np.inf,
            'radius_bias_norm': np.inf,
            'condition_number': np.inf,
            'score': np.inf,
            'count': 0,
        }
    else:
        ellipse_stats = _weighted_plane_circle_support_metrics(points3d, weights, ellipse_center_3d, target_radius_mm, plane_basis)

    sat = 0.0 if not np.isfinite(saturation_ratio) else max(float(saturation_ratio), 0.0)
    plateau = 0.0 if not np.isfinite(plateau_support_mean) else max(float(plateau_support_mean), 0.0)
    halo_penalty = float(np.sqrt(sat * plateau))

    ellipse_cov = np.asarray(ellipse_cov, dtype=np.float64).reshape(2, 2)
    geo_cov = np.asarray(geo_cov, dtype=np.float64).reshape(2, 2)
    ellipse_unc = float(np.sqrt(max(np.trace(ellipse_cov), 0.0))) if np.all(np.isfinite(ellipse_cov)) else np.inf
    geo_unc = float(np.sqrt(max(np.trace(geo_cov), 0.0))) if np.all(np.isfinite(geo_cov)) else np.inf

    ellipse_score = float(ellipse_stats['score'] + ellipse_unc / max(target_radius_mm, 1e-12)) if ellipse_stats.get('valid', False) else np.inf
    geom_score = float(geom_stats['score'] + geo_unc / max(target_radius_mm, 1e-12) + halo_penalty) if geom_stats.get('valid', False) else np.inf

    use_geometry = bool(np.isfinite(geom_score) and (geom_score < ellipse_score or not np.isfinite(ellipse_score)))
    decision = 'use_geometry_by_physical_solvability' if use_geometry else 'fallback_to_ellipse_by_physical_solvability'
    return {
        'use_geometry': use_geometry,
        'decision': decision,
        'geometry_score': None if not np.isfinite(geom_score) else float(geom_score),
        'ellipse_score': None if not np.isfinite(ellipse_score) else float(ellipse_score),
        'halo_penalty': float(halo_penalty),
        'geometry_stats': geom_stats,
        'ellipse_stats': ellipse_stats,
        'geometry_uncertainty': None if not np.isfinite(geo_unc) else float(geo_unc),
        'ellipse_uncertainty': None if not np.isfinite(ellipse_unc) else float(ellipse_unc),
        'ellipse_center_3d': None if ellipse_center_3d is None or not np.all(np.isfinite(ellipse_center_3d)) else ellipse_center_3d.astype(np.float64),
    }


def estimate_geometric_center(edge_set, measurement, K, n_c1, d_c1, proj_ctx=None):
    bbox_center = np.asarray(measurement.get('bbox_center', [np.nan, np.nan]), dtype=np.float64)
    gray_center = np.asarray(measurement.get('gray_center', bbox_center), dtype=np.float64)
    ellipse_center_ref = np.asarray(measurement.get('ellipse_center', bbox_center), dtype=np.float64)
    ellipse_cov = np.asarray(measurement.get('ellipse_cov', np.eye(2, dtype=np.float64) * 25.0), dtype=np.float64)
    work_ellipse = measurement.get('work_ellipse')
    point_debug = dict(measurement.get('point_debug_base', {}))
    point_debug.update(edge_set.get('debug', {}))

    if proj_ctx is None:
        proj_ctx = _prepare_projection_context(K, n_c1, d_c1)

    pts_refined = edge_set.get('points')
    if pts_refined is None or len(pts_refined) < 6:
        geo_cov = estimate_geometry_center_covariance(point_debug, work_ellipse)
        point_debug['geometry_regularization_alpha'] = 0.0
        point_debug['geometry_gate'] = {'decision': 'fallback_ellipse_insufficient_shoulder_points', 'reference_center': ellipse_center_ref.tolist(), 'raw_geometry_center': None}
        point_debug['geometry_output_center'] = ellipse_center_ref.tolist()
        point_debug['geometry_diagnostic_center'] = ellipse_center_ref.tolist()
        return {'center': ellipse_center_ref.copy(), 'cov': geo_cov, 'point_debug': point_debug, 'points_3d': None, 'weights': None, 'diagnostic_center': ellipse_center_ref.copy(), 'diagnostic_cov': geo_cov.copy(), 'fusion_debug': {'alpha': 0.0, 'mode': 'ellipse_fallback'}}

    pts3d, proj_weights, dbg_proj = project_pixels_to_plane_filtered(pts_refined, K, n_c1, d_c1, den_eps=0.0, keep_ratio=0.0, proj_ctx=proj_ctx)
    point_debug.update(dbg_proj)
    if pts3d is None or len(pts3d) < 6:
        geo_cov = estimate_geometry_center_covariance(point_debug, work_ellipse)
        point_debug['geometry_regularization_alpha'] = 0.0
        point_debug['geometry_gate'] = {'decision': 'fallback_ellipse_projection_fail', 'reference_center': ellipse_center_ref.tolist(), 'raw_geometry_center': None}
        point_debug['geometry_output_center'] = ellipse_center_ref.tolist()
        point_debug['geometry_diagnostic_center'] = ellipse_center_ref.tolist()
        return {'center': ellipse_center_ref.copy(), 'cov': geo_cov, 'point_debug': point_debug, 'points_3d': None, 'weights': None, 'diagnostic_center': ellipse_center_ref.copy(), 'diagnostic_cov': geo_cov.copy(), 'fusion_debug': {'alpha': 0.0, 'mode': 'ellipse_fallback'}}

    weights = np.asarray(proj_weights, dtype=np.float64)
    ew = edge_set.get('weights')
    if ew is not None:
        ew = np.asarray(ew, dtype=np.float64).reshape(-1)
        if len(ew) == len(weights):
            weights = np.clip(weights * ew, 1e-9, None)
            weights = weights / (np.mean(weights) + 1e-12)

    center_3d, dbg_fit = fit_circle_on_plane_weighted_debug(
        pts3d,
        n_c1,
        weights=weights,
        plane_basis=proj_ctx.get('plane_basis'),
    )
    point_debug.update(dbg_fit)

    rep = measurement.get('point_debug_base', {}).get('representation', {}) if isinstance(measurement.get('point_debug_base', {}), dict) else {}
    confidence_light = float(rep.get('confidence_light', 1.0 / (1.0 + max(float(point_debug.get('fit_residual_median', 0.0) or 0.0), 0.0))))
    plateau_support_mean = float(rep.get('plateau_support_mean', edge_set.get('debug', {}).get('edge_plateau_ratio_mean', 0.0)) or 0.0)
    coverage = float(rep.get('effective_arc_coverage', edge_set.get('debug', {}).get('effective_arc_coverage', 0.0)) or 0.0)
    saturation_ratio = float(measurement.get('point_debug_base', {}).get('saturation_ratio', np.nan))
    sigma_mean = float(edge_set.get('debug', {}).get('edge_sigma_mean', np.nan))
    scale = _ellipse_geom_scale(work_ellipse)
    bloom_badness = _bloom_badness_from_stats(scale, saturation_ratio, plateau_support_mean, coverage, sigma_mean)

    base_cov = estimate_geometry_center_covariance(point_debug, work_ellipse)

    if center_3d is None or not np.all(np.isfinite(center_3d)):
        geo_center = ellipse_center_ref.copy()
        diagnostic_center = ellipse_center_ref.copy()
        diagnostic_cov = base_cov.copy()
        alpha = 0.0
        decision = 'fallback_ellipse_fit_fail'
        raw_geometry_center_img = None
        geo_cov = base_cov.copy()
    else:
        raw_geometry_center_img = project_center_to_image(np.asarray(center_3d, dtype=np.float64), proj_ctx['K'])
        diagnostic_center = raw_geometry_center_img.copy()
        diagnostic_cov = base_cov.copy()
        geo_center = raw_geometry_center_img.copy()
        geo_cov = base_cov.copy()
        alpha = 1.0
        decision = 'use_edge_first_known_radius_circle_fit'

        point_debug['geometry_plane_support_gate'] = {
            'decision': decision,
            'use_geometry': True,
            'geometry_score': point_debug.get('fit_residual_median'),
            'ellipse_score': None,
            'halo_penalty': None,
            'geometry_stats': {
                'fit_model': point_debug.get('fit_model'),
                'fit_radius': point_debug.get('fit_radius'),
                'fit_residual_median': point_debug.get('fit_residual_median'),
                'fit_used_points': point_debug.get('fit_used_points'),
            },
            'ellipse_stats': None,
            'geometry_uncertainty': point_debug.get('fit_center_cov_trace'),
            'ellipse_uncertainty': None,
        }
        point_debug['geometry_feasibility_ratio'] = None
        point_debug['geometry_support_term'] = None
        point_debug['geometry_plateau_term'] = float(plateau_support_mean)
        point_debug['geometry_residual_term'] = None
        point_debug['geometry_sharp_term'] = None
        point_debug['geometry_evidence_term'] = None
        point_debug['geometry_shift_consistency'] = None
        point_debug['geometry_expected_shift_ratio'] = None
        point_debug['geometry_raw_to_ellipse_ratio'] = float(np.linalg.norm(raw_geometry_center_img - ellipse_center_ref) / max(scale, np.finfo(np.float64).eps))
        point_debug['geometry_raw_to_gray_ratio'] = float(np.linalg.norm(raw_geometry_center_img - gray_center) / max(scale, np.finfo(np.float64).eps))
        point_debug['geometry_exposure_burden'] = float(np.sqrt(max(0.0, saturation_ratio) * max(0.0, plateau_support_mean))) if np.isfinite(saturation_ratio) else 0.0

    point_debug['geometry_regularization_alpha'] = float(alpha)
    point_debug['geometry_gate'] = {
        'decision': decision,
        'reference_center': ellipse_center_ref.tolist(),
        'raw_geometry_center': None if raw_geometry_center_img is None or not np.all(np.isfinite(raw_geometry_center_img)) else raw_geometry_center_img.tolist(),
        'confidence_light': confidence_light,
        'plateau_support_mean': plateau_support_mean,
        'effective_arc_coverage': coverage,
        'bloom_badness': bloom_badness,
        'alpha': float(alpha),
    }
    point_debug['geometry_output_center'] = geo_center.tolist()
    point_debug['geometry_diagnostic_center'] = diagnostic_center.tolist()
    return {
        'center': geo_center.astype(np.float64),
        'cov': geo_cov.astype(np.float64),
        'point_debug': point_debug,
        'points_3d': pts3d,
        'weights': weights,
        'diagnostic_center': diagnostic_center.astype(np.float64),
        'diagnostic_cov': diagnostic_cov.astype(np.float64),
        'fusion_debug': {
            'mode': 'physical_solvability_gate_geometry_switch',
            'alpha': float(alpha),
            'decision': decision,
            'confidence_light': float(confidence_light),
            'bloom_badness': float(bloom_badness),
            'feasibility_ratio': _safe_float(point_debug.get('geometry_feasibility_ratio')),
            'plane_support_gate': point_debug.get('geometry_plane_support_gate', {}),
        },
    }

def _diagnostic_risk_scores(measurements):
    n = len(measurements)
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64), {'mode': 'empty'}
    risk = np.zeros((n,), dtype=np.float64)
    metric_debug = []
    for i, item in enumerate(measurements):
        dbg = item.get('point_debug', {}) or {}
        rep = item.get('measurement', {}).get('point_debug_base', {}).get('representation', {}) or {}
        cov = np.asarray(item.get('geo_est', {}).get('cov', np.eye(2) * 36.0), dtype=np.float64)
        ell = item.get('measurement', {}).get('work_ellipse', item.get('ellipse_cache'))
        scale = _ellipse_geom_scale(ell)
        conf_light = float(rep.get('confidence_light', dbg.get('confidence_light', np.nan)))
        plateau_support = float(rep.get('plateau_support_mean', dbg.get('edge_plateau_ratio_mean', np.nan)))
        coverage = float(rep.get('effective_arc_coverage', dbg.get('effective_arc_coverage', np.nan)))
        fit_radius = dbg.get('fit_radius', np.nan)
        fit_residual = dbg.get('fit_residual_median', dbg.get('fit_residual_mean', np.nan))
        fit_ratio = float(fit_residual / fit_radius) if np.isfinite(fit_residual) and np.isfinite(fit_radius) and abs(fit_radius) > np.finfo(np.float64).eps else np.nan
        cov_ratio = float(np.sqrt(max(np.trace(cov), 0.0)) / scale) if np.all(np.isfinite(cov)) else np.nan
        sigma_mean = float(dbg.get('edge_sigma_mean', np.nan))
        sat_ratio = float(item.get('measurement', {}).get('point_debug_base', {}).get('saturation_ratio', np.nan))
        bloom_badness = _bloom_badness_from_stats(scale, sat_ratio, plateau_support, coverage, sigma_mean)
        geo_center = np.asarray(item.get('geo_est', {}).get('center', np.full((2,), np.nan)), dtype=np.float64).reshape(2)
        gray_center = np.asarray(item.get('measurement', {}).get('gray_center', np.full((2,), np.nan)), dtype=np.float64).reshape(2)
        ellipse_center = np.asarray(item.get('measurement', {}).get('ellipse_center', np.full((2,), np.nan)), dtype=np.float64).reshape(2)
        d_geo_gray = float(np.linalg.norm(geo_center - gray_center) / scale) if np.all(np.isfinite(geo_center)) and np.all(np.isfinite(gray_center)) else np.nan
        d_geo_ell = float(np.linalg.norm(geo_center - ellipse_center) / scale) if np.all(np.isfinite(geo_center)) and np.all(np.isfinite(ellipse_center)) else np.nan
        pose_ctr = np.asarray(rep.get('assigned_pose_center_image', [np.nan, np.nan]), dtype=np.float64).reshape(2)
        d_geo_pose = float(np.linalg.norm(geo_center - pose_ctr) / scale) if np.all(np.isfinite(geo_center)) and np.all(np.isfinite(pose_ctr)) else np.nan
        d_ell_pose = float(np.linalg.norm(ellipse_center - pose_ctr) / scale) if np.all(np.isfinite(ellipse_center)) and np.all(np.isfinite(pose_ctr)) else np.nan
        pose_gain = float(d_ell_pose - d_geo_pose) if np.isfinite(d_geo_pose) and np.isfinite(d_ell_pose) else np.nan
        parts = []
        if np.isfinite(conf_light): parts.append(float(1.0 - np.clip(conf_light, 0.0, 1.0)))
        if np.isfinite(plateau_support): parts.append(float(1.0 - np.clip(plateau_support, 0.0, 1.0)))
        if np.isfinite(fit_ratio): parts.append(float(fit_ratio / (1.0 + fit_ratio)))
        if np.isfinite(cov_ratio): parts.append(float(cov_ratio / (1.0 + cov_ratio)))
        if np.isfinite(d_geo_gray): parts.append(float(d_geo_gray / (1.0 + d_geo_gray)))
        if np.isfinite(d_geo_ell): parts.append(float(d_geo_ell / (1.0 + d_geo_ell)))
        if np.isfinite(pose_gain): parts.append(float(max(-pose_gain, 0.0) / (1.0 + abs(pose_gain))))
        parts.append(float(np.clip(bloom_badness, 0.0, 1.0)))
        r = float(np.mean(parts)) if parts else 1.0
        risk[i] = float(np.clip(r, 0.0, 1.0))
        metric_debug.append({'confidence_light': None if not np.isfinite(conf_light) else float(conf_light), 'plateau_support': None if not np.isfinite(plateau_support) else float(plateau_support), 'coverage': None if not np.isfinite(coverage) else float(coverage), 'fit_ratio': None if not np.isfinite(fit_ratio) else float(fit_ratio), 'cov_ratio': None if not np.isfinite(cov_ratio) else float(cov_ratio), 'geo_gray_ratio': None if not np.isfinite(d_geo_gray) else float(d_geo_gray), 'geo_ellipse_ratio': None if not np.isfinite(d_geo_ell) else float(d_geo_ell), 'geo_pose_ratio': None if not np.isfinite(d_geo_pose) else float(d_geo_pose), 'ellipse_pose_ratio': None if not np.isfinite(d_ell_pose) else float(d_ell_pose), 'pose_gain': None if not np.isfinite(pose_gain) else float(pose_gain), 'bloom_badness': float(bloom_badness), 'risk': risk[i]})
    confidence = 1.0 - risk
    for i, item in enumerate(measurements):
        dbg = item.get('point_debug', {}) or {}
        dbg['diagnostic_risk'] = float(risk[i]); dbg['diagnostic_confidence'] = float(confidence[i]); dbg['diagnostic_metric_badness'] = metric_debug[i]
        item['point_debug'] = dbg
        geo_cov = np.asarray(item.get('geo_est', {}).get('cov', np.eye(2) * 36.0), dtype=np.float64)
        inflation = 1.0 / max(confidence[i], 1e-6)
        adj_cov = geo_cov * inflation
        item.setdefault('geo_est', {})['cov'] = adj_cov
        item.setdefault('fused_est', {})['center'] = np.asarray(item.get('geo_center', item['measurement'].get('ellipse_center')), dtype=np.float64).tolist()
        item.setdefault('fused_est', {})['cov'] = adj_cov.tolist()
        item.setdefault('fusion_debug', {})
        item['fusion_debug']['fusion'] = {'used_terms': ['geometry_only'], 'cov_trace': float(np.trace(adj_cov)), 'cov_det': float(np.linalg.det(adj_cov)), 'diagnostic_confidence': float(confidence[i])}
    return confidence.astype(np.float64), risk.astype(np.float64), {'mode': 'absolute_plateau_locked_shoulder_pose_guarded_v11', 'metrics': metric_debug, 'confidence': confidence.tolist(), 'risk': risk.tolist()}



# ============================================================
# v12 覆盖：棋盘格位姿 + 四灯联合几何一致候选边选择器
# 仅在候选边选择层升级；后端几何校正主链保持不变
# ============================================================
from itertools import product
try:
    import fast_normal_offset_edge_selector_v7_fast_visualized as _joint_edge_mod
except Exception:
    _joint_edge_mod = None

LIGHT_RADIUS_MM = 10.0
ANTIPODAL_EDGE_CORRECTION_MAX_GAIN = 0.25
ANTIPODAL_EDGE_ECC_GAIN_CENTER = 1.18
ANTIPODAL_EDGE_ECC_GAIN_WIDTH = 0.025
JOINT_EDGE_HOMOGRAPHY_MAX_POINTS_PER_LIGHT = 48
JOINT_EDGE_HOMOGRAPHY_ITERS = 5
JOINT_EDGE_HOMOGRAPHY_CENTER_PRIOR_WEIGHT = 0.18
JOINT_EDGE_HOMOGRAPHY_OUTPUT_BLEND = 0.35
JOINT_EDGE_LOCAL_CIRCLE_PRIOR_WEIGHT = 0.08
JOINT_EDGE_LOCAL_CIRCLE_ITERS = 5
JOINT_EDGE_PERSPECTIVE_BLEND_CENTER = 1.105
JOINT_EDGE_PERSPECTIVE_BLEND_WIDTH = 0.018
FINAL_EDGE_LAYOUT_JOINT_BLEND_BASE = 0.20
FINAL_EDGE_LAYOUT_JOINT_BLEND_MIN = 0.04
FINAL_EDGE_LAYOUT_JOINT_BLEND_MAX = 0.78
FINAL_EDGE_LAYOUT_AGREEMENT_SCALE_RADIUS = 0.45
PHYSICAL_EDGE_MAX_POINTS_PER_LIGHT = 32
PHYSICAL_EDGE_ITERS = 4
PHYSICAL_EDGE_CENTER_PRIOR_WEIGHT = 0.22
PHYSICAL_EDGE_OFFSET_PRIOR_WEIGHT = 0.18
PHYSICAL_EDGE_OUTPUT_BLEND = 0.06
PAGC_SAVE_POSE_EVAL_INPUTS_DEFAULT = "0"
PAGC_SAVE_REFINED_EDGE_POINTS_DEFAULT = "0"


def _normalize_scale_from_positive(vals):
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return 1.0
    return float(np.median(arr))


def _ellipse_param_angles(ellipse, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    if ellipse is None or len(pts) == 0:
        return np.zeros((0,), dtype=np.float64)
    (cx, cy), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, np.finfo(np.float64).eps)
    b = max(float(ma) * 0.5, np.finfo(np.float64).eps)
    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    return np.arctan2(q[:, 1] / b, q[:, 0] / a).astype(np.float64)


def _sample_contour_points_for_arc(contour, ellipse):
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    target = int(_auto_sample_count_from_ellipse(ellipse)) if ellipse is not None else int(len(pts))
    target = int(min(len(pts), max(6, target)))
    if target >= len(pts):
        sampled = pts
    else:
        idx = np.linspace(0.0, len(pts) - 1, target, endpoint=False)
        idx = np.round(idx).astype(np.int64)
        sampled = pts[idx]
    ang = _ellipse_param_angles(ellipse, sampled)
    w = np.ones((len(sampled),), dtype=np.float64)
    return sampled.astype(np.float64), ang.astype(np.float64), w


def _project_world_circle_to_image_ellipse(center_w, radius_mm, K, R1, t1, num_samples=180):
    ts = np.linspace(0.0, 2.0 * np.pi, int(num_samples), endpoint=False, dtype=np.float64)
    circle_w = np.column_stack([
        center_w[0] + radius_mm * np.cos(ts),
        center_w[1] + radius_mm * np.sin(ts),
        np.full_like(ts, center_w[2]),
    ]).astype(np.float64)
    pts_img = []
    for Xw in circle_w:
        Xc = R1 @ Xw + t1
        x = K @ Xc
        if x[2] <= np.finfo(np.float64).eps:
            continue
        pts_img.append([x[0] / x[2], x[1] / x[2]])
    pts_img = np.asarray(pts_img, dtype=np.float64)
    if pts_img.shape[0] < 6:
        return None
    try:
        return cv2.fitEllipse(pts_img.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None


def _ellipse_shape_metrics(cand_ellipse, exp_ellipse):
    if cand_ellipse is None or exp_ellipse is None:
        return np.inf, np.inf, np.inf
    cand_area = max(ellipse_area_from_fit(cand_ellipse), np.finfo(np.float64).eps)
    exp_area = max(ellipse_area_from_fit(exp_ellipse), np.finfo(np.float64).eps)
    cand_ecc = max(ellipse_ecc_ratio_from_fit(cand_ellipse), np.finfo(np.float64).eps)
    exp_ecc = max(ellipse_ecc_ratio_from_fit(exp_ellipse), np.finfo(np.float64).eps)
    area_err = float(abs(np.log(cand_area / exp_area)))
    ecc_err = float(abs(np.log(cand_ecc / exp_ecc)))
    ang_c = float(cand_ellipse[2]) % 180.0
    ang_e = float(exp_ellipse[2]) % 180.0
    ang_diff = abs(((ang_c - ang_e + 90.0) % 180.0) - 90.0)
    return area_err, ecc_err, float(np.deg2rad(ang_diff))


def _candidate_from_contour(contour, ellipse, K, n_c, d_c, proj_ctx=None):
    cnt = _normalize_contour_for_opencv(contour)
    pts2d = cnt.reshape(-1, 2).astype(np.float64)
    arc_points, arc_angles, arc_weights = _sample_contour_points_for_arc(cnt, ellipse)
    if proj_ctx is None:
        proj_ctx = _prepare_projection_context(K, n_c, d_c)
    pts3d, weights3d, proj_dbg = project_pixels_to_plane_filtered(pts2d, K, n_c, d_c, proj_ctx=proj_ctx)
    center_3d, fit_dbg = fit_circle_on_plane_weighted_debug(pts3d, n_c, weights=weights3d, iters=5, plane_basis=proj_ctx.get('plane_basis'))
    center_img = None
    if center_3d is not None and np.all(np.isfinite(center_3d)):
        center_img = project_center_to_image(center_3d, proj_ctx['K'])
    resid = _ellipse_contour_residual(ellipse, pts2d) if ellipse is not None and len(pts2d) >= 6 else np.inf
    return {
        'contour': cnt,
        'ellipse': ellipse,
        'arc_points': arc_points,
        'arc_angles': arc_angles,
        'arc_weights': arc_weights,
        'pts3d': pts3d,
        'weights3d': weights3d,
        'center_3d': None if center_3d is None else np.asarray(center_3d, dtype=np.float64),
        'center_img': None if center_img is None else np.asarray(center_img, dtype=np.float64),
        'fit_radius': _safe_float(fit_dbg.get('fit_radius')) if isinstance(fit_dbg, dict) else None,
        'ellipse_resid': float(resid),
        'proj_dbg': proj_dbg,
        'fit_dbg': fit_dbg,
    }


def _observation_candidates_joint(obs, K, n_c, d_c, proj_ctx=None):
    candidates = []
    if proj_ctx is None:
        proj_ctx = _prepare_projection_context(K, n_c, d_c)
    if obs.source_ellipse is not None:
        candidates.append(_candidate_from_contour(obs.contour, obs.source_ellipse, K, n_c, d_c, proj_ctx=proj_ctx))
    ml_cs = list(getattr(obs, 'multilevel_contours', []) or [])
    ml_es = list(getattr(obs, 'multilevel_ellipses', []) or [])
    for cc, ee in zip(ml_cs, ml_es):
        if ee is None:
            continue
        candidates.append(_candidate_from_contour(cc, ee, K, n_c, d_c, proj_ctx=proj_ctx))
    if not candidates and obs.source_ellipse is not None:
        candidates.append(_candidate_from_contour(obs.contour, obs.source_ellipse, K, n_c, d_c, proj_ctx=proj_ctx))
    return candidates


def _joint_candidate_score_tables(candidates_by_light, expected_img, expected_cam, expected_ellipses):
    atom_tables = []
    center_img_vals = []
    center_3d_vals = []
    radius_vals = []
    resid_vals = []
    area_vals = []
    ecc_vals = []
    ang_vals = []

    for light_cands in candidates_by_light:
        table = []
        for cand in light_cands:
            row = []
            for j in range(len(expected_img)):
                center_img_err = np.inf
                if cand['center_img'] is not None and np.all(np.isfinite(cand['center_img'])):
                    center_img_err = float(np.linalg.norm(cand['center_img'] - expected_img[j]))
                center_3d_err = np.inf
                if cand['center_3d'] is not None and np.all(np.isfinite(cand['center_3d'])):
                    center_3d_err = float(np.linalg.norm(cand['center_3d'] - expected_cam[j]))
                fit_radius = cand['fit_radius']
                radius_err = np.inf if fit_radius is None or not np.isfinite(fit_radius) else float(abs(fit_radius - LIGHT_RADIUS_MM))
                area_err, ecc_err, ang_err = _ellipse_shape_metrics(cand['ellipse'], expected_ellipses[j])
                resid_err = float(cand['ellipse_resid']) if np.isfinite(cand['ellipse_resid']) else np.inf
                row.append({
                    'center_img_err': center_img_err,
                    'center_3d_err': center_3d_err,
                    'radius_err': radius_err,
                    'ellipse_resid': resid_err,
                    'area_err': area_err,
                    'ecc_err': ecc_err,
                    'ang_err': ang_err,
                })
                center_img_vals.append(center_img_err)
                center_3d_vals.append(center_3d_err)
                radius_vals.append(radius_err)
                resid_vals.append(resid_err)
                area_vals.append(area_err)
                ecc_vals.append(ecc_err)
                ang_vals.append(ang_err)
            table.append(row)
        atom_tables.append(table)

    scales = {
        'center_img_err': _normalize_scale_from_positive(center_img_vals),
        'center_3d_err': _normalize_scale_from_positive(center_3d_vals),
        'radius_err': _normalize_scale_from_positive(radius_vals),
        'ellipse_resid': _normalize_scale_from_positive(resid_vals),
        'area_err': _normalize_scale_from_positive(area_vals),
        'ecc_err': _normalize_scale_from_positive(ecc_vals),
        'ang_err': _normalize_scale_from_positive(ang_vals),
    }

    unary_tables = []
    for table in atom_tables:
        utable = []
        for row in table:
            scores = []
            for atom in row:
                s = 0.0
                for key, scale in scales.items():
                    val = atom[key]
                    if np.isfinite(val):
                        s += float(val / max(scale, np.finfo(np.float64).eps))
                    else:
                        s += float(1.0 / np.finfo(np.float64).eps)
                scores.append(float(s))
            utable.append(scores)
        unary_tables.append(utable)
    return atom_tables, unary_tables, scales


def _pairwise_layout_score(combo, perm, candidates_by_light, expected_cam):
    sel_centers = []
    exp_centers = []
    for i, cand_idx in enumerate(combo):
        cand = candidates_by_light[i][cand_idx]
        c3 = cand['center_3d']
        if c3 is None or not np.all(np.isfinite(c3)):
            return np.inf, {'pairwise_center_error': np.inf}
        sel_centers.append(np.asarray(c3, dtype=np.float64))
        exp_centers.append(np.asarray(expected_cam[perm[i]], dtype=np.float64))
    sel_centers = np.asarray(sel_centers, dtype=np.float64)
    exp_centers = np.asarray(exp_centers, dtype=np.float64)
    sel_d = np.linalg.norm(sel_centers[:, None, :] - sel_centers[None, :, :], axis=2)
    exp_d = np.linalg.norm(exp_centers[:, None, :] - exp_centers[None, :, :], axis=2)
    tri = np.triu_indices(len(sel_centers), k=1)
    diff = np.abs(sel_d[tri] - exp_d[tri])
    if diff.size == 0:
        return 0.0, {'pairwise_center_error': 0.0}
    scale = _normalize_scale_from_positive(exp_d[tri])
    score = float(np.sum(diff / max(scale, np.finfo(np.float64).eps)))
    return score, {'pairwise_center_error': float(np.median(diff))}


def _joint_expected_subset_scale(perm, expected_cam):
    idx = list(perm)
    if len(idx) < 2:
        return 1.0
    exp_centers = np.asarray(expected_cam, dtype=np.float64)[idx]
    exp_d = np.linalg.norm(exp_centers[:, None, :] - exp_centers[None, :, :], axis=2)
    tri = np.triu_indices(len(idx), k=1)
    return _normalize_scale_from_positive(exp_d[tri])


def _joint_pairwise_term_table(candidates_by_light, expected_cam, perm, subset_scale):
    n_lights = len(candidates_by_light)
    pair_tables = {}
    for i in range(n_lights):
        for k in range(i + 1, n_lights):
            exp_dist = float(np.linalg.norm(np.asarray(expected_cam[perm[i]], dtype=np.float64) - np.asarray(expected_cam[perm[k]], dtype=np.float64)))
            mat = np.full((len(candidates_by_light[i]), len(candidates_by_light[k])), np.inf, dtype=np.float64)
            for ci, cand_i in enumerate(candidates_by_light[i]):
                c3_i = cand_i.get('center_3d')
                if c3_i is None or not np.all(np.isfinite(c3_i)):
                    continue
                c3_i = np.asarray(c3_i, dtype=np.float64)
                for ck, cand_k in enumerate(candidates_by_light[k]):
                    c3_k = cand_k.get('center_3d')
                    if c3_k is None or not np.all(np.isfinite(c3_k)):
                        continue
                    c3_k = np.asarray(c3_k, dtype=np.float64)
                    diff = abs(float(np.linalg.norm(c3_i - c3_k)) - exp_dist)
                    mat[ci, ck] = diff / max(float(subset_scale), np.finfo(np.float64).eps)
            pair_tables[(i, k)] = mat
    return pair_tables


def _joint_pairwise_score_from_selection(selection, pair_tables):
    score = 0.0
    pair_vals = []
    for i, k in sorted(pair_tables.keys()):
        ci = selection[i]
        ck = selection[k]
        val = float(pair_tables[(i, k)][ci, ck])
        if not np.isfinite(val):
            return np.inf, {'pairwise_center_error': np.inf}
        score += val
        pair_vals.append(val)
    pair_median = None if len(pair_vals) == 0 else float(np.median(pair_vals))
    return float(score), {'pairwise_center_error': pair_median}


def _joint_search_best_combo_for_perm(candidates_by_light, unary_tables, pair_tables, perm, current_best=np.inf):
    n_lights = len(candidates_by_light)
    min_unary_by_light = []
    for i in range(n_lights):
        vals = np.asarray([unary_tables[i][cand_idx][perm[i]] for cand_idx in range(len(candidates_by_light[i]))], dtype=np.float64)
        min_unary_by_light.append(float(np.min(vals)) if vals.size else np.inf)
    suffix_min_unary = np.zeros((n_lights + 1,), dtype=np.float64)
    for i in range(n_lights - 1, -1, -1):
        suffix_min_unary[i] = suffix_min_unary[i + 1] + min_unary_by_light[i]

    best_combo = None
    best_score = float(current_best)
    selection = [-1] * n_lights
    candidate_orders = []
    for i in range(n_lights):
        order = sorted(range(len(candidates_by_light[i])), key=lambda cand_idx: float(unary_tables[i][cand_idx][perm[i]]))
        candidate_orders.append(order)

    def dfs(light_idx, unary_accum, pair_accum):
        nonlocal best_combo, best_score
        lower_bound = float(unary_accum + pair_accum + suffix_min_unary[light_idx])
        if lower_bound >= best_score:
            return
        if light_idx >= n_lights:
            best_combo = tuple(selection)
            best_score = float(unary_accum + pair_accum)
            return
        for cand_idx in candidate_orders[light_idx]:
            unary_val = float(unary_tables[light_idx][cand_idx][perm[light_idx]])
            if not np.isfinite(unary_val):
                continue
            added_pair = 0.0
            bad = False
            for prev_idx in range(light_idx):
                prev_cand = selection[prev_idx]
                pair_val = float(pair_tables[(prev_idx, light_idx)][prev_cand, cand_idx])
                if not np.isfinite(pair_val):
                    bad = True
                    break
                added_pair += pair_val
            if bad:
                continue
            selection[light_idx] = int(cand_idx)
            dfs(light_idx + 1, unary_accum + unary_val, pair_accum + added_pair)
            selection[light_idx] = -1

    dfs(0, 0.0, 0.0)
    if best_combo is None:
        return None, np.inf, None
    pair_score, pair_dbg = _joint_pairwise_score_from_selection(best_combo, pair_tables)
    total_score = float(sum(float(unary_tables[i][best_combo[i]][perm[i]]) for i in range(n_lights)) + pair_score)
    return best_combo, total_score, pair_dbg
def build_selected_measurements_for_pipeline_joint_geometry(
    gray_light: np.ndarray,
    contours,
    K: np.ndarray,
    pose_ctx: Dict[str, Any],
    circle_centers_w: np.ndarray,
):
    if _joint_edge_mod is None:
        return build_selected_measurements_for_pipeline(
            gray_light=gray_light,
            contours=contours,
            K=K,
            plane_normal_camera=pose_ctx['n_c1'],
            plane_distance=pose_ctx['d_c1'],
            circle_centers_w=circle_centers_w,
        )

    observations = _joint_edge_mod.build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=pose_ctx['n_c1'],
        plane_distance=pose_ctx['d_c1'],
    )
    if len(observations) == 0:
        return [], {'mode': 'joint_geometry_selector', 'reason': 'no_observation'}

    runtime_cache = pose_ctx.get('runtime_cache')
    if not isinstance(runtime_cache, dict):
        runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
    proj_ctx = runtime_cache['projection_ctx']
    expected_img = np.asarray(runtime_cache['expected_img'], dtype=np.float64).reshape(-1, 2)
    expected_cam = np.asarray(runtime_cache['expected_cam'], dtype=np.float64).reshape(-1, 3)
    expected_ellipses = list(runtime_cache['expected_ellipses'])

    candidates_by_light = [
        _observation_candidates_joint(obs, K, pose_ctx['n_c1'], pose_ctx['d_c1'], proj_ctx=proj_ctx)
        for obs in observations
    ]
    if any(len(cands) == 0 for cands in candidates_by_light):
        return build_selected_measurements_for_pipeline(
            gray_light=gray_light,
            contours=contours,
            K=K,
            plane_normal_camera=pose_ctx['n_c1'],
            plane_distance=pose_ctx['d_c1'],
            circle_centers_w=circle_centers_w,
        )

    atom_tables, unary_tables, scales = _joint_candidate_score_tables(candidates_by_light, expected_img, expected_cam, expected_ellipses)

    best = None
    best_score = np.inf
    best_pair_dbg = None
    n_lights = len(candidates_by_light)
    for perm in permutations(range(len(circle_centers_w)), n_lights):
        unary_lower = 0.0
        feasible = True
        for i in range(n_lights):
            vals = np.asarray([unary_tables[i][cand_idx][perm[i]] for cand_idx in range(len(candidates_by_light[i]))], dtype=np.float64)
            if vals.size == 0 or not np.any(np.isfinite(vals)):
                feasible = False
                break
            unary_lower += float(np.min(vals))
        if not feasible or unary_lower >= best_score:
            continue

        subset_scale = _joint_expected_subset_scale(perm, expected_cam)
        pair_tables = _joint_pairwise_term_table(candidates_by_light, expected_cam, perm, subset_scale)
        combo, total_score, pair_dbg = _joint_search_best_combo_for_perm(candidates_by_light, unary_tables, pair_tables, perm, current_best=best_score)
        if combo is None:
            continue
        if total_score < best_score:
            best_score = float(total_score)
            best = (combo, perm)
            best_pair_dbg = pair_dbg

    if best is None:
        return build_selected_measurements_for_pipeline(
            gray_light=gray_light,
            contours=contours,
            K=K,
            plane_normal_camera=pose_ctx['n_c1'],
            plane_distance=pose_ctx['d_c1'],
            circle_centers_w=circle_centers_w,
        )

    combo, perm = best
    selected = []
    per_light_debug = []
    for i, (obs, cand_idx, assigned_j) in enumerate(zip(observations, combo, perm)):
        cand = candidates_by_light[i][cand_idx]
        atom = atom_tables[i][cand_idx][assigned_j]
        unary = unary_tables[i][cand_idx][assigned_j]
        rep_dbg = {
            'used': True,
            'reason': 'four_light_joint_geometric_consistency_selector_v1',
            'assigned_layout_index': int(assigned_j),
            'assigned_pose_center_image': expected_img[assigned_j].astype(np.float64).tolist(),
            'assigned_pose_center_camera': expected_cam[assigned_j].astype(np.float64).tolist(),
            'assigned_pose_ellipse': _serialize_cv_ellipse(expected_ellipses[assigned_j]),
            'joint_unary_score': float(unary),
            'joint_total_score': float(best_score),
            'joint_pairwise_center_error': None if best_pair_dbg is None else _safe_float(best_pair_dbg.get('pairwise_center_error')),
            'center_img_err': _safe_float(atom['center_img_err']),
            'center_3d_err': _safe_float(atom['center_3d_err']),
            'radius_err': _safe_float(atom['radius_err']),
            'ellipse_resid': _safe_float(atom['ellipse_resid']),
            'area_err': _safe_float(atom['area_err']),
            'ecc_err': _safe_float(atom['ecc_err']),
            'ang_err': _safe_float(atom['ang_err']),
            'scales': {k: float(v) for k, v in scales.items()},
            'confidence_light': float(1.0 / (1.0 + unary)),
            'plateau_support_mean': float(obs.feature_debug.get('plateau_support_mean', 0.0) if isinstance(obs.feature_debug, dict) else 0.0),
            'effective_arc_coverage': float(_effective_arc_coverage_from_angles(cand['arc_angles'])) if len(cand['arc_angles']) >= 2 else 0.0,
            'arc_points_count': int(len(cand['arc_points'])),
            'candidate_index': int(cand_idx),
            'num_candidates_for_light': int(len(candidates_by_light[i])),
        }
        selected.append({
            'source_contour': obs.contour,
            'source_ellipse': obs.source_ellipse,
            'bbox_center': obs.bbox_center,
            'gray_center': obs.gray_center,
            'work_contour': cand['contour'],
            'work_ellipse': cand['ellipse'],
            'arc_points': cand['arc_points'],
            'arc_angles': cand['arc_angles'],
            'arc_weights': cand['arc_weights'],
            'rep_dbg': rep_dbg,
        })
        per_light_debug.append(rep_dbg)

    global_debug = {
        'mode': 'joint_geometry_selector',
        'selector': 'four_light_joint_geometric_consistency_selector_v1',
        'best_total_score': float(best_score),
        'assignment': [int(x) for x in perm],
        'combo': [int(x) for x in combo],
        'pairwise': best_pair_dbg,
        'per_light': per_light_debug,
    }
    return selected, global_debug




def _serialize_cv_ellipse(ellipse):
    if ellipse is None:
        return None
    try:
        (cx, cy), (MA, ma), ang = ellipse
        return {
            'center': [float(cx), float(cy)],
            'axes': [float(MA), float(ma)],
            'angle_deg': float(ang),
        }
    except Exception:
        return None


def _apply_frontend_mode_to_selected_item(gray_light_proc, selected_item, frontend_mode):
    if frontend_mode == 'legacy':
        return selected_item
    if frontend_mode != 'randomized_physical':
        raise ValueError(f'未知 frontend_mode: {frontend_mode}')
    if select_randomized_physical_contour is None:
        raise RuntimeError('未找到 physical_randomized_contour_selector 模块，无法使用 randomized_physical 前端')
    try:
        refined = select_randomized_physical_contour(
            gray_light_proc=gray_light_proc,
            selected_item=selected_item,
            light_radius_mm=LIGHT_RADIUS_MM,
        )
    except Exception as e:
        refined = dict(selected_item)
        dbg = dict(refined.get('rep_dbg', {})) if isinstance(refined.get('rep_dbg', {}), dict) else {}
        dbg['randomized_physical_fallback'] = True
        dbg['randomized_physical_fallback_reason'] = f'{type(e).__name__}: {e}'
        refined['rep_dbg'] = dbg
    return refined


def _apply_frontend_mode_to_selected_measurements(gray_light_proc, selected_measurements, frontend_mode):
    out = []
    for item in selected_measurements:
        out.append(_apply_frontend_mode_to_selected_item(gray_light_proc, item, frontend_mode))
    return out


def _contour_center_for_pose_prefilter(contour):
    contour = np.asarray(contour)
    if contour is None or len(contour) == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            return np.asarray(ellipse[0], dtype=np.float64).reshape(2)
        except cv2.error:
            pass
    m = cv2.moments(contour)
    if abs(float(m.get('m00', 0.0))) > 1e-9:
        return np.array([m['m10'] / m['m00'], m['m01'] / m['m00']], dtype=np.float64)
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    return np.mean(pts, axis=0)


def _select_four_candidate_contours_by_pose_layout(contours, pose_ctx):
    contours = list(contours or [])
    if len(contours) <= 4:
        return contours
    expected = pose_ctx.get('proj_pose_final', pose_ctx.get('pose_reference_eval', None))
    if expected is None:
        runtime_cache = pose_ctx.get('runtime_cache', {}) if isinstance(pose_ctx.get('runtime_cache'), dict) else {}
        expected = runtime_cache.get('expected_img')
    if expected is None:
        return contours[:4]
    expected = np.asarray(expected, dtype=np.float64).reshape(-1, 2)
    expected = expected[np.all(np.isfinite(expected), axis=1)]
    if len(expected) != 4:
        return contours[:4]
    centers = np.asarray([_contour_center_for_pose_prefilter(c) for c in contours], dtype=np.float64).reshape(-1, 2)
    finite = np.all(np.isfinite(centers), axis=1)
    if np.count_nonzero(finite) < 4:
        return contours[:4]

    best = None
    best_cost = np.inf
    idxs = [i for i, ok in enumerate(finite) if ok]
    for combo in itertools.combinations(idxs, 4):
        for perm in itertools.permutations(combo, 4):
            pts = centers[list(perm)]
            d = np.linalg.norm(pts - expected, axis=1)
            cost = float(np.sum(d))
            if cost < best_cost:
                best_cost = cost
                best = tuple(perm)
    if best is None:
        return contours[:4]
    selected = [contours[i] for i in best]
    return selected


def _recover_pose_seeded_green_core_contours(light_img_proc, contours, pose_ctx, search_radius_px=90):
    contours = list(contours or [])
    if light_img_proc is None or light_img_proc.ndim != 3:
        return contours
    expected = pose_ctx.get('proj_pose_final', pose_ctx.get('pose_reference_eval', None))
    if expected is None:
        runtime_cache = pose_ctx.get('runtime_cache', {}) if isinstance(pose_ctx.get('runtime_cache'), dict) else {}
        expected = runtime_cache.get('expected_img')
    if expected is None:
        return contours
    expected = np.asarray(expected, dtype=np.float64).reshape(-1, 2)
    h, w = light_img_proc.shape[:2]
    existing_centers = [_contour_center_for_pose_prefilter(c) for c in contours]

    b = light_img_proc[:, :, 0].astype(np.float32)
    g = light_img_proc[:, :, 1].astype(np.float32)
    r = light_img_proc[:, :, 2].astype(np.float32)
    green_excess = g - 0.45 * (r + b)

    for pt in expected:
        if not np.all(np.isfinite(pt)):
            continue
        if any(np.linalg.norm(np.asarray(cen, dtype=np.float64) - pt) < 0.75 * float(search_radius_px) for cen in existing_centers if np.all(np.isfinite(cen))):
            continue
        cx, cy = float(pt[0]), float(pt[1])
        x0 = max(0, int(round(cx - search_radius_px)))
        y0 = max(0, int(round(cy - search_radius_px)))
        x1 = min(w, int(round(cx + search_radius_px + 1)))
        y1 = min(h, int(round(cy + search_radius_px + 1)))
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            continue
        roi = green_excess[y0:y1, x0:x1]
        roi_g = g[y0:y1, x0:x1]
        vals = roi[np.isfinite(roi)]
        if vals.size < 16:
            continue
        thr = max(float(np.percentile(vals, 96.5)), float(np.percentile(roi_g, 97.5)) - 0.45 * 255.0)
        bw = np.zeros(roi.shape, dtype=np.uint8)
        bw[roi >= thr] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        local_contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best = None
        best_score = np.inf
        for cc in local_contours:
            if len(cc) < 5:
                continue
            area = float(cv2.contourArea(cc))
            if area < 12.0:
                continue
            cc_global = cc.copy()
            cc_global[:, :, 0] += x0
            cc_global[:, :, 1] += y0
            cen = _contour_center_for_pose_prefilter(cc_global)
            if not np.all(np.isfinite(cen)):
                continue
            dist = float(np.linalg.norm(cen - pt))
            x, y, ww, hh = cv2.boundingRect(cc_global)
            shape_penalty = abs(math.log((ww + 1.0) / (hh + 1.0)))
            score = dist + 0.02 * shape_penalty * float(search_radius_px) - 0.015 * min(area, 400.0)
            if score < best_score:
                best_score = score
                best = cc_global
        if best is not None:
            contours.append(best)
            existing_centers.append(_contour_center_for_pose_prefilter(best))
    return contours


def _pose_ctx_from_light_measurement_centers(base_pose_ctx, K, measurements, circle_centers_w):
    obj_pts = []
    img_pts = []
    for meas in measurements:
        rep = meas.get("point_debug_base", {}).get("representation", {}) if isinstance(meas.get("point_debug_base", {}), dict) else {}
        idx = rep.get("assigned_layout_index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if idx < 0 or idx >= len(circle_centers_w):
            continue
        uv = np.asarray(meas.get("ellipse_center", meas.get("weighted_gray_center", [np.nan, np.nan])), dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(uv)):
            continue
        obj_pts.append(np.asarray(circle_centers_w[idx], dtype=np.float64).reshape(3))
        img_pts.append(uv)

    if len(obj_pts) < 4:
        return base_pose_ctx, {"used": False, "reason": "too_few_labeled_measurements", "num_points": int(len(obj_pts))}

    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    base_rvec = np.asarray(base_pose_ctx["rvec1"], dtype=np.float64).reshape(3, 1)
    base_tvec = np.asarray(base_pose_ctx["tvec1"], dtype=np.float64).reshape(3, 1)

    solutions = []
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            if flag == cv2.SOLVEPNP_ITERATIVE:
                ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5), rvec=base_rvec, tvec=base_tvec, useExtrinsicGuess=True, flags=flag)
                if ok:
                    solutions.append((np.asarray(rv, dtype=np.float64).reshape(3, 1), np.asarray(tv, dtype=np.float64).reshape(3, 1), int(flag)))
            else:
                ret = cv2.solvePnPGeneric(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                if bool(ret[0]):
                    for rv, tv in zip(ret[1], ret[2]):
                        solutions.append((np.asarray(rv, dtype=np.float64).reshape(3, 1), np.asarray(tv, dtype=np.float64).reshape(3, 1), int(flag)))
        except cv2.error:
            continue

    best = None
    best_score = np.inf
    for rv, tv, flag in solutions:
        proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, np.zeros(5))
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_pts, axis=1)
        R, _ = cv2.Rodrigues(rv)
        depths = (R @ obj_pts.T + tv.reshape(3, 1))[2]
        if np.min(depths) <= 1.0:
            continue
        pose_shift = float(np.linalg.norm(tv.reshape(3) - base_tvec.reshape(3)) / 100.0)
        score = float(np.median(err) + 0.15 * pose_shift)
        if score < best_score:
            best_score = score
            best = (rv, tv, flag, err)

    if best is None:
        return base_pose_ctx, {"used": False, "reason": "no_positive_depth_solution", "num_points": int(len(obj_pts))}

    rv, tv, flag, err = best
    R, _ = cv2.Rodrigues(rv)
    t = tv.reshape(3)
    n_c = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_c /= max(np.linalg.norm(n_c), 1e-12)
    d_c = -float(np.dot(n_c, t))
    proj_all, _ = cv2.projectPoints(np.asarray(circle_centers_w, dtype=np.float64), rv, tv, K, np.zeros(5))
    proj_all = proj_all.reshape(-1, 2)
    pose_sigma_px = float(np.clip(max(0.75, 0.75 * float(np.median(err))), 0.5, 8.0))

    pose_ctx = dict(base_pose_ctx)
    pose_ctx.update({
        "rvec1": rv,
        "tvec1": tv,
        "R1": R,
        "t1": t,
        "n_c1": n_c,
        "d_c1": d_c,
        "proj_pose_final": proj_all,
        "proj_true_eval": proj_all.copy(),
        "pose_reference_eval": proj_all.copy(),
        "pose_sigma_px": pose_sigma_px,
        "pose_reference_covariances": [(np.eye(2) * pose_sigma_px * pose_sigma_px).tolist() for _ in range(len(proj_all))],
        "pose_source": "light_only_measurement_refined",
    })
    dbg0 = dict(base_pose_ctx.get("pose_ref_debug", {}))
    dbg0["measurement_pose_refine"] = {
        "used": True,
        "num_points": int(len(obj_pts)),
        "solver_flag": int(flag),
        "median_reprojection_px": float(np.median(err)),
        "max_reprojection_px": float(np.max(err)),
        "source": "roi_frontend_ellipse_centers_no_checkerboard",
    }
    pose_ctx["pose_ref_debug"] = dbg0
    _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
    return pose_ctx, dbg0["measurement_pose_refine"]


def _assigned_layout_index_from_packed_measurement(item):
    sources = []
    if isinstance(item, dict):
        sources.append(item.get("point_debug", {}))
        meas = item.get("measurement", {})
        if isinstance(meas, dict):
            sources.append(meas.get("point_debug_base", {}))
    for src in sources:
        if not isinstance(src, dict):
            continue
        rep = src.get("representation", {})
        if not isinstance(rep, dict):
            continue
        idx = rep.get("assigned_layout_index")
        try:
            idx = int(idx)
        except Exception:
            continue
        return idx
    return None


def _expected_projected_ellipse_stats_from_pose(pose_ctx, K, circle_centers_w):
    runtime_cache = pose_ctx.get("runtime_cache") if isinstance(pose_ctx, dict) else None
    if not isinstance(runtime_cache, dict):
        try:
            runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
        except Exception:
            runtime_cache = {}
    ellipses = list(runtime_cache.get("expected_ellipses", [])) if isinstance(runtime_cache, dict) else []
    majors = []
    ratios = []
    for ell in ellipses:
        if ell is None:
            continue
        try:
            axes = np.asarray(ell[1], dtype=np.float64).reshape(2)
        except Exception:
            continue
        if axes.size != 2 or not np.all(np.isfinite(axes)) or np.min(axes) <= 1e-6:
            continue
        majors.append(float(np.max(axes)))
        ratios.append(float(np.max(axes) / max(np.min(axes), 1e-6)))
    if not majors:
        return {"median_major_px": None, "median_axis_ratio": None, "num_expected_ellipses": 0}
    return {
        "median_major_px": float(np.median(np.asarray(majors, dtype=np.float64))),
        "median_axis_ratio": float(np.median(np.asarray(ratios, dtype=np.float64))),
        "num_expected_ellipses": int(len(majors)),
    }


def _choose_layout_pnp_center_source(pose_ctx, K, circle_centers_w):
    stats = _expected_projected_ellipse_stats_from_pose(pose_ctx, K, circle_centers_w)
    ratio = stats.get("median_axis_ratio")
    major = stats.get("median_major_px")
    use_ellipse = (
        ratio is not None
        and major is not None
        and np.isfinite(float(ratio))
        and np.isfinite(float(major))
        and float(ratio) > 1.18
        and float(major) > 25.0
    )
    return ("direct_ellipse_center" if use_ellipse else "weighted_gray_center"), {
        "rule": "use_direct_ellipse_when_projected_axis_ratio_gt_1p18_and_major_gt_25px_else_weighted_gray",
        "chosen_source": "direct_ellipse_center" if use_ellipse else "weighted_gray_center",
        **stats,
    }


def _solve_layout_pnp_from_measurement_centers(measurements, K, pose_ctx, circle_centers_w):
    center_key, debug = _choose_layout_pnp_center_source(pose_ctx, K, circle_centers_w)
    obj_pts = []
    img_pts = []
    used_layout_indices = []
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        if idx is None or idx < 0 or idx >= len(circle_centers_w):
            continue
        uv = np.asarray(item.get(center_key, [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if uv.shape != (2,) or not np.all(np.isfinite(uv)):
            continue
        obj_pts.append(np.asarray(circle_centers_w[idx], dtype=np.float64).reshape(3))
        img_pts.append(uv)
        used_layout_indices.append(int(idx))

    debug = dict(debug)
    debug["used_layout_indices"] = used_layout_indices
    debug["num_correspondences"] = int(len(obj_pts))
    if len(obj_pts) < 4:
        debug.update({"used": False, "reason": "too_few_layout_labeled_measurements"})
        return None, debug

    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    base_rvec = np.asarray(pose_ctx.get("rvec1", np.zeros((3, 1))), dtype=np.float64).reshape(3, 1)
    base_tvec = np.asarray(pose_ctx.get("tvec1", np.zeros((3, 1))), dtype=np.float64).reshape(3, 1)

    solutions = []
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            if flag == cv2.SOLVEPNP_ITERATIVE:
                ok, rv, tv = cv2.solvePnP(
                    obj_pts, img_pts, K, np.zeros(5),
                    rvec=base_rvec.copy(), tvec=base_tvec.copy(),
                    useExtrinsicGuess=True, flags=flag,
                )
                if ok:
                    solutions.append((np.asarray(rv, dtype=np.float64).reshape(3, 1), np.asarray(tv, dtype=np.float64).reshape(3, 1), int(flag)))
            else:
                ret = cv2.solvePnPGeneric(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                if bool(ret[0]):
                    for rv, tv in zip(ret[1], ret[2]):
                        solutions.append((np.asarray(rv, dtype=np.float64).reshape(3, 1), np.asarray(tv, dtype=np.float64).reshape(3, 1), int(flag)))
        except cv2.error:
            continue

    best = None
    best_score = np.inf
    for rv, tv, flag in solutions:
        R, _ = cv2.Rodrigues(rv)
        depths = (R @ obj_pts.T + tv.reshape(3, 1))[2]
        if depths.size == 0 or np.min(depths) <= 1.0:
            continue
        reproj, _ = cv2.projectPoints(obj_pts, rv, tv, K, np.zeros(5))
        reproj = reproj.reshape(-1, 2)
        err = np.linalg.norm(reproj - img_pts, axis=1)
        pose_shift = float(np.linalg.norm(tv.reshape(3) - base_tvec.reshape(3)) / 100.0)
        score = float(np.median(err) + 0.10 * pose_shift)
        if score < best_score:
            best_score = score
            best = (rv, tv, flag, err)

    if best is None:
        debug.update({"used": False, "reason": "no_positive_depth_layout_pnp_solution"})
        return None, debug

    rv, tv, flag, err = best
    proj_all, _ = cv2.projectPoints(np.asarray(circle_centers_w, dtype=np.float64), rv, tv, K, np.zeros(5))
    proj_all = proj_all.reshape(-1, 2).astype(np.float64)
    debug.update({
        "used": True,
        "reason": "layout_pnp_from_roi_frontend_centers_no_checkerboard",
        "solver_flag": int(flag),
        "median_internal_reprojection_px": float(np.median(err)),
        "max_internal_reprojection_px": float(np.max(err)),
        "score": float(best_score),
    })
    return proj_all, debug


def _attach_layout_constrained_pnp_centers(measurements, K, pose_ctx, circle_centers_w):
    proj_all, debug = _solve_layout_pnp_from_measurement_centers(measurements, K, pose_ctx, circle_centers_w)
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        if proj_all is not None and idx is not None and 0 <= idx < len(proj_all):
            item["layout_constrained_pnp_center"] = np.asarray(proj_all[idx], dtype=np.float64)
        else:
            item["layout_constrained_pnp_center"] = np.array([np.nan, np.nan], dtype=np.float64)
        item["layout_constrained_pnp_debug"] = debug
    return measurements, debug


def _camera_normalize_points_uv(points_uv, K):
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = (pts[:, 0] - cx) / max(fx, 1e-12)
    out[:, 1] = (pts[:, 1] - cy) / max(fy, 1e-12)
    return out


def _camera_denormalize_points_xy(points_xy, K):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = pts[:, 0] * float(K[0, 0]) + float(K[0, 2])
    out[:, 1] = pts[:, 1] * float(K[1, 1]) + float(K[1, 2])
    return out


def _apply_homography_2d(H, points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    ph = np.column_stack([pts, np.ones((len(pts),), dtype=np.float64)])
    qh = (H @ ph.T).T
    den = qh[:, 2:3]
    valid = np.abs(den[:, 0]) > 1e-12
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = qh[valid, :2] / den[valid]
    return out


def _params_to_homography(params8):
    p = np.asarray(params8, dtype=np.float64).reshape(8)
    return np.array([
        [p[0], p[1], p[2]],
        [p[3], p[4], p[5]],
        [p[6], p[7], 1.0],
    ], dtype=np.float64)


def _homography_to_params(H):
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if abs(float(H[2, 2])) > 1e-12:
        H = H / float(H[2, 2])
    return np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]], dtype=np.float64)


def _subsample_edge_points_for_joint_homography(item, max_points):
    pts = np.asarray(item.get("refined_pts", []), dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        pts = np.asarray((item.get("edge_set", {}) or {}).get("points", []), dtype=np.float64).reshape(-1, 2)
    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]
    if len(pts) <= max_points:
        return pts
    ell = item.get("ellipse_cache")
    if ell is None and isinstance(item.get("measurement", {}), dict):
        ell = item["measurement"].get("work_ellipse")
    try:
        ang = np.mod(_ellipse_param_angles(ell, pts), 2.0 * np.pi)
        order = np.argsort(ang)
        take = np.linspace(0, len(order) - 1, int(max_points), dtype=np.int32)
        return pts[order[take]]
    except Exception:
        take = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int32)
        return pts[take]


def _joint_edge_homography_residuals(params8, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight):
    G = _params_to_homography(params8)
    residuals = []
    if len(edge_xyn):
        xy = _apply_homography_2d(G, edge_xyn)
        for q, idx in zip(xy, edge_layout_idx):
            if not np.all(np.isfinite(q)):
                residuals.append(25.0)
                continue
            d = float(np.linalg.norm(q - centers_n[int(idx)]))
            residuals.append((d - radius_n) / max(radius_n, 1e-12))
    if len(center_xyn):
        xy_c = _apply_homography_2d(G, center_xyn)
        for q, idx in zip(xy_c, center_layout_idx):
            if not np.all(np.isfinite(q)):
                residuals.extend([25.0 * center_weight, 25.0 * center_weight])
                continue
            diff = (q - centers_n[int(idx)]) / max(radius_n, 1e-12)
            residuals.extend((center_weight * diff).tolist())
    return np.asarray(residuals, dtype=np.float64)


def _optimize_joint_edge_homography(G0, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight, iters):
    p = _homography_to_params(G0)
    history = []
    lam = 2e-3
    eps = 1e-5
    for _ in range(int(iters)):
        r = _joint_edge_homography_residuals(p, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight)
        if r.size < 8 or not np.all(np.isfinite(r)):
            break
        scale = max(0.20, 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9)
        robust = (1.0 + (r / (2.5 * scale)) ** 2) ** -1
        J = np.zeros((r.size, 8), dtype=np.float64)
        for j in range(8):
            pp = p.copy()
            pp[j] += eps
            rp = _joint_edge_homography_residuals(pp, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight)
            J[:, j] = (rp - r) / eps
        WJ = J * robust.reshape(-1, 1)
        A = J.T @ WJ + np.eye(8, dtype=np.float64) * lam
        b = J.T @ (robust * r)
        try:
            delta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm):
            break
        best_p = p
        best_cost = float(np.mean(robust * r * r))
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125):
            cand_p = p + step * delta
            rc = _joint_edge_homography_residuals(cand_p, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight)
            if rc.size != r.size or not np.all(np.isfinite(rc)):
                continue
            cand_cost = float(np.mean((1.0 + (rc / (2.5 * scale)) ** 2) ** -1 * rc * rc))
            if cand_cost < best_cost:
                best_p = cand_p
                best_cost = cand_cost
                accepted = True
                break
        history.append({"cost": best_cost, "delta_norm": delta_norm, "accepted": bool(accepted)})
        p = best_p
        if delta_norm < 1e-7:
            break
    r_final = _joint_edge_homography_residuals(p, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx, centers_n, radius_n, center_weight)
    return _params_to_homography(p), {
        "iterations": int(len(history)),
        "history": history,
        "final_residual_median": float(np.median(np.abs(r_final))) if r_final.size else None,
        "final_residual_mean": float(np.mean(np.abs(r_final))) if r_final.size else None,
    }


def _physical_edge_params_to_model(params, num_lights=4):
    H = _params_to_homography(params[:8])
    offsets = np.asarray(params[8:8 + 3 * int(num_lights)], dtype=np.float64).reshape(int(num_lights), 3)
    return H, offsets


def _physical_edge_circle_residuals(params, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
                                    centers_n, radius_n, center_weight, offset_weight):
    G, offsets = _physical_edge_params_to_model(params, len(centers_n))
    residuals = []
    if len(edge_xyn):
        xy = _apply_homography_2d(G, edge_xyn)
        for q, idx in zip(xy, edge_layout_idx):
            idx = int(idx)
            if idx < 0 or idx >= len(centers_n) or not np.all(np.isfinite(q)):
                residuals.append(25.0)
                continue
            vec = q - centers_n[idx]
            d = float(np.linalg.norm(vec))
            if not np.isfinite(d) or d <= 1e-12:
                residuals.append(25.0)
                continue
            theta = math.atan2(float(vec[1]), float(vec[0]))
            off = float(offsets[idx, 0] + offsets[idx, 1] * math.cos(theta) + offsets[idx, 2] * math.sin(theta))
            residuals.append((d - radius_n - off) / max(radius_n, 1e-12))
    if len(center_xyn):
        xy_c = _apply_homography_2d(G, center_xyn)
        for q, idx in zip(xy_c, center_layout_idx):
            idx = int(idx)
            if idx < 0 or idx >= len(centers_n) or not np.all(np.isfinite(q)):
                residuals.extend([25.0 * center_weight, 25.0 * center_weight])
                continue
            diff = (q - centers_n[idx]) / max(radius_n, 1e-12)
            residuals.extend((center_weight * diff).tolist())
    if offset_weight > 0:
        for idx in range(len(centers_n)):
            residuals.append(offset_weight * offsets[idx, 0] / max(radius_n, 1e-12))
            residuals.append(0.75 * offset_weight * offsets[idx, 1] / max(radius_n, 1e-12))
            residuals.append(0.75 * offset_weight * offsets[idx, 2] / max(radius_n, 1e-12))
    return np.asarray(residuals, dtype=np.float64)


def _optimize_physical_edge_circle_homography(G0, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
                                              centers_n, radius_n, center_weight, offset_weight, iters):
    p = np.zeros(8 + 3 * len(centers_n), dtype=np.float64)
    p[:8] = _homography_to_params(G0)
    history = []
    lam = 3e-3
    eps = 1e-5
    for _ in range(int(max(0, iters))):
        r = _physical_edge_circle_residuals(
            p, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
            centers_n, radius_n, center_weight, offset_weight
        )
        if r.size < p.size or not np.all(np.isfinite(r)):
            break
        scale = max(0.08, 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9)
        robust = (1.0 + (r / (2.5 * scale)) ** 2) ** -1
        J = np.zeros((r.size, p.size), dtype=np.float64)
        for j in range(p.size):
            pp = p.copy()
            pp[j] += eps
            rp = _physical_edge_circle_residuals(
                pp, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
                centers_n, radius_n, center_weight, offset_weight
            )
            if rp.size != r.size:
                return _physical_edge_params_to_model(p, len(centers_n))[0], _physical_edge_params_to_model(p, len(centers_n))[1], {
                    "iterations": int(len(history)),
                    "reason": "residual_size_changed",
                    "history": history,
                }
            J[:, j] = (rp - r) / eps
        WJ = J * robust.reshape(-1, 1)
        A = J.T @ WJ + np.eye(p.size, dtype=np.float64) * lam
        b = J.T @ (robust * r)
        try:
            delta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm):
            break
        if delta_norm > 1.0:
            delta *= 1.0 / max(delta_norm, 1e-12)
        best_p = p
        best_cost = float(np.mean(robust * r * r))
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125):
            cand = p + step * delta
            rc = _physical_edge_circle_residuals(
                cand, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
                centers_n, radius_n, center_weight, offset_weight
            )
            if rc.size != r.size or not np.all(np.isfinite(rc)):
                continue
            rc_robust = (1.0 + (rc / (2.5 * scale)) ** 2) ** -1
            cand_cost = float(np.mean(rc_robust * rc * rc))
            if cand_cost < best_cost:
                best_p = cand
                best_cost = cand_cost
                accepted = True
                break
        p = best_p
        history.append({"cost": best_cost, "delta_norm": delta_norm, "accepted": bool(accepted)})
        if not accepted or delta_norm < 1e-6:
            break
    G, offsets = _physical_edge_params_to_model(p, len(centers_n))
    r_final = _physical_edge_circle_residuals(
        p, edge_xyn, edge_layout_idx, center_xyn, center_layout_idx,
        centers_n, radius_n, center_weight, offset_weight
    )
    return G, offsets, {
        "iterations": int(len(history)),
        "history": history,
        "final_residual_median": float(np.median(np.abs(r_final))) if r_final.size else None,
        "final_residual_mean": float(np.mean(np.abs(r_final))) if r_final.size else None,
        "offsets_radius_units": (offsets / max(radius_n, 1e-12)).astype(np.float64).tolist(),
    }


def _attach_physical_edge_circle_correction_centers(measurements, K, circle_centers_w):
    centers_xy = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    origin = np.mean(centers_xy, axis=0)
    layout_scale = float(np.median(np.linalg.norm(centers_xy - origin.reshape(1, 2), axis=1)))
    if not np.isfinite(layout_scale) or layout_scale <= 1e-9:
        layout_scale = 400.0
    centers_n = (centers_xy - origin.reshape(1, 2)) / layout_scale
    radius_n = float(LIGHT_RADIUS_MM / layout_scale)
    max_pts = int(globals().get("PHYSICAL_EDGE_MAX_POINTS_PER_LIGHT", 32))

    center_uv, center_idx, edge_uv, edge_idx = [], [], [], []
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        if idx is None or idx < 0 or idx >= len(centers_n):
            continue
        step4 = (item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy")
        if step4 is not None:
            uv = np.asarray(step4, dtype=np.float64).reshape(2)
            if np.all(np.isfinite(uv)):
                center_uv.append(uv)
                center_idx.append(int(idx))
        pts = _subsample_edge_points_for_joint_homography(item, max_pts)
        if len(pts) >= 8:
            edge_uv.extend(np.asarray(pts, dtype=np.float64).reshape(-1, 2).tolist())
            edge_idx.extend([int(idx)] * len(pts))

    debug = {
        "method": "four_light_edge_level_physical_circle_constraint_with_low_order_edge_offset",
        "used": False,
        "num_centers": int(len(center_uv)),
        "num_edge_points": int(len(edge_uv)),
        "max_points_per_light": int(max_pts),
        "radius_mm": float(LIGHT_RADIUS_MM),
        "uses_checkerboard_pose": False,
    }
    if len(center_uv) < 4 or len(edge_uv) < 32:
        debug["reason"] = "insufficient_labeled_centers_or_edges"
        for item in measurements:
            item["physical_edge_circle_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["physical_edge_circle_debug"] = debug
        return measurements, debug

    center_uv = np.asarray(center_uv, dtype=np.float64).reshape(-1, 2)
    center_idx = np.asarray(center_idx, dtype=np.int32).reshape(-1)
    edge_uv = np.asarray(edge_uv, dtype=np.float64).reshape(-1, 2)
    edge_idx = np.asarray(edge_idx, dtype=np.int32).reshape(-1)
    center_xyn = _camera_normalize_points_uv(center_uv, K)
    edge_xyn = _camera_normalize_points_uv(edge_uv, K)
    G0, _ = cv2.findHomography(center_xyn.astype(np.float64), centers_n[center_idx].astype(np.float64), method=0)
    if G0 is None or not np.all(np.isfinite(G0)):
        debug["reason"] = "initial_homography_failed"
        for item in measurements:
            item["physical_edge_circle_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["physical_edge_circle_debug"] = debug
        return measurements, debug

    center_weight = float(globals().get("PHYSICAL_EDGE_CENTER_PRIOR_WEIGHT", 0.22))
    offset_weight = float(globals().get("PHYSICAL_EDGE_OFFSET_PRIOR_WEIGHT", 0.18))
    iters = int(globals().get("PHYSICAL_EDGE_ITERS", 4))
    G_opt, offsets, opt_dbg = _optimize_physical_edge_circle_homography(
        np.asarray(G0, dtype=np.float64).reshape(3, 3),
        edge_xyn,
        edge_idx,
        center_xyn,
        center_idx,
        centers_n,
        radius_n,
        center_weight,
        offset_weight,
        iters,
    )
    try:
        H_opt = np.linalg.inv(G_opt)
        proj_center_n = _apply_homography_2d(H_opt, centers_n)
        proj_center_uv = _camera_denormalize_points_xy(proj_center_n, K)
    except Exception as e:
        debug.update({"reason": f"optimized_projection_failed:{type(e).__name__}", "optimization": opt_dbg})
        for item in measurements:
            item["physical_edge_circle_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["physical_edge_circle_debug"] = debug
        return measurements, debug

    beta = float(np.clip(globals().get("PHYSICAL_EDGE_OUTPUT_BLEND", 0.18), 0.0, 1.0))
    debug.update({
        "used": True,
        "reason": "optimized_edge_level_circle_constraint_with_low_order_offset",
        "center_prior_weight": float(center_weight),
        "offset_prior_weight": float(offset_weight),
        "output_blend": float(beta),
        "layout_scale_mm": float(layout_scale),
        "optimization": opt_dbg,
    })
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        selected = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if idx is not None and 0 <= idx < len(proj_center_uv):
            raw = np.asarray(proj_center_uv[int(idx)], dtype=np.float64).reshape(2)
            base = np.asarray(item.get("antipodal_edge_center", selected), dtype=np.float64).reshape(2)
            if np.all(np.isfinite(base)) and np.all(np.isfinite(raw)):
                center = base + beta * (raw - base)
            else:
                center = raw if np.all(np.isfinite(raw)) else selected
            item["physical_edge_circle_center"] = center.astype(np.float64)
            item["physical_edge_circle_raw_center"] = raw.astype(np.float64)
            item["physical_edge_circle_offset_radius_units"] = (offsets[int(idx)] / max(radius_n, 1e-12)).astype(np.float64).tolist()
        else:
            item["physical_edge_circle_center"] = selected
            item["physical_edge_circle_raw_center"] = np.array([np.nan, np.nan], dtype=np.float64)
            item["physical_edge_circle_offset_radius_units"] = None
        item["physical_edge_circle_debug"] = debug
    return measurements, debug


def _fit_known_radius_center_2d(points_xy, radius, init_center, prior_center=None, prior_weight=0.08, iters=5):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    radius = float(max(radius, 1e-12))
    center = np.asarray(init_center, dtype=np.float64).reshape(2).copy()
    prior = np.asarray(prior_center if prior_center is not None else init_center, dtype=np.float64).reshape(2)
    prior_weight = float(max(prior_weight, 0.0))
    history = []
    if len(pts) < 6 or not np.all(np.isfinite(pts)) or not np.all(np.isfinite(center)):
        return center, {"used": False, "reason": "insufficient_points_or_invalid_init"}

    for _ in range(int(max(0, iters))):
        diff = pts - center.reshape(1, 2)
        dist = np.linalg.norm(diff, axis=1)
        good = np.isfinite(dist) & (dist > 1e-12)
        if np.count_nonzero(good) < 6:
            break
        diff = diff[good]
        dist = dist[good]
        residual = (dist - radius) / radius
        J = -diff / dist.reshape(-1, 1) / radius
        scale = max(0.05, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-9)
        robust = (1.0 + (residual / (2.5 * scale)) ** 2) ** -1
        A = J.T @ (J * robust.reshape(-1, 1))
        b = J.T @ (robust * residual)
        if prior_weight > 0.0 and np.all(np.isfinite(prior)):
            sw = math.sqrt(prior_weight)
            rp = sw * (center - prior) / radius
            Jp = sw * np.eye(2, dtype=np.float64) / radius
            A += Jp.T @ Jp
            b += Jp.T @ rp
        A += np.eye(2, dtype=np.float64) * 1e-6
        try:
            delta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm):
            break
        max_step = 0.75 * radius
        if delta_norm > max_step:
            delta *= max_step / max(delta_norm, 1e-12)
        old_center = center.copy()
        old_cost = float(np.mean(robust * residual * residual))
        accepted = False
        for step in (1.0, 0.5, 0.25):
            cand = old_center + step * delta
            cand_dist = np.linalg.norm(pts - cand.reshape(1, 2), axis=1)
            cand_res = (cand_dist - radius) / radius
            cand_cost = float(np.mean((1.0 + (cand_res / (2.5 * scale)) ** 2) ** -1 * cand_res * cand_res))
            if np.isfinite(cand_cost) and cand_cost <= old_cost:
                center = cand
                old_cost = cand_cost
                accepted = True
                break
        history.append({"cost": old_cost, "delta_norm": delta_norm, "accepted": bool(accepted)})
        if not accepted or delta_norm < 1e-7:
            break

    final_dist = np.linalg.norm(pts - center.reshape(1, 2), axis=1)
    final_res = (final_dist - radius) / radius
    return center, {
        "used": True,
        "iterations": int(len(history)),
        "history": history,
        "final_residual_median": float(np.median(np.abs(final_res))) if final_res.size else None,
        "final_residual_mean": float(np.mean(np.abs(final_res))) if final_res.size else None,
        "center_shift_radius_units": float(np.linalg.norm(center - prior) / radius) if np.all(np.isfinite(prior)) else None,
    }


def _attach_joint_layout_edge_homography_centers(measurements, K, circle_centers_w):
    centers_xy = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    origin = np.mean(centers_xy, axis=0)
    layout_scale = float(np.median(np.linalg.norm(centers_xy - origin.reshape(1, 2), axis=1)))
    if not np.isfinite(layout_scale) or layout_scale <= 1e-9:
        layout_scale = 400.0
    centers_n = (centers_xy - origin.reshape(1, 2)) / layout_scale
    radius_n = float(LIGHT_RADIUS_MM / layout_scale)

    center_uv = []
    center_idx = []
    edge_uv = []
    edge_idx = []
    max_pts = int(globals().get("JOINT_EDGE_HOMOGRAPHY_MAX_POINTS_PER_LIGHT", 48))
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        if idx is None or idx < 0 or idx >= len(centers_n):
            continue
        step4 = (item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy")
        if step4 is None:
            step4 = item.get("ellipse_cache", ((np.nan, np.nan),))[0] if item.get("ellipse_cache") is not None else None
        if step4 is not None:
            uv = np.asarray(step4, dtype=np.float64).reshape(2)
            if np.all(np.isfinite(uv)):
                center_uv.append(uv)
                center_idx.append(int(idx))
        pts = _subsample_edge_points_for_joint_homography(item, max_pts)
        if len(pts) >= 6:
            edge_uv.extend(pts.tolist())
            edge_idx.extend([int(idx)] * len(pts))

    debug = {
        "method": "joint_four_light_homography_plus_local_known_radius_edge_fit_v2",
        "used": False,
        "num_centers": int(len(center_uv)),
        "num_edge_points": int(len(edge_uv)),
        "max_points_per_light": int(max_pts),
        "radius_mm": float(LIGHT_RADIUS_MM),
    }
    if len(center_uv) < 4 or len(edge_uv) < 24:
        debug["reason"] = "insufficient_labeled_centers_or_edges"
        for item in measurements:
            item["joint_layout_edge_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["joint_layout_edge_debug"] = debug
        return measurements, debug

    center_uv = np.asarray(center_uv, dtype=np.float64).reshape(-1, 2)
    center_idx = np.asarray(center_idx, dtype=np.int32).reshape(-1)
    edge_uv = np.asarray(edge_uv, dtype=np.float64).reshape(-1, 2)
    edge_idx = np.asarray(edge_idx, dtype=np.int32).reshape(-1)
    center_xyn = _camera_normalize_points_uv(center_uv, K)
    edge_xyn = _camera_normalize_points_uv(edge_uv, K)
    center_plane_n = centers_n[center_idx]

    G0, _ = cv2.findHomography(center_xyn.astype(np.float64), center_plane_n.astype(np.float64), method=0)
    if G0 is None or not np.all(np.isfinite(G0)):
        debug["reason"] = "initial_homography_failed"
        for item in measurements:
            item["joint_layout_edge_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["joint_layout_edge_debug"] = debug
        return measurements, debug

    G0 = np.asarray(G0, dtype=np.float64).reshape(3, 3)
    center_weight = float(globals().get("JOINT_EDGE_HOMOGRAPHY_CENTER_PRIOR_WEIGHT", 0.18))
    iters = int(globals().get("JOINT_EDGE_HOMOGRAPHY_ITERS", 5))
    G_opt, opt_dbg = _optimize_joint_edge_homography(
        G0,
        edge_xyn,
        edge_idx,
        center_xyn,
        center_idx,
        centers_n,
        radius_n,
        center_weight,
        iters,
    )
    try:
        H_opt = np.linalg.inv(G_opt)
        proj_layout_center_n = _apply_homography_2d(H_opt, centers_n)
        proj_layout_center_uv = _camera_denormalize_points_xy(proj_layout_center_n, K)
    except Exception as e:
        debug.update({"reason": f"optimized_projection_failed:{type(e).__name__}", "optimization": opt_dbg})
        for item in measurements:
            item["joint_layout_edge_center"] = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64)
            item["joint_layout_edge_debug"] = debug
        return measurements, debug

    debug.update({
        "used": True,
        "reason": "global_four_light_homography_then_local_subpixel_known_radius_circle_fit",
        "center_prior_weight": float(center_weight),
        "output_blend": float(globals().get("JOINT_EDGE_HOMOGRAPHY_OUTPUT_BLEND", 0.15)),
        "local_circle_prior_weight": float(globals().get("JOINT_EDGE_LOCAL_CIRCLE_PRIOR_WEIGHT", 0.08)),
        "local_circle_iters": int(globals().get("JOINT_EDGE_LOCAL_CIRCLE_ITERS", 5)),
        "layout_scale_mm": float(layout_scale),
        "optimization": opt_dbg,
    })
    axis_ratios = []
    for item in measurements:
        ell = item.get("ellipse_cache")
        if ell is not None:
            try:
                axes = np.asarray(ell[1], dtype=np.float64).reshape(2)
                if np.all(np.isfinite(axes)) and np.min(axes) > 1e-9:
                    axis_ratios.append(float(np.max(axes) / np.min(axes)))
            except Exception:
                pass
    median_axis_ratio = float(np.median(axis_ratios)) if axis_ratios else 1.0
    blend_center = float(globals().get("JOINT_EDGE_PERSPECTIVE_BLEND_CENTER", 1.105))
    blend_width = float(max(globals().get("JOINT_EDGE_PERSPECTIVE_BLEND_WIDTH", 0.018), 1e-6))
    z = float(np.clip((median_axis_ratio - blend_center) / blend_width, -60.0, 60.0))
    edge_perspective_weight = float(1.0 / (1.0 + math.exp(z)))
    debug.update({
        "median_selected_boundary_axis_ratio": median_axis_ratio,
        "edge_perspective_weight": edge_perspective_weight,
        "perspective_blend_center": blend_center,
        "perspective_blend_width": blend_width,
    })
    local_debug = []
    for item in measurements:
        idx = _assigned_layout_index_from_packed_measurement(item)
        step4 = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if idx is not None and 0 <= idx < len(proj_layout_center_uv):
            known_center_n = centers_n[int(idx)]
            pts_uv = _subsample_edge_points_for_joint_homography(item, max_pts)
            local_center_n = known_center_n
            fit_dbg = {"used": False, "reason": "no_local_edge_fit"}
            if len(pts_uv) >= 6:
                pts_xyn = _camera_normalize_points_uv(np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2), K)
                pts_plane_n = _apply_homography_2d(G_opt, pts_xyn)
                finite = np.all(np.isfinite(pts_plane_n), axis=1)
                if np.count_nonzero(finite) >= 6:
                    local_center_n, fit_dbg = _fit_known_radius_center_2d(
                        pts_plane_n[finite],
                        radius_n,
                        known_center_n,
                        prior_center=known_center_n,
                        prior_weight=float(globals().get("JOINT_EDGE_LOCAL_CIRCLE_PRIOR_WEIGHT", 0.08)),
                        iters=int(globals().get("JOINT_EDGE_LOCAL_CIRCLE_ITERS", 5)),
                    )
            try:
                raw_center_xyn = _apply_homography_2d(H_opt, np.asarray(local_center_n, dtype=np.float64).reshape(1, 2))
                raw_center = _camera_denormalize_points_xy(raw_center_xyn, K).reshape(2)
            except Exception:
                raw_center = np.asarray(proj_layout_center_uv[int(idx)], dtype=np.float64)
            if np.all(np.isfinite(step4)):
                beta = float(globals().get("JOINT_EDGE_HOMOGRAPHY_OUTPUT_BLEND", 0.15))
                damped_local_center = step4 + beta * (raw_center - step4)
            else:
                damped_local_center = raw_center
            layout_anchor = np.asarray(item.get("layout_constrained_pnp_center", np.asarray(proj_layout_center_uv[int(idx)], dtype=np.float64)), dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(layout_anchor)):
                layout_anchor = np.asarray(proj_layout_center_uv[int(idx)], dtype=np.float64)
            if np.all(np.isfinite(layout_anchor)) and np.all(np.isfinite(damped_local_center)):
                item["joint_layout_edge_center"] = layout_anchor + edge_perspective_weight * (damped_local_center - layout_anchor)
            else:
                item["joint_layout_edge_center"] = damped_local_center
            item["joint_layout_edge_raw_center"] = raw_center
            item["joint_layout_edge_local_center"] = damped_local_center
            item["joint_layout_edge_layout_anchor"] = layout_anchor
            item["joint_layout_edge_layout_center"] = np.asarray(proj_layout_center_uv[int(idx)], dtype=np.float64)
            item["joint_layout_edge_local_fit_debug"] = fit_dbg
            local_debug.append({
                "layout_index": int(idx),
                "fit": fit_dbg,
                "raw_shift_from_selected_boundary_px": float(np.linalg.norm(raw_center - step4)) if np.all(np.isfinite(step4)) and np.all(np.isfinite(raw_center)) else None,
                "layout_anchor_shift_from_selected_boundary_px": float(np.linalg.norm(layout_anchor - step4)) if np.all(np.isfinite(step4)) and np.all(np.isfinite(layout_anchor)) else None,
                "edge_perspective_weight": edge_perspective_weight,
            })
        else:
            item["joint_layout_edge_center"] = step4
            item["joint_layout_edge_raw_center"] = np.array([np.nan, np.nan], dtype=np.float64)
            item["joint_layout_edge_local_center"] = np.array([np.nan, np.nan], dtype=np.float64)
            item["joint_layout_edge_layout_anchor"] = np.array([np.nan, np.nan], dtype=np.float64)
            item["joint_layout_edge_layout_center"] = np.array([np.nan, np.nan], dtype=np.float64)
            item["joint_layout_edge_local_fit_debug"] = {"used": False, "reason": "missing_layout_assignment"}
        item["joint_layout_edge_debug"] = debug
    debug["local_circle_fit"] = local_debug
    return measurements, debug


def _attach_final_edge_layout_backend_centers(measurements):
    base = float(globals().get("FINAL_EDGE_LAYOUT_JOINT_BLEND_BASE", 0.20))
    blend_min = float(globals().get("FINAL_EDGE_LAYOUT_JOINT_BLEND_MIN", 0.04))
    blend_max = float(globals().get("FINAL_EDGE_LAYOUT_JOINT_BLEND_MAX", 0.35))
    shifts = []
    radii = []
    edge_success = []
    for item in measurements:
        anti = np.asarray(item.get("antipodal_edge_center", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        joint = np.asarray(item.get("joint_layout_edge_center", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if np.all(np.isfinite(anti)) and np.all(np.isfinite(joint)):
            shifts.append(float(np.linalg.norm(joint - anti)))
        ell = item.get("ellipse_cache")
        if ell is not None:
            try:
                axes = np.asarray(ell[1], dtype=np.float64).reshape(2)
                if np.all(np.isfinite(axes)) and np.min(axes) > 1e-9:
                    radii.append(float(0.25 * np.sum(axes)))
            except Exception:
                pass
        dbg = item.get("point_debug", {})
        if isinstance(dbg, dict) and dbg.get("edge_fit_success_ratio") is not None:
            try:
                edge_success.append(float(dbg.get("edge_fit_success_ratio")))
            except Exception:
                pass

    med_radius = float(np.median(radii)) if radii else 20.0
    med_shift = float(np.median(shifts)) if shifts else 0.0
    shift_radius_units = float(med_shift / max(med_radius, 1e-9))
    agreement_scale = float(max(globals().get("FINAL_EDGE_LAYOUT_AGREEMENT_SCALE_RADIUS", 0.45), 1e-6))
    agreement_factor = float(math.exp(-((shift_radius_units / agreement_scale) ** 2)))

    joint_debug = None
    for item in measurements:
        jd = item.get("joint_layout_edge_debug")
        if isinstance(jd, dict) and jd.get("used"):
            joint_debug = jd
            break
    opt = (joint_debug or {}).get("optimization", {}) if isinstance(joint_debug, dict) else {}
    residual = opt.get("final_residual_median", None) if isinstance(opt, dict) else None
    try:
        residual = float(residual)
    except Exception:
        residual = None
    residual_factor = float(1.0 / (1.0 + (max(residual or 0.0, 0.0) / 0.35) ** 2))

    axis_ratio = (joint_debug or {}).get("median_selected_boundary_axis_ratio", None) if isinstance(joint_debug, dict) else None
    try:
        axis_ratio = float(axis_ratio)
    except Exception:
        axis_ratio = 1.0
    perspective_boost = float(1.0 / (1.0 + math.exp(-float(np.clip((axis_ratio - 1.18) / 0.035, -60.0, 60.0)))))
    perspective_factor = float(0.85 + 1.75 * perspective_boost)

    edge_factor = float(np.clip(0.75 + 0.35 * (float(np.mean(edge_success)) if edge_success else 0.75), 0.75, 1.10))
    blend = base * (0.65 + 0.70 * agreement_factor) * (0.75 + 0.35 * residual_factor) * perspective_factor * edge_factor
    blend = float(np.clip(blend, blend_min, blend_max))
    debug = {
        "method": "final_selected_boundary_subpixel_edge_plus_four_light_layout_backend",
        "used": True,
        "blend_mode": "image_adaptive_observable_geometry",
        "base_joint_weight": float(base),
        "antipodal_edge_weight": float(1.0 - blend),
        "joint_four_light_backend_weight": float(blend),
        "blend_min": float(blend_min),
        "blend_max": float(blend_max),
        "median_joint_antipodal_shift_px": med_shift,
        "median_selected_boundary_radius_px": med_radius,
        "shift_radius_units": shift_radius_units,
        "agreement_factor": agreement_factor,
        "joint_residual_factor": residual_factor,
        "median_selected_boundary_axis_ratio": axis_ratio,
        "perspective_boost": perspective_boost,
        "perspective_factor": perspective_factor,
        "edge_success_factor": edge_factor,
        "uses_checkerboard_pose": False,
    }
    for item in measurements:
        anti = np.asarray(item.get("antipodal_edge_center", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        joint = np.asarray(item.get("joint_layout_edge_center", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        selected = np.asarray((item.get("step4_work_ellipse_ablation", {}) or {}).get("center_xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if np.all(np.isfinite(anti)) and np.all(np.isfinite(joint)):
            center = anti + blend * (joint - anti)
            source = "antipodal_edge_plus_joint_four_light_backend"
        elif np.all(np.isfinite(anti)):
            center = anti
            source = "antipodal_edge_fallback"
        elif np.all(np.isfinite(joint)):
            center = joint
            source = "joint_four_light_backend_fallback"
        else:
            center = selected
            source = "selected_boundary_fallback"
        item["final_edge_layout_center"] = center.astype(np.float64)
        item["final_edge_layout_debug"] = dict(debug, source=source)
    return measurements, debug


def process_single_image(chess_img_path, light_img_path, K, save_dir=None, show=False, roi_points=None, final_measurement_vis=False, frontend_mode='legacy', pose_source='light_only'):
    pose_source = str(pose_source)
    if pose_source != 'light_only':
        raise ValueError("主算法流程禁止使用棋盘格位姿；pose_source 必须为 light_only")
    light_tmp = cv2.imread(light_img_path)
    if light_tmp is None:
        raise RuntimeError(f"图像读取失败: light={light_img_path}")
    chess_img = light_tmp.copy()
    gray_chess = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)
    pose_ctx = estimate_pose_from_light_layout(light_tmp, K, roi_points=roi_points)
    _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)

    light_img = cv2.imread(light_img_path)
    if light_img is None:
        raise RuntimeError(f"图像读取失败: light={light_img_path}")

    inputs_ctx = prepare_manual_roi({
        'chess_img': chess_img,
        'light_img': light_img,
        'gray_chess': gray_chess,
        'gray_light': cv2.cvtColor(light_img, cv2.COLOR_BGR2GRAY),
    }, roi_points=roi_points)
    roi_ctx = {'roi_points': roi_points}
    proj_ctx = pose_ctx.get('runtime_cache', {}).get('projection_ctx') if isinstance(pose_ctx.get('runtime_cache'), dict) else None
    if proj_ctx is None:
        _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
        proj_ctx = pose_ctx['runtime_cache']['projection_ctx']

    max_frontend_contours = 8
    candidates = detect_light_candidates(inputs_ctx['light_img_proc'], max_contours=max_frontend_contours)
    raw_candidate_count = len(candidates)
    candidates = _select_four_candidate_contours_by_pose_layout(candidates, pose_ctx)

    selected_measurements, arc_global_debug = build_selected_measurements_for_pipeline_joint_geometry(
        gray_light=inputs_ctx['gray_light_proc'],
        contours=candidates,
        K=K,
        pose_ctx=pose_ctx,
        circle_centers_w=circle_centers_w,
    )
    selected_measurements = _apply_frontend_mode_to_selected_measurements(inputs_ctx['gray_light_proc'], selected_measurements, frontend_mode)

    initial_measurements = [
        build_candidate_measurement_from_preselected(sel, inputs_ctx['gray_light_proc'])
        for sel in selected_measurements
    ]
    enable_measurement_pose_refine = os.environ.get("PAGC_ENABLE_MEASUREMENT_POSE_REFINE", "0").strip().lower() in ("1", "true", "yes", "on")
    if pose_source == 'light_only' and enable_measurement_pose_refine:
        pose_ctx, measurement_pose_refine_debug = _pose_ctx_from_light_measurement_centers(
            pose_ctx,
            K,
            initial_measurements,
            circle_centers_w,
        )
        proj_ctx = pose_ctx['runtime_cache']['projection_ctx']
    else:
        measurement_pose_refine_debug = {
            "used": False,
            "reason": "disabled_by_default" if pose_source == 'light_only' else "pose_source_not_light_only",
            "enable_env": "PAGC_ENABLE_MEASUREMENT_POSE_REFINE=1",
        }

    # 消融分支：Step 4 工作轮廓 -> 图像平面椭圆中心
    step4_work_ellipse_results = []
    proj_pose_final = pose_ctx.get('proj_true_eval', None)
    for light_idx, sel in enumerate(selected_measurements):
        ablation_item = fit_step4_work_ellipse_result(sel, proj_pose_final=proj_pose_final)
        ablation_item['light_index'] = int(light_idx)
        step4_work_ellipse_results.append(ablation_item)

    measurements = []
    for light_idx, meas in enumerate(initial_measurements):
        edge_set = estimate_subpixel_edge_set(inputs_ctx['gray_light_proc'], meas, proj_ctx=proj_ctx)
        geo_est = estimate_geometric_center(edge_set, meas, K, pose_ctx['n_c1'], pose_ctx['d_c1'], proj_ctx=proj_ctx)
        extra_baselines = compute_additional_baselines_for_measurement(meas, inputs_ctx['gray_light_proc'], proj_ctx=proj_ctx)
        packed = pack_candidate_result(meas['contour'], meas, edge_set, geo_est, None, extra_baselines=extra_baselines)
        packed['step4_work_ellipse_ablation'] = step4_work_ellipse_results[light_idx]
        measurements.append(packed)

    measurements, layout_constrained_pnp_debug = _attach_layout_constrained_pnp_centers(
        measurements,
        K,
        pose_ctx,
        circle_centers_w,
    )
    measurements, physical_edge_circle_debug = _attach_physical_edge_circle_correction_centers(
        measurements,
        K,
        circle_centers_w,
    )
    measurements, joint_layout_edge_debug = _attach_joint_layout_edge_homography_centers(
        measurements,
        K,
        circle_centers_w,
    )
    measurements, final_edge_layout_debug = _attach_final_edge_layout_backend_centers(measurements)

    measurements, pose_assignment = attach_uncertainty_fusion(
        measurements,
        proj_pose_final=pose_ctx['proj_pose_final'],
        pose_sigma_px=pose_ctx.get('pose_sigma_px'),
    )
    measurement_ctx = build_candidate_measurements(measurements)
    measurement_ctx['pose_assignment'] = pose_assignment
    measurement_ctx['roi_ctx'] = roi_ctx
    measurement_ctx['fast_normal_offset_edge_debug'] = arc_global_debug

    eval_ctx = evaluate_method_suite(
        light_img_path,
        measurement_ctx['pred_bbox'],
        measurement_ctx['pred_gray_centroid'],
        measurement_ctx['pred_ellipse'],
        measurement_ctx['pred_step4_work_ellipse'],
        measurement_ctx['pred_geo_pass1'],
        measurement_ctx['pred_geo_bias_pass1'],
        pose_ctx['proj_true_eval'],
        pred_weighted_gray=measurement_ctx.get('pred_weighted_gray'),
        pred_circle=measurement_ctx.get('pred_circle'),
        pred_ellipse_backproj=measurement_ctx.get('pred_ellipse_backproj'),
        pred_homography_rectified=measurement_ctx.get('pred_homography_rectified'),
        pred_layout_constrained_pnp=measurement_ctx.get('pred_layout_constrained_pnp'),
        pred_antipodal_edge=measurement_ctx.get('pred_antipodal_edge'),
        pred_joint_layout_edge=measurement_ctx.get('pred_joint_layout_edge'),
        pred_final_edge_layout=measurement_ctx.get('pred_final_edge_layout'),
        pred_physical_edge_circle=measurement_ctx.get('pred_physical_edge_circle'),
        reference_label=_reference_label_from_pose_ctx(pose_ctx),
    )

    results, summary = assemble_single_image_results(
        chess_img_path,
        light_img_path,
        save_dir,
        roi_points,
        max_frontend_contours,
        pose_ctx,
        measurement_ctx,
        eval_ctx,
    )

    results['algorithm_params']['measurement_contour'] = 'four_light_joint_geometric_consistency_selector_v1'
    results['algorithm_params']['edge_localizer'] = 'edge_first_joint_arc_normal_profile_subpixel'
    results['algorithm_params']['center_fusion'] = 'edge_first_known_radius_circle_fit_no_center_gate'
    results['algorithm_params']['evaluation_reference'] = _reference_label_from_pose_ctx(pose_ctx)
    results['algorithm_params']['pose_source'] = pose_source
    results['algorithm_params']['frontend_mode'] = str(frontend_mode)
    results['algorithm_params']['frontend_selector'] = 'physics_constrained_randomized_contour_hypothesis_search_v3_with_inward_halfmax_refiner' if frontend_mode == 'randomized_physical' else 'legacy_joint_arc_profile_subpixel_edge_first'
    results['algorithm_params']['frontend_candidate_pool_count'] = int(raw_candidate_count)
    results['algorithm_params']['frontend_candidate_prefilter'] = 'pose_layout_global_min_distance_8_to_4'
    results['algorithm_params']['ablation_step4_work_ellipse'] = 'enabled'
    results['algorithm_params']['measurement_pose_refine'] = measurement_pose_refine_debug
    results['algorithm_params']['layout_constrained_pnp'] = layout_constrained_pnp_debug
    results['algorithm_params']['physical_edge_circle_correction'] = physical_edge_circle_debug
    results['algorithm_params']['joint_layout_edge_correction'] = joint_layout_edge_debug
    results['algorithm_params']['final_edge_layout_correction'] = final_edge_layout_debug
    results['fast_normal_offset_edge_debug'] = arc_global_debug
    results['step4_work_ellipse_ablation'] = step4_work_ellipse_results

    finalize_single_image_outputs(
        inputs_ctx,
        light_img_path,
        save_dir,
        show,
        roi_points,
        measurement_ctx,
        eval_ctx,
        results,
        final_measurement_vis=final_measurement_vis,
    )
    return results, summary


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        gc.collect()
    sys.exit(0)

# python pagc_experiment_baselines_v17_with_pose_eval_inputs.py ^
#  --pose_source light_only ^
#  --frontend_mode randomized_physical ^
#  --final_measurement_vis


# ============================================================
# Geometry-guided boundary selection + subpixel edge ellipse center
# Final frontend extension: geometry is used at boundary-selection stage;
# the final center is fitted from localized subpixel edge points.
# ============================================================

GEOMETRY_BOUNDARY_SUBPIXEL_CONFIG = {
    "min_edge_points": 12,
    "min_edge_fit_success_ratio": 0.18,
    "min_arc_coverage": 0.16,
    "trim_keep_ratio": 0.78,
    "max_fit_residual_median": 0.45,
    "max_center_shift_vs_boundary_px": 8.0,
    "prefer_subpixel_source": True,
}


def _ggbs_center_or_nan(value):
    try:
        p = np.asarray(value, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(p)):
            return p.astype(np.float64)
    except Exception:
        pass
    return np.array([np.nan, np.nan], dtype=np.float64)


def _ggbs_fit_ellipse_from_points(points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 6:
        return None
    try:
        return cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return None


def _ggbs_ellipse_residuals(ellipse, points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if ellipse is None or pts.shape[0] == 0:
        return np.full((pts.shape[0],), np.inf, dtype=np.float64)
    (cx, cy), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, np.finfo(np.float64).eps)
    b = max(float(ma) * 0.5, np.finfo(np.float64).eps)
    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    return np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0).astype(np.float64)


def fit_subpixel_edge_ellipse_center(edge_set, boundary_center=None, boundary_ellipse=None, config=None):
    """Fit final center from localized subpixel edge points.

    This function is intentionally not a backend correction. It only consumes
    the already geometry-selected frontend boundary's subpixel edge points and
    robustly fits an image-plane ellipse. The returned center is the proposed
    frontend center when enough reliable subpixel edge support is available.
    """
    cfg = dict(GEOMETRY_BOUNDARY_SUBPIXEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)

    boundary_center = _ggbs_center_or_nan(boundary_center)
    pts = np.asarray((edge_set or {}).get("points", []), dtype=np.float64)
    if pts.size == 0:
        return {
            "center": boundary_center,
            "ellipse": None,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": "no_subpixel_edge_points",
            "debug": {
                "edge_point_count": 0,
                "edge_fit_success_ratio": 0.0,
                "arc_coverage": 0.0,
            },
        }
    pts = pts.reshape(-1, 2)
    finite_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_mask]

    dbg0 = (edge_set or {}).get("debug", {}) if isinstance((edge_set or {}).get("debug", {}), dict) else {}
    edge_count = int(pts.shape[0])
    edge_fit_ratio = float(dbg0.get("edge_fit_success_ratio", 0.0) or 0.0)
    arc_cov = float(dbg0.get("effective_arc_coverage", 0.0) or 0.0)

    if edge_count < int(cfg["min_edge_points"]):
        return {
            "center": boundary_center,
            "ellipse": None,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": "insufficient_subpixel_edge_points",
            "debug": {
                "edge_point_count": edge_count,
                "edge_fit_success_ratio": edge_fit_ratio,
                "arc_coverage": arc_cov,
            },
        }
    if edge_fit_ratio < float(cfg["min_edge_fit_success_ratio"]):
        return {
            "center": boundary_center,
            "ellipse": None,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": "low_subpixel_fit_success_ratio",
            "debug": {
                "edge_point_count": edge_count,
                "edge_fit_success_ratio": edge_fit_ratio,
                "arc_coverage": arc_cov,
            },
        }
    if arc_cov < float(cfg["min_arc_coverage"]):
        return {
            "center": boundary_center,
            "ellipse": None,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": "insufficient_subpixel_arc_coverage",
            "debug": {
                "edge_point_count": edge_count,
                "edge_fit_success_ratio": edge_fit_ratio,
                "arc_coverage": arc_cov,
            },
        }

    e0 = _ggbs_fit_ellipse_from_points(pts)
    if e0 is None:
        return {
            "center": boundary_center,
            "ellipse": None,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": "subpixel_ellipse_fit_failed_initial",
            "debug": {
                "edge_point_count": edge_count,
                "edge_fit_success_ratio": edge_fit_ratio,
                "arc_coverage": arc_cov,
            },
        }

    weights = np.asarray((edge_set or {}).get("weights", []), dtype=np.float64).reshape(-1)
    if weights.size != edge_count:
        weights = np.ones((edge_count,), dtype=np.float64)
    else:
        weights = weights[finite_mask] if finite_mask.size == weights.size else weights
        if weights.size != edge_count:
            weights = np.ones((edge_count,), dtype=np.float64)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / (np.median(weights) + 1e-12)

    resid0 = _ggbs_ellipse_residuals(e0, pts)
    score = resid0 / np.sqrt(np.clip(weights, 1e-9, None))
    keep_ratio = float(np.clip(cfg.get("trim_keep_ratio", 0.78), 0.50, 1.00))
    keep_n = int(np.clip(np.ceil(keep_ratio * edge_count), 6, edge_count))
    keep_idx = np.argsort(score)[:keep_n]
    pts_keep = pts[keep_idx]
    e1 = _ggbs_fit_ellipse_from_points(pts_keep)
    if e1 is None:
        e1 = e0
        pts_keep = pts

    resid1 = _ggbs_ellipse_residuals(e1, pts_keep)
    residual_median = float(np.median(resid1)) if resid1.size else np.inf
    residual_p90 = float(np.percentile(resid1, 90.0)) if resid1.size else np.inf
    center = _ggbs_center_or_nan(e1[0])
    boundary_shift = float(np.linalg.norm(center - boundary_center)) if np.all(np.isfinite(center)) and np.all(np.isfinite(boundary_center)) else np.inf

    reject_reason = None
    if not np.all(np.isfinite(center)):
        reject_reason = "nonfinite_subpixel_ellipse_center"
    elif residual_median > float(cfg["max_fit_residual_median"]):
        reject_reason = "subpixel_ellipse_residual_too_large"
    elif boundary_shift > float(cfg["max_center_shift_vs_boundary_px"]):
        reject_reason = "subpixel_center_shift_too_large"

    debug = {
        "edge_point_count": edge_count,
        "edge_fit_success_ratio": edge_fit_ratio,
        "arc_coverage": arc_cov,
        "trim_keep_ratio": keep_ratio,
        "fit_used_points": int(len(pts_keep)),
        "residual_median": residual_median,
        "residual_p90": residual_p90,
        "center_shift_vs_boundary_px": boundary_shift,
        "initial_residual_median": float(np.median(resid0)) if resid0.size else None,
        "edge_debug": dbg0,
    }

    if reject_reason is not None:
        return {
            "center": boundary_center,
            "ellipse": e1,
            "available": False,
            "final_source": "geometry_selected_boundary_ellipse",
            "fallback_reason": reject_reason,
            "debug": debug,
        }

    return {
        "center": center.astype(np.float64),
        "ellipse": e1,
        "available": True,
        "final_source": "subpixel_edge_ellipse",
        "fallback_reason": None,
        "debug": debug,
    }


def geometry_guided_boundary_subpixel_center_from_selected(
    gray_light_proc,
    selected_item,
    K=None,
    pose_ctx=None,
    proj_ctx=None,
    config=None,
):
    """Geometry-selected boundary -> subpixel edge localization -> ellipse center.

    The boundary has already been selected by
    build_selected_measurements_for_pipeline_joint_geometry(), which scores
    candidate contours against current light-only pose, projected circle shape,
    3D circle radius, and four-light layout consistency. This function then
    performs subpixel edge localization only on that selected boundary and makes
    the subpixel edge fitted ellipse center the final frontend output.
    """
    if pose_ctx is not None and proj_ctx is None:
        runtime_cache = pose_ctx.get("runtime_cache") if isinstance(pose_ctx, dict) else None
        if not isinstance(runtime_cache, dict) and K is not None:
            try:
                runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
            except Exception:
                runtime_cache = None
        if isinstance(runtime_cache, dict):
            proj_ctx = runtime_cache.get("projection_ctx")

    measurement = build_candidate_measurement_from_preselected(selected_item, gray_light_proc)
    edge_set = estimate_subpixel_edge_set(gray_light_proc, measurement, proj_ctx=proj_ctx)

    rep_dbg = measurement.get("point_debug_base", {}).get("representation", {}) if isinstance(measurement.get("point_debug_base", {}), dict) else {}
    boundary_center = _ggbs_center_or_nan(measurement.get("ellipse_center"))
    if not np.all(np.isfinite(boundary_center)):
        try:
            boundary_center = _ggbs_center_or_nan(measurement.get("work_ellipse", [None])[0])
        except Exception:
            boundary_center = np.array([np.nan, np.nan], dtype=np.float64)
    if not np.all(np.isfinite(boundary_center)):
        boundary_center = _ggbs_center_or_nan(selected_item.get("bbox_center"))

    fit = fit_subpixel_edge_ellipse_center(
        edge_set=edge_set,
        boundary_center=boundary_center,
        boundary_ellipse=measurement.get("work_ellipse"),
        config=config,
    )
    final_center = _ggbs_center_or_nan(fit.get("center"))
    final_source = str(fit.get("final_source", "unknown"))
    if not np.all(np.isfinite(final_center)):
        final_center = boundary_center.copy()
        final_source = "selected_boundary_center_last_resort"

    debug = {
        "algorithm_stage": "geometry_guided_boundary_selection_then_subpixel_edge_ellipse_fit",
        "boundary_selection_used": True,
        "boundary_selection_reason": rep_dbg.get("reason"),
        "assigned_layout_index": rep_dbg.get("assigned_layout_index"),
        "assigned_pose_center_image": rep_dbg.get("assigned_pose_center_image"),
        "assigned_pose_ellipse": rep_dbg.get("assigned_pose_ellipse"),
        "joint_unary_score": rep_dbg.get("joint_unary_score"),
        "joint_total_score": rep_dbg.get("joint_total_score"),
        "center_img_err_to_pose_prior": rep_dbg.get("center_img_err"),
        "center_3d_err_to_pose_prior": rep_dbg.get("center_3d_err"),
        "radius_err_to_physical_prior": rep_dbg.get("radius_err"),
        "candidate_ellipse_resid": rep_dbg.get("ellipse_resid"),
        "effective_arc_coverage_boundary": rep_dbg.get("effective_arc_coverage"),
        "arc_points_count_boundary": rep_dbg.get("arc_points_count"),
        "subpixel_fit": fit.get("debug", {}),
        "final_source": final_source,
        "fallback_reason": fit.get("fallback_reason"),
    }

    return {
        "center": final_center.astype(np.float64),
        "geometry_selected_boundary_center": boundary_center.astype(np.float64),
        "subpixel_edge_fitted_center": _ggbs_center_or_nan(fit.get("center")) if fit.get("available") else np.array([np.nan, np.nan], dtype=np.float64),
        "subpixel_edge_fitted_ellipse": fit.get("ellipse"),
        "boundary_ellipse": measurement.get("work_ellipse"),
        "boundary_contour": measurement.get("work_contour"),
        "edge_set": edge_set,
        "measurement": measurement,
        "debug": debug,
    }


# ============================================================
# Shell-constrained inward subpixel frontend
# New revision: geometry is not a post-hoc center correction. It selects a
# halo shell whose inverse projection is a common-scale concentric shell on the
# lamp plane, then shrinks that shell to the known physical radius and performs
# one-sided/inward subpixel edge localization around the predicted true boundary.
# ============================================================

SHELL_INWARD_CONFIG = {
    "candidate_topk_per_light": 5,
    "max_joint_combinations": 625,
    "min_candidate_points": 18,
    "max_candidate_points_for_scoring": 96,
    "num_inner_arc_points": 144,
    "min_shell_scale_k": 0.85,
    "preferred_shell_scale_k": 1.08,
    "max_shell_scale_k": 6.0,
    "max_axis_ratio_log_error": 0.45,
    "joint_circularity_weight": 0.25,
    "joint_under_shell_weight": 0.35,
    "joint_axis_ratio_weight": 0.08,
    "enable_pointwise_boundary_pool": False,
    "point_pool_topk_shells": 4,
    "point_pool_samples_per_shell": 96,
    "point_pool_max_points": 144,
    "point_pool_min_points": 28,
    "point_pool_num_angle_bins": 36,
    "point_pool_gradient_weight": 0.85,
    "point_pool_source_weight": 0.55,
    "point_pool_expected_ellipse_weight": 0.20,
    "subpixel_config": {
        "min_edge_points": 16,
        "min_edge_fit_success_ratio": 0.18,
        "min_arc_coverage": 0.18,
        "trim_keep_ratio": 0.78,
        "max_fit_residual_median": 0.42,
        "max_center_shift_vs_boundary_px": 9.0,
    },
}


def _shell_as_cv_contour(points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _shell_project_cam_to_image(points_cam, K):
    X = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    valid = np.isfinite(X).all(axis=1) & (np.abs(X[:, 2]) > 1e-9)
    out = np.full((len(X), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = (K @ X[valid].T).T
        out[valid, 0] = x[:, 0] / x[:, 2]
        out[valid, 1] = x[:, 1] / x[:, 2]
    return out


def _shell_expected_axis_ratio(expected_ellipse):
    if expected_ellipse is None:
        return None
    try:
        (_, _), (MA, ma), _ = expected_ellipse
        MA = float(MA); ma = float(ma)
        if not np.isfinite(MA) or not np.isfinite(ma) or MA <= 1e-6 or ma <= 1e-6:
            return None
        return float(min(MA, ma) / max(MA, ma))
    except Exception:
        return None


def _shell_ellipse_axis_ratio(ellipse):
    if ellipse is None:
        return None
    try:
        (_, _), (MA, ma), _ = ellipse
        MA = float(MA); ma = float(ma)
        if not np.isfinite(MA) or not np.isfinite(ma) or MA <= 1e-6 or ma <= 1e-6:
            return None
        return float(min(MA, ma) / max(MA, ma))
    except Exception:
        return None


def _shell_sample_contour_points(contour, max_points=96):
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return pts
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int32)
    return pts[idx]


def _shell_project_points_to_plane(points_uv, proj_ctx):
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    out = []
    valid_idx = []
    for idx, uv in enumerate(pts):
        X = _project_single_pixel_to_plane(uv, proj_ctx)
        if X is not None and np.all(np.isfinite(X)):
            out.append(X)
            valid_idx.append(idx)
    if not out:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3), np.asarray(valid_idx, dtype=np.int32)


def generate_shell_boundary_candidates(gray, contour, ellipse, max_candidates=12):
    """Generate multi-level halo-shell boundary candidates.

    The function deliberately does not decide which boundary is correct. It
    returns plausible elliptical shells from intensity levels inside the detected
    lamp/halo region plus the original working contour as a candidate.
    """
    candidates = []
    if gray is None or contour is None:
        return candidates
    contour_np = np.asarray(contour)
    if contour_np.size == 0:
        return candidates
    if ellipse is None:
        try:
            if len(contour_np.reshape(-1, 2)) >= 6:
                ellipse = cv2.fitEllipse(contour_np.astype(np.float32).reshape(-1, 1, 2))
        except Exception:
            ellipse = None
    if ellipse is None:
        return candidates

    # Original shell candidate.
    try:
        e0 = cv2.fitEllipse(contour_np.astype(np.float32).reshape(-1, 1, 2)) if len(contour_np.reshape(-1, 2)) >= 6 else ellipse
    except Exception:
        e0 = ellipse
    candidates.append({
        "source": "original_work_contour",
        "level_index": -1,
        "threshold": None,
        "contour": contour_np.reshape(-1, 1, 2).astype(np.int32),
        "ellipse": e0,
        "area": float(cv2.contourArea(contour_np.reshape(-1, 1, 2).astype(np.int32))) if contour_np.size else 0.0,
    })

    try:
        roi_gray, (ox, oy), _ = crop_local(gray, ellipse, pad=max(ROI_PAD, 18))
    except Exception:
        return candidates[:max_candidates]
    if roi_gray is None or roi_gray.size == 0:
        return candidates[:max_candidates]

    contour_local = contour_np.reshape(-1, 2).astype(np.int32).copy()
    contour_local[:, 0] -= int(ox)
    contour_local[:, 1] -= int(oy)
    contour_local[:, 0] = np.clip(contour_local[:, 0], 0, roi_gray.shape[1] - 1)
    contour_local[:, 1] = np.clip(contour_local[:, 1], 0, roi_gray.shape[0] - 1)
    mask = np.zeros(roi_gray.shape[:2], dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [contour_local.reshape(-1, 1, 2)], 255)
    except Exception:
        return candidates[:max_candidates]
    vals = roi_gray[mask > 0]
    if vals.size < 32:
        return candidates[:max_candidates]

    lo = float(np.percentile(vals, 38.0))
    hi = float(np.percentile(vals, 98.0))
    if hi - lo < 4.0:
        return candidates[:max_candidates]
    # Low threshold gives outer shell; high threshold gives inner saturated core.
    level_fracs = [0.08, 0.14, 0.22, 0.32, 0.44, 0.58, 0.72, 0.86]
    raw_area = max(1.0, float(cv2.contourArea(contour_np.reshape(-1, 1, 2).astype(np.int32))))
    ksz = int(np.clip(round(np.sqrt(raw_area) / 45.0), 3, 7))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    base_center = np.asarray(ellipse[0], dtype=np.float64).reshape(2)

    for level_idx, frac in enumerate(level_fracs):
        thr = float(lo + frac * (hi - lo))
        bw = np.zeros_like(roi_gray, dtype=np.uint8)
        bw[(roi_gray >= thr) & (mask > 0)] = 255
        if np.count_nonzero(bw) < 20:
            continue
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        best = None
        best_rank = -np.inf
        for cc in cs:
            if len(cc) < 6:
                continue
            area = float(cv2.contourArea(cc))
            if area < 8.0:
                continue
            try:
                e = cv2.fitEllipse(cc.astype(np.float32).reshape(-1, 1, 2))
            except Exception:
                continue
            ctr_img = np.asarray([e[0][0] + ox, e[0][1] + oy], dtype=np.float64)
            drift = float(np.linalg.norm(ctr_img - base_center))
            # Prefer the component around the original shell center.
            rank = area / raw_area - 0.04 * drift
            if rank > best_rank:
                best_rank = rank
                best = (cc, e, area)
        if best is None:
            continue
        cc, e, area = best
        cc_img = cc.reshape(-1, 2).astype(np.float64) + np.array([ox, oy], dtype=np.float64)
        e_img = ((float(e[0][0] + ox), float(e[0][1] + oy)), e[1], e[2])
        candidates.append({
            "source": "intensity_level_shell",
            "level_index": int(level_idx),
            "threshold": float(thr),
            "contour": _shell_as_cv_contour(cc_img),
            "ellipse": e_img,
            "area": float(area),
            "level_fraction": float(frac),
        })

    # Deduplicate candidates with nearly identical centers/axes.
    unique = []
    seen = []
    for c in candidates:
        e = c.get("ellipse")
        if e is None:
            continue
        try:
            feat = np.array([e[0][0], e[0][1], e[1][0], e[1][1]], dtype=np.float64)
        except Exception:
            continue
        duplicate = False
        for f in seen:
            if np.linalg.norm((feat - f) / np.maximum(np.abs(f), 1.0)) < 0.015:
                duplicate = True
                break
        if not duplicate:
            unique.append(c)
            seen.append(feat)
    return unique[:max_candidates]


def score_shell_candidate_by_plane_scale(candidate, proj_ctx, center_cam, expected_ellipse=None, lamp_radius_mm=None, config=None):
    cfg = dict(SHELL_INWARD_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    R_lamp = float(lamp_radius_mm if lamp_radius_mm is not None else LIGHT_RADIUS_MM)
    ctr_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
    if candidate is None or candidate.get("contour") is None or not np.all(np.isfinite(ctr_cam)):
        return None
    contour = np.asarray(candidate.get("contour"), dtype=np.float64).reshape(-1, 2)
    if len(contour) < int(cfg["min_candidate_points"]):
        return None
    pts = _shell_sample_contour_points(contour, int(cfg["max_candidate_points_for_scoring"]))
    X, valid_idx = _shell_project_points_to_plane(pts, proj_ctx)
    if len(X) < max(8, int(0.35 * min(len(pts), cfg["max_candidate_points_for_scoring"]))):
        return None
    radii = np.linalg.norm(X - ctr_cam.reshape(1, 3), axis=1)
    finite = np.isfinite(radii) & (radii > 1e-6)
    radii = radii[finite]
    if radii.size < 8:
        return None
    k_vals = radii / max(R_lamp, 1e-9)
    k_med = float(np.median(k_vals))
    mad = float(np.median(np.abs(k_vals - k_med)))
    circularity = float(mad / max(abs(k_med), 1e-6))

    axis_ratio = _shell_ellipse_axis_ratio(candidate.get("ellipse"))
    exp_ratio = _shell_expected_axis_ratio(expected_ellipse)
    if axis_ratio is None or exp_ratio is None:
        axis_ratio_log_error = 0.0
        axis_ratio_gate_pass = True
    else:
        axis_ratio_log_error = float(abs(np.log(max(axis_ratio, 1e-9) / max(exp_ratio, 1e-9))))
        axis_ratio_gate_pass = axis_ratio_log_error <= float(cfg["max_axis_ratio_log_error"])

    under_shell_penalty = float(max(0.0, float(cfg["preferred_shell_scale_k"]) - k_med) / max(float(cfg["preferred_shell_scale_k"]), 1e-6))
    invalid_scale = bool(k_med < float(cfg["min_shell_scale_k"]) or k_med > float(cfg["max_shell_scale_k"]))
    valid = bool((not invalid_scale) and axis_ratio_gate_pass)

    out = dict(candidate)
    out.update({
        "shell_valid": valid,
        "shell_scale_k": k_med,
        "shell_scale_mad": mad,
        "shell_circularity": circularity,
        "shell_radius_mm_median": float(np.median(radii)),
        "shell_radius_mm_mad": float(np.median(np.abs(radii - np.median(radii)))),
        "axis_ratio": axis_ratio,
        "expected_axis_ratio": exp_ratio,
        "axis_ratio_log_error": axis_ratio_log_error,
        "axis_ratio_gate_pass": bool(axis_ratio_gate_pass),
        "under_shell_penalty": under_shell_penalty,
        "invalid_scale": invalid_scale,
        "score_local": float(circularity + cfg["joint_under_shell_weight"] * under_shell_penalty + cfg["joint_axis_ratio_weight"] * axis_ratio_log_error),
    })
    return out


def _select_layout_id_from_measurement(measurement, fallback_idx=0):
    try:
        rep = measurement.get("point_debug_base", {}).get("representation", {})
        v = rep.get("assigned_layout_index", None)
        if v is not None:
            return int(v)
    except Exception:
        pass
    return int(fallback_idx)


def _make_shell_measurement_from_inner_boundary(base_measurement, shell_candidate, inner_points_img, inner_ellipse, layout_idx, center_cam, shell_debug):
    meas = dict(base_measurement) if isinstance(base_measurement, dict) else {}
    inner_pts = np.asarray(inner_points_img, dtype=np.float64).reshape(-1, 2)
    inner_pts = inner_pts[np.all(np.isfinite(inner_pts), axis=1)]
    if inner_ellipse is None and len(inner_pts) >= 6:
        inner_ellipse = _ggbs_fit_ellipse_from_points(inner_pts)
    if inner_ellipse is None:
        inner_ellipse = shell_candidate.get("ellipse")
    shell_contour = shell_candidate.get("contour")
    inner_contour = _shell_as_cv_contour(inner_pts)
    meas["work_ellipse"] = inner_ellipse
    meas["ellipse_center"] = np.asarray(inner_ellipse[0], dtype=np.float64) if inner_ellipse is not None else _ggbs_center_or_nan(meas.get("ellipse_center"))
    # Keep shell contour for photometric baselines, but use inner contour for edge sampling.
    meas["work_contour"] = inner_contour if len(inner_contour) >= 6 else shell_contour
    meas["contour"] = shell_contour if shell_contour is not None else meas.get("contour")
    try:
        angles = _ellipse_param_angles(inner_ellipse, inner_pts) if inner_ellipse is not None and len(inner_pts) else np.zeros((0,), dtype=np.float64)
    except Exception:
        angles = np.zeros((len(inner_pts),), dtype=np.float64)
    meas.setdefault("point_debug_base", {})
    if not isinstance(meas["point_debug_base"], dict):
        meas["point_debug_base"] = {}
    meas["point_debug_base"]["fast_normal_offset_edge"] = {
        "arc_points": inner_pts.tolist(),
        "arc_angles": np.asarray(angles, dtype=np.float64).reshape(-1).tolist(),
        "arc_weights": np.ones((len(inner_pts),), dtype=np.float64).tolist(),
        "edge_source": "shell_shrunk_physical_radius_prediction",
    }
    rep_old = meas["point_debug_base"].get("representation", {})
    if not isinstance(rep_old, dict):
        rep_old = {}
    rep = dict(rep_old)
    rep.update({
        "reason": "shell_scale_consistent_boundary_shrunk_to_physical_radius",
        "assigned_layout_index": int(layout_idx),
        "assigned_pose_center_camera": np.asarray(center_cam, dtype=np.float64).reshape(3).tolist(),
        "shell_candidate_source": shell_candidate.get("source"),
        "shell_level_index": shell_candidate.get("level_index"),
        "shell_scale_k": shell_debug.get("shell_scale_k"),
        "shell_scale_mad": shell_debug.get("shell_scale_mad"),
        "shell_circularity": shell_debug.get("shell_circularity"),
        "shell_joint_score": shell_debug.get("joint_score"),
        "shell_axis_ratio_log_error": shell_debug.get("axis_ratio_log_error"),
        "confidence_light": max(float(rep_old.get("confidence_light", 0.0) or 0.0), 0.65),
    })
    meas["point_debug_base"]["representation"] = rep
    return meas


def _shrink_shell_candidate_to_physical_boundary(shell_candidate, proj_ctx, center_cam, K, lamp_radius_mm=None, num_points=None):
    R_lamp = float(lamp_radius_mm if lamp_radius_mm is not None else LIGHT_RADIUS_MM)
    n = int(num_points if num_points is not None else SHELL_INWARD_CONFIG["num_inner_arc_points"])
    ellipse = shell_candidate.get("ellipse")
    if ellipse is not None:
        shell_pts_img = sample_ellipse_points(ellipse, num_samples=n)
    else:
        contour = np.asarray(shell_candidate.get("contour"), dtype=np.float64).reshape(-1, 2)
        shell_pts_img = _shell_sample_contour_points(contour, n)
    X_shell, valid_idx = _shell_project_points_to_plane(shell_pts_img, proj_ctx)
    ctr = np.asarray(center_cam, dtype=np.float64).reshape(3)
    if len(X_shell) < 6:
        return None, None, {"used": False, "reason": "too_few_shell_points_projected"}
    V = X_shell - ctr.reshape(1, 3)
    nr = np.linalg.norm(V, axis=1)
    valid = np.isfinite(nr) & (nr > 1e-9)
    if np.count_nonzero(valid) < 6:
        return None, None, {"used": False, "reason": "invalid_shell_vectors"}
    V = V[valid]
    nr = nr[valid]
    X_inner = ctr.reshape(1, 3) + R_lamp * (V / nr.reshape(-1, 1))
    inner_img = _shell_project_cam_to_image(X_inner, K)
    inner_img = inner_img[np.all(np.isfinite(inner_img), axis=1)]
    if len(inner_img) < 6:
        return None, None, {"used": False, "reason": "inner_projection_failed"}
    inner_ellipse = _ggbs_fit_ellipse_from_points(inner_img)
    return inner_img, inner_ellipse, {
        "used": True,
        "reason": "shell_plane_inward_shrink_to_known_physical_radius",
        "shell_radius_mm_median": float(np.median(nr)),
        "shell_scale_k": float(np.median(nr) / max(R_lamp, 1e-9)),
        "inward_shrink_mm_median": float(max(0.0, np.median(nr) - R_lamp)),
        "inner_point_count": int(len(inner_img)),
    }


def _pointwise_select_physical_boundary_from_shells(
    gray,
    scored_shells,
    anchor_shell,
    proj_ctx,
    center_cam,
    K,
    expected_ellipse=None,
    lamp_radius_mm=None,
    config=None,
    joint_debug=None,
):
    """Build a physical-boundary point set from high-quality points of many shells.

    This replaces the strict "select one whole shell" behavior with a point-wise
    selection: each candidate shell contributes inward-projected physical-boundary
    points, and only points with good local image support and geometric consistency
    are kept. The returned point set is still a frontend boundary prior; subpixel
    edge localization is performed afterwards.
    """
    cfg = dict(SHELL_INWARD_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if not bool(cfg.get("enable_pointwise_boundary_pool", True)):
        return None, None, None, {"used": False, "reason": "disabled"}
    shells = [c for c in (scored_shells or []) if c is not None]
    if anchor_shell is not None and all(id(anchor_shell) != id(c) for c in shells):
        shells.append(anchor_shell)
    if not shells:
        return None, None, None, {"used": False, "reason": "no_shell_candidates"}

    shells.sort(key=lambda c: float(c.get("score_local", np.inf)))
    topk = int(max(1, cfg.get("point_pool_topk_shells", 4)))
    shells = shells[:topk]

    gray_f = np.asarray(gray, dtype=np.float64)
    if gray_f.ndim == 3:
        gray_f = cv2.cvtColor(gray_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    blur = cv2.GaussianBlur(gray_f.astype(np.float32), (3, 3), 0.0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    nz = gmag[np.isfinite(gmag) & (gmag > 0)]
    grad_norm = float(np.percentile(nz, 90.0)) if nz.size else 1.0
    grad_norm = max(grad_norm, 1e-6)

    joint_k = None
    if isinstance(joint_debug, dict):
        try:
            joint_k = float(joint_debug.get("shell_scale_median"))
        except Exception:
            joint_k = None
    point_rows = []
    shrink_debugs = []
    samples_per_shell = int(max(24, cfg.get("point_pool_samples_per_shell", 96)))
    w_grad = float(cfg.get("point_pool_gradient_weight", 0.85))
    w_src = float(cfg.get("point_pool_source_weight", 0.55))
    w_exp = float(cfg.get("point_pool_expected_ellipse_weight", 0.20))

    for shell_rank, shell in enumerate(shells):
        inner_pts, _, shrink_dbg = _shrink_shell_candidate_to_physical_boundary(
            shell,
            proj_ctx=proj_ctx,
            center_cam=center_cam,
            K=K,
            lamp_radius_mm=lamp_radius_mm,
            num_points=samples_per_shell,
        )
        shrink_debugs.append(shrink_dbg)
        if inner_pts is None:
            continue
        pts = np.asarray(inner_pts, dtype=np.float64).reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 6:
            continue

        source_penalty = float(shell.get("score_local", 0.0) or 0.0)
        source_penalty += 0.15 * float(max(shell_rank, 0))
        if joint_k is not None and np.isfinite(joint_k):
            k = float(shell.get("shell_scale_k", np.nan))
            if np.isfinite(k) and k > 1e-9 and joint_k > 1e-9:
                source_penalty += abs(math.log(k / joint_k))
        exp_res = _ggbs_ellipse_residuals(expected_ellipse, pts) if expected_ellipse is not None else np.zeros((len(pts),), dtype=np.float64)
        exp_res = np.where(np.isfinite(exp_res), exp_res, 2.0)

        for pi, pt in enumerate(pts):
            gv = _bilinear_at(gmag, float(pt[0]), float(pt[1]))
            if gv is None or not np.isfinite(gv):
                continue
            grad_score = float(np.clip(gv / grad_norm, 0.0, 2.5))
            score = w_grad * grad_score - w_src * source_penalty - w_exp * float(exp_res[pi])
            point_rows.append({
                "point": pt.astype(np.float64),
                "score": float(score),
                "gradient_score": grad_score,
                "source_penalty": source_penalty,
                "expected_residual": float(exp_res[pi]),
                "shell_rank": int(shell_rank),
                "shell_source": shell.get("source"),
            })

    if len(point_rows) < int(cfg.get("point_pool_min_points", 28)):
        return None, None, None, {
            "used": False,
            "reason": "too_few_scored_points",
            "num_scored_points": int(len(point_rows)),
            "shrink_debugs": shrink_debugs,
        }

    # Angle-balanced non-maximum selection prevents one strong arc from
    # dominating the fitted boundary.
    all_pts = np.asarray([r["point"] for r in point_rows], dtype=np.float64).reshape(-1, 2)
    ref_ellipse = expected_ellipse if expected_ellipse is not None else _ggbs_fit_ellipse_from_points(all_pts)
    if ref_ellipse is not None:
        ctr = np.asarray(ref_ellipse[0], dtype=np.float64).reshape(2)
    else:
        ctr = np.nanmean(all_pts, axis=0)
    for r in point_rows:
        v = r["point"] - ctr
        r["angle"] = float(np.mod(np.arctan2(v[1], v[0]), 2.0 * np.pi))

    max_points = int(max(cfg.get("point_pool_min_points", 28), cfg.get("point_pool_max_points", 144)))
    num_bins = int(np.clip(cfg.get("point_pool_num_angle_bins", 36), 8, 72))
    per_bin_limit = int(max(2, math.ceil(max_points / max(num_bins, 1))))
    selected = []
    bin_counts = np.zeros((num_bins,), dtype=np.int32)
    used_ids = set()
    sorted_rows = sorted(enumerate(point_rows), key=lambda x: x[1]["score"], reverse=True)
    for idx, row in sorted_rows:
        b = int(np.floor(row["angle"] / (2.0 * np.pi) * num_bins)) % num_bins
        if bin_counts[b] >= per_bin_limit:
            continue
        selected.append(row)
        used_ids.add(idx)
        bin_counts[b] += 1
        if len(selected) >= max_points:
            break
    if len(selected) < int(cfg.get("point_pool_min_points", 28)):
        for idx, row in sorted_rows:
            if idx in used_ids:
                continue
            selected.append(row)
            if len(selected) >= int(cfg.get("point_pool_min_points", 28)):
                break

    selected_pts = np.asarray([r["point"] for r in selected], dtype=np.float64).reshape(-1, 2)
    inner_ellipse = _ggbs_fit_ellipse_from_points(selected_pts)
    if inner_ellipse is None:
        return None, None, None, {
            "used": False,
            "reason": "point_pool_ellipse_fit_failed",
            "num_selected_points": int(len(selected_pts)),
            "num_scored_points": int(len(point_rows)),
            "shrink_debugs": shrink_debugs,
        }

    pseudo_shell = dict(anchor_shell or shells[0])
    pseudo_shell.update({
        "source": "pointwise_multi_shell_boundary_pool",
        "level_index": -200,
        "contour": _shell_as_cv_contour(selected_pts),
        "ellipse": inner_ellipse,
        "shell_valid": True,
        "score_local": float(-np.mean([r["score"] for r in selected])),
        "point_pool_used": True,
        "point_pool_num_candidates": int(len(shells)),
        "point_pool_num_scored": int(len(point_rows)),
        "point_pool_num_selected": int(len(selected_pts)),
    })
    debug = {
        "used": True,
        "reason": "pointwise_multi_shell_boundary_selection",
        "num_candidate_shells": int(len(shells)),
        "num_scored_points": int(len(point_rows)),
        "num_selected_points": int(len(selected_pts)),
        "num_angle_bins": int(num_bins),
        "occupied_angle_bins": int(np.count_nonzero(bin_counts)),
        "score_mean": float(np.mean([r["score"] for r in selected])),
        "score_min": float(np.min([r["score"] for r in selected])),
        "score_max": float(np.max([r["score"] for r in selected])),
        "gradient_score_mean": float(np.mean([r["gradient_score"] for r in selected])),
        "source_penalty_mean": float(np.mean([r["source_penalty"] for r in selected])),
        "expected_residual_mean": float(np.mean([r["expected_residual"] for r in selected])),
        "selected_shell_sources": [r["shell_source"] for r in selected[:12]],
        "shrink_debugs": shrink_debugs,
    }
    return pseudo_shell, selected_pts, inner_ellipse, debug


def _joint_select_shell_candidates(scored_per_light, config=None):
    cfg = dict(SHELL_INWARD_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    topk = int(cfg["candidate_topk_per_light"])
    lists = []
    for cand_list in scored_per_light:
        valid = [c for c in cand_list if c is not None and c.get("shell_valid", False)]
        if not valid:
            valid = [c for c in cand_list if c is not None]
        valid.sort(key=lambda c: float(c.get("score_local", np.inf)))
        lists.append(valid[:max(1, topk)])
    if any(len(v) == 0 for v in lists):
        return None, {"used": False, "reason": "empty_candidate_list"}

    best_combo = None
    best_score = np.inf
    best_dbg = None
    combo_iter = itertools.product(*lists)
    max_combos = int(cfg.get("max_joint_combinations", 625))
    for combo_idx, combo in enumerate(combo_iter):
        if combo_idx >= max_combos:
            break
        ks = np.asarray([float(c.get("shell_scale_k", np.nan)) for c in combo], dtype=np.float64)
        circ = np.asarray([float(c.get("shell_circularity", np.nan)) for c in combo], dtype=np.float64)
        under = np.asarray([float(c.get("under_shell_penalty", 0.0)) for c in combo], dtype=np.float64)
        axis = np.asarray([float(c.get("axis_ratio_log_error", 0.0)) for c in combo], dtype=np.float64)
        if not np.all(np.isfinite(ks)):
            continue
        k_med = float(np.median(ks))
        k_std_norm = float(np.std(ks) / max(abs(k_med), 1e-6))
        score = (
            k_std_norm
            + float(cfg["joint_circularity_weight"]) * float(np.nanmean(circ))
            + float(cfg["joint_under_shell_weight"]) * float(np.nanmean(under))
            + float(cfg["joint_axis_ratio_weight"]) * float(np.nanmean(axis))
        )
        if score < best_score:
            best_score = float(score)
            best_combo = list(combo)
            best_dbg = {
                "used": True,
                "reason": "min_common_shell_scale_consistency",
                "joint_score": float(score),
                "shell_scale_values": ks.tolist(),
                "shell_scale_median": k_med,
                "shell_scale_std_norm": k_std_norm,
                "shell_circularity_mean": float(np.nanmean(circ)),
                "under_shell_penalty_mean": float(np.nanmean(under)),
                "axis_ratio_log_error_mean": float(np.nanmean(axis)),
            }
    if best_combo is None:
        return None, {"used": False, "reason": "no_finite_joint_score"}
    return best_combo, best_dbg


def shell_constrained_inward_subpixel_frontend(gray_light_proc, selected_items, K=None, pose_ctx=None, proj_ctx=None, config=None):
    """Run shell-constrained boundary selection for all four lights jointly.

    Returns one result dict per selected item. The checkerboard is not used.
    Geometry uses only the light-only pose, four-lamp layout, and known radius.
    """
    cfg = dict(SHELL_INWARD_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if pose_ctx is not None and proj_ctx is None:
        runtime_cache = pose_ctx.get("runtime_cache") if isinstance(pose_ctx, dict) else None
        if not isinstance(runtime_cache, dict) and K is not None:
            try:
                runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
            except Exception:
                runtime_cache = None
        if isinstance(runtime_cache, dict):
            proj_ctx = runtime_cache.get("projection_ctx")
    runtime_cache = pose_ctx.get("runtime_cache", {}) if isinstance(pose_ctx, dict) else {}
    expected_cam = np.asarray(runtime_cache.get("expected_cam", []), dtype=np.float64).reshape(-1, 3) if runtime_cache else np.zeros((0, 3), dtype=np.float64)
    expected_ellipses = runtime_cache.get("expected_ellipses", []) if isinstance(runtime_cache, dict) else []
    if proj_ctx is None or K is None or expected_cam.shape[0] < 1:
        # Last-resort compatibility: fall back to previous per-light function.
        return [geometry_guided_boundary_subpixel_center_from_selected(gray_light_proc, s, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg.get("subpixel_config")) for s in selected_items]

    base_measurements = []
    scored_per_light = []
    layout_ids = []
    for i, sel in enumerate(selected_items):
        meas = build_candidate_measurement_from_preselected(sel, gray_light_proc)
        base_measurements.append(meas)
        layout_idx = _select_layout_id_from_measurement(meas, fallback_idx=i)
        layout_idx = int(np.clip(layout_idx, 0, max(0, expected_cam.shape[0] - 1)))
        layout_ids.append(layout_idx)
        contour = meas.get("work_contour", meas.get("contour", sel.get("work_contour")))
        ellipse = meas.get("work_ellipse", None)
        if ellipse is None:
            try:
                ellipse = cv2.fitEllipse(np.asarray(contour).astype(np.float32).reshape(-1, 1, 2))
            except Exception:
                ellipse = None
        raw_candidates = generate_shell_boundary_candidates(gray_light_proc, contour, ellipse)
        exp_e = expected_ellipses[layout_idx] if layout_idx < len(expected_ellipses) else None
        scored = []
        for cand in raw_candidates:
            sc = score_shell_candidate_by_plane_scale(
                cand,
                proj_ctx=proj_ctx,
                center_cam=expected_cam[layout_idx],
                expected_ellipse=exp_e,
                lamp_radius_mm=LIGHT_RADIUS_MM,
                config=cfg,
            )
            if sc is not None:
                scored.append(sc)
        if not scored:
            # Fallback candidate from the original contour even if scoring failed.
            if ellipse is not None and contour is not None:
                scored.append({
                    "source": "fallback_unscored_original",
                    "level_index": -99,
                    "contour": np.asarray(contour).reshape(-1, 1, 2).astype(np.int32),
                    "ellipse": ellipse,
                    "shell_valid": False,
                    "shell_scale_k": np.nan,
                    "shell_circularity": np.inf,
                    "score_local": np.inf,
                })
        scored_per_light.append(scored)

    selected_shells, joint_dbg = _joint_select_shell_candidates(scored_per_light, config=cfg)
    if selected_shells is None:
        return [geometry_guided_boundary_subpixel_center_from_selected(gray_light_proc, s, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg.get("subpixel_config")) for s in selected_items]

    results = []
    for i, (sel, meas, shell) in enumerate(zip(selected_items, base_measurements, selected_shells)):
        layout_idx = layout_ids[i]
        shell_dbg = {k: shell.get(k) for k in [
            "source", "level_index", "threshold", "level_fraction", "shell_scale_k", "shell_scale_mad",
            "shell_circularity", "shell_radius_mm_median", "shell_radius_mm_mad", "axis_ratio",
            "expected_axis_ratio", "axis_ratio_log_error", "axis_ratio_gate_pass", "under_shell_penalty",
            "score_local", "shell_valid"
        ]}
        shell_dbg.update(joint_dbg)
        point_shell, point_pts, point_ellipse, point_dbg = _pointwise_select_physical_boundary_from_shells(
            gray_light_proc,
            scored_per_light[i],
            shell,
            proj_ctx=proj_ctx,
            center_cam=expected_cam[layout_idx],
            K=K,
            expected_ellipse=expected_ellipses[layout_idx] if layout_idx < len(expected_ellipses) else None,
            lamp_radius_mm=LIGHT_RADIUS_MM,
            config=cfg,
            joint_debug=joint_dbg,
        )
        if point_shell is not None and point_pts is not None and point_ellipse is not None:
            shell = point_shell
            inner_pts = point_pts
            inner_ellipse = point_ellipse
            shrink_dbg = {
                "used": True,
                "reason": "pointwise_boundary_pool_preselected_physical_points",
                "inner_point_count": int(len(inner_pts)),
                "pointwise_selection": point_dbg,
            }
            shell_dbg.update({
                "pointwise_boundary_pool": point_dbg,
                "source": shell.get("source"),
                "level_index": shell.get("level_index"),
                "score_local": shell.get("score_local"),
            })
        else:
            inner_pts, inner_ellipse, shrink_dbg = _shrink_shell_candidate_to_physical_boundary(
                shell,
                proj_ctx=proj_ctx,
                center_cam=expected_cam[layout_idx],
                K=K,
                lamp_radius_mm=LIGHT_RADIUS_MM,
                num_points=cfg.get("num_inner_arc_points"),
            )
            shell_dbg["pointwise_boundary_pool"] = point_dbg
        if inner_pts is None or inner_ellipse is None:
            # If shell shrink fails, fall back to old geoselect path for this lamp.
            res = geometry_guided_boundary_subpixel_center_from_selected(gray_light_proc, sel, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg.get("subpixel_config"))
            dbg = res.get("debug", {}) if isinstance(res.get("debug", {}), dict) else {}
            dbg["shell_inward"] = {"used": False, "fallback_reason": shrink_dbg.get("reason"), "shell_debug": shell_dbg}
            res["debug"] = dbg
            results.append(res)
            continue
        shell_measurement = _make_shell_measurement_from_inner_boundary(
            base_measurement=meas,
            shell_candidate=shell,
            inner_points_img=inner_pts,
            inner_ellipse=inner_ellipse,
            layout_idx=layout_idx,
            center_cam=expected_cam[layout_idx],
            shell_debug=shell_dbg,
        )
        edge_set = estimate_subpixel_edge_set(gray_light_proc, shell_measurement, proj_ctx=proj_ctx)
        fit = fit_subpixel_edge_ellipse_center(
            edge_set=edge_set,
            boundary_center=np.asarray(inner_ellipse[0], dtype=np.float64) if inner_ellipse is not None else None,
            boundary_ellipse=inner_ellipse,
            config=cfg.get("subpixel_config"),
        )
        final_center = _ggbs_center_or_nan(fit.get("center"))
        final_source = str(fit.get("final_source", "unknown"))
        if not np.all(np.isfinite(final_center)):
            final_center = _ggbs_center_or_nan(inner_ellipse[0]) if inner_ellipse is not None else _ggbs_center_or_nan(meas.get("ellipse_center"))
            final_source = "shell_inner_predicted_ellipse_last_resort"
        subpx_center = _ggbs_center_or_nan(fit.get("center")) if fit.get("available") else np.array([np.nan, np.nan], dtype=np.float64)
        shell_center = _ggbs_center_or_nan(shell.get("ellipse", ((np.nan, np.nan), (0, 0), 0))[0])
        inner_center = _ggbs_center_or_nan(inner_ellipse[0]) if inner_ellipse is not None else np.array([np.nan, np.nan], dtype=np.float64)
        edge_dbg = edge_set.get("debug", {}) if isinstance(edge_set, dict) and isinstance(edge_set.get("debug", {}), dict) else {}
        debug = {
            "algorithm_stage": "shell_consistent_boundary_selection_inward_subpixel_edge_fit",
            "boundary_selection_used": True,
            "geometry_principle": "common_scale_concentric_halo_shell_then_inward_shrink_to_known_radius",
            "assigned_layout_index": int(layout_idx),
            "final_source": final_source,
            "fallback_reason": fit.get("fallback_reason"),
            "shell_selection": shell_dbg,
            "shell_shrink": shrink_dbg,
            "subpixel_fit": fit.get("debug", {}),
            "edge_debug": edge_dbg,
        }
        results.append({
            "center": final_center.astype(np.float64),
            "shell_selected_boundary_center": shell_center.astype(np.float64),
            "shell_inner_predicted_center": inner_center.astype(np.float64),
            "subpixel_edge_fitted_center": subpx_center.astype(np.float64),
            "subpixel_edge_fitted_ellipse": fit.get("ellipse"),
            "boundary_ellipse": shell.get("ellipse"),
            "inner_boundary_ellipse": inner_ellipse,
            "boundary_contour": shell.get("contour"),
            "inner_boundary_points": np.asarray(inner_pts, dtype=np.float64).reshape(-1, 2),
            "edge_set": edge_set,
            "measurement": shell_measurement,
            "base_measurement": meas,
            "debug": debug,
        })
    return results

# ============================================================
# Projective shell model frontend
# New revision: geometry estimates the planar projection model H from halo-shell
# candidates and known four-lamp layout/radius. The final center is the
# projection of the physical lamp center through the estimated H, not the center
# of an image ellipse. Checkerboard/GT is never used here.
# ============================================================

PROJECTIVE_SHELL_MODEL_CONFIG = {
    "candidate_topk_per_light": 4,
    "max_joint_combinations": 256,
    "optimize_top_combinations": 18,
    "max_candidate_points_for_model": 36,
    "min_candidate_points_for_model": 18,
    "min_shell_scale_k": 1.01,
    "preferred_shell_scale_k": 1.12,
    "max_shell_scale_k": 3.20,
    "k_initial_bounds": [1.0, 3.2],
    "max_shell_model_residual_norm": 0.22,
    "max_edge_model_residual_norm": 0.18,
    "max_center_shift_from_selected_mean_px": 8.0,
    "max_center_shift_from_selected_px": 16.0,
    "enable_edge_refine": True,
    "num_physical_boundary_points": 144,
    "homography_regularization_weight": 0.025,
    "homography_translation_regularization_weight": 0.015,
    "homography_projective_regularization_weight": 25.0,
    "max_h_scale_shear": 0.14,
    "max_h_translation_mm": 28.0,
    "max_h_projective": 7e-4,
    "subpixel_config": {
        "min_edge_points": 24,
        "min_edge_fit_success_ratio": 0.25,
        "min_arc_coverage": 0.25,
        "trim_keep_ratio": 0.78,
        "max_fit_residual_median": 0.45,
        "max_center_shift_vs_boundary_px": 12.0,
    },
}


def _psm_homography_from_pose_ctx(pose_ctx, K):
    """Plane world coordinates (X,Y,1) -> image pixels using light-only pose."""
    if pose_ctx is None or K is None:
        return None
    runtime_cache = pose_ctx.get("runtime_cache", {}) if isinstance(pose_ctx, dict) else {}
    R = runtime_cache.get("R1", None)
    t = runtime_cache.get("t1", None)
    if R is None:
        R = pose_ctx.get("R1", None) if isinstance(pose_ctx, dict) else None
    if t is None:
        t = pose_ctx.get("t1", None) if isinstance(pose_ctx, dict) else None
    try:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t)) and np.all(np.isfinite(K))):
        return None
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    return H.astype(np.float64)


def _psm_project_xy(H, xy):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    homog = np.column_stack([xy, np.ones((len(xy),), dtype=np.float64)])
    uvh = (H @ homog.T).T
    out = np.full((len(xy), 2), np.nan, dtype=np.float64)
    ok = np.isfinite(uvh).all(axis=1) & (np.abs(uvh[:, 2]) > 1e-12)
    out[ok] = uvh[ok, :2] / uvh[ok, 2:3]
    return out


def _psm_backproject_uv(H, uv):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    try:
        Hinv = np.linalg.inv(H)
    except Exception:
        return np.full((len(uv), 2), np.nan, dtype=np.float64)
    homog = np.column_stack([uv, np.ones((len(uv),), dtype=np.float64)])
    xyh = (Hinv @ homog.T).T
    out = np.full((len(uv), 2), np.nan, dtype=np.float64)
    ok = np.isfinite(xyh).all(axis=1) & (np.abs(xyh[:, 2]) > 1e-12)
    out[ok] = xyh[ok, :2] / xyh[ok, 2:3]
    return out


def _psm_circle_points(center_xy, radius_mm, n=144):
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.column_stack([
        center_xy[0] + float(radius_mm) * np.cos(theta),
        center_xy[1] + float(radius_mm) * np.sin(theta),
    ])


def _psm_ellipse_from_projected_circle(H, center_xy, radius_mm, n=144):
    uv = _psm_project_xy(H, _psm_circle_points(center_xy, radius_mm, n=n))
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if len(uv) < 6:
        return None, uv
    try:
        e = cv2.fitEllipse(uv.astype(np.float32).reshape(-1, 1, 2))
        return e, uv
    except Exception:
        return None, uv


def _psm_plane_correction_from_params(params):
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size < 8:
        q = np.zeros((8,), dtype=np.float64)
        q[:p.size] = p
        p = q
    return np.array([
        [1.0 + p[0], p[1], p[2]],
        [p[3], 1.0 + p[4], p[5]],
        [p[6], p[7], 1.0],
    ], dtype=np.float64)


def _psm_candidate_points(candidate, max_points=36):
    if candidate is None:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.asarray(candidate.get("contour", []), dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        e = candidate.get("ellipse")
        if e is not None:
            try:
                (cx, cy), (a, b), ang = e
                theta = np.linspace(0, 2*np.pi, max_points, endpoint=False)
                aa = float(a) * 0.5
                bb = float(b) * 0.5
                ca = np.cos(np.deg2rad(float(ang)))
                sa = np.sin(np.deg2rad(float(ang)))
                x = aa * np.cos(theta)
                y = bb * np.sin(theta)
                pts = np.column_stack([float(cx) + ca*x - sa*y, float(cy) + sa*x + ca*y])
            except Exception:
                pts = np.zeros((0, 2), dtype=np.float64)
    if len(pts) <= max_points:
        return pts.astype(np.float64)
    idx = np.linspace(0, len(pts)-1, int(max_points), dtype=np.int32)
    return pts[idx].astype(np.float64)


def _psm_initial_score_candidate(candidate, H0, center_xy, radius_mm, config=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    pts = _psm_candidate_points(candidate, int(cfg["max_candidate_points_for_model"]))
    if len(pts) < int(cfg["min_candidate_points_for_model"]):
        return None
    xy = _psm_backproject_uv(H0, pts)
    ok = np.all(np.isfinite(xy), axis=1)
    if np.count_nonzero(ok) < int(cfg["min_candidate_points_for_model"]):
        return None
    xy = xy[ok]
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    rr = np.linalg.norm(xy - c.reshape(1, 2), axis=1)
    rr = rr[np.isfinite(rr)]
    if len(rr) < int(cfg["min_candidate_points_for_model"]):
        return None
    R = max(float(radius_mm), 1e-9)
    k = float(np.median(rr) / R)
    mad = float(np.median(np.abs(rr - np.median(rr))) / max(float(np.median(rr)), 1e-9))
    min_k = float(cfg["min_shell_scale_k"])
    max_k = float(cfg["max_shell_scale_k"])
    # Hardly-expanded shells are weak evidence; keep them but penalize strongly.
    under = float(max(0.0, min_k - k) / max(min_k, 1e-9))
    over = float(max(0.0, k - max_k) / max(max_k, 1e-9))
    preferred = float(cfg.get("preferred_shell_scale_k", 1.12))
    pref_pen = 0.04 * abs(np.log(max(k, 1e-9) / max(preferred, 1e-9)))
    score = mad + 1.5 * under + 2.0 * over + pref_pen
    out = dict(candidate)
    out.update({
        "psm_k0": k,
        "psm_circularity0": mad,
        "psm_under_penalty0": under,
        "psm_over_penalty0": over,
        "psm_score0": float(score),
        "psm_point_count": int(len(rr)),
    })
    return out


def _psm_select_top_candidates(gray, selected_items, H0, config=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    centers = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    per_light = []
    base_measurements = []
    layout_ids = []
    for i, sel in enumerate(selected_items):
        meas = build_candidate_measurement_from_preselected(sel, gray)
        base_measurements.append(meas)
        layout_idx = _select_layout_id_from_measurement(meas, fallback_idx=i)
        layout_idx = int(np.clip(layout_idx, 0, len(centers)-1))
        layout_ids.append(layout_idx)
        contour = meas.get("work_contour", meas.get("contour", sel.get("work_contour")))
        ellipse = meas.get("work_ellipse", None)
        if ellipse is None:
            try:
                ellipse = cv2.fitEllipse(np.asarray(contour).astype(np.float32).reshape(-1, 1, 2))
            except Exception:
                ellipse = None
        raw = generate_shell_boundary_candidates(gray, contour, ellipse, max_candidates=14)
        # Add a small family of same-axis shells so the projective model can test
        # whether a slightly larger or smaller halo shell better explains all four lamps.
        if ellipse is not None:
            try:
                (cx, cy), (aa, bb), ang = ellipse
                for sf in [0.92, 0.98, 1.04, 1.10, 1.18, 1.28]:
                    e = ((float(cx), float(cy)), (float(aa)*sf, float(bb)*sf), float(ang))
                    pts = _psm_ellipse_points(e, int(cfg.get("max_candidate_points_for_model", 36)) * 2)
                    raw.append({
                        "source": "scaled_ellipse_family_shell",
                        "level_index": int(round(sf * 1000)),
                        "level_fraction": float(sf),
                        "threshold": None,
                        "contour": _shell_as_cv_contour(pts),
                        "ellipse": e,
                        "area": float(np.pi * 0.25 * float(aa) * float(bb) * sf * sf),
                    })
            except Exception:
                pass
        scored = []
        for cand in raw:
            sc = _psm_initial_score_candidate(cand, H0, centers[layout_idx], LIGHT_RADIUS_MM, config=cfg)
            if sc is not None:
                scored.append(sc)
        if not scored and ellipse is not None:
            pts = _psm_ellipse_points(ellipse, 72)
            fallback = {
                "source": "fallback_ellipse_shell",
                "level_index": -99,
                "contour": _shell_as_cv_contour(pts),
                "ellipse": ellipse,
                "area": 0.0,
            }
            sc = _psm_initial_score_candidate(fallback, H0, centers[layout_idx], LIGHT_RADIUS_MM, config=cfg)
            if sc is not None:
                scored.append(sc)
        scored.sort(key=lambda c: float(c.get("psm_score0", np.inf)))
        per_light.append(scored[:max(1, int(cfg["candidate_topk_per_light"]))])
    return per_light, base_measurements, layout_ids


def _psm_ellipse_points(ellipse, n=96):
    try:
        (cx, cy), (aa, bb), ang = ellipse
        theta = np.linspace(0.0, 2.0*np.pi, int(n), endpoint=False)
        a = 0.5 * float(aa)
        b = 0.5 * float(bb)
        ca = np.cos(np.deg2rad(float(ang)))
        sa = np.sin(np.deg2rad(float(ang)))
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        return np.column_stack([float(cx) + ca*x - sa*y, float(cy) + sa*x + ca*y]).astype(np.float64)
    except Exception:
        return np.zeros((0, 2), dtype=np.float64)


def _psm_residuals_for_combo(params, H0, combo, layout_ids, radius_mm, centers_xy, config=None, target_k=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if target_k is None:
        k = float(p[8]) if p.size >= 9 else float(cfg.get("preferred_shell_scale_k", 1.12))
        hp = p[:8]
    else:
        k = float(target_k)
        hp = p[:8]
    G = _psm_plane_correction_from_params(hp)
    try:
        H = np.asarray(H0, dtype=np.float64).reshape(3, 3) @ G
        H = H / H[2, 2]
        Hinv = np.linalg.inv(H)
    except Exception:
        return np.ones((64,), dtype=np.float64) * 1e3
    residuals = []
    R = max(float(radius_mm), 1e-9)
    max_pts = int(cfg.get("max_candidate_points_for_model", 36))
    for cand, layout_idx in zip(combo, layout_ids):
        pts = _psm_candidate_points(cand, max_pts)
        if len(pts) < 6:
            continue
        homog = np.column_stack([pts, np.ones((len(pts),), dtype=np.float64)])
        xyh = (Hinv @ homog.T).T
        ok = np.isfinite(xyh).all(axis=1) & (np.abs(xyh[:, 2]) > 1e-12)
        if not np.any(ok):
            continue
        xy = xyh[ok, :2] / xyh[ok, 2:3]
        c = centers_xy[int(layout_idx)]
        rr = np.linalg.norm(xy - c.reshape(1, 2), axis=1)
        res = (rr - k * R) / R
        # robust clip: keep optimization focused on the shell family, not outliers.
        res = np.clip(res, -4.0, 4.0)
        residuals.extend(res.tolist())
    if not residuals:
        residuals = [1e3]
    # Regularize against unrealistic homography corrections; this prevents fitting
    # arbitrary H to one noisy shell. These terms are not GT-dependent.
    hp = np.asarray(hp, dtype=np.float64).reshape(8)
    w = float(cfg.get("homography_regularization_weight", 0.025))
    wt = float(cfg.get("homography_translation_regularization_weight", 0.015))
    wp = float(cfg.get("homography_projective_regularization_weight", 25.0))
    reg = [w*hp[0], w*hp[1], wt*hp[2]/max(R, 1e-9), w*hp[3], w*hp[4], wt*hp[5]/max(R, 1e-9), wp*hp[6], wp*hp[7]]
    if target_k is None:
        # weak preference for a physical halo shell, not an inner boundary.
        pref = float(cfg.get("preferred_shell_scale_k", 1.12))
        reg.append(0.05 * np.log(max(k, 1e-9) / max(pref, 1e-9)))
    residuals.extend(reg)
    return np.asarray(residuals, dtype=np.float64)


def _psm_fit_combo_projective_model(combo, layout_ids, H0, centers_xy, radius_mm, config=None, fixed_k=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    ks = []
    circs = []
    for cand in combo:
        k0 = cand.get("psm_k0", np.nan)
        c0 = cand.get("psm_circularity0", np.nan)
        if np.isfinite(k0):
            ks.append(float(k0))
        if np.isfinite(c0):
            circs.append(float(c0))
    if not ks:
        return None
    k_init = float(np.median(ks))
    lo_k, hi_k = cfg.get("k_initial_bounds", [1.0, 3.2])
    k_init = float(np.clip(k_init, float(lo_k), float(hi_k)))
    x0 = np.zeros((9,), dtype=np.float64)
    x0[8] = k_init
    hmax = float(cfg.get("max_h_scale_shear", 0.14))
    tmax = float(cfg.get("max_h_translation_mm", 28.0))
    pmax = float(cfg.get("max_h_projective", 7e-4))
    lower = np.array([-hmax, -hmax, -tmax, -hmax, -hmax, -tmax, -pmax, -pmax, float(lo_k)], dtype=np.float64)
    upper = np.array([ hmax,  hmax,  tmax,  hmax,  hmax,  tmax,  pmax,  pmax, float(hi_k)], dtype=np.float64)
    opt_used = False
    x = x0.copy()
    cost = None
    try:
        from scipy.optimize import least_squares
        res = least_squares(
            lambda z: _psm_residuals_for_combo(z, H0, combo, layout_ids, radius_mm, centers_xy, config=cfg, target_k=fixed_k),
            x0[:8] if fixed_k is not None else x0,
            bounds=(lower[:8], upper[:8]) if fixed_k is not None else (lower, upper),
            loss="soft_l1",
            f_scale=0.12,
            max_nfev=80,
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        opt_used = True
        if fixed_k is not None:
            x[:8] = res.x[:8]
            x[8] = float(fixed_k)
        else:
            x = res.x.astype(np.float64)
        cost = float(res.cost)
    except Exception as e:
        # Fall back to initial light-only H and common k. This keeps the script
        # runnable even when scipy is not installed.
        opt_used = False
        cost = None
    G = _psm_plane_correction_from_params(x[:8])
    try:
        H = np.asarray(H0, dtype=np.float64).reshape(3, 3) @ G
        H = H / H[2, 2]
    except Exception:
        return None
    r = _psm_residuals_for_combo(x if fixed_k is None else x[:8], H0, combo, layout_ids, radius_mm, centers_xy, config=cfg, target_k=fixed_k)
    # Remove the small regularization tail when reporting model residual.
    data_r = r[:-9] if len(r) > 16 else r
    med_abs = float(np.median(np.abs(data_r))) if len(data_r) else np.inf
    p90_abs = float(np.percentile(np.abs(data_r), 90)) if len(data_r) else np.inf
    k = float(fixed_k if fixed_k is not None else x[8])
    return {
        "H": H,
        "G": G,
        "params": x.tolist(),
        "k": k,
        "model_residual_median_norm": med_abs,
        "model_residual_p90_norm": p90_abs,
        "optimizer_used": bool(opt_used),
        "optimizer_cost": cost,
        "initial_shell_k_values": ks,
        "initial_shell_k_median": float(np.median(ks)),
        "initial_shell_k_std_norm": float(np.std(ks) / max(abs(float(np.median(ks))), 1e-9)),
        "initial_circularity_mean": float(np.nanmean(circs)) if circs else None,
    }


def _psm_choose_projective_shell_combo(candidates_per_light, layout_ids, H0, centers_xy, radius_mm, config=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if any(len(v) == 0 for v in candidates_per_light):
        return None, None, {"used": False, "reason": "empty_candidate_list"}
    combos = []
    max_combos = int(cfg.get("max_joint_combinations", 256))
    for idx, combo in enumerate(itertools.product(*candidates_per_light)):
        if idx >= max_combos:
            break
        ks = np.asarray([float(c.get("psm_k0", np.nan)) for c in combo], dtype=np.float64)
        circ = np.asarray([float(c.get("psm_circularity0", np.nan)) for c in combo], dtype=np.float64)
        if not np.all(np.isfinite(ks)):
            continue
        kmed = float(np.median(ks))
        kstd = float(np.std(ks) / max(abs(kmed), 1e-9))
        under = float(np.mean(np.maximum(0.0, float(cfg["min_shell_scale_k"]) - ks)))
        score0 = kstd + 0.35 * float(np.nanmean(circ)) + 1.3 * under
        combos.append((score0, combo, {"k_values_h0": ks.tolist(), "k_std_norm_h0": kstd, "circularity_mean_h0": float(np.nanmean(circ)), "score_h0": float(score0)}))
    if not combos:
        return None, None, {"used": False, "reason": "no_initial_combo"}
    combos.sort(key=lambda x: x[0])
    best = None
    best_combo = None
    best_dbg = None
    for rank, (score0, combo, dbg0) in enumerate(combos[:int(cfg.get("optimize_top_combinations", 18))]):
        fit = _psm_fit_combo_projective_model(combo, layout_ids, H0, centers_xy, radius_mm, config=cfg)
        if fit is None:
            continue
        # Main criterion: candidate shell points should become a common-scale
        # circle family in the known lamp plane under a single H.
        score = float(fit["model_residual_median_norm"] + 0.35 * fit["model_residual_p90_norm"] + 0.15 * fit["initial_shell_k_std_norm"])
        if best is None or score < best[0]:
            best = (score, fit)
            best_combo = list(combo)
            best_dbg = dict(dbg0)
            best_dbg.update({
                "used": True,
                "reason": "min_projective_shell_family_residual",
                "rank_initial": int(rank),
                "projective_score": float(score),
                "fit": fit,
            })
    if best is None:
        return None, None, {"used": False, "reason": "all_projective_fits_failed"}
    return best_combo, best[1], best_dbg


def _psm_fit_h_from_edge_points(edge_points_per_light, layout_ids, H_init, centers_xy, radius_mm, config=None):
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    pseudo_combo = []
    valid_layout = []
    for pts, lid in zip(edge_points_per_light, layout_ids):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 12:
            continue
        if len(pts) > int(cfg.get("max_candidate_points_for_model", 36)):
            idx = np.linspace(0, len(pts)-1, int(cfg.get("max_candidate_points_for_model", 36)), dtype=np.int32)
            pts = pts[idx]
        pseudo_combo.append({"source": "subpixel_edge_points", "contour": pts.reshape(-1, 1, 2).astype(np.float32), "ellipse": None})
        valid_layout.append(int(lid))
    if len(pseudo_combo) < 3:
        return None
    return _psm_fit_combo_projective_model(pseudo_combo, valid_layout, H_init, centers_xy, radius_mm, config=cfg, fixed_k=1.0)


def _psm_centers_from_H(H, layout_ids=None):
    centers_xy = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    if layout_ids is None:
        ids = list(range(len(centers_xy)))
    else:
        ids = [int(i) for i in layout_ids]
    uv = _psm_project_xy(H, centers_xy[ids])
    return uv.astype(np.float64)


def _psm_make_measurement_for_projected_physical_circle(base_measurement, H, layout_idx, num_points=144):
    centers_xy = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    cxy = centers_xy[int(layout_idx)]
    circle_xy = _psm_circle_points(cxy, LIGHT_RADIUS_MM, int(num_points))
    pts_img = _psm_project_xy(H, circle_xy)
    pts_img = pts_img[np.all(np.isfinite(pts_img), axis=1)]
    if len(pts_img) < 6:
        return None, None, None
    ell = _ggbs_fit_ellipse_from_points(pts_img)
    if ell is None:
        return None, None, None
    # The existing edge localizer expects measurement debug with arc points.
    meas = dict(base_measurement) if isinstance(base_measurement, dict) else {}
    meas["work_ellipse"] = ell
    meas["ellipse"] = ell
    meas["work_contour"] = _shell_as_cv_contour(pts_img)
    meas["contour"] = meas["work_contour"]
    meas["ellipse_center"] = np.asarray(ell[0], dtype=np.float64).reshape(2)
    meas["direct_ellipse_center"] = meas["ellipse_center"].copy()
    meas["point_debug_base"] = dict(meas.get("point_debug_base", {})) if isinstance(meas.get("point_debug_base", {}), dict) else {}
    rep = dict(meas["point_debug_base"].get("representation", {})) if isinstance(meas["point_debug_base"].get("representation", {}), dict) else {}
    rep.update({
        "representation_source": "projective_physical_circle_prediction",
        "assigned_layout_index": int(layout_idx),
        "confidence_light": max(float(rep.get("confidence_light", 0.0) or 0.0), 0.65),
    })
    meas["point_debug_base"]["representation"] = rep
    angles = _ellipse_param_angles(ell, pts_img)
    meas["point_debug_base"]["fast_normal_offset_edge"] = {
        "arc_points": pts_img.tolist(),
        "arc_angles": np.asarray(angles, dtype=np.float64).reshape(-1).tolist(),
        "arc_weights": np.ones((len(pts_img),), dtype=np.float64).tolist(),
        "edge_source": "projective_physical_circle_prediction",
    }
    return meas, ell, pts_img


def projective_shell_model_frontend(gray_light_proc, selected_items, K=None, pose_ctx=None, proj_ctx=None, config=None):
    """Projective-shell model center localization.

    Runtime-only inputs: light image measurements, camera intrinsics, known lamp
    layout/radius, and light-only pose initialization. The final center is the
    projection of the known physical circle center through a homography fitted
    from halo-shell / inward subpixel edge constraints. No chessboard pose or GT
    appears in this function.
    """
    cfg = dict(PROJECTIVE_SHELL_MODEL_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if pose_ctx is not None and proj_ctx is None:
        runtime_cache = pose_ctx.get("runtime_cache") if isinstance(pose_ctx, dict) else None
        if not isinstance(runtime_cache, dict) and K is not None:
            try:
                runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
            except Exception:
                runtime_cache = None
        if isinstance(runtime_cache, dict):
            proj_ctx = runtime_cache.get("projection_ctx")
    H0 = _psm_homography_from_pose_ctx(pose_ctx, K)
    if H0 is None:
        # Compatibility fallback.
        base = shell_constrained_inward_subpixel_frontend(gray_light_proc, selected_items, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg)
        for r in base:
            c = _ggbs_center_or_nan(r.get("center"))
            r["projective_shell_model_center"] = c.copy()
            r["projective_shell_edge_refined_center"] = c.copy()
            r["debug"] = dict(r.get("debug", {}))
            r["debug"]["projective_shell_model"] = {"used": False, "reason": "no_light_only_homography"}
        return base

    centers_xy = np.asarray(circle_centers_w, dtype=np.float64).reshape(-1, 3)[:, :2]
    candidates_per_light, base_measurements, layout_ids = _psm_select_top_candidates(gray_light_proc, selected_items, H0, config=cfg)
    combo, shell_fit, select_dbg = _psm_choose_projective_shell_combo(candidates_per_light, layout_ids, H0, centers_xy, LIGHT_RADIUS_MM, config=cfg)
    if combo is None or shell_fit is None:
        base = shell_constrained_inward_subpixel_frontend(gray_light_proc, selected_items, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg)
        for r in base:
            c = _ggbs_center_or_nan(r.get("center"))
            r["projective_shell_model_center"] = c.copy()
            r["projective_shell_edge_refined_center"] = c.copy()
            r["debug"] = dict(r.get("debug", {}))
            r["debug"]["projective_shell_model"] = select_dbg
        return base

    H_shell = np.asarray(shell_fit["H"], dtype=np.float64).reshape(3, 3)
    projective_shell_centers_all = _psm_centers_from_H(H_shell, layout_ids)

    # Build predicted physical circle boundary from H_shell, perform subpixel
    # localization near it, then optionally refine H with target physical radius.
    edge_points_per_light = []
    edge_sets = []
    subpixel_fits = []
    physical_meas = []
    for i, (meas, lid) in enumerate(zip(base_measurements, layout_ids)):
        pm, pell, ppts = _psm_make_measurement_for_projected_physical_circle(
            meas,
            H_shell,
            lid,
            num_points=int(cfg.get("num_physical_boundary_points", 144)),
        )
        if pm is None:
            edge_points_per_light.append(np.zeros((0, 2), dtype=np.float64))
            edge_sets.append({})
            subpixel_fits.append({"available": False, "fallback_reason": "physical_circle_measurement_failed"})
            physical_meas.append(meas)
            continue
        edge_set = estimate_subpixel_edge_set(gray_light_proc, pm, proj_ctx=proj_ctx)
        edge_sets.append(edge_set)
        pts = np.asarray(edge_set.get("points", []), dtype=np.float64).reshape(-1, 2) if isinstance(edge_set, dict) and edge_set.get("points") is not None else np.zeros((0, 2), dtype=np.float64)
        edge_points_per_light.append(pts)
        fit = fit_subpixel_edge_ellipse_center(
            edge_set=edge_set,
            boundary_center=np.asarray(pell[0], dtype=np.float64).reshape(2) if pell is not None else None,
            boundary_ellipse=pell,
            config=cfg.get("subpixel_config"),
        )
        subpixel_fits.append(fit)
        physical_meas.append(pm)

    H_edge = H_shell.copy()
    edge_fit_dbg = {"used": False, "reason": "disabled_or_failed"}
    if bool(cfg.get("enable_edge_refine", True)):
        refined = _psm_fit_h_from_edge_points(edge_points_per_light, layout_ids, H_shell, centers_xy, LIGHT_RADIUS_MM, config=cfg)
        if refined is not None and np.isfinite(refined.get("model_residual_median_norm", np.inf)):
            H_candidate = np.asarray(refined["H"], dtype=np.float64).reshape(3, 3)
            centers_candidate = _psm_centers_from_H(H_candidate, layout_ids)
            selected_centers = []
            for r in base_measurements:
                selected_centers.append(_ggbs_center_or_nan(r.get("ellipse_center", r.get("direct_ellipse_center"))))
            selected_centers = np.asarray(selected_centers, dtype=np.float64).reshape(-1, 2)
            shifts = np.linalg.norm(centers_candidate - selected_centers, axis=1)
            mean_shift = float(np.nanmean(shifts)) if shifts.size else np.inf
            max_shift = float(np.nanmax(shifts)) if shifts.size else np.inf
            residual_ok = float(refined.get("model_residual_median_norm", np.inf)) <= float(cfg.get("max_edge_model_residual_norm", 0.18))
            shift_ok = mean_shift <= float(cfg.get("max_center_shift_from_selected_mean_px", 8.0)) and max_shift <= float(cfg.get("max_center_shift_from_selected_px", 16.0))
            if residual_ok and shift_ok:
                H_edge = H_candidate
                edge_fit_dbg = dict(refined)
                edge_fit_dbg.update({"used": True, "reason": "accepted_edge_refined_projective_model", "mean_center_shift_from_selected_px": mean_shift, "max_center_shift_from_selected_px": max_shift})
            else:
                edge_fit_dbg = dict(refined)
                edge_fit_dbg.update({"used": False, "reason": "edge_refine_rejected_by_residual_or_shift_gate", "residual_ok": bool(residual_ok), "shift_ok": bool(shift_ok), "mean_center_shift_from_selected_px": mean_shift, "max_center_shift_from_selected_px": max_shift})

    projective_edge_centers_all = _psm_centers_from_H(H_edge, layout_ids)
    # If shell-only model itself is too weak, mark fallback but still expose values
    # for diagnostics. Final falls back to shell-inward subpixel in that case.
    shell_model_ok = float(shell_fit.get("model_residual_median_norm", np.inf)) <= float(cfg.get("max_shell_model_residual_norm", 0.22))
    if not shell_model_ok:
        fallback_base = shell_constrained_inward_subpixel_frontend(gray_light_proc, selected_items, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg)
    else:
        fallback_base = [None] * len(selected_items)

    results = []
    for i, (sel, meas, lid, shell_cand) in enumerate(zip(selected_items, base_measurements, layout_ids, combo)):
        shell_model_center = _ggbs_center_or_nan(projective_shell_centers_all[i])
        edge_model_center = _ggbs_center_or_nan(projective_edge_centers_all[i])
        fb = fallback_base[i] if i < len(fallback_base) else None
        shell_inward_center = _ggbs_center_or_nan(fb.get("center")) if isinstance(fb, dict) else _ggbs_center_or_nan(edge_model_center)
        final = edge_model_center.copy()
        final_source = "projective_shell_edge_refined_physical_center" if edge_fit_dbg.get("used") else "projective_shell_physical_center"
        if not shell_model_ok:
            final = shell_inward_center.copy()
            final_source = "fallback_shell_inward_subpixel_due_to_weak_projective_model"
        if not np.all(np.isfinite(final)):
            final = shell_model_center.copy() if np.all(np.isfinite(shell_model_center)) else shell_inward_center.copy()
        edge_dbg = edge_sets[i].get("debug", {}) if isinstance(edge_sets[i], dict) and isinstance(edge_sets[i].get("debug", {}), dict) else {}
        fit = subpixel_fits[i] if isinstance(subpixel_fits[i], dict) else {}
        subpx_center = _ggbs_center_or_nan(fit.get("center")) if fit.get("available") else np.array([np.nan, np.nan], dtype=np.float64)
        predicted_ellipse, predicted_pts = _psm_ellipse_from_projected_circle(H_shell, centers_xy[int(lid)], LIGHT_RADIUS_MM, n=int(cfg.get("num_physical_boundary_points", 144)))
        debug = {
            "algorithm_stage": "projective_shell_model_guided_physical_center_projection",
            "geometry_principle": "fit_single_planar_homography_and_common_shell_scale_from_halo_candidates_then_project_known_physical_centers",
            "assigned_layout_index": int(lid),
            "final_source": final_source,
            "fallback_reason": None if shell_model_ok else "weak_projective_shell_model_residual",
            "selected_shell_candidate": {k: shell_cand.get(k) for k in ["source", "level_index", "level_fraction", "threshold", "psm_k0", "psm_circularity0", "psm_score0", "psm_point_count"]},
            "projective_shell_selection": select_dbg,
            "projective_shell_fit": shell_fit,
            "projective_edge_refine": edge_fit_dbg,
            "subpixel_fit": fit.get("debug", {}),
            "edge_debug": edge_dbg,
        }
        results.append({
            "center": final.astype(np.float64),
            "projective_shell_model_center": shell_model_center.astype(np.float64),
            "projective_shell_edge_refined_center": edge_model_center.astype(np.float64),
            "shell_inward_subpixel_frontend_center": shell_inward_center.astype(np.float64),
            "shell_selected_boundary_center": _ggbs_center_or_nan(meas.get("ellipse_center", meas.get("direct_ellipse_center"))).astype(np.float64),
            "shell_inner_predicted_center": _ggbs_center_or_nan(predicted_ellipse[0] if predicted_ellipse is not None else shell_model_center).astype(np.float64),
            "subpixel_edge_fitted_center": subpx_center.astype(np.float64),
            "subpixel_edge_fitted_ellipse": fit.get("ellipse"),
            "boundary_ellipse": shell_cand.get("ellipse"),
            "inner_boundary_ellipse": predicted_ellipse,
            "boundary_contour": shell_cand.get("contour"),
            "inner_boundary_points": np.asarray(predicted_pts, dtype=np.float64).reshape(-1, 2),
            "edge_set": edge_sets[i],
            "measurement": physical_meas[i],
            "base_measurement": meas,
            "debug": debug,
        })
    return results

# ============================================================
# One-dimensional projective bias correction frontend
# New revision: do not recover a full H from the halo shell. Instead, use the
# image-derived four-light geometry to estimate a dominant projected compression
# axis and apply a bounded one-dimensional center shift after selected/subpixel
# center localization. Checkerboard/GT is never used here.
# ============================================================
PROJECTIVE_BIAS_1D_CONFIG = {
    # Base center selection
    "prefer_subpixel_base": True,
    "min_subpixel_arc_coverage": 0.85,
    "min_subpixel_success_ratio": 0.90,
    "max_subpixel_center_shift_px": 3.0,
    "max_subpixel_residual_p90": 0.12,

    # Direction / gate
    "min_complete_lights": 4,
    "min_axis_anisotropy": 0.005,   # abs(len_x-len_y)/max(len_x,len_y)
    "max_axis_ratio_q_for_correction": 0.96,
    "min_axis_ratio_q_for_correction": 0.35,
    "min_shape_signal_px": 0.40,    # median radius*(1-q)
    "min_shift_px": 0.25,
    "max_shift_px": 7.0,

    # Shift magnitude: s = median(sqrt(a*b)*(1-q)); lambda is fixed to 1.0.
    "shift_lambda": 1.0,
    "shape_power": 1.0,

    # Sign selection. In the safe version, auto first uses the image-only
    # vanishing point of the compressed layout axis and moves the center away
    # from that vanishing point. It no longer silently falls back to a fixed
    # default sign unless allow_default_sign_fallback=True.
    "sign_mode": "auto",           # auto | positive | negative | none
    "default_sign": 1.0,
    "allow_default_sign_fallback": False,
    "min_pose_sign_signal_px": 0.20,
    "min_vp_sign_signal_px": 20.0,

    # Optional safety: if shift is large but geometry is weak, do not apply.
    "require_direction_for_large_shift": True,
    "large_shift_px": 3.0,
}


def _pb1d_center_or_nan(x):
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return arr[:2].astype(np.float64)
    except Exception:
        pass
    return np.array([np.nan, np.nan], dtype=np.float64)


def _pb1d_unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return v / n


def _pb1d_ellipse_stats(ellipse):
    """Return center, major radius a, minor radius b, q=b/a, and minor axis unit vector."""
    if ellipse is None:
        return None
    try:
        (cx, cy), (MA, ma), angle_deg = ellipse
        w = float(MA)
        h = float(ma)
        ang = float(angle_deg)
        if not np.isfinite([cx, cy, w, h, ang]).all() or w <= 1e-6 or h <= 1e-6:
            return None
        # OpenCV fitEllipse angle corresponds to the first listed axis. The
        # following keeps only a reliable minor-axis direction; sign is axis-like.
        r1 = 0.5 * w
        r2 = 0.5 * h
        theta = np.deg2rad(ang)
        axis1 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        axis2 = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        if r1 >= r2:
            a, b = r1, r2
            minor_dir = axis2
        else:
            a, b = r2, r1
            minor_dir = axis1
        q = float(b / max(a, 1e-9))
        if minor_dir[1] < 0 or (abs(minor_dir[1]) < 1e-9 and minor_dir[0] < 0):
            minor_dir = -minor_dir
        return {
            "center": np.array([float(cx), float(cy)], dtype=np.float64),
            "a": float(a),
            "b": float(b),
            "q": q,
            "r_eff": float(np.sqrt(max(a * b, 0.0))),
            "minor_dir": _pb1d_unit(minor_dir),
        }
    except Exception:
        return None


def _pb1d_subpixel_is_valid(res, cfg):
    dbg = res.get("debug", {}) if isinstance(res, dict) else {}
    subdbg = dbg.get("subpixel_fit", {}) if isinstance(dbg.get("subpixel_fit", {}), dict) else {}
    # shell_constrained debug stores fit details directly under subpixel_fit.
    arc = subdbg.get("arc_coverage", subdbg.get("effective_arc_coverage"))
    succ = subdbg.get("edge_fit_success_ratio")
    shift = subdbg.get("center_shift_vs_boundary_px")
    p90 = subdbg.get("residual_p90")
    try:
        if arc is not None and float(arc) < float(cfg.get("min_subpixel_arc_coverage", 0.85)):
            return False, "low_subpixel_arc_coverage"
        if succ is not None and float(succ) < float(cfg.get("min_subpixel_success_ratio", 0.90)):
            return False, "low_subpixel_success_ratio"
        if shift is not None and float(shift) > float(cfg.get("max_subpixel_center_shift_px", 3.0)):
            return False, "large_subpixel_center_shift"
        if p90 is not None and float(p90) > float(cfg.get("max_subpixel_residual_p90", 0.12)):
            return False, "large_subpixel_residual_p90"
    except Exception:
        return False, "invalid_subpixel_debug"
    return True, None


def _pb1d_select_base_center(res, cfg):
    """Choose subpixel center if valid; otherwise use selected/shell boundary center."""
    boundary = _pb1d_center_or_nan(res.get("shell_selected_boundary_center"))
    if not np.all(np.isfinite(boundary)):
        boundary = _pb1d_center_or_nan(res.get("geometry_selected_boundary_center"))
    if not np.all(np.isfinite(boundary)):
        boundary = _pb1d_center_or_nan(res.get("center"))

    subpx = _pb1d_center_or_nan(res.get("subpixel_edge_fitted_center"))
    sub_ok, reason = _pb1d_subpixel_is_valid(res, cfg)
    if bool(cfg.get("prefer_subpixel_base", True)) and np.all(np.isfinite(subpx)) and sub_ok:
        return subpx.astype(np.float64), "subpixel_edge_fitted_center", None
    return boundary.astype(np.float64), "selected_boundary_center", reason or "subpixel_not_available"


def _pb1d_get_layout_id(res, fallback_idx=0):
    dbg = res.get("debug", {}) if isinstance(res, dict) else {}
    for key in ("assigned_layout_index",):
        try:
            v = dbg.get(key)
            if v is not None and np.isfinite(float(v)):
                return int(np.clip(int(v), 0, 3))
        except Exception:
            pass
    return int(np.clip(fallback_idx, 0, 3))


def _pb1d_order_by_layout(centers, layout_ids):
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    out = np.full((4, 2), np.nan, dtype=np.float64)
    for c, lid in zip(centers, layout_ids):
        lid = int(lid)
        if 0 <= lid < 4 and np.all(np.isfinite(c)):
            out[lid] = c
    return out


def _pb1d_layout_axes_from_centers(centers_by_layout):
    """Estimate the dominant 1D projective-bias axis from four-light layout.

    Earlier versions used the shorter projected side as the compressed axis.
    The full experiment shows that this can be wrong under oblique planar
    views: the apparent side length is affected by both foreshortening and
    perspective scale. The center bias follows the physical-axis family with
    stronger projective convergence, which is better indicated by the nearer
    vanishing point of the two layout-axis families.

    Therefore this version chooses the axis whose two parallel layout lines
    intersect at the nearer finite vanishing point. If vanishing-point estimation
    is not stable, it falls back to the shorter projected side.
    """
    c = np.asarray(centers_by_layout, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(c)):
        return None

    x_axis = 0.5 * ((c[1] - c[0]) + (c[3] - c[2]))
    y_axis = 0.5 * ((c[2] - c[0]) + (c[3] - c[1]))
    lx = float(np.linalg.norm(x_axis))
    ly = float(np.linalg.norm(y_axis))
    if lx < 1e-6 or ly < 1e-6:
        return None

    dx = _pb1d_unit(x_axis)
    dy = _pb1d_unit(y_axis)
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        return None

    # Canonical signs for reproducibility. The actual correction sign is
    # selected later from the vanishing point.
    if dx[1] < 0 or (abs(dx[1]) < 1e-9 and dx[0] < 0):
        dx = -dx
    if dy[1] < 0 or (abs(dy[1]) < 1e-9 and dy[0] < 0):
        dy = -dy

    center_mean = np.mean(c, axis=0)
    vpx = _pb1d_line_intersection(c[0], c[1], c[2], c[3])
    vpy = _pb1d_line_intersection(c[0], c[2], c[1], c[3])
    dist_x = float(np.linalg.norm(vpx - center_mean)) if vpx is not None else np.inf
    dist_y = float(np.linalg.norm(vpy - center_mean)) if vpy is not None else np.inf

    use_vp = np.isfinite(dist_x) or np.isfinite(dist_y)
    if use_vp and dist_x <= dist_y:
        d = dx
        name = "layout_x_axis_nearest_vanishing_point"
        compressed_len = lx
        other_len = ly
        selected_vp = vpx
        selected_vp_dist = dist_x
        other_vp_dist = dist_y
        selection_reason = "nearest_layout_vanishing_point"
    elif use_vp:
        d = dy
        name = "layout_y_axis_nearest_vanishing_point"
        compressed_len = ly
        other_len = lx
        selected_vp = vpy
        selected_vp_dist = dist_y
        other_vp_dist = dist_x
        selection_reason = "nearest_layout_vanishing_point"
    else:
        if lx <= ly:
            d = dx
            name = "layout_x_axis_shorter_fallback"
            compressed_len = lx
            other_len = ly
        else:
            d = dy
            name = "layout_y_axis_shorter_fallback"
            compressed_len = ly
            other_len = lx
        selected_vp = None
        selected_vp_dist = None
        other_vp_dist = None
        selection_reason = "shorter_axis_fallback_no_valid_vp"

    anis = abs(lx - ly) / max(lx, ly, 1e-9)
    vp_ratio = None
    if selected_vp_dist is not None and np.isfinite(selected_vp_dist) and other_vp_dist is not None and np.isfinite(other_vp_dist):
        vp_ratio = float(selected_vp_dist / max(other_vp_dist, 1e-9))

    return {
        "direction": d.astype(np.float64),
        "axis_name": name,
        "axis_selection_reason": selection_reason,
        "layout_x_direction": dx.astype(np.float64).tolist(),
        "layout_y_direction": dy.astype(np.float64).tolist(),
        "layout_x_len_px": lx,
        "layout_y_len_px": ly,
        "compressed_len_px": compressed_len,
        "other_len_px": other_len,
        "axis_anisotropy": float(anis),
        "layout_x_vp": vpx.tolist() if vpx is not None else None,
        "layout_y_vp": vpy.tolist() if vpy is not None else None,
        "layout_x_vp_distance_px": dist_x if np.isfinite(dist_x) else None,
        "layout_y_vp_distance_px": dist_y if np.isfinite(dist_y) else None,
        "selected_vp_distance_px": selected_vp_dist if selected_vp_dist is not None and np.isfinite(selected_vp_dist) else None,
        "other_vp_distance_px": other_vp_dist if other_vp_dist is not None and np.isfinite(other_vp_dist) else None,
        "vp_distance_ratio": vp_ratio,
    }



def _pb1d_line_intersection(p1, p2, p3, p4):
    """Intersection of two 2D lines p1-p2 and p3-p4; returns None if unstable."""
    try:
        p1 = np.asarray(p1, dtype=np.float64).reshape(2)
        p2 = np.asarray(p2, dtype=np.float64).reshape(2)
        p3 = np.asarray(p3, dtype=np.float64).reshape(2)
        p4 = np.asarray(p4, dtype=np.float64).reshape(2)
        A = np.column_stack([p2 - p1, -(p4 - p3)])
        b = p3 - p1
        det = float(np.linalg.det(A))
        if not np.isfinite(det) or abs(det) < 1e-9:
            return None
        t_u = np.linalg.solve(A, b)
        vp = p1 + float(t_u[0]) * (p2 - p1)
        if not np.all(np.isfinite(vp)):
            return None
        return vp.astype(np.float64)
    except Exception:
        return None


def _pb1d_vanishing_point_sign_from_layout(base_by_layout, axis_dbg, direction, cfg):
    """Choose the 1D correction sign from the image-only layout vanishing point.

    For a tilted planar rectangle, the shorter projected layout axis gives the
    dominant compression direction. The vanishing point of that physical-axis
    family indicates the projective near/far direction. Empirically and
    geometrically, the physical circle-center projection is corrected away from
    the halo-shell/projected-axis vanishing point along this dominant axis.

    This uses only base centers and known four-light topology; no GT/chessboard.
    """
    try:
        c = np.asarray(base_by_layout, dtype=np.float64).reshape(4, 2)
        d = np.asarray(direction, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
            return None, "vp_sign_invalid_centers_or_direction", None
        axis_name = str((axis_dbg or {}).get("axis_name", ""))
        if "layout_x" in axis_name:
            vp = _pb1d_line_intersection(c[0], c[1], c[2], c[3])
            vp_axis = "layout_x_vanishing_point"
        elif "layout_y" in axis_name:
            vp = _pb1d_line_intersection(c[0], c[2], c[1], c[3])
            vp_axis = "layout_y_vanishing_point"
        else:
            return None, "vp_sign_unknown_axis", None
        if vp is None:
            return None, "vp_sign_parallel_or_unstable_lines", None
        center = np.nanmean(c, axis=0)
        signal = float(np.dot(vp - center, d))
        if not np.isfinite(signal):
            return None, "vp_sign_nonfinite_signal", None
        if abs(signal) < float(cfg.get("min_vp_sign_signal_px", 20.0)):
            return None, "vp_sign_signal_too_small", signal
        # Move away from the vanishing point along the compressed axis.
        sign = -1.0 if signal > 0 else 1.0
        return sign, "auto_from_layout_vanishing_point_away_from_vp", {
            "signal_px": signal,
            "vanishing_point": vp.tolist(),
            "vanishing_axis": vp_axis,
            "rule": "sign = -sign(dot(vp - center, compressed_direction))",
        }
    except Exception as exc:
        return None, f"vp_sign_exception:{exc}", None

def _pb1d_pose_projection_by_layout(pose_ctx):
    if not isinstance(pose_ctx, dict):
        return None
    for key in ("proj_pose_final", "projected_centers", "pose_projection"):
        arr = pose_ctx.get(key)
        try:
            pts = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
            if pts.shape[0] >= 4 and np.all(np.isfinite(pts[:4])):
                return pts[:4].astype(np.float64)
        except Exception:
            continue
    return None


def _pb1d_choose_sign(base_by_layout, direction, pose_ctx, cfg, axis_dbg=None):
    """Choose correction sign without GT.

    Safe order:
      1) user-fixed positive/negative/none;
      2) image-only vanishing-point sign from the four-light layout;
      3) optional light-only pose projection sign;
      4) optional default sign only if explicitly allowed.

    If all automatic evidence is weak, return sign=0 so the gate falls back to
    the base center instead of applying a risky default correction.
    """
    mode = str(cfg.get("sign_mode", "auto")).lower()
    if mode in ("none", "off", "zero"):
        return 0.0, "disabled_by_sign_mode", {"signal": 0.0}
    if mode in ("positive", "+", "+1"):
        return 1.0, "fixed_positive", {"signal": 0.0}
    if mode in ("negative", "-", "-1"):
        return -1.0, "fixed_negative", {"signal": 0.0}

    # 1. Prefer a pure image-geometry sign: move away from the vanishing point
    # of the shorter projected layout-axis family.
    vp_sign, vp_reason, vp_info = _pb1d_vanishing_point_sign_from_layout(
        base_by_layout, axis_dbg, direction, cfg
    )
    if vp_sign is not None:
        return float(vp_sign), vp_reason, vp_info

    # 2. Optional fallback: runtime light-only pose projection, if available.
    proj = _pb1d_pose_projection_by_layout(pose_ctx)
    signal = 0.0
    if proj is not None and base_by_layout is not None and np.all(np.isfinite(base_by_layout)):
        diffs = proj - np.asarray(base_by_layout, dtype=np.float64).reshape(4, 2)
        vals = diffs @ np.asarray(direction, dtype=np.float64).reshape(2)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            signal = float(np.median(vals))
            if abs(signal) >= float(cfg.get("min_pose_sign_signal_px", 0.20)):
                return (1.0 if signal >= 0 else -1.0), "auto_from_light_only_pose_projection", {
                    "signal_px": signal,
                    "vp_fallback_reason": vp_reason,
                    "vp_fallback_info": vp_info,
                }

    # 3. No silent default sign unless explicitly enabled for ablation.
    if bool(cfg.get("allow_default_sign_fallback", False)):
        default = 1.0 if float(cfg.get("default_sign", 1.0)) >= 0 else -1.0
        return default, "auto_fallback_default_sign_explicitly_enabled", {
            "signal_px": signal,
            "vp_fallback_reason": vp_reason,
            "vp_fallback_info": vp_info,
        }
    return 0.0, "auto_sign_uncertain_fallback_to_base", {
        "signal_px": signal,
        "vp_fallback_reason": vp_reason,
        "vp_fallback_info": vp_info,
    }

def _pb1d_estimate_shift_from_ellipses(results, cfg):
    vals = []
    qs = []
    radii = []
    minor_dirs = []
    ellipse_sources = []
    for res in results:
        ell = res.get("subpixel_edge_fitted_ellipse")
        ell_source = "subpixel_edge_fitted_ellipse"
        if ell is None:
            ell = res.get("boundary_ellipse")
            ell_source = "boundary_ellipse"
        if ell is None:
            ell = res.get("inner_boundary_ellipse")
            ell_source = "inner_boundary_ellipse"
        st = _pb1d_ellipse_stats(ell)
        if st is None:
            continue
        q = float(st["q"])
        r_eff = float(st["r_eff"])
        if np.isfinite(q) and np.isfinite(r_eff):
            qs.append(q)
            radii.append(r_eff)
            ellipse_sources.append(ell_source)
            vals.append(r_eff * max(0.0, 1.0 - q) ** float(cfg.get("shape_power", 1.0)))
            if np.all(np.isfinite(st["minor_dir"])):
                minor_dirs.append(st["minor_dir"])
    if not vals:
        return 0.0, {
            "raw_shape_signal_px": None,
            "raw_shape_signal_per_light_px": None,
            "median_axis_ratio_q": None,
            "axis_ratio_q_per_light": None,
            "median_radius_px": None,
            "radius_per_light_px": None,
            "ellipse_source_per_light": None,
            "ellipse_axis_consistency": None,
            "reason": "no_valid_ellipse_stats",
        }
    vals = np.asarray(vals, dtype=np.float64)
    qs_arr = np.asarray(qs, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    raw_shape = float(np.nanmedian(vals))
    lam = float(cfg.get("shift_lambda", 1.0))
    s_raw = lam * raw_shape
    s = float(np.clip(s_raw, float(cfg.get("min_shift_px", 0.0)), float(cfg.get("max_shift_px", 7.0))))

    # Axis consistency is computed with orientation ambiguity removed.
    axis_cons = None
    if len(minor_dirs) >= 2:
        ds = np.asarray(minor_dirs, dtype=np.float64).reshape(-1, 2)
        ref = ds[0]
        aligned = []
        for d in ds:
            aligned.append(d if np.dot(d, ref) >= 0 else -d)
        axis_cons = float(np.linalg.norm(np.mean(np.asarray(aligned), axis=0)))

    return s, {
        "raw_shape_signal_px": raw_shape,
        "raw_shape_signal_per_light_px": vals.tolist(),
        "shift_lambda": lam,
        "shift_raw_px": float(s_raw),
        "shift_clipped_px": s,
        "median_axis_ratio_q": float(np.nanmedian(qs_arr)) if qs_arr.size else None,
        "axis_ratio_q_per_light": qs_arr.tolist(),
        "median_radius_px": float(np.nanmedian(radii_arr)) if radii_arr.size else None,
        "radius_per_light_px": radii_arr.tolist(),
        "ellipse_source_per_light": ellipse_sources,
        "ellipse_axis_consistency": axis_cons,
        "num_ellipses_for_shift": int(len(vals)),
    }


def _pb1d_per_light_shifts_from_debug(shift_dbg, cfg, n_lights):
    vals = shift_dbg.get("raw_shape_signal_per_light_px")
    if vals is None:
        return None
    try:
        arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    lam = float(cfg.get("shift_lambda", 1.0))
    min_s = float(cfg.get("min_shift_px", 0.0))
    max_s = float(cfg.get("max_shift_px", 7.0))
    arr = lam * arr
    arr = np.clip(arr, min_s, max_s)
    if arr.size < int(n_lights):
        global_shift = float(shift_dbg.get("shift_clipped_px", 0.0) or 0.0)
        padded = np.full((int(n_lights),), global_shift, dtype=np.float64)
        padded[:arr.size] = arr
        arr = padded
    elif arr.size > int(n_lights):
        arr = arr[:int(n_lights)]
    return arr.astype(np.float64)


def _pb1d_gate(axis_dbg, shift_dbg, sign, cfg):
    reasons = []
    if axis_dbg is None:
        reasons.append("no_valid_layout_axis")
    else:
        if float(axis_dbg.get("axis_anisotropy", 0.0)) < float(cfg.get("min_axis_anisotropy", 0.005)):
            reasons.append("axis_anisotropy_too_small")
    q = shift_dbg.get("median_axis_ratio_q")
    if q is None or not np.isfinite(float(q)):
        reasons.append("missing_axis_ratio")
    else:
        if float(q) > float(cfg.get("max_axis_ratio_q_for_correction", 0.96)):
            reasons.append("ellipse_too_close_to_circle_low_tilt")
        if float(q) < float(cfg.get("min_axis_ratio_q_for_correction", 0.35)):
            reasons.append("ellipse_axis_ratio_unreliable")
    raw_signal = shift_dbg.get("raw_shape_signal_px")
    if raw_signal is None or float(raw_signal) < float(cfg.get("min_shape_signal_px", 0.40)):
        reasons.append("shape_signal_too_weak")
    shift = shift_dbg.get("shift_clipped_px")
    if shift is None or float(shift) < float(cfg.get("min_shift_px", 0.25)):
        reasons.append("shift_too_small")
    if abs(float(sign)) < 0.5:
        reasons.append("sign_disabled")
    # For large shifts, require a valid direction; axis consistency is optional
    # because halo minor axes can be noisy under saturation.
    if bool(cfg.get("require_direction_for_large_shift", True)) and shift is not None:
        if float(shift) >= float(cfg.get("large_shift_px", 3.0)) and axis_dbg is None:
            reasons.append("large_shift_without_direction")
    return len(reasons) == 0, reasons


def projective_bias_1d_frontend(gray_light_proc, selected_items, K=None, pose_ctx=None, proj_ctx=None, config=None):
    """Selected/subpixel center followed by bounded 1D projective-bias correction.

    Runtime inputs are only light image derived features, camera intrinsics,
    known four-light layout, lamp radius through existing measurement modules,
    and optional ROI. Chessboard/GT is never used.
    """
    cfg = dict(PROJECTIVE_BIAS_1D_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if pose_ctx is not None and proj_ctx is None:
        runtime_cache = pose_ctx.get("runtime_cache") if isinstance(pose_ctx, dict) else None
        if not isinstance(runtime_cache, dict) and K is not None:
            try:
                runtime_cache = _build_pose_runtime_cache(pose_ctx, K, circle_centers_w)
            except Exception:
                runtime_cache = None
        if isinstance(runtime_cache, dict):
            proj_ctx = runtime_cache.get("projection_ctx")

    # Reuse the best existing image-domain stage; this keeps baselines unchanged
    # and adds geometry only as a post-center bounded correction.
    base_results = shell_constrained_inward_subpixel_frontend(
        gray_light_proc, selected_items, K=K, pose_ctx=pose_ctx, proj_ctx=proj_ctx, config=cfg
    )

    base_centers = []
    base_sources = []
    base_fallbacks = []
    layout_ids = []
    for i, res in enumerate(base_results):
        c, src, freason = _pb1d_select_base_center(res, cfg)
        base_centers.append(c)
        base_sources.append(src)
        base_fallbacks.append(freason)
        layout_ids.append(_pb1d_get_layout_id(res, fallback_idx=i))
    base_centers = np.asarray(base_centers, dtype=np.float64).reshape(-1, 2)
    layout_ids = list(map(int, layout_ids))

    axis_dbg = None
    direction = np.array([np.nan, np.nan], dtype=np.float64)
    base_by_layout = None
    if len(base_centers) >= int(cfg.get("min_complete_lights", 4)) and len(set(layout_ids[:4])) == 4:
        base_by_layout = _pb1d_order_by_layout(base_centers, layout_ids)
        axis_dbg = _pb1d_layout_axes_from_centers(base_by_layout)
        if axis_dbg is not None:
            direction = np.asarray(axis_dbg["direction"], dtype=np.float64).reshape(2)

    shift_px, shift_dbg = _pb1d_estimate_shift_from_ellipses(base_results, cfg)
    sign, sign_reason, sign_signal = _pb1d_choose_sign(base_by_layout, direction, pose_ctx, cfg, axis_dbg=axis_dbg)
    used, gate_reasons = _pb1d_gate(axis_dbg, shift_dbg, sign, cfg)

    shift_mode = str(cfg.get("shift_mode", "global_median")).lower()
    per_light_shift_px = None

    if not used or not np.all(np.isfinite(direction)):
        final_centers = base_centers.copy()
        applied_shift = 0.0
        final_source = "projective_bias_1d_fallback_base_center"
    else:
        if shift_mode in ("per_light", "per_lamp", "individual"):
            per_light_shift_px = _pb1d_per_light_shifts_from_debug(shift_dbg, cfg, len(base_centers))
        if per_light_shift_px is not None:
            signed_shifts = float(sign) * per_light_shift_px.reshape(-1, 1)
            final_centers = base_centers + signed_shifts * direction.reshape(1, 2)
            applied_shift = float(np.nanmean(np.abs(per_light_shift_px)))
            final_source = "projective_bias_1d_center_per_light_shift"
        else:
            applied_shift = float(sign) * float(shift_px)
            final_centers = base_centers + applied_shift * direction.reshape(1, 2)
            final_source = "projective_bias_1d_center"

    group_debug = {
        "used": bool(used),
        "fallback_reasons": gate_reasons,
        "base_sources": base_sources,
        "base_fallback_reasons": base_fallbacks,
        "layout_ids": layout_ids,
        "direction": direction.tolist() if np.all(np.isfinite(direction)) else [None, None],
        "axis_debug": axis_dbg,
        "shift_debug": shift_dbg,
        "shift_px_abs": float(abs(applied_shift)),
        "signed_shift_px": float(applied_shift),
        "shift_mode": shift_mode,
        "per_light_shift_px_abs": None if per_light_shift_px is None else per_light_shift_px.tolist(),
        "per_light_signed_shift_px": None if per_light_shift_px is None else (float(sign) * per_light_shift_px).tolist(),
        "sign": float(sign),
        "sign_reason": sign_reason,
        "sign_debug": sign_signal,
        "sign_signal_from_light_only_pose_px": (float(sign_signal) if isinstance(sign_signal, (int, float, np.floating)) and np.isfinite(sign_signal) else None),
        "final_source": final_source,
        "geometry_principle": "bounded_1d_projective_bias_shift_along_image_estimated_layout_compression_axis",
        "uses_checkerboard_or_gt": False,
    }

    out = []
    for i, res in enumerate(base_results):
        r = dict(res)
        base_c = base_centers[i]
        final_c = final_centers[i]
        pre_bias_shell_c = _pb1d_center_or_nan(r.get("shell_inward_subpixel_frontend_center"))
        if not np.all(np.isfinite(pre_bias_shell_c)):
            pre_bias_shell_c = base_c.copy()
        dbg = dict(r.get("debug", {})) if isinstance(r.get("debug", {}), dict) else {}
        per_dbg = dict(group_debug)
        per_dbg.update({
            "algorithm_stage": "selected_or_subpixel_center_plus_bounded_1d_projective_bias_correction",
            "light_index": int(i),
            "assigned_layout_index": int(layout_ids[i]) if i < len(layout_ids) else int(i),
            "base_center_source": base_sources[i] if i < len(base_sources) else None,
            "base_center": base_c.tolist(),
            "projective_bias_1d_center": final_c.tolist(),
        })
        dbg["projective_bias_1d"] = per_dbg
        dbg["final_source"] = final_source if used else (base_sources[i] if i < len(base_sources) else final_source)
        dbg["fallback_reason"] = None if used else ";".join(gate_reasons)
        r["debug"] = dbg
        r["center"] = final_c.astype(np.float64)
        r["projective_bias_1d_base_center"] = base_c.astype(np.float64)
        r["projective_bias_1d_center"] = final_c.astype(np.float64)
        r["projective_bias_1d_debug"] = per_dbg
        # Keep the pre-bias frontend output as an actual ablation value. Do not
        # fill it from r["center"], because center has already been replaced by
        # the projective-bias corrected final center at this stage.
        r["shell_inward_subpixel_frontend_center"] = pre_bias_shell_c.astype(np.float64)
        out.append(r)
    return out
