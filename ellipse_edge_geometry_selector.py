"""
ellipse_edge_geometry_selector.py

独立模块：面向几何校正的椭圆边缘搜索与候选选择

核心思想
--------
不是寻找“图像拟合最稳定”的椭圆，而是寻找：
    反投影到灯平面后最接近真实圆盘投影边界的椭圆/轮廓。

适用方式
--------
在原主程序中，用这个模块替代/封装：
    fit_core_ellipse_from_multilevel(...)
    choose_measurement_representation(...)

你只需要在主程序里把原来的：
    work_ellipse, work_contour, rep_dbg = choose_measurement_representation(gray, contour, ellipse)
换成：
    from ellipse_edge_geometry_selector import choose_geometry_consistent_measurement_representation
    work_ellipse, work_contour, rep_dbg = choose_geometry_consistent_measurement_representation(
        gray=gray_light_proc,
        contour=contour,
        ellipse=ellipse,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )

如果暂时拿不到 K / plane_normal_camera / plane_distance，也可以先传 None；
此时模块会退化为仅图像域评分，但仍保留统一接口。

输出保持兼容：
    work_ellipse, work_contour, rep_dbg
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GeometryEdgeSelectorConfig:
    roi_pad: int = 14
    min_inside_pixels: int = 24
    n_candidate_levels: int = 11
    level_low_q: float = 55.0
    level_high_q: float = 98.8
    sat_q: float = 99.5
    min_dyn: float = 6.0

    min_component_area: float = 10.0
    morph_scale_div: float = 16.0
    geo_sample_count: int = 96

    # 这些不是任务经验阈值，而是稳健统计中的“权重”与采样分辨率。
    # 真正的筛选边界由当前 ROI / 当前候选集合的分布自适应估计。
    w_img_resid: float = 1.2
    w_img_drift: float = 0.35
    w_img_axis: float = 0.35
    w_img_grad: float = 0.65
    w_img_edge_drop: float = 0.95
    w_img_bloom: float = 0.35

    w_geo_circle_rms: float = 1.20
    w_geo_radius_cv: float = 1.00
    w_geo_center_shift: float = 0.60
    w_geo_reproj_resid: float = 0.85
    w_geo_valid_ratio: float = 0.60

    # 自适应目标带的稳健带宽下限（数值稳定项，不是语义阈值）
    level_sigma_floor: float = 0.06
    suspicious_center_shift_px: float = 8.0


def _ellipse_contour_residual(ellipse, contour_pts) -> float:
    if ellipse is None or contour_pts is None or len(contour_pts) < 6:
        return float("inf")
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


def _fit_ellipse_best(contour) -> Optional[Tuple]:
    if contour is None or len(contour) < 6:
        return None
    funcs = []
    if hasattr(cv2, "fitEllipseAMS"):
        funcs.append(cv2.fitEllipseAMS)
    if hasattr(cv2, "fitEllipseDirect"):
        funcs.append(cv2.fitEllipseDirect)
    funcs.append(cv2.fitEllipse)
    best = None
    best_resid = float("inf")
    for fn in funcs:
        try:
            e = fn(np.asarray(contour, dtype=np.float32))
            resid = _ellipse_contour_residual(e, contour.reshape(-1, 2))
            if np.isfinite(resid) and resid < best_resid:
                best_resid = resid
                best = e
        except Exception:
            pass
    return best


def _crop_local(gray: np.ndarray, ellipse, pad: int = 10):
    (cx, cy), (MA, ma), _ = ellipse
    r = int(0.5 * max(MA, ma) + pad)
    x0 = max(int(np.floor(cx)) - r, 0)
    y0 = max(int(np.floor(cy)) - r, 0)
    x1 = min(int(np.ceil(cx)) + r, gray.shape[1] - 1)
    y1 = min(int(np.ceil(cy)) + r, gray.shape[0] - 1)
    roi_crop = gray[y0:y1 + 1, x0:x1 + 1]
    e_local = ((cx - x0, cy - y0), (MA, ma), ellipse[2])
    return roi_crop, (x0, y0), e_local


def _sample_ellipse_points(ellipse, num_samples: int = 96) -> np.ndarray:
    (cx, cy), (MA, ma), angle_deg = ellipse
    a, b = max(MA * 0.5, 1e-6), max(ma * 0.5, 1e-6)
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    ts = np.linspace(0, 2 * np.pi, int(num_samples), endpoint=False)
    x = cx + a * np.cos(ts) * c - b * np.sin(ts) * s
    y = cy + a * np.cos(ts) * s + b * np.sin(ts) * c
    return np.column_stack([x, y]).astype(np.float64)


def _ellipse_center(ellipse) -> np.ndarray:
    return np.asarray(ellipse[0], dtype=np.float64)


def _ellipse_axes(ellipse) -> np.ndarray:
    return np.asarray(ellipse[1], dtype=np.float64)


def _backproject_pixels_to_plane(pts_uv: np.ndarray, K: np.ndarray, plane_normal_camera: np.ndarray, plane_distance: float):
    pts = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    n = np.asarray(plane_normal_camera, dtype=np.float64).reshape(3)
    d = float(plane_distance)
    K_inv = np.linalg.inv(K)
    homog = np.column_stack([pts, np.ones((len(pts),), dtype=np.float64)])
    rays = (K_inv @ homog.T).T
    denom = rays @ n
    valid = np.abs(denom) > 1e-9
    X = np.full((len(pts), 3), np.nan, dtype=np.float64)
    if np.any(valid):
        t = -d / denom[valid]
        X[valid] = rays[valid] * t[:, None]
    return X, valid


def _project_camera_points_to_image(Xc: np.ndarray, K: np.ndarray) -> np.ndarray:
    Xc = np.asarray(Xc, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    z = Xc[:, 2]
    uv = np.full((len(Xc), 2), np.nan, dtype=np.float64)
    valid = np.abs(z) > 1e-9
    if np.any(valid):
        x = Xc[valid, 0] / z[valid]
        y = Xc[valid, 1] / z[valid]
        homog = np.column_stack([x, y, np.ones_like(x)])
        proj = (K @ homog.T).T
        uv[valid] = proj[:, :2]
    return uv


def _plane_basis_from_normal(n_c: np.ndarray):
    n = np.asarray(n_c, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    anchor = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(n, anchor)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return n, e1, e2


def _project_3d_points_to_plane_coords(Xc: np.ndarray, n_c: np.ndarray):
    X = np.asarray(Xc, dtype=np.float64).reshape(-1, 3)
    n, e1, e2 = _plane_basis_from_normal(n_c)
    X0 = np.nanmean(X, axis=0)
    d = X - X0
    u = d @ e1
    v = d @ e2
    return X0, e1, e2, np.column_stack([u, v])


def _fit_circle_2d_algebraic(points_uv: np.ndarray):
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 6:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    D, E, F = theta
    cx = -0.5 * D
    cy = -0.5 * E
    r2 = cx * cx + cy * cy - F
    if r2 <= 1e-12 or not np.isfinite(r2):
        return None
    r = float(np.sqrt(r2))
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    resid = rr - r
    return {
        "center_2d": np.array([cx, cy], dtype=np.float64),
        "radius": r,
        "radius_std": float(np.std(rr)),
        "radius_cv": float(np.std(rr) / max(r, 1e-6)),
        "resid_rms": float(np.sqrt(np.mean(resid * resid))),
        "resid_median_abs": float(np.median(np.abs(resid))),
    }


def _compute_edge_drop(roi_gray: np.ndarray, contour_local: np.ndarray, mask_inside: np.ndarray) -> float:
    h, w = roi_gray.shape[:2]
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour_local.reshape(-1, 1, 2)], -1, 255, thickness=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(mask_inside, kernel, iterations=1)
    outer = cv2.dilate(mask_inside, kernel, iterations=1)
    ring_out = np.clip(outer.astype(np.int16) - mask_inside.astype(np.int16), 0, 255).astype(np.uint8)
    inner_band = cv2.bitwise_and(inner, contour_mask)
    outer_band = cv2.bitwise_and(ring_out, contour_mask)
    if np.count_nonzero(inner_band) < 8:
        inner_band = inner
    if np.count_nonzero(outer_band) < 8:
        outer_band = ring_out
    if np.count_nonzero(inner_band) == 0 or np.count_nonzero(outer_band) == 0:
        return 0.0
    inner_val = float(np.mean(roi_gray[inner_band > 0]))
    outer_val = float(np.mean(roi_gray[outer_band > 0]))
    return max(0.0, inner_val - outer_val)


def _evaluate_candidate_geometry_consistency(ellipse_img, K, plane_normal_camera, plane_distance, cfg: GeometryEdgeSelectorConfig):
    if K is None or plane_normal_camera is None or plane_distance is None:
        return {
            "geo_enabled": False,
            "geo_valid_ratio": np.nan,
            "geo_circle_rms": np.nan,
            "geo_radius_cv": np.nan,
            "geo_center_shift": np.nan,
            "geo_reproj_resid": np.nan,
        }
    pts_img = _sample_ellipse_points(ellipse_img, cfg.geo_sample_count)
    Xc, valid = _backproject_pixels_to_plane(pts_img, K, plane_normal_camera, plane_distance)
    valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
    if np.count_nonzero(valid) < max(12, cfg.geo_sample_count // 4):
        return {
            "geo_enabled": True,
            "geo_valid_ratio": valid_ratio,
            "geo_circle_rms": np.inf,
            "geo_radius_cv": np.inf,
            "geo_center_shift": np.inf,
            "geo_reproj_resid": np.inf,
        }
    Xv = Xc[valid]
    X0, e1, e2, plane_pts = _project_3d_points_to_plane_coords(Xv, plane_normal_camera)
    circle = _fit_circle_2d_algebraic(plane_pts)
    if circle is None:
        return {
            "geo_enabled": True,
            "geo_valid_ratio": valid_ratio,
            "geo_circle_rms": np.inf,
            "geo_radius_cv": np.inf,
            "geo_center_shift": np.inf,
            "geo_reproj_resid": np.inf,
        }
    center3d = X0 + circle["center_2d"][0] * e1 + circle["center_2d"][1] * e2
    center_uv = _project_camera_points_to_image(center3d[None, :], K)[0]
    center_shift = float(np.linalg.norm(center_uv - _ellipse_center(ellipse_img))) if np.all(np.isfinite(center_uv)) else np.inf
    rr_img = np.linalg.norm(pts_img - center_uv[None, :], axis=1)
    reproj_resid = float(np.std(rr_img) / max(np.mean(rr_img), 1e-6)) if np.all(np.isfinite(rr_img)) else np.inf
    return {
        "geo_enabled": True,
        "geo_valid_ratio": valid_ratio,
        "geo_circle_rms": float(circle["resid_rms"]),
        "geo_radius_cv": float(circle["radius_cv"]),
        "geo_center_shift": center_shift,
        "geo_reproj_resid": reproj_resid,
        "geo_center_uv": center_uv.tolist() if np.all(np.isfinite(center_uv)) else None,
        "geo_circle_center_3d": center3d.tolist(),
        "geo_circle_radius": float(circle["radius"]),
    }


def _normalize_scores(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if np.count_nonzero(finite) == 0:
        return out
    vals = arr[finite]
    lo = np.percentile(vals, 5.0)
    hi = np.percentile(vals, 95.0)
    if hi - lo < 1e-12:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    return out

def _robust_center_scale(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) == 0:
        return 0.0, 1.0
    vals = arr[finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(1.4826 * mad, 1e-6)
    return med, scale


def _robust_z_positive(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    med, scale = _robust_center_scale(arr)
    z = (arr - med) / scale
    z = np.where(np.isfinite(z), np.maximum(z, 0.0), 0.0)
    return z.astype(np.float64)


def _rank_unit(arr: np.ndarray, high_is_bad: bool = True) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) <= 1:
        return out
    vals = arr[finite]
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(vals))
    if high_is_bad:
        out[finite] = ranks
    else:
        out[finite] = 1.0 - ranks
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(mask) == 0:
        return float(np.nanmedian(v)) if np.any(np.isfinite(v)) else 0.5
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cw /= max(cw[-1], 1e-12)
    return float(np.interp(np.clip(q, 0.0, 1.0), cw, v))


def _adaptive_level_target(candidates, global_sat_ratio: float, dyn: float, cfg: GeometryEdgeSelectorConfig):
    level = np.asarray([c["level_norm"] for c in candidates], dtype=np.float64)
    edge_drop = np.asarray([c["edge_drop"] for c in candidates], dtype=np.float64)
    grad = np.asarray([c["boundary_grad"] for c in candidates], dtype=np.float64)
    sat = np.asarray([c["sat_ratio"] for c in candidates], dtype=np.float64)
    plateau = np.asarray([c["plateau_core_ratio"] for c in candidates], dtype=np.float64)
    halo = np.asarray([c["halo_low_ratio"] for c in candidates], dtype=np.float64)

    edge_n = _normalize_scores(edge_drop / max(dyn, 1e-6))
    grad_n = _normalize_scores(grad)
    sat_good = 1.0 - _rank_unit(sat, high_is_bad=True)
    plateau_good = 1.0 - _rank_unit(plateau, high_is_bad=True)
    halo_good = 1.0 - _rank_unit(halo, high_is_bad=True)
    weights = np.clip(0.45 * edge_n + 0.35 * grad_n + 0.10 * sat_good + 0.05 * plateau_good + 0.05 * halo_good, 1e-6, None)

    target = _weighted_quantile(level, weights, 0.5)
    q25 = _weighted_quantile(level, weights, 0.25)
    q75 = _weighted_quantile(level, weights, 0.75)
    sigma = max(0.7413 * (q75 - q25), cfg.level_sigma_floor)

    # 轻度饱和时目标带允许向外稍移；高饱和时允许向内略收，但完全由当前图像饱和统计连续控制
    sat_shift = float(np.clip((global_sat_ratio - 0.03) / 0.10, -1.0, 1.0))
    target = float(np.clip(target + 0.08 * sat_shift, 0.12, 0.88))
    return target, float(sigma)


def _adaptive_inner_outer_penalties(candidates, dyn: float):
    level = np.asarray([c["level_norm"] for c in candidates], dtype=np.float64)
    sat = np.asarray([c["sat_ratio"] for c in candidates], dtype=np.float64)
    plateau = np.asarray([c["plateau_core_ratio"] for c in candidates], dtype=np.float64)
    halo = np.asarray([c["halo_low_ratio"] for c in candidates], dtype=np.float64)
    grad = np.asarray([c["boundary_grad"] for c in candidates], dtype=np.float64)
    edge = np.asarray([c["edge_drop"] / max(dyn, 1e-6) for c in candidates], dtype=np.float64)

    r_level_hi = _rank_unit(level, high_is_bad=True)
    r_level_lo = _rank_unit(level, high_is_bad=False)
    r_sat_hi = _rank_unit(sat, high_is_bad=True)
    r_plateau_hi = _rank_unit(plateau, high_is_bad=True)
    r_halo_hi = _rank_unit(halo, high_is_bad=True)
    r_edge_lo = _rank_unit(edge, high_is_bad=False)
    r_grad_lo = _rank_unit(grad, high_is_bad=False)

    inner = 0.45 * r_level_hi + 0.35 * r_plateau_hi + 0.20 * r_sat_hi
    outer = 0.35 * r_level_lo + 0.30 * r_halo_hi + 0.20 * r_edge_lo + 0.15 * r_grad_lo
    return inner.astype(np.float64), outer.astype(np.float64)


def choose_geometry_consistent_measurement_representation(
    gray: np.ndarray,
    contour: np.ndarray,
    ellipse,
    K: Optional[np.ndarray] = None,
    plane_normal_camera: Optional[np.ndarray] = None,
    plane_distance: Optional[float] = None,
    config: Optional[GeometryEdgeSelectorConfig] = None,
):
    cfg = config or GeometryEdgeSelectorConfig()
    if ellipse is None or contour is None:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "invalid_input", "config": asdict(cfg)}
    roi_gray, (ox, oy), _ = _crop_local(gray, ellipse, pad=max(cfg.roi_pad, 10))
    if roi_gray.size == 0:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "empty_roi", "config": asdict(cfg)}

    contour_local = contour.reshape(-1, 2).astype(np.int32).copy()
    contour_local[:, 0] -= ox
    contour_local[:, 1] -= oy

    mask = np.zeros_like(roi_gray, dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [contour_local.reshape(-1, 1, 2)], 255)
    except Exception:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "mask_fail", "config": asdict(cfg)}

    vals = roi_gray[mask > 0]
    if vals.size < cfg.min_inside_pixels:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "too_few_pixels", "config": asdict(cfg)}

    q_low = float(np.percentile(vals, cfg.level_low_q))
    q_high = float(np.percentile(vals, cfg.level_high_q))
    sat_thr = float(np.percentile(vals, cfg.sat_q))
    dyn = float(q_high - q_low)
    if dyn < cfg.min_dyn:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "low_dynamic_range", "config": asdict(cfg)}

    raw_center = _ellipse_center(ellipse)
    raw_axes = _ellipse_axes(ellipse)
    raw_area = max(1.0, float(cv2.contourArea(contour)))
    global_sat_ratio = float(np.mean(vals >= sat_thr)) if vals.size else 0.0

    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    grad_ref = float(np.percentile(gmag[mask > 0], 90.0) + 1e-9)

    quantiles = np.linspace(cfg.level_low_q, cfg.level_high_q, cfg.n_candidate_levels)
    levels = np.unique(np.round(np.percentile(vals, quantiles)).astype(np.int32))
    if levels.size < 3:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "too_few_levels", "config": asdict(cfg)}

    ksz = int(np.clip(round(min(raw_axes) / cfg.morph_scale_div), 3, 7))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    candidates: List[Dict[str, Any]] = []
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
        best_local_score = float("inf")
        for cc in cs:
            if len(cc) < 6:
                continue
            area = float(cv2.contourArea(cc))
            if area < cfg.min_component_area:
                continue
            e_try = _fit_ellipse_best(cc)
            if e_try is None:
                continue

            ctr_img = np.array([e_try[0][0] + ox, e_try[0][1] + oy], dtype=np.float64)
            axes = _ellipse_axes(e_try)
            drift = float(np.linalg.norm(ctr_img - raw_center))
            axis_delta = float(np.linalg.norm((axes - raw_axes) / (raw_axes + 1e-6)))
            area_ratio = float(area / raw_area)
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

            cc_pts = cc.reshape(-1, 2).astype(np.int32)
            cc_pts[:, 0] = np.clip(cc_pts[:, 0], 0, roi_gray.shape[1] - 1)
            cc_pts[:, 1] = np.clip(cc_pts[:, 1], 0, roi_gray.shape[0] - 1)
            boundary_grad = float(np.mean(gmag[cc_pts[:, 1], cc_pts[:, 0]]) / grad_ref)
            edge_drop = _compute_edge_drop(roi_gray, cc.reshape(-1, 2).astype(np.int32), cc_mask)

            local_score = (
                cfg.w_img_resid * resid +
                cfg.w_img_drift * drift / 8.0 +
                cfg.w_img_axis * axis_delta +
                cfg.w_img_sat * sat_ratio +
                cfg.w_img_plateau * plateau_core_ratio +
                cfg.w_img_halo * halo_low_ratio +
                cfg.w_img_level * abs(level_norm - 0.62) +
                cfg.w_img_grad * (1.0 - np.clip(boundary_grad, 0.0, 1.5)) -
                cfg.w_img_edge_drop * np.clip(edge_drop / max(dyn, 1e-6), 0.0, 2.0)
            )
            if local_score < best_local_score:
                best_local_score = local_score
                best_local = {
                    "level_idx": int(lvl_idx),
                    "thr": int(thr),
                    "contour_local": cc.copy(),
                    "ellipse_local": e_try,
                    "area": area,
                    "center_img": ctr_img,
                    "axes": axes,
                    "drift": drift,
                    "axis_delta": axis_delta,
                    "area_ratio": area_ratio,
                    "resid": resid,
                    "sat_ratio": sat_ratio,
                    "plateau_core_ratio": plateau_core_ratio,
                    "halo_low_ratio": halo_low_ratio,
                    "boundary_grad": boundary_grad,
                    "edge_drop": float(edge_drop),
                    "level_norm": level_norm,
                    "local_score": float(local_score),
                }
        if best_local is not None:
            candidates.append(best_local)

    if len(candidates) < 1:
        return ellipse, contour, {"enabled": True, "used": False, "reason": "no_candidate", "config": asdict(cfg)}

    candidates.sort(key=lambda d: d["level_idx"])
    growths = []
    for i, cand in enumerate(candidates):
        prev_c = candidates[i - 1] if i > 0 else None
        next_c = candidates[i + 1] if i + 1 < len(candidates) else None
        stability = 0.0
        support = 0
        for nb in (prev_c, next_c):
            if nb is None:
                continue
            stability += float(np.linalg.norm(cand["center_img"] - nb["center_img"]))
            stability += 0.45 * float(np.linalg.norm((cand["axes"] - nb["axes"]) / (raw_axes + 1e-6)))
            stability += 2.0 * abs(cand["resid"] - nb["resid"])
            support += 1
        stability = stability / max(support, 1)
        outward_growth = 0.0
        if next_c is not None:
            outward_growth = max(0.0, np.log(max(next_c["area"], 1e-6) / max(cand["area"], 1e-6)))
        cand["stability"] = float(stability)
        cand["bloom_expansion"] = float(outward_growth)
        growths.append(outward_growth)

    growth_norm = _normalize_scores(np.asarray(growths, dtype=np.float64)) if growths else np.zeros((len(candidates),), dtype=np.float64)

    # ===== 数据驱动的自适应目标带与内/外风险 =====
    for cand, bloom_penalty in zip(candidates, growth_norm.tolist()):
        cand["bloom_expansion"] = float(bloom_penalty)
    level_target, level_sigma = _adaptive_level_target(candidates, global_sat_ratio, dyn, cfg)
    inner_arr, outer_arr = _adaptive_inner_outer_penalties(candidates, dyn)

    # 先补齐几何一致性诊断
    for cand in candidates:
        e_try = cand["ellipse_local"]
        e_img = ((e_try[0][0] + ox, e_try[0][1] + oy), e_try[1], e_try[2])
        geo_dbg = _evaluate_candidate_geometry_consistency(e_img, K, plane_normal_camera, plane_distance, cfg)
        cand.update(geo_dbg)

    # 基于当前候选集合的稳健标准化，而不是经验阈值
    arr_resid = _robust_z_positive([c["resid"] for c in candidates])
    arr_drift = _robust_z_positive([c["drift"] for c in candidates])
    arr_axis = _robust_z_positive([c["axis_delta"] for c in candidates])
    arr_bloom = _robust_z_positive([c["bloom_expansion"] for c in candidates])
    arr_grad_bad = _robust_z_positive([1.0 - np.clip(c["boundary_grad"], 0.0, 1.5) for c in candidates])
    arr_edge_bad = _robust_z_positive([1.0 - np.clip(c["edge_drop"] / max(dyn, 1e-6), 0.0, 2.0) / 2.0 for c in candidates])
    arr_level = np.asarray([abs(c["level_norm"] - level_target) / max(level_sigma, 1e-6) for c in candidates], dtype=np.float64)
    arr_level = _robust_z_positive(arr_level)

    arr_geo_circle = _robust_z_positive([c.get("geo_circle_rms", np.nan) for c in candidates])
    arr_geo_radius = _robust_z_positive([c.get("geo_radius_cv", np.nan) for c in candidates])
    arr_geo_shift = _robust_z_positive([c.get("geo_center_shift", np.nan) for c in candidates])
    arr_geo_reproj = _robust_z_positive([c.get("geo_reproj_resid", np.nan) for c in candidates])
    arr_geo_invalid = np.asarray([1.0 - float(np.clip(c.get("geo_valid_ratio", 0.0), 0.0, 1.0)) for c in candidates], dtype=np.float64)

    for i, cand in enumerate(candidates):
        score = 0.0
        score += cfg.w_img_resid * arr_resid[i]
        score += cfg.w_img_drift * arr_drift[i]
        score += cfg.w_img_axis * arr_axis[i]
        score += cfg.w_img_level * arr_level[i]
        score += cfg.w_img_grad * arr_grad_bad[i]
        score += cfg.w_img_edge_drop * arr_edge_bad[i]
        score += cfg.w_img_bloom * arr_bloom[i]
        score += 1.15 * inner_arr[i]
        score += 0.95 * outer_arr[i]

        if cand.get("geo_enabled", False):
            score += cfg.w_geo_circle_rms * arr_geo_circle[i]
            score += cfg.w_geo_radius_cv * arr_geo_radius[i]
            score += cfg.w_geo_center_shift * arr_geo_shift[i]
            score += cfg.w_geo_reproj_resid * arr_geo_reproj[i]
            score += cfg.w_geo_valid_ratio * arr_geo_invalid[i]
        else:
            score += 3.0

        cand["effective_level_target"] = float(level_target)
        cand["effective_level_sigma"] = float(level_sigma)
        cand["penalty_too_inner"] = float(inner_arr[i])
        cand["penalty_too_outer"] = float(outer_arr[i])
        cand["global_score"] = float(score)
    chosen = min(candidates, key=lambda d: d["global_score"])
    cc_img = chosen["contour_local"].reshape(-1, 2).astype(np.float64) + np.array([ox, oy], dtype=np.float64)
    e_try = chosen["ellipse_local"]
    e_img = ((e_try[0][0] + ox, e_try[0][1] + oy), e_try[1], e_try[2])

    dbg_candidates = []
    for cand in candidates:
        dbg_candidates.append({
            "level_idx": int(cand["level_idx"]),
            "thr": int(cand["thr"]),
            "drift": float(cand["drift"]),
            "resid": float(cand["resid"]),
            "axis_delta": float(cand["axis_delta"]),
            "sat_ratio": float(cand["sat_ratio"]),
            "plateau_core_ratio": float(cand["plateau_core_ratio"]),
            "halo_low_ratio": float(cand["halo_low_ratio"]),
            "boundary_grad": float(cand["boundary_grad"]),
            "edge_drop": float(cand["edge_drop"]),
            "level_norm": float(cand["level_norm"]),
            "stability": float(cand["stability"]),
            "bloom_expansion": float(cand["bloom_expansion"]),
            "geo_enabled": bool(cand.get("geo_enabled", False)),
            "geo_valid_ratio": float(cand.get("geo_valid_ratio", np.nan)),
            "geo_circle_rms": float(cand.get("geo_circle_rms", np.nan)),
            "geo_radius_cv": float(cand.get("geo_radius_cv", np.nan)),
            "geo_center_shift": float(cand.get("geo_center_shift", np.nan)),
            "geo_reproj_resid": float(cand.get("geo_reproj_resid", np.nan)),
            "global_score": float(cand["global_score"]),
        })

    rep_dbg = {
        "enabled": True,
        "used": True,
        "reason": "geometry_consistent_candidate_selection",
        "config": asdict(cfg),
        "global_saturation_ratio": float(global_sat_ratio),
        "local_saturation_threshold": float(sat_thr),
        "num_levels": int(levels.size),
        "num_candidates": int(len(candidates)),
        "chosen_thr": int(chosen["thr"]),
        "chosen_level_idx": int(chosen["level_idx"]),
        "chosen_drift": float(chosen["drift"]),
        "chosen_resid": float(chosen["resid"]),
        "chosen_axis_delta": float(chosen["axis_delta"]),
        "chosen_sat_ratio": float(chosen["sat_ratio"]),
        "chosen_plateau_core_ratio": float(chosen["plateau_core_ratio"]),
        "chosen_halo_low_ratio": float(chosen["halo_low_ratio"]),
        "chosen_boundary_grad": float(chosen["boundary_grad"]),
        "chosen_edge_drop": float(chosen["edge_drop"]),
        "chosen_bloom_expansion": float(chosen["bloom_expansion"]),
        "chosen_level_norm": float(chosen["level_norm"]),
        "chosen_geo_enabled": bool(chosen.get("geo_enabled", False)),
        "chosen_geo_valid_ratio": float(chosen.get("geo_valid_ratio", np.nan)),
        "chosen_geo_circle_rms": float(chosen.get("geo_circle_rms", np.nan)),
        "chosen_geo_radius_cv": float(chosen.get("geo_radius_cv", np.nan)),
        "chosen_geo_center_shift": float(chosen.get("geo_center_shift", np.nan)),
        "chosen_geo_reproj_resid": float(chosen.get("geo_reproj_resid", np.nan)),
        "candidates": dbg_candidates,
    }
    return e_img, cc_img.reshape(-1, 1, 2).astype(np.int32), rep_dbg


def choose_measurement_representation(gray, contour, ellipse, K=None, plane_normal_camera=None, plane_distance=None, config=None):
    return choose_geometry_consistent_measurement_representation(
        gray=gray,
        contour=contour,
        ellipse=ellipse,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        config=config,
    )


def example_integration_call(gray_light_proc, contour, ellipse, K, plane_normal_camera, plane_distance):
    return choose_measurement_representation(
        gray=gray_light_proc,
        contour=contour,
        ellipse=ellipse,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        config=GeometryEdgeSelectorConfig(),
    )
