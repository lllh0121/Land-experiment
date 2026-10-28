import argparse
import itertools
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


# Target-frame geometry, in millimetres. This is mechanical layout data,
# not a camera-pose prior.
DEFAULT_LIGHT_CENTERS_MM = np.array(
    [
        [-25.0, 275.0, 0.0],
        [375.0, 275.0, 0.0],
        [-25.0, 675.0, 0.0],
        [375.0, 675.0, 0.0],
    ],
    dtype=np.float64,
)
DEFAULT_LIGHT_RADIUS_MM = 15.0
EDGE_MODE = os.environ.get("LIGHT_ONLY_EDGE_MODE", "profile").strip().lower()


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("new_K", "camera_matrix", "K", "mtx", "intrinsic_matrix"):
        if key in data:
            K = np.asarray(data[key], dtype=np.float64).reshape(3, 3)
            break
    else:
        if all(k in data for k in ("fx", "fy", "cx", "cy")):
            K = np.array(
                [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        else:
            raise ValueError("Cannot find camera intrinsics in JSON.")
    dist = np.asarray(data.get("dist", data.get("D", np.zeros(5))), dtype=np.float64).reshape(-1)
    return K, dist, data


def load_layout(path=None, radius_mm=DEFAULT_LIGHT_RADIUS_MM):
    if not path:
        return DEFAULT_LIGHT_CENTERS_MM.copy(), float(radius_mm)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    centers = data.get("light_centers_mm", data.get("centers_mm", data.get("centers")))
    if centers is None:
        raise ValueError("Layout JSON must contain light_centers_mm / centers_mm / centers.")
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if len(centers) < 4:
        raise ValueError("At least four light centers are required.")
    radius = float(data.get("light_radius_mm", data.get("radius_mm", radius_mm)))
    return centers, radius


def sample_contour(contour, max_points=128):
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False)
    return pts[np.round(idx).astype(np.int64)]


def ellipse_residual(ellipse, points):
    if ellipse is None or points is None or len(points) < 6:
        return np.inf
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    (cx, cy), (major, minor), angle = ellipse
    a = max(float(major) * 0.5, 1e-6)
    b = max(float(minor) * 0.5, 1e-6)
    theta = math.radians(float(angle))
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ rot
    vals = np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0)
    return float(np.median(vals))


def candidate_nms(candidates, min_dist_px=18.0):
    kept = []
    for cand in sorted(candidates, key=lambda x: x["score"]):
        center = np.asarray(cand["center_px"], dtype=np.float64)
        duplicate = False
        for old in kept:
            old_center = np.asarray(old["center_px"], dtype=np.float64)
            old_axes = np.asarray(old["axes_px"], dtype=np.float64)
            radius = max(float(min_dist_px), 0.30 * float(np.max(old_axes)))
            if np.linalg.norm(center - old_center) <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def attach_candidate_quality_priors(candidates):
    """Add image-domain priors that reject small halo fragments before PnP scoring."""
    if not candidates:
        return candidates
    out = [dict(cand) for cand in candidates]
    areas = np.asarray([max(float(c.get("area_px", 0.0)), 1.0) for c in out], dtype=np.float64)
    axes = np.asarray([np.sort(np.asarray(c.get("axes_px", [1.0, 1.0]), dtype=np.float64)) for c in out], dtype=np.float64)
    order_area = np.argsort(areas)[::-1]
    top = order_area[: min(4, len(order_area))]
    area_ref = float(np.median(areas[top])) if top.size else float(np.median(areas))
    minor_ref = float(np.median(axes[top, 0])) if top.size else float(np.median(axes[:, 0]))
    major_ref = float(np.median(axes[top, 1])) if top.size else float(np.median(axes[:, 1]))
    area_ref = max(area_ref, 1.0)
    minor_ref = max(minor_ref, 1.0)
    major_ref = max(major_ref, 1.0)

    score_order = np.argsort([float(c.get("score", 0.0)) for c in out])
    rank_by_score = np.empty((len(out),), dtype=np.int32)
    rank_by_score[score_order] = np.arange(len(out), dtype=np.int32)

    for i, cand in enumerate(out):
        area_ratio = float(areas[i] / area_ref)
        minor_ratio = float(axes[i, 0] / minor_ref)
        major_ratio = float(axes[i, 1] / major_ref)
        axis_ratio = float(max(cand.get("axis_ratio", axes[i, 1] / max(axes[i, 0], 1e-6)), 1.0))
        small_area_pen = max(0.0, (0.45 - area_ratio) / 0.45) ** 2
        small_minor_pen = max(0.0, (0.50 - minor_ratio) / 0.50) ** 2
        long_fragment_pen = max(0.0, math.log(axis_ratio / 1.9)) ** 2
        rank_pen = max(0, int(rank_by_score[i]) - 3) * 0.20
        prior = 9.0 * small_area_pen + 6.0 * small_minor_pen + 2.0 * long_fragment_pen + rank_pen
        cand["quality_prior"] = {
            "area_ratio_to_top4_median": area_ratio,
            "minor_ratio_to_top4_median": minor_ratio,
            "major_ratio_to_top4_median": major_ratio,
            "axis_ratio": axis_ratio,
            "score_rank": int(rank_by_score[i]),
            "penalty": float(prior),
        }
    return out


def bright_core_center(gray, cand, percentile=75.0):
    contour = cand.get("contour")
    if contour is None or len(contour) < 3:
        return np.asarray(cand["center_px"], dtype=np.float64)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 1 or h <= 1:
        return np.asarray(cand["center_px"], dtype=np.float64)
    local_contour = contour.copy()
    local_contour[:, :, 0] -= x
    local_contour[:, :, 1] -= y
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, thickness=-1)
    patch = gray[y : y + h, x : x + w].astype(np.float64)
    valid = mask > 0
    if not np.any(valid):
        return np.asarray(cand["center_px"], dtype=np.float64)
    vals = patch[valid]
    thr = float(np.percentile(vals, percentile))
    weights = np.maximum(patch - thr, 0.0) * valid
    total = float(np.sum(weights))
    if total <= 1e-9:
        return np.asarray(cand["center_px"], dtype=np.float64)
    yy, xx = np.indices((h, w), dtype=np.float64)
    return np.array([x + np.sum(weights * xx) / total, y + np.sum(weights * yy) / total], dtype=np.float64)


def refine_assigned_candidate_centers_from_core(image, candidates, init, K, centers_w, radius_mm, trigger_px=0.45):
    """Use a bright-core centroid only when the coarse center geometry is not already subpixel."""
    center_med = float(init.get("debug", {}).get("median_center_reprojection_px", np.inf))
    if not np.isfinite(center_med) or center_med <= float(trigger_px):
        return init, candidates, {
            "used": False,
            "reason": "initial_light_center_fit_is_already_subpixel",
            "trigger_px": float(trigger_px),
            "initial_center_median_px": center_med,
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    refined = [dict(cand) for cand in candidates]
    shifts = []
    for _, cand_idx in init["assignments"]:
        old = np.asarray(refined[cand_idx]["center_px"], dtype=np.float64)
        new = bright_core_center(gray, refined[cand_idx], percentile=75.0)
        refined[cand_idx]["ellipse_center_px"] = old
        refined[cand_idx]["center_px"] = new
        refined[cand_idx]["center_refinement"] = {
            "method": "bright_core_centroid_p75",
            "shift_px": float(np.linalg.norm(new - old)),
        }
        shifts.append(float(np.linalg.norm(new - old)))

    best = None
    assignments = list(init["assignments"])
    img_pts = np.array([refined[cand_idx]["center_px"] for _, cand_idx in assignments], dtype=np.float64)
    for rvec, tvec, flag in solve_pnp_solutions(centers_w, img_pts, K):
        score, debug = pose_score(rvec, tvec, K, assignments, refined, centers_w, radius_mm)
        if not np.isfinite(score):
            continue
        if best is None or score < best["score"]:
            best = {
                "rvec": rvec.copy(),
                "tvec": tvec.copy(),
                "assignments": assignments,
                "score": float(score),
                "solver_flag": int(flag),
                "debug": debug,
                "coarse_pose_before_center_refinement": init,
            }
    if best is None:
        return init, candidates, {
            "used": False,
            "reason": "refined_center_pnp_failed",
            "trigger_px": float(trigger_px),
            "initial_center_median_px": center_med,
        }
    return best, refined, {
        "used": True,
        "method": "bright_core_centroid_p75",
        "trigger_px": float(trigger_px),
        "initial_center_median_px": center_med,
        "median_shift_px": float(np.median(shifts)) if shifts else 0.0,
        "max_shift_px": float(np.max(shifts)) if shifts else 0.0,
        "refined_center_median_px": float(best["debug"].get("median_center_reprojection_px", np.nan)),
    }


def roi_mask_from_json(path, image_shape):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("roi_points", data.get("points", data))
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def detect_light_candidates(image, roi_mask=None, max_candidates=14):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
    h, w = gray.shape[:2]
    valid = np.ones_like(gray, dtype=np.uint8) * 255 if roi_mask is None else roi_mask.copy()
    vals = gray[valid > 0]
    if vals.size == 0:
        raise RuntimeError("ROI mask is empty.")

    otsu_thr, _ = cv2.threshold(
        vals.reshape(-1, 1).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_thr = float(otsu_thr)
    roi_mean = float(np.mean(vals))

    min_area = max(16.0, 1e-5 * h * w)
    max_area = max(4000.0, 0.08 * h * w)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = []

    bw = np.zeros_like(gray, dtype=np.uint8)
    bw[(gray >= otsu_thr) & (valid > 0)] = 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        if len(contour) < 6:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        (cx, cy), (major, minor), angle = ellipse
        axes = np.array([major, minor], dtype=np.float64)
        if np.min(axes) < 3.0 or np.max(axes) > 0.75 * max(h, w):
            continue
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        x, y, ww, hh = cv2.boundingRect(contour)
        local_mask = np.zeros((hh, ww), dtype=np.uint8)
        local_contour = contour.copy()
        local_contour[:, :, 0] -= x
        local_contour[:, :, 1] -= y
        cv2.drawContours(local_mask, [local_contour], -1, 255, thickness=-1)
        pix = gray[y : y + hh, x : x + ww][local_mask > 0]
        resid = ellipse_residual(ellipse, contour.reshape(-1, 2))
        mean_brightness = float(np.mean(pix)) if pix.size else 0.0
        peak_brightness = float(np.max(pix)) if pix.size else 0.0
        circularity = float(4.0 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1e-6))
        axis_ratio = float(max(axes) / max(min(axes), 1e-6))
        score = (
            2.0 * resid
            + 0.12 * abs(math.log(max(axis_ratio, 1e-6)))
            - 0.015 * (mean_brightness - roi_mean)
            - 0.010 * (peak_brightness - otsu_thr)
            - 0.20 * circularity
        )
        candidates.append(
            {
                "center_px": np.array([cx, cy], dtype=np.float64),
                "ellipse": ellipse,
                "axes_px": axes,
                "angle_deg": float(angle),
                "area_px": area,
                "threshold": int(round(otsu_thr)),
                "threshold_method": "roi_otsu",
                "mean_brightness": mean_brightness,
                "peak_brightness": peak_brightness,
                "ellipse_residual": resid,
                "axis_ratio": axis_ratio,
                "contour": contour.copy(),
                "boundary_px": sample_contour(contour, max_points=128),
                "score": float(score),
            }
        )

    return attach_candidate_quality_priors(candidate_nms(candidates)[:max_candidates])


def project_points(K, rvec, tvec, points_w):
    points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    img, _ = cv2.projectPoints(points_w, np.asarray(rvec).reshape(3, 1), np.asarray(tvec).reshape(3, 1), K, np.zeros(5))
    return img.reshape(-1, 2)


def project_circle_points(K, rvec, tvec, center_w, radius_mm, samples=240):
    ts = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    center_w = np.asarray(center_w, dtype=np.float64).reshape(3)
    pts_w = np.column_stack(
        [
            center_w[0] + radius_mm * np.cos(ts),
            center_w[1] + radius_mm * np.sin(ts),
            np.full_like(ts, center_w[2]),
        ]
    )
    return project_points(K, rvec, tvec, pts_w)


def positive_depth_margin(rvec, tvec, points_w):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    pts_c = (R @ np.asarray(points_w, dtype=np.float64).reshape(-1, 3).T + t.reshape(3, 1)).T
    return float(np.min(pts_c[:, 2]))


def plane_from_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    n_c = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_c /= max(np.linalg.norm(n_c), 1e-12)
    d_c = -float(np.dot(n_c, t))
    return R, t, n_c, d_c


def backproject_pixels_to_target_plane(uv, Kinv, R, t, n_c, d_c):
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    rays = (Kinv @ np.column_stack([pts, np.ones(len(pts))]).T).T
    denom = rays @ n_c
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-10)
    scale = np.full(len(pts), np.nan, dtype=np.float64)
    scale[valid] = -d_c / denom[valid]
    valid &= scale > 0.0
    pts_c = rays * scale[:, None]
    pts_w = (R.T @ (pts_c - t.reshape(1, 3)).T).T
    return pts_w, valid


def circle_boundary_residuals(rvec, tvec, K, assignments, candidates, centers_w, radius_mm, robust_clip_mm=8.0):
    Kinv = np.linalg.inv(K)
    R, t, n_c, d_c = plane_from_pose(rvec, tvec)
    all_res = []
    per_light = []

    for obj_idx, cand_idx in assignments:
        cand = candidates[cand_idx]
        pts_img = cand.get("iter_edge_px", None)
        if pts_img is None or len(pts_img) < 8:
            pts_img = cand.get("boundary_px", [])
        pts_w, valid = backproject_pixels_to_target_plane(pts_img, Kinv, R, t, n_c, d_c)
        pts_w = pts_w[valid]
        if len(pts_w) < 8:
            fill = float(robust_clip_mm) if robust_clip_mm is not None else 20.0
            all_res.append(np.full(8, fill, dtype=np.float64))
            per_light.append({"valid_points": int(len(pts_w)), "median_radius_error_mm": None})
            continue
        radial = np.linalg.norm(pts_w[:, :2] - centers_w[obj_idx, :2].reshape(1, 2), axis=1)
        res = radial - float(radius_mm)
        if robust_clip_mm is not None:
            res = np.clip(res, -float(robust_clip_mm), float(robust_clip_mm))
        all_res.append(res)
        per_light.append(
            {
                "valid_points": int(len(pts_w)),
                "median_radius_error_mm": float(np.median(np.abs(res))),
                "mean_radius_error_mm": float(np.mean(np.abs(res))),
            }
        )

    if not all_res:
        return np.empty((0,), dtype=np.float64), per_light
    return np.concatenate(all_res).astype(np.float64), per_light


def solve_pnp_solutions(obj_pts, img_pts, K):
    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(cv2.SOLVEPNP_IPPE)
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags.append(cv2.SOLVEPNP_SQPNP)
    flags.extend([cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE])

    out = []
    for flag in flags:
        try:
            if flag == cv2.SOLVEPNP_ITERATIVE:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                if ok:
                    out.append((rvec.reshape(3), tvec.reshape(3), int(flag)))
            else:
                ret = cv2.solvePnPGeneric(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                if not bool(ret[0]):
                    continue
                for rvec, tvec in zip(ret[1], ret[2]):
                    out.append((np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3), int(flag)))
        except cv2.error:
            continue
    return out


def pose_score(rvec, tvec, K, assignments, candidates, centers_w, radius_mm):
    if positive_depth_margin(rvec, tvec, centers_w) <= 1.0:
        return np.inf, {}
    obj = np.array([centers_w[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
    obs = np.array([candidates[cand_idx]["center_px"] for _, cand_idx in assignments], dtype=np.float64)
    pred = project_points(K, rvec, tvec, obj)
    center_err = np.linalg.norm(pred - obs, axis=1)
    boundary_res, per_light = circle_boundary_residuals(
        rvec, tvec, K, assignments, candidates, centers_w, radius_mm
    )
    boundary_med = float(np.median(np.abs(boundary_res))) if boundary_res.size else 0.0
    quality_penalties = np.asarray(
        [
            float((candidates[cand_idx].get("quality_prior", {}) or {}).get("penalty", 0.0))
            for _, cand_idx in assignments
        ],
        dtype=np.float64,
    )
    quality_med = float(np.median(quality_penalties)) if quality_penalties.size else 0.0
    quality_max = float(np.max(quality_penalties)) if quality_penalties.size else 0.0
    score = float(np.median(center_err) + 2.0 * boundary_med + 8.0 * quality_med + 3.0 * quality_max)
    return score, {
        "median_center_reprojection_px": float(np.median(center_err)),
        "max_center_reprojection_px": float(np.max(center_err)),
        "median_boundary_radius_error_mm": boundary_med,
        "median_candidate_quality_penalty": quality_med,
        "max_candidate_quality_penalty": quality_max,
        "per_light_boundary": per_light,
    }


def initialize_pose(candidates, K, centers_w, radius_mm, top_k=10):
    if len(candidates) < 4:
        raise RuntimeError(f"At least 4 light candidates are required; got {len(candidates)}.")
    centers_w = np.asarray(centers_w, dtype=np.float64).reshape(-1, 3)
    if len(centers_w) != 4:
        raise ValueError("This fast initializer expects exactly four known light centers.")
    candidates = candidates[:top_k]
    best = None

    for cand_combo in itertools.combinations(range(len(candidates)), 4):
        for cand_order in itertools.permutations(cand_combo, 4):
            assignments = [(obj_idx, cand_idx) for obj_idx, cand_idx in enumerate(cand_order)]
            img_pts = np.array([candidates[cand_idx]["center_px"] for cand_idx in cand_order], dtype=np.float64)
            for rvec, tvec, flag in solve_pnp_solutions(centers_w, img_pts, K):
                score, debug = pose_score(rvec, tvec, K, assignments, candidates, centers_w, radius_mm)
                if not np.isfinite(score):
                    continue
                if best is None or score < best["score"]:
                    best = {
                        "rvec": rvec.copy(),
                        "tvec": tvec.copy(),
                        "assignments": assignments,
                        "score": float(score),
                        "solver_flag": int(flag),
                        "debug": debug,
                    }

    if best is None:
        raise RuntimeError("No valid positive-depth pose was initialized from light candidates.")
    return best, candidates


def edge_samples_near_projected_circle(gray, projected_circle_px, max_points=180):
    circle = np.asarray(projected_circle_px, dtype=np.float64).reshape(-1, 2)
    if len(circle) < 8:
        return None, {"used": False, "reason": "invalid_projected_circle"}
    if EDGE_MODE in ("profile", "normal_profile", "halfmax"):
        profile_pts, profile_debug = normal_profile_edges_near_projected_circle(gray, circle, max_points=max_points)
        if profile_pts is not None and len(profile_pts) >= 12:
            return profile_pts, profile_debug
    elif EDGE_MODE in ("none", "contour"):
        return None, {"used": False, "source": "disabled", "reason": "edge_mode_disabled"}

    h, w = gray.shape[:2]
    x0 = max(0, int(np.floor(np.min(circle[:, 0]) - 18)))
    y0 = max(0, int(np.floor(np.min(circle[:, 1]) - 18)))
    x1 = min(w, int(np.ceil(np.max(circle[:, 0]) + 18)))
    y1 = min(h, int(np.ceil(np.max(circle[:, 1]) + 18)))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return None, {"used": False, "reason": "empty_roi"}

    roi = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (3, 3), 0.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    nz = gmag[gmag > 0]
    if nz.size == 0:
        return None, {"used": False, "reason": "no_gradient"}
    high = float(np.percentile(nz, 88.0))
    low = max(5.0, 0.40 * high)
    edges = cv2.Canny(blur, low, high)
    ys, xs = np.nonzero(edges)
    if len(xs) < 12:
        return None, {"used": False, "reason": "few_edges", "edge_count": int(len(xs))}

    edge_pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    dist = np.min(np.linalg.norm(edge_pts[:, None, :] - circle[None, :, :], axis=2), axis=1)
    median_radius_px = float(np.median(np.linalg.norm(circle - np.mean(circle, axis=0), axis=1)))
    near_thr = max(8.0, 0.65 * median_radius_px)
    near = np.where(dist <= near_thr)[0]
    if len(near) < 12:
        return None, {
            "used": False,
            "reason": "no_edge_close_to_projected_circle",
            "edge_count": int(len(edge_pts)),
            "near_count": int(len(near)),
            "near_threshold_px": float(near_thr),
            "median_distance_px": float(np.median(dist)),
        }
    near = near[np.argsort(dist[near])][:max_points]
    pts = edge_pts[near]
    return pts, {
        "used": True,
        "source": "canny_near_projected_circle_fallback",
        "edge_count": int(len(edge_pts)),
        "selected_count": int(len(pts)),
        "near_threshold_px": float(near_thr),
        "median_distance_px": float(np.median(dist[near])),
    }


def _smooth_profile(vals):
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    if vals.size < 5:
        return vals.copy()
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(vals, (2, 2), mode="edge"), kernel, mode="valid")


def _bilinear_gray(gray, x, y):
    h, w = gray.shape[:2]
    if x < 0.0 or y < 0.0 or x >= w - 1 or y >= h - 1:
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = float(x - x0)
    dy = float(y - y0)
    v00 = float(gray[y0, x0])
    v10 = float(gray[y0, x1])
    v01 = float(gray[y1, x0])
    v11 = float(gray[y1, x1])
    return (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v10 + (1.0 - dx) * dy * v01 + dx * dy * v11


def normal_profile_edges_near_projected_circle(gray, projected_circle_px, max_points=180):
    """Subpixel edge extraction along normals of the projected physical circle."""
    circle = np.asarray(projected_circle_px, dtype=np.float64).reshape(-1, 2)
    if len(circle) < 16:
        return None, {"used": False, "source": "normal_profile", "reason": "few_projected_points"}
    center = np.mean(circle, axis=0)
    radius_px = float(np.median(np.linalg.norm(circle - center.reshape(1, 2), axis=1)))
    if not np.isfinite(radius_px) or radius_px < 3.0:
        return None, {"used": False, "source": "normal_profile", "reason": "invalid_projected_radius"}

    n_samples = int(np.clip(min(max_points, len(circle)), 48, 160))
    idx = np.linspace(0, len(circle) - 1, n_samples, endpoint=False)
    idx = np.round(idx).astype(np.int64)
    base_points = circle[idx]
    search_half = float(np.clip(0.55 * radius_px, 6.0, 22.0))
    s_grid = np.arange(-search_half, search_half + 0.25, 0.5, dtype=np.float64)

    edge_pts = []
    weights = []
    reject = {"low_contrast": 0, "no_crossing": 0, "outside": 0, "bad_gradient": 0}
    gray_f = np.asarray(gray, dtype=np.float64)

    for p in base_points:
        normal = p - center
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-6:
            continue
        normal /= nrm
        vals = []
        valid_s = []
        for s in s_grid:
            q = p + s * normal
            v = _bilinear_gray(gray_f, q[0], q[1])
            if v is None:
                continue
            vals.append(v)
            valid_s.append(float(s))
        if len(vals) < 9:
            reject["outside"] += 1
            continue
        vals = _smooth_profile(np.asarray(vals, dtype=np.float64))
        ss = np.asarray(valid_s, dtype=np.float64)
        inner = vals[ss <= -0.25 * search_half]
        outer = vals[ss >= 0.25 * search_half]
        if inner.size < 3 or outer.size < 3:
            reject["outside"] += 1
            continue
        inner_val = float(np.percentile(inner, 75.0))
        outer_val = float(np.percentile(outer, 25.0))
        contrast = inner_val - outer_val
        if contrast < 5.0:
            reject["low_contrast"] += 1
            continue
        half = outer_val + 0.50 * contrast
        grad = np.gradient(vals, ss)
        crossing_s = None
        crossing_grad = None
        for k in range(len(ss) - 1):
            if vals[k] >= half and vals[k + 1] <= half:
                denom = vals[k + 1] - vals[k]
                alpha = 0.0 if abs(denom) < 1e-9 else float((half - vals[k]) / denom)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                crossing_s = float(ss[k] + alpha * (ss[k + 1] - ss[k]))
                crossing_grad = float((1.0 - alpha) * grad[k] + alpha * grad[k + 1])
                break
        if crossing_s is None:
            band = np.where(np.abs(ss) <= 0.55 * search_half)[0]
            if band.size == 0:
                reject["no_crossing"] += 1
                continue
            k = int(band[np.argmin(grad[band])])
            if grad[k] >= -0.08 * contrast:
                reject["bad_gradient"] += 1
                continue
            crossing_s = float(ss[k])
            crossing_grad = float(grad[k])
        if crossing_grad is not None and crossing_grad > -0.03 * contrast:
            reject["bad_gradient"] += 1
            continue
        edge_pts.append(p + crossing_s * normal)
        weights.append(max(contrast, 1.0) * max(-float(crossing_grad or 0.0), 1.0))

    if len(edge_pts) < 12:
        return None, {
            "used": False,
            "source": "normal_profile",
            "reason": "too_few_profile_edges",
            "selected_count": int(len(edge_pts)),
            "reject": reject,
            "search_half_px": float(search_half),
        }

    edge_pts = np.asarray(edge_pts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(edge_pts) > max_points:
        order = np.argsort(weights)[::-1][:max_points]
        edge_pts = edge_pts[order]
        weights = weights[order]
    return edge_pts, {
        "used": True,
        "source": "normal_profile_halfmax_projected_circle",
        "selected_count": int(len(edge_pts)),
        "median_weight": float(np.median(weights)) if weights.size else None,
        "search_half_px": float(search_half),
        "reject": reject,
    }


def enrich_candidates_with_iterative_edges(image, candidates, assignments, K, rvec, tvec, centers_w, radius_mm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    enriched = [dict(cand) for cand in candidates]
    for cand in enriched:
        cand["iter_edge_debug"] = {"used": False, "reason": "not_assigned"}
    for obj_idx, cand_idx in assignments:
        circle = project_circle_points(K, rvec, tvec, centers_w[obj_idx], radius_mm, samples=240)
        pts, debug = edge_samples_near_projected_circle(gray, circle)
        if pts is not None:
            enriched[cand_idx]["iter_edge_px"] = pts
        enriched[cand_idx]["iter_edge_debug"] = debug
    return enriched


def lm_refine_pose(
    rvec,
    tvec,
    K,
    assignments,
    candidates,
    centers_w,
    radius_mm,
    iterations=16,
    center_sigma_px=1.5,
    boundary_sigma_mm=8.0,
    center_guard_margin_px=0.75,
):
    centers_w = np.asarray(centers_w, dtype=np.float64).reshape(-1, 3)
    obj = np.array([centers_w[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
    obs = np.array([candidates[cand_idx]["center_px"] for _, cand_idx in assignments], dtype=np.float64)
    p = np.concatenate([np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)])
    eps = np.array([1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2], dtype=np.float64)
    lam = 1e-2
    history = []

    def center_median(params):
        pred = project_points(K, params[:3], params[3:], obj)
        return float(np.median(np.linalg.norm(pred - obs, axis=1)))

    def residuals(params):
        rv = params[:3]
        tv = params[3:]
        if positive_depth_margin(rv, tv, centers_w) <= 1.0:
            return np.full(64, 1e3, dtype=np.float64)
        pred = project_points(K, rv, tv, obj)
        center_res = ((pred - obs) / max(float(center_sigma_px), 1e-6)).reshape(-1)
        boundary_res, _ = circle_boundary_residuals(
            rv, tv, K, assignments, candidates, centers_w, radius_mm, robust_clip_mm=None
        )
        if boundary_res.size:
            sigma = max(float(boundary_sigma_mm), 1e-6)
            boundary_res = boundary_res / (sigma + np.abs(boundary_res) / 6.0)
        return np.concatenate([center_res, boundary_res]).astype(np.float64)

    r = residuals(p)
    cost = float(np.mean(r * r))
    center_guard_px = center_median(p) + float(center_guard_margin_px)
    history.append({"iter": 0, "cost": cost, "lambda": lam, "center_guard_px": center_guard_px})

    for it in range(1, int(iterations) + 1):
        r = residuals(p)
        J = np.zeros((len(r), 6), dtype=np.float64)
        for j in range(6):
            pp = p.copy()
            pp[j] += eps[j]
            rp = residuals(pp)
            m = min(len(r), len(rp))
            J[:m, j] = (rp[:m] - r[:m]) / eps[j]

        H = J.T @ J
        g = J.T @ r
        diag = np.diag(np.maximum(np.diag(H), 1.0))
        try:
            delta = -np.linalg.solve(H + lam * diag, g)
        except np.linalg.LinAlgError:
            lam = min(lam * 10.0, 1e8)
            history.append({"iter": it, "cost": cost, "lambda": lam, "accepted": False, "reason": "singular"})
            continue

        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta[3:]) > 300.0 or np.linalg.norm(delta[:3]) > 0.75:
            lam = min(lam * 10.0, 1e8)
            history.append({"iter": it, "cost": cost, "lambda": lam, "accepted": False, "reason": "bad_delta"})
            continue

        trial = p + delta
        rt = residuals(trial)
        trial_cost = float(np.mean(rt * rt))
        trial_center = center_median(trial)
        if np.isfinite(trial_cost) and trial_cost < cost and trial_center <= center_guard_px:
            p = trial
            cost = trial_cost
            lam = max(lam * 0.35, 1e-7)
            accepted = True
        else:
            lam = min(lam * 4.0, 1e8)
            accepted = False
        history.append(
            {
                "iter": it,
                "cost": cost,
                "trial_cost": trial_cost,
                "trial_center_median_px": trial_center,
                "lambda": lam,
                "accepted": accepted,
                "step_norm": float(np.linalg.norm(delta)),
            }
        )

    final_res, per_light = circle_boundary_residuals(
        p[:3], p[3:], K, assignments, candidates, centers_w, radius_mm, robust_clip_mm=None
    )
    pred = project_points(K, p[:3], p[3:], obj)
    center_err = np.linalg.norm(pred - obs, axis=1)
    return p[:3].copy(), p[3:].copy(), {
        "history": history,
        "final_cost": cost,
        "iterations": int(iterations),
        "median_center_reprojection_px": float(np.median(center_err)),
        "max_center_reprojection_px": float(np.max(center_err)),
        "median_boundary_radius_error_mm": float(np.median(np.abs(final_res))) if final_res.size else None,
        "mean_boundary_radius_error_mm": float(np.mean(np.abs(final_res))) if final_res.size else None,
        "per_light_boundary": per_light,
    }


def final_metrics(rvec, tvec, K, assignments, candidates, centers_w, radius_mm):
    obj = np.array([centers_w[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
    obs = np.array([candidates[cand_idx]["center_px"] for _, cand_idx in assignments], dtype=np.float64)
    pred = project_points(K, rvec, tvec, obj)
    center_err = np.linalg.norm(pred - obs, axis=1)
    boundary_res, per_light = circle_boundary_residuals(
        rvec, tvec, K, assignments, candidates, centers_w, radius_mm, robust_clip_mm=None
    )
    return {
        "projected_centers_px": pred,
        "observed_candidate_centers_px": obs,
        "center_reprojection_errors_px": center_err,
        "median_center_reprojection_px": float(np.median(center_err)),
        "max_center_reprojection_px": float(np.max(center_err)),
        "median_boundary_radius_error_mm": float(np.median(np.abs(boundary_res))) if boundary_res.size else None,
        "mean_boundary_radius_error_mm": float(np.mean(np.abs(boundary_res))) if boundary_res.size else None,
        "positive_depth_margin_mm": positive_depth_margin(rvec, tvec, centers_w),
        "per_light_boundary": per_light,
    }


def pose_ctx_from_rt(rvec, tvec, K, centers_w, pose_sigma_px=None, debug=None):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    n_c = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_c /= max(np.linalg.norm(n_c), 1e-12)
    d_c = -float(np.dot(n_c, t))
    proj = project_points(K, rvec, tvec, centers_w)
    sigma = 1.5 if pose_sigma_px is None else float(pose_sigma_px)
    return {
        "rvec1": rvec,
        "tvec1": tvec,
        "R1": R,
        "t1": t,
        "n_c1": n_c,
        "d_c1": d_c,
        "proj_pose_final": proj,
        "proj_true_eval": proj.copy(),
        "pose_reference_eval": proj.copy(),
        "pose_sigma_px": sigma,
        "pose_reference_covariances": [(np.eye(2) * sigma * sigma).tolist() for _ in range(len(proj))],
        "pose_ref_debug": {} if debug is None else debug,
        "pose_source": "light_only_iterative_layout_radius",
    }


def estimate_pose_ctx_from_light_only(
    light_image_or_path,
    K,
    light_centers_mm=None,
    light_radius_mm=DEFAULT_LIGHT_RADIUS_MM,
    roi_mask=None,
    max_candidates=14,
    lm_iterations=16,
):
    if isinstance(light_image_or_path, (str, Path)):
        image = cv2.imread(str(light_image_or_path))
        if image is None:
            raise RuntimeError(f"Cannot read light image: {light_image_or_path}")
    else:
        image = np.asarray(light_image_or_path).copy()
    centers_w = DEFAULT_LIGHT_CENTERS_MM.copy() if light_centers_mm is None else np.asarray(light_centers_mm, dtype=np.float64).reshape(-1, 3)

    candidates = detect_light_candidates(image, roi_mask=roi_mask, max_candidates=max_candidates)
    init, candidates = initialize_pose(candidates, K, centers_w, light_radius_mm, top_k=min(max_candidates, len(candidates)))
    init, candidates, center_refine_debug = refine_assigned_candidate_centers_from_core(
        image, candidates, init, K, centers_w, light_radius_mm
    )
    candidates = enrich_candidates_with_iterative_edges(
        image, candidates, init["assignments"], K, init["rvec"], init["tvec"], centers_w, light_radius_mm
    )
    init_metrics = final_metrics(init["rvec"], init["tvec"], K, init["assignments"], candidates, centers_w, light_radius_mm)
    init_center_med = float(init_metrics["median_center_reprojection_px"])
    rollback_reason = None
    if lm_iterations > 0 and init_center_med <= 0.45:
        rvec = np.asarray(init["rvec"], dtype=np.float64).copy()
        tvec = np.asarray(init["tvec"], dtype=np.float64).copy()
        metrics = init_metrics
        refine_debug = {
            "history": [],
            "iterations": 0,
            "skipped_lm": True,
            "skip_reason": "initial_light_center_fit_is_already_subpixel",
            "initial_center_threshold_px": 0.45,
            "rollback_to_initial": True,
        }
    else:
        rvec, tvec, refine_debug = lm_refine_pose(
            init["rvec"],
            init["tvec"],
            K,
            init["assignments"],
            candidates,
            centers_w,
            light_radius_mm,
            iterations=lm_iterations,
        )
        metrics = final_metrics(rvec, tvec, K, init["assignments"], candidates, centers_w, light_radius_mm)
        boundary_med = metrics.get("median_boundary_radius_error_mm")
        init_boundary_med = init_metrics.get("median_boundary_radius_error_mm")
        if (
            lm_iterations > 0
            and boundary_med is not None
            and np.isfinite(boundary_med)
            and boundary_med > 4.25
            and (init_boundary_med is None or boundary_med >= float(init_boundary_med) - 0.10)
        ):
            rollback_reason = (
                "iterative_edge_boundary_residual_too_large; "
                "using light-center PnP initialization"
            )
            rvec = np.asarray(init["rvec"], dtype=np.float64).copy()
            tvec = np.asarray(init["tvec"], dtype=np.float64).copy()
            metrics = init_metrics
            refine_debug = dict(refine_debug)
            refine_debug["rollback_to_initial"] = True
            refine_debug["rollback_reason"] = rollback_reason
            refine_debug["rolled_back_boundary_threshold_mm"] = 4.25
        else:
            refine_debug = dict(refine_debug)
            refine_debug["rollback_to_initial"] = False
    pose_sigma_px = float(np.clip(max(0.75, 0.75 * metrics["median_center_reprojection_px"]), 0.5, 8.0))
    debug = {
        "algorithm": "light_only_iterative_layout_radius_lm_v3_core_center_guarded",
        "logic_contract": {
            "uses_checkerboard": False,
            "uses_external_pose_prior": False,
            "allowed_inputs": [
                "light image",
                "camera intrinsics",
                "known metric light-center layout",
                "known light radius",
                "optional image ROI",
            ],
        },
        "initial_pose": init,
        "center_refinement": center_refine_debug,
        "initial_metrics_after_edge_enrichment": init_metrics,
        "iterative_refine": refine_debug,
        "metrics": metrics,
        "assignment": init["assignments"],
        "rollback_reason": rollback_reason,
    }
    return pose_ctx_from_rt(rvec, tvec, K, centers_w, pose_sigma_px=pose_sigma_px, debug=debug), {
        "image": image,
        "candidates": candidates,
        "assignments": init["assignments"],
        "initial_pose": init,
        "refined_pose": {"rvec": rvec, "tvec": tvec, "R": cv2.Rodrigues(rvec)[0]},
        "metrics": metrics,
        "pose_ctx": pose_ctx_from_rt(rvec, tvec, K, centers_w, pose_sigma_px=pose_sigma_px, debug=debug),
    }


def draw_visualization(image, result, K, centers_w, radius_mm, output_path):
    vis = image.copy()
    rvec = result["refined_pose"]["rvec"]
    tvec = result["refined_pose"]["tvec"]
    proj = project_points(K, rvec, tvec, centers_w)
    colors = [(0, 80, 255), (0, 210, 0), (255, 80, 0), (230, 0, 230)]

    for cand in result["candidates"]:
        cv2.ellipse(vis, cand["ellipse"], (110, 110, 110), 1, cv2.LINE_AA)
        c = tuple(np.round(cand["center_px"]).astype(int))
        cv2.circle(vis, c, 3, (110, 110, 110), -1, cv2.LINE_AA)

    for obj_idx, cand_idx in result["assignments"]:
        color = colors[obj_idx % len(colors)]
        cand = result["candidates"][cand_idx]
        cv2.ellipse(vis, cand["ellipse"], color, 2, cv2.LINE_AA)
        obs = tuple(np.round(cand["center_px"]).astype(int))
        prj = tuple(np.round(proj[obj_idx]).astype(int))
        cv2.circle(vis, obs, 5, color, -1, cv2.LINE_AA)
        cv2.circle(vis, prj, 7, color, 2, cv2.LINE_AA)
        cv2.line(vis, obs, prj, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"L{obj_idx}", (prj[0] + 8, prj[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        circle = project_circle_points(K, rvec, tvec, centers_w[obj_idx], radius_mm, samples=240)
        if len(circle) >= 6:
            try:
                ell = cv2.fitEllipse(circle.astype(np.float32).reshape(-1, 1, 2))
                cv2.ellipse(vis, ell, color, 1, cv2.LINE_AA)
            except cv2.error:
                pass
        for pt in np.asarray(cand.get("iter_edge_px", []), dtype=np.float64).reshape(-1, 2)[::5]:
            q = tuple(np.round(pt).astype(int))
            if 0 <= q[0] < vis.shape[1] and 0 <= q[1] < vis.shape[0]:
                cv2.circle(vis, q, 1, color, -1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)


def run_single(args):
    image = cv2.imread(str(args.light))
    if image is None:
        raise RuntimeError(f"Cannot read light image: {args.light}")
    K, dist, cam_data = load_intrinsics(args.cam)
    if dist.size and np.max(np.abs(dist)) > 1e-12:
        image = cv2.undistort(image, K, dist)
    centers_w, radius_mm = load_layout(args.layout_json, radius_mm=args.radius_mm)
    roi_mask = roi_mask_from_json(args.roi_json, image.shape) if args.roi_json else None

    pose_ctx, result = estimate_pose_ctx_from_light_only(
        image,
        K,
        light_centers_mm=centers_w,
        light_radius_mm=radius_mm,
        roi_mask=roi_mask,
        max_candidates=args.max_candidates,
        lm_iterations=args.lm_iterations,
    )

    output = {
        "algorithm": "light_only_iterative_layout_radius_lm_v1",
        "logic_contract": {
            "uses_checkerboard": False,
            "uses_external_pose_prior": False,
            "pose_source": "estimated_from_light_image_known_layout_known_radius",
            "allowed_inputs": ["light image", "camera intrinsics", "known light-center layout", "known light radius", "optional ROI"],
        },
        "inputs": {
            "light_image": str(args.light),
            "camera_intrinsics": str(args.cam),
            "layout_json": None if args.layout_json is None else str(args.layout_json),
            "roi_json": None if args.roi_json is None else str(args.roi_json),
        },
        "camera": {"K": K, "intrinsics_raw": cam_data},
        "target_geometry": {"light_centers_mm": centers_w, "light_radius_mm": float(radius_mm)},
        "detections": [
            {
                "candidate_index": i,
                "center_px": cand["center_px"],
                "axes_px": cand["axes_px"],
                "angle_deg": cand["angle_deg"],
                "area_px": cand["area_px"],
                "score": cand["score"],
                "iter_edge_debug": cand.get("iter_edge_debug"),
            }
            for i, cand in enumerate(result["candidates"])
        ],
        "assignment": [
            {
                "object_index": int(obj_idx),
                "candidate_index": int(cand_idx),
                "object_point_mm": centers_w[obj_idx],
                "candidate_center_px": result["candidates"][cand_idx]["center_px"],
            }
            for obj_idx, cand_idx in result["assignments"]
        ],
        "initial_pose": result["initial_pose"],
        "refined_pose": result["refined_pose"],
        "metrics": result["metrics"],
        "pose_ctx_compatible": pose_ctx,
    }

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.light).stem
        json_path = out_dir / f"{stem}_light_only_iterative_pose.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(output), f, indent=2, ensure_ascii=False)
        output["output_json"] = str(json_path)
        if args.vis:
            vis_path = out_dir / f"{stem}_light_only_iterative_pose_vis.jpg"
            draw_visualization(image, result, K, centers_w, radius_mm, vis_path)
            output["output_vis"] = str(vis_path)

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Light-only pose estimation using known lamp layout/radius and a fast LM refinement."
    )
    parser.add_argument("--light", required=True, help="Light image path.")
    parser.add_argument("--cam", required=True, help="Camera intrinsics JSON.")
    parser.add_argument("--save_dir", default=None, help="Optional output directory.")
    parser.add_argument("--layout_json", default=None, help="Optional JSON containing light_centers_mm and light_radius_mm.")
    parser.add_argument("--radius_mm", type=float, default=DEFAULT_LIGHT_RADIUS_MM, help="Known light radius in millimetres.")
    parser.add_argument("--roi_json", default=None, help="Optional image ROI JSON; it must not contain any pose.")
    parser.add_argument("--max_candidates", type=int, default=14)
    parser.add_argument("--lm_iterations", type=int, default=16)
    parser.add_argument("--vis", action="store_true", help="Save overlay visualization.")
    return parser.parse_args()


def main():
    args = parse_args()
    output = run_single(args)
    brief = {
        "algorithm": output["algorithm"],
        "uses_checkerboard": output["logic_contract"]["uses_checkerboard"],
        "initial_score": output["initial_pose"]["score"],
        "median_center_reprojection_px": output["metrics"]["median_center_reprojection_px"],
        "median_boundary_radius_error_mm": output["metrics"]["median_boundary_radius_error_mm"],
        "output_json": output.get("output_json"),
        "output_vis": output.get("output_vis"),
    }
    print(json.dumps(to_jsonable(brief), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
