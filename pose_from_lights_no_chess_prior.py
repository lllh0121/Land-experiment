import argparse
import itertools
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


# Known physical target geometry. These are target-frame coordinates, not a pose prior.
CIRCLE_CENTERS_W = np.array(
    [
        [-25.0, 275.0, 0.0],
        [375.0, 275.0, 0.0],
        [-25.0, 675.0, 0.0],
        [375.0, 675.0, 0.0],
    ],
    dtype=np.float64,
)
LIGHT_RADIUS_MM = 15.0


def load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.asarray(data.get("new_K", data.get("K")), dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data.get("dist", data.get("D", np.zeros(5))), dtype=np.float64).reshape(-1)
    return K, dist, data


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
    val = np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0)
    return float(np.median(val))


def ellipse_area(ellipse):
    if ellipse is None:
        return 0.0
    (_, _), (major, minor), _ = ellipse
    return float(np.pi * max(major, 0.0) * max(minor, 0.0) * 0.25)


def sample_contour(contour, max_points=96):
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False)
    return pts[np.round(idx).astype(np.int64)]


def nms_candidates(candidates, center_radius=18.0):
    kept = []
    for cand in sorted(candidates, key=lambda x: x["score"]):
        center = np.asarray(cand["center"], dtype=np.float64)
        duplicate = False
        for old in kept:
            old_center = np.asarray(old["center"], dtype=np.float64)
            old_axes = np.asarray(old["axes"], dtype=np.float64)
            radius = max(center_radius, 0.35 * float(np.max(old_axes)))
            if np.linalg.norm(center - old_center) <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def detect_light_candidates(image, roi_mask=None, max_candidates=14):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
    h, w = gray.shape[:2]

    valid_mask = np.ones_like(gray, dtype=np.uint8) * 255 if roi_mask is None else roi_mask.copy()
    valid_vals = gray[valid_mask > 0]
    if valid_vals.size == 0:
        raise RuntimeError("ROI mask is empty; no pixels are available for detection.")

    otsu_thr, _ = cv2.threshold(
        valid_vals.reshape(-1, 1).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_thr = float(otsu_thr)
    roi_mean = float(np.mean(valid_vals))

    min_area = max(18.0, 0.00001 * h * w)
    max_area = max(4000.0, 0.08 * h * w)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = []

    bw = np.zeros_like(gray, dtype=np.uint8)
    bw[(gray >= otsu_thr) & (valid_mask > 0)] = 255
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
        if not (0.0 <= cx < w and 0.0 <= cy < h):
            continue

        pts = contour.reshape(-1, 2)
        resid = ellipse_residual(ellipse, pts)
        x, y, ww, hh = cv2.boundingRect(contour)
        local_mask = np.zeros((hh, ww), dtype=np.uint8)
        local_contour = contour.copy()
        local_contour[:, :, 0] -= x
        local_contour[:, :, 1] -= y
        cv2.drawContours(local_mask, [local_contour], -1, 255, thickness=-1)
        pix = gray[y : y + hh, x : x + ww][local_mask > 0]
        mean_brightness = float(np.mean(pix)) if pix.size else 0.0
        peak_brightness = float(np.max(pix)) if pix.size else 0.0
        circularity = float(4.0 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1e-6))
        axis_ratio = float(max(axes) / max(min(axes), 1e-6))
        score = (
            2.0 * resid
            + 0.15 * abs(math.log(max(axis_ratio, 1e-6)))
            - 0.015 * (mean_brightness - roi_mean)
            - 0.010 * (peak_brightness - otsu_thr)
            - 0.20 * circularity
        )
        candidates.append(
            {
                "center": np.array([cx, cy], dtype=np.float64),
                "ellipse": ellipse,
                "axes": axes,
                "angle_deg": float(angle),
                "area": area,
                "threshold": int(round(otsu_thr)),
                "threshold_method": "roi_otsu",
                "level": 0,
                "mean_brightness": mean_brightness,
                "peak_brightness": peak_brightness,
                "ellipse_residual": resid,
                "axis_ratio": axis_ratio,
                "contour": contour.copy(),
                "sampled_contour": sample_contour(contour),
                "score": float(score),
            }
        )

    candidates = nms_candidates(candidates)
    return candidates[:max_candidates]


def project_points(K, rvec, tvec, points_w):
    points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    img, _ = cv2.projectPoints(points_w, rvec, tvec, K, np.zeros(5))
    return img.reshape(-1, 2)


def project_world_circle_points(K, rvec, tvec, center_w, radius_mm=LIGHT_RADIUS_MM, samples=240):
    ts = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False, dtype=np.float64)
    center_w = np.asarray(center_w, dtype=np.float64).reshape(3)
    circle_w = np.column_stack(
        [
            center_w[0] + radius_mm * np.cos(ts),
            center_w[1] + radius_mm * np.sin(ts),
            np.full_like(ts, center_w[2]),
        ]
    )
    return project_points(K, rvec, tvec, circle_w)


def fit_projected_circle_ellipse(K, rvec, tvec, center_w, radius_mm=LIGHT_RADIUS_MM):
    pts = project_world_circle_points(K, rvec, tvec, center_w, radius_mm=radius_mm, samples=240)
    if len(pts) < 6:
        return None, pts
    try:
        ellipse = cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        ellipse = None
    return ellipse, pts


def positive_depth_margin(rvec, tvec, points_w):
    R, _ = cv2.Rodrigues(rvec)
    pts_c = (R @ np.asarray(points_w, dtype=np.float64).reshape(-1, 3).T + tvec.reshape(3, 1)).T
    return float(np.min(pts_c[:, 2]))


def plane_from_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    n_c = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_c /= max(np.linalg.norm(n_c), 1e-12)
    d_c = -float(np.dot(n_c, t))
    return R, t, n_c, d_c


def backproject_pixels_to_target_plane(uv, Kinv, R, t, n_c, d_c):
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    rays = (Kinv @ hom.T).T
    denom = rays @ n_c
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-10)
    scales = np.full(len(pts), np.nan, dtype=np.float64)
    scales[valid] = -d_c / denom[valid]
    valid &= scales > 0.0
    pts_c = rays * scales[:, None]
    pts_w = (R.T @ (pts_c - t.reshape(1, 3)).T).T
    return pts_w, valid


def circle_boundary_residuals(rvec, tvec, K, assignments, candidates, robust_clip=8.0, require_pose_guided_edges=False):
    Kinv = np.linalg.inv(K)
    R, t, n_c, d_c = plane_from_pose(rvec, tvec)
    all_residuals = []
    per_light = []

    for obj_idx, cand_idx in assignments:
        cand = candidates[cand_idx]
        boundary_debug = cand.get("boundary_debug", {})
        if require_pose_guided_edges and not (isinstance(boundary_debug, dict) and boundary_debug.get("used")):
            per_light.append({"valid_points": 0, "median_radius_error_mm": None, "skipped": True})
            continue
        pts_img = cand["sampled_contour"]
        pts_w, valid = backproject_pixels_to_target_plane(pts_img, Kinv, R, t, n_c, d_c)
        pts_w = pts_w[valid]
        if len(pts_w) < 8:
            all_residuals.append(np.full(8, robust_clip, dtype=np.float64))
            per_light.append({"valid_points": int(len(pts_w)), "median_radius_error_mm": None})
            continue
        xy = pts_w[:, :2]
        center_xy = CIRCLE_CENTERS_W[obj_idx, :2]
        radial = np.linalg.norm(xy - center_xy.reshape(1, 2), axis=1)
        res = radial - LIGHT_RADIUS_MM
        res = np.clip(res, -robust_clip, robust_clip)
        all_residuals.append(res)
        per_light.append(
            {
                "valid_points": int(len(pts_w)),
                "median_radius_error_mm": float(np.median(np.abs(res))),
                "mean_radius_error_mm": float(np.mean(np.abs(res))),
            }
        )

    if not all_residuals:
        return np.empty((0,), dtype=np.float64), per_light
    return np.concatenate(all_residuals).astype(np.float64), per_light


def pose_score(rvec, tvec, K, assignments, candidates):
    if positive_depth_margin(rvec, tvec, CIRCLE_CENTERS_W) <= 1.0:
        return np.inf, {}

    obs = np.array([candidates[cand_idx]["center"] for _, cand_idx in assignments], dtype=np.float64)
    obj = np.array([CIRCLE_CENTERS_W[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
    pred = project_points(K, rvec, tvec, obj)
    reproj = np.linalg.norm(pred - obs, axis=1)
    has_pose_guided_edges = any(
        isinstance(candidates[cand_idx].get("boundary_debug", {}), dict)
        and candidates[cand_idx].get("boundary_debug", {}).get("used")
        for _, cand_idx in assignments
    )
    per_light = []
    boundary_cost = 0.0
    if has_pose_guided_edges:
        boundary_res, per_light = circle_boundary_residuals(
            rvec, tvec, K, assignments, candidates, require_pose_guided_edges=True
        )
        boundary_cost = float(np.median(np.abs(boundary_res))) if boundary_res.size else 0.0
    score = float(np.median(reproj) + 2.5 * boundary_cost)
    return score, {
        "median_reprojection_px": float(np.median(reproj)),
        "max_reprojection_px": float(np.max(reproj)),
        "median_boundary_radius_error_mm": boundary_cost if has_pose_guided_edges else None,
        "per_light_boundary": per_light,
    }


def solve_pnp_candidates(obj_pts, img_pts, K):
    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(cv2.SOLVEPNP_IPPE)
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags.append(cv2.SOLVEPNP_SQPNP)
    flags.extend([cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE])

    solutions = []
    for flag in flags:
        try:
            if flag == cv2.SOLVEPNP_ITERATIVE:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                if ok:
                    solutions.append((rvec.reshape(3), tvec.reshape(3), int(flag)))
            else:
                ret = cv2.solvePnPGeneric(obj_pts, img_pts, K, np.zeros(5), flags=flag)
                ok = bool(ret[0])
                if not ok:
                    continue
                rvecs = ret[1]
                tvecs = ret[2]
                for rvec, tvec in zip(rvecs, tvecs):
                    solutions.append((np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3), int(flag)))
        except cv2.error:
            continue
    return solutions


def edge_samples_near_projected_circle(gray, projected_circle, max_points=160):
    projected_circle = np.asarray(projected_circle, dtype=np.float64).reshape(-1, 2)
    if projected_circle.shape[0] < 8:
        return None, {"used": False, "reason": "invalid_projected_circle"}

    h, w = gray.shape[:2]
    x0 = max(0, int(np.floor(np.min(projected_circle[:, 0]) - 18)))
    y0 = max(0, int(np.floor(np.min(projected_circle[:, 1]) - 18)))
    x1 = min(w, int(np.ceil(np.max(projected_circle[:, 0]) + 18)))
    y1 = min(h, int(np.ceil(np.max(projected_circle[:, 1]) + 18)))
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
        return None, {"used": False, "reason": "few_canny_edges", "edge_count": int(len(xs))}

    edge_pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
    diff = edge_pts[:, None, :] - projected_circle[None, :, :]
    dist = np.min(np.linalg.norm(diff, axis=2), axis=1)
    median_radius_px = float(np.median(np.linalg.norm(projected_circle - np.mean(projected_circle, axis=0), axis=1)))
    near_thr = max(4.0, 0.35 * median_radius_px)
    near = dist <= near_thr
    near_count = int(np.count_nonzero(near))
    if near_count < 12:
        return None, {
            "used": False,
            "reason": "no_observed_edge_close_to_physical_circle_projection",
            "edge_count": int(len(edge_pts)),
            "near_count": near_count,
            "median_distance_to_projected_circle_px": float(np.median(dist)),
            "near_threshold_px": float(near_thr),
        }

    order = np.where(near)[0]
    order = order[np.argsort(dist[order])]
    order = order[:max_points]

    pts = edge_pts[order]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False)
        pts = pts[np.round(idx).astype(np.int64)]
    if len(pts) < 12:
        return None, {"used": False, "reason": "too_few_near_edges", "edge_count": int(len(edge_pts))}

    return pts.astype(np.float64), {
        "used": True,
        "source": "pose_guided_image_edges",
        "edge_count": int(len(edge_pts)),
        "near_count": near_count,
        "selected_count": int(len(pts)),
        "median_distance_to_projected_circle_px": float(np.median(dist[order])),
        "near_threshold_px": float(near_thr),
    }


def enrich_candidates_with_pose_guided_boundaries(image, candidates, assignments, K, rvec, tvec):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    enriched = []
    for cand in candidates:
        item = dict(cand)
        item["boundary_debug"] = {"used": False, "reason": "not_assigned"}
        enriched.append(item)

    for obj_idx, cand_idx in assignments:
        projected_ellipse, projected_circle = fit_projected_circle_ellipse(K, rvec, tvec, CIRCLE_CENTERS_W[obj_idx])
        samples, debug = edge_samples_near_projected_circle(gray, projected_circle)
        if samples is not None:
            enriched[cand_idx]["sampled_contour"] = samples
            enriched[cand_idx]["boundary_debug"] = debug
            enriched[cand_idx]["projected_physical_ellipse"] = projected_ellipse
        else:
            enriched[cand_idx]["boundary_debug"] = debug
    return enriched


def initialize_pose_from_candidates(candidates, K, top_k=10):
    if len(candidates) < 4:
        raise RuntimeError(f"At least 4 light candidates are required, got {len(candidates)}.")

    candidates = candidates[:top_k]
    best = None
    obj_base = CIRCLE_CENTERS_W.astype(np.float64)

    for cand_combo in itertools.combinations(range(len(candidates)), 4):
        for cand_order in itertools.permutations(cand_combo, 4):
            assignments = [(obj_idx, cand_idx) for obj_idx, cand_idx in enumerate(cand_order)]
            img_pts = np.array([candidates[cand_idx]["center"] for cand_idx in cand_order], dtype=np.float64)
            solutions = solve_pnp_candidates(obj_base, img_pts, K)
            for rvec, tvec, flag in solutions:
                score, debug = pose_score(rvec, tvec, K, assignments, candidates)
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
        raise RuntimeError("No valid positive-depth pose could be initialized from light candidates.")
    return best, candidates


def lm_refine_pose_on_circle_edges(rvec, tvec, K, assignments, candidates, iterations=18, center_px_weight=1.0):
    usable_points = 0
    for _, cand_idx in assignments:
        dbg = candidates[cand_idx].get("boundary_debug", {})
        if isinstance(dbg, dict) and dbg.get("used"):
            usable_points += int(len(candidates[cand_idx].get("sampled_contour", [])))
    if usable_points < 32:
        return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3), {
            "skipped": True,
            "reason": "insufficient_observed_edges_close_to_physical_circle_projection",
            "usable_boundary_points": int(usable_points),
            "median_boundary_radius_error_mm": None,
            "mean_boundary_radius_error_mm": None,
            "per_light_boundary": [],
        }

    p = np.concatenate([np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)])
    lam = 1e-2
    eps = np.array([1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2], dtype=np.float64)
    history = []

    def residuals_from_params(params):
        rv = params[:3]
        tv = params[3:]
        if positive_depth_margin(rv, tv, CIRCLE_CENTERS_W) <= 1.0:
            return np.full(64, 1e3, dtype=np.float64)
        boundary_res, _ = circle_boundary_residuals(rv, tv, K, assignments, candidates, require_pose_guided_edges=True)
        if boundary_res.size == 0:
            return np.full(64, 1e3, dtype=np.float64)
        obj = np.array([CIRCLE_CENTERS_W[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
        obs = np.array([candidates[cand_idx]["center"] for _, cand_idx in assignments], dtype=np.float64)
        reproj = project_points(K, rv, tv, obj)
        center_res = (reproj - obs).reshape(-1) * float(center_px_weight)
        return np.concatenate([boundary_res, center_res]).astype(np.float64)

    r = residuals_from_params(p)
    cost = float(np.mean(r * r))
    history.append({"iter": 0, "cost": cost, "lambda": lam})

    for it in range(1, iterations + 1):
        r = residuals_from_params(p)
        J = np.zeros((len(r), 6), dtype=np.float64)
        for j in range(6):
            pp = p.copy()
            pp[j] += eps[j]
            rp = residuals_from_params(pp)
            if len(rp) != len(r):
                m = min(len(rp), len(r))
                J[:m, j] = (rp[:m] - r[:m]) / eps[j]
            else:
                J[:, j] = (rp - r) / eps[j]

        H = J.T @ J
        g = J.T @ r
        diag = np.diag(np.maximum(np.diag(H), 1.0))
        try:
            delta = -np.linalg.solve(H + lam * diag, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            history.append({"iter": it, "cost": cost, "lambda": lam, "accepted": False, "reason": "singular"})
            continue

        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > 200.0:
            lam *= 10.0
            history.append({"iter": it, "cost": cost, "lambda": lam, "accepted": False, "reason": "bad_delta"})
            continue

        trial = p + delta
        rt = residuals_from_params(trial)
        trial_cost = float(np.mean(rt * rt))
        if np.isfinite(trial_cost) and trial_cost < cost:
            p = trial
            cost = trial_cost
            lam = max(lam * 0.35, 1e-7)
            accepted = True
        else:
            lam = min(lam * 4.0, 1e8)
            accepted = False
        history.append({"iter": it, "cost": cost, "trial_cost": trial_cost, "lambda": lam, "accepted": accepted})

    final_rvec = p[:3].copy()
    final_tvec = p[3:].copy()
    final_res, per_light = circle_boundary_residuals(
        final_rvec, final_tvec, K, assignments, candidates, require_pose_guided_edges=True
    )
    return final_rvec, final_tvec, {
        "history": history,
        "center_px_weight": float(center_px_weight),
        "median_boundary_radius_error_mm": float(np.median(np.abs(final_res))) if final_res.size else None,
        "mean_boundary_radius_error_mm": float(np.mean(np.abs(final_res))) if final_res.size else None,
        "per_light_boundary": per_light,
    }


def final_pose_metrics(rvec, tvec, K, assignments, candidates):
    obj = np.array([CIRCLE_CENTERS_W[obj_idx] for obj_idx, _ in assignments], dtype=np.float64)
    obs = np.array([candidates[cand_idx]["center"] for _, cand_idx in assignments], dtype=np.float64)
    proj = project_points(K, rvec, tvec, obj)
    center_err = np.linalg.norm(proj - obs, axis=1)
    boundary_res, per_light = circle_boundary_residuals(
        rvec, tvec, K, assignments, candidates, require_pose_guided_edges=True
    )
    return {
        "projected_centers_px": proj,
        "candidate_center_observations_px": obs,
        "center_reprojection_errors_px": center_err,
        "median_center_reprojection_px": float(np.median(center_err)),
        "max_center_reprojection_px": float(np.max(center_err)),
        "median_boundary_radius_error_mm": float(np.median(np.abs(boundary_res))) if boundary_res.size else None,
        "mean_boundary_radius_error_mm": float(np.mean(np.abs(boundary_res))) if boundary_res.size else None,
        "positive_depth_margin_mm": positive_depth_margin(rvec, tvec, CIRCLE_CENTERS_W),
        "per_light_boundary": per_light,
    }


def draw_visualization(image, candidates, assignments, K, rvec, tvec, output_path):
    vis = image.copy()
    proj = project_points(K, rvec, tvec, CIRCLE_CENTERS_W)
    colors = [(0, 80, 255), (0, 210, 0), (255, 80, 0), (230, 0, 230)]

    for idx, cand in enumerate(candidates):
        color = (100, 100, 100)
        cv2.ellipse(vis, cand["ellipse"], color, 1, cv2.LINE_AA)
        c = tuple(np.round(cand["center"]).astype(int))
        cv2.circle(vis, c, 3, color, -1, cv2.LINE_AA)

    for obj_idx, cand_idx in assignments:
        color = colors[obj_idx % len(colors)]
        cand = candidates[cand_idx]
        cv2.ellipse(vis, cand["ellipse"], color, 2, cv2.LINE_AA)
        obs = tuple(np.round(cand["center"]).astype(int))
        prj = tuple(np.round(proj[obj_idx]).astype(int))
        cv2.circle(vis, obs, 5, color, -1, cv2.LINE_AA)
        cv2.circle(vis, prj, 7, color, 2, cv2.LINE_AA)
        cv2.line(vis, obs, prj, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"L{obj_idx}", (prj[0] + 8, prj[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        projected_ellipse, _ = fit_projected_circle_ellipse(K, rvec, tvec, CIRCLE_CENTERS_W[obj_idx])
        if projected_ellipse is not None:
            cv2.ellipse(vis, projected_ellipse, color, 1, cv2.LINE_AA)
        for pt in np.asarray(cand.get("sampled_contour", []), dtype=np.float64).reshape(-1, 2)[::4]:
            q = tuple(np.round(pt).astype(int))
            if 0 <= q[0] < vis.shape[1] and 0 <= q[1] < vis.shape[0]:
                cv2.circle(vis, q, 1, color, -1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)


def load_roi_mask(path, image_shape):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("roi_points", data.get("points", data))
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def run_single(light_path, cam_path, save_dir=None, roi_json=None, vis=False, max_candidates=14):
    image = cv2.imread(str(light_path))
    if image is None:
        raise RuntimeError(f"Could not read light image: {light_path}")
    K, dist, cam_data = load_intrinsics(cam_path)
    if dist.size and np.max(np.abs(dist)) > 1e-12:
        # The current dataset normally uses rectified images. If nonzero distortion is supplied,
        # undistort once so the rest of the geometry remains pinhole-only.
        image = cv2.undistort(image, K, dist)
    roi_mask = load_roi_mask(roi_json, image.shape) if roi_json else None

    candidates = detect_light_candidates(image, roi_mask=roi_mask, max_candidates=max_candidates)
    init, used_candidates = initialize_pose_from_candidates(candidates, K, top_k=min(max_candidates, len(candidates)))
    used_candidates = enrich_candidates_with_pose_guided_boundaries(
        image, used_candidates, init["assignments"], K, init["rvec"], init["tvec"]
    )
    rvec_refined, tvec_refined, refine_debug = lm_refine_pose_on_circle_edges(
        init["rvec"], init["tvec"], K, init["assignments"], used_candidates
    )
    metrics = final_pose_metrics(rvec_refined, tvec_refined, K, init["assignments"], used_candidates)

    results = {
        "algorithm": "light_only_boundary_constrained_pose_v1",
        "logic_contract": {
            "uses_chessboard_prior": False,
            "uses_external_pose_prior": False,
            "pose_source": "estimated_from_light_image_candidates_then_refined_on_circle_boundary_residuals",
            "allowed_inputs": ["light image", "camera intrinsics", "known 3D lamp layout", "known lamp radius", "optional ROI mask"],
            "forbidden_inputs": ["chessboard image", "chessboard solvePnP pose", "chessboard-projected lamp centers"],
        },
        "inputs": {
            "light_image": str(light_path),
            "camera_intrinsics": str(cam_path),
            "roi_json": None if roi_json is None else str(roi_json),
        },
        "target_geometry": {
            "circle_centers_w_mm": CIRCLE_CENTERS_W,
            "light_radius_mm": LIGHT_RADIUS_MM,
        },
        "camera": {"K": K, "intrinsics_raw": cam_data},
        "detections": [
            {
                "candidate_index": i,
                "center_px": cand["center"],
                "axes_px": cand["axes"],
                "angle_deg": cand["angle_deg"],
                "area_px": cand["area"],
                "threshold": cand["threshold"],
                "mean_brightness": cand["mean_brightness"],
                "peak_brightness": cand["peak_brightness"],
                "ellipse_residual": cand["ellipse_residual"],
                "score": cand["score"],
                "boundary_debug": cand.get("boundary_debug"),
            }
            for i, cand in enumerate(used_candidates)
        ],
        "assignment": [
            {
                "object_index": int(obj_idx),
                "candidate_index": int(cand_idx),
                "object_point_w_mm": CIRCLE_CENTERS_W[obj_idx],
                "candidate_center_px": used_candidates[cand_idx]["center"],
            }
            for obj_idx, cand_idx in init["assignments"]
        ],
        "initial_pose": {
            "rvec": init["rvec"],
            "tvec": init["tvec"],
            "score": init["score"],
            "solver_flag": init["solver_flag"],
            "debug": init["debug"],
        },
        "refined_pose": {
            "rvec": rvec_refined,
            "tvec": tvec_refined,
            "R": cv2.Rodrigues(rvec_refined)[0],
            "debug": refine_debug,
        },
        "metrics": metrics,
    }

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(light_path).stem
        json_path = out_dir / f"{stem}_light_only_pose.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(results), f, indent=2, ensure_ascii=False)
        results["output_json"] = str(json_path)
        if vis:
            vis_path = out_dir / f"{stem}_light_only_pose_vis.jpg"
            draw_visualization(image, used_candidates, init["assignments"], K, rvec_refined, tvec_refined, vis_path)
            results["output_vis"] = str(vis_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Estimate target pose from light images only; no chessboard pose or chessboard projections are used."
    )
    parser.add_argument("--light", required=True, help="Light image path.")
    parser.add_argument("--cam", required=True, help="Camera intrinsics JSON. For rectified images, use new_K and zero distortion.")
    parser.add_argument("--save_dir", default=None, help="Optional output directory for JSON and visualization.")
    parser.add_argument("--roi_json", default=None, help="Optional polygon ROI JSON; independent of chessboard pose.")
    parser.add_argument("--max_candidates", type=int, default=14)
    parser.add_argument("--vis", action="store_true", help="Save an overlay visualization.")
    parser.add_argument("--chess", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.chess:
        raise SystemExit("This light-only pipeline refuses --chess by design. Use chessboard data only in a separate evaluator.")

    results = run_single(
        light_path=args.light,
        cam_path=args.cam,
        save_dir=args.save_dir,
        roi_json=args.roi_json,
        vis=args.vis,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(to_jsonable({
        "algorithm": results["algorithm"],
        "uses_chessboard_prior": results["logic_contract"]["uses_chessboard_prior"],
        "initial_score": results["initial_pose"]["score"],
        "median_center_reprojection_px": results["metrics"]["median_center_reprojection_px"],
        "median_boundary_radius_error_mm": results["metrics"]["median_boundary_radius_error_mm"],
        "output_json": results.get("output_json"),
        "output_vis": results.get("output_vis"),
    }), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
