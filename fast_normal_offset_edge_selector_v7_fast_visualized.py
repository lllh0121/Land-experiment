from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LightObservation:
    contour: np.ndarray
    bbox_center: np.ndarray
    gray_center: np.ndarray
    source_ellipse: Any
    multilevel_contours: List[np.ndarray]
    multilevel_ellipses: List[Any]
    feature_debug: Dict[str, Any]
    profile_bank: Dict[str, Any]
    provisional_plane_center: Optional[np.ndarray]
    provisional_plane_radius: Optional[float]
    provisional_center_image: Optional[np.ndarray]


@dataclass
class AnchorFrame:
    anchor_indices: List[int]
    provisional_plane_centers: np.ndarray
    layout_assignment: List[int]
    layout_transform_R: np.ndarray
    layout_transform_t: np.ndarray
    predicted_plane_centers: np.ndarray
    predicted_image_centers: np.ndarray
    debug: Dict[str, Any]


@dataclass
class ArcSelectionResult:
    work_contour: np.ndarray
    work_ellipse: Any
    arc_points: np.ndarray
    arc_angles: np.ndarray
    arc_score: float
    score_gap: float
    debug: Dict[str, Any]


# -----------------------------
# 基础几何工具
# -----------------------------
def _fit_ellipse_best(contour) -> Optional[Tuple]:
    pts = np.asarray(contour, dtype=np.float32)
    if len(pts) < 5:
        return None
    fns = []
    if hasattr(cv2, "fitEllipseAMS"):
        fns.append(cv2.fitEllipseAMS)
    if hasattr(cv2, "fitEllipseDirect"):
        fns.append(cv2.fitEllipseDirect)
    fns.append(cv2.fitEllipse)

    best = None
    best_res = np.inf
    for fn in fns:
        try:
            e = fn(pts)
            res = _ellipse_contour_residual(e, pts.reshape(-1, 2))
            if np.isfinite(res) and res < best_res:
                best = e
                best_res = res
        except Exception:
            pass
    return best


def _ellipse_contour_residual(ellipse, contour_pts) -> float:
    pts = np.asarray(contour_pts, dtype=np.float64).reshape(-1, 2)
    if ellipse is None or len(pts) < 5:
        return np.inf
    (cx, cy), (MA, ma), ang = ellipse
    a = max(float(MA) * 0.5, 1e-12)
    b = max(float(ma) * 0.5, 1e-12)
    th = np.deg2rad(float(ang))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    resid = np.abs((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2 - 1.0)
    return float(np.median(resid))


def _gray_centroid_in_contour(gray, contour) -> np.ndarray:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cnt = np.asarray(contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)
    vals = gray[ys, xs].astype(np.float64)
    denom = float(np.sum(vals))
    if denom <= 1e-12:
        return np.array([np.mean(xs), np.mean(ys)], dtype=np.float64)
    cx = float(np.sum(xs * vals) / denom)
    cy = float(np.sum(ys * vals) / denom)
    return np.array([cx, cy], dtype=np.float64)


def _sample_ellipse_points_and_normals(ellipse, num_samples: Optional[int] = None):
    if ellipse is None:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    (cx, cy), (MA, ma), angle_deg = ellipse
    a = max(float(MA) * 0.5, 1.0)
    b = max(float(ma) * 0.5, 1.0)
    if num_samples is None:
        perim = np.pi * (3.0 * (a + b) - np.sqrt(max((3.0 * a + b) * (a + 3.0 * b), 1e-12)))
        num_samples = int(np.clip(np.ceil(perim / 3.0), 48, 144))
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    ts = np.linspace(0.0, 2.0 * np.pi, int(num_samples), endpoint=False)
    ct = np.cos(ts)
    st = np.sin(ts)
    q = np.column_stack([a * ct, b * st])
    pts = np.array([cx, cy], dtype=np.float64).reshape(1, 2) + q @ R.T
    nrms = np.column_stack([ct / a, st / b]) @ R.T
    nrms /= (np.linalg.norm(nrms, axis=1, keepdims=True) + 1e-12)
    return pts.astype(np.float64), nrms.astype(np.float64), ts.astype(np.float64)


def _point_polygon_signed_distance(contour, point) -> float:
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    p = (float(point[0]), float(point[1]))
    return float(cv2.pointPolygonTest(cnt, p, True))


def _backproject_pixels_to_plane(points_uv, K, plane_normal_camera, plane_distance):
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    n = np.asarray(plane_normal_camera, dtype=np.float64).reshape(3)
    d = float(plane_distance)
    homog = np.column_stack([pts, np.ones((len(pts),), dtype=np.float64)])
    rays = (Kinv @ homog.T).T
    denom = rays @ n
    valid = np.abs(denom) > 1e-12
    X = np.full((len(pts), 3), np.nan, dtype=np.float64)
    if np.any(valid):
        lam = -d / denom[valid]
        X[valid] = rays[valid] * lam[:, None]
    return X, valid


def _project_plane_points_to_image(points_3d, K):
    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    proj = (np.asarray(K, dtype=np.float64) @ X.T).T
    z = np.clip(proj[:, 2:3], 1e-12, None)
    return (proj[:, :2] / z).astype(np.float64)


def _plane_basis_from_current_plane(plane_normal_camera, plane_distance):
    n = np.asarray(plane_normal_camera, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    basis = np.eye(3, dtype=np.float64)
    ref = basis[int(np.argmin(np.abs(basis @ n)))]
    e1 = np.cross(n, ref)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-12)
    origin = -float(plane_distance) * n
    return origin.astype(np.float64), e1.astype(np.float64), e2.astype(np.float64)


def _plane3d_to_plane2d(points_3d, origin, e1, e2):
    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    rel = X - origin.reshape(1, 3)
    return np.column_stack([rel @ e1, rel @ e2]).astype(np.float64)


def _plane2d_to_plane3d(points_2d, origin, e1, e2):
    X = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    return origin.reshape(1, 3) + X[:, :1] * e1.reshape(1, 3) + X[:, 1:2] * e2.reshape(1, 3)


def _free_circle_fit(points_xy) -> Dict[str, Any]:
    P = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(P) < 3:
        return {"center": None, "radius": None, "residuals": None, "mad": np.inf}
    x = P[:, 0]
    y = P[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return {"center": None, "radius": None, "residuals": None, "mad": np.inf}
    c = np.asarray(sol[:2], dtype=np.float64)
    beta = float(sol[2])
    r2 = max(beta + float(c @ c), 0.0)
    r = float(np.sqrt(r2))
    dist = np.linalg.norm(P - c.reshape(1, 2), axis=1)
    resid = dist - r
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return {"center": c, "radius": r, "residuals": resid, "mad": mad}


def _weighted_rigid_alignment(template_xy, observed_xy, weights=None):
    X = np.asarray(template_xy, dtype=np.float64).reshape(-1, 2)
    Y = np.asarray(observed_xy, dtype=np.float64).reshape(-1, 2)
    if weights is None:
        w = np.ones((len(X),), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(X):
            w = np.ones((len(X),), dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    w = w / np.sum(w)
    mx = np.sum(X * w[:, None], axis=0)
    my = np.sum(Y * w[:, None], axis=0)
    X0 = X - mx
    Y0 = Y - my
    H = X0.T @ (Y0 * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = my - R @ mx
    return R.astype(np.float64), t.astype(np.float64)


def _rank_unit(values, higher_is_better: bool = False) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros_like(v, dtype=np.float64)
    finite = np.isfinite(v)
    if np.count_nonzero(finite) <= 1:
        return out
    vals = v[finite]
    order = np.argsort(vals)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.linspace(0.0, 1.0, len(vals))
    out[finite] = 1.0 - rank if higher_is_better else rank
    return out


def _circular_smooth(arr: np.ndarray, kernel_len: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return a
    k = max(3, int(kernel_len))
    if k % 2 == 0:
        k += 1
    if a.size <= 3:
        return a.copy()
    pad = k // 2
    ap = np.r_[a[-pad:], a, a[:pad]]
    kernel = np.hanning(k).astype(np.float64)
    kernel /= np.sum(kernel)
    sm = np.convolve(ap, kernel, mode='same')[pad:-pad]
    return sm.astype(np.float64)


# -----------------------------
# 观测构建
# -----------------------------
def _build_multilevel_contours(gray, contour) -> Tuple[List[np.ndarray], List[Any], Dict[str, Any]]:
    contour = np.asarray(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return [], [], {"levels": []}

    mask = np.zeros((h, w), dtype=np.uint8)
    local = contour.copy()
    local[:, 0, 0] -= x
    local[:, 0, 1] -= y
    cv2.drawContours(mask, [local], -1, 255, thickness=-1)
    vals = roi[mask > 0]
    if vals.size < 8:
        return [], [], {"levels": []}

    q = np.linspace(0.1, 0.9, 9)
    levels = np.unique(np.quantile(vals.astype(np.float64), q).astype(np.float64))
    contours_out = []
    ellipses_out = []
    for lv in levels:
        bw = np.zeros_like(mask, dtype=np.uint8)
        bw[(roi.astype(np.float64) >= lv) & (mask > 0)] = 255
        cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        best = max(cs, key=lambda c: cv2.contourArea(c))
        best = best.copy()
        best[:, 0, 0] += x
        best[:, 0, 1] += y
        e = _fit_ellipse_best(best)
        if e is None:
            continue
        contours_out.append(best.astype(np.int32))
        ellipses_out.append(e)
    return contours_out, ellipses_out, {"levels": [float(v) for v in levels.tolist()]}


def _profile_bank_from_ellipse(gray, ellipse, contour):
    pts, nrms, ang = _sample_ellipse_points_and_normals(ellipse, None)
    if len(pts) == 0:
        return {"points": pts, "normals": nrms, "angles": ang, "profiles": []}
    signed = np.array([_point_polygon_signed_distance(contour, p) for p in pts], dtype=np.float64)
    max_abs = np.nanmax(np.abs(signed)) if np.any(np.isfinite(signed)) else 1.0
    search = np.linspace(-max_abs, max_abs, int(np.clip(np.ceil(2.0 * max_abs) + 1, 11, 81)))
    profiles = []
    for p, n in zip(pts, nrms):
        samples = []
        for d in search:
            q = p + d * n
            x = float(np.clip(q[0], 0.0, gray.shape[1] - 1.001))
            y = float(np.clip(q[1], 0.0, gray.shape[0] - 1.001))
            x0 = int(np.floor(x)); y0 = int(np.floor(y))
            x1 = min(x0 + 1, gray.shape[1] - 1); y1 = min(y0 + 1, gray.shape[0] - 1)
            dx = x - x0; dy = y - y0
            v = (
                (1.0 - dx) * (1.0 - dy) * float(gray[y0, x0])
                + dx * (1.0 - dy) * float(gray[y0, x1])
                + (1.0 - dx) * dy * float(gray[y1, x0])
                + dx * dy * float(gray[y1, x1])
            )
            samples.append(v)
        samples = np.asarray(samples, dtype=np.float64)
        profiles.append({
            "offsets": search.copy(),
            "values": samples,
            "grad": np.gradient(samples, search),
            "grad2": np.gradient(np.gradient(samples, search), search),
        })
    return {"points": pts, "normals": nrms, "angles": ang, "profiles": profiles}


def _provisional_plane_center_from_contour(contour, K, plane_normal_camera, plane_distance):
    e = _fit_ellipse_best(contour)
    if e is None:
        return None, None, None
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    X3, valid = _backproject_pixels_to_plane(pts, K, plane_normal_camera, plane_distance)
    X3 = X3[valid]
    if len(X3) < 6:
        return None, None, None
    origin, e1, e2 = _plane_basis_from_current_plane(plane_normal_camera, plane_distance)
    P2 = _plane3d_to_plane2d(X3, origin, e1, e2)
    fit = _free_circle_fit(P2)
    c2 = fit["center"]
    r = fit["radius"]
    if c2 is None:
        return None, None, None
    c3 = _plane2d_to_plane3d(np.asarray(c2, dtype=np.float64).reshape(1, 2), origin, e1, e2)[0]
    c_img = _project_plane_points_to_image(c3.reshape(1, 3), K)[0]
    return np.asarray(c2, dtype=np.float64), float(r), np.asarray(c_img, dtype=np.float64)


def build_light_observations(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
) -> List[LightObservation]:
    observations = []
    for contour in contours:
        contour = np.asarray(contour, dtype=np.int32)
        bbox = cv2.boundingRect(contour)
        bbox_center = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]], dtype=np.float64)
        gray_center = _gray_centroid_in_contour(gray_light, contour)
        source_ellipse = _fit_ellipse_best(contour)
        ml_contours, ml_ellipses, level_debug = _build_multilevel_contours(gray_light, contour)
        pb = _profile_bank_from_ellipse(gray_light, source_ellipse, contour) if source_ellipse is not None else {"points": np.empty((0, 2)), "normals": np.empty((0, 2)), "angles": np.empty((0,)), "profiles": []}
        p_center, p_radius, p_center_img = _provisional_plane_center_from_contour(contour, K, plane_normal_camera, plane_distance)

        areas = [float(cv2.contourArea(c)) for c in ml_contours] if ml_contours else [float(cv2.contourArea(contour))]
        centers = [np.asarray(e[0], dtype=np.float64) for e in ml_ellipses] if ml_ellipses else ([np.asarray(source_ellipse[0], dtype=np.float64)] if source_ellipse is not None else [bbox_center])
        layer_drift = 0.0
        if len(centers) > 1:
            base = np.median(np.asarray(centers), axis=0)
            layer_drift = float(np.median(np.linalg.norm(np.asarray(centers) - base.reshape(1, 2), axis=1)))

        mask = np.zeros(gray_light.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        vals = gray_light[mask > 0].astype(np.float64)
        energy_concentration = float(np.sum(vals ** 2) / ((np.sum(vals) ** 2) + 1e-12)) if vals.size else 0.0
        compactness = float(4.0 * np.pi * max(cv2.contourArea(contour), 1e-12) / (max(cv2.arcLength(contour, True), 1e-12) ** 2))
        area_growth = float(np.max(areas) - np.min(areas)) if len(areas) > 1 else 0.0

        obs = LightObservation(
            contour=contour,
            bbox_center=bbox_center,
            gray_center=gray_center,
            source_ellipse=source_ellipse,
            multilevel_contours=ml_contours,
            multilevel_ellipses=ml_ellipses,
            feature_debug={
                "compactness": compactness,
                "energy_concentration": energy_concentration,
                "layer_drift": layer_drift,
                "area_growth": area_growth,
                "num_levels": int(len(ml_contours)),
                "levels": level_debug["levels"],
            },
            profile_bank=pb,
            provisional_plane_center=p_center,
            provisional_plane_radius=p_radius,
            provisional_center_image=p_center_img,
        )
        observations.append(obs)
    return observations


# -----------------------------
# 锚点与参考框架
# -----------------------------
def select_anchor_indices(observations: Sequence[LightObservation]) -> List[int]:
    if not observations:
        return []
    compactness = _rank_unit([o.feature_debug.get("compactness", np.nan) for o in observations], higher_is_better=True)
    energy = _rank_unit([o.feature_debug.get("energy_concentration", np.nan) for o in observations], higher_is_better=True)
    drift = _rank_unit([o.feature_debug.get("layer_drift", np.nan) for o in observations], higher_is_better=False)
    gray_ellipse_consistency = _rank_unit([
        np.linalg.norm(np.asarray(o.gray_center) - np.asarray(o.source_ellipse[0], dtype=np.float64)) if o.source_ellipse is not None else np.inf
        for o in observations
    ], higher_is_better=False)
    score = compactness + energy + drift + gray_ellipse_consistency
    order = np.argsort(-score)
    num_anchor = min(2, len(observations))
    return [int(v) for v in order[:num_anchor].tolist()]


def build_anchor_frame(
    observations: Sequence[LightObservation],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> AnchorFrame:
    origin, e1, e2 = _plane_basis_from_current_plane(plane_normal_camera, plane_distance)
    provisional = []
    valid_ids = []
    for i, obs in enumerate(observations):
        c = obs.provisional_plane_center
        if c is not None and np.all(np.isfinite(c)):
            provisional.append(c)
            valid_ids.append(i)
        else:
            provisional.append([np.nan, np.nan])
    provisional = np.asarray(provisional, dtype=np.float64)

    anchor_indices = select_anchor_indices(observations)
    template_xy = np.asarray(circle_centers_w, dtype=np.float64)[:, :2]

    if len(valid_ids) < 2:
        assignment = list(range(len(observations)))
        R2 = np.eye(2, dtype=np.float64)
        t2 = np.zeros((2,), dtype=np.float64)
        pred_xy = template_xy.copy()
    else:
        best = None
        valid_ids = np.asarray(valid_ids, dtype=np.int32)
        provisional_valid = provisional[valid_ids]
        for perm in permutations(range(len(template_xy)), len(valid_ids)):
            tmpl = template_xy[list(perm)]
            try:
                R2, t2 = _weighted_rigid_alignment(tmpl, provisional_valid, None)
            except Exception:
                continue
            pred = (tmpl @ R2.T) + t2.reshape(1, 2)
            err = np.linalg.norm(pred - provisional_valid, axis=1)
            score = float(np.median(err))
            if best is None or score < best["score"]:
                best = {"perm": perm, "R2": R2, "t2": t2, "score": score}
        assignment = [-1] * len(observations)
        for det_idx, layout_idx in zip(valid_ids.tolist(), list(best["perm"])):
            assignment[int(det_idx)] = int(layout_idx)
        remaining_layout = [j for j in range(len(template_xy)) if j not in best["perm"]]
        remaining_det = [i for i in range(len(observations)) if assignment[i] < 0]
        for det_idx, layout_idx in zip(remaining_det, remaining_layout):
            assignment[int(det_idx)] = int(layout_idx)
        R2 = best["R2"]
        t2 = best["t2"]
        pred_xy = (template_xy @ R2.T) + t2.reshape(1, 2)

    pred_3d = _plane2d_to_plane3d(pred_xy, origin, e1, e2)
    pred_img = _project_plane_points_to_image(pred_3d, K)

    debug = {
        "provisional_plane_centers": provisional.tolist(),
        "anchor_indices": anchor_indices,
        "layout_assignment": assignment,
        "predicted_plane_centers": pred_xy.tolist(),
        "predicted_image_centers": pred_img.tolist(),
    }

    return AnchorFrame(
        anchor_indices=anchor_indices,
        provisional_plane_centers=provisional,
        layout_assignment=assignment,
        layout_transform_R=R2,
        layout_transform_t=t2,
        predicted_plane_centers=pred_xy,
        predicted_image_centers=pred_img,
        debug=debug,
    )


# -----------------------------
# 新的前端搜索：角域压缩 + 低候选扇区
# -----------------------------
def _sample_level_offsets(reference_ellipse, level_contour):
    ref_pts, ref_nrms, ref_ang = _sample_ellipse_points_and_normals(reference_ellipse, None)
    offsets = []
    valid = []
    for p, n in zip(ref_pts, ref_nrms):
        dist = _point_polygon_signed_distance(level_contour, p)
        offsets.append(-dist)
        valid.append(np.isfinite(dist))
    return ref_pts, ref_nrms, ref_ang, np.asarray(offsets, dtype=np.float64), np.asarray(valid, dtype=bool)


def _stable_offset_field(observation: LightObservation):
    ref_ellipse = observation.source_ellipse
    if ref_ellipse is None or not observation.multilevel_contours:
        return None
    all_offsets = []
    all_valid = []
    ref_pts = ref_nrms = ref_ang = None
    for contour in observation.multilevel_contours:
        ref_pts, ref_nrms, ref_ang, off, val = _sample_level_offsets(ref_ellipse, contour)
        all_offsets.append(off)
        all_valid.append(val)
    O = np.asarray(all_offsets, dtype=np.float64)
    V = np.asarray(all_valid, dtype=bool)
    med = np.full((O.shape[1],), np.nan, dtype=np.float64)
    mad = np.full((O.shape[1],), np.nan, dtype=np.float64)
    support = np.zeros((O.shape[1],), dtype=np.int32)
    for j in range(O.shape[1]):
        vals = O[V[:, j], j]
        support[j] = int(len(vals))
        if len(vals):
            med[j] = float(np.median(vals))
            mad[j] = float(np.median(np.abs(vals - np.median(vals))))

    finite_mad = mad[np.isfinite(mad)]
    base_scale = float(np.median(finite_mad)) if finite_mad.size else 1.0
    quality = support.astype(np.float64) / (mad + base_scale + 1e-12)
    quality[~np.isfinite(quality)] = 0.0
    smooth_len = int(np.clip(np.ceil(np.sqrt(max(len(quality), 1))), 5, 31))
    quality_smooth = _circular_smooth(quality, smooth_len)
    offset_smooth = _circular_smooth(np.where(np.isfinite(med), med, np.nanmedian(med[np.isfinite(med)]) if np.any(np.isfinite(med)) else 0.0), smooth_len)
    return {
        "ref_points": ref_pts,
        "ref_normals": ref_nrms,
        "angles": ref_ang,
        "offset_median": med,
        "offset_mad": mad,
        "support": support,
        "quality": quality,
        "quality_smooth": quality_smooth,
        "offset_smooth": offset_smooth,
    }


def _effective_arc_coverage_from_angles(angles: np.ndarray) -> float:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if ang.size < 2:
        return 0.0
    ang = np.sort(np.mod(ang, 2.0 * np.pi))
    gaps = np.diff(np.r_[ang, ang[0] + 2.0 * np.pi])
    if gaps.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.max(gaps) / (2.0 * np.pi), 0.0, 1.0))


def _angle_balance_from_angles(angles: np.ndarray) -> float:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if ang.size == 0:
        return 0.0
    z = np.exp(1j * ang)
    resultant = float(np.abs(np.mean(z)))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))

def _local_maxima_indices(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 0:
        return np.empty((0,), dtype=np.int32)
    prev = np.r_[q[-1], q[:-1]]
    nxt = np.r_[q[1:], q[0]]
    mask = (q >= prev) & (q > nxt)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        idx = np.array([int(np.argmax(q))], dtype=np.int32)
    return idx.astype(np.int32)


def _candidate_sector_indices(field: Dict[str, Any]) -> List[np.ndarray]:
    q = np.asarray(field["quality_smooth"], dtype=np.float64)
    N = len(q)
    if N == 0:
        return []
    seeds_all = _local_maxima_indices(q)
    order = np.argsort(q[seeds_all])[::-1]
    seeds_all = seeds_all[order]
    keep_count = int(max(1, np.ceil(np.sqrt(len(seeds_all)))))
    seeds = seeds_all[:keep_count]

    max_half = max(3, N // 4)
    width_count = int(np.clip(np.ceil(np.log2(max_half + 1)), 4, 8))
    half_widths = np.unique(np.ceil(np.geomspace(3, max_half, num=width_count)).astype(np.int32))

    sectors = []
    seen = set()
    all_peaks = np.asarray(seeds_all, dtype=np.int32)
    for s in seeds.tolist():
        opp_target = int((s + N // 2) % N)
        if all_peaks.size:
            cyc = np.minimum((all_peaks - opp_target) % N, (opp_target - all_peaks) % N)
            opp_peak = int(all_peaks[int(np.argmin(cyc))])
        else:
            opp_peak = opp_target
        for hw in half_widths.tolist():
            idx1 = np.sort((s + np.arange(-hw, hw + 1, dtype=np.int32)) % N)
            key1 = tuple(idx1.tolist())
            if key1 not in seen:
                seen.add(key1)
                sectors.append(idx1)

            idx2 = np.sort((opp_peak + np.arange(-hw, hw + 1, dtype=np.int32)) % N)
            idx12 = np.unique(np.r_[idx1, idx2])
            key12 = tuple(idx12.tolist())
            if key12 not in seen:
                seen.add(key12)
                sectors.append(idx12)
    return sectors

def _arc_points_from_sector(field, sector_idx):
    off = field["offset_smooth"][sector_idx]
    pts = field["ref_points"][sector_idx] + off[:, None] * field["ref_normals"][sector_idx]
    ang = field["angles"][sector_idx]
    return pts.astype(np.float64), ang.astype(np.float64)


def _circle_center_stability(points_img, K, plane_normal_camera, plane_distance):
    X3, valid = _backproject_pixels_to_plane(points_img, K, plane_normal_camera, plane_distance)
    X3 = X3[valid]
    if len(X3) < 6:
        return np.inf, None
    origin, e1, e2 = _plane_basis_from_current_plane(plane_normal_camera, plane_distance)
    P2 = _plane3d_to_plane2d(X3, origin, e1, e2)
    centers = []
    for i in range(len(P2)):
        keep = np.ones((len(P2),), dtype=bool)
        keep[i] = False
        if np.count_nonzero(keep) < 3:
            continue
        fit = _free_circle_fit(P2[keep])
        if fit["center"] is not None:
            centers.append(fit["center"])
    if not centers:
        return np.inf, None
    centers = np.asarray(centers, dtype=np.float64)
    c_med = np.median(centers, axis=0)
    stability = float(np.median(np.linalg.norm(centers - c_med.reshape(1, 2), axis=1)))
    return stability, c_med


def _layout_consistency_score(
    detected_idx: int,
    candidate_center_plane: np.ndarray,
    anchor_frame: AnchorFrame,
    circle_centers_w: np.ndarray,
):
    centers = np.asarray(anchor_frame.provisional_plane_centers, dtype=np.float64).copy()
    if candidate_center_plane is not None and np.all(np.isfinite(candidate_center_plane)):
        centers[detected_idx] = candidate_center_plane.reshape(2)
    valid = np.flatnonzero(np.all(np.isfinite(centers), axis=1))
    if len(valid) < 2:
        return np.inf
    tmpl = np.asarray(circle_centers_w, dtype=np.float64)[:, :2]
    assigned = [anchor_frame.layout_assignment[i] for i in valid]
    tmpl_valid = tmpl[assigned]
    obs_valid = centers[valid]
    R2, t2 = _weighted_rigid_alignment(tmpl_valid, obs_valid, None)
    pred = (tmpl_valid @ R2.T) + t2.reshape(1, 2)
    err = np.linalg.norm(pred - obs_valid, axis=1)
    return float(np.median(err))


def collect_arc_hypotheses_for_light(
    detected_idx: int,
    observation: LightObservation,
    anchor_frame: AnchorFrame,
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> List[Dict[str, Any]]:
    field = _stable_offset_field(observation)
    if field is None:
        return []

    sectors = _candidate_sector_indices(field)
    hyps = []
    for sector in sectors:
        support = field["support"][sector]
        offset_mad = field["offset_mad"][sector]
        if not np.any(np.isfinite(offset_mad)):
            continue
        pts_img, ang = _arc_points_from_sector(field, sector)
        if len(pts_img) < 6:
            continue
        ellipse = _fit_ellipse_best(pts_img.reshape(-1, 1, 2).astype(np.float32))
        if ellipse is None:
            continue
        stability, c_plane = _circle_center_stability(pts_img, K, plane_normal_camera, plane_distance)
        layout_score = _layout_consistency_score(detected_idx, c_plane, anchor_frame, circle_centers_w)
        support_score = float(np.median(support.astype(np.float64)))
        coverage = _effective_arc_coverage_from_angles(ang)
        angle_balance = _angle_balance_from_angles(ang)
        hyps.append({
            "points_img": pts_img,
            "angles": ang,
            "ellipse": ellipse,
            "layer_stability": float(np.median(offset_mad[np.isfinite(offset_mad)])),
            "plane_center_stability": float(stability),
            "layout_consistency": float(layout_score),
            "support_score": float(support_score),
            "effective_arc_coverage": float(coverage),
            "angle_balance": float(angle_balance),
            "sector_idx": sector.astype(np.int32),
            "candidate_plane_center": None if c_plane is None else np.asarray(c_plane, dtype=np.float64),
        })
    return hyps

def score_and_select_arc_family(hypotheses: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[float], List[Dict[str, Any]]]:
    if not hypotheses:
        return None, None, []

    layer = _rank_unit([h["layer_stability"] for h in hypotheses], higher_is_better=False)
    plane = _rank_unit([h["plane_center_stability"] for h in hypotheses], higher_is_better=False)
    layout = _rank_unit([h["layout_consistency"] for h in hypotheses], higher_is_better=False)
    support = _rank_unit([h["support_score"] for h in hypotheses], higher_is_better=True)
    coverage = _rank_unit([h["effective_arc_coverage"] for h in hypotheses], higher_is_better=True)
    balance = _rank_unit([h["angle_balance"] for h in hypotheses], higher_is_better=True)

    scored = []
    for i, h in enumerate(hypotheses):
        score = float(layer[i] + plane[i] + layout[i] + support[i] + coverage[i] + balance[i])
        item = dict(h)
        item["score_terms"] = {
            "layer_stability_rank": float(layer[i]),
            "plane_center_stability_rank": float(plane[i]),
            "layout_consistency_rank": float(layout[i]),
            "support_rank": float(support[i]),
            "coverage_rank": float(coverage[i]),
            "angle_balance_rank": float(balance[i]),
        }
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"])
    best = scored[0]
    gap = None if len(scored) == 1 else float(scored[1]["score"] - scored[0]["score"])
    return best, gap, scored

def select_physical_edges(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[ArcSelectionResult], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )

    anchor_frame = build_anchor_frame(
        observations=observations,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        circle_centers_w=circle_centers_w,
    )

    results: List[ArcSelectionResult] = []
    per_light_debug = []
    for idx, obs in enumerate(observations):
        hypotheses = collect_arc_hypotheses_for_light(
            detected_idx=idx,
            observation=obs,
            anchor_frame=anchor_frame,
            K=K,
            plane_normal_camera=plane_normal_camera,
            plane_distance=plane_distance,
            circle_centers_w=circle_centers_w,
        )
        best, gap, scored = score_and_select_arc_family(hypotheses)

        if best is None:
            work_contour = np.asarray(obs.contour, dtype=np.float32)
            work_ellipse = obs.source_ellipse
            arc_points = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
            arc_angles = np.empty((0,), dtype=np.float64)
            arc_score = np.inf
            score_gap = np.nan
            debug = {
                "is_anchor": idx in anchor_frame.anchor_indices,
                "num_arc_hypotheses": 0,
                "reason": "fallback_source_contour",
                "feature_debug": obs.feature_debug,
                "anchor_frame": anchor_frame.debug,
            }
        else:
            work_contour = best["points_img"].reshape(-1, 1, 2).astype(np.float32)
            work_ellipse = best["ellipse"]
            arc_points = best["points_img"]
            arc_angles = best["angles"]
            arc_score = float(best["score"])
            score_gap = np.nan if gap is None else float(gap)
            debug = {
                "is_anchor": idx in anchor_frame.anchor_indices,
                "num_arc_hypotheses": int(len(hypotheses)),
                "best_arc_score": float(best["score"]),
                "best_second_gap": score_gap,
                "layer_stability": float(best["layer_stability"]),
                "plane_center_stability": float(best["plane_center_stability"]),
                "layout_consistency": float(best["layout_consistency"]),
                "support_score": float(best["support_score"]),
                "effective_arc_coverage": float(best["effective_arc_coverage"]),
                "angle_balance": float(best["angle_balance"]),
                "candidate_plane_center": None if best["candidate_plane_center"] is None else best["candidate_plane_center"].tolist(),
                "score_terms": best["score_terms"],
                "feature_debug": obs.feature_debug,
                "anchor_frame": anchor_frame.debug,
                "used": True,
                "reason": "fast_normal_offset_edge_selection_v3",
                "selected_arc_points_count": int(len(best["points_img"])),
                "selected_arc_angles_count": int(len(best["angles"])),
            }

        results.append(
            ArcSelectionResult(
                work_contour=work_contour,
                work_ellipse=work_ellipse,
                arc_points=arc_points,
                arc_angles=arc_angles,
                arc_score=arc_score,
                score_gap=score_gap,
                debug=debug,
            )
        )
        per_light_debug.append(debug)

    global_debug = {
        "anchor_indices": anchor_frame.anchor_indices,
        "anchor_frame": anchor_frame.debug,
        "num_lights": int(len(observations)),
        "per_light": per_light_debug,
    }
    return results, global_debug


def build_selected_measurements_for_pipeline(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    arc_results, global_debug = select_physical_edges(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        circle_centers_w=circle_centers_w,
    )

    out = []
    for obs, sel in zip(observations, arc_results):
        out.append({
            "source_contour": obs.contour,
            "source_ellipse": obs.source_ellipse,
            "bbox_center": obs.bbox_center,
            "gray_center": obs.gray_center,
            "work_contour": sel.work_contour,
            "work_ellipse": sel.work_ellipse,
            "arc_points": sel.arc_points,
            "arc_angles": sel.arc_angles,
            "rep_dbg": sel.debug,
        })
    return out, global_debug


# ============================================================
# v4 覆盖：单灯“核心外侧第一稳定过渡边”选择
# ============================================================

def _core_center_from_contour(gray_light: np.ndarray, contour: np.ndarray, source_ellipse: Optional[Tuple]) -> np.ndarray:
    cnt = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros(gray_light.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        if source_ellipse is not None:
            return np.asarray(source_ellipse[0], dtype=np.float64)
        return _gray_centroid_in_contour(gray_light, contour)

    vals = gray_light[ys, xs].astype(np.uint8)
    if vals.size < 9:
        if source_ellipse is not None:
            return np.asarray(source_ellipse[0], dtype=np.float64)
        return _gray_centroid_in_contour(gray_light, contour)

    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core_mask = np.zeros_like(mask)
    core_mask[ys, xs] = np.where(vals >= thr, 255, 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(core_mask)
    if num_labels <= 1:
        if source_ellipse is not None:
            return np.asarray(source_ellipse[0], dtype=np.float64)
        return _gray_centroid_in_contour(gray_light, contour)

    peak_idx = int(np.argmax(vals))
    peak_y = int(ys[peak_idx])
    peak_x = int(xs[peak_idx])
    peak_label = int(labels[peak_y, peak_x])

    if peak_label <= 0:
        if source_ellipse is not None:
            return np.asarray(source_ellipse[0], dtype=np.float64)
        return _gray_centroid_in_contour(gray_light, contour)

    comp = labels == peak_label
    ys2, xs2 = np.where(comp)
    w = gray_light[ys2, xs2].astype(np.float64)
    w = np.clip(w - float(thr) + 1.0, 1.0, None)
    denom = float(np.sum(w))
    if denom <= 1e-12:
        return np.array([np.mean(xs2), np.mean(ys2)], dtype=np.float64)
    cx = float(np.sum(xs2 * w) / denom)
    cy = float(np.sum(ys2 * w) / denom)
    return np.array([cx, cy], dtype=np.float64)


def _ray_exit_radius(contour: np.ndarray, center: np.ndarray, angle: float) -> float:
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    dmax = float(np.max(np.linalg.norm(pts - c.reshape(1, 2), axis=1)))
    if not np.isfinite(dmax) or dmax <= 1.0:
        return 0.0
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    num = int(max(32, np.ceil(2.0 * dmax)))
    rs = np.linspace(0.0, dmax, num=num, endpoint=True)
    inside = []
    for r in rs:
        p = c + r * direction
        inside.append(cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), False) >= 0)
    inside = np.asarray(inside, dtype=bool)
    if not np.any(inside):
        return 0.0
    idx = int(np.max(np.flatnonzero(inside)))
    return float(rs[idx])


def _segment_prefix_stats(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ones = np.ones_like(x)
    return {
        'n': np.r_[0.0, np.cumsum(ones)],
        'x': np.r_[0.0, np.cumsum(x)],
        'y': np.r_[0.0, np.cumsum(y)],
        'xx': np.r_[0.0, np.cumsum(x * x)],
        'xy': np.r_[0.0, np.cumsum(x * y)],
        'yy': np.r_[0.0, np.cumsum(y * y)],
    }


def _segment_linfit_from_prefix(pref: Dict[str, np.ndarray], i: int, j: int):
    m = float(pref['n'][j] - pref['n'][i])
    if m <= 1.0:
        return 0.0, float(pref['y'][j] - pref['y'][i]), 0.0
    sx = float(pref['x'][j] - pref['x'][i])
    sy = float(pref['y'][j] - pref['y'][i])
    sxx = float(pref['xx'][j] - pref['xx'][i])
    sxy = float(pref['xy'][j] - pref['xy'][i])
    syy = float(pref['yy'][j] - pref['yy'][i])
    den = m * sxx - sx * sx
    if abs(den) <= 1e-12:
        a = 0.0
        b = sy / m
    else:
        a = (m * sxy - sx * sy) / den
        b = (sy - a * sx) / m
    sse = syy - 2.0 * a * sxy - 2.0 * b * sy + a * a * sxx + 2.0 * a * b * sx + m * b * b
    return float(a), float(b), float(max(sse, 0.0))


def _piecewise_three_segment_edge(r: np.ndarray, profile: np.ndarray) -> Optional[Dict[str, Any]]:
    x = np.asarray(r, dtype=np.float64).reshape(-1)
    y = np.asarray(profile, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 15:
        return None
    min_seg = int(max(4, n // 12))
    pref = _segment_prefix_stats(x, y)

    best = None
    best_any = None
    for a in range(min_seg, n - 2 * min_seg):
        for b in range(a + min_seg, n - min_seg):
            s1, c1, e1 = _segment_linfit_from_prefix(pref, 0, a)
            s2, c2, e2 = _segment_linfit_from_prefix(pref, a, b)
            s3, c3, e3 = _segment_linfit_from_prefix(pref, b, n)
            sse = e1 + e2 + e3
            cand = {
                'a': int(a), 'b': int(b),
                's1': float(s1), 's2': float(s2), 's3': float(s3),
                'sse': float(sse),
            }
            if best_any is None or cand['sse'] < best_any['sse']:
                best_any = cand
            if s2 <= s1 and s2 <= s3:
                if best is None or cand['sse'] < best['sse']:
                    best = cand

    if best is None:
        best = best_any
    if best is None:
        return None

    a = best['a']
    b = best['b']
    r_edge = float(x[a])
    slope_contrast = float(abs(best['s2'] - best['s1']) + abs(best['s2'] - best['s3']))
    seg2_len = float(max(x[b - 1] - x[a], 1e-12))
    rmse = float(np.sqrt(best['sse'] / max(n, 1)))
    quality = float(slope_contrast * seg2_len / max(rmse, 1e-12))

    return {
        'radius': r_edge,
        'quality': quality,
        'rmse': rmse,
        'a': a,
        'b': b,
        's1': best['s1'],
        's2': best['s2'],
        's3': best['s3'],
    }


def _sample_radial_profile(gray_light: np.ndarray, center: np.ndarray, angle: float, r_max: float):
    if not np.isfinite(r_max) or r_max <= 4.0:
        return None
    step = 0.5
    rs = np.arange(0.0, float(r_max) + step, step, dtype=np.float64)
    cs = np.asarray(center, dtype=np.float64).reshape(2)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    vals = np.array([
        _bilinear_at(gray_light, cs[0] + rr * direction[0], cs[1] + rr * direction[1])
        for rr in rs
    ], dtype=np.float64)
    vals_sm = _smooth_profile_1d(vals)
    return {
        'r': rs,
        'profile': vals_sm,
        'raw_profile': vals,
        'direction': direction,
    }


def _circular_smooth_valid(values: np.ndarray, weights: np.ndarray, win: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(v) == 0:
        return v
    n = len(v)
    win = int(max(1, min(win, n)))
    ext_v = np.r_[v[-win:], v, v[:win]]
    ext_w = np.r_[w[-win:], w, w[:win]]
    out = np.empty((n,), dtype=np.float64)
    for i in range(n):
        sl = slice(i, i + 2 * win + 1)
        ww = np.clip(ext_w[sl], 0.0, None)
        vv = ext_v[sl]
        if np.sum(ww) <= 1e-12:
            out[i] = v[i]
        else:
            out[i] = float(np.sum(vv * ww) / np.sum(ww))
    return out


def _detect_shoulder_arc_for_observation(gray_light: np.ndarray, obs: LightObservation) -> ArcSelectionResult:
    source_ellipse = obs.source_ellipse
    if source_ellipse is None:
        work_contour = np.asarray(obs.contour, dtype=np.float32).reshape(-1, 1, 2)
        arc_points = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(arc_points), endpoint=False, dtype=np.float64)
        return ArcSelectionResult(
            work_contour=work_contour,
            work_ellipse=None,
            arc_points=arc_points,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={'used': False, 'reason': 'fallback_no_source_ellipse', 'arc_weights': [1.0] * len(arc_points)},
        )

    core_center = _core_center_from_contour(gray_light, obs.contour, source_ellipse)
    _, _, base_angles = _sample_ellipse_points_and_normals(source_ellipse, None)
    num_angles = int(len(base_angles))
    records = []
    for ang in base_angles:
        r_exit = _ray_exit_radius(obs.contour, core_center, float(ang))
        prof = _sample_radial_profile(gray_light, core_center, float(ang), r_exit)
        if prof is None:
            continue
        edge = _piecewise_three_segment_edge(prof['r'], prof['profile'])
        if edge is None or not np.isfinite(edge['radius']):
            continue
        pt = core_center + edge['radius'] * prof['direction']
        if not np.all(np.isfinite(pt)):
            continue
        records.append({
            'angle': float(ang),
            'radius': float(edge['radius']),
            'point': pt.astype(np.float64),
            'quality': float(edge['quality']),
            'rmse': float(edge['rmse']),
            'segment_a': int(edge['a']),
            'segment_b': int(edge['b']),
            'slopes': [float(edge['s1']), float(edge['s2']), float(edge['s3'])],
        })

    if len(records) < 12:
        work_contour = np.asarray(obs.contour, dtype=np.float32).reshape(-1, 1, 2)
        arc_points = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(arc_points), endpoint=False, dtype=np.float64)
        return ArcSelectionResult(
            work_contour=work_contour,
            work_ellipse=source_ellipse,
            arc_points=arc_points,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={'used': False, 'reason': 'fallback_too_few_profiles', 'core_center': core_center.tolist(), 'arc_weights': [1.0] * len(arc_points)},
        )

    records.sort(key=lambda d: d['angle'])
    angles = np.asarray([d['angle'] for d in records], dtype=np.float64)
    radii = np.asarray([d['radius'] for d in records], dtype=np.float64)
    qualities = np.asarray([d['quality'] for d in records], dtype=np.float64)
    directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)

    qscale = float(np.median(qualities[np.isfinite(qualities)])) if np.any(np.isfinite(qualities)) else 1.0
    qnorm = qualities / max(qscale, 1e-12)
    qnorm = 1.0 / (1.0 + np.maximum(qnorm, 0.0) ** (-1))
    win = int(max(3, min(len(radii) // 8, 15)))
    radii_sm = _circular_smooth_valid(radii, np.clip(qnorm, 1e-6, None), win)
    points = core_center.reshape(1, 2) + radii_sm[:, None] * directions

    fit_e = _fit_ellipse_best(points.astype(np.float32).reshape(-1, 1, 2))
    if fit_e is None:
        fit_e = source_ellipse

    residuals = np.array([
        _ellipse_contour_residual(fit_e, points[i:i+1])
        for i in range(len(points))
    ], dtype=np.float64)
    rmed = float(np.median(residuals[np.isfinite(residuals)])) if np.any(np.isfinite(residuals)) else 0.0
    rmad = float(np.median(np.abs(residuals - rmed))) if np.any(np.isfinite(residuals)) else 0.0
    scale = max(rmad, 1e-12)
    robust = 1.0 / (1.0 + ((residuals - rmed) / scale) ** 2)
    final_weights = np.clip(qnorm * robust, 1e-6, None)

    if fit_e is not None:
        work_pts, _, work_angles = _sample_ellipse_points_and_normals(fit_e, None)
        work_contour = work_pts.astype(np.float32).reshape(-1, 1, 2)
    else:
        work_contour = points.astype(np.float32).reshape(-1, 1, 2)
        work_angles = angles

    coverage = _effective_arc_coverage_from_angles(angles)
    balance = _angle_balance_from_angles(angles)
    score = float(np.median(residuals[np.isfinite(residuals)])) if np.any(np.isfinite(residuals)) else np.inf

    debug = {
        'used': True,
        'reason': 'single_light_stable_shoulder_edge_v4',
        'core_center': core_center.tolist(),
        'num_profiles': int(len(records)),
        'selected_arc_points_count': int(len(points)),
        'selected_arc_angles_count': int(len(angles)),
        'effective_arc_coverage': float(coverage),
        'angle_balance': float(balance),
        'profile_rmse_mean': float(np.mean([r['rmse'] for r in records])),
        'profile_quality_mean': float(np.mean(qualities)),
        'arc_weights': final_weights.astype(np.float64).tolist(),
    }

    return ArcSelectionResult(
        work_contour=work_contour,
        work_ellipse=fit_e,
        arc_points=points.astype(np.float64),
        arc_angles=angles.astype(np.float64),
        arc_score=score,
        score_gap=np.nan,
        debug=debug,
    )


def select_physical_edges(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[ArcSelectionResult], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    results = []
    per_light = []
    for obs in observations:
        res = _detect_shoulder_arc_for_observation(gray_light, obs)
        results.append(res)
        per_light.append(res.debug)
    global_debug = {
        'mode': 'single_light_stable_shoulder_edge_v4',
        'num_lights': int(len(observations)),
        'per_light': per_light,
    }
    return results, global_debug


def build_selected_measurements_for_pipeline(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    arc_results, global_debug = select_physical_edges(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        circle_centers_w=circle_centers_w,
    )
    out = []
    for obs, sel in zip(observations, arc_results):
        out.append({
            'source_contour': obs.contour,
            'source_ellipse': obs.source_ellipse,
            'bbox_center': obs.bbox_center,
            'gray_center': obs.gray_center,
            'work_contour': sel.work_contour,
            'work_ellipse': sel.work_ellipse,
            'arc_points': sel.arc_points,
            'arc_angles': sel.arc_angles,
            'arc_weights': np.asarray(sel.debug.get('arc_weights', []), dtype=np.float64),
            'rep_dbg': sel.debug,
        })
    return out, global_debug


# ==========================
# v5: 按《几何校正椭圆边缘提取创新》重构
# 核心：径向分段回归 + BIC 模型选择 + shoulder 绝对置信度
# 保持主程序接口不变，仅替换前端边缘点生成逻辑。
# ==========================

def _bilinear(gray: np.ndarray, x: float, y: float) -> float:
    h, w = gray.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1); y1 = min(y0 + 1, h - 1)
    dx = x - x0; dy = y - y0
    v00 = float(gray[y0, x0]); v10 = float(gray[y0, x1])
    v01 = float(gray[y1, x0]); v11 = float(gray[y1, x1])
    return (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v10 + (1.0 - dx) * dy * v01 + dx * dy * v11


def _prefix_sums_bic(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Sx = np.concatenate([[0.0], np.cumsum(x)])
    Sy = np.concatenate([[0.0], np.cumsum(y)])
    Sxx = np.concatenate([[0.0], np.cumsum(x * x)])
    Sxy = np.concatenate([[0.0], np.cumsum(x * y)])
    Syy = np.concatenate([[0.0], np.cumsum(y * y)])
    return Sx, Sy, Sxx, Sxy, Syy


def _seg_stats_bic(Sx, Sy, Sxx, Sxy, Syy, i: int, j: int):
    n = int(j - i)
    sx = float(Sx[j] - Sx[i])
    sy = float(Sy[j] - Sy[i])
    sxx = float(Sxx[j] - Sxx[i])
    sxy = float(Sxy[j] - Sxy[i])
    syy = float(Syy[j] - Syy[i])
    return n, sx, sy, sxx, sxy, syy


def _sse_const_bic(Sy, Syy, i: int, j: int) -> float:
    n = int(j - i)
    if n <= 0:
        return np.inf
    sy = float(Sy[j] - Sy[i])
    syy = float(Syy[j] - Syy[i])
    return float(max(0.0, syy - (sy * sy) / n))


def _fit_line_sse_bic(Sx, Sy, Sxx, Sxy, Syy, i: int, j: int):
    n, sx, sy, sxx, sxy, syy = _seg_stats_bic(Sx, Sy, Sxx, Sxy, Syy, i, j)
    if n <= 1:
        return np.inf, 0.0, float(sy / max(n, 1))
    denom = float(n * sxx - sx * sx)
    if abs(denom) <= np.finfo(np.float64).eps:
        b = float(sy / n)
        sse = float(max(0.0, syy - (sy * sy) / n))
        return sse, 0.0, b
    m = float((n * sxy - sx * sy) / denom)
    b = float((sy - m * sx) / n)
    sse = float(syy - 2.0 * m * sxy - 2.0 * b * sy + (m * m) * sxx + 2.0 * m * b * sx + n * (b * b))
    return max(0.0, sse), m, b


def _bic_score(n: int, rss: float, k_params: int) -> float:
    eps = np.finfo(np.float64).tiny
    rss = float(max(rss, eps))
    return float(n * np.log(rss / n) + k_params * np.log(n))


def _best_piecewise_model_bic(r: np.ndarray, y: np.ndarray):
    x = np.asarray(r, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 12:
        return None

    Sx, Sy, Sxx, Sxy, Syy = _prefix_sums_bic(x, yy)

    rss1, m1, b1 = _fit_line_sse_bic(Sx, Sy, Sxx, Sxy, Syy, 0, n)
    bic1 = _bic_score(n, rss1, 2)

    best2 = (np.inf, None)
    for t in range(2, n - 2):
        rss0 = _sse_const_bic(Sy, Syy, 0, t)
        rssL, mL, bL = _fit_line_sse_bic(Sx, Sy, Sxx, Sxy, Syy, t, n)
        if not np.isfinite(rss0 + rssL):
            continue
        if mL > 0.0:
            continue
        rss = rss0 + rssL
        bic = _bic_score(n, rss, 3)
        if bic < best2[0]:
            mu0 = float((Sy[t] - Sy[0]) / t)
            best2 = (bic, (t, mu0, mL, bL, rss))
    bic2 = float(best2[0])

    best3 = (np.inf, None)
    for t1 in range(2, n - 4):
        rss0 = _sse_const_bic(Sy, Syy, 0, t1)
        if not np.isfinite(rss0):
            continue
        mu0 = float((Sy[t1] - Sy[0]) / t1)
        for t2 in range(t1 + 2, n - 2):
            rssA, mA, bA = _fit_line_sse_bic(Sx, Sy, Sxx, Sxy, Syy, t1, t2)
            rssB, mB, bB = _fit_line_sse_bic(Sx, Sy, Sxx, Sxy, Syy, t2, n)
            if not np.isfinite(rssA + rssB):
                continue
            if mA > 0.0 or mB > 0.0:
                continue
            if abs(mB) > abs(mA):
                continue
            rss = rss0 + rssA + rssB
            bic = _bic_score(n, rss, 5)
            if bic < best3[0]:
                best3 = (bic, (t1, t2, mu0, mA, bA, mB, bB, rss))
    bic3 = float(best3[0])

    bics = np.array([bic1, bic2, bic3], dtype=np.float64)
    finite = np.isfinite(bics)
    if not np.any(finite):
        return None
    bmin = float(np.min(bics[finite]))
    w = np.exp(-0.5 * (bics - bmin))
    w[~np.isfinite(w)] = 0.0
    sw = float(np.sum(w))
    if sw > 0.0:
        w /= sw

    idx = int(np.nanargmin(bics))
    if idx == 0:
        g = np.abs(np.gradient(yy, x))
        k = int(np.argmax(g))
        return {
            'model': 'M1', 'radius': float(x[k]), 'confidence': float(w[1] + w[2]),
            'bic': bics, 'weights': w, 'rmse': float(np.sqrt(max(rss1 / n, 0.0))),
            'm1': float(m1), 'b1': float(b1), 's1': float(m1), 's2': float(m1), 's3': 0.0,
            'a': k, 'b': k,
        }
    if idx == 1 and best2[1] is not None:
        t, mu0, mL, bL, rss = best2[1]
        if abs(mL) <= np.finfo(np.float64).eps:
            r_star = float(x[t])
        else:
            r_star = float(np.clip((mu0 - bL) / mL, x[0], x[-1]))
        return {
            'model': 'M2', 'radius': r_star, 'confidence': float(w[1] + w[2]),
            'bic': bics, 'weights': w, 'rmse': float(np.sqrt(max(rss / n, 0.0))),
            'mu0': float(mu0), 'm1': float(mL), 'b1': float(bL), 's1': 0.0, 's2': float(mL), 's3': float(mL),
            'a': int(t), 'b': int(t),
        }
    if idx == 2 and best3[1] is not None:
        t1, t2, mu0, mA, bA, mB, bB, rss = best3[1]
        if abs(mA) <= np.finfo(np.float64).eps:
            r_star = float(x[t1])
        else:
            r_star = float(np.clip((mu0 - bA) / mA, x[0], x[-1]))
        return {
            'model': 'M3', 'radius': r_star, 'confidence': float(w[1] + w[2]),
            'bic': bics, 'weights': w, 'rmse': float(np.sqrt(max(rss / n, 0.0))),
            'mu0': float(mu0), 'm1': float(mA), 'b1': float(bA), 'm2': float(mB), 'b2': float(bB),
            's1': 0.0, 's2': float(mA), 's3': float(mB), 'a': int(t1), 'b': int(t2),
        }
    return None


def _sample_radial_profile_bic(gray_light: np.ndarray, center: np.ndarray, angle: float, rmax: float) -> Optional[Dict[str, Any]]:
    if not np.isfinite(rmax) or rmax < 8.0:
        return None
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    r = np.arange(0.0, float(rmax) + 1.0, 1.0, dtype=np.float64)
    if len(r) < 12:
        return None
    vals = np.array([_bilinear(gray_light, center[0] + rr * direction[0], center[1] + rr * direction[1]) for rr in r], dtype=np.float64)
    return {'r': r, 'profile': vals, 'direction': direction}


def _fit_ellipse_weighted_by_conf(points: np.ndarray, conf: np.ndarray) -> Optional[Tuple]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 6:
        return None
    ww = np.asarray(conf, dtype=np.float64).reshape(-1)
    if len(ww) != len(pts) or not np.all(np.isfinite(ww)):
        ww = np.ones((len(pts),), dtype=np.float64)
    ww = np.clip(ww, 1e-9, None)
    ww = ww / (np.mean(ww) + 1e-12)

    centroid = np.average(pts, axis=0, weights=ww)
    X = pts - centroid
    C = (X.T * ww) @ X / np.sum(ww)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj = X @ evecs
    sx = np.sqrt(max(np.average(proj[:, 0] ** 2, weights=ww), 1e-12))
    sy = np.sqrt(max(np.average(proj[:, 1] ** 2, weights=ww), 1e-12))
    scale = np.sqrt(2.0)
    a0 = max(2.0 * scale * sx, 1.0)
    b0 = max(2.0 * scale * sy, 1.0)
    ang0 = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    e0 = ((float(centroid[0]), float(centroid[1])), (float(2.0 * a0), float(2.0 * b0)), ang0)

    try:
        e1 = _fit_ellipse_best(pts.astype(np.float32).reshape(-1, 1, 2))
        return e1 if e1 is not None else e0
    except Exception:
        return e0


def _detect_shoulder_arc_for_observation_bic(gray_light: np.ndarray, obs: LightObservation) -> ArcSelectionResult:
    source_ellipse = obs.source_ellipse
    if source_ellipse is None:
        contour = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
        if len(contour) < 6:
            contour = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(contour), endpoint=False, dtype=np.float64) if len(contour) else np.empty((0,), dtype=np.float64)
        return ArcSelectionResult(
            work_contour=contour.astype(np.float32).reshape(-1, 1, 2),
            work_ellipse=None,
            arc_points=contour,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={'used': False, 'reason': 'fallback_no_source_ellipse', 'arc_weights': [1.0] * len(contour)},
        )

    center = _core_center_from_contour(gray_light, obs.contour, source_ellipse)
    _, _, base_angles = _sample_ellipse_points_and_normals(source_ellipse, None)

    records = []
    for ang in base_angles:
        rmax = _ray_exit_radius(obs.contour, center, float(ang))
        prof = _sample_radial_profile_bic(gray_light, center, float(ang), rmax)
        if prof is None:
            continue
        edge = _best_piecewise_model_bic(prof['r'], prof['profile'])
        if edge is None or not np.isfinite(edge['radius']):
            continue
        pt = center + edge['radius'] * prof['direction']
        if not np.all(np.isfinite(pt)):
            continue
        records.append({
            'angle': float(ang),
            'point': pt.astype(np.float64),
            'radius': float(edge['radius']),
            'conf': float(edge['confidence']),
            'rmse': float(edge['rmse']),
            'model': edge['model'],
            'a': int(edge['a']),
            'b': int(edge['b']),
            'bic_weights': np.asarray(edge['weights'], dtype=np.float64),
            'bic_values': np.asarray(edge['bic'], dtype=np.float64),
            'slopes': [float(edge.get('s1', 0.0)), float(edge.get('s2', 0.0)), float(edge.get('s3', 0.0))],
        })

    if len(records) < 6:
        contour = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(contour), endpoint=False, dtype=np.float64) if len(contour) else np.empty((0,), dtype=np.float64)
        return ArcSelectionResult(
            work_contour=contour.astype(np.float32).reshape(-1, 1, 2),
            work_ellipse=source_ellipse,
            arc_points=contour,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={
                'used': False,
                'reason': 'fallback_too_few_bic_profiles',
                'core_center': center.tolist(),
                'arc_weights': [1.0] * len(contour),
                'confidence_light': 0.0,
                'risk_light': 1.0,
            },
        )

    records.sort(key=lambda d: d['angle'])
    angles = np.asarray([d['angle'] for d in records], dtype=np.float64)
    points = np.asarray([d['point'] for d in records], dtype=np.float64)
    conf = np.asarray([d['conf'] for d in records], dtype=np.float64)
    conf = np.clip(conf, 1e-9, 1.0)

    radii = np.asarray([d['radius'] for d in records], dtype=np.float64)
    radii_sm = _circular_smooth_valid(radii, conf, int(max(1, min(len(radii) // 12, 7))))
    directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)
    points_sm = center.reshape(1, 2) + radii_sm[:, None] * directions

    fit_e = _fit_ellipse_weighted_by_conf(points_sm, conf)
    if fit_e is None:
        fit_e = source_ellipse
    work_contour = points_sm.astype(np.float32).reshape(-1, 1, 2)

    residuals = np.array([_ellipse_contour_residual(fit_e, points_sm[i:i+1]) for i in range(len(points_sm))], dtype=np.float64)
    residuals[~np.isfinite(residuals)] = np.nan
    res_scale = float(np.nanmedian(residuals)) if np.any(np.isfinite(residuals)) else np.nan

    confidence_light = float(np.exp(np.mean(np.log(conf))))
    risk_light = float(1.0 - confidence_light)
    model_counts = {
        'M1': int(sum(1 for r in records if r['model'] == 'M1')),
        'M2': int(sum(1 for r in records if r['model'] == 'M2')),
        'M3': int(sum(1 for r in records if r['model'] == 'M3')),
    }
    work_angles = angles.astype(np.float64)
    coverage = _effective_arc_coverage_from_angles(work_angles)
    balance = _angle_balance_from_angles(work_angles)

    debug = {
        'used': True,
        'reason': 'bic_piecewise_radial_shoulder_v5',
        'core_center': center.tolist(),
        'num_profiles': int(len(records)),
        'selected_arc_points_count': int(len(points_sm)),
        'selected_arc_angles_count': int(len(work_angles)),
        'effective_arc_coverage': float(coverage),
        'angle_balance': float(balance),
        'profile_rmse_mean': float(np.mean([r['rmse'] for r in records])),
        'profile_rmse_median': float(np.median([r['rmse'] for r in records])),
        'conf_ray_mean': float(np.mean(conf)),
        'conf_ray_min': float(np.min(conf)),
        'conf_ray_max': float(np.max(conf)),
        'confidence_light': confidence_light,
        'risk_light': risk_light,
        'bic_model_counts': model_counts,
        'ellipse_residual_median': None if not np.any(np.isfinite(residuals)) else float(np.nanmedian(residuals)),
        'arc_weights': conf.astype(np.float64).tolist(),
    }

    score = float(np.nanmedian(residuals)) if np.any(np.isfinite(residuals)) else np.inf
    return ArcSelectionResult(
        work_contour=work_contour,
        work_ellipse=fit_e,
        arc_points=points_sm.astype(np.float64),
        arc_angles=work_angles,
        arc_score=score,
        score_gap=np.nan,
        debug=debug,
    )


def select_physical_edges(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[ArcSelectionResult], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    results = []
    per_light = []
    for obs in observations:
        res = _detect_shoulder_arc_for_observation_bic(gray_light, obs)
        results.append(res)
        per_light.append(res.debug)
    global_debug = {
        'mode': 'bic_piecewise_radial_shoulder_v5',
        'num_lights': int(len(observations)),
        'per_light': per_light,
    }
    return results, global_debug


def build_selected_measurements_for_pipeline(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    arc_results, global_debug = select_physical_edges(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
        circle_centers_w=circle_centers_w,
    )
    out = []
    for obs, sel in zip(observations, arc_results):
        out.append({
            'source_contour': obs.contour,
            'source_ellipse': obs.source_ellipse,
            'bbox_center': obs.bbox_center,
            'gray_center': obs.gray_center,
            'work_contour': sel.work_contour,
            'work_ellipse': sel.work_ellipse,
            'arc_points': sel.arc_points,
            'arc_angles': sel.arc_angles,
            'arc_weights': np.asarray(sel.debug.get('arc_weights', []), dtype=np.float64),
            'rep_dbg': sel.debug,
        })
    return out, global_debug




# ============================================================
# v6 覆盖：锁定“平台结束后的第一强下降拐点”，不再追逐整圈平滑外包络
# ============================================================

def _smooth_profile_shoulder(y: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if yy.size <= 3:
        return yy.copy()
    k = int(max(3, 2 * (yy.size // 32) + 1))
    if k % 2 == 0:
        k += 1
    k = min(k, yy.size if yy.size % 2 == 1 else yy.size - 1)
    if k < 3:
        return yy.copy()
    prof = yy.astype(np.float32).reshape(1, -1)
    sm = cv2.GaussianBlur(prof, (k, 1), 0.0).reshape(-1).astype(np.float64)
    return sm


def _mad1d(x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    if xx.size == 0:
        return 0.0
    med = np.median(xx)
    return float(np.median(np.abs(xx - med)))


def _otsu_mask_from_values(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.zeros((0,), dtype=bool)
    if not np.any(np.isfinite(v)):
        return np.zeros_like(v, dtype=bool)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.zeros_like(v, dtype=bool)
    lo = float(np.min(vv))
    hi = float(np.max(vv))
    if hi - lo <= np.finfo(np.float64).eps:
        return np.ones_like(v, dtype=bool)
    u8 = np.clip(np.round((v - lo) / (hi - lo) * 255.0), 0.0, 255.0).astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_n = lo + (hi - lo) * (float(thr) / 255.0)
    return np.asarray(v >= thr_n, dtype=bool)


def _best_piecewise_model_shoulder_locked(r: np.ndarray, y: np.ndarray):
    x = np.asarray(r, dtype=np.float64).reshape(-1)
    yy = _smooth_profile_shoulder(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size != yy.size or x.size < 12:
        return None

    model = _best_piecewise_model_bic(x, yy)
    if model is None:
        return None
    if model.get('model') not in ('M2', 'M3'):
        return None

    a = int(model.get('a', 0))
    b = int(model.get('b', a))
    a = int(np.clip(a, 1, len(x) - 2))
    b = int(np.clip(b, a + 1, len(x) - 1))

    g = np.gradient(yy, x)
    grad_noise = _mad1d(g) + np.finfo(np.float64).eps

    plateau = yy[:a]
    descent = yy[a:b + 1] if b >= a else yy[a:a + 1]
    tail = yy[b:] if b < len(yy) else yy[-1:]

    plateau_var = _mad1d(np.gradient(plateau)) if plateau.size >= 3 else _mad1d(g[:max(a, 1)])
    plateau_support = float(np.exp(-plateau_var / grad_noise))

    descent_grad = -g[a:b + 1] if b >= a else np.maximum(-g[a:], 0.0)
    if descent_grad.size == 0:
        return None
    descent_strength = float(np.max(np.maximum(descent_grad, 0.0)))
    descent_support = float(1.0 - np.exp(-descent_strength / grad_noise))

    tail_grad = _mad1d(np.gradient(tail)) if tail.size >= 3 else _mad1d(g[b:])
    tail_support = float(np.exp(-tail_grad / grad_noise))

    dynamic = float(np.max(yy) - np.min(yy))
    if dynamic <= np.finfo(np.float64).eps:
        return None

    shoulder_idx = a
    shoulder_radius = float(x[shoulder_idx])
    shoulder_level = float(np.clip((yy[shoulder_idx] - np.min(yy)) / dynamic, 0.0, 1.0))
    plateau_ratio = float(a / max(len(x) - 1, 1))
    transition_ratio = float((b - a) / max(len(x) - 1, 1))

    model_conf = float(np.clip(model.get('confidence', 0.0), 0.0, 1.0))
    # geometric mean of data-driven supports
    conf = float(np.exp(np.mean(np.log(np.clip(np.array([model_conf, plateau_support, descent_support, tail_support]), 1e-12, 1.0)))))

    return {
        'model': model.get('model'),
        'radius': shoulder_radius,
        'confidence': conf,
        'rmse': float(model.get('rmse', np.nan)),
        'bic': np.asarray(model.get('bic', []), dtype=np.float64),
        'weights': np.asarray(model.get('weights', []), dtype=np.float64),
        'a': a,
        'b': b,
        's1': float(model.get('s1', 0.0)),
        's2': float(model.get('s2', 0.0)),
        's3': float(model.get('s3', 0.0)),
        'plateau_support': plateau_support,
        'descent_support': descent_support,
        'tail_support': tail_support,
        'plateau_ratio': plateau_ratio,
        'transition_ratio': transition_ratio,
        'shoulder_level': shoulder_level,
        'grad_noise': float(grad_noise),
        'descent_strength': descent_strength,
        'profile_smooth': yy.astype(np.float64),
        'gradient': g.astype(np.float64),
    }


def _fit_ellipse_iterative_shoulder(points: np.ndarray, weights: np.ndarray) -> Optional[Tuple]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(pts) < 6:
        return None
    if len(ww) != len(pts) or not np.all(np.isfinite(ww)):
        ww = np.ones((len(pts),), dtype=np.float64)
    ww = np.clip(ww, 1e-9, None)
    ww = ww / (np.mean(ww) + 1e-12)

    keep = np.ones((len(pts),), dtype=bool)
    best = None
    for _ in range(3):
        cur_pts = pts[keep]
        cur_w = ww[keep]
        if len(cur_pts) < 6:
            break
        e = _fit_ellipse_weighted_by_conf(cur_pts, cur_w)
        if e is None:
            break
        best = e
        resid = np.array([_ellipse_contour_residual(e, cur_pts[i:i+1]) for i in range(len(cur_pts))], dtype=np.float64)
        rs = _mad1d(resid) + np.finfo(np.float64).eps
        rw = np.exp(-0.5 * (resid / rs) ** 2)
        mask_local = _otsu_mask_from_values(rw)
        if np.count_nonzero(mask_local) < 6:
            break
        full_keep = np.zeros_like(keep)
        full_keep[np.flatnonzero(keep)[mask_local]] = True
        keep = full_keep
    return best


def _detect_shoulder_arc_for_observation_bic(gray_light: np.ndarray, obs: LightObservation) -> ArcSelectionResult:
    source_ellipse = obs.source_ellipse
    contour = np.asarray(obs.contour, dtype=np.float64).reshape(-1, 2)
    if source_ellipse is None:
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(contour), endpoint=False, dtype=np.float64) if len(contour) else np.empty((0,), dtype=np.float64)
        return ArcSelectionResult(
            work_contour=contour.astype(np.float32).reshape(-1, 1, 2),
            work_ellipse=None,
            arc_points=contour,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={'used': False, 'reason': 'fallback_no_source_ellipse', 'arc_weights': [1.0] * len(contour), 'confidence_light': 0.0, 'plateau_support_mean': 0.0},
        )

    center = _core_center_from_contour(gray_light, obs.contour, source_ellipse)
    _, _, base_angles = _sample_ellipse_points_and_normals(source_ellipse, None)

    raw_records = []
    for ang in base_angles:
        rmax = _ray_exit_radius(obs.contour, center, float(ang))
        prof = _sample_radial_profile_bic(gray_light, center, float(ang), rmax)
        if prof is None:
            continue
        edge = _best_piecewise_model_shoulder_locked(prof['r'], prof['profile'])
        if edge is None or not np.isfinite(edge['radius']):
            continue
        pt = center + edge['radius'] * prof['direction']
        if not np.all(np.isfinite(pt)):
            continue
        raw_records.append({
            'angle': float(ang),
            'point': pt.astype(np.float64),
            'radius': float(edge['radius']),
            'conf': float(edge['confidence']),
            'rmse': float(edge['rmse']),
            'model': edge['model'],
            'a': int(edge['a']),
            'b': int(edge['b']),
            'bic_weights': np.asarray(edge['weights'], dtype=np.float64),
            'bic_values': np.asarray(edge['bic'], dtype=np.float64),
            'slopes': [float(edge.get('s1', 0.0)), float(edge.get('s2', 0.0)), float(edge.get('s3', 0.0))],
            'plateau_support': float(edge['plateau_support']),
            'descent_support': float(edge['descent_support']),
            'tail_support': float(edge['tail_support']),
            'plateau_ratio': float(edge['plateau_ratio']),
            'transition_ratio': float(edge['transition_ratio']),
            'shoulder_level': float(edge['shoulder_level']),
            'grad_noise': float(edge['grad_noise']),
            'descent_strength': float(edge['descent_strength']),
        })

    if len(raw_records) < 6:
        arc_angles = np.linspace(0.0, 2.0 * np.pi, len(contour), endpoint=False, dtype=np.float64) if len(contour) else np.empty((0,), dtype=np.float64)
        return ArcSelectionResult(
            work_contour=contour.astype(np.float32).reshape(-1, 1, 2),
            work_ellipse=source_ellipse,
            arc_points=contour,
            arc_angles=arc_angles,
            arc_score=np.inf,
            score_gap=np.nan,
            debug={
                'used': False,
                'reason': 'fallback_too_few_shoulder_profiles_v6',
                'core_center': center.tolist(),
                'arc_weights': [1.0] * len(contour),
                'confidence_light': 0.0,
                'risk_light': 1.0,
                'plateau_support_mean': 0.0,
            },
        )

    raw_records.sort(key=lambda d: d['angle'])
    conf_all = np.asarray([r['conf'] for r in raw_records], dtype=np.float64)
    plateau_all = np.asarray([r['plateau_support'] for r in raw_records], dtype=np.float64)
    joint_score = np.sqrt(np.clip(conf_all, 0.0, 1.0) * np.clip(plateau_all, 0.0, 1.0))
    mask = _otsu_mask_from_values(joint_score)
    if np.count_nonzero(mask) >= 6:
        records = [r for r, k in zip(raw_records, mask.tolist()) if k]
    else:
        records = list(raw_records)

    angles = np.asarray([d['angle'] for d in records], dtype=np.float64)
    points = np.asarray([d['point'] for d in records], dtype=np.float64)
    weights = np.asarray([np.sqrt(max(d['conf'], 1e-12) * max(d['plateau_support'], 1e-12)) for d in records], dtype=np.float64)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / (np.mean(weights) + 1e-12)

    fit_e = _fit_ellipse_iterative_shoulder(points, weights)
    if fit_e is None:
        fit_e = source_ellipse
    work_contour = points.astype(np.float32).reshape(-1, 1, 2)

    residuals = np.array([_ellipse_contour_residual(fit_e, points[i:i+1]) for i in range(len(points))], dtype=np.float64)
    residuals[~np.isfinite(residuals)] = np.nan
    res_med = float(np.nanmedian(residuals)) if np.any(np.isfinite(residuals)) else np.inf

    coverage = _effective_arc_coverage_from_angles(angles)
    balance = _angle_balance_from_angles(angles)
    conf_light = float(np.exp(np.mean(np.log(np.clip(np.asarray([r['conf'] for r in records], dtype=np.float64), 1e-12, 1.0)))))
    plateau_mean = float(np.mean([r['plateau_support'] for r in records]))
    tail_mean = float(np.mean([r['tail_support'] for r in records]))
    selected_ratio = float(len(records) / max(len(raw_records), 1))
    confidence_light = float(np.clip(conf_light * np.sqrt(max(plateau_mean, 1e-12)) * np.sqrt(max(selected_ratio, 1e-12)), 0.0, 1.0))
    risk_light = float(1.0 - confidence_light)

    debug = {
        'used': True,
        'reason': 'plateau_locked_first_descent_shoulder_v6',
        'core_center': center.tolist(),
        'num_profiles_total': int(len(raw_records)),
        'num_profiles_selected': int(len(records)),
        'selected_ratio': selected_ratio,
        'selected_arc_points_count': int(len(points)),
        'selected_arc_angles_count': int(len(angles)),
        'effective_arc_coverage': float(coverage),
        'angle_balance': float(balance),
        'profile_rmse_mean': float(np.mean([r['rmse'] for r in records])),
        'profile_rmse_median': float(np.median([r['rmse'] for r in records])),
        'conf_ray_mean': float(np.mean([r['conf'] for r in records])),
        'conf_ray_min': float(np.min([r['conf'] for r in records])),
        'conf_ray_max': float(np.max([r['conf'] for r in records])),
        'plateau_support_mean': plateau_mean,
        'plateau_support_min': float(np.min([r['plateau_support'] for r in records])),
        'descent_support_mean': float(np.mean([r['descent_support'] for r in records])),
        'tail_support_mean': tail_mean,
        'shoulder_level_mean': float(np.mean([r['shoulder_level'] for r in records])),
        'plateau_ratio_mean': float(np.mean([r['plateau_ratio'] for r in records])),
        'transition_ratio_mean': float(np.mean([r['transition_ratio'] for r in records])),
        'confidence_light': confidence_light,
        'risk_light': risk_light,
        'arc_weights': weights.astype(np.float64).tolist(),
        'ellipse_residual_median': None if not np.any(np.isfinite(residuals)) else float(np.nanmedian(residuals)),
    }

    return ArcSelectionResult(
        work_contour=work_contour,
        work_ellipse=fit_e,
        arc_points=points.astype(np.float64),
        arc_angles=angles.astype(np.float64),
        arc_score=res_med,
        score_gap=np.nan,
        debug=debug,
    )


def _select_physical_edges_from_observations(gray_light: np.ndarray, observations: Sequence[LightObservation]) -> Tuple[List[ArcSelectionResult], Dict[str, Any]]:
    results = []
    per_light = []
    for obs in observations:
        res = _detect_shoulder_arc_for_observation_bic(gray_light, obs)
        results.append(res)
        per_light.append(res.debug)
    global_debug = {
        'mode': 'plateau_locked_first_descent_shoulder_v6_fast',
        'num_lights': int(len(observations)),
        'per_light': per_light,
    }
    return results, global_debug


def select_physical_edges(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
    observations: Optional[Sequence[LightObservation]] = None,
) -> Tuple[List[ArcSelectionResult], Dict[str, Any]]:
    obs_seq = observations
    if obs_seq is None:
        obs_seq = build_light_observations(
            gray_light=gray_light,
            contours=contours,
            K=K,
            plane_normal_camera=plane_normal_camera,
            plane_distance=plane_distance,
        )
    return _select_physical_edges_from_observations(gray_light, obs_seq)


def build_selected_measurements_for_pipeline(
    gray_light: np.ndarray,
    contours: Sequence[np.ndarray],
    K: np.ndarray,
    plane_normal_camera: np.ndarray,
    plane_distance: float,
    circle_centers_w: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    observations = build_light_observations(
        gray_light=gray_light,
        contours=contours,
        K=K,
        plane_normal_camera=plane_normal_camera,
        plane_distance=plane_distance,
    )
    arc_results, global_debug = _select_physical_edges_from_observations(gray_light, observations)
    out = []
    for obs, sel in zip(observations, arc_results):
        out.append({
            'source_contour': obs.contour,
            'source_ellipse': obs.source_ellipse,
            'bbox_center': obs.bbox_center,
            'gray_center': obs.gray_center,
            'work_contour': sel.work_contour,
            'work_ellipse': sel.work_ellipse,
            'arc_points': sel.arc_points,
            'arc_angles': sel.arc_angles,
            'arc_weights': np.asarray(sel.debug.get('arc_weights', []), dtype=np.float64),
            'rep_dbg': sel.debug,
        })
    return out, global_debug
