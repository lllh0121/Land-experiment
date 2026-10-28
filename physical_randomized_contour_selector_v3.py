
import cv2
import numpy as np
from typing import Dict, Any, Optional

GAUSS_95_Z = 1.959963984540054
EPS = np.sqrt(np.finfo(np.float64).eps)


def _normalize_contour(contour) -> np.ndarray:
    arr = np.asarray(contour, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.reshape(-1, 2)
    else:
        arr = arr.reshape(-1, 2)
    return arr.astype(np.float64)


def _deserialize_cv_ellipse(obj: Optional[Dict[str, Any]]):
    if obj is None:
        return None
    try:
        center = obj.get('center', None)
        axes = obj.get('axes', None)
        angle_deg = obj.get('angle_deg', None)
        if center is None or axes is None or angle_deg is None:
            return None
        return ((float(center[0]), float(center[1])), (float(axes[0]), float(axes[1])), float(angle_deg))
    except Exception:
        return None


def _safe_fit_ellipse(pts_xy: np.ndarray):
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 6:
        return None
    try:
        return cv2.fitEllipse(pts)
    except cv2.error:
        return None


def _ellipse_point_from_angle(ellipse, theta: np.ndarray) -> np.ndarray:
    (cx, cy), (MA, ma), ang_deg = ellipse
    a = max(float(MA) * 0.5, EPS)
    b = max(float(ma) * 0.5, EPS)
    th = np.asarray(theta, dtype=np.float64)
    ang = np.deg2rad(float(ang_deg))
    c, s = np.cos(ang), np.sin(ang)
    ct, st = np.cos(th), np.sin(th)
    x = cx + a * ct * c - b * st * s
    y = cy + a * ct * s + b * st * c
    return np.column_stack([x, y]).astype(np.float64)


def _ellipse_param_angles(ellipse, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    if ellipse is None or len(pts) == 0:
        return np.zeros((0,), dtype=np.float64)
    (cx, cy), (MA, ma), ang_deg = ellipse
    a = max(float(MA) * 0.5, EPS)
    b = max(float(ma) * 0.5, EPS)
    ang = np.deg2rad(float(ang_deg))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    return np.arctan2(q[:, 1] / b, q[:, 0] / a).astype(np.float64)


def _ellipse_normal_from_angle(ellipse, theta: np.ndarray) -> np.ndarray:
    pts = _ellipse_point_from_angle(ellipse, theta)
    (cx, cy), (MA, ma), ang_deg = ellipse
    a = max(float(MA) * 0.5, EPS)
    b = max(float(ma) * 0.5, EPS)
    ang = np.deg2rad(float(ang_deg))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    q = (pts - np.array([cx, cy], dtype=np.float64)) @ R
    n_local = np.column_stack([q[:, 0] / (a * a), q[:, 1] / (b * b)])
    n_img = n_local @ R.T
    norms = np.linalg.norm(n_img, axis=1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return (n_img / norms).astype(np.float64)


def _ellipse_geom_scale(ellipse) -> float:
    if ellipse is None:
        return 1.0
    (_, _), (MA, ma), _ = ellipse
    return max(EPS, 0.25 * min(float(MA), float(ma)))


def _effective_arc_coverage_from_angles(angles: np.ndarray) -> float:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if ang.size < 2:
        return 0.0
    ang = np.sort(np.mod(ang, 2.0 * np.pi))
    gaps = np.diff(np.r_[ang, ang[0] + 2.0 * np.pi])
    if gaps.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.max(gaps) / (2.0 * np.pi), 0.0, 1.0))


def _support_observability(angles: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(ang)
    if n < 3:
        return {'rank': 0, 'cond': np.inf, 'coverage': 0.0, 'n_eff': float(n)}
    w = np.clip(w, EPS, None)
    M = np.column_stack([np.ones((n,), dtype=np.float64), np.cos(ang), np.sin(ang)])
    Mw = M * np.sqrt(w / (np.mean(w) + 1e-12))[:, None]
    try:
        s = np.linalg.svd(Mw, compute_uv=False)
    except np.linalg.LinAlgError:
        return {'rank': 0, 'cond': np.inf, 'coverage': 0.0, 'n_eff': 0.0}
    smax = float(s[0]) if s.size else 0.0
    tol = max(np.sqrt(np.finfo(np.float64).eps) * smax, EPS)
    rank = int(np.sum(s > tol))
    smin = float(s[-1]) if s.size else 0.0
    cond = float(smax / max(smin, EPS)) if smax > 0.0 else np.inf
    n_eff = float((np.sum(w) ** 2) / max(np.sum(w * w), EPS))
    coverage = _effective_arc_coverage_from_angles(ang)
    return {'rank': rank, 'cond': cond, 'coverage': float(coverage), 'n_eff': n_eff}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    w = np.clip(w[order], EPS, None)
    cw = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cw, 0.5, side='left'))
    idx = int(np.clip(idx, 0, len(x) - 1))
    return float(x[idx])


def _robust_scale_mad(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    med = _weighted_median(x, w)
    mad = _weighted_median(np.abs(x - med), w)
    scale = 1.4826 * float(mad)
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.sqrt(np.average((x - med) ** 2, weights=np.clip(w, EPS, None)) + EPS))
    return max(scale, EPS)


def _sobel_gradient(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def _bilinear_at(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = float(img[y0, x0])
    v10 = float(img[y0, x1])
    v01 = float(img[y1, x0])
    v11 = float(img[y1, x1])
    return (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v10 + (1.0 - dx) * dy * v01 + dx * dy * v11


def _solve_first_harmonic_offset(angles: np.ndarray, offsets: np.ndarray) -> Optional[np.ndarray]:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    d = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if len(ang) != 3 or len(d) != 3:
        return None
    A = np.column_stack([np.ones((3,), dtype=np.float64), np.cos(ang), np.sin(ang)])
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0:
        return None
    cond = float(s[0] / max(s[-1], EPS))
    if not np.isfinite(cond) or cond > 1.0 / np.sqrt(np.finfo(np.float64).eps):
        return None
    try:
        beta = np.linalg.solve(A, d)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta.astype(np.float64)


def _weighted_ls_first_harmonic(angles: np.ndarray, offsets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    d = np.asarray(offsets, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    A = np.column_stack([np.ones((len(ang),), dtype=np.float64), np.cos(ang), np.sin(ang)])
    w = np.clip(w, EPS, None)
    Aw = A * np.sqrt(w / (np.mean(w) + 1e-12))[:, None]
    dw = d * np.sqrt(w / (np.mean(w) + 1e-12))
    beta, _, _, _ = np.linalg.lstsq(Aw, dw, rcond=None)
    return np.asarray(beta, dtype=np.float64).reshape(3)


def _ellipse_area(ellipse) -> float:
    if ellipse is None:
        return np.inf
    (_, _), (MA, ma), _ = ellipse
    return float(np.pi * max(float(MA), EPS) * max(float(ma), EPS) * 0.25)


def _extract_support(selected_item: Dict[str, Any], theory_ellipse):
    arc_pts = _normalize_contour(selected_item.get('arc_points', None))
    arc_ang = np.asarray(selected_item.get('arc_angles', []), dtype=np.float64).reshape(-1)
    arc_w = np.asarray(selected_item.get('arc_weights', []), dtype=np.float64).reshape(-1)
    if arc_pts.shape[0] >= 6:
        if arc_ang.shape[0] != arc_pts.shape[0]:
            arc_ang = _ellipse_param_angles(theory_ellipse, arc_pts)
        if arc_w.shape[0] != arc_pts.shape[0]:
            arc_w = np.ones((arc_pts.shape[0],), dtype=np.float64)
        return arc_pts.astype(np.float64), arc_ang.astype(np.float64), np.clip(arc_w.astype(np.float64), EPS, None), 'legacy_arc_points'

    work_contour = _normalize_contour(selected_item.get('work_contour', None))
    if work_contour.shape[0] >= 6:
        ang = _ellipse_param_angles(theory_ellipse, work_contour)
        w = np.ones((work_contour.shape[0],), dtype=np.float64)
        return work_contour.astype(np.float64), ang, w, 'work_contour'

    source_contour = _normalize_contour(selected_item.get('source_contour', None))
    if source_contour.shape[0] >= 6:
        ang = _ellipse_param_angles(theory_ellipse, source_contour)
        w = np.ones((source_contour.shape[0],), dtype=np.float64)
        return source_contour.astype(np.float64), ang, w, 'source_contour'

    return np.empty((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), 'none'


def _smooth_profile_binomial(prof: np.ndarray) -> np.ndarray:
    x = np.asarray(prof, dtype=np.float64).reshape(-1)
    if x.size < 5:
        return x.copy()
    k = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    k /= np.sum(k)
    return np.convolve(np.pad(x, (2, 2), mode='edge'), k, mode='valid')


def _line_profile(gray: np.ndarray, base_pt: np.ndarray, normal: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    vals = [_bilinear_at(gray, float(base_pt[0] + s * normal[0]), float(base_pt[1] + s * normal[1])) for s in s_grid]
    return np.asarray(vals, dtype=np.float64)


def _halfmax_inward_edge(gray: np.ndarray, base_pt: np.ndarray, normal: np.ndarray, s_current: float, s_center: float, env_half: float) -> Optional[Dict[str, Any]]:
    # 在当前已接受边界与理论轮廓之间，利用模糊阶跃边在半高处对应真实边位置的性质，
    # 沿法线向内寻找更接近物理轮廓的交叉位置。
    search_low = min(s_center - env_half, s_current)
    search_high = s_current + env_half
    if not np.isfinite(search_low) or not np.isfinite(search_high) or search_high <= search_low + EPS:
        return None

    span = search_high - search_low
    n_samples = int(max(9, np.ceil(span / 0.5)))
    s_grid = np.linspace(search_low, search_high, n_samples, dtype=np.float64)
    prof = _line_profile(gray, base_pt, normal, s_grid)
    prof_s = _smooth_profile_binomial(prof)
    grad = np.gradient(prof_s, s_grid)

    inward_mask = s_grid <= min(0.5 * (search_low + s_current), s_current)
    outward_mask = s_grid >= s_current
    if np.count_nonzero(inward_mask) < 2 or np.count_nonzero(outward_mask) < 2:
        return None

    fg_val = float(np.median(prof_s[inward_mask]))
    bg_val = float(np.median(prof_s[outward_mask]))
    dyn = fg_val - bg_val
    if not np.isfinite(dyn) or dyn <= EPS:
        return None

    half = 0.5 * (fg_val + bg_val)
    valid_mask = s_grid <= s_current
    idx = np.where(valid_mask)[0]
    if idx.size < 2:
        return None

    cross = []
    val = prof_s - half
    for k in range(int(idx[0]), int(idx[-1])):
        v1 = val[k]
        v2 = val[k + 1]
        if v1 >= 0.0 and v2 <= 0.0:
            denom = max(abs(v1 - v2), EPS)
            t = float(v1 / denom)
            s_cross = float(s_grid[k] * (1.0 - t) + s_grid[k + 1] * t)
            g_local = float(max(-(grad[k] * (1.0 - t) + grad[k + 1] * t), 0.0))
            cross.append((s_cross, g_local))

    if len(cross) == 0:
        neg_idx = idx[grad[idx] < 0.0]
        if neg_idx.size == 0:
            return None
        k = int(neg_idx[np.argmin(prof_s[neg_idx])])
        s_cross = float(s_grid[k])
        g_local = float(max(-grad[k], 0.0))
    else:
        g_arr = np.asarray([c[1] for c in cross], dtype=np.float64)
        # 选择具有最大负梯度的半高交叉；这是对模糊阶跃边模型的直接实现。
        best = int(np.argmax(g_arr))
        s_cross, g_local = cross[best]

    if not np.isfinite(s_cross) or s_cross > s_current + EPS:
        return None

    pt = base_pt + s_cross * normal
    return {
        'point': pt.astype(np.float64),
        's_refined': float(s_cross),
        's_current': float(s_current),
        'fg_val': fg_val,
        'bg_val': bg_val,
        'half_val': float(half),
        'grad_mag': float(g_local),
        'dynamic_range': float(dyn),
    }


def _current_offset_model(angles: np.ndarray, offsets: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    beta = _weighted_ls_first_harmonic(angles, offsets, weights)
    pred = beta[0] + beta[1] * np.cos(angles) + beta[2] * np.sin(angles)
    resid = offsets - pred
    scale = _robust_scale_mad(resid, weights)
    return {'beta': beta, 'pred': pred, 'resid': resid, 'scale': float(scale)}


def _inward_refine_kept_points(gray: np.ndarray,
                               theory_ellipse,
                               keep_pts: np.ndarray,
                               keep_ang: np.ndarray,
                               keep_w: np.ndarray,
                               center_off: float,
                               env_half: float,
                               current_ellipse,
                               current_model: Dict[str, Any]) -> Dict[str, Any]:
    base = _ellipse_point_from_angle(theory_ellipse, keep_ang)
    normals = _ellipse_normal_from_angle(theory_ellipse, keep_ang)
    cur_off = np.sum((keep_pts - base) * normals, axis=1)

    refined_pts = keep_pts.copy()
    refined_off = cur_off.copy()
    per_point = []
    moved = np.zeros((len(keep_pts),), dtype=bool)

    for i in range(len(keep_pts)):
        ref = _halfmax_inward_edge(gray, base[i], normals[i], float(cur_off[i]), float(center_off), float(env_half))
        if ref is None:
            per_point.append({'used': False, 'reason': 'no_halfmax_edge'})
            continue
        s_new = float(ref['s_refined'])
        # 只接受向内修正，不允许向外扩张。
        if s_new <= cur_off[i] + EPS:
            refined_pts[i] = ref['point']
            refined_off[i] = s_new
            moved[i] = s_new < cur_off[i] - EPS
            per_point.append({'used': True, 'delta_inward': float(cur_off[i] - s_new), 'grad_mag': ref['grad_mag'], 'dynamic_range': ref['dynamic_range']})
        else:
            per_point.append({'used': False, 'reason': 'outward_solution'})

    if np.count_nonzero(np.isfinite(refined_off)) < 6:
        return {'accepted': False, 'reason': 'refined_points_too_few'}

    ell_ref = _safe_fit_ellipse(refined_pts)
    if ell_ref is None:
        return {'accepted': False, 'reason': 'ellipse_fit_failed'}

    area_cur = _ellipse_area(current_ellipse)
    area_ref = _ellipse_area(ell_ref)
    model_ref = _current_offset_model(keep_ang, refined_off, keep_w)

    median_cur = _weighted_median(cur_off, keep_w)
    median_ref = _weighted_median(refined_off, keep_w)
    inward_ok = median_ref <= median_cur + EPS
    stability_ok = model_ref['scale'] <= current_model['scale'] + EPS
    area_ok = area_ref <= area_cur + EPS

    accepted = bool(inward_ok and stability_ok and area_ok)
    return {
        'accepted': accepted,
        'reason': 'accepted' if accepted else 'not_better_than_current',
        'points': refined_pts,
        'offsets': refined_off,
        'ellipse': ell_ref,
        'area_cur': float(area_cur),
        'area_ref': float(area_ref),
        'scale_cur': float(current_model['scale']),
        'scale_ref': float(model_ref['scale']),
        'median_cur': float(median_cur),
        'median_ref': float(median_ref),
        'moved_count': int(np.count_nonzero(moved)),
        'per_point': per_point,
    }


def select_randomized_physical_contour(gray_light_proc: np.ndarray, selected_item: Dict[str, Any], light_radius_mm: float = 15.0) -> Dict[str, Any]:
    """
    物理约束随机轮廓假设搜索前端（v3，加入高曝光向内修正）。

    理论依据：
    1) 理论物理轮廓只定义当前帧边界假设的可行域中心，而不直接提供答案；
    2) 随机搜索阶段仅用于在已有图像支持弧点上估计低维法线偏移模型 delta(theta)=c0+c1*cos(theta)+s1*sin(theta)；
    3) 在高曝光下，已接受边界往往稳定但整体偏外；对模糊阶跃边，真实边位置更接近法线剖面的半高交叉位置；
    4) 因此在随机搜索选出稳定边族后，再沿理论法线执行一次受图像证据支撑的向内半高修正；
    5) 只有当向内修正同时满足“更向内、不降低稳定性、不放大面积”时才采用，否则保留当前已接受边界。
    """
    out = dict(selected_item)
    rep_dbg = dict(out.get('rep_dbg', {}))

    theory_ellipse = _deserialize_cv_ellipse(rep_dbg.get('assigned_pose_ellipse'))
    if theory_ellipse is None:
        theory_ellipse = out.get('work_ellipse', None)
    if theory_ellipse is None:
        theory_ellipse = out.get('source_ellipse', None)
    if theory_ellipse is None:
        rep_dbg['frontend_selector'] = {'used': False, 'reason': 'no_theory_ellipse'}
        out['rep_dbg'] = rep_dbg
        return out

    gray = np.asarray(gray_light_proc, dtype=np.float64)
    if gray.ndim != 2:
        rep_dbg['frontend_selector'] = {'used': False, 'reason': 'gray_invalid'}
        out['rep_dbg'] = rep_dbg
        return out

    support_pts, support_ang, support_w0, support_src = _extract_support(out, theory_ellipse)
    if support_pts.shape[0] < 6:
        rep_dbg['frontend_selector'] = {'used': False, 'reason': 'insufficient_support', 'support_source': support_src}
        out['rep_dbg'] = rep_dbg
        return out

    theory_base = _ellipse_point_from_angle(theory_ellipse, support_ang)
    theory_norm = _ellipse_normal_from_angle(theory_ellipse, support_ang)
    support_off = np.sum((support_pts - theory_base) * theory_norm, axis=1)

    gx, gy = _sobel_gradient(gray)
    grad_n = []
    for p, n in zip(support_pts, theory_norm):
        gxv = _bilinear_at(gx, p[0], p[1])
        gyv = _bilinear_at(gy, p[0], p[1])
        grad_n.append(abs(gxv * n[0] + gyv * n[1]))
    grad_n = np.asarray(grad_n, dtype=np.float64)
    grad_scale = max(np.median(grad_n[grad_n > 0]) if np.any(grad_n > 0) else 1.0, EPS)
    grad_w = np.clip(grad_n / grad_scale, EPS, None)
    support_w = np.clip(support_w0 * grad_w, EPS, None)

    center_off = _weighted_median(support_off, support_w)
    scale_off = _robust_scale_mad(support_off, support_w)
    env_half = max(GAUSS_95_Z * scale_off, EPS)
    base_gate = np.abs(support_off - center_off) <= env_half
    if np.count_nonzero(base_gate) < 6:
        base_gate = np.isfinite(support_off)
    pts_s = support_pts[base_gate]
    ang_s = support_ang[base_gate]
    off_s = support_off[base_gate]
    w_s = support_w[base_gate]
    if len(pts_s) < 6:
        rep_dbg['frontend_selector'] = {'used': False, 'reason': 'insufficient_support_after_gate', 'support_source': support_src}
        out['rep_dbg'] = rep_dbg
        return out

    off_hi = center_off + env_half
    off_lo = center_off - env_half
    (_, _), (MA0, ma0), _ = theory_ellipse
    area_upper = float(np.pi * max(MA0 * 0.5 + off_hi, EPS) * max(ma0 * 0.5 + off_hi, EPS))
    area_lower = float(np.pi * max(MA0 * 0.5 + off_lo, EPS) * max(ma0 * 0.5 + off_lo, EPS))

    beta_base = _weighted_ls_first_harmonic(ang_s, off_s, w_s)

    def evaluate_beta(beta: np.ndarray):
        pred_off = beta[0] + beta[1] * np.cos(ang_s) + beta[2] * np.sin(ang_s)
        env_violation = np.maximum(np.abs(pred_off - center_off) - env_half, 0.0) / env_half
        resid = off_s - pred_off
        res_scale = _robust_scale_mad(resid, w_s)
        z = resid / res_scale
        robust = 1.0 / (1.0 + z * z)
        inlier_mask = (np.abs(z) <= GAUSS_95_Z) & (env_violation <= 0.0)
        if np.count_nonzero(inlier_mask) < 6:
            return {
                'score': 0.0,
                'resid': resid,
                'res_scale': float(res_scale),
                'inlier_mask': inlier_mask,
                'coverage': 0.0,
                'cond': np.inf,
                'area': np.inf,
                'area_violation': np.inf,
                'env_violation_mean': float(np.mean(env_violation)),
            }
        keep_pts = pts_s[inlier_mask]
        keep_ang = ang_s[inlier_mask]
        keep_w = w_s[inlier_mask]
        obs_i = _support_observability(keep_ang, keep_w)
        cond = float(obs_i.get('cond', np.inf))
        conditioning = 0.0 if (not np.isfinite(cond) or cond <= 0.0) else float(1.0 / (1.0 + np.log1p(cond)))
        ell = _safe_fit_ellipse(keep_pts)
        area = _ellipse_area(ell)
        if not np.isfinite(area):
            area = np.inf
        area_violation = 0.0
        if np.isfinite(area):
            if area > area_upper:
                area_violation = (area - area_upper) / max(area_upper, EPS)
            elif area < area_lower:
                area_violation = (area_lower - area) / max(area_lower, EPS)
        image_support = float(np.average(robust[inlier_mask], weights=keep_w))
        envelope_penalty = float(1.0 / (1.0 + np.mean(env_violation[inlier_mask] * env_violation[inlier_mask])))
        area_penalty = float(1.0 / (1.0 + area_violation * area_violation))
        coverage = float(obs_i.get('coverage', 0.0))
        score = image_support * np.sqrt(max(coverage, 0.0)) * (0.5 + 0.5 * conditioning) * envelope_penalty * area_penalty
        return {
            'score': float(score),
            'resid': resid,
            'res_scale': float(res_scale),
            'inlier_mask': inlier_mask,
            'coverage': coverage,
            'cond': cond,
            'area': float(area),
            'area_violation': float(area_violation),
            'env_violation_mean': float(np.mean(env_violation[inlier_mask])) if np.count_nonzero(inlier_mask) > 0 else float(np.mean(env_violation)),
        }

    base_eval = evaluate_beta(beta_base)
    best_beta = beta_base.copy()
    best_eval = dict(base_eval)

    obs = _support_observability(ang_s, w_s)
    n_support = len(ang_s)
    trial_count = int(max(3, min(n_support, int(np.ceil(max(obs.get('n_eff', 0.0), 3.0))))))
    w_sum = float(np.sum(w_s))
    if not np.isfinite(w_sum) or w_sum <= EPS:
        prob = np.full(n_support, 1.0 / max(n_support, 1), dtype=np.float64)
    else:
        prob = np.clip(w_s / w_sum, EPS, 1.0)
        prob_sum = float(np.sum(prob))
        if not np.isfinite(prob_sum) or prob_sum <= EPS:
            prob = np.full(n_support, 1.0 / max(n_support, 1), dtype=np.float64)
        else:
            prob = prob / prob_sum
    rng_seed = int(np.round(np.sum(np.abs(np.asarray(theory_ellipse[0], dtype=np.float64))) * 1000.0)) % (2 ** 32 - 1)
    rng = np.random.default_rng(rng_seed)

    for _ in range(trial_count):
        idx = rng.choice(n_support, size=3, replace=False, p=prob)
        beta = _solve_first_harmonic_offset(ang_s[idx], off_s[idx])
        if beta is None:
            continue
        trial_eval = evaluate_beta(beta)
        better = trial_eval['score'] > best_eval['score']
        no_worse_shape = trial_eval['area_violation'] <= best_eval['area_violation'] and trial_eval['res_scale'] <= max(base_eval['res_scale'], EPS)
        if better and no_worse_shape:
            best_beta = beta
            best_eval = trial_eval

    adopt_random = (
        best_eval['score'] > base_eval['score'] and
        best_eval['res_scale'] <= base_eval['res_scale'] and
        best_eval['area_violation'] <= base_eval['area_violation'] and
        best_eval['coverage'] >= base_eval['coverage']
    )
    if not adopt_random:
        best_beta = beta_base
        best_eval = base_eval

    keep_mask = np.asarray(best_eval['inlier_mask'], dtype=bool)
    if np.count_nonzero(keep_mask) < 6:
        rep_dbg['frontend_selector'] = {
            'used': False,
            'reason': 'final_inliers_too_few',
            'support_source': support_src,
            'support_count': int(n_support),
            'inlier_count': int(np.count_nonzero(keep_mask)),
        }
        out['rep_dbg'] = rep_dbg
        return out

    keep_pts = pts_s[keep_mask]
    keep_ang = ang_s[keep_mask]
    keep_w = np.clip(w_s[keep_mask], EPS, None)
    order = np.argsort(np.mod(keep_ang, 2.0 * np.pi))
    keep_pts = keep_pts[order]
    keep_ang = keep_ang[order]
    keep_w = keep_w[order]

    work_ellipse = _safe_fit_ellipse(keep_pts)
    if work_ellipse is None:
        rep_dbg['frontend_selector'] = {
            'used': False,
            'reason': 'fit_ellipse_failed_after_selection',
            'support_source': support_src,
        }
        out['rep_dbg'] = rep_dbg
        return out

    area_final = _ellipse_area(work_ellipse)
    veto = False
    veto_reason = None
    if not np.isfinite(area_final) or area_final < area_lower or area_final > area_upper:
        veto = True
        veto_reason = 'area_outside_physical_envelope'
    elif best_eval['res_scale'] > base_eval['res_scale']:
        veto = True
        veto_reason = 'residual_worse_than_baseline'
    elif best_eval['coverage'] < base_eval['coverage']:
        veto = True
        veto_reason = 'coverage_worse_than_baseline'

    if veto:
        rep_dbg['frontend_selector'] = {
            'used': False,
            'reason': veto_reason,
            'support_source': support_src,
            'support_count': int(n_support),
            'inlier_count': int(np.count_nonzero(keep_mask)),
            'base_score': float(base_eval['score']),
            'best_score': float(best_eval['score']),
            'base_res_scale': float(base_eval['res_scale']),
            'best_res_scale': float(best_eval['res_scale']),
        }
        out['rep_dbg'] = rep_dbg
        return out

    current_base = _ellipse_point_from_angle(theory_ellipse, keep_ang)
    current_norm = _ellipse_normal_from_angle(theory_ellipse, keep_ang)
    current_off = np.sum((keep_pts - current_base) * current_norm, axis=1)
    current_model = _current_offset_model(keep_ang, current_off, keep_w)
    inward = _inward_refine_kept_points(gray, theory_ellipse, keep_pts, keep_ang, keep_w, center_off, env_half, work_ellipse, current_model)

    if inward.get('accepted', False):
        keep_pts = np.asarray(inward['points'], dtype=np.float64)
        work_ellipse = inward['ellipse']
        area_final = inward['area_ref']

    out['work_ellipse'] = work_ellipse
    out['work_contour'] = keep_pts.reshape(-1, 1, 2).astype(np.float64)
    out['arc_points'] = keep_pts.astype(np.float64)
    out['arc_angles'] = keep_ang.astype(np.float64)
    out['arc_weights'] = (keep_w / (np.mean(keep_w) + 1e-12)).astype(np.float64)

    rep_dbg['frontend_selector'] = {
        'used': True,
        'mode': 'physics_constrained_randomized_contour_hypothesis_search_v3_with_inward_halfmax_refiner',
        'support_source': support_src,
        'offset_model': 'delta(theta)=c0+c1*cos(theta)+s1*sin(theta)',
        'random_minimal_sample_size': 3,
        'random_trial_count': int(trial_count),
        'support_count': int(n_support),
        'kept_count': int(len(keep_pts)),
        'coverage_support': float(obs.get('coverage', 0.0)),
        'coverage_kept': float(best_eval['coverage']),
        'observability_cond_support': float(obs.get('cond', np.inf)),
        'base_score': float(base_eval['score']),
        'best_score': float(best_eval['score']),
        'res_scale_base': float(base_eval['res_scale']),
        'res_scale_best': float(best_eval['res_scale']),
        'area_upper': float(area_upper),
        'area_lower': float(area_lower),
        'area_final': float(area_final),
        'beta': [float(best_beta[0]), float(best_beta[1]), float(best_beta[2])],
        'light_radius_mm': float(light_radius_mm),
        'inward_refiner': {
            'accepted': bool(inward.get('accepted', False)),
            'reason': inward.get('reason', 'not_run'),
            'moved_count': int(inward.get('moved_count', 0)),
            'area_cur': float(inward.get('area_cur', area_final)),
            'area_ref': float(inward.get('area_ref', area_final)),
            'scale_cur': float(inward.get('scale_cur', current_model['scale'])),
            'scale_ref': float(inward.get('scale_ref', current_model['scale'])),
            'median_cur': float(inward.get('median_cur', _weighted_median(current_off, keep_w))),
            'median_ref': float(inward.get('median_ref', _weighted_median(current_off, keep_w))),
        },
    }
    out['rep_dbg'] = rep_dbg
    return out
