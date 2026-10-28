"""Unchanged numerical stage helpers extracted from method_v3_experiment.py.

Only the four inference stages are exposed. No baseline or ground-truth runner.
"""
import cv2
import numpy as np
import pagc_experiment_baselines_v17_with_pose_eval_inputs_bias1d_vpaxis as core

BIAS1D_CONFIG = {
    "shift_lambda": 1.0, "max_shift_px": 7.0, "min_shift_px": 0.25,
    "sign_mode": "auto", "default_sign": 1.0,
    "allow_default_sign_fallback": False, "min_vp_sign_signal_px": 20.0,
    "min_axis_anisotropy": 0.005, "max_axis_ratio_q_for_correction": 0.96,
    "min_shape_signal_px": 0.40, "shift_mode": "global_median",
}

def _finite_center(value):
    try:
        p = np.asarray(value, dtype=np.float64).reshape(-1)
        if p.size >= 2 and np.all(np.isfinite(p[:2])):
            return p[:2].copy()
    except Exception:
        pass
    return np.array([np.nan, np.nan], dtype=np.float64)


def _roi_mask(image_shape, roi_points):
    if roi_points is None:
        return None
    points = np.asarray(roi_points, dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        return None
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def _ordered_by_layout(values, observation_to_layout):
    values = list(values)
    layout_ids = [int(x) for x in observation_to_layout]
    if len(values) != 4 or len(layout_ids) != 4 or sorted(layout_ids) != [0, 1, 2, 3]:
        raise RuntimeError("A complete one-to-one four-lamp correspondence is required.")
    out = [None] * 4
    for value, layout_id in zip(values, layout_ids):
        out[layout_id] = value
    return out


def _method_v3_subpixel_stage(gray, selected_item, proj_ctx):
    """Run Section 3.3 directly on the boundary selected in Section 3.2."""
    measurement = core.build_candidate_measurement_from_preselected(selected_item, gray)

    # B_i is the selected multilevel contour and E_i^sup is its fitted ellipse.
    # Suppress the archived wrapper's optional Sampson replacement so the stage
    # boundary remains exactly the output of the joint candidate selector.
    support_ellipse = selected_item.get("work_ellipse")
    if support_ellipse is None:
        support_ellipse = core.fit_ellipse_from_contour(selected_item.get("work_contour"))
    if support_ellipse is None:
        support_ellipse = measurement.get("work_ellipse")
    if support_ellipse is None:
        raise RuntimeError("The selected boundary has no valid support ellipse.")

    support_center = _finite_center(support_ellipse[0])
    if not np.all(np.isfinite(support_center)):
        raise RuntimeError("The selected support-ellipse center is not finite.")
    measurement["work_ellipse"] = support_ellipse
    measurement["ellipse_center"] = support_center.copy()
    if selected_item.get("work_contour") is not None:
        measurement["work_contour"] = selected_item["work_contour"]

    edge_set = core.estimate_subpixel_edge_set(gray, measurement, proj_ctx=proj_ctx)
    edge_points = edge_set.get("points") if isinstance(edge_set, dict) else None
    if edge_points is None or np.asarray(edge_points).size < 2:
        # This is the fallback required by Section 3.3, not a dropped sample.
        fit = {
            "center": support_center.copy(),
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
    else:
        fit = core.fit_subpixel_edge_ellipse_center(
            edge_set=edge_set,
            boundary_center=support_center,
            boundary_ellipse=support_ellipse,
            config=core.GEOMETRY_BOUNDARY_SUBPIXEL_CONFIG,
        )
    refined_center = _finite_center(fit.get("center"))
    if not np.all(np.isfinite(refined_center)):
        refined_center = support_center.copy()
    refined_available = bool(fit.get("available"))
    refined_ellipse = fit.get("ellipse") if refined_available else None
    rep_dbg = selected_item.get("rep_dbg", {}) if isinstance(selected_item.get("rep_dbg"), dict) else {}

    return {
        "center": refined_center,
        "geometry_selected_boundary_center": support_center,
        "subpixel_edge_fitted_center": refined_center,
        "subpixel_edge_fitted_ellipse": refined_ellipse,
        "boundary_ellipse": support_ellipse,
        "boundary_contour": measurement.get("work_contour"),
        "edge_set": edge_set,
        "measurement": measurement,
        "debug": {
            "algorithm_stage": "selected_boundary_normal_profile_physical_radius_subpixel_refit",
            "assigned_layout_index": rep_dbg.get("assigned_layout_index"),
            "candidate_index": rep_dbg.get("candidate_index"),
            "subpixel_available": refined_available,
            "final_source": "subpixel_edge_ellipse" if refined_available else "support_ellipse_fallback",
            "fallback_reason": fit.get("fallback_reason"),
            "subpixel_fit": fit.get("debug", {}),
            "edge_debug": edge_set.get("debug", {}) if isinstance(edge_set, dict) else {},
        },
    }


def _normalized_squared_eccentricity_shift_from_ellipses(shape_results):
    """Estimate the common shift from ellipse deformation without a gain.

    For semi-axes ``a >= b`` and ``rho=b/a``, the per-lamp signal is
    ``a*(1-rho^2)=(a^2-b^2)/a``.  It is already in pixel units, vanishes for a
    circle, and decays linearly for small deformation instead of amplifying
    near-circular axis noise through a square root.  This is a deformation
    scale used by Section 3.4, not the exact projective conic-center offset.
    """
    values = []
    ratios = []
    radii = []
    minor_dirs = []
    sources = []
    for result in shape_results:
        ellipse = result.get("boundary_ellipse")
        stats = core._pb1d_ellipse_stats(ellipse)
        if stats is None:
            continue
        a = float(stats["a"])
        b = float(stats["b"])
        q = float(stats["q"])
        r_eff = float(stats["r_eff"])
        value = float(max(0.0, a * a - b * b) / max(a, 1e-12))
        if not np.all(np.isfinite([value, q, r_eff])):
            continue
        values.append(value)
        ratios.append(q)
        radii.append(r_eff)
        sources.append("boundary_ellipse")
        minor = np.asarray(stats.get("minor_dir", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if np.all(np.isfinite(minor)):
            minor_dirs.append(minor)

    if not values:
        return 0.0, {
            "raw_shape_signal_px": None,
            "raw_shape_signal_per_light_px": None,
            "median_axis_ratio_q": None,
            "axis_ratio_q_per_light": None,
            "median_radius_px": None,
            "radius_per_light_px": None,
            "ellipse_source_per_light": None,
            "ellipse_axis_consistency": None,
            "shift_rule": "median_normalized_squared_eccentricity",
            "shift_lambda_used": False,
            "reason": "no_valid_ellipse_stats",
        }

    values_arr = np.asarray(values, dtype=np.float64)
    ratios_arr = np.asarray(ratios, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    shift = float(np.median(values_arr))
    axis_consistency = None
    if len(minor_dirs) >= 2:
        ref = minor_dirs[0]
        aligned = [direction if np.dot(direction, ref) >= 0.0 else -direction for direction in minor_dirs]
        axis_consistency = float(np.linalg.norm(np.mean(np.asarray(aligned), axis=0)))

    return shift, {
        # Retain these field names because the reliability gate and the compact
        # result schema consume them; the formula is identified explicitly.
        "raw_shape_signal_px": shift,
        "raw_shape_signal_per_light_px": values_arr.tolist(),
        "raw_deformation_signal_px": shift,
        "deformation_signal_per_light_px": values_arr.tolist(),
        "shift_rule": "median_normalized_squared_eccentricity",
        "formula": "median_i (a_i^2 - b_i^2) / a_i = median_i a_i (1 - rho_i^2)",
        "shift_lambda_used": False,
        "shift_lambda": None,
        "shift_raw_px": shift,
        "shift_clipped_px": shift,
        "shift_clipping_used": False,
        "median_axis_ratio_q": float(np.median(ratios_arr)),
        "axis_ratio_q_per_light": ratios_arr.tolist(),
        "median_radius_px": float(np.median(radii_arr)),
        "radius_per_light_px": radii_arr.tolist(),
        "ellipse_source_per_light": sources,
        "ellipse_axis_consistency": axis_consistency,
        "num_ellipses_for_shift": int(len(values_arr)),
    }


def _method_v3_projective_correction(stage_results, pose_ctx, config=None):
    """Run Section 3.4 on c^sp without the archived extra base-center gate."""
    cfg = dict(core.PROJECTIVE_BIAS_1D_CONFIG)
    cfg.update(BIAS1D_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)

    base_centers = np.asarray([_finite_center(r.get("center")) for r in stage_results], dtype=np.float64)
    layout_ids = [core._pb1d_get_layout_id(r, fallback_idx=i) for i, r in enumerate(stage_results)]
    complete = bool(
        base_centers.shape == (4, 2)
        and np.all(np.isfinite(base_centers))
        and sorted(map(int, layout_ids)) == [0, 1, 2, 3]
    )

    base_by_layout = core._pb1d_order_by_layout(base_centers, layout_ids) if complete else None
    axis_debug = core._pb1d_layout_axes_from_centers(base_by_layout) if complete else None
    direction = (
        np.asarray(axis_debug["direction"], dtype=np.float64).reshape(2)
        if axis_debug is not None
        else np.array([np.nan, np.nan], dtype=np.float64)
    )

    # Section 3.4 uses E^sp when available and E^sup only as its fallback.
    shape_results = []
    shape_sources = []
    for result in stage_results:
        refined = result.get("subpixel_edge_fitted_ellipse")
        support = result.get("boundary_ellipse")
        chosen = refined if refined is not None else support
        shape_sources.append("E_sp" if refined is not None else "E_sup")
        shape_results.append({"boundary_ellipse": chosen})

    kappa, shift_debug = _normalized_squared_eccentricity_shift_from_ellipses(shape_results)
    cfg.pop("shift_lambda", None)
    cfg.pop("max_shift_px", None)
    cfg["shift_rule"] = "median_normalized_squared_eccentricity"
    cfg["shift_lambda_used"] = False
    cfg["shift_clipping_used"] = False
    mu, sign_reason, sign_debug = core._pb1d_choose_sign(
        base_by_layout,
        direction,
        pose_ctx,
        cfg,
        axis_dbg=axis_debug,
    )
    used, gate_reasons = core._pb1d_gate(axis_debug, shift_debug, mu, cfg)
    gate_reasons = list(gate_reasons)
    if not complete:
        gate_reasons.append("incomplete_four_lamp_correspondence")
        used = False
    if not np.all(np.isfinite(direction)):
        if "no_valid_layout_axis" not in gate_reasons:
            gate_reasons.append("no_valid_layout_axis")
        used = False

    signed_shift = float(mu) * float(kappa) if used else 0.0
    corrected = base_centers + signed_shift * direction.reshape(1, 2) if used else base_centers.copy()
    debug = {
        "used": bool(used),
        "fallback_reasons": gate_reasons,
        "layout_ids": list(map(int, layout_ids)),
        "direction": direction.tolist() if np.all(np.isfinite(direction)) else [None, None],
        "sign": float(mu),
        "sign_reason": sign_reason,
        "sign_debug": sign_debug,
        "kappa_raw": shift_debug.get("raw_shape_signal_px"),
        "kappa": float(kappa),
        "signed_shift_px": signed_shift,
        "eta_axis": None if axis_debug is None else axis_debug.get("axis_anisotropy"),
        "median_rho": shift_debug.get("median_axis_ratio_q"),
        "axis_debug": axis_debug,
        "shift_debug": shift_debug,
        "ellipse_sources": shape_sources,
        "base_center_definition": "c_sp_with_c_sup_fallback",
        "uses_extra_subpixel_quality_gate": False,
        "uses_checkerboard_or_gt": False,
        "shift_rule": "median_normalized_squared_eccentricity",
        "shift_lambda_used": False,
    }
    return corrected, debug, cfg
