"""Minimal light-only planar pose initializer used by the 2026-08-31 experiment.

The experiment intentionally removes the multi-solver/joint-cost initializer and
the custom LM refinement used by ``light_only_iterative_pose``.  Its pose logic is
limited to:

1. detect the same image-domain lamp candidates as the production pipeline;
2. generate pose hypotheses only with ``SOLVEPNP_IPPE``;
3. reject hypotheses for which any known lamp center has non-positive depth;
4. retain the hypothesis with the smallest lamp-center reprojection RMSE.

No circle-boundary residual, candidate-quality term, pose prior, or nonlinear
pose refinement is used to select or update the returned pose.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

import light_only_iterative_pose as light_pose


def _solve_ippe(obj_pts, img_pts, K):
    """Return every IPPE solution supplied by OpenCV for one correspondence set."""
    ret = cv2.solvePnPGeneric(
        np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3),
        np.asarray(img_pts, dtype=np.float64).reshape(-1, 2),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not bool(ret[0]):
        return []
    return [
        (
            np.asarray(rvec, dtype=np.float64).reshape(3),
            np.asarray(tvec, dtype=np.float64).reshape(3),
        )
        for rvec, tvec in zip(ret[1], ret[2])
    ]


def initialize_pose_ippe_reprojection(candidates, K, centers_w, top_k=8):
    """Select a pure-IPPE pose using cheirality and center reprojection RMSE only."""
    centers_w = np.asarray(centers_w, dtype=np.float64).reshape(-1, 3)
    if len(centers_w) != 4:
        raise ValueError("The IPPE-only experiment expects exactly four known lamp centers.")
    if len(candidates) < 4:
        raise RuntimeError(f"At least four lamp candidates are required; got {len(candidates)}.")

    used_candidates = list(candidates[: int(top_k)])
    best = None
    num_correspondence_sets = 0
    num_ippe_solutions = 0
    num_positive_depth_solutions = 0

    for candidate_indices in itertools.combinations(range(len(used_candidates)), 4):
        for candidate_order in itertools.permutations(candidate_indices, 4):
            num_correspondence_sets += 1
            img_pts = np.asarray(
                [used_candidates[candidate_idx]["center_px"] for candidate_idx in candidate_order],
                dtype=np.float64,
            ).reshape(4, 2)
            try:
                solutions = _solve_ippe(centers_w, img_pts, K)
            except cv2.error:
                continue
            num_ippe_solutions += len(solutions)

            for rvec, tvec in solutions:
                depth_margin = light_pose.positive_depth_margin(rvec, tvec, centers_w)
                if not np.isfinite(depth_margin) or depth_margin <= 0.0:
                    continue
                num_positive_depth_solutions += 1
                projected = light_pose.project_points(K, rvec, tvec, centers_w)
                errors = np.linalg.norm(projected - img_pts, axis=1)
                reprojection_rmse = float(np.sqrt(np.mean(errors * errors)))
                if not np.isfinite(reprojection_rmse):
                    continue
                if best is None or reprojection_rmse < best["reprojection_rmse_px"]:
                    best = {
                        "rvec": rvec.copy(),
                        "tvec": tvec.copy(),
                        "assignments": [
                            (int(object_idx), int(candidate_idx))
                            for object_idx, candidate_idx in enumerate(candidate_order)
                        ],
                        "solver_flag": int(cv2.SOLVEPNP_IPPE),
                        "reprojection_rmse_px": reprojection_rmse,
                        "reprojection_mean_px": float(np.mean(errors)),
                        "reprojection_median_px": float(np.median(errors)),
                        "reprojection_max_px": float(np.max(errors)),
                        "reprojection_errors_px": errors.copy(),
                        "projected_centers_px": projected.copy(),
                        "observed_centers_px": img_pts.copy(),
                        "positive_depth_margin_mm": float(depth_margin),
                    }

    if best is None:
        raise RuntimeError("IPPE produced no positive-depth pose for the detected lamp candidates.")

    best["search_debug"] = {
        "num_candidates_used": int(len(used_candidates)),
        "num_correspondence_sets": int(num_correspondence_sets),
        "num_ippe_solutions": int(num_ippe_solutions),
        "num_positive_depth_solutions": int(num_positive_depth_solutions),
        "selection_rule": "positive_depth_then_minimum_lamp_center_reprojection_rmse",
    }
    return best, used_candidates


def estimate_pose_ctx_from_light_only_ippe_no_refine(
    light_image,
    K,
    light_centers_mm,
    light_radius_mm,
    roi_mask=None,
    max_candidates=8,
    return_candidates=False,
):
    """Return a pipeline-compatible pose context without any pose refinement.

    ``return_candidates`` is an experiment-interface option.  When enabled, the
    exact candidate objects consumed by the IPPE search are returned together
    with the pose context so later stages can build their multilevel boundaries
    inside the same coarse observations.  The default keeps the original API.
    """
    image = np.asarray(light_image).copy()
    centers_w = np.asarray(light_centers_mm, dtype=np.float64).reshape(-1, 3)
    candidates = light_pose.detect_light_candidates(
        image,
        roi_mask=roi_mask,
        max_candidates=int(max_candidates),
    )
    initial_pose, candidates = initialize_pose_ippe_reprojection(
        candidates,
        K,
        centers_w,
        top_k=min(int(max_candidates), len(candidates)),
    )

    # Preserve the production pipeline's uncertainty packaging so that the
    # experiment changes only pose initialization/selection/refinement logic.
    reprojection_median = float(initial_pose["reprojection_median_px"])
    pose_sigma_px = float(np.clip(max(0.75, 0.75 * reprojection_median), 0.5, 8.0))
    debug = {
        "algorithm": "ippe_only_positive_depth_min_reprojection_no_refinement",
        "logic_contract": {
            "uses_checkerboard": False,
            "uses_external_pose_prior": False,
            "pnp_solver": "SOLVEPNP_IPPE",
            "multi_solution_selection": "positive_depth_then_minimum_lamp_center_reprojection_rmse",
            "uses_custom_joint_cost": False,
            "uses_pose_refinement": False,
            "uses_circle_boundary_residual_for_pose": False,
            "known_light_radius_used_for_pose": False,
        },
        "initial_pose": initial_pose,
        "assignment": initial_pose["assignments"],
        "num_candidates": int(len(candidates)),
        "light_radius_mm_received_but_unused_for_pose": float(light_radius_mm),
    }
    pose_ctx = light_pose.pose_ctx_from_rt(
        initial_pose["rvec"],
        initial_pose["tvec"],
        K,
        centers_w,
        pose_sigma_px=pose_sigma_px,
        debug=debug,
    )
    pose_ctx["pose_source"] = "light_only_ippe_min_reprojection_no_refinement"
    if bool(return_candidates):
        return pose_ctx, candidates
    return pose_ctx
