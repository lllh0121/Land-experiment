"""Run the four-stage lamp-center localization pipeline on a rectified image."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def localize(image, K, roi_points=None):
    """Return layout-ordered centers and diagnostics; no GT or baselines are used.

    image: decoded, rectified BGR image; K: matching 3x3 rectified intrinsics.
    roi_points: optional polygon in original image pixel coordinates.
    World coordinates and radius use the original fixed four-lamp target.
    """
    import cv2
    import numpy as np
    import ippe_only_pose_experiment as ippe
    import pipeline_stages as stages
    core = stages.core
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Coarse observations, IPPE pose, and projected geometric prior.
    pose, candidates = ippe.estimate_pose_ctx_from_light_only_ippe_no_refine(
        image, K, light_centers_mm=core.circle_centers_w,
        light_radius_mm=float(core.LIGHT_RADIUS_MM),
        roi_mask=stages._roi_mask(image.shape, roi_points),
        max_candidates=8, return_candidates=True,
    )
    core._build_pose_runtime_cache(pose, K, core.circle_centers_w)
    projection = pose["runtime_cache"]["projection_ctx"]
    initial = pose.get("pose_ref_debug", {}).get("initial_pose", {})
    assignments = sorted(tuple(map(int, pair)) for pair in initial.get("assignments", []))
    if len(assignments) != 4 or sorted(pair[0] for pair in assignments) != [0, 1, 2, 3]:
        raise RuntimeError("IPPE did not return a complete four-lamp correspondence.")

    # 2. Joint boundary selection and observation-to-layout assignment.
    selected, selection = core.build_selected_measurements_for_pipeline_joint_geometry(
        gray_light=gray,
        contours=[candidates[index]["contour"] for _, index in assignments],
        K=K, pose_ctx=pose, circle_centers_w=core.circle_centers_w,
    )
    pi_star = [int(value) for value in selection.get("assignment", [])]
    if (selection.get("mode") != "joint_geometry_selector" or len(selected) != 4
            or len(selection.get("combo", [])) != 4 or sorted(pi_star) != [0, 1, 2, 3]):
        raise RuntimeError("Joint geometry selection did not return a complete solution.")

    # 3. Subpixel normal-profile edges and ellipse refitting.
    refined = [stages._method_v3_subpixel_stage(gray, item, projection) for item in selected]

    # 4. Shared one-dimensional shift, gated by shape/layout/sign reliability.
    corrected, correction, _ = stages._method_v3_projective_correction(refined, pose)
    ordered = stages._ordered_by_layout
    centers = {
        "v0_coarse_center": np.asarray(initial["observed_centers_px"], dtype=float).reshape(4, 2).tolist(),
        "v1_support_ellipse_center": np.asarray(ordered(
            [item["geometry_selected_boundary_center"] for item in refined], pi_star)).tolist(),
        "v2_subpixel_center": np.asarray(ordered([item["center"] for item in refined], pi_star)).tolist(),
        "full_corrected_center": np.asarray(ordered(corrected, pi_star)).tolist(),
    }
    return ippe.light_pose.to_jsonable({
        "centers_px": centers["full_corrected_center"],
        "stage_centers_by_layout": centers,
        "observation_to_layout": pi_star,
        "subpixel_available_by_layout": ordered(
            [bool(item["debug"]["subpixel_available"]) for item in refined], pi_star),
        "projective_correction": correction,
        "physical_radius_mm": float(core.LIGHT_RADIUS_MM),
        "manual_roi_used": roi_points is not None,
        "coordinate_note": "Layout order is a geometric symmetry branch, not uniquely identified physical lamp labels.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Rectified light image.")
    parser.add_argument("--camera", type=Path, required=True, help="JSON with new_K or K (3x3).")
    parser.add_argument("--roi", type=Path, help="Optional JSON with roi_points polygon.")
    parser.add_argument("--output", type=Path, default=Path("outputs/centers.json"))
    args = parser.parse_args()
    for path in (args.image, args.camera, *([args.roi] if args.roi else [])):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")

    import cv2
    import numpy as np
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    K = np.asarray(camera.get("new_K", camera.get("K")), dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12):
        parser.error("Camera matrix must be finite, invertible, 3x3, with positive focal lengths.")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Unable to decode image: {args.image}")
    roi = None
    if args.roi:
        payload = json.loads(args.roi.read_text(encoding="utf-8"))
        roi = np.asarray(payload.get("roi_points", []), dtype=float)
        if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3 or not np.isfinite(roi).all():
            parser.error("roi_points must contain at least three finite [x, y] pairs.")
    result = localize(image, K, roi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "centers_px": result["centers_px"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
