"""Validate recorded Blender captures and independently solve image ChArUco PnP.

Run in ZMSoftware: python validate_collection.py --dataset PATH [--partial].
The partial mode checks every completed record, without claiming a full dataset.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from prepare_charuco import make_board

POSITION_TOLERANCE_MM = 0.002
ROTATION_TOLERANCE_RAD = 2e-6
RECORD_PROJECTION_RMSE_LIMIT_PX = 1.5
RECORD_PROJECTION_MAX_LIMIT_PX = 3.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_board_rectangle(vision, width, height, pattern_size, camera, distortion):
    """Rectangle centred on the pattern, expressed in its top-left M frame."""
    half = np.asarray([width, height]) / 2
    center = np.asarray(pattern_size) / 2
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half + center
    points = np.column_stack((xy, np.zeros(4)))
    return cv2.projectPoints(
        points, cv2.Rodrigues(vision[:3, :3])[0], vision[:3, 3], camera, distortion,
    )[0].reshape(-1, 2)


def margin(pixels, resolution):
    width, height = resolution
    return float(min(pixels[:, 0].min(), pixels[:, 1].min(),
                     width - pixels[:, 0].max(), height - pixels[:, 1].max()))


def matrix(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
        raise ValueError("Invalid homogeneous last row")
    rotation = pose[:3, :3]
    if (np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 2e-6
            or abs(np.linalg.det(rotation) - 1) > 2e-6):
        raise ValueError("Invalid rotation matrix")
    return pose


def rotation_difference(first, second):
    # Remove float32 orthogonality roundoff before measuring a tiny rotation.
    left, _, right = np.linalg.svd(first @ second.T)
    relative = left @ right
    sine = np.linalg.norm([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) / 2
    cosine = (np.trace(relative) - 1) / 2
    return float(math.atan2(sine, cosine))


def pose_difference(first, second):
    return {
        "translation_error_mm": float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
        "rotation_error_rad": rotation_difference(first[:3, :3], second[:3, :3]),
    }


def check_pose(first, second, label, errors):
    difference = pose_difference(first, second)
    if (difference["translation_error_mm"] > POSITION_TOLERANCE_MM
            or difference["rotation_error_rad"] > ROTATION_TOLERANCE_RAD):
        errors.append(f"{label}: {difference}")
    return difference


def image_pnp(gray, board, detector, camera_matrix, distortion, truth):
    corners, ids, _, marker_ids = detector.detectBoard(gray)
    found_markers = [] if marker_ids is None else marker_ids.reshape(-1).tolist()
    found_corners = [] if ids is None else ids.reshape(-1).tolist()
    errors = []
    result = {
        "marker_count": len(found_markers), "charuco_corner_count": len(found_corners),
        "detected_marker_ids": sorted(found_markers),
        "detected_charuco_ids": sorted(found_corners),
        "vision_pose": None, "reprojection_rmse_px": None,
        "vision_pose_source": "Image-detected ChArUco IDs, IPPE then LM; no ground-truth initialization",
        "length_unit": "mm", "transform_convention": "C_T_M",
    }
    if sorted(found_markers) != sorted(board.getIds().reshape(-1).tolist()):
        errors.append(f"Incomplete or incorrect marker IDs: {len(found_markers)}/31")
    if sorted(found_corners) != list(range(len(board.getChessboardCorners()))):
        errors.append(f"Incomplete or incorrect ChArUco IDs: {len(found_corners)}/48")
        return result, errors

    objects = board.getChessboardCorners()[found_corners].astype(np.float64) * 1000
    pixels = corners.reshape(-1, 2).astype(np.float64)
    truth_pixels = cv2.projectPoints(
        objects, cv2.Rodrigues(truth[:3, :3])[0], truth[:3, 3], camera_matrix, distortion,
    )[0].reshape(-1, 2)
    truth_pixel_errors = np.linalg.norm(pixels - truth_pixels, axis=1)
    record_rmse = float(np.sqrt(np.mean(truth_pixel_errors ** 2)))
    record_max = float(np.max(truth_pixel_errors))
    record_matches = bool(
        record_rmse <= RECORD_PROJECTION_RMSE_LIMIT_PX
        and record_max <= RECORD_PROJECTION_MAX_LIMIT_PX
    )
    result["render_record_consistency"] = {
        "rmse_px": record_rmse, "max_error_px": record_max,
        "rmse_limit_px": RECORD_PROJECTION_RMSE_LIMIT_PX,
        "max_error_limit_px": RECORD_PROJECTION_MAX_LIMIT_PX,
        "passed": record_matches,
        "note": "Rendered image/record correspondence tolerance allows raster and corner detection errors; not physical measurement accuracy. Subpixel-similar swapped images may be indistinguishable.",
    }
    if not record_matches:
        errors.append(f"Image corners disagree with recorded projection: RMS {record_rmse:.6g}px, max {record_max:.6g}px")
    found, rotations, translations, _ = cv2.solvePnPGeneric(
        objects, pixels, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE,
    )
    candidates = []
    if found:
        for rvec, tvec in zip(rotations, translations):
            rotation = cv2.Rodrigues(rvec)[0]
            if np.any((objects @ rotation.T + tvec.reshape(3))[:, 2] <= 0):
                continue
            projected = cv2.projectPoints(objects, rvec, tvec, camera_matrix, distortion)[0]
            rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
            candidates.append((rms, rvec, tvec))
    if not candidates:
        errors.append("PnP has no positive-depth solution")
        return result, errors
    _, rvec, tvec = min(candidates, key=lambda item: item[0])
    rvec, tvec = cv2.solvePnPRefineLM(objects, pixels, camera_matrix, distortion, rvec, tvec)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.reshape(3)
    pose = matrix(pose)
    if np.any((objects @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        errors.append("Refined PnP places a corner behind the camera")
    projected = cv2.projectPoints(objects, rvec, tvec, camera_matrix, distortion)[0].reshape(-1, 2)
    result.update({
        "vision_pose": pose.tolist(),
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1)))),
        "ground_truth_comparison": pose_difference(pose, truth),
    })
    return result, errors


def validate_capture(row, directory, inertia, reference, hand_eye, target, board, detector, camera_matrix, distortion, extrinsics):
    result = {
        "run_id": directory.name,
        "capture_index": row.get("capture_index"),
        "point_id": row.get("point_id"), "direction_id": row.get("direction_id"),
        "batch_id": row.get("batch_id"), "arrival_index": row.get("arrival_index"),
        "frame_index": row.get("frame_index"),
        "sample_id": row.get("sample_id"), "image_filename": row.get("image_filename"),
        "errors": [], "pnp": None,
    }
    errors = result["errors"]
    try:
        ideal = matrix(row["B_T_E_ideal_mm"])
        actual = matrix(row["B_T_E_actual_mm"])
        camera = matrix(row["B_T_C_mm"])
        vision_truth = matrix(row["C_T_M_mm"])
        extrinsic_error_start = len(errors)
        expected_extrinsic_id = f"{directory.name}/{row['point_id']}/{row['direction_id']}"
        if row["run_id"] != directory.name or row["extrinsic_id"] != expected_extrinsic_id:
            errors.append("Capture run/extrinsic identity does not match its directory and point/direction")
        parameter_extrinsic = extrinsics.get(expected_extrinsic_id)
        result["parameter_extrinsics"] = {"expected_extrinsic_id": expected_extrinsic_id}
        if parameter_extrinsic is None:
            errors.append("parameters.json has no extrinsic entry for this capture")
        else:
            expected_image = f"{directory.name}/{row['image_filename']}"
            if parameter_extrinsic["image_filename"] != expected_image:
                errors.append("Parameter extrinsic image_filename differs from the capture record")
            result["parameter_extrinsics"].update({
                "B_T_C_comparison": check_pose(
                    matrix(parameter_extrinsic["B_T_C"]), camera,
                    "Parameter B_T_C differs from capture record", errors,
                ),
                "C_T_M_comparison": check_pose(
                    matrix(parameter_extrinsic["C_T_M"]), vision_truth,
                    "Parameter C_T_M differs from capture record", errors,
                ),
            })
        result["parameter_extrinsics"]["passed"] = len(errors) == extrinsic_error_start
        direction = np.asarray(row["direction_B"], dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction) - 1) > 1e-6:
            raise ValueError("direction_B must be a finite unit vector")
        if row["length_unit"] != "mm" or not math.isclose(row["inertia"], inertia, abs_tol=1e-12):
            errors.append("Record units or inertia do not match the run")
        bias = np.asarray(row.get("target_bias_B_mm", [0, 0, 0]), dtype=float)
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("target_bias_B_mm must be a finite three-component vector")
        expected_delta = bias + 2 * inertia * direction
        expected_distance = float(np.linalg.norm(expected_delta))
        delta = actual[:3, 3] - ideal[:3, 3]
        distance = float(np.linalg.norm(delta))
        direction_error = float(np.linalg.norm(delta - expected_delta))
        rotation_error = rotation_difference(actual[:3, :3], ideal[:3, :3])
        result["geometry"] = {
            "offset_norm_mm": distance, "expected_offset_mm": expected_distance,
            "direction_error_mm": direction_error, "rotation_difference_rad": rotation_error,
            "E_H_equals_C": check_pose(actual @ hand_eye, camera, "E * H != C", errors),
            "C_V_equals_target": check_pose(camera @ vision_truth, target, "C * C_T_M != B_T_M", errors),
        }
        if "target_bias_B_mm" in row:
            biased_target = ideal.copy()
            biased_target[:3, 3] += bias
            result["geometry"].update({
                "target_bias_B_mm": bias.tolist(),
                "target_bias_norm_mm": float(np.linalg.norm(bias)),
                "inertia_offset_mm": 2 * inertia,
                "biased_target": check_pose(
                    matrix(row["B_T_E_biased_target_mm"]), biased_target,
                    "Biased target differs from original ideal plus target bias", errors,
                ),
            })
        if abs(distance - expected_distance) > POSITION_TOLERANCE_MM:
            errors.append("Offset norm differs from the combined target bias and inertia offset")
        if direction_error > POSITION_TOLERANCE_MM:
            errors.append("Offset is inconsistent with the target bias and approach direction")
        if rotation_error > ROTATION_TOLERANCE_RAD:
            errors.append("Actual endpoint changes the ideal orientation")
        if reference is not None:
            result["geometry"]["fixed_plan_ideal"] = check_pose(
                ideal, matrix(reference["B_T_E_ideal_mm"]), "Ideal pose differs from fixed plan", errors,
            )
            expected_direction = next(
                item["direction_B"] for item in reference["directions"]
                if item["direction_id"] == row["direction_id"]
            )
            if not np.allclose(direction, expected_direction, rtol=0, atol=1e-10):
                errors.append("Direction differs from fixed observation plan")

        filename = f"calibration_images/{row['batch_id']}_{row['point_id']}_D{int(row['direction_id'][1:]):03d}.png"
        if row["image_filename"] != filename:
            errors.append("Image filename does not match batch/point/three-digit direction identity")
        if row["arrival_index"] != 1 or row["frame_index"] != 1 or row["sample_id"] != "001":
            errors.append("This protocol requires exactly R001 and F001 for each point/direction")
        path = directory / row["image_filename"]
        encoded = np.fromfile(path, dtype=np.uint8)
        if encoded[:8].tobytes() != b"\x89PNG\r\n\x1a\n":
            errors.append("PNG filename does not contain a PNG image")
        color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError("Image cannot be decoded")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        result["image_size_px"] = [gray.shape[1], gray.shape[0]]
        if result["image_size_px"] != [3072, 2048]:
            errors.append("Image resolution is not 3072 x 2048")
        result["pnp"], image_errors = image_pnp(
            gray, board, detector, camera_matrix, distortion, vision_truth,
        )
        errors.extend(image_errors)
    except (OSError, ValueError, KeyError, StopIteration, cv2.error) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["passed"] = not errors
    return result


def validate(dataset, partial=False):
    parameters = read_json(dataset / "parameters.json")
    plan = read_json(dataset / parameters["fixed_plan"])
    board = make_board(0)
    detector = cv2.aruco.CharucoDetector(board)
    camera_matrix = np.asarray(parameters["camera"]["camera_matrix"], dtype=float)
    distortion = np.asarray(parameters["camera"]["dist_coeffs"], dtype=float)
    transforms = parameters["system_transforms"]
    extrinsics = parameters["extrinsics_by_capture"]
    hand_eye = matrix(transforms["E_T_C_hand_eye"])
    target = matrix(transforms["B_T_M_charuco_top_left"])
    center = matrix(transforms["B_T_target_center"])
    errors = []
    if parameters["length_unit"] != "mm":
        errors.append("Dataset parameters must use mm")
    if parameters["camera"]["image_size_px"] != [3072, 2048]:
        errors.append("Declared image resolution is not 3072 x 2048")
    width, height = np.asarray(board.getChessboardSize()) * board.getSquareLength() * 1000
    charuco = parameters["charuco"]
    if (charuco["squares_xy"] != list(board.getChessboardSize())
            or not np.allclose(charuco["pattern_size_mm"], [width, height], rtol=0, atol=1e-5)
            or not math.isclose(charuco["marker_length_mm"], board.getMarkerLength() * 1000, abs_tol=1e-5)
            or charuco["dictionary"] != "DICT_5X5_1000"):
        raise ValueError("Dataset ChArUco definition differs from prepare_charuco.make_board")
    center_to_top_left = np.eye(4)
    center_to_top_left[:3, 3] = [-width / 2, -height / 2, 0]
    center_checks = {
        "expected_target_center_T_M_mm": center_to_top_left.tolist(),
        "definition": charuco["center_definition"],
        "target_transform": check_pose(center @ center_to_top_left, target, "Target center/top-left mismatch", errors),
        "recorded_offset": check_pose(
            matrix(transforms["target_center_T_M"]), center_to_top_left,
            "Stored target_center_T_M mismatch", errors,
        ),
    }
    points = {point["point_id"]: point for point in plan["points"]}
    expected_pairs = {
        (point["point_id"], direction["direction_id"])
        for point in plan["points"] for direction in point["directions"]
    }
    if len(points) != 30 or len(expected_pairs) != 600 or any(len(p["directions"]) != 20 for p in points.values()):
        errors.append("Fixed plan must contain 30 points x 20 distinct directions")
    runs = parameters["runs"]
    if len(runs) != 2 or not np.allclose([r["inertia"] for r in runs], [0.1, 0.2], rtol=0, atol=1e-12):
        errors.append("Expected two runs with inertia 0.1 and 0.2")
    results = []
    run_summaries = []
    first_ideals = {}
    cross_run_max_mm = 0.0
    recorded_extrinsic_ids = set()
    with (dataset / "per_image_validation.jsonl").open("w", encoding="utf-8") as stream:
        for run_index, run in enumerate(runs):
            directory = dataset / run["run_id"]
            record_path = directory / "records.json"
            run_errors = []
            if run["batch_id"] != f"B{run_index:03d}":
                run_errors.append("Run batch_id does not match B000/B001 order")
            if record_path.exists():
                records = read_json(record_path)
                rows = records["captures"]
                if records["run_id"] != run["run_id"] or records["inertia"] != run["inertia"]:
                    run_errors.append("records.json run identity differs from parameters")
                if records["batch_id"] != run["batch_id"] or any(row["batch_id"] != run["batch_id"] for row in rows):
                    run_errors.append("Record batch identity differs from parameters")
                if records["expected_images"] != 600 or (not partial and not records["complete"]):
                    run_errors.append("records.json does not declare a complete 600-image run")
            else:
                rows = []
                if not partial:
                    run_errors.append("Missing records.json")
            if run["expected_images"] != 600 or (not partial and len(rows) != 600):
                run_errors.append("Expected 600 captures in the run")
            pairs = Counter((row["point_id"], row["direction_id"]) for row in rows)
            if any(count != 1 for count in pairs.values()) or not set(pairs).issubset(expected_pairs):
                run_errors.append("Duplicate or unknown point/direction pair")
            if not partial and set(pairs) != expected_pairs:
                run_errors.append("Captured point/direction pairs do not cover the frozen plan")
            indices = [row["capture_index"] for row in rows]
            if len(set(indices)) != len(rows) or any(index not in range(600) for index in indices):
                run_errors.append("Duplicate or invalid capture indices")
            recorded_files = {row["image_filename"] for row in rows}
            actual_files = {
                path.relative_to(directory).as_posix()
                for path in (directory / "calibration_images").iterdir() if path.is_file()
            }
            if recorded_files != actual_files:
                run_errors.append("Image file set and capture records differ")
            run_start = len(results)
            for row in rows:
                recorded_extrinsic_ids.add(f"{run['run_id']}/{row['point_id']}/{row['direction_id']}")
                result = validate_capture(
                    row, directory, run["inertia"], points.get(row["point_id"]),
                    hand_eye, target, board, detector, camera_matrix, distortion, extrinsics,
                )
                key = (row["point_id"], row["direction_id"])
                if "geometry" in result:
                    ideal = matrix(row["B_T_E_ideal_mm"])
                    if key in first_ideals:
                        difference = check_pose(ideal, first_ideals[key], "Cross-run ideal pose mismatch", result["errors"])
                        cross_run_max_mm = max(cross_run_max_mm, difference["translation_error_mm"])
                        result["geometry"]["cross_run_ideal"] = difference
                    else:
                        first_ideals[key] = ideal
                result["passed"] = not result["errors"]
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if len(results) % 50 == 0:
                    stream.flush()
                    print(f"VALIDATION_PROGRESS {len(results)}/1200", flush=True)
            checked = results[run_start:]
            run_summaries.append({
                "run_id": run["run_id"], "inertia": run["inertia"],
                "batch_id": run["batch_id"],
                "expected_images": 600, "recorded_images": len(rows), "image_files": len(actual_files),
                "unique_points": len({pair[0] for pair in pairs}), "unique_point_direction_pairs": len(pairs),
                "failed_images": sum(not item["passed"] for item in checked),
                "errors": run_errors,
                "passed": not run_errors and all(item["passed"] for item in checked),
            })
    if not results:
        errors.append("No completed captures were available to validate")
    expected_extrinsic_ids = {
        f"{run['run_id']}/{point}/{direction}" for run in runs for point, direction in expected_pairs
    }
    extrinsic_keys = set(extrinsics)
    keys_match_records = extrinsic_keys == recorded_extrinsic_ids
    keys_match_full_plan = extrinsic_keys == expected_extrinsic_ids
    if not keys_match_records:
        errors.append("Parameter extrinsic keys do not exactly match completed capture records")
    if not partial and not keys_match_full_plan:
        errors.append("Parameter extrinsic keys do not cover exactly two complete 600-image runs")
    pnp_results = [item["pnp"] for item in results if item["pnp"] and item["pnp"]["vision_pose"] is not None]
    projection_results = [
        item["pnp"]["render_record_consistency"] for item in results
        if item["pnp"] and "render_record_consistency" in item["pnp"]
    ]
    geometry_results = [item["geometry"] for item in results if "geometry" in item]
    summary = {
        "dataset": str(dataset), "opencv_version": cv2.__version__,
        "mode": "partial" if partial else "complete",
        "expected_images": 1200, "validated_images": len(results),
        "full_dataset_validated": not partial and len(results) == 1200,
        "failed_images": sum(not item["passed"] for item in results),
        "tolerances": {
            "position_mm": POSITION_TOLERANCE_MM, "rotation_rad": ROTATION_TOLERANCE_RAD,
            "render_record_projection_rmse_px": RECORD_PROJECTION_RMSE_LIMIT_PX,
            "render_record_projection_max_px": RECORD_PROJECTION_MAX_LIMIT_PX,
        },
        "target_center_check": center_checks, "runs": run_summaries,
        "parameter_extrinsics": {
            "entries": len(extrinsics), "matches_completed_record_keys": keys_match_records,
            "matches_complete_plan_keys": keys_match_full_plan,
            "missing_record_entries": sorted(recorded_extrinsic_ids - extrinsic_keys),
            "entries_without_records": sorted(extrinsic_keys - recorded_extrinsic_ids),
        },
        "maximum_cross_run_ideal_translation_difference_mm": cross_run_max_mm,
        "maximum_direction_error_mm": max((r["direction_error_mm"] for r in geometry_results), default=None),
        "maximum_endpoint_rotation_difference_rad": max((r["rotation_difference_rad"] for r in geometry_results), default=None),
        "pnp_images": len(pnp_results),
        "maximum_reprojection_rmse_px": max((r["reprojection_rmse_px"] for r in pnp_results), default=None),
        "maximum_render_record_projection_rmse_px": max((r["rmse_px"] for r in projection_results), default=None),
        "maximum_render_record_projection_error_px": max((r["max_error_px"] for r in projection_results), default=None),
        "failed_render_record_consistency_images": sum(not r["passed"] for r in projection_results),
        "maximum_pnp_truth_translation_difference_mm": max((r["ground_truth_comparison"]["translation_error_mm"] for r in pnp_results), default=None),
        "errors": errors,
        "per_image_results": "per_image_validation.jsonl",
        "notes": "PnP is estimated from image corners without truth initialization. Truth projection checks only rendered image/record correspondence (RMS <= 1.5 px, max <= 3 px), allowing raster and detection errors; it cannot distinguish every subpixel-similar image swap. No real-camera measurement accuracy or PnP pose accuracy threshold is asserted. Twenty different directions are not twenty repetitions of one direction.",
    }
    summary["passed"] = not errors and all(run["passed"] for run in run_summaries)
    (dataset / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Validate only completed captures; do not claim full coverage")
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.dataset.resolve(), args.partial) else 1)
