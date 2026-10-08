"""Append one completed fixed-plan batch without rewriting earlier batches.

python append_batch.py [export|validate|all] --dataset PATH --source PATH/_work/B003
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from acquisition_geometry import AcquisitionGeometry
from collect_images import save_json
from export_grouped import MINIMUM_MARGIN_PX, save_record
from prepare_charuco import make_board
from validate_collection import (
    check_pose, margin, matrix, project_board_rectangle, read_json, validate_capture,
)

ROOT = Path(__file__).resolve().parent


def load_batch(dataset, source):
    parameters = read_json(dataset / "parameters.json")
    capture_parameters = read_json(source / "parameters.json")
    if read_json(source / "target_bias.json") != capture_parameters["target_bias"]:
        raise ValueError("Saved point biases differ from capture parameters")
    if len(capture_parameters["runs"]) != 1:
        raise ValueError("Append requires a source containing one completed run")
    run = capture_parameters["runs"][0]
    plan = read_json(dataset / parameters["fixed_plan"])
    if read_json(source / capture_parameters["fixed_plan"]) != plan:
        raise ValueError("The source must use the same frozen targets and directions")
    for key in ("camera", "charuco", "system_transforms", "length_unit"):
        if parameters[key] != capture_parameters[key]:
            raise ValueError(f"The appended batch has different {key}")
    directory = source / run["run_id"]
    records = read_json(directory / "records.json")
    rows = sorted(records["captures"], key=lambda row: row["capture_index"])
    pairs = {(p["point_id"], d["direction_id"])
             for p in plan["points"] for d in p["directions"]}
    if (len(plan["points"]) != 30 or len(pairs) != 600
            or any(len(p["directions"]) != 20 for p in plan["points"])
            or not records["complete"] or run["expected_images"] != 600
            or Counter((r["point_id"], r["direction_id"]) for r in rows) != Counter(pairs)
            or len({r["image_filename"] for r in rows}) != 600
            or any(r["batch_id"] != run["batch_id"] for r in rows)):
        raise ValueError("Append requires exactly 30 fixed poses x 20 directions")
    return parameters, capture_parameters, run, plan, directory, rows


def public_run(source, dataset, run, bias):
    batch = run["batch_id"]
    return {
        "run_id": batch, "batch_id": batch, "inertia": run["inertia"],
        "expected_images": 600, "record": f"{batch}/record.json",
        "direction_ids": [f"D{i:02d}" for i in range(1, 21)],
        "target_bias": bias, "source": source.relative_to(dataset).as_posix(),
    }


def export(dataset, source):
    parameters, capture_parameters, run, _, directory, rows = load_batch(dataset, source)
    batch = run["batch_id"]
    if batch in {"B000", "B001", "B002"}:
        raise ValueError("The original three batches must not be replaced")
    bias = capture_parameters["target_bias"]
    metadata = public_run(source, dataset, run, bias)
    previous = next((item for item in parameters["runs"] if item["batch_id"] == batch), None)
    if previous is not None and previous != metadata:
        raise ValueError(f"Existing {batch} metadata differs from this source")
    destination = dataset / batch
    expected_record = {
        "batch": batch, "inertia": run["inertia"], "target_bias": bias["coefficient"],
        "frames": [{"image": row["image_filename"], "ideal": row["B_T_E_ideal_mm"],
                    "actual": row["B_T_E_actual_mm"]} for row in rows],
    }
    record_path = destination / "record.json"
    if record_path.exists() and read_json(record_path) != expected_record:
        raise ValueError(f"Existing {batch} record differs from this source")
    (destination / "calibration_images").mkdir(parents=True, exist_ok=True)
    for row in rows:
        original = directory / row["image_filename"]
        image_path = destination / row["image_filename"]
        if image_path.exists():
            if not os.path.samefile(original, image_path):
                raise ValueError(f"Existing image is not the source hard link: {image_path}")
        else:
            os.link(original, image_path)
        key = f"{batch}/{row['point_id']}/{row['direction_id']}"
        value = {"B_T_C": row["B_T_C_mm"], "C_T_M": row["C_T_M_mm"],
                 "image_filename": f"{batch}/{row['image_filename']}"}
        if key in parameters["extrinsics_by_capture"] and parameters["extrinsics_by_capture"][key] != value:
            raise ValueError(f"Existing extrinsic differs from this source: {key}")
        parameters["extrinsics_by_capture"][key] = value
    if not record_path.exists():
        save_record(record_path, batch, run["inertia"], rows, target_bias=bias["coefficient"])
    if previous is None:
        parameters["runs"].append(metadata)
    parameters["purpose"] = "Approved 30 fixed poses with per-batch inertia and optional fixed point bias"
    parameters["record_definition"]["batch"] = "Batch identifier; coefficients are defined in runs"
    parameters["record_definition"]["inertia"] = (
        "Dimensionless stopping coefficient; inertial component = 2 * inertia mm along the approach direction"
    )
    parameters["record_definition"]["target_bias"] = (
        "Optional coefficient in [0,1]; maximum point translation = 5 * coefficient mm. "
        "The one-time sampled vectors in runs[].target_bias.offsets_B_mm are fixed across directions."
    )
    parameters["record_definition"]["actual"] = (
        "4x4 B_T_E at the rendered stop in mm; includes fixed point bias plus approach-direction inertia"
    )
    parameters["motion_model"].update({
        "formula": "p_stop = p_ideal + point_bias_B + (2 * inertia mm) * direction_B; R_stop = R_ideal",
        "maximum_offset_mm": 2.0,
        "maximum_offset_definition": "Upper bound for the inertia component only; excludes target point bias",
        "target_bias_maximum_mm": 5.0,
        "target_bias_definition": "One fixed translation per point, shared across its directions; zero when absent from run metadata",
    })
    save_json(dataset / "parameters.json", parameters)
    print(f"EXPORTED {batch}: {len(rows)} images; earlier batches unchanged", flush=True)


def update_validation_index(dataset, batch, summary):
    path = dataset / "validation_index.json"
    if path.exists():
        index = read_json(path)
    else:
        previous = read_json(dataset / "validation.json")
        index = {"reports": [{
            "file": "validation.json", "batches": [item["batch"] for item in previous["batches"]],
            "images": previous["images"], "passed": previous["passed"],
        }]}
    index["reports"] = [item for item in index["reports"] if batch not in item["batches"]]
    index["reports"].append({
        "file": f"validation_{batch}.json", "batches": [batch],
        "images": summary["images"], "passed": summary["passed"],
    })
    index["images"] = sum(item["images"] for item in index["reports"])
    index["passed"] = all(item["passed"] for item in index["reports"])
    index["note"] = "Earlier batch reports retain their original scope; appending validates only the new batch."
    save_json(path, index)


def validate(dataset, source):
    parameters, capture_parameters, run, plan, directory, rows = load_batch(dataset, source)
    batch, inertia = run["batch_id"], run["inertia"]
    bias = capture_parameters["target_bias"]
    points = {point["point_id"]: point for point in plan["points"]}
    errors = []
    offsets = bias["offsets_B_mm"]
    if (set(offsets) != set(points) or not 0 <= bias["coefficient"] <= 1
            or bias["maximum_mm"] != 5.0
            or not np.isclose(bias["limit_mm"], 5 * bias["coefficient"], rtol=0, atol=1e-12)):
        errors.append("Target bias definition does not match the fixed points and 5 mm maximum")
    norms = []
    for point_id, values in offsets.items():
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid frozen target bias for {point_id}")
        norms.append(float(np.linalg.norm(vector)))
        if norms[-1] > bias["limit_mm"] + 1e-12:
            errors.append(f"Target bias exceeds its limit: {point_id}")
    if bias["coefficient"] and np.ptp(norms) <= 1e-12:
        errors.append("Nonzero point biases must have sampled distances, not one fixed radius")
    if "reference" in bias:
        reference = read_json(Path(bias["reference"]["file"]))
        scale = bias["coefficient"] / reference["coefficient"]
        if bias["reference"]["scale"] != scale:
            errors.append("Bias reference scale differs from the coefficient ratio")
        for point_id in points:
            expected_bias = np.asarray(reference["offsets_B_mm"][point_id]) * scale
            if not np.array_equal(offsets[point_id], expected_bias):
                errors.append(f"Point bias does not reuse the saved reference: {point_id}")
    public_runs = [item for item in parameters["runs"] if item["batch_id"] == batch]
    if public_runs != [public_run(source, dataset, run, bias)]:
        errors.append("Published run metadata differs from the capture source")
    destination = dataset / batch
    record = read_json(destination / "record.json")
    if (set(record) != {"batch", "inertia", "target_bias", "frames"}
            or record["batch"] != batch or record["inertia"] != inertia
            or record["target_bias"] != bias["coefficient"]):
        errors.append("Public record metadata differs from the capture source")
    originals = {row["image_filename"]: row for row in rows}
    frames = record["frames"]
    files = {path.relative_to(destination).as_posix()
             for path in (destination / "calibration_images").iterdir() if path.is_file()}
    if Counter(frame["image"] for frame in frames) != Counter(originals.keys()) or files != set(originals):
        errors.append("Public images and records do not exactly match the 600 source captures")
    original_ideals = {
        Path(frame["image"]).stem.split("_")[1]: matrix(frame["ideal"])
        for frame in read_json(dataset / "B000/record.json")["frames"]
    }
    previous_actuals = {
        tuple(Path(frame["image"]).stem.split("_")[1:]): matrix(frame["actual"])
        for frame in read_json(dataset / "B002/record.json")["frames"]
    }
    job = read_json(source / "capture_job.json")
    commands = {item["image_filename"]: item for item in job["runs"][0]["captures"]}
    if set(commands) != set(originals):
        errors.append("Frozen job images differ from the completed source records")
    geometry = AcquisitionGeometry(
        plan["scene_config_snapshot"], read_json(ROOT / "assets/ur10/robot_description.json"),
    )
    camera = np.asarray(parameters["camera"]["camera_matrix"], dtype=float)
    distortion = np.asarray(parameters["camera"]["dist_coeffs"], dtype=float)
    transforms = parameters["system_transforms"]
    hand_eye = matrix(transforms["E_T_C_hand_eye"])
    target = matrix(transforms["B_T_M_charuco_top_left"])
    board = make_board(0)
    detector = cv2.aruco.CharucoDetector(board)
    extrinsics = parameters["extrinsics_by_capture"]
    expected_keys = {f"{batch}/{r['point_id']}/{r['direction_id']}" for r in rows}
    if {key for key in extrinsics if key.startswith(batch + "/")} != expected_keys:
        errors.append("Published extrinsics do not exactly cover this batch")
    results = []
    report_path = source / "per_image_validation.jsonl"
    with report_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            source_row = originals[frame["image"]]
            point_id = source_row["point_id"]
            point = points[point_id]
            row = dict(source_row)
            row.update({"run_id": batch, "B_T_E_ideal_mm": frame["ideal"],
                        "B_T_E_actual_mm": frame["actual"],
                        "extrinsic_id": f"{batch}/{point_id}/{source_row['direction_id']}"})
            result = validate_capture(
                row, destination, inertia, point, hand_eye, target, board, detector,
                camera, distortion, extrinsics,
            )
            issues = result["errors"]
            if (set(frame) != {"image", "ideal", "actual"}
                    or frame["ideal"] != source_row["B_T_E_ideal_mm"]
                    or frame["actual"] != source_row["B_T_E_actual_mm"]):
                issues.append("Public frame differs from its original render record")
            if (source_row["q_ideal_rad"] != point["q_ideal_rad"]
                    or source_row["B_T_E_ideal_requested_mm"] != point["B_T_E_ideal_mm"]):
                issues.append("Original commanded target or ideal joints changed")
            if source_row["target_bias_B_mm"] != offsets[point_id]:
                issues.append("Point bias differs from its frozen vector")
            command = commands[frame["image"]]
            for key in ("q_ideal_rad", "direction_B", "target_bias_B_mm",
                        "B_T_E_biased_target_mm", "B_T_E_approach_start_mm"):
                if source_row[key] != command[key]:
                    issues.append(f"Rendered record differs from its frozen job: {key}")
            if source_row["B_T_E_ideal_requested_mm"] != command["B_T_E_ideal_mm"]:
                issues.append("Recorded original target differs from its frozen job")
            direction = next(item for item in point["directions"]
                             if item["direction_id"] == row["direction_id"])
            if source_row["direction_B"] != direction["direction_B"]:
                issues.append("Approach direction changed from the original fixed plan")
            expected_target = matrix(point["B_T_E_ideal_mm"]).copy()
            expected_target[:3, 3] += np.asarray(offsets[point_id])
            result["fixed_biased_target"] = check_pose(
                matrix(source_row["B_T_E_biased_target_mm"]), expected_target,
                "Biased target differs from original target plus frozen point bias", issues,
            )
            expected_start = expected_target.copy()
            expected_start[:3, 3] -= direction["distance_mm"] * np.asarray(direction["direction_B"])
            result["fixed_approach_start"] = check_pose(
                matrix(source_row["B_T_E_approach_start_mm"]), expected_start,
                "Approach start differs from the biased target and frozen approach distance", issues,
            )
            result["reference_ideal"] = check_pose(
                matrix(frame["ideal"]), original_ideals[point_id],
                "Original approved ideal pose changed", issues,
            )
            pair = (point_id, f"D{int(row['direction_id'][1:]):03d}")
            expected_actual = previous_actuals[pair].copy()
            expected_actual[:3, 3] += (
                np.asarray(offsets[point_id])
                + 2 * (inertia - 0.2) * np.asarray(direction["direction_B"])
            )
            result["cross_batch_reference"] = check_pose(
                matrix(frame["actual"]), expected_actual,
                "Change from B002 differs from point bias plus the inertia change", issues,
            )
            if inertia == 0.2:
                result["same_inertia_reference"] = result["cross_batch_reference"]
            result["original_image_bytes_equal"] = os.path.samefile(
                directory / frame["image"], destination / frame["image"],
            )
            if not result["original_image_bytes_equal"]:
                issues.append("Public image is not the original rendered hard link")
            visibility = geometry.visibility(matrix(frame["actual"]), margin_px=MINIMUM_MARGIN_PX)
            result["plate_margin_px"] = visibility["minimum_margin_px"]
            if not visibility["complete_plate_visible"]:
                issues.append("Actual white plate has less than 60 px image margin")
            pnp = result.get("pnp")
            if pnp and pnp["vision_pose"] is not None:
                projected = project_board_rectangle(
                    matrix(pnp["vision_pose"]), *geometry.plate_size_mm,
                    geometry.pattern_size_mm, camera, distortion,
                )
                result["pnp_plate_margin_px"] = margin(projected, geometry.resolution_px)
                if result["pnp_plate_margin_px"] < MINIMUM_MARGIN_PX:
                    issues.append("Image PnP white plate has less than 60 px margin")
            result["passed"] = not issues
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            if len(results) % 50 == 0:
                stream.flush()
                print(f"BATCH_VALIDATION {batch} {len(results)}/600", flush=True)
    pnp_results = [item["pnp"] for item in results
                   if item.get("pnp") and item["pnp"]["vision_pose"] is not None]
    summary = {
        "batch": batch, "inertia": inertia, "target_bias": bias["coefficient"],
        "passed": not errors and len(results) == 600 and all(item["passed"] for item in results),
        "images": len(results), "expected_images": 600,
        "failed_images": sum(not item["passed"] for item in results),
        "target_bias_distance_range_mm": [min(norms), max(norms)],
        "target_bias_limit_mm": bias["limit_mm"],
        "all_31_markers_48_corners_detected": len(pnp_results) == 600 and all(
            item["marker_count"] == 31 and item["charuco_corner_count"] == 48 for item in pnp_results),
        "minimum_plate_margin_px": min((item["plate_margin_px"] for item in results), default=None),
        "minimum_pnp_plate_margin_px": min((item["pnp_plate_margin_px"] for item in results
                                            if "pnp_plate_margin_px" in item), default=None),
        "maximum_direction_error_mm": max((item["geometry"]["direction_error_mm"] for item in results
                                            if "geometry" in item), default=None),
        "maximum_reference_ideal_difference_mm": max((item["reference_ideal"]["translation_error_mm"]
                                                       for item in results), default=None),
        "maximum_same_inertia_bias_error_mm": max((item["same_inertia_reference"]["translation_error_mm"]
                                                    for item in results if "same_inertia_reference" in item), default=None),
        "maximum_cross_batch_error_mm": max((item["cross_batch_reference"]["translation_error_mm"]
                                              for item in results), default=None),
        "maximum_pnp_truth_translation_difference_mm": max((item["ground_truth_comparison"]["translation_error_mm"]
                                                            for item in pnp_results), default=None),
        "parameter_extrinsics": len(expected_keys), "errors": errors,
        "per_image_results": report_path.relative_to(dataset).as_posix(),
        "note": "Only this appended batch was validated; earlier validation.json retains its original scope. Synthetic image checks are not hardware accuracy certification.",
    }
    save_json(dataset / f"validation_{batch}.json", summary)
    update_validation_index(dataset, batch, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "validate", "all"], nargs="?", default="all")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    dataset, source = args.dataset.resolve(), args.source.resolve()
    if args.action in ("export", "all"):
        export(dataset, source)
    if args.action in ("validate", "all"):
        raise SystemExit(0 if validate(dataset, source) else 1)
