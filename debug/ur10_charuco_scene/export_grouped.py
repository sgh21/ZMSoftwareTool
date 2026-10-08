"""Publish and check the accepted preview plus two fixed-pose inertia batches.

Usage: python export_grouped.py [export|validate|all] --dataset PATH
Detailed render jobs and journals remain in PATH/_work; public records stay small.
"""

import argparse
import copy
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from acquisition_geometry import AcquisitionGeometry
from collect_images import save_json
from prepare_charuco import make_board
from validate_collection import (
    check_pose, margin, matrix, project_board_rectangle, read_json, validate_capture,
)

ROOT = Path(__file__).resolve().parent
MINIMUM_MARGIN_PX = 60


def sources(dataset):
    """Return the existing render records; never alter the approved preview."""
    work = dataset / "_work"
    preview = (work / Path(read_json(work / "grouped_source.json")["preview_dir"])).resolve()
    preview_parameters = read_json(preview / "parameters.json")
    parameters = read_json(work / "parameters.json")
    runs = [(preview, preview_parameters["runs"][0]),
            *((work, run) for run in parameters["runs"])]
    if len(runs) != 3:
        raise ValueError("Expected the reference preview and two rendered runs")
    result = []
    for index, ((directory, run), inertia, count) in enumerate(zip(
            runs, [0.0, 0.1, 0.2], [30, 600, 600])):
        batch = f"B{index:03d}"
        if run["batch_id"] != batch or run["inertia"] != inertia:
            raise ValueError(f"Expected {batch} with inertia {inertia}")
        records = read_json(directory / run["run_id"] / "records.json")
        rows = sorted(records["captures"], key=lambda row: row["capture_index"])
        if not records["complete"] or len(rows) != count:
            raise ValueError(f"{batch} must be complete with {count} images before export")
        result.append({
            "batch": batch, "inertia": inertia, "count": count,
            "directory": directory / run["run_id"], "rows": rows,
        })
    for key in ("camera", "charuco", "system_transforms"):
        if preview_parameters[key] != parameters[key]:
            raise ValueError(f"Preview and new captures have different {key}")
    return result, parameters


def save_record(path, batch, inertia, rows, target_bias=None):
    """Keep each matrix row on one line without rounding any stored values."""
    lines = ["{", f'  "batch": {json.dumps(batch)},',
             f'  "inertia": {json.dumps(inertia)},']
    if target_bias is not None:
        lines.append(f'  "target_bias": {json.dumps(target_bias)},')
    lines.append('  "frames": [')
    for index, row in enumerate(rows):
        lines += ["    {", f'      "image": {json.dumps(row["image_filename"])},']
        for field, key in (("ideal", "B_T_E_ideal_mm"), ("actual", "B_T_E_actual_mm")):
            lines.append(f'      "{field}": [')
            lines.extend("        " + json.dumps(values) + ("," if number < 3 else "")
                         for number, values in enumerate(row[key]))
            lines.append("      ]" + ("," if field == "ideal" else ""))
        lines.append("    }" + ("," if index < len(rows) - 1 else ""))
    lines += ["  ]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(dataset):
    dataset = Path(dataset).resolve()
    public_parameters = dataset / "parameters.json"
    if public_parameters.exists():
        batches = {run["batch_id"] for run in read_json(public_parameters)["runs"]}
        if batches - {"B000", "B001", "B002"}:
            raise ValueError("Additional batches exist; use append_batch.py to preserve them")
    groups, parameters = sources(dataset)
    parameters = copy.deepcopy(parameters)
    parameters["runs"] = []
    parameters["extrinsics_by_capture"] = {}
    parameters["fixed_plan"] = "_work/fixed_observation_plan.json"
    parameters["scene_snapshot"] = "_work/scene_snapshot.blend"
    parameters["purpose"] = "Accepted 30 target poses: imported inertia-0 reference and new inertia-0.1/0.2 captures"
    parameters["record_definition"] = {
        "file": "Bxxx/record.json",
        "batch": "Acquisition batch; B000=0.0, B001=0.1, B002=0.2",
        "inertia": "Dimensionless stopping coefficient; displacement = 2 * inertia mm",
        "frames": "One entry per image: image, ideal, actual",
        "image": "Image path relative to its batch directory; filename identifies point and direction",
        "ideal": "4x4 nominal B_T_E evaluated at the frozen target joint angles in Blender; E is tool0; translation in mm. The exact commanded target is in fixed_plan; the difference is floating-point roundoff.",
        "actual": "4x4 B_T_E at the rendered stop; translation in mm",
    }
    reference = (dataset / "_work" / Path(
        read_json(dataset / "_work/grouped_source.json")["preview_dir"],
    )).resolve()
    reference_path = (reference.relative_to(dataset).as_posix()
                      if reference.is_relative_to(dataset) else str(reference))
    parameters["reference_source"] = {
        "preview_dir": reference_path,
        "batch": "B000", "images": 30,
        "method": "Original PNG bytes copied without re-rendering or re-encoding",
        "coverage": "One existing D001 image per pose; not twenty independently captured reference directions",
    }
    for group in groups:
        batch = group["batch"]
        destination = dataset / batch
        (destination / "calibration_images").mkdir(parents=True, exist_ok=True)
        for row in group["rows"]:
            original = group["directory"] / row["image_filename"]
            target = destination / row["image_filename"]
            if target.exists():
                if original.read_bytes() != target.read_bytes():
                    raise ValueError(f"Existing public image differs from its source: {target}")
            elif batch == "B000":
                shutil.copy2(original, target)
            else:
                os.link(original, target)
            if batch == "B000" and original.read_bytes() != target.read_bytes():
                raise ValueError(f"Reference copy changed image bytes: {target}")
            extrinsic_id = f"{batch}/{row['point_id']}/{row['direction_id']}"
            parameters["extrinsics_by_capture"][extrinsic_id] = {
                "B_T_C": row["B_T_C_mm"], "C_T_M": row["C_T_M_mm"],
                "image_filename": f"{batch}/{row['image_filename']}",
            }
        save_record(destination / "record.json", batch, group["inertia"], group["rows"])
        parameters["runs"].append({
            "run_id": batch, "batch_id": batch, "inertia": group["inertia"],
            "expected_images": group["count"], "record": f"{batch}/record.json",
            "direction_ids": ["D01"] if batch == "B000" else [f"D{i:02d}" for i in range(1, 21)],
        })
        print(f"EXPORTED {batch}: {group['count']} images", flush=True)
    save_json(dataset / "parameters.json", parameters)
    return dataset


def validate(dataset):
    dataset = Path(dataset).resolve()
    groups, source_parameters = sources(dataset)
    parameters = read_json(dataset / "parameters.json")
    plan = read_json(dataset / parameters["fixed_plan"])
    geometry = AcquisitionGeometry(
        plan["scene_config_snapshot"], read_json(ROOT / "assets/ur10/robot_description.json"),
    )
    points = {point["point_id"]: point for point in plan["points"]}
    full_pairs = {(point["point_id"], direction["direction_id"])
                  for point in plan["points"] for direction in point["directions"]}
    errors = []
    if (len(points) != 30 or len(full_pairs) != 600
            or any(len(point["directions"]) != 20 for point in points.values())):
        errors.append("Frozen plan must contain 30 poses and 20 directions per pose")
    for key in ("camera", "charuco", "system_transforms", "length_unit"):
        if parameters[key] != source_parameters[key]:
            errors.append(f"Published {key} differs from the capture parameters")
    camera = np.asarray(parameters["camera"]["camera_matrix"], dtype=float)
    distortion = np.asarray(parameters["camera"]["dist_coeffs"], dtype=float)
    transforms = parameters["system_transforms"]
    hand_eye = matrix(transforms["E_T_C_hand_eye"])
    target = matrix(transforms["B_T_M_charuco_top_left"])
    center = matrix(transforms["B_T_target_center"])
    check_pose(hand_eye, geometry.H_E_T_C_mm, "Frozen hand-eye mismatch", errors)
    check_pose(center, geometry.B_T_marker_center_mm, "Frozen target center mismatch", errors)
    check_pose(target, geometry.B_T_M_mm, "Frozen ChArUco frame mismatch", errors)
    check_pose(center @ matrix(transforms["target_center_T_M"]), target,
               "Target center/top-left chain mismatch", errors)
    board = make_board(0)
    detector = cv2.aruco.CharucoDetector(board)
    extrinsics = parameters["extrinsics_by_capture"]
    expected_extrinsics = set()
    reference_ideals = {}
    reference_starts = {}
    results, batch_summaries = [], []
    public_runs = {run["batch_id"]: run for run in parameters["runs"]}
    if set(public_runs) != {group["batch"] for group in groups}:
        errors.append("Published parameter batches must be B000, B001 and B002")
    report_path = dataset / "_work/per_image_validation.jsonl"
    with report_path.open("w", encoding="utf-8") as stream:
        for group in groups:
            batch = group["batch"]
            directory = dataset / batch
            record = read_json(directory / "record.json")
            batch_errors = []
            if (set(record) != {"batch", "inertia", "frames"}
                    or record["batch"] != batch or record["inertia"] != group["inertia"]):
                batch_errors.append("Public record must contain only the correct batch, inertia and frames")
            frames = record["frames"]
            run = public_runs.get(batch, {})
            if (run.get("inertia") != group["inertia"]
                    or run.get("expected_images") != group["count"]
                    or run.get("record") != f"{batch}/record.json"):
                batch_errors.append("Parameter batch metadata differs from expected captures")
            originals = {row["image_filename"]: row for row in group["rows"]}
            files = {path.relative_to(directory).as_posix()
                     for path in (directory / "calibration_images").iterdir() if path.is_file()}
            names = [frame["image"] for frame in frames]
            if (len(frames) != group["count"] or Counter(names) != Counter(originals.keys())
                    or files != set(originals)):
                batch_errors.append("Public images and records do not exactly match source captures")
            expected_pairs = ({(point, "D01") for point in points}
                              if batch == "B000" else full_pairs)
            source_pairs = Counter((row["point_id"], row["direction_id"]) for row in group["rows"])
            if source_pairs != Counter(expected_pairs):
                batch_errors.append("Captured point/direction pairs differ from the fixed protocol")
            first_result = len(results)
            for frame in frames:
                if frame["image"] not in originals:
                    continue
                source_row = originals[frame["image"]]
                row = dict(source_row)
                row.update({
                    "run_id": batch, "batch_id": batch,
                    "B_T_E_ideal_mm": frame["ideal"], "B_T_E_actual_mm": frame["actual"],
                    "extrinsic_id": f"{batch}/{source_row['point_id']}/{source_row['direction_id']}",
                })
                expected_extrinsics.add(row["extrinsic_id"])
                result = validate_capture(
                    row, directory, group["inertia"], points.get(row["point_id"]),
                    hand_eye, target, board, detector, camera, distortion, extrinsics,
                )
                if set(frame) != {"image", "ideal", "actual"}:
                    result["errors"].append("Public frame must contain only image, ideal and actual")
                if (frame["ideal"] != source_row["B_T_E_ideal_mm"]
                        or frame["actual"] != source_row["B_T_E_actual_mm"]):
                    result["errors"].append("Public pose values differ from original render records")
                original_image = group["directory"] / frame["image"]
                public_image = directory / frame["image"]
                bytes_equal = (batch != "B000" and os.path.samefile(original_image, public_image))
                if not bytes_equal:
                    bytes_equal = original_image.read_bytes() == public_image.read_bytes()
                result["original_image_bytes_equal"] = bytes_equal
                if not bytes_equal:
                    result["errors"].append("Published image differs from the original capture bytes")
                ideal = matrix(frame["ideal"])
                point = points[row["point_id"]]
                approach = next(item for item in point["directions"]
                                if item["direction_id"] == row["direction_id"])
                expected_start = matrix(point["B_T_E_ideal_mm"]).copy()
                expected_start[:3, 3] -= approach["distance_mm"] * np.asarray(approach["direction_B"])
                actual_start = matrix(source_row["B_T_E_approach_start_mm"])
                result["fixed_approach_start"] = check_pose(
                    actual_start, expected_start, "Approach start differs from its fixed path distance",
                    result["errors"],
                )
                pair = (row["point_id"], row["direction_id"])
                if pair in reference_starts:
                    result["cross_batch_approach_start"] = check_pose(
                        actual_start, reference_starts[pair],
                        "Approach start changed between batches", result["errors"],
                    )
                else:
                    reference_starts[pair] = actual_start
                if batch == "B000":
                    reference_ideals[row["point_id"]] = ideal
                elif row["point_id"] in reference_ideals:
                    result["reference_ideal"] = check_pose(
                        ideal, reference_ideals[row["point_id"]],
                        "Target pose changed from the approved preview", result["errors"],
                    )
                visibility = geometry.visibility(matrix(frame["actual"]), margin_px=MINIMUM_MARGIN_PX)
                result["plate_margin_px"] = visibility["minimum_margin_px"]
                if not visibility["complete_plate_visible"]:
                    result["errors"].append("Actual white plate has less than 60 px image margin")
                pnp = result.get("pnp")
                if pnp and pnp["vision_pose"] is not None:
                    projected = project_board_rectangle(
                        matrix(pnp["vision_pose"]), *geometry.plate_size_mm,
                        geometry.pattern_size_mm, camera, distortion,
                    )
                    result["pnp_plate_margin_px"] = margin(projected, geometry.resolution_px)
                    if result["pnp_plate_margin_px"] < MINIMUM_MARGIN_PX:
                        result["errors"].append("Image PnP white plate has less than 60 px margin")
                result["passed"] = not result["errors"]
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if len(results) % 50 == 0:
                    stream.flush()
                    print(f"GROUPED_VALIDATION {len(results)}/1230", flush=True)
            checked = results[first_result:]
            batch_summaries.append({
                "batch": batch, "inertia": group["inertia"], "images": len(checked),
                "expected_images": group["count"],
                "failed_images": sum(not row["passed"] for row in checked),
                "errors": batch_errors,
                "passed": not batch_errors and all(row["passed"] for row in checked),
            })
    if set(extrinsics) != expected_extrinsics:
        errors.append("Parameter extrinsics do not exactly match all published images")
    if len(results) != 1230:
        errors.append("Expected 30 reference images plus two new batches of 600 images")
    geometry_results = [row["geometry"] for row in results if "geometry" in row]
    pnp_results = [row["pnp"] for row in results if row.get("pnp") and row["pnp"]["vision_pose"] is not None]
    summary = {
        "passed": not errors and all(batch["passed"] for batch in batch_summaries),
        "images": len(results), "expected_images": 1230,
        "failed_images": sum(not row["passed"] for row in results),
        "batches": batch_summaries,
        "all_original_image_bytes_preserved": all(row["original_image_bytes_equal"] for row in results),
        "all_31_markers_48_corners_detected": len(pnp_results) == 1230 and all(
            row["marker_count"] == 31 and row["charuco_corner_count"] == 48 for row in pnp_results),
        "minimum_plate_margin_px": min((row["plate_margin_px"] for row in results), default=None),
        "minimum_pnp_plate_margin_px": min((row["pnp_plate_margin_px"] for row in results
                                             if "pnp_plate_margin_px" in row), default=None),
        "maximum_direction_error_mm": max((row["direction_error_mm"] for row in geometry_results), default=None),
        "maximum_reference_ideal_difference_mm": max((row["reference_ideal"]["translation_error_mm"]
                                                       for row in results if "reference_ideal" in row), default=None),
        "maximum_pnp_truth_translation_difference_mm": max((row["ground_truth_comparison"]["translation_error_mm"]
                                                            for row in pnp_results), default=None),
        "parameter_extrinsics": len(extrinsics), "errors": errors,
        "per_image_results": "_work/per_image_validation.jsonl",
        "note": "Synthetic pinhole results, not hardware measurement accuracy. B000 retains only its 30 original captures.",
    }
    save_json(dataset / "validation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "validate", "all"], nargs="?", default="all")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    if args.action in ("export", "all"):
        export(args.dataset)
    if args.action in ("validate", "all"):
        raise SystemExit(0 if validate(args.dataset) else 1)
