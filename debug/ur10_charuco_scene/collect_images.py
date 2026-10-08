"""Prepare one fixed observation plan and collect independent Blender datasets."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from acquisition_geometry import AcquisitionGeometry, inverse

ROOT = Path(__file__).resolve().parent


def save_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_filename(batch_id, point_id, direction_id):
    direction = int(direction_id.removeprefix("D"))
    return f"calibration_images/{batch_id}_{point_id}_D{direction:03d}.png"


def prepare_dataset(args):
    geometry = AcquisitionGeometry.from_root(ROOT)
    plan_path = args.plan.resolve()
    # A missing plan must never silently replace the approved target/direction set.
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan["point_count"] != args.points or plan["direction_count"] != args.directions
            or any(len(point["directions"]) != args.directions for point in plan["points"])):
        raise ValueError("Existing fixed plan has a different size; choose a matching --plan")
    if plan["scene_config_snapshot"] != geometry.config:
        raise ValueError("Scene configuration differs from the frozen plan; use its original scene")
    print(f"REUSE_FIXED_PLAN {plan_path}", flush=True)

    dataset = args.dataset.resolve()
    dataset.mkdir(parents=True, exist_ok=False)
    scene_source = getattr(args, "scene_source", None) or ROOT / "output/ur10_charuco_scene.blend"
    shutil.copy2(scene_source, dataset / "scene_snapshot.blend")
    save_json(dataset / "fixed_observation_plan.json", plan)
    bias_coefficient = getattr(args, "target_bias", None) or 0.0
    target_bias = None
    if bias_coefficient:
        bias_source = getattr(args, "bias_source", None)
        if bias_source:
            reference = json.loads(Path(bias_source).read_text(encoding="utf-8"))
            if reference["coefficient"] <= 0:
                raise ValueError("The reference bias coefficient must be positive")
            if set(reference["offsets_B_mm"]) != {p["point_id"] for p in plan["points"]}:
                raise ValueError("Reference bias points differ from the fixed plan")
            scale = bias_coefficient / reference["coefficient"]
            offsets = {key: (np.asarray(value) * scale).tolist()
                       for key, value in reference["offsets_B_mm"].items()}
            seed = reference["seed"]
        else:
            rng = np.random.default_rng(args.bias_seed)
            offsets = {}
            for point in plan["points"]:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                offsets[point["point_id"]] = (
                    direction * rng.uniform(0, 5 * bias_coefficient)
                ).tolist()
            seed = args.bias_seed
        target_bias = {
            "coefficient": bias_coefficient, "maximum_mm": 5.0,
            "limit_mm": 5 * bias_coefficient, "seed": seed,
            "distribution": "isotropic direction; radius uniform [0, limit_mm]",
            "offsets_B_mm": offsets,
        }
        if bias_source:
            target_bias["reference"] = {
                "file": str(Path(bias_source).resolve()),
                "coefficient": reference["coefficient"], "scale": scale,
                "method": "Scale saved vectors; no new random sampling",
            }
        # Save the sampled vectors before planning. Resume reads the saved job;
        # neither approaches nor subsequent captures draw new random offsets.
        save_json(dataset / "target_bias.json", target_bias)
    camera = geometry.config["camera"]
    charuco = geometry.config["charuco"]
    parameters = {
        "schema": "ur10_simulated_camera_parameters_v1",
        "source": "Blender nominal scene ground truth, not an estimated hardware calibration",
        "length_unit": "mm",
        "joint_angle_unit": "rad",
        "transform_convention": "A_T_B maps coordinates in B into A; 4x4 homogeneous matrices",
        "coordinate_systems": plan["coordinate_convention"],
        "camera": {
            "reference_model": camera["reference_model"],
            "image_size_px": geometry.resolution_px.tolist(),
            "focal_length_mm": camera["focal_length_mm"],
            "pixel_pitch_um": camera["pixel_pitch_um"],
            "camera_matrix": geometry.K_px.tolist(),
            "distortion_model": "OpenCV Brown-Conrady [k1,k2,p1,p2,k3]; ideal pinhole",
            "dist_coeffs": [0.0] * 5,
            "axes": "+X right, +Y down, +Z forward",
        },
        "charuco": {
            "dictionary": "DICT_5X5_1000", "squares_xy": charuco["squares_xy"],
            "square_length_mm": charuco["square_length_m"] * 1000,
            "marker_length_mm": charuco["marker_length_m"] * 1000,
            "pattern_size_mm": geometry.pattern_size_mm.tolist(),
            "plate_size_mm": geometry.plate_size_mm.tolist(),
            "marker_ids": list(range(31)),
            "center_definition": "Center of the printed pattern surface; same axes as M",
        },
        "system_transforms": {
            "W_T_B": geometry.W_T_B_mm.tolist(),
            "E_T_C_hand_eye": geometry.H_E_T_C_mm.tolist(),
            "B_T_target_center": geometry.B_T_marker_center_mm.tolist(),
            "B_T_M_charuco_top_left": geometry.B_T_M_mm.tolist(),
            "target_center_T_M": (inverse(geometry.B_T_marker_center_mm) @ geometry.B_T_M_mm).tolist(),
        },
        "intrinsics_and_hand_eye_constant": True,
        "extrinsics_definition": "B_T_C is camera pose in robot base; C_T_M maps ChArUco points into camera",
        "extrinsics_by_capture": {},
        "fixed_plan": "fixed_observation_plan.json",
        "scene_snapshot": "scene_snapshot.blend",
        "image_naming": {
            "pattern": "B{batch:03d}_P{point:03d}_D{direction:03d}.png",
            "batch": "Zero-based acquisition batch: B000, B001, ...",
            "point": "Fixed observation point, one-based",
            "direction": "Fixed approach direction, one-based; internal D01 is written as D001",
        },
        "motion_model": {
            "name": "Prescribed direction-dependent stopping overshoot, not rigid-body dynamics",
            "formula": "p_stop = p_ideal + 2 * inertia * unit_direction_B [mm]",
            "inertia_range": [0, 1], "maximum_offset_mm": 2,
            "orientation": "Same ideal orientation for every direction at a given point",
            "directions_per_point": args.directions,
            "arrivals_per_point_direction": 1,
            "settle_frames": 6, "fps": 30,
        },
        "renderer": {"engine": "CYCLES", "device": "OPTIX", "denoiser": "OPTIX",
                     "samples": args.samples, "image_format": "PNG", "png_compression": 15,
                     "color_mode": "RGB", "color_depth": 8, "render_seed": 0,
                     "exposure_ev": geometry.config["render_exposure_ev"]},
        "limitations": "No calibrated lens distortion, sensor noise, rolling shutter or physical inertia dynamics. Different directions are not same-direction repeatability samples.",
    }
    if target_bias:
        parameters["target_bias"] = target_bias
        parameters["motion_model"]["formula"] = (
            "p_target = p_ideal + target_bias_B; "
            "p_start = p_target - approach_distance * unit_direction_B; "
            "p_stop = p_target + 2 * inertia * unit_direction_B [mm]"
        )
    runs = []
    for run_index, inertia in enumerate(args.inertia, 1):
        run_id = f"run_{run_index:02d}_inertia_{inertia:g}"
        batch_id = f"B{getattr(args, 'batch_start', 0) + run_index - 1:03d}"
        run_dir = dataset / run_id
        (run_dir / "calibration_images").mkdir(parents=True)
        captures = []
        for point in plan["points"]:
            ideal = np.asarray(point["B_T_E_ideal_mm"])
            target = ideal.copy()
            target_q = point["q_ideal_rad"]
            if target_bias:
                bias = target_bias["offsets_B_mm"][point["point_id"]]
                target[:3, 3] += bias
                target_q = geometry.ik(target, target_q)
            for approach in point["directions"]:
                path = geometry.approach(
                    target, target_q, approach["direction_B"],
                    distance_mm=approach.get("distance_mm", plan["sampling"]["approach_distance_mm"]),
                    step_mm=plan["sampling"]["maximum_path_step_mm"],
                    endpoint_offset_mm=2 * inertia,
                )
                actual = np.asarray(path["actual_B_T_E_mm"])
                visible = geometry.visibility(actual, margin_px=plan["sampling"]["image_margin_px"])
                if not visible["complete_plate_visible"]:
                    raise ValueError(f"Actual endpoint clips the plate: {point['point_id']}")
                index = len(captures)
                captures.append({
                    "capture_index": index, "batch_id": batch_id, "point_id": point["point_id"],
                    "direction_id": approach["direction_id"], "sample_id": "001",
                    "arrival_index": 1, "frame_index": 1,
                    "image_filename": image_filename(batch_id, point["point_id"], approach["direction_id"]),
                    "inertia": inertia, "offset_distance_mm": 2 * inertia,
                    "direction_B": approach["direction_B"],
                    "B_T_E_ideal_mm": point["B_T_E_ideal_mm"],
                    "q_ideal_rad": point["q_ideal_rad"],
                    "B_T_E_stop_requested_mm": actual.tolist(),
                    "q_stop_rad": path["q_actual_rad"],
                    "B_T_E_approach_start_mm": path["start_B_T_E_mm"],
                    "path_q_rad": path["path_q_rad"],
                    "endpoint_visibility": visible,
                })
                if target_bias:
                    captures[-1].update({
                        "target_bias_B_mm": bias,
                        "B_T_E_biased_target_mm": target.tolist(),
                    })
        runs.append({"run_id": run_id, "batch_id": batch_id, "inertia": inertia, "captures": captures})
        print(f"PREPARED {run_id}: {len(captures)} independent arrivals", flush=True)
    parameters["runs"] = [
        {"run_id": r["run_id"], "batch_id": r["batch_id"], "inertia": r["inertia"],
         "expected_images": len(r["captures"])}
        for r in runs
    ]
    save_json(dataset / "parameters.json", parameters)
    job = {
        "schema": "ur10_blender_capture_job_v1", "dataset_dir": str(dataset),
        "scene_file": str(dataset / "scene_snapshot.blend"),
        "robot_description": str((ROOT / "assets/ur10/robot_description.json").resolve()),
        "scene_config_snapshot": geometry.config,
        "parameters_file": "parameters.json", "runs": runs,
    }
    if target_bias:
        job["validation_script"] = "append_batch.py (provide the grouped dataset and this source directory)"
    save_json(dataset / "capture_job.json", job)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inertia", type=float, nargs="+")
    parser.add_argument("--points", type=int, default=30)
    parser.add_argument("--directions", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--target-bias", type=float,
                        help="Fixed random per-point translation, upper bound 5 * coefficient mm")
    parser.add_argument("--bias-seed", type=int, default=20260928)
    parser.add_argument("--bias-source", type=Path,
                        help="Scale an existing target_bias.json to --target-bias without resampling")
    parser.add_argument("--scene-source", type=Path, help="Use a previously approved scene snapshot")
    parser.add_argument("--plan", type=Path, default=ROOT / "fixed_observation_plan.json")
    parser.add_argument("--blender", type=Path, default=Path("D:/Softwares/Blender/blender.exe"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Render up to N new captures, then resume later")
    args = parser.parse_args()
    if args.resume:
        dataset = args.dataset.resolve()
        if not (dataset / "capture_job.json").exists():
            parser.error("--resume requires a prepared dataset")
        if args.inertia is not None or args.target_bias is not None or args.bias_source is not None:
            parser.error("On resume, inertia and target bias come from the saved job; do not change them")
    else:
        if not args.inertia or any(not 0 <= k <= 1 for k in args.inertia):
            parser.error("Provide --inertia with one or more values in [0,1]")
        if args.target_bias is not None and not 0 <= args.target_bias <= 1:
            parser.error("--target-bias must be in [0,1]")
        if args.bias_source is not None and not args.target_bias:
            parser.error("--bias-source requires a positive --target-bias")
        dataset = prepare_dataset(args)
    if args.prepare_only:
        return
    job = json.loads((dataset / "capture_job.json").read_text(encoding="utf-8"))
    command = [str(args.blender), "--background", "--factory-startup", "--python-exit-code", "1", job["scene_file"],
               "--python", str(ROOT / "render_collection.py"), "--", "--job", str(dataset / "capture_job.json")]
    if args.limit:
        command += ["--limit", str(args.limit)]
    with (dataset / "render.log").open("a", encoding="utf-8") as log:
        print(f"RENDER_STARTED {dataset}", flush=True)
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Blender exited {result.returncode}; see {dataset / 'render.log'}")
    validator = job.get("validation_script", "validate_collection.py")
    print(f"RENDER_FINISHED {dataset}; run {validator} before using the dataset", flush=True)


if __name__ == "__main__":
    main()
