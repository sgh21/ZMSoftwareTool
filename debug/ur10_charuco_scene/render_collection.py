"""Blender-side trajectory playback, stationary capture and per-image truth records."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def save_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def matrix_mm(matrix):
    result = np.array(matrix, dtype=float)
    result[:3, 3] *= 1000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    job = json.loads(args.job.read_text(encoding="utf-8"))
    dataset = Path(job["dataset_dir"])
    parameters = json.loads((dataset / job["parameters_file"]).read_text(encoding="utf-8"))
    robot = json.loads(Path(job["robot_description"]).read_text(encoding="utf-8"))
    joint_links = [bpy.data.objects["UR10 | " + j["child"]] for j in robot["joints"] if j["type"] == "revolute"]
    scene = bpy.data.scenes["02_Wrist_camera_3072x2048"]
    bpy.context.window.scene = scene
    camera = scene.camera
    tool = bpy.data.objects["UR10 | tool0"]
    base = bpy.data.objects["UR10 | base_link"]
    board = bpy.data.objects["board_01 | fixed to workbench"]
    if sum(bool(o.get("fixed_marker")) for o in scene.objects) != 1:
        raise ValueError("The capture scene must contain exactly one fixed board")
    board_world = board.matrix_world.copy()
    B_T_W = np.linalg.inv(np.asarray(parameters["system_transforms"]["W_T_B"]))
    B_T_M = np.asarray(parameters["system_transforms"]["B_T_M_charuco_top_left"])
    CV_AXIS = Matrix.Diagonal((1, -1, -1, 1))
    H_actual = np.linalg.inv(matrix_mm(tool.matrix_world)) @ matrix_mm(camera.matrix_world @ CV_AXIS)
    if not np.allclose(H_actual, parameters["system_transforms"]["E_T_C_hand_eye"], atol=.001, rtol=0):
        raise ValueError("Scene camera mount differs from frozen hand-eye parameters")
    pattern_width, pattern_height = parameters["charuco"]["pattern_size_mm"]
    board_top_left = board.matrix_world @ Matrix.Translation((-pattern_width / 2000, pattern_height / 2000, .0001)) @ CV_AXIS
    if not np.allclose(B_T_W @ matrix_mm(board_top_left), B_T_M, atol=.001, rtol=0):
        raise ValueError("Scene board differs from frozen target coordinates")
    width_px = parameters["camera"]["image_size_px"][0]
    if abs(camera.data.lens / camera.data.sensor_width * width_px - parameters["camera"]["camera_matrix"][0][0]) > .001:
        raise ValueError("Scene focal length/sensor differs from frozen intrinsics")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    if not any(d.type == "OPTIX" for d in prefs.devices):
        raise RuntimeError("The configured OPTIX capture device is unavailable")
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = parameters["renderer"]["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX"
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.resolution_x, scene.render.resolution_y = parameters["camera"]["image_size_px"]
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    scene.render.fps = 30
    scene.frame_set(1)
    root_before = base.matrix_world.copy()
    moving_meshes = [o for o in scene.objects if o.type == "MESH" and "official mesh" in o.name and "base_link" not in o.name]
    table_z = job["scene_config_snapshot"]["table"]["top_z_m"]

    def set_q(q):
        for link, angle in zip(joint_links, q):
            link.rotation_euler.z = angle
        bpy.context.view_layer.update()

    def tool_pose():
        return B_T_W @ matrix_mm(tool.matrix_world)

    def check_table_clearance():
        # The local paths stay near the approved posture. This is a worktop
        # clearance check, not a complete self-collision/dynamics solver.
        bottom = min((o.matrix_world @ Vector(corner)).z for o in moving_meshes for corner in o.bound_box)
        if bottom <= table_z:
            raise ValueError("A moving robot link reaches the worktop envelope")
        return (bottom - table_z) * 1000

    ideal_by_point = {}
    for capture in job["runs"][0]["captures"]:
        if capture["point_id"] not in ideal_by_point:
            set_q(capture["q_ideal_rad"])
            ideal_by_point[capture["point_id"]] = tool_pose()
    current_q = np.deg2rad(job["scene_config_snapshot"]["robot_joint_angles_deg"])
    set_q(current_q)
    rendered = 0
    total = sum(len(r["captures"]) for r in job["runs"])
    all_records = {}
    for run in job["runs"]:
        directory = dataset / run["run_id"]
        journal = directory / "capture_journal.jsonl"
        records = []
        if journal.exists():
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
        completed = {r["capture_index"] for r in records}
        all_records[run["run_id"]] = records
        for row in records:
            capture = run["captures"][row["capture_index"]]
            row.setdefault("B_T_E_ideal_requested_mm", capture["B_T_E_ideal_mm"])
            row.setdefault("batch_id", capture["batch_id"])
            row.setdefault("arrival_index", capture["arrival_index"])
            row.setdefault("frame_index", capture["frame_index"])
            if not (directory / row["image_filename"]).is_file():
                raise ValueError("A journal entry has no image; restore it or start a new dataset")
            parameters["extrinsics_by_capture"][row["extrinsic_id"]] = {
                "B_T_C": row["B_T_C_mm"], "C_T_M": row["C_T_M_mm"],
                "image_filename": f"{run['run_id']}/{row['image_filename']}",
            }
        with journal.open("a", encoding="utf-8") as stream:
            for capture in run["captures"]:
                if capture["capture_index"] in completed:
                    continue
                if args.limit and rendered >= args.limit:
                    break
                started = time.monotonic()
                ideal = ideal_by_point[capture["point_id"]]
                clearance = float("inf")
                # Every sample makes its own departure and prescribed approach;
                # each image below is independently rendered, with no image reuse.
                start_q = np.asarray(capture["path_q_rad"][0])
                departure_steps = max(2, int(np.ceil(np.max(np.abs(start_q - current_q)) / np.deg2rad(.5))))
                for q in np.linspace(current_q, start_q, departure_steps + 1)[1:]:
                    set_q(q)
                    clearance = min(clearance, check_table_clearance())
                for q in capture["path_q_rad"]:
                    set_q(q)
                    clearance = min(clearance, check_table_clearance())
                current_q = np.asarray(capture["q_stop_rad"])
                for _ in range(parameters["motion_model"]["settle_frames"]):
                    bpy.context.view_layer.update()
                actual = tool_pose()
                target = np.asarray(capture["B_T_E_stop_requested_mm"])
                if np.linalg.norm(actual[:3, 3] - target[:3, 3]) > .002:
                    raise ValueError("Blender joint hierarchy does not reproduce the planned stop")
                delta = actual[:3, 3] - ideal[:3, 3]
                expected = (np.asarray(capture.get("target_bias_B_mm", [0, 0, 0]))
                            + capture["offset_distance_mm"] * np.asarray(capture["direction_B"]))
                if np.linalg.norm(delta - expected) > .002:
                    raise ValueError("Rendered stop does not match target bias plus inertia offset")
                if board.matrix_world != board_world or base.matrix_world != root_before:
                    raise ValueError("Fixed board or robot base moved during capture")
                camera_world_cv = camera.matrix_world @ CV_AXIS
                B_T_C = B_T_W @ matrix_mm(camera_world_cv)
                C_T_M = np.linalg.inv(B_T_C) @ B_T_M
                scene.render.filepath = str(directory / capture["image_filename"])
                bpy.ops.render.render(write_still=True)
                extrinsic_id = f"{run['run_id']}/{capture['point_id']}/{capture['direction_id']}"
                record = {
                    "capture_index": capture["capture_index"], "run_id": run["run_id"],
                    "batch_id": capture["batch_id"],
                    "point_id": capture["point_id"], "direction_id": capture["direction_id"],
                    "sample_id": capture["sample_id"], "image_filename": capture["image_filename"],
                    "arrival_index": capture["arrival_index"], "frame_index": capture["frame_index"],
                    "length_unit": "mm", "joint_angle_unit": "rad", "inertia": capture["inertia"],
                    "direction_B": capture["direction_B"], "commanded_offset_mm": capture["offset_distance_mm"],
                    "actual_offset_B_mm": delta.tolist(), "actual_offset_norm_mm": float(np.linalg.norm(delta)),
                    "B_T_E_ideal_mm": ideal.tolist(), "B_T_E_actual_mm": actual.tolist(),
                    "B_T_E_ideal_requested_mm": capture["B_T_E_ideal_mm"],
                    "B_T_E_approach_start_mm": capture["B_T_E_approach_start_mm"],
                    "q_ideal_rad": capture["q_ideal_rad"], "q_actual_rad": capture["q_stop_rad"],
                    "B_T_C_mm": B_T_C.tolist(), "C_T_M_mm": C_T_M.tolist(), "extrinsic_id": extrinsic_id,
                    "minimum_link_worktop_clearance_mm": clearance,
                    "approach_path_samples": len(capture["path_q_rad"]),
                    "departure_path_samples": departure_steps,
                    "stationary_hold_s": parameters["motion_model"]["settle_frames"] / 30,
                    "time_model": "Quasistatic samples with prescribed hold; not real-time dynamics",
                    "image_reused": False, "render_seconds": time.monotonic() - started,
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                if "target_bias_B_mm" in capture:
                    record["target_bias_B_mm"] = capture["target_bias_B_mm"]
                    record["B_T_E_biased_target_mm"] = capture["B_T_E_biased_target_mm"]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                parameters["extrinsics_by_capture"][extrinsic_id] = {
                    "B_T_C": B_T_C.tolist(), "C_T_M": C_T_M.tolist(),
                    "image_filename": f"{run['run_id']}/{capture['image_filename']}",
                }
                rendered += 1
                done = sum(len(v) for v in all_records.values())
                save_json(dataset / "progress.json", {"completed": done, "expected": total,
                          "last_image": f"{run['run_id']}/{capture['image_filename']}",
                          "last_render_seconds": record["render_seconds"],
                          "updated_at_utc": record["captured_at_utc"]})
                print(f"CAPTURE_PROGRESS {done}/{total} {run['run_id']} {capture['image_filename']}", flush=True)
        records.sort(key=lambda item: item["capture_index"])
        save_json(directory / "records.json", {"run_id": run["run_id"], "batch_id": run["batch_id"],
                  "inertia": run["inertia"],
                  "length_unit": "mm", "transform_convention": "B_T_E: tool0 into UR controller Base",
                  "expected_images": len(run["captures"]), "complete": len(records) == len(run["captures"]),
                  "captures": records})
        if args.limit and rendered >= args.limit:
            break
    save_json(dataset / job["parameters_file"], parameters)
    done = sum(len(v) for v in all_records.values())
    print(f"CAPTURE_SESSION_FINISHED {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
