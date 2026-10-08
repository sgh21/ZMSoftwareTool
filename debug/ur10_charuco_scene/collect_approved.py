"""Reuse the approved 30 poses and zero-inertia images; collect two 20-direction groups."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from collect_images import ROOT, prepare_dataset, save_json


def store_reference(work, preview):
    """Copy the approved reference into a dataset; also migrate an existing dataset."""
    work, preview = Path(work).resolve(), Path(preview).resolve()
    reference = work / "reference"
    if reference != preview:
        parameters = json.loads((preview / "parameters.json").read_text(encoding="utf-8"))
        run_id = parameters["runs"][0]["run_id"]
        records = json.loads((preview / run_id / "records.json").read_text(encoding="utf-8"))
        reference.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preview / "fixed_observation_plan.json", reference / "fixed_observation_plan.json")
        run_directory = reference / run_id
        run_directory.mkdir(exist_ok=True)
        shutil.copy2(preview / run_id / "records.json", run_directory / "records.json")
        for row in records["captures"]:
            image = Path(row["image_filename"])
            (run_directory / image).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview / run_id / image, run_directory / image)
        parameters["scene_snapshot"] = "../scene_snapshot.blend"
        save_json(reference / "parameters.json", parameters)
    save_json(work / "grouped_source.json", {"preview_dir": "reference"})
    return reference


def prepare(dataset, preview, plan_path):
    """Read the frozen targets and directions, and retain the approved reference."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    preview_plan = json.loads((preview / "fixed_observation_plan.json").read_text(encoding="utf-8"))
    preview_parameters = json.loads((preview / "parameters.json").read_text(encoding="utf-8"))
    if preview_plan["point_count"] != 30 or preview_plan["direction_count"] != 1:
        raise ValueError("Expected the approved 30-pose, one-direction preview")
    if plan["point_count"] != 30 or plan["direction_count"] != 20:
        raise ValueError("Expected the approved frozen 30-pose, 20-direction plan")
    reference_points = {p["point_id"]: p for p in preview_plan["points"]}
    for point in plan["points"]:
        reference = reference_points[point["point_id"]]
        if (point["B_T_E_ideal_mm"] != reference["B_T_E_ideal_mm"]
                or point["q_ideal_rad"] != reference["q_ideal_rad"]):
            raise ValueError(f"Target differs from the approved zero-inertia reference: {point['point_id']}")
    dataset.mkdir(parents=True, exist_ok=False)
    snapshot = dataset / "fixed_observation_plan.json"
    shutil.copy2(plan_path, snapshot)
    work = prepare_dataset(SimpleNamespace(
        dataset=dataset / "_work", plan=snapshot, points=30, directions=20,
        inertia=[0.1, 0.2], samples=32, batch_start=1,
        scene_source=(preview / preview_parameters["scene_snapshot"]).resolve(),
    ))
    store_reference(work, preview)
    job_path = work / "capture_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["validation_script"] = "export_grouped.py (use the parent grouped dataset path)"
    save_json(job_path, job)
    print("APPROVED_TARGETS_REUSED 30; BASELINE_IMAGES_REUSED 30; NEW_CAPTURES 1200", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output/approved_30poses_three_inertias")
    parser.add_argument("--preview", type=Path,
                        default=ROOT / "output/approved_30poses_three_inertias/_work/reference")
    parser.add_argument("--plan", type=Path, default=ROOT / "fixed_observation_plan.json")
    parser.add_argument("--blender", type=Path, default=Path("D:/Softwares/Blender/blender.exe"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset, preview = args.dataset.resolve(), args.preview.resolve()
    if not args.resume:
        prepare(dataset, preview, args.plan.resolve())
    if args.prepare_only:
        return
    subprocess.run([
        sys.executable, str(ROOT / "collect_images.py"), "--dataset", str(dataset / "_work"),
        "--resume", "--blender", str(args.blender),
    ], check=True)
    # The public records contain only images and ideal/actual poses. The detailed
    # renderer journals are retained under _work for resuming and verification.
    subprocess.run([
        sys.executable, str(ROOT / "export_grouped.py"), "--dataset", str(dataset),
    ], check=True)


if __name__ == "__main__":
    main()
