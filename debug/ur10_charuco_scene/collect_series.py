"""Prepare and run a frozen batch queue, with progress polling every 30 minutes."""

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from collect_images import ROOT, prepare_dataset, save_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(config_path):
    config = read_json(config_path)
    dataset = (ROOT / config["dataset"]).resolve()
    existing = read_json(dataset / "parameters.json")
    batches = config["batches"]
    names = [row["batch"] for row in batches]
    if len(set(names)) != len(names) or set(names) & {row["batch_id"] for row in existing["runs"]}:
        raise ValueError("New batch identifiers must be unique and not already published")
    for row in batches:
        if not 0 <= row["inertia"] <= 1 or not 0 < row["target_bias"] <= 1:
            raise ValueError("Inertia must be in [0,1] and target bias in (0,1]")
    if config["poll_minutes"] <= 0:
        raise ValueError("Polling interval must be positive")
    queue_dir = dataset / "_work" / f"series_{names[0]}_{names[-1]}"
    queue_dir.mkdir(exist_ok=False)
    queue = dict(config, dataset=str(dataset), created_at_utc=now())
    for row in queue["batches"]:
        source = dataset / "_work" / row["batch"]
        prepare_dataset(SimpleNamespace(
            dataset=source, plan=dataset / config["fixed_plan"], points=30, directions=20,
            inertia=[row["inertia"]], target_bias=row["target_bias"],
            bias_source=dataset / config["bias_source"], bias_seed=20260928,
            samples=config["samples"], batch_start=int(row["batch"][1:]),
            scene_source=dataset / config["scene_source"],
        ))
        row["source"] = source.relative_to(dataset).as_posix()
        row["expected_images"] = 600
        print(f"SERIES_PREPARED {row['batch']}", flush=True)
    queue_path = queue_dir / "queue.json"
    save_json(queue_path, queue)
    print(f"QUEUE_READY {queue_path}", flush=True)
    return queue_path


def run_command(command):
    # Wait for process completion rather than repeatedly polling capture progress.
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8",
                          creationflags=subprocess.CREATE_NO_WINDOW) as process:
        for line in process.stdout:
            if line.strip() != "libpng warning: eXIf: duplicate":
                print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {command}")


def recycle_render_log(source):
    path = source / "render.log"
    if not path.exists():
        return
    # Only this completed batch's own verbose log is removed; keep failed logs.
    env = dict(os.environ, UR10_COMPLETED_RENDER_LOG=str(path.resolve()))
    subprocess.run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Add-Type -AssemblyName Microsoft.VisualBasic; "
        "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
        "$env:UR10_COMPLETED_RENDER_LOG, "
        "[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs, "
        "[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)",
    ], env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


def run(queue_path, blender):
    queue = read_json(queue_path)
    dataset = Path(queue["dataset"])
    directory = queue_path.parent
    state = {"state": "running", "pid": os.getpid(), "started_at_utc": now(),
             "poll_minutes": queue["poll_minutes"], "expected_images": 600 * len(queue["batches"]),
             "batches": [dict(row, state="pending", images=0) for row in queue["batches"]]}
    lock = threading.Lock()
    finished = threading.Event()

    def snapshot(event):
        with lock:
            for row in state["batches"]:
                progress = dataset / row["source"] / "progress.json"
                if progress.exists():
                    row["images"] = read_json(progress)["completed"]
            state["images"] = sum(row["images"] for row in state["batches"])
            state["updated_at_utc"] = now()
            temporary = directory / "status.json.tmp"
            save_json(temporary, state)
            temporary.replace(directory / "status.json")
            entry = {"time": state["updated_at_utc"], "event": event,
                     "state": state["state"], "images": state["images"],
                     "expected_images": state["expected_images"],
                     "batches": [{"batch": row["batch"], "state": row["state"],
                                  "images": row["images"]} for row in state["batches"]]}
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print("SERIES_STATUS " + json.dumps(entry), flush=True)

    def monitor():
        while not finished.wait(queue["poll_minutes"] * 60):
            snapshot("periodic_poll")

    def phase(row, value):
        with lock:
            row["state"] = value
        snapshot("phase_change")

    # OS releases this lock if the runner exits; resuming cannot duplicate a live queue.
    with (directory / "runner.lock").open("a+b") as runner_lock:
        if runner_lock.tell() == 0:
            runner_lock.write(b"0")
            runner_lock.flush()
        runner_lock.seek(0)
        msvcrt.locking(runner_lock.fileno(), msvcrt.LK_NBLCK, 1)
        original_parameters = read_json(dataset / "parameters.json")
        protected = [dataset / "fixed_observation_plan.json"]
        for old in original_parameters["runs"]:
            if old["batch_id"] not in {row["batch"] for row in queue["batches"]}:
                protected.append(dataset / old["record"])
        original_bytes = {path: path.read_bytes() for path in protected}
        snapshot("started")
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            for row in state["batches"]:
                source = dataset / row["source"]
                job = read_json(source / "capture_job.json")
                params = read_json(source / "parameters.json")
                capture = job["runs"][0]
                if (capture["batch_id"] != row["batch"] or capture["inertia"] != row["inertia"]
                        or params["target_bias"]["coefficient"] != row["target_bias"]):
                    raise ValueError("Saved job does not match the frozen series coefficients")
                report_path = dataset / f"validation_{row['batch']}.json"
                if report_path.exists() and read_json(report_path)["passed"]:
                    phase(row, "validated")
                    continue
                phase(row, "rendering")
                run_command([
                    sys.executable, "-B", "-X", "utf8", str(ROOT / "collect_images.py"),
                    "--dataset", str(source), "--resume", "--blender", str(blender),
                ])
                phase(row, "validating")
                run_command([
                    sys.executable, "-B", "-X", "utf8", str(ROOT / "append_batch.py"),
                    "--dataset", str(dataset), "--source", str(source),
                ])
                report = read_json(report_path)
                if not report["passed"] or report["images"] != row["expected_images"]:
                    raise ValueError(f"Validation failed for {row['batch']}")
                recycle_render_log(source)
                phase(row, "validated")
            current = read_json(dataset / "parameters.json")
            for path, content in original_bytes.items():
                if path.read_bytes() != content:
                    raise ValueError(f"An original record or fixed plan changed: {path}")
            for key, value in original_parameters["extrinsics_by_capture"].items():
                if current["extrinsics_by_capture"][key] != value:
                    raise ValueError(f"An original camera extrinsic changed: {key}")
            if any(row not in current["runs"] for row in original_parameters["runs"]):
                raise ValueError("An original batch's parameters changed")
            with lock:
                state["state"] = "complete"
                state["previous_data_preserved"] = True
                state["total_published_images"] = len(current["extrinsics_by_capture"])
        except Exception as exc:
            with lock:
                state["state"] = "failed"
                state["error"] = str(exc)
            raise
        finally:
            finished.set()
            monitor_thread.join()
            snapshot("finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "status"])
    parser.add_argument("--config", type=Path, default=ROOT / "capture_series.json")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--blender", type=Path, default=Path("D:/Softwares/Blender/blender.exe"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config.resolve())
    elif args.queue is None:
        parser.error("run/status requires --queue")
    elif args.action == "run":
        run(args.queue.resolve(), args.blender)
    else:
        print(json.dumps(read_json(args.queue.resolve().parent / "status.json"),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
