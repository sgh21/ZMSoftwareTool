"""Pack raw robot CSV and laser-tracker TXT files into calibration pkl data."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROBOT_JOINT_COLUMNS = ("end_j1", "end_j2", "end_j3", "end_j4", "end_j5", "end_j6")
COMMAND_POSE_COLUMNS = ("cmd_x", "cmd_y", "cmd_z", "cmd_rx", "cmd_ry", "cmd_rz")
END_POSE_COLUMNS = ("end_x", "end_y", "end_z", "end_rx", "end_ry", "end_rz")


def pack_raw_calibration_pair(
    csv_path: str | Path,
    txt_path: str | Path,
    output_path: str | Path,
    *,
    payload_kg: float = 0.0,
    speed_mode: str = "raw",
) -> Path:
    """Pack one robot-pose CSV and one laser-tracker TXT file into a pkl dataset.

    The parser follows the previous data-packing convention: CSV rows and TXT
    points are paired by row order. Laser tracker coordinates are stored in
    meters because the calibration pipeline consumes meter-scale positions.
    """
    csv_file = Path(csv_path).resolve()
    txt_file = Path(txt_path).resolve()
    output_file = Path(output_path).resolve()

    csv_data = load_robot_pose_csv(csv_file)
    laser_points = load_laser_tracker_txt(txt_file, output_unit="m")
    sample_count = int(len(csv_data["joints"]))
    if sample_count != len(laser_points):
        raise ValueError(
            "CSV/TXT sample counts must match: "
            f"{csv_file.name} has {sample_count}, {txt_file.name} has {len(laser_points)}."
        )

    payloads = np.full(sample_count, float(payload_kg), dtype=np.float32)
    speed_modes = np.asarray([speed_mode] * sample_count)
    dataset: dict[str, Any] = {
        "meta": {
            "source_format": "raw_csv_txt",
            "robot_csv": str(csv_file),
            "laser_txt": str(txt_file),
            "laser_input_unit": "mm",
            "position_unit": "m",
            "payload_kg": float(payload_kg),
            "speed_mode": speed_mode,
            "n_points": sample_count,
        },
        "indices": csv_data["indices"].astype(int),
        "joints": csv_data["joints"].astype(np.float32),
        "cmd_pose": csv_data["cmd_pose"].astype(np.float32),
        "end_pose": csv_data["end_pose"].astype(np.float32),
        "laser_points": laser_points.astype(np.float32),
        "measured_positions": laser_points.astype(np.float32),
        "payload": payloads,
        "payloads": payloads,
        "speed_mode": speed_modes,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(dataset, file)
    return output_file


def load_robot_pose_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load stable-point robot pose CSV data produced by the collection tool."""
    csv_file = Path(path)
    rows = _read_robot_csv(csv_file)
    _require_columns(rows[0], ROBOT_JOINT_COLUMNS + COMMAND_POSE_COLUMNS + END_POSE_COLUMNS)
    if "index" in rows[0]:
        indices = np.asarray([int(float(row["index"])) for row in rows], dtype=int)
    else:
        indices = np.arange(len(rows), dtype=int)
    return {
        "indices": indices,
        "joints": _columns_to_array(rows, ROBOT_JOINT_COLUMNS),
        "cmd_pose": _columns_to_array(rows, COMMAND_POSE_COLUMNS),
        "end_pose": _columns_to_array(rows, END_POSE_COLUMNS),
    }


def load_laser_tracker_txt(path: str | Path, *, output_unit: str = "m") -> np.ndarray:
    """Load laser tracker point TXT data with rows like ``index x y z`` in mm."""
    txt_file = Path(path)
    points: list[list[float]] = []
    with txt_file.open("r", encoding="utf-8-sig", errors="ignore") as file:
        for line in file:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    point_array = np.asarray(points, dtype=float).reshape(-1, 3)
    if output_unit.lower() in {"m", "meter", "meters"}:
        return point_array / 1000.0
    if output_unit.lower() in {"mm", "millimeter", "millimeters"}:
        return point_array
    raise ValueError(f"Unsupported laser tracker output unit: {output_unit!r}")


def default_packed_dataset_path(
    project_root: str | Path,
    csv_path: str | Path,
    txt_path: str | Path,
) -> Path:
    """Return the default processed pkl path for a raw CSV/TXT pair."""
    csv_stem = Path(csv_path).stem
    txt_stem = Path(txt_path).stem
    filename = f"{csv_stem}__{txt_stem}.pkl"
    return Path(project_root).resolve() / "data" / "processed" / "calibration" / filename


def split_calibration_input_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split input files into processed pkl, raw CSV, and raw TXT groups."""
    pkl_paths: list[Path] = []
    csv_paths: list[Path] = []
    txt_paths: list[Path] = []
    for path in paths:
        item = Path(path).resolve()
        suffix = item.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            pkl_paths.append(item)
        elif suffix == ".csv":
            csv_paths.append(item)
        elif suffix == ".txt":
            txt_paths.append(item)
        else:
            raise ValueError(f"Unsupported calibration data file type: {item}")
    return pkl_paths, csv_paths, txt_paths


def _read_robot_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    for start in (1, 0):
        rows = list(csv.DictReader(lines[start:]))
        if rows and all(column in rows[0] for column in ROBOT_JOINT_COLUMNS):
            return rows
    raise ValueError(f"Robot pose CSV is missing required columns: {path}")


def _require_columns(row: dict[str, str], columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Robot pose CSV is missing required columns: {missing}")


def _columns_to_array(rows: list[dict[str, str]], columns: Iterable[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)
