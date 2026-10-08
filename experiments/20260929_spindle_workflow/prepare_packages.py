"""Prepare replay ZIPs, optionally one raw acquisition per sample package.

The legacy daily mode reuses July HDF5 files. Single-run mode converts all
selected dates from raw acceleration with the application algorithm.
No pretrained weights are read or included in a package.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from core.services.position_persistence import read_json  # noqa: E402


def load_helpers(research):
    path = research / "src/build_reconstruction_dataset.py"
    spec = importlib.util.spec_from_file_location("spindle_build_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_august_run(run, destination, helpers):
    from core.algorithms.spindle_monitoring import acceleration_to_velocity

    if destination.exists():
        return destination
    telemetry = pd.read_csv(run / "spindle_telemetry.csv")
    good = (
        (telemetry.target_rpm == 7000)
        & telemetry.speed_ok.astype(str).str.lower().eq("true")
        & ((telemetry.actual_speed_rpm - 7000).abs() <= 10)
    )
    platform_start = float(telemetry.loc[good, "time_s"].min())
    platform_end = float(telemetry.loc[good, "time_s"].max())
    records = pd.read_csv(run / "segment_records.csv")
    records = records[
        records.signal_type.eq("acceleration")
        & records.partial.astype(str).str.lower().eq("false")
        & records.sample_count.eq(256000)
        & (records.time_start_s >= platform_start + 600)
        & (records.time_end_s <= platform_end)
    ].sort_values("time_start_s")
    selected = []
    for row in records.itertuples():
        inside = (telemetry.time_s >= row.time_start_s) & (telemetry.time_s < row.time_end_s)
        if inside.any() and float(good[inside].mean()) >= 0.95:
            selected.append(row)
    if not selected:
        raise ValueError(f"No compatible steady windows: {run.name}")
    temperatures = [helpers.load_temperature_series(run, folder) for folder in helpers.TEMPERATURE_SOURCE_FOLDERS]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as h5:
        h5.attrs.update({
            "schema_version": "1.0", "run_id": run.name, "run_name": run.name,
            "run_date": run.parent.name, "target_speed_rpm": 7000.0,
            "platform_start_s": platform_start, "stable_wait_s": 600.0,
            "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
            "velocity_channels": np.array(["ACC1", "ACC2", "ACC3"], dtype=h5py.string_dtype()),
            "velocity_frequency_band_hz": np.array([10., 900.]),
            "preprocessing_id": "velocity_2048_10_900_v1",
        })
        windows = helpers.initialize_window_group(h5, "windows_10s", len(selected), 10)
        for index, row in enumerate(selected):
            source = run / "acceleration_25600Hz_256000samples" / row.data_file
            raw = helpers.load_npz_xz(source)
            velocity = acceleration_to_velocity(raw["data"])
            target_times = row.time_start_s + np.arange(100) / 10
            values, masks = [], []
            for times, samples, _ in temperatures:
                value, mask = helpers.align_temperature(times, samples, target_times)
                values.append(value)
                masks.append(mask)
            windows["velocity"][index] = velocity
            windows["temperature"][index] = values
            windows["temperature_valid"][index] = masks
            windows["sample_valid"][index] = bool(np.isfinite(velocity).all() and np.all(masks))
            windows["start_time_s"][index] = row.time_start_s
            windows["end_time_s"][index] = row.time_end_s
            windows["source_segment_index"][index] = row.segment_index
        h5.attrs["window_10s_count"] = len(selected)
    return destination


def condition_name(run):
    for name in ("unbalance1", "unbalance2", "unbalance3", "unbalance4", "balance"):
        if run.name.endswith("_" + name):
            return name
    return "normal"


def write_package(day, pairs, output, preprocessing):
    manifest = {
        "schema_version": 1, "package_id": "spindle_" + day,
        "captured_date": f"{day[:4]}-{day[4:6]}-{day[6:]}",
        "source_type": "historical_replay", "preprocessing": preprocessing, "runs": [],
    }
    path = output / f"spindle_{day}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as package:
        for run, h5_path in pairs:
            source = read_json(run / "manifest.json")
            folder = "runs/" + run.name
            entry = {
                "run_id": run.name, "captured_at": source["created_at_local"],
                "speed_rpm": 7000, "operation": "idle",
                "condition": {"tool_remounted": "reset" in run.name,
                              "seal_pressure_mpa": 0.2 if "0.2Mpa" in run.name else None},
                "data_file": folder + "/signals.h5",
                "telemetry_file": folder + "/spindle_telemetry.csv",
                "source_metadata_file": folder + "/manifest.json",
                "experiment_condition": condition_name(run),
            }
            manifest["runs"].append(entry)
            package.write(h5_path, entry["data_file"])
            for filename in ("spindle_telemetry.csv", "manifest.json", "sensor_info.csv", "experiment_record.csv", "segment_records.csv"):
                package.write(run / filename, folder + "/" + filename)
        package.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"PACKAGE {day}: {len(pairs)} runs, {path.stat().st_size / 1024 ** 2:.1f} MiB", flush=True)
    return {"date": day, "path": str(path), "run_count": len(pairs)}


def select_raw_segments(run, telemetry):
    """Keep complete 10 s records from the second half of the 7000 rpm hold."""
    good = (
        telemetry.target_rpm.eq(7000)
        & telemetry.speed_ok.astype(str).str.lower().isin(("true", "1"))
        & telemetry.actual_speed_rpm.sub(7000).abs().le(10)
    )
    if not good.any():
        raise ValueError(f"No 7000 rpm platform: {run.name}")
    platform_start = float(telemetry.loc[good, "time_s"].min())
    platform_end = float(telemetry.loc[good, "time_s"].max())
    cutoff = platform_start + 600
    evaluation_end = min(platform_end, cutoff + 600)
    records = pd.read_csv(run / "segment_records.csv")
    acceleration = records[records.signal_type.eq("acceleration")].sort_values("time_start_s")
    selected, excluded = [], []
    for row in acceleration.itertuples():
        item = {"segment_index": int(row.segment_index), "data_file": row.data_file,
                "start_time_s": float(row.time_start_s), "end_time_s": float(row.time_end_s)}
        reason = None
        if (str(row.partial).lower() not in ("false", "0") or row.sample_count != 256000
                or not np.isclose(row.time_end_s - row.time_start_s, 10, atol=1e-8)):
            reason = "incomplete_10s"
        elif row.time_start_s < cutoff:
            reason = "ramp_or_first_600s"
        elif row.time_end_s > evaluation_end:
            reason = "after_evaluation_interval"
        else:
            inside = telemetry.time_s.ge(row.time_start_s) & telemetry.time_s.lt(row.time_end_s)
            fraction = float(good[inside].mean()) if inside.any() else 0.0
            item["steady_telemetry_fraction"] = fraction
            if fraction < 0.95:
                reason = "steady_telemetry_below_95_percent"
        if reason:
            item["reason"] = reason
            excluded.append(item)
        else:
            item["minute_in_evaluation"] = int((row.time_start_s - cutoff) // 60) + 1
            selected.append(item)
    return selected, excluded, {
        "platform_start_s": platform_start, "platform_end_s": platform_end,
        "stable_wait_s": 600.0, "evaluation_start_s": cutoff,
        "evaluation_end_s": evaluation_end, "rpm_tolerance": 10.0,
        "minimum_steady_telemetry_fraction": 0.95,
        "source_acceleration_segment_count": len(acceleration),
    }


def convert_raw_sample(run, source, h5_path, helpers):
    from core.algorithms.spindle_monitoring import acceleration_to_velocity

    telemetry = pd.read_csv(run / "spindle_telemetry.csv")
    selected, excluded, rule = select_raw_segments(run, telemetry)
    groups = source["configuration"]["groups"]
    acceleration = next(group for group in groups if group["signal_type"] == "acceleration")
    channel_names = [channel["physical_name"] for channel in acceleration["channels"]]
    sensor_ids = [channel["sensor"]["sensor_id"] for channel in acceleration["channels"]]
    order = [sensor_ids.index(name) for name in ("ACC1", "ACC2", "ACC3")]
    temperatures = [helpers.load_temperature_series(run, folder)
                    for folder in helpers.TEMPERATURE_SOURCE_FOLDERS]

    def convert(item):
        path = run / "acceleration_25600Hz_256000samples" / item["data_file"]
        raw = helpers.load_npz_xz(path)
        raw_channels = np.asarray(raw["channels"]).astype(str).tolist()
        if (raw_channels != channel_names or float(raw["sample_rate_hz"].reshape(-1)[0]) != 25600
                or str(raw["unit"].reshape(-1)[0]) != "g"
                or str(raw["signal_type"].reshape(-1)[0]) != "acceleration"):
            raise ValueError(f"Raw acceleration metadata mismatch: {path}")
        expected = item["start_time_s"] + np.arange(256000) / 25600
        if not np.allclose(raw["time_s"], expected, rtol=0, atol=1e-8):
            raise ValueError(f"Raw acceleration time axis mismatch: {path}")
        velocity = acceleration_to_velocity(np.asarray(raw["data"])[order])
        target_time = item["start_time_s"] + np.arange(100) / 10
        values, masks = zip(*(helpers.align_temperature(times, samples, target_time)
                              for times, samples, _ in temperatures))
        return item, velocity, np.asarray(values), np.asarray(masks)

    if not selected:
        raise ValueError(f"No complete stable segments: {run.name}")
    # Workers only read/decompress/convert; HDF5 writes stay in the calling thread.
    with ThreadPoolExecutor(max_workers=4) as workers:
        converted = list(workers.map(convert, selected))
    with h5py.File(h5_path, "w") as h5:
        h5.attrs.update({
            "schema_version": "1.0", "run_id": run.name, "run_name": run.name,
            "source_run_id": source["run_id"], "run_date": run.parent.name,
            "target_speed_rpm": 7000.0, "captured_at": source["created_at_local"],
            "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
            "velocity_channels": np.array(["ACC1", "ACC2", "ACC3"], dtype=h5py.string_dtype()),
            "velocity_frequency_band_hz": np.array([10., 900.]),
            "temperature_channels": np.array(["NTC1", "RTD1"], dtype=h5py.string_dtype()),
            "preprocessing_id": "velocity_2048_10_900_v1",
            "platform_start_s": rule["platform_start_s"], "stable_wait_s": 600.0,
            "evaluation_start_s": rule["evaluation_start_s"],
            "evaluation_end_s": rule["evaluation_end_s"],
            "window_10s_count": len(converted), "window_1s_count": len(converted) * 10,
            "source_velocity_dataset": "raw_acceleration_25600Hz_256000samples",
            "missing_temperature_policy": "NaN with False mask; retain valid vibration",
        })
        windows = helpers.initialize_window_group(h5, "windows_10s", len(converted), 10)
        windows["sample_valid"].attrs["meaning"] = "True means vibration is finite and steady telemetry passes"
        for index, (item, velocity, temperature, mask) in enumerate(converted):
            windows["velocity"][index] = velocity
            windows["temperature"][index] = temperature
            windows["temperature_valid"][index] = mask
            windows["sample_valid"][index] = True
            windows["start_time_s"][index] = item["start_time_s"]
            windows["end_time_s"][index] = item["end_time_s"]
            windows["source_segment_index"][index] = item["segment_index"]
        windows["temperature"].attrs["channel_order"] = np.array(["NTC1", "RTD1"], dtype=h5py.string_dtype())
    temperature_masks = np.asarray([item[3] for item in converted])
    report = {
        "run_id": run.name, "source_run_id": source["run_id"], "raw_directory": str(run.resolve()),
        "captured_at": source["created_at_local"], "selection": rule,
        "window_10s_count": len(converted), "window_1s_count": len(converted) * 10,
        "effective_duration_s": len(converted) * 10,
        "selected_segments": selected, "excluded_segments": excluded,
        "temperature_valid_fraction": {
            name: float(temperature_masks[:, index].mean())
            for index, name in enumerate(("NTC1", "RTD1"))
        },
        "minute_groups_are_provenance_only": True,
        "score_aggregation": "P95 over all retained non-overlapping 1s windows; no minute averaging",
        "validation": "pending",
    }
    return report


def prepare_single_run(run, output, preprocessing, helpers):
    from core.algorithms.spindle_monitoring import read_run
    from core.services.spindle_monitoring_service import SpindleMonitoringService

    source = read_json(run / "manifest.json")
    day = run.parent.name
    destination = output / day / (run.name + ".zip")
    if destination.exists():
        raise FileExistsError(f"Sample package already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="spindle_sample_") as temporary:
        h5_path = Path(temporary) / "signals.h5"
        report = convert_raw_sample(run, source, h5_path, helpers)
        data = read_run(h5_path, run / "spindle_telemetry.csv")
        if len(data["velocity"]) != report["window_10s_count"] or not data["sample_valid"].all():
            raise ValueError(f"Sample quality verification failed: {run.name}")
        report["validation"] = "application_read_run_passed"
        folder = "runs/" + run.name
        entry = {
            "run_id": run.name, "source_run_id": source["run_id"], "run_name": run.name,
            "captured_at": source["created_at_local"], "speed_rpm": 7000, "operation": "idle",
            "condition": {"tool_remounted": True if "reset" in run.name else None,
                          "seal_pressure_mpa": 0.2 if "0.2Mpa" in run.name else None},
            "experiment_condition": condition_name(run),
            "source_group": "normal" if day.startswith("202607") else "daily_test",
            "sample_role": ("modeling" if day in ("20260723", "20260724", "20260725")
                            else "normal_daily" if day == "20260726" else "daily_condition_test"),
            "expected_source_health": "normal" if day.startswith("202607") else condition_name(run),
            "data_file": folder + "/signals.h5",
            "telemetry_file": folder + "/spindle_telemetry.csv",
            "source_metadata_file": folder + "/source_manifest.json",
            "quality_report_file": folder + "/quality_report.json",
        }
        manifest = {
            "schema_version": 1, "package_id": "spindle_sample_" + run.name,
            "captured_date": f"{day[:4]}-{day[4:6]}-{day[6:]}",
            "source_type": "historical_replay", "preprocessing": preprocessing, "runs": [entry],
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as package:
            package.write(h5_path, entry["data_file"])
            package.write(run / "spindle_telemetry.csv", entry["telemetry_file"])
            package.write(run / "manifest.json", entry["source_metadata_file"])
            for filename in ("sensor_info.csv", "experiment_record.csv", "segment_records.csv"):
                package.write(run / filename, folder + "/" + filename)
            package.writestr(entry["quality_report_file"], json.dumps(report, ensure_ascii=False, indent=2))
            package.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(destination) as package:
            # Protocol validation reads only config and archive metadata; no application storage is created.
            validator = SpindleMonitoringService.__new__(SpindleMonitoringService)
            validator.config = {"preprocessing": preprocessing, "target_speed_rpm": 7000}
            validator._validate_package(package)
            if package.testzip() is not None:
                raise ValueError(f"ZIP CRC verification failed: {destination}")
    row = {
        "date": day, "run_id": run.name, "source_run_id": source["run_id"],
        "condition": entry["experiment_condition"], "source_group": entry["source_group"],
        "role": entry["sample_role"], "expected_source_health": entry["expected_source_health"],
        "zip_file": str(destination.resolve()), "window_10s_count": report["window_10s_count"],
        "window_1s_count": report["window_1s_count"], "effective_duration_s": report["effective_duration_s"],
        "ntc_valid_fraction": report["temperature_valid_fraction"]["NTC1"],
        "rtd_valid_fraction": report["temperature_valid_fraction"]["RTD1"],
        "size_mib": round(destination.stat().st_size / 1024 ** 2, 3),
        "validation": "H5_read_and_ZIP_protocol_and_CRC_passed",
    }
    print(f"SAMPLE {day}: {run.name}, {row['window_1s_count']} windows, {row['size_mib']:.1f} MiB", flush=True)
    return row, report


def prepare_samples(args, settings, helpers):
    summaries, reports, excluded_runs = [], [], []
    for day in args.dates:
        for run in sorted((args.raw / day).glob("run_*")):
            source = read_json(run / "manifest.json")
            flow = source.get("standard_acquisition_flow", {}).get("configuration", {})
            if flow.get("max_rpm") != 7000 or flow.get("max_hold_s") != 1200:
                excluded_runs.append({"date": day, "run_id": run.name,
                                      "reason": "not_7000rpm_1200s_fixed_hold"})
                continue
            print("CONVERT RAW " + run.name, flush=True)
            row, report = prepare_single_run(run, args.output, settings["preprocessing"], helpers)
            summaries.append(row)
            reports.append(report)
            with (args.output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(summaries)
    quality = {"samples": reports, "excluded_runs": excluded_runs,
               "total_packages": len(summaries),
               "total_window_1s_count": sum(row["window_1s_count"] for row in summaries)}
    (args.output / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE: {len(summaries)} single-run packages; output={args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=PROJECT.parent / "Self-SupervisedReconstruction")
    parser.add_argument("--august", type=Path, default=PROJECT / "data/spindle_error/20260823")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-run", action="store_true", help="Convert raw data and package each acquisition separately")
    parser.add_argument("--raw", type=Path, default=PROJECT / "data/spindle_error")
    parser.add_argument("--dates", nargs="+", default=["20260723", "20260724", "20260725", "20260726", "20260823"])
    args = parser.parse_args()
    if args.output is None:
        folder = "spindle_sample_packages" if args.single_run else "spindle_daily_packages"
        args.output = PROJECT / "data/processed" / folder
    args.output.mkdir(parents=True, exist_ok=True)
    settings = read_json(PROJECT / "config/spindle_monitoring.json")
    helpers = load_helpers(args.research)
    if args.single_run:
        prepare_samples(args, settings, helpers)
        return
    summary = []
    organized = args.research / "data/organized_data/reconstruction_velocity_temperature/7000rpm"
    for day in ("20260722", "20260723", "20260724", "20260725", "20260726"):
        pairs = [(args.research / "data/runs" / day / h5.stem, h5) for h5 in sorted((organized / day).glob("*.h5"))]
        summary.append(write_package(day, pairs, args.output, settings["preprocessing"]))
    pairs = []
    for run in sorted(args.august.glob("run_*")):
        print("CONVERT " + run.name, flush=True)
        h5 = prepare_august_run(run, args.output / "converted_august" / (run.name + ".h5"), helpers)
        pairs.append((run, h5))
    summary.append(write_package("20260823", pairs, args.output, settings["preprocessing"]))
    with (args.output / "packages.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["date", "path", "run_count"])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
