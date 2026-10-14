"""用真实 Qt 界面按 1× 流速验证完整实测振动回放。"""

import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import psutil
import pyqtgraph as pg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from main import create_application  # noqa: E402
from app.pages.spindle_rotation_page import SpindleRotationPage  # noqa: E402
from core.services.spindle_monitoring_service import SpindleMonitoringService  # noqa: E402


def percentiles(values):
    return {key: float(np.percentile(values, percentile)) if values else None
            for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    app = create_application()
    process = psutil.Process()
    paint_times = []
    original_paint = pg.PlotWidget.paintEvent

    def measured_paint(widget, event):
        started = perf_counter()
        original_paint(widget, event)
        paint_times.append((perf_counter() - started) * 1000)

    pg.PlotWidget.paintEvent = measured_paint
    metrics = {"interval_ms": [], "update_ms": [], "lag_ms": []}
    memory = []
    waveform_heights = []
    snapshots = []
    fixed_ranges = None
    started_at = None
    next_progress = 0
    next_memory = 0
    snapshot_times = [5, 60, 600, 1200]
    max_points = 0
    y_stable = True
    result = {}
    with TemporaryDirectory(prefix="zm_vibration_validation_") as store:
        page = SpindleRotationPage(SpindleMonitoringService(Path(store)))
        page.ui_scale = 1.0
        page.resize(1600, 1000)
        page.show()
        page.realtime_button.click()
        dialog = page.realtime_dialog
        dialog.resize(1400, 1000)
        frame_file = (report / "frames.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(frame_file, fieldnames=["wall_s", "source_s", "interval_ms", "update_ms", "lag_ms"])
        writer.writeheader()

        def save_snapshot(source_s, name):
            dialog.grab().save(str(report / name))
            occupancy = {}
            for key, wave in dialog.wave_plots.items():
                samples = np.concatenate([y for _, y in wave.buffer.display_segments()])
                lower, upper = wave.plot.viewRange()[1]
                occupancy[key] = float(np.ptp(samples) / (upper - lower)) if samples.size else 0
            snapshots.append({"source_s": source_s, "file": name, "waveform_height_fraction": occupancy})

        def observe():
            nonlocal started_at, next_progress, next_memory, fixed_ranges, max_points, y_stable, result
            frame = dialog.frame_stats
            if not frame:
                return
            if started_at is None:
                started_at = perf_counter() - dialog.replay_elapsed_s()
                fixed_ranges = {key: wave.plot.viewRange()[1] for key, wave in dialog.wave_plots.items()}
                print(json.dumps({"event": "started", "duration_s": dialog.recording.duration_s,
                                  "sample_rate_hz": dialog.recording.sample_rate_hz,
                                  "total_samples": dialog.recording.total_samples}), flush=True)
            wall_s = perf_counter() - started_at
            source_s = frame["source_s"]
            for key in metrics:
                metrics[key].append(frame[key])
            writer.writerow({"wall_s": wall_s, **frame})
            for key, wave in dialog.wave_plots.items():
                max_points = max(max_points, *(buffer.count for buffer in wave.buffers.values()))
                y_stable &= wave.plot.viewRange()[1] == fixed_ranges[key]
            if source_s >= next_memory:
                memory.append({"source_s": source_s, "rss_mb": process.memory_info().rss / 1024 ** 2})
                height = {"source_s": source_s}
                for key, wave in dialog.wave_plots.items():
                    values = np.concatenate([y for _, y in wave.buffer.display_segments()]) if wave.buffer.count else np.array([])
                    limits = wave.plot.viewRange()[1]
                    height[key] = float(np.ptp(values) / (limits[1] - limits[0])) if values.size else 0
                waveform_heights.append(height)
                next_memory += 10
            if snapshot_times and source_s >= snapshot_times[0]:
                moment = snapshot_times.pop(0)
                save_snapshot(source_s, f"waveform_{moment:04d}s.png")
            if source_s >= next_progress:
                progress = {"event": "progress", "source_s": source_s, "wall_s": wall_s,
                            "samples": dialog.replay.samples_consumed, "rss_mb": memory[-1]["rss_mb"],
                            "frame_interval_ms": percentiles(metrics["interval_ms"]), "lag_ms": frame["lag_ms"]}
                (report / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                print(json.dumps(progress), flush=True)
                frame_file.flush()
                next_progress += 60
            requested_s = args.duration_seconds or dialog.recording.duration_s
            if dialog.replay.finished or source_s >= min(requested_s, dialog.recording.duration_s):
                save_snapshot(source_s, "waveform_final.png")
                complete = dialog.replay.finished and dialog.replay.samples_consumed == dialog.recording.total_samples
                rss_baseline = next((row["rss_mb"] for row in memory if row["source_s"] >= 60), memory[0]["rss_mb"])
                result = {"complete": complete, "at_least_20_minutes": complete and source_s >= 1200,
                          "recording": str(args.run_dir.resolve()), "duration_s": dialog.recording.duration_s,
                          "source_s": source_s, "wall_s": wall_s, "speed": "1x",
                          "samples_consumed": dialog.replay.samples_consumed, "total_samples": dialog.recording.total_samples,
                          "sample_rate_hz": dialog.recording.sample_rate_hz,
                          "display_rates_hz": dialog.recording.display_rates, "display_passbands_hz": dialog.recording.passbands,
                          "frame_interval_ms": percentiles(metrics["interval_ms"]),
                          "ui_data_update_ms": percentiles(metrics["update_ms"]),
                          "qt_paint_ms": percentiles(paint_times), "replay_lag_ms": percentiles(metrics["lag_ms"]),
                          "frame_intervals_over_250ms": sum(value > 250 for value in metrics["interval_ms"]),
                          "frames": len(metrics["interval_ms"]), "max_curve_points": max_points,
                          "y_axis_fixed": y_stable, "fixed_y_ranges": fixed_ranges,
                          "rss_baseline_after_warmup_mb": rss_baseline, "rss_final_mb": memory[-1]["rss_mb"],
                          "rss_peak_mb": max(row["rss_mb"] for row in memory),
                          "rss_growth_after_warmup_mb": memory[-1]["rss_mb"] - rss_baseline,
                          "memory_samples": memory, "screenshots": snapshots,
                          "waveform_height_samples": waveform_heights,
                          "waveform_height_median": {key: float(np.median([sample[key] for sample in waveform_heights
                                                                          if sample["source_s"] >= 60]))
                                                       for key in dialog.wave_plots} if source_s >= 60 else {}}
                result["checks"] = {"all_samples_consumed": complete, "fixed_y_axis": y_stable,
                                    "bounded_plot_buffers": max_points <= 4096,
                                    "frame_p99_under_100ms": result["frame_interval_ms"]["p99"] < 100,
                                    "no_frame_gap_over_250ms": not result["frame_intervals_over_250ms"],
                                    "rss_growth_under_64mb": result["rss_growth_after_warmup_mb"] < 64}
                (report / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps({"event": "finished", "complete": complete, "source_s": source_s,
                                  "checks": result["checks"]}), flush=True)
                dialog.shutdown()
                dialog.close()
                page.close()
                app.quit()

        dialog.replay_timer.timeout.connect(observe)
        dialog.load_recording(args.run_dir)
        app.exec()
        dialog.shutdown()
        frame_file.close()
    pg.PlotWidget.paintEvent = original_paint
    return 0 if result and all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
