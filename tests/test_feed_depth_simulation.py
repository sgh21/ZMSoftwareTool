"""模拟数据须可复现、可回算，且与实测数据明确分离。"""

import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from statistics import fmean, pstdev, pvariance

import pytest

from core.services.feed_depth_simulation import generate_feed_depth_simulation


def test_exported_holes_reproduce_history_and_degradation(tmp_path):
    progress = []
    result = generate_feed_depth_simulation(
        tmp_path / "simulation", lambda value, message: progress.append(value),
    )
    with Path(result["csv_path"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with Path(result["summary_path"]).open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(rows) == 2700
    assert len(summary) == len(result["history"]) == 90
    assert len(list((Path(result["output_dir"]) / "batches").glob("*.csv"))) == 90
    assert rows[0]["measured_at"] == "2026-07-11T09:00:00+08:00"
    assert rows[-1]["measured_at"] == "2026-10-08T09:09:40+08:00"
    for batch_index, item in enumerate(result["history"]):
        batch = rows[30 * batch_index:30 * (batch_index + 1)]
        depths = [float(row["actual_depth_mm"]) for row in batch]
        assert item["mean_depth_mm"] == pytest.approx(fmean(depths))
        assert item["bias_mm"] == pytest.approx(fmean(depths) - 1.5)
        assert item["variance_mm2"] == pytest.approx(pvariance(depths))
        assert item["stddev_mm"] == pytest.approx(pstdev(depths))
        assert float(summary[batch_index]["variance_mm2"]) == item["variance_mm2"]
        assert item["measured_at"] == batch[-1]["measured_at"]
        assert {row["batch_id"] for row in batch} == {item["batch_id"]}
        assert {row["hole_id"] for row in batch} == {f"{hole:03d}" for hole in range(1, 31)}
        assert all(row["data_source"] == "simulation" and row["is_simulated"] == "True" for row in batch)
        assert all(len(row["actual_depth_mm"].split(".")[1]) == 3 for row in batch)
        batch_path = Path(result["output_dir"]) / "batches" / f"{item['batch_id']}.csv"
        with batch_path.open(encoding="utf-8-sig", newline="") as stream:
            assert list(csv.DictReader(stream)) == batch

    history = result["history"]
    early_bias = fmean(item["bias_mm"] for item in history[:15])
    late_bias = fmean(item["bias_mm"] for item in history[-10:])
    early_sigma = fmean(item["stddev_mm"] for item in history[:15])
    late_sigma = fmean(item["stddev_mm"] for item in history[-10:])
    assert -0.008 < early_bias < 0.002
    assert late_bias < -0.055
    assert late_sigma > 2 * early_sigma
    assert any(b["bias_mm"] > a["bias_mm"] for a, b in zip(history, history[1:]))
    assert any(b["stddev_mm"] < a["stddev_mm"] for a, b in zip(history, history[1:]))
    assert progress == sorted(progress)
    assert progress[-1] == 99


def test_simulation_reproducible_without_global_random_state(tmp_path):
    random_state = random.getstate()
    first = generate_feed_depth_simulation(tmp_path / "first", days=3, holes_per_day=5, seed=21)
    second = generate_feed_depth_simulation(tmp_path / "second", days=3, holes_per_day=5, seed=21)
    assert first["rows"] == second["rows"]
    assert first["history"] == second["history"]
    for first_path in Path(first["output_dir"]).rglob("*"):
        if first_path.is_file():
            relative_path = first_path.relative_to(first["output_dir"])
            assert first_path.read_bytes() == (Path(second["output_dir"]) / relative_path).read_bytes()
    changed = generate_feed_depth_simulation(tmp_path / "changed", days=3, holes_per_day=5, seed=22)
    assert first["rows"] != changed["rows"]
    assert random.getstate() == random_state


def test_one_hole_and_leap_date_are_preserved(tmp_path):
    result = generate_feed_depth_simulation(
        tmp_path / "single", start_date="2024-02-29", days=2, holes_per_day=1,
    )
    assert result["history"][0]["variance_mm2"] == 0
    assert result["history"][0]["stddev_mm"] == 0
    measured_times = [datetime.fromisoformat(row["measured_at"]) for row in result["rows"]]
    assert measured_times[1] - measured_times[0] == timedelta(days=1)
    assert measured_times[0].utcoffset() == timedelta(hours=8)
    assert measured_times[1].date().isoformat() == "2024-03-01"
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["is_simulated"] is True
    assert metadata["resolution_mm"] == 0.001
    assert metadata["start_date"] == "2024-02-29"
    assert metadata["end_date"] == "2024-03-01"


@pytest.mark.parametrize("parameters", [
    {"days": 0}, {"days": 1.5}, {"holes_per_day": 0}, {"start_date": "2026-02-29"},
])
def test_invalid_input_does_not_create_output(tmp_path, parameters):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        generate_feed_depth_simulation(destination, **parameters)
    assert not destination.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    original = destination / "previous.csv"
    original.write_text("original measurement\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="新目录"):
        generate_feed_depth_simulation(destination)
    assert original.read_text(encoding="utf-8") == "original measurement\n"
    assert list(destination.iterdir()) == [original]
