"""交付样例必须经真实服务接受；无 mock、网络训练或生产存储写入。"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from core.services.feed_depth_service import DEFAULT_SETTINGS, evaluate_feed_depth, load_feed_depth_data
from core.services.position_monitoring_service import PositionMonitoringService, metric_values
from core.services.spindle_monitoring_service import SpindleMonitoringService
from tools.release.prepare_test_data import ROOT, prepare_test_data


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("release_test_data")
    archive_path = prepare_test_data(folder)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert all(Path(name).suffix in {".json", ".png", ".zip", ".csv", ".xlsx", ".txt"}
                   for name in archive.namelist())
        archive.extractall(folder / "unpacked")
    return folder / "unpacked"


def test_robot_observations_have_analytic_metrics_and_persist(inputs, tmp_path):
    source = inputs / "01_机器人"
    service = PositionMonitoringService(root=tmp_path)
    service.load_parameters(source / "parameters.json")
    baseline = service.load_observations(source / "baseline_observations.json")
    assert baseline["source_type"] == "simulation"
    service.create_baseline("软件验证模拟基准")
    service.load_observations(source / "current_observations.json")
    result = service.evaluate()
    rms = 0.4 * np.sqrt(2 / 3)
    assert metric_values(result, "repeatability") == pytest.approx([rms, 0, 0, rms], abs=1e-10)
    assert metric_values(result, "repeatability_change") == pytest.approx([rms / 2, 0, 0, rms / 2], abs=1e-10)
    assert metric_values(result, "absolute_change") == pytest.approx([0.6, 0, 0, 0.6], abs=1e-10)
    restored = PositionMonitoringService(root=tmp_path)
    assert restored.latest_result["batch_id"] == "B002"
    assert len(restored.list_history()) == 2


def test_robot_images_are_detected_without_truth_substitution(inputs, tmp_path):
    source = inputs / "01_机器人"
    service = PositionMonitoringService(root=tmp_path)
    service.load_parameters(source / "parameters.json")
    for batch_id in ("B001", "B002"):
        batch = service.load_observations(source / "images" / batch_id / "record.json")
        assert batch["comparison_status"] == "simulation"
        assert len(batch["samples"]) == 3
        for sample in batch["samples"]:
            assert sample["corner_count"] == 48
            assert sample["reprojection_error_px"] < 0.15
            assert Path(sample["image_path"]).is_file()
        if batch_id == "B001":
            service.create_baseline("软件验证图片基准")
    result = service.evaluate()
    assert metric_values(result, "absolute_change")[3] == pytest.approx(0.6, abs=0.03)


def test_spindle_zips_are_imported_analyzed_and_deduplicated(inputs, tmp_path):
    config = json.loads((ROOT / "config/spindle_monitoring.json").read_text(encoding="utf-8"))
    service = SpindleMonitoringService(tmp_path, config)
    packages = sorted((inputs / "02_主轴").glob("simulated_*.zip"))
    records = service.import_packages(packages)
    assert len(records) == 2
    for index, record in enumerate(records):
        assert record["is_simulated"] and record["manual_label"] == "unconfirmed"
        result = service.latest_result(record["run_id"])
        assert result["score"] is None and result["window_count"] == 10
        amplitude = (0.05, 0.2)[index]
        assert result["rms_mm_s"][0] == pytest.approx(np.sqrt((amplitude**2 + 0.01**2) / 2), rel=1e-5)
        spectrum = result["spectrum"]
        assert spectrum["frequency_hz"][np.argmax(spectrum["amplitude_mm_s"][0])] == pytest.approx(120)
        assert result["telemetry"]["current_a"] == [0.4, 0.6]
    service.import_packages(packages)
    assert len(service.runs) == len(service.history()) == 2
    assert SpindleMonitoringService(tmp_path, config).latest_result(records[0]["run_id"]) is not None


def test_training_bundle_provides_five_distinct_matching_candidates(inputs, tmp_path):
    config = json.loads((ROOT / "config/spindle_monitoring.json").read_text(encoding="utf-8"))
    service = SpindleMonitoringService(tmp_path, config)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    report = service.import_batch([inputs / manifest["spindle"]["training_package"]], label="healthy")
    assert report["new_count"] == 5 and report["failed_count"] == 0
    candidates = service.training_candidates()
    assert len(candidates) == len({record["run_id"] for record in candidates}) == 5
    assert all(record["is_simulated"] and record["training_eligible"] for record in candidates)
    runs = [service._read_run(record) for record in candidates]
    assert all(run["velocity"].shape == (1, 3, 20480) for run in runs)
    assert all(run["calibration_context"] == runs[0]["calibration_context"] for run in runs)
    for index, run in enumerate(runs):
        assert all(not np.array_equal(run["velocity"], other["velocity"]) for other in runs[:index])
    assert service.current_model is None  # 数据生成和格式回归不偷偷训练网络。


@pytest.mark.parametrize("suffix", ["csv", "xlsx"])
@pytest.mark.parametrize(("name", "variance", "alarms"), [("normal", 0.0002, 0), ("out_of_limit", 0.02, 2)])
def test_feed_csv_and_excel_preserve_simulation_missing_values_and_limits(inputs, suffix, name, variance, alarms):
    source = inputs / "03_窝深" / f"simulated_{name}.{suffix}"
    data = load_feed_depth_data(source, DEFAULT_SETTINGS)
    result = evaluate_feed_depth(data, {"error_lower_mm": -0.1, "error_upper_mm": 0.1})
    assert all(row["is_simulated"] for row in result["rows"])
    summary = result["history"][0]
    assert summary["count"] == 5 and summary["mean_depth_mm"] == pytest.approx(1.5)
    assert summary["variance_mm2"] == pytest.approx(variance)
    assert summary["alarm_count"] == alarms
    if name == "out_of_limit":
        assert result["rows"][-1]["actual_depth_mm"] is None
        assert [row["alarm"] for row in result["rows"]] == [True, False, False, False, True, False]
