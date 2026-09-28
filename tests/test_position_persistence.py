"""日记录、重启恢复、图像托管与真实/调试时间的服务回归。"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.services.position_monitoring_service import PositionMonitoringService, read_document, write_document
from core.services.position_persistence import observation_time


@pytest.fixture
def service(tmp_path):
    result = PositionMonitoringService(root=tmp_path / "application")
    result.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    return result


def observations(tmp_path, batch_id="B001", radius=0.2, simulation=True, image=None, **extra):
    samples = []
    for index, offset in enumerate((-radius, radius), 1):
        end = np.eye(4)
        end[:3, 3] = [100 + offset, 200, 300]
        ideal = end.copy()
        ideal[0, 3] = 100
        sample = {"point_id": "P001", "direction_id": f"D{index:03d}", "sample_id": str(index),
                  "vision_pose": np.linalg.inv(end).tolist(), "ideal_pose": ideal.tolist()}
        if image is not None:
            sample["image_path"] = str(image)
        samples.append(sample)
    document = {
        "batch_id": batch_id, "length_unit": "mm", "sampling_protocol": "multidirectional",
        "program_id": "fixed", "target_id": "board", "samples": samples,
        "comparison_status": "simulation" if simulation else "observed", **extra,
    }
    path = tmp_path / f"{batch_id}.json"
    write_document(path, document)
    return path


def test_baseline_and_two_measurements_form_three_debug_days_and_restore(service, tmp_path):
    first = service.load_observations(observations(tmp_path))
    baseline = service.create_baseline("B001")
    assert service.latest_result["role"] == "baseline"
    assert service.latest_result["summary"]["rp_current"] > 0
    assert service.latest_result["summary"]["rp_change"] == pytest.approx(0)
    assert service.latest_result["summary"]["absolute_ap_change"] == pytest.approx(0)
    for batch_id, radius in (("B002", 0.4), ("B003", 0.4)):
        service.load_observations(observations(tmp_path, batch_id, radius))
        service.evaluate()
    history = service.list_history()
    assert [row["debug_day_index"] for row in history] == [0, 1, 2]
    assert len(list((service.storage / "observations").glob("*.json"))) == 3
    assert not (service.storage / "history").exists()
    for day in range(3):
        document = read_document(service.storage / "debug" / f"day_{day:04d}.json")
        assert len(document["evaluations"]) == 1
        assert document["parameters"][service.parameters["version"]] == service.parameters
    # 基准保存观测路径；不重复复制两份完整观测。
    assert read_document(baseline["path"])["batch_path"] == first["saved_path"]
    assert "batch" not in read_document(baseline["path"])
    restored = PositionMonitoringService(root=service.root)
    assert restored.current_batch["batch_id"] == "B003"
    assert restored.latest_result == service.latest_result
    assert restored.load_evaluation_samples(restored.latest_result)["baseline"]["batch_id"] == "B001"
    service.evaluate()
    assert len(read_document(service.storage / "debug" / "day_0002.json")["evaluations"]) == 2
    assert len(list((service.storage / "observations").glob("*.json"))) == 3


def test_unassessed_import_survives_restart_without_previous_result(service, tmp_path):
    service.load_observations(observations(tmp_path))
    service.create_baseline()
    service.load_observations(observations(tmp_path, "B002", 0.4))
    restored = PositionMonitoringService(root=service.root)
    assert restored.current_batch["batch_id"] == "B002"
    assert restored.latest_result is None
    assert restored.baseline["batch"]["batch_id"] == "B001"
    assert len(restored.list_history()) == 1


def test_real_capture_time_wins_and_repeated_evaluations_append_same_day(service, tmp_path):
    capture = "2026-09-20T10:30:00+08:00"
    service.load_observations(observations(tmp_path, simulation=False, captured_at=capture))
    service.create_baseline()
    result = service.evaluate()
    assert result["observed_at"] == capture
    assert result["time_source"] == "captured_at"
    assert result["debug_day_index"] is None
    date = datetime.fromisoformat(capture).astimezone().date().isoformat()
    daily = read_document(service.storage / "daily" / f"{date}.json")
    assert len(daily["evaluations"]) == 2
    assert len(daily["observations"]) == 1
    assert len(daily["baselines"]) == 1


def test_frame_capture_dates_are_ordered_by_instant_not_timestamp_text():
    batch = {"batch_id": "B004", "comparison_status": "simulation", "imported_at": "2026-10-01T00:00:00Z",
             "samples": [{"captured_at": "2026-09-21T00:30:00+08:00"},
                         {"captured_at": "2026-09-20T20:30:00Z"}]}
    observation_time(batch)
    assert batch["observed_at"] == "2026-09-20T20:30:00+00:00"
    assert batch["debug_day_index"] == 3
    batch["batch_id"] = "B000"
    observation_time(batch)
    assert batch["debug_day_index"] is None
    batch["comparison_status"] = "observed"
    batch["batch_id"] = "B001"
    observation_time(batch)
    assert "debug_day_index" not in batch


def test_missing_capture_time_is_explicit_import_time_fallback(service, tmp_path):
    imported = service.load_observations(observations(tmp_path, simulation=False))
    assert imported["observed_at"] == imported["imported_at"]
    assert imported["time_source"] == "imported_at"


def test_images_are_copied_once_remain_available_without_source_and_reused_on_reevaluation(service, tmp_path):
    image = tmp_path / "capture.png"
    image.write_bytes(b"original captured pixels")
    source = observations(tmp_path, image=image)
    imported = service.load_observations(source)
    managed = Path(imported["samples"][0]["image_path"])
    assert managed.is_relative_to(service.storage / "images")
    assert managed.read_bytes() == image.read_bytes()
    assert managed != image
    assert imported["samples"][0]["source_image_path"] == str(image)
    service.create_baseline()
    service.evaluate()
    imported_again = service.load_observations(source)
    assert imported_again["samples"][0]["image_path"] == str(managed)
    assert len(list((service.storage / "images").rglob("*.png"))) == 1
    image.unlink()
    restored = PositionMonitoringService(root=service.root)
    assert Path(restored.current_batch["samples"][0]["image_path"]).read_bytes() == b"original captured pixels"


def test_old_history_and_embedded_baseline_migrate_without_recomputing(service, tmp_path):
    service.load_observations(observations(tmp_path))
    service.create_baseline()
    result = deepcopy(service.latest_result)
    old_root = tmp_path / "legacy"
    storage = old_root / "storage" / "position_monitoring"
    old_baseline = deepcopy(service.baseline)
    old_baseline.pop("batch_path")
    for name in ("observed_at", "time_source", "debug_day_index"):
        old_baseline["batch"].pop(name, None)
    old_baseline["path"] = str(storage / "baselines" / "old.json")
    write_document(old_baseline["path"], old_baseline)
    old_snapshot = deepcopy(service.current_batch)
    old_snapshot["saved_path"] = "original-import-no-longer-used.json"
    for name in ("observed_at", "time_source", "debug_day_index"):
        old_snapshot.pop(name, None)
    result["current_batch_path"] = str(storage / "observations" / "old.json")
    result["baseline_path"] = old_baseline["path"]
    for name in ("observed_at", "time_source", "debug_day_index"):
        result.pop(name, None)
        result.pop(f"baseline_{name}", None)
    write_document(result["current_batch_path"], old_snapshot)
    write_document(storage / "history" / f"{result['id']}.json", result)
    write_document(storage / "parameters" / "current.json", service.parameters)
    write_document(storage / "state.json", {"baseline_path": old_baseline["path"], "settings": service.settings})
    old_history_path = storage / "history" / f"{result['id']}.json"
    old_bytes = old_history_path.read_bytes()
    restored = PositionMonitoringService(root=old_root)
    completed = restored.latest_result
    assert completed["summary"] == result["summary"]
    assert completed["groups"] == result["groups"]
    assert completed["debug_day_index"] == completed["baseline_debug_day_index"] == 0
    assert completed["time_source"] == completed["baseline_time_source"] == "imported_at"
    assert restored.current_batch["saved_path"] == result["current_batch_path"]
    assert restored.list_history() == [completed]
    assert PositionMonitoringService(root=old_root).latest_result == completed
    assert old_history_path.read_bytes() == old_bytes
    assert restored.evaluate()["time_source"] == "imported_at"


@pytest.mark.parametrize("simulation", [True, False])
def test_old_and_new_history_keep_shared_trend_time_metadata(service, tmp_path, simulation):
    service.load_observations(observations(tmp_path, simulation=simulation, captured_at="2026-09-20T10:00:00+08:00"))
    service.create_baseline()
    service.load_observations(observations(tmp_path, "B002", 0.4, simulation, captured_at="2026-09-21T10:00:00+08:00"))
    old = service.evaluate()
    # 构造升级前的单独 history 记录；真实旧记录没有新的六个时间字段。
    daily_path = Path(service._latest_result_ref["path"])
    daily = read_document(daily_path)
    daily["evaluations"] = [row for row in daily["evaluations"] if row["id"] != old["id"]]
    write_document(daily_path, daily)
    for name in ("observed_at", "time_source", "debug_day_index"):
        old.pop(name, None)
        old.pop(f"baseline_{name}", None)
    old_path = service.storage / "history" / f"{old['id']}.json"
    write_document(old_path, old)
    before = old_path.read_bytes()
    service.load_observations(observations(tmp_path, "B003", 0.4, simulation, captured_at="2026-09-22T10:00:00+08:00"))
    service.evaluate()
    history = service.list_history()
    assert [row["batch_id"] for row in history] == ["B001", "B002", "B003"]
    assert all(row["time_source"] == row["baseline_time_source"] == "captured_at" for row in history)
    assert history[1]["observed_at"] == "2026-09-21T10:00:00+08:00"
    assert [row["debug_day_index"] for row in history] == ([0, 1, 2] if simulation else [None] * 3)
    assert service._store.result({"path": str(old_path), "id": old["id"]}) == history[1]
    assert old_path.read_bytes() == before


def test_history_without_capture_or_import_time_declares_evaluation_fallback(service):
    old = {"id": "legacy", "created_at": "2026-09-20T10:00:00+08:00", "batch_id": "B002",
           "baseline_id": "baseline", "baseline_created_at": "2026-09-19T10:00:00+08:00",
           "comparison_status": "observed", "summary": {"rp_current": 0.2}, "groups": []}
    path = service.storage / "history" / "legacy.json"
    write_document(path, old)
    restored = service.list_history()[0]
    assert restored["observed_at"] == old["created_at"]
    assert restored["time_source"] == "evaluated_at"
    assert restored["baseline_observed_at"] == old["baseline_created_at"]
    assert restored["baseline_time_source"] == "baseline_created_at"
    assert restored["debug_day_index"] is None
    assert read_document(path) == old


def test_logs_survive_restart_and_do_not_rewrite_daily_results(service, tmp_path):
    service.load_observations(observations(tmp_path))
    service.create_baseline()
    daily_path = service.storage / "debug" / "day_0000.json"
    before = daily_path.read_bytes()
    first = service.append_log("观测已保存")
    second = service.append_log("没有设置阈值", "WARN")
    restored = PositionMonitoringService(root=service.root)
    assert restored.list_logs() == [first, second]
    assert daily_path.read_bytes() == before
