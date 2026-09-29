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


def test_unassessed_import_survives_restart_separate_from_latest_evaluated_result(service, tmp_path):
    service.load_observations(observations(tmp_path))
    service.create_baseline()
    service.load_observations(observations(tmp_path, "B002", 0.4))
    restored = PositionMonitoringService(root=service.root)
    assert restored.current_batch["batch_id"] == "B002"
    assert restored.latest_batch["batch_id"] == restored.latest_result["batch_id"] == "B001"
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
    result = deepcopy(service.list_history()[0])
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
    assert restored.list_history()[0]["summary"] == completed["summary"]
    assert restored.list_history()[0]["created_at"] == completed["created_at"]
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


def three_evaluated_batches(service, tmp_path):
    service.load_observations(observations(tmp_path, "B001", 0.2))
    first = service.create_baseline("first")
    service.load_observations(observations(tmp_path, "B002", 0.4))
    service.evaluate()
    service.load_observations(observations(tmp_path, "B003", 0.6))
    service.evaluate()
    return first


def test_switch_baseline_recomputes_latest_and_all_observations_without_writing_history(service, tmp_path, monkeypatch):
    first = three_evaluated_batches(service, tmp_path)
    third = service.create_baseline("third")
    service.evaluate()  # 同一次观测重复评估，不应成为第二个趋势点。
    archived = service.list_history()
    protected = {path: path.read_bytes() for path in service.storage.rglob("*.json")
                 if path.name != "state.json"}

    def no_pnp(*args, **kwargs):
        raise AssertionError("切换基准不得重算图像")

    monkeypatch.setattr("core.services.position_monitoring_service.estimate_charuco_pose", no_pnp)
    monkeypatch.setattr("core.services.position_monitoring_service.estimate_board_pose", no_pnp)
    curves = service.history_comparisons(include_before=True)
    assert [row["batch_id"] for row in curves] == ["B001", "B002", "B003"]
    assert [row["comparison_days"] for row in curves] == [-2, -1, 0]
    assert [row["before_baseline"] for row in curves] == [True, True, False]
    assert [row["debug_day_index"] for row in curves] == [0, 1, 2]
    assert [row["batch_id"] for row in service.history_comparisons()] == ["B003"]
    assert curves[0]["summary"]["rp_change"] == pytest.approx(-0.4)
    assert service.latest_result["summary"]["rp_change"] == pytest.approx(0)
    service.select_baseline(first["path"])
    assert service.latest_batch["batch_id"] == service.current_batch["batch_id"] == "B003"
    assert service.latest_result["summary"]["rp_change"] == pytest.approx(0.4)
    assert service.latest_result["summary"]["absolute_ap_change"] == pytest.approx(0.4)
    assert service.list_history() == archived
    assert [row["id"] for row in service.list_baselines()] == [first["id"], third["id"]]
    service.save_settings({"multidirectional_thresholds": {"repeatability_change": {"distance": 0.3}}})
    assert service.compare_latest()["metric_assessments"]["repeatability_change"]["status"] == "超限"
    assert service.history_comparisons()[-1]["metric_thresholds"]["repeatability_change"]["distance"] == 0.3
    assert all(row["metric_thresholds"]["repeatability_change"]["distance"] is None for row in service.list_history())
    assert {path: path.read_bytes() for path in protected} == protected


def test_older_baseline_evaluation_and_pending_import_never_replace_latest_observation(service, tmp_path):
    first = three_evaluated_batches(service, tmp_path)
    service.current_batch = deepcopy(first["batch"])
    assert service.create_baseline("older observation")["duplicate"]
    service.evaluate()  # 已有基准不重复创建，但显式重评旧观测仍可追溯。
    assert service.current_batch["batch_id"] == "B001"
    assert service.latest_batch["batch_id"] == service.latest_result["batch_id"] == "B003"
    assert service.latest_result["summary"]["rp_change"] == pytest.approx(0.4)
    service.load_observations(observations(tmp_path, "B004", 0.8))
    assert service.current_batch["batch_id"] == "B004"
    assert service.latest_batch["batch_id"] == "B003"
    restored = PositionMonitoringService(root=service.root)
    assert restored.current_batch["batch_id"] == "B004"
    assert restored.latest_result["batch_id"] == "B003"
    # 兼容旧版切换基准留下的空指针；原 history 的最后一条是 B001。
    state_path = service.storage / "state.json"
    state = read_document(state_path)
    state.update({"current_batch_path": None, "latest_result": None})
    write_document(state_path, state)
    restored = PositionMonitoringService(root=service.root)
    assert restored.current_batch["batch_id"] == restored.latest_batch["batch_id"] == "B003"
    assert restored.latest_result["batch_id"] == "B003"


@pytest.mark.parametrize("change, reason", [
    ({"program_id": "different"}, "program_id"),
    ({"target_id": "different"}, "target_id"),
    ({"sampling_protocol": "same_direction"}, "采样方式"),
    ({"different_groups": True}, "测点"),
    ({"different_ideal": True}, "理想位姿"),
    ({"different_parameters": True}, "参数版本"),
])
def test_latest_incompatible_observation_is_not_replaced_by_older_compatible_data(service, tmp_path, change, reason):
    first = three_evaluated_batches(service, tmp_path)
    path = observations(tmp_path, "B004", 0.8)
    document = read_document(path)
    if change.get("different_groups"):
        for sample in document["samples"]:
            sample["point_id"] = "other"
    elif change.get("different_ideal"):
        for sample in document["samples"]:
            sample["ideal_pose"][0][3] += 1
    elif change.get("different_parameters"):
        service.save_parameters({"square_size_mm": service.parameters["square_size_mm"] + 1})
    else:
        document.update(change)
    write_document(path, document)
    service.load_observations(path)
    current_only = service.evaluate_current()
    before = service.list_history()
    service.select_baseline(first["path"])
    assert service.latest_result["batch_id"] == "B004"
    assert reason in service.latest_result["comparison_error"]
    assert service.latest_result["summary"]["rp_current"] == current_only["summary"]["rp_current"]
    assert service.latest_result["summary"]["rp_change"] is None
    assert service.latest_result["summary"]["absolute_ap_change"] is None
    assert [row["batch_id"] for row in service.history_comparisons()] == ["B001", "B002", "B003"]
    assert reason in service.history_comparison_warnings[0]
    assert service.list_history() == before


def test_real_observation_order_uses_capture_instant_and_keeps_single_period_history(service, tmp_path):
    earlier = service.load_observations(observations(
        tmp_path, "B001", 0.2, simulation=False, captured_at="2026-09-21T00:30:00+08:00"))
    service.evaluate_current()
    service.load_observations(observations(
        tmp_path, "B002", 0.4, simulation=False, captured_at="2026-09-20T20:30:00Z"))
    service.evaluate_current()
    service.current_batch = earlier
    service.evaluate_current()
    assert service.latest_batch["batch_id"] == "B002"
    curves = service.history_comparisons()
    assert [row["batch_id"] for row in curves] == ["B001", "B002"]
    assert all(row["summary"]["rp_current"] is not None for row in curves)
    assert all(row["summary"]["rp_change"] is None for row in curves)
    baseline = service.create_baseline()
    assert baseline["batch"]["batch_id"] == "B001"
    assert service.latest_result["batch_id"] == "B002"
    assert service.latest_result["comparison_days"] == pytest.approx(4 / 24)
    assert PositionMonitoringService(root=service.root).latest_result == service.latest_result


@pytest.mark.parametrize("change, reason", [("groups", "测点/接近方向"), ("ideal", "理想位姿")])
def test_current_only_trend_excludes_changed_points_or_ideal_targets(service, tmp_path, change, reason):
    service.load_observations(observations(tmp_path, "B001"))
    service.evaluate_current()
    path = observations(tmp_path, "B002")
    document = read_document(path)
    if change == "groups":
        document["samples"].pop()
    else:
        for sample in document["samples"]:
            sample["ideal_pose"][0][3] += 1
    write_document(path, document)
    service.load_observations(path)
    service.evaluate_current()
    assert [row["batch_id"] for row in service.history_comparisons()] == ["B002"]
    assert reason in service.history_comparison_warnings[0]
    assert len(service.list_history()) == 2


def test_duplicate_baseline_after_reimport_has_no_writes_or_selection_changes(service, tmp_path):
    first = service.load_observations(observations(tmp_path, "B001"))
    original = service.create_baseline("original")
    service.load_observations(observations(tmp_path, "B002", 0.4))
    selected = service.create_baseline("selected")
    source = deepcopy(first)
    source.update({"batch_id": "B009", "label": "new name", "imported_at": "2027-01-01T00:00:00Z"})
    for index, sample in enumerate(source["samples"]):
        sample["sample_id"] = f"renamed-{index}"
    path = tmp_path / "renamed-copy.json"
    write_document(path, source)
    current = service.load_observations(path)
    before = {path: path.read_bytes() for path in service.storage.rglob("*") if path.is_file()}
    latest = deepcopy(service.latest_result)
    events = []
    result = service.create_baseline("do not create", progress=lambda value, message: events.append((value, message)))
    assert result["duplicate"] is True
    assert result["existing_baseline"]["id"] == original["id"]
    assert service.baseline["id"] == selected["id"]
    assert service.current_batch == current
    assert service.latest_result == latest
    assert {path: path.read_bytes() for path in service.storage.rglob("*") if path.is_file()} == before
    assert events[0][0] == 0 and events[-1][0] == 100


@pytest.mark.parametrize("change", ["bias", "rotation", "tiny", "ideal", "q", "initial_error",
                                        "program", "target", "protocol", "status", "parameters", "count"])
def test_equal_rounded_metrics_do_not_replace_exact_measurement_and_condition_checks(service, tmp_path, change):
    first_path = observations(tmp_path, "B001")
    service.load_observations(first_path)
    service.create_baseline("original")
    original_rp = service.latest_result["summary"]["rp_current"]
    document = read_document(first_path)
    if change in ("bias", "tiny"):
        for sample in document["samples"]:
            sample["vision_pose"][0][3] -= 0.1 if change == "bias" else 1e-8
    elif change == "rotation":
        angle = 1e-6
        document["samples"][0]["vision_pose"][0][:3] = [np.cos(angle), -np.sin(angle), 0]
        document["samples"][0]["vision_pose"][1][:3] = [np.sin(angle), np.cos(angle), 0]
    elif change == "ideal":
        for sample in document["samples"]:
            sample["ideal_pose"][0][3] += 0.1
    elif change == "q":
        document["base_rotations"] = {"P001": np.eye(3).tolist()}
    elif change == "initial_error":
        document["initial_errors"] = {"P001": {"D001": [0.1, 0, 0]}}
    elif change in ("program", "target"):
        document[f"{change}_id"] = "another"
    elif change == "protocol":
        document["sampling_protocol"] = "same_direction"
    elif change == "status":
        document["comparison_status"] = "observed"
    elif change == "parameters":
        service.save_parameters({"square_size_mm": service.parameters["square_size_mm"] + 0.1})
    else:
        document["samples"].pop()
    write_document(tmp_path / "changed.json", document)
    service.load_observations(tmp_path / "changed.json")
    created = service.create_baseline("different")
    assert not created.get("duplicate")
    assert len(service.list_baselines()) == 2
    if change == "bias":
        assert service.latest_result["summary"]["rp_current"] == pytest.approx(original_rp)


def test_existing_duplicate_options_keep_selected_identity_and_all_original_files(service, tmp_path):
    service.load_observations(observations(tmp_path))
    original = service.create_baseline("first")
    old_duplicate = read_document(original["path"])
    old_duplicate.update({"id": "old-duplicate", "label": "selected old duplicate",
                          "created_at": "2027-01-01T00:00:00Z",
                          "path": str(service.storage / "baselines" / "old-duplicate.json")})
    write_document(old_duplicate["path"], old_duplicate)
    service.select_baseline(old_duplicate["path"])
    before = {path: path.read_bytes() for path in service.storage.rglob("*.json")}
    options = service.list_baselines()
    assert len(options) == 1
    assert options[0]["id"] == old_duplicate["id"]
    assert options[0]["created_at"] == old_duplicate["created_at"]
    assert service.create_baseline()["existing_baseline"]["id"] == old_duplicate["id"]
    assert {path: path.read_bytes() for path in before} == before


def test_duplicate_sample_order_is_normalized_but_first_reference_is_preserved(service, tmp_path):
    path = observations(tmp_path)
    document = read_document(path)
    third = deepcopy(document["samples"][1])
    third.update({"direction_id": "D003", "sample_id": "3"})
    third["vision_pose"][0][3] -= 0.1
    document["samples"].append(third)
    write_document(path, document)
    service.load_observations(path)
    service.create_baseline()
    document["samples"][1:] = reversed(document["samples"][1:])
    write_document(path, document)
    service.load_observations(path)
    assert service.create_baseline()["duplicate"]
    document["samples"].reverse()
    write_document(path, document)
    service.load_observations(path)
    assert not service.create_baseline().get("duplicate")


def test_statistics_cache_reuses_all_periods_and_only_rechecks_current_thresholds(service, tmp_path, monkeypatch):
    first = three_evaluated_batches(service, tmp_path)
    third = service.create_baseline("third")
    calls = []
    compare = service._compare_batches

    def counted(*args):
        calls.append(args[0]["batch_id"])
        return compare(*args)

    monkeypatch.setattr(service, "_compare_batches", counted)
    events = []
    assert len(service.history_comparisons(progress=lambda value, message: events.append((value, message)))) == 1
    assert calls == ["B001", "B002", "B003"]
    assert all(any(f"{phase}批次 {batch}" in message for _, message in events)
               for batch in ("B001", "B002", "B003") for phase in ("读取", "比较"))
    assert [value for value, _ in events] == sorted(value for value, _ in events)
    assert events[-1][0] == 100
    service.save_settings({"show_before_baseline": True})
    service.save_settings({"multidirectional_thresholds": {"repeatability": {"distance": 0.3}}})
    assert service.compare_latest()["metric_assessments"]["repeatability"]["status"] == "超限"
    rows = service.history_comparisons(include_before=True)
    assert len(rows) == 3
    assert rows[-1]["metric_thresholds"]["repeatability"]["distance"] == 0.3
    rows[-1]["summary"]["rp_current"] = -100
    assert service.history_comparisons()[-1]["summary"]["rp_current"] == pytest.approx(0.6)
    assert calls == ["B001", "B002", "B003"]
    service.select_baseline(first["path"])
    assert len(service.history_comparisons()) == 3
    assert len(calls) == 7  # 重新比较最新观测及3批历史。
    service.evaluate()
    service.history_comparisons()
    assert len(calls) > 7
    service.select_baseline(third["path"])


def test_deferred_restore_and_import_progress_cover_copy_and_save(service, tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"existing pixels with saved pose")
    events = []
    saved = service.load_observations(observations(tmp_path, image=image),
                                      progress=lambda value, message: events.append((value, message)))
    assert any(value < 100 and "托管" in message for value, message in events)
    assert any(value < 100 and "保存" in message for value, message in events)
    assert [value for value, _ in events] == sorted(value for value, _ in events)
    assert events[-1][0] == 100
    assert Path(saved["saved_path"]).exists()
    service.create_baseline()
    delayed = PositionMonitoringService(root=service.root, defer_restore=True)
    assert delayed.current_batch is None and delayed.latest_result is None
    restored_events = []
    delayed.restore(progress=lambda value, message: restored_events.append((value, message)))
    assert delayed.latest_result == service.latest_result
    assert delayed.current_batch["saved_path"] == saved["saved_path"]
    assert [value for value, _ in restored_events] == sorted(value for value, _ in restored_events)
    assert restored_events[-1][0] == 100


@pytest.mark.parametrize("operation", ["baseline", "settings"])
def test_failed_selection_or_setting_save_restores_previous_view(service, tmp_path, monkeypatch, operation):
    first = three_evaluated_batches(service, tmp_path)
    service.create_baseline("third")
    before = (deepcopy(service.baseline), deepcopy(service.settings), deepcopy(service.latest_result),
              service.history_comparisons(include_before=True))
    state = (service.storage / "state.json").read_bytes()

    def fail_save():
        raise OSError("write failed")

    monkeypatch.setattr(service, "_save_state", fail_save)
    with pytest.raises(OSError, match="write failed"):
        if operation == "baseline":
            service.select_baseline(first["path"])
        else:
            service.save_settings({"multidirectional_thresholds": {"repeatability": {"distance": 0.3}}})
    assert (service.baseline, service.settings, service.latest_result,
            service.history_comparisons(include_before=True)) == before
    assert (service.storage / "state.json").read_bytes() == state
