"""包边界、入集权限、分 run 校准及模型历史的业务回归。"""

from copy import deepcopy
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from core.services import spindle_monitoring_service as service_module
from core.services.spindle_monitoring_service import (
    SpindleMonitoringService, assess_compatibility, assess_score,
)


@pytest.fixture
def config():
    return {
        "storage_root": "unused", "target_speed_rpm": 7000,
        "preprocessing": {
            "id": "velocity_2048_10_900_v1", "sample_rate_hz": 2048,
            "frequency_band_hz": [10, 900], "unit": "mm/s", "channels": ["ACC1", "ACC2", "ACC3"],
        },
        "thresholds": {"warning": None, "fault": None},
        "training": {"seed": 17, "validation_fraction": 0.2, "epochs": 2},
    }


@pytest.fixture
def algorithm(monkeypatch):
    calls = {"train": [], "evaluate": []}

    def read_run(path, telemetry_path=None):
        assert Path(telemetry_path).is_file()
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def train_model(training, validation, output_path, config, progress=None):
        calls["train"].append({
            "training": [run["run_id"] for run in training],
            "validation": [run["run_id"] for run in validation],
            "output": str(output_path), "config": deepcopy(config),
        })
        Path(output_path).write_bytes(b"test checkpoint")
        if progress:
            progress(100, "训练完成")
        return {"normal_reference": {}, "best_epoch": 1}

    def evaluate_run(run, model_path=None, config=None, progress=None):
        if model_path is not None:
            assert Path(model_path).is_file()
        calls["evaluate"].append((run["run_id"], str(model_path)))
        config = config or {}
        alphas = {feature: config.get(f"{feature}_alpha", config.get("alpha", .05))
                  for feature in ("vibration", "network")}
        p_values = {"vibration": float(run.get("vibration_p_value", .5)),
                    "network": float(run.get("score", .3)) * .05}
        scores = {feature: min(1.0, p_values[feature] / alphas[feature]) if model_path else None
                  for feature in alphas}
        return {
            "score": scores["network"], "analysis_score": scores["vibration"],
            "score_kind": "normal_compatibility_v1", "alpha": alphas["network"],
            "vibration_p_value": p_values["vibration"] if model_path else None,
            "network_p_value": p_values["network"] if model_path else None,
            "reconstruction_error_p95": 0.25 if model_path else None,
            "compatibility": {feature: {"status": "valid" if model_path else "invalid",
                                        "message": "初步校准" if model_path else "尚未校准",
                                        "alpha": alphas[feature], "score": scores[feature],
                                        "p_value": p_values[feature] if model_path else None}
                              for feature in ("vibration", "network")},
            "rms_mm_s": [0.02, 0.03, 0.05], "waveform": {"time_s": [0, 1], "values": [[1, 2]]},
            "xy_rms_mm_s": (0.02 ** 2 + 0.03 ** 2) ** .5,
            "window_scores": [1, 3] if model_path else [], "reconstruction": None,
        }

    monkeypatch.setattr(service_module.algorithm, "read_run", read_run)
    monkeypatch.setattr(service_module.algorithm, "train_model", train_model)
    monkeypatch.setattr(service_module.algorithm, "evaluate_run", evaluate_run)
    return calls


@pytest.fixture
def service(tmp_path, config, algorithm):
    return SpindleMonitoringService(tmp_path / "store", config)


def package(tmp_path, config, run_ids=("run1",), *, name="data.zip", condition="normal", extra=None):
    manifest = {
        "schema_version": 1, "package_id": name, "captured_date": "2026-07-22",
        "source_type": "historical_replay", "preprocessing": deepcopy(config["preprocessing"]), "runs": [],
    }
    files = {}
    for index, run_id in enumerate(run_ids):
        directory = f"runs/{run_id}"
        manifest["runs"].append({
            "run_id": run_id, "captured_at": f"2026-07-22T12:{index:02d}:00+08:00",
            "speed_rpm": 7000, "operation": "idle", "condition": {"tool_remounted": True},
            "experiment_condition": condition,
            "data_file": directory + "/signals.h5", "telemetry_file": directory + "/telemetry.csv",
            "source_metadata_file": directory + "/manifest.json",
        })
        files[directory + "/signals.h5"] = json.dumps({"run_id": run_id, "target_speed_rpm": 7000, "velocity": [1]})
        files[directory + "/telemetry.csv"] = "time_s,current_a\n0,0.2\n"
        files[directory + "/manifest.json"] = "{}"
    if extra:
        extra(manifest, files)
    path = tmp_path / name
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for filename, content in files.items():
            archive.writestr(filename, content)
    return path


def h5_package(tmp_path, *, name="run1", telemetry=True, speed=7000):
    """无任何 JSON 的真实信号包，文件名和包内目录均不固定。"""
    import h5py
    from test_spindle_algorithms import make_run

    run = make_run(name)
    h5_path = tmp_path / (name + ".hdf5")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs.update({
            "run_id": name, "run_name": name, "run_date": "20260929",
            "captured_at": "2026-09-29T12:34:56+08:00", "target_speed_rpm": speed,
            "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
            "velocity_frequency_band_hz": [10, 900], "velocity_channels": ["ACC1", "ACC2", "ACC3"],
        })
        group = h5.create_group("windows_10s")
        for key in ("velocity", "start_time_s", "temperature", "temperature_valid",
                    "sample_valid", "source_segment_index"):
            group[key] = run[key]
    path = tmp_path / (name + ".zip")
    with ZipFile(path, "w") as archive:
        archive.write(h5_path, "任意目录/振动.hdf5")
        archive.writestr("任意目录/其他.csv", "sensor,label\nACC1,前轴承\n")
        if telemetry:
            archive.writestr("任意目录/驱动器.csv", "time_s,actual_speed_rpm,current_a\n610,7000,0.4\n")
    return path


@pytest.mark.parametrize("telemetry", [True, False])
def test_batch_accepts_real_h5_without_any_manifest(tmp_path, config, telemetry):
    service = SpindleMonitoringService(tmp_path / "store", config)
    path = h5_package(tmp_path, telemetry=telemetry)
    report = service.import_batch([path])
    assert report["new_count"] == 1 and report["failed_count"] == 0
    record = service.runs["run1"]
    assert record["captured_at"] == "2026-09-29T12:34:56+08:00"
    assert record["source_metadata_file"] is None
    assert bool(record["telemetry_file"]) == telemetry
    result = service.latest_result("run1")
    assert result["window_count"] == 10
    assert bool(result["telemetry"]["current_a"]) == telemetry
    assert record["manual_label"] == "unconfirmed" and not record["training_eligible"]
    assert service.current_model is None
    assert not list(service.root.glob("spindle_*"))


def test_batch_recurses_folders_and_zips_continues_failures_and_deduplicates(service, tmp_path, config):
    first = package(tmp_path, config, ("one", "two"), name="first.zip")
    second = package(tmp_path, config, ("three",), name="second.zip")
    day = tmp_path / "day.zip"
    with ZipFile(day, "w") as archive:
        archive.write(second, "工况/深层/second.ZIP")
        archive.writestr("broken.zip", b"not a zip")
    outer = tmp_path / "23-25.zip"
    with ZipFile(outer, "w") as archive:
        archive.write(first, "23/上午/first.zip")
        archive.write(day, "24/任意目录/day.zip")
        archive.write(first, "25/repeated.zip")
        archive.writestr("README.txt", "无须清单")
    progress = []
    report = service.import_batch([outer], lambda value, message: progress.append((value, message)))
    assert (report["new_count"], report["duplicate_count"], report["failed_count"]) == (3, 2, 1)
    assert set(service.runs) == {"one", "two", "three"}
    assert len(service.history()) == 3 and len(service.state["packages"]) == 2
    assert service.runs["three"]["source_filename"] == "23-25.zip!/24/任意目录/day.zip!/工况/深层/second.ZIP"
    assert "broken.zip" in next(item["message"] for item in report["packages"] if item["status"] == "failed")
    assert [value for value, _ in progress] == sorted(value for value, _ in progress)
    assert progress[0][0] == 0 and progress[-1][0] == 100
    assert not list(service.root.glob("spindle_*"))
    service.set_label("one", "abnormal")
    repeated = service.import_batch([outer])
    assert repeated["new_count"] == 0 and len(service.history()) == 3
    assert service.runs["one"]["manual_label"] == "abnormal"
    assert not service.runs["one"]["training_eligible"]


def test_batch_keeps_valid_sample_terminal_and_reports_conflicting_ids(service, tmp_path, config):
    hidden = package(tmp_path, config, ("hidden",), name="hidden.zip")
    valid = package(tmp_path, config, name="valid.zip")
    with ZipFile(valid, "a") as archive:
        archive.write(hidden, "attachment.zip")
    conflict = package(tmp_path, config, name="conflict.zip", extra=lambda manifest, files:
                       manifest["runs"][0].update(captured_at="2026-07-23T12:00:00+08:00"))
    report = service.import_batch([valid, conflict])
    assert (report["new_count"], report["failed_count"]) == (1, 1)
    assert set(service.runs) == {"run1"}
    assert "时间或工况不同" in report["packages"][-1]["message"]


def test_batch_retains_imported_runs_and_retries_failed_analysis(service, tmp_path, config, monkeypatch):
    path = package(tmp_path, config)
    original = service.evaluate
    monkeypatch.setattr(service, "evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("分析中断")))
    report = service.import_batch([path])
    assert report["new_count"] == 1 and report["failed_count"] == 1
    assert "已保存 1 条，分析未完成" in report["packages"][0]["message"]
    assert len(service.runs) == 1 and not service.history()
    monkeypatch.setattr(service, "evaluate", original)
    report = service.import_batch([path])
    assert report["new_count"] == 0 and report["duplicate_count"] == 1 and report["failed_count"] == 0
    assert len(service.history()) == 1


def test_batch_rejects_wrong_h5_operating_speed_without_partial_records(tmp_path, config):
    service = SpindleMonitoringService(tmp_path / "store", config)
    report = service.import_batch([h5_package(tmp_path, speed=7400)])
    assert report["failed_count"] == 1 and report["new_count"] == 0
    assert "转速" in report["packages"][0]["message"]
    assert not service.runs and not service.state["packages"]
    assert not list(service.root.glob("spindle_*"))


def test_initial_runs_require_confirmed_healthy_batch_before_training(service, tmp_path, config):
    path = package(tmp_path, config, ("run1", "run2", "run3"))
    records = service.import_packages([path], purpose="initial")
    assert len(records) == 3 and not service.training_candidates()
    assert all(not run["training_eligible"] for run in records)
    with pytest.raises(ValueError, match="5 个"):
        service.train()
    assert all(run["manual_label"] == "unconfirmed" for run in records)
    assert all(run["label_history"] == [] for run in records)
    assert len(service.history()) == 3
    assert service.latest_result("run1")["score"] is None
    assert service.latest_result("run1")["waveform"]["values"] == [[1, 2]]


def test_daily_experiment_label_is_not_a_human_label(service, tmp_path, config):
    path = package(tmp_path, config, condition="normal")
    service.import_packages([path])
    assert service.runs["run1"]["manual_label"] == "unconfirmed"
    assert service.runs["run1"]["experiment_condition"] == "normal"
    assert service.training_candidates() == []


def test_batch_label_is_shared_and_only_healthy_data_can_train(service, tmp_path, config, algorithm):
    first = package(tmp_path, config, ("one", "two", "three", "four", "five"), name="first.zip")
    second = package(tmp_path, config, ("other",), name="second.zip")
    service.import_batch([first], label="healthy")
    service.import_batch([second], label="abnormal")
    assert {run["run_id"] for run in service.training_candidates()} == {"one", "two", "three", "four", "five"}
    assert len({service.runs[key]["import_batch_id"] for key in ("one", "two", "three", "four", "five")}) == 1
    service.set_label("two", "abnormal")
    assert all(service.runs[key]["manual_label"] == "abnormal" for key in ("one", "two", "three", "four", "five"))
    assert not service.training_candidates()
    service.set_label("one", "unconfirmed", include_in_training=True)
    assert not service.training_candidates()
    assert all(not service.runs[key]["training_eligible"] for key in ("one", "two", "three", "four", "five"))
    service.set_label("three", "healthy")
    assert len(service.training_candidates()) == 5
    assert service.runs["other"]["manual_label"] == "abnormal"
    model = service.train()
    assert all(run["manual_label"] == "healthy" for run in model["training_samples"])
    assert set(algorithm["train"][0]["training"] + algorithm["train"][0]["validation"]) == {"one", "two", "three", "four", "five"}
    with pytest.raises(ValueError, match="不能修改人工判定"):
        service.set_label("two", "abnormal")
    assert len(service.training_candidates()) == 5


def test_old_batch_group_is_recovered_and_unconfirmed_flags_cannot_enable_training(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("one", "two", "three"))], "initial")
    for index, run in enumerate(service.runs.values()):
        run.pop("import_batch_id")
        run["source_filename"] = f"202607.zip!/day{index}/sample.zip"
        run["training_eligible"] = True
    service._save()
    restored = SpindleMonitoringService(service.root, config)
    assert not restored.training_candidates()
    assert all(not run["training_eligible"] for run in restored.runs.values())
    # 即使旧调用误设入训标志，训练入口也独立检查标签。
    for run in restored.runs.values():
        run["training_eligible"] = True
    with pytest.raises(ValueError, match="5 个"):
        restored.train()
    restored.set_label("two", "healthy")
    assert len(restored.training_candidates()) == 3
    assert all(run["manual_label"] == "healthy" for run in restored.runs.values())


def test_failed_batch_label_save_restores_every_member(service, tmp_path, config, monkeypatch):
    service.import_batch([package(tmp_path, config, ("one", "two", "three"))])
    original = deepcopy(service.runs)

    def fail():
        raise OSError("整批保存失败")

    monkeypatch.setattr(service, "_save", fail)
    with pytest.raises(OSError, match="整批保存失败"):
        service.set_label("two", "healthy")
    assert service.runs == original
    assert SpindleMonitoringService(service.root, config).runs == original


def test_duplicate_daily_run_cannot_become_initial(service, tmp_path, config):
    path = package(tmp_path, config)
    service.import_packages([path], purpose="daily")
    service.import_packages([path], purpose="initial")
    assert service.runs["run1"]["purpose"] == "daily"
    assert not service.runs["run1"]["training_eligible"]
    assert len(service.history()) == len(service.state["packages"]) == 1


def test_labels_control_daily_candidates_and_survive_restart(service, tmp_path, config):
    service.import_packages([package(tmp_path, config)])
    service.set_label("run1", "healthy", "人工复核", include_in_training=False)
    assert not service.training_candidates()
    service.set_label("run1", "healthy", include_in_training=True)
    assert len(service.training_candidates()) == 1
    service.set_label("run1", "abnormal", "需检修", include_in_training=True)
    assert not service.training_candidates()
    restored = SpindleMonitoringService(service.root, config)
    assert restored.runs["run1"]["manual_label"] == "abnormal"
    assert restored.runs["run1"]["label_note"] == "需检修"
    assert len(restored.runs["run1"]["label_history"]) == 3
    assert restored.latest_result("run1")["manual_label_at_evaluation"] == "unconfirmed"


def test_unconfirmed_daily_run_is_never_a_candidate(service, tmp_path, config):
    service.import_packages([package(tmp_path, config)])
    service.set_label("run1", "healthy")
    service.set_label("run1", "unconfirmed")
    assert not service.runs["run1"]["training_eligible"]


def test_training_requires_separate_complete_runs(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2"))], purpose="initial")
    with pytest.raises(ValueError, match="5 个"):
        service.train()


def test_training_rejects_candidates_processed_with_another_version(service, tmp_path, config, algorithm):
    ids = tuple(f"run{i}" for i in range(5))
    service.import_batch([package(tmp_path, config, ids)], label="healthy")
    service.runs[ids[0]]["preprocessing"]["id"] = "previous_velocity_pipeline"
    with pytest.raises(ValueError, match="预处理版本"):
        service.train()
    assert not algorithm["train"] and not service.models


@pytest.mark.parametrize("boundary", ["condition", "length"])
def test_training_does_not_pool_small_incompatible_reference_groups(service, tmp_path, config, algorithm, boundary):
    def separate_groups(manifest, files):
        for run in manifest["runs"][3:]:
            if boundary == "condition":
                run["condition"]["tool_remounted"] = False
            else:
                data = json.loads(files[run["data_file"]])
                data["velocity"] = [1, 2]
                files[run["data_file"]] = json.dumps(data)

    ids = tuple(f"run{i}" for i in range(6))
    service.import_batch([package(tmp_path, config, ids, extra=separate_groups)], label="healthy")
    with pytest.raises(ValueError, match="同工况、同长度"):
        service.train()
    assert not algorithm["train"] and not service.models


def test_training_reserves_three_independent_calibrations_in_each_condition(service, tmp_path, config, algorithm):
    def separate_groups(manifest, files):
        for run in manifest["runs"][4:]:
            run["condition"]["tool_remounted"] = False

    ids = tuple(f"run{i}" for i in range(8))
    service.import_batch([package(tmp_path, config, ids, extra=separate_groups)], label="healthy")
    model = service.train()
    assert len(model["training_run_ids"]) == 2
    assert len(model["calibration_run_ids"]) == 6
    for condition in (True, False):
        assert sum(service.runs[key]["condition"]["tool_remounted"] == condition
                   for key in model["calibration_run_ids"]) == 3
    assert set(algorithm["train"][0]["training"]).isdisjoint(algorithm["train"][0]["validation"])


def test_training_split_versions_and_history_reanalysis(service, tmp_path, config, algorithm):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip", condition="unbalance1")])
    service.set_label("daily", "abnormal", "检查轴承")
    first = service.train()
    assert first["version"].startswith("tcn_1s_2ep_")
    assert first["model_path"] == f"models/{first['version']}/model.pt"
    call = algorithm["train"][0]
    assert len(call["training"]) == 2 and len(call["validation"]) == 3
    assert set(call["training"]).isdisjoint(call["validation"])
    assert "daily" not in call["training"] + call["validation"]
    assert len(service.history(model_version=first["version"])) == 6
    assert service.latest_result("daily")["role"] == "independent"
    assert {result["role"] for result in service.history(model_version=first["version"])} == {
        "training", "calibration", "independent",
    }
    assert all(result["captured_at"].startswith("2026-07-22") for result in service.history())
    first_results = deepcopy(service.history(model_version=first["version"]))
    second = service.train()
    assert first["version"] != second["version"]
    assert first["model_path"] != second["model_path"]
    assert first["initialization"] == second["initialization"] == "random"
    assert algorithm["train"][1]["training"] == call["training"]
    assert algorithm["train"][1]["validation"] == call["validation"]
    assert service.history(model_version=first["version"]) == first_results
    assert len(service.history(model_version=second["version"])) == 6
    assert len(service.history()) == 18
    assert service.runs["daily"]["manual_label"] == "abnormal"
    restored = SpindleMonitoringService(service.root, config)
    assert restored.current_model["version"] == second["version"]
    assert restored.current_model["reanalysis_status"] == "complete"
    assert len(restored.history()) == 18


def test_legacy_reference_uses_only_original_calibration_and_never_daily_candidates(
        service, tmp_path, config, monkeypatch):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], "initial")
    service.set_label("run1", "healthy")
    model = service.train()
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip")])
    service.set_label("daily", "healthy")
    model.pop("normal_reference")
    calibration_calls = []

    def calibrate(runs, model_path, preprocessing):
        calibration_calls.append([run["run_id"] for run in runs])
        assert Path(model_path).is_file()
        assert preprocessing == config["preprocessing"]
        return {"groups": [], "reference_marker": "fixed"}

    monkeypatch.setattr(service_module.algorithm, "calibrate_model", calibrate)
    history = deepcopy(service.history())
    result = service.evaluate("daily")
    assert calibration_calls == [model["calibration_run_ids"]]
    assert set(calibration_calls[0]).isdisjoint(model["training_run_ids"] + ["daily"])
    service.evaluate("daily")
    assert len(calibration_calls) == 1
    assert model["normal_reference"]["model_version"] == model["version"]
    assert service.history()[:len(history)] == history
    assert result["analysis_score"] == 1
    restored = SpindleMonitoringService(service.root, config)
    assert restored.current_model["normal_reference"] == model["normal_reference"]
    assert restored.analysis_metrics(result)[0] == 1


@pytest.mark.parametrize("change", ["path", "mtime", "size"])
def test_legacy_reference_cache_rebuilds_after_model_file_changes(
        service, tmp_path, config, monkeypatch, change):
    ids = tuple(f"run{i}" for i in range(5))
    service.import_batch([package(tmp_path, config, ids)], label="healthy")
    model = service.train()
    model_path = service.root / model["model_path"]
    cached = {"network_id": service_module.algorithm._network_id({}, model_path),
              "groups": [], "model_version": model["version"]}
    model["normal_reference"] = deepcopy(cached)
    assert service._normal_reference(model) == cached

    if change == "path":
        moved = model_path.with_name("moved_model.pt")
        model_path.rename(moved)
        model_path = moved
        model["model_path"] = moved.relative_to(service.root).as_posix()
    elif change == "mtime":
        stat = model_path.stat()
        os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        model_path.write_bytes(b"updated test checkpoint with different size")

    calls = []

    def calibrate(runs, path, preprocessing):
        calls.append([run["run_id"] for run in runs])
        assert Path(path) == model_path
        assert preprocessing == config["preprocessing"]
        return {"network_id": service_module.algorithm._network_id({}, path), "groups": []}

    monkeypatch.setattr(service_module.algorithm, "calibrate_model", calibrate)
    rebuilt = service._normal_reference(model)
    assert calls == [model["calibration_run_ids"]]
    assert rebuilt["network_id"] != cached["network_id"]
    assert rebuilt["model_version"] == model["version"]
    assert service._normal_reference(model) == rebuilt
    assert len(calls) == 1
    restored = SpindleMonitoringService(service.root, config)
    assert restored.current_model["normal_reference"] == rebuilt
    assert restored._normal_reference(restored.current_model) == rebuilt
    assert len(calls) == 1


def test_legacy_history_and_thresholds_are_not_reinterpreted(service, config):
    legacy = {"score": 3.0, "analysis_score": 1.2, "alpha": 0.05,
              "assessment": {"status": "normal", "message": "旧判定"}}
    original = deepcopy(legacy)
    service.settings.pop("score_kind")
    service.settings["thresholds"] = {"warning": 5, "fault": 15}
    service._save()
    restored = SpindleMonitoringService(service.root, config)
    assert restored.settings["thresholds"] == {"warning": None, "fault": None}
    assert restored.settings["legacy_thresholds"] == {"warning": 5, "fault": 15}
    assert restored.settings["alpha"] == 0.05
    shown = restored.display_result(legacy, current_settings=True)
    assert shown["score"] is None and shown["analysis_score"] is None
    assert shown["assessment"]["status"] == "legacy"
    assert restored.analysis_metrics(legacy) == (None, None)
    assert legacy == original


@pytest.mark.parametrize("problem", ["duplicate", "training_overlap", "missing", "abnormal"])
def test_invalid_legacy_calibration_members_are_rejected(service, tmp_path, config, monkeypatch, problem):
    service.import_batch([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], label="healthy")
    model = service.train()
    model.pop("normal_reference")
    calibration = model["calibration_run_ids"]
    if problem == "duplicate":
        calibration.append(calibration[0])
    elif problem == "training_overlap":
        calibration.append(model["training_run_ids"][0])
    elif problem == "missing":
        calibration.append("missing")
    else:
        service.runs[calibration[0]]["manual_label"] = "abnormal"

    def unexpected_calibration(*args, **kwargs):
        pytest.fail("无效校准清单不能进入参考计算")

    monkeypatch.setattr(service_module.algorithm, "calibrate_model", unexpected_calibration)
    reference = service._normal_reference(model)
    assert reference["groups"] == []
    assert "无效" in reference["message"] or "重叠" in reference["message"]
    assert "normal_reference" not in model


def test_initial_import_is_closed_after_first_model(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    path = package(tmp_path, config, ("late",), name="late.zip")
    with pytest.raises(ValueError, match="日常导入"):
        service.import_packages([path], purpose="initial")
    service.import_packages([path])
    result = service.latest_result("late")
    assert result["score"] == 0.3
    assert result["reason"] == "import"
    assert result["model_version"] == service.current_model["version"]
    assert service.runs["late"]["manual_label"] == "unconfirmed"


def test_trained_and_calibration_labels_stay_locked_after_restart(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    model = service.train()
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip")])
    service.train()
    restored = SpindleMonitoringService(service.root, config)
    before = deepcopy(restored.state)
    for key in ("training_run_ids", "calibration_run_ids"):
        with pytest.raises(ValueError, match="不能修改人工判定"):
            restored.set_labels(["daily", model[key][0]], "abnormal", "不能部分修改")
        assert restored.state == before
    assert SpindleMonitoringService(service.root, config).state == before
    restored.set_label("daily", "healthy")
    assert restored.runs["daily"]["manual_label"] == "healthy"


def test_historical_model_locks_labels_even_when_current_model_did_not_use_run(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("one", "two", "three", "four", "five"))], "initial")
    service.set_label("one", "healthy")
    first = service.train()
    service.train()
    # 当前模型成员不同，旧版本仍然决定标签是否可改。
    service.current_model["training_run_ids"] = []
    service.current_model["calibration_run_ids"] = []
    service._save()
    restored = SpindleMonitoringService(service.root, config)
    assert restored.current_model["version"] != first["version"]
    with pytest.raises(ValueError, match="不能修改人工判定"):
        restored.set_label("one", "abnormal")


def test_threshold_changes_only_apply_to_new_evaluations(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    old = service.latest_result("run1")
    assert old["assessment"]["status"] == "unconfigured"
    service.set_thresholds(warning=0.5, fault=0.3)
    new = service.evaluate("run1")
    assert new["assessment"]["status"] == "fault"
    assert "建议检修" in new["assessment"]["message"]
    assert service.history("run1")[-2] == old
    assert old["thresholds"] == {"warning": None, "fault": None}


@pytest.mark.parametrize(("score", "thresholds", "status"), [
    (None, {"warning": 0.5, "fault": 0.2}, "unavailable"),
    (float("nan"), {"warning": 0.5, "fault": 0.2}, "unavailable"),
    (float("inf"), {"warning": 0.5, "fault": 0.2}, "unavailable"),
    (-0.1, {"warning": 0.5, "fault": 0.2}, "unavailable"),
    (1.1, {"warning": 0.5, "fault": 0.2}, "unavailable"),
    (1, {"warning": None, "fault": None}, "unconfigured"),
    (0.3, {"warning": None, "fault": None}, "unconfigured"),
    (1, {"warning": 0.5, "fault": 0.2}, "normal"),
    (0.8, {"warning": 0.5, "fault": 0.2}, "deviation"),
    (0.5, {"warning": 0.5, "fault": 0.2}, "warning"),
    (0.2, {"warning": 0.5, "fault": 0.2}, "fault"),
    (0, {"warning": None, "fault": 0}, "fault"),
])
def test_threshold_priority(score, thresholds, status):
    assert assess_score(score, thresholds)["status"] == status


@pytest.mark.parametrize(("score", "thresholds", "status", "message"), [
    (0.0, {"warning": None, "fault": None}, "unconfigured", "偏离正常参考 · 未设置报警阈值"),
    (0.8, {"warning": None, "fault": None}, "unconfigured", "偏离正常参考 · 未设置报警阈值"),
    (0.8, {"warning": 0.5, "fault": 0.2}, "deviation", "偏离正常参考，未达报警阈值"),
    (1.0, {"warning": 0.5, "fault": 0.2}, "normal", "阈值内"),
])
def test_reference_deviation_is_not_hidden_by_missing_or_untriggered_alarm_thresholds(
        score, thresholds, status, message):
    assert assess_score(score, thresholds) == {"status": status, "message": message}


@pytest.mark.parametrize(("vibration", "network"), [(0.8, 1.0), (1.0, 0.8), (0.8, 0.9)])
@pytest.mark.parametrize("configured", [False, True])
def test_combined_assessment_preserves_deviation_of_either_valid_feature(vibration, network, configured):
    thresholds = {"warning": 0.5, "fault": 0.2} if configured else {"warning": None, "fault": None}
    result = {"analysis_score": vibration, "score": network}
    assessment = assess_compatibility(result, thresholds)
    assert assessment["status"] == ("deviation" if configured else "unconfigured")
    assert assessment["message"] == (
        "偏离正常参考，未达报警阈值" if configured else "偏离正常参考 · 未设置报警阈值")


@pytest.mark.parametrize(("warning", "fault"), [
    (0.2, 0.3), (0.3, 0.3), (-1, None), (float("nan"), None),
    (1, None), (None, 1), (float("inf"), None),
])
def test_invalid_thresholds_do_not_change_settings(service, warning, fault):
    before = deepcopy(service.settings)
    with pytest.raises(ValueError):
        service.set_thresholds(warning, fault)
    assert service.settings == before


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_alpha_does_not_change_settings(service, alpha):
    before = deepcopy(service.settings)
    with pytest.raises(ValueError, match="alpha"):
        service.set_thresholds(0.5, 0.2, alpha=alpha)
    assert service.settings == before


@pytest.mark.parametrize(("vibration", "network", "status"), [
    (1.0, 1.0, "normal"), (0.5, 1.0, "warning"), (1.0, 0.5, "warning"),
    (0.2, 0.5, "fault"), (1.0, 0.2, "fault"), (None, 1.0, "unavailable"),
    (1.0, None, "unavailable"), (None, 0.0, "fault"), (0.0, None, "fault"),
    (None, 0.5, "warning"), (0.5, None, "warning"),
])
def test_both_scores_drive_alarm_and_invalid_cannot_be_normal(vibration, network, status):
    result = {"analysis_score": vibration, "score": network}
    assert assess_compatibility(result, {"warning": 0.5, "fault": 0.2})["status"] == status


@pytest.mark.parametrize("invalid_feature", ["vibration", "network"])
@pytest.mark.parametrize(("valid_score", "review_required", "status"), [
    (0.2, False, "fault"), (0.5, False, "warning"), (1.0, False, "unavailable"),
    (0.2, True, "unavailable"), (0.5, True, "unavailable"),
])
def test_invalid_metric_does_not_hide_another_alarm_or_bypass_review(
        invalid_feature, valid_score, review_required, status):
    result = {
        "analysis_score": None if invalid_feature == "vibration" else valid_score,
        "score": None if invalid_feature == "network" else valid_score,
        "compatibility": {invalid_feature: {"status": "invalid", "message": "匹配参考不足 3 次"}},
    }
    assessment = assess_compatibility(result, {"warning": 0.5, "fault": 0.2}, review_required)
    assert assessment["status"] == status
    assert "匹配参考不足 3 次" in assessment["message"]


@pytest.mark.parametrize(("vibration", "network"), [(0.2, 1.0), (1.0, 0.2), (0.5, 0.5)])
def test_unreviewed_thresholds_never_trigger_an_alarm(vibration, network):
    result = {"analysis_score": vibration, "score": network}
    assessment = assess_compatibility(result, {"warning": 0.5, "fault": 0.2}, review_required=True)
    assert assessment["status"] == "review_required"


@pytest.mark.parametrize("invalid_feature", ["vibration", "network"])
def test_invalid_metric_without_configured_thresholds_is_unavailable(invalid_feature):
    result = {"analysis_score": None if invalid_feature == "vibration" else 0.0,
              "score": None if invalid_feature == "network" else 0.0}
    assert assess_compatibility(result, {"warning": None, "fault": None})["status"] == "unavailable"


def test_current_alpha_rescores_from_saved_p_without_rewriting_history(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], "initial")
    service.set_label("run1", "healthy")
    service.train()
    original = service.latest_result("run1")
    original_copy = deepcopy(original)
    service.set_thresholds(0.5, 0.2, alpha=0.1)
    shown = service.display_result(original, current_settings=True)
    assert shown["alpha"] == 0.1
    assert shown["score"] == pytest.approx(0.15)
    assert shown["analysis_score"] == 1
    for feature, score_key in (("vibration", "analysis_score"), ("network", "score")):
        assert shown["compatibility"][feature]["alpha"] == shown["alpha"]
        assert shown["compatibility"][feature]["score"] == shown[score_key]
        assert original["compatibility"][feature]["alpha"] == 0.05
    assert shown["assessment"]["status"] == "fault"
    assert service.display_result(original) == original_copy
    assert original == original_copy
    assert service.latest_result("run1") == original_copy
    assert SpindleMonitoringService(service.root, config).settings["alpha"] == 0.1


@pytest.mark.parametrize(("feature", "score_key", "other_feature", "other_score_key", "changed", "alarm"), [
    ("vibration", "analysis_score", "network", "score", {"alpha": .8, "warning": .7, "fault": .3}, "warning"),
    ("network", "score", "vibration", "analysis_score", {"alpha": .1, "warning": .25, "fault": .2}, "fault"),
])
def test_independent_metric_preview_changes_only_selected_score_and_alarm_without_saving(
        service, tmp_path, config, algorithm, feature, score_key, other_feature, other_score_key, changed, alarm):
    service.import_batch([package(tmp_path, config, tuple(f"run{i}" for i in range(5)))], label="healthy")
    model = service.train()
    service.set_metric_settings({name: {"alpha": .05, "warning": .2, "fault": .1}
                                 for name in ("vibration", "network")})
    original = service.latest_result("run0")
    baseline = service.display_result(original, current_settings=True)
    settings_before = deepcopy(service.settings)
    reference_before = deepcopy(model["normal_reference"])
    bytes_before = service.state_path.read_bytes()
    calls_before = len(algorithm["evaluate"])
    draft = deepcopy(service.metric_settings)
    draft[feature] = changed

    preview = service.display_result(original, current_settings=True, metric_settings=draft)
    assert preview[score_key] != baseline[score_key]
    assert preview[score_key] == pytest.approx(min(1, preview[f"{feature}_p_value"] / changed["alpha"]))
    assert preview[other_score_key] == baseline[other_score_key]
    assert preview["compatibility"][other_feature] == baseline["compatibility"][other_feature]
    assert preview["compatibility"][feature]["alpha"] == changed["alpha"]
    assert preview["metric_settings"] == draft
    assert preview["alpha"] == draft["network"]["alpha"]
    assert preview["thresholds"] == {key: draft["network"][key] for key in ("warning", "fault")}
    assert preview["assessment"]["status"] == alarm
    assert assess_score(preview[other_score_key], draft[other_feature]) == assess_score(
        baseline[other_score_key], baseline["metric_settings"][other_feature])
    for key in ("xy_rms_mm_s", "reconstruction_error_p95", "vibration_p_value", "network_p_value"):
        assert preview[key] == baseline[key]
    assert service.settings == settings_before
    assert service.state_path.read_bytes() == bytes_before
    assert service.latest_result("run0") == original
    assert model["normal_reference"] == reference_before
    assert len(algorithm["evaluate"]) == calls_before


def test_independent_metric_settings_are_used_by_evaluation_saved_as_snapshot_and_restored(service, tmp_path, config):
    service.import_batch([package(tmp_path, config, tuple(f"run{i}" for i in range(5)))], label="healthy")
    model = service.train()
    original = service.latest_result("run0")
    reference_before = deepcopy(model["normal_reference"])
    metrics = {"vibration": {"alpha": .8, "warning": .7, "fault": .3},
               "network": {"alpha": .1, "warning": .25, "fault": .2}}
    service.set_metric_settings(metrics)
    evaluated = service.evaluate("run0")
    assert evaluated["analysis_score"] == pytest.approx(.625)
    assert evaluated["score"] == pytest.approx(.15)
    assert evaluated["metric_settings"] == metrics
    assert evaluated["alpha"] == .1
    assert evaluated["thresholds"] == {"warning": .25, "fault": .2}
    assert evaluated["compatibility"]["vibration"]["alpha"] == .8
    assert evaluated["compatibility"]["network"]["alpha"] == .1
    assert evaluated["assessment"]["status"] == "fault"
    assert service.history("run0")[-2] == original
    assert model["normal_reference"] == reference_before
    for key in ("xy_rms_mm_s", "reconstruction_error_p95", "vibration_p_value", "network_p_value"):
        assert evaluated[key] == original[key]
    restored = SpindleMonitoringService(service.root, config)
    assert restored.metric_settings == metrics
    assert restored.latest_result("run0") == evaluated
    assert restored.settings["alpha"] == metrics["network"]["alpha"]
    assert restored.settings["thresholds"] == {"warning": .25, "fault": .2}
    assert restored.result_metric_settings(original) == original["metric_settings"]


def test_shared_historical_settings_expand_equally_without_using_current_metrics(service, config):
    common = {"alpha": .08, "warning": .45, "fault": .12}
    historical = {"alpha": common["alpha"], "thresholds": {"warning": .45, "fault": .12}}
    expected = {feature: deepcopy(common) for feature in ("vibration", "network")}
    split = {"vibration": {"alpha": .2, "warning": .8, "fault": .4},
             "network": {"alpha": .04, "warning": .3, "fault": .1}}
    service.set_metric_settings(split)
    mapped = service_module.result_metric_settings(historical)
    assert mapped == expected
    mapped["vibration"]["alpha"] = .3
    assert mapped["network"] == common
    assert service_module.result_metric_settings(historical) == expected
    assert service.metric_settings == split

    service.settings.pop("metric_settings")
    service.settings.update(alpha=common["alpha"], thresholds=deepcopy(historical["thresholds"]))
    service._save()
    restored = SpindleMonitoringService(service.root, config)
    assert restored.metric_settings == expected


def test_legacy_threshold_setter_applies_same_values_to_both_metrics(service):
    service.set_metric_settings({"vibration": {"alpha": .2, "warning": .8, "fault": .4},
                                 "network": {"alpha": .04, "warning": .3, "fault": .1}})
    service.set_thresholds(.6, .2, alpha=.1)
    assert service.metric_settings == {
        feature: {"alpha": .1, "warning": .6, "fault": .2} for feature in ("vibration", "network")}


@pytest.mark.parametrize("feature", ["vibration", "network"])
@pytest.mark.parametrize(("field", "invalid"), [("alpha", 0), ("alpha", float("nan")),
                                                 ("warning", 1), ("fault", .8)])
def test_invalid_metric_form_does_not_partially_save_other_metric(service, config, feature, field, invalid):
    baseline = {"vibration": {"alpha": .05, "warning": .5, "fault": .2},
                "network": {"alpha": .08, "warning": .6, "fault": .1}}
    service.set_metric_settings(baseline)
    settings_before = deepcopy(service.settings)
    bytes_before = service.state_path.read_bytes()
    draft = deepcopy(baseline)
    for values in draft.values():
        values["alpha"] = .15
    draft[feature][field] = invalid
    with pytest.raises(ValueError):
        service.set_metric_settings(draft)
    assert service.settings == settings_before
    assert service.state_path.read_bytes() == bytes_before
    assert SpindleMonitoringService(service.root, config).metric_settings == baseline


def test_failed_split_settings_save_restores_both_metrics_and_compatibility_fields(service, config, monkeypatch):
    baseline = {"vibration": {"alpha": .05, "warning": .5, "fault": .2},
                "network": {"alpha": .08, "warning": .6, "fault": .1}}
    service.set_metric_settings(baseline)
    settings_before = deepcopy(service.settings)

    def failed_save():
        raise OSError("写盘失败")

    monkeypatch.setattr(service, "_save", failed_save)
    with pytest.raises(OSError, match="写盘失败"):
        service.set_metric_settings({"vibration": {"alpha": .2, "warning": .8, "fault": .4},
                                     "network": {"alpha": .04, "warning": .3, "fault": .1}})
    assert service.settings == settings_before
    assert SpindleMonitoringService(service.root, config).metric_settings == baseline


@pytest.mark.parametrize("member", ["../outside.txt", "/outside.txt", "C:/outside.txt", "runs/../outside.txt"])
def test_archive_traversal_rejected_before_import(service, tmp_path, config, member):
    path = package(tmp_path, config, extra=lambda manifest, files: files.update({member: "bad"}))
    with pytest.raises(ValueError, match="不安全"):
        service.import_packages([path])
    assert not service.runs
    assert not service.history()
    assert not (tmp_path / "outside.txt").exists()


def test_preprocessing_mismatch_is_not_silently_mixed(service, tmp_path, config):
    path = package(tmp_path, config, extra=lambda manifest, files: manifest["preprocessing"].update({"unit": "g"}))
    with pytest.raises(ValueError, match="预处理参数"):
        service.import_packages([path])
    assert not service.runs


def test_missing_member_does_not_register_partial_run(service, tmp_path, config):
    path = package(tmp_path, config, extra=lambda manifest, files: files.pop("runs/run1/telemetry.csv"))
    with pytest.raises(ValueError, match="缺少文件"):
        service.import_packages([path])
    assert not service.runs


def test_invalid_payload_does_not_register_partial_run(service, tmp_path, config):
    path = package(tmp_path, config, extra=lambda manifest, files: files.update({"runs/run1/signals.h5": "not JSON"}))
    with pytest.raises(json.JSONDecodeError):
        service.import_packages([path])
    assert not service.runs
    assert not service.state["packages"]


def test_nonstandard_speed_is_rejected(service, tmp_path, config):
    path = package(tmp_path, config, extra=lambda manifest, files: manifest["runs"][0].update({"speed_rpm": 7400}))
    with pytest.raises(ValueError, match="转速"):
        service.import_packages([path])
    assert not service.runs


def test_backslash_member_is_rejected_before_zip_normalization():
    with pytest.raises(ValueError, match="不安全"):
        service_module._relative_member(r"runs\outside.txt")


def test_latest_result_does_not_fall_back_after_failed_reanalysis(service, tmp_path, config, monkeypatch):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    first = service.train()
    previous = service.latest_result("run2")
    evaluate_run = service_module.algorithm.evaluate_run

    def fail_second_run(run, **kwargs):
        if run["run_id"] == "run2":
            raise ValueError("历史重算测试中断")
        return evaluate_run(run, **kwargs)

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", fail_second_run)
    with pytest.raises(RuntimeError, match="历史重算未全部完成"):
        service.train()
    assert service.current_model["version"] != first["version"]
    assert service.current_model["reanalysis_status"] == "failed"
    assert service.latest_result("run1")["model_version"] == service.current_model["version"]
    assert service.latest_result("run2") is None
    assert service.latest_result("run2", first["version"]) == previous
    restored = SpindleMonitoringService(service.root, config)
    assert restored.latest_result("run2") is None
    assert restored.latest_result("run2", first["version"]) == previous


@pytest.mark.parametrize("previous_kind", ["missing", "legacy"])
def test_selection_refreshes_only_missing_or_legacy_result_once_without_changing_labels_or_reference(
        service, tmp_path, config, algorithm, previous_kind):
    service.import_batch([package(tmp_path, config, tuple(f"training{i}" for i in range(5)))], label="healthy")
    model = service.train()
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip")])
    service.set_label("daily", "healthy", "人工确认，不修改计算分数", include_in_training=True)
    previous = service.latest_result("daily")
    entry = service.state["results"][-1]
    if previous_kind == "missing":
        service.state["results"].remove(entry)
    else:
        previous.pop("score_kind")
        previous.update(score=4.0, analysis_score=1.4)
        entry.pop("score_kind")
        entry.update(score=4.0, analysis_score=1.4)
        service_module.write_document(service.root / entry["result_file"], previous)
    service._save()
    labels_before = deepcopy(service.runs)
    reference_before = deepcopy(model["normal_reference"])
    history_before = service.history()
    call_count = len(algorithm["evaluate"])

    refreshed = service.ensure_latest_result("daily")
    assert refreshed["reason"] == "selection_refresh"
    assert refreshed["score_kind"] == "normal_compatibility_v1"
    assert refreshed["score"] == 0.3  # 人工“正常”标签不强制相容度变成 1。
    assert refreshed["manual_label_at_evaluation"] == "healthy"
    assert refreshed["role"] == "independent"
    assert len(algorithm["evaluate"]) == call_count + 1
    assert algorithm["evaluate"][-1][0] == "daily"
    assert service.history()[:-1] == history_before
    assert service.runs == labels_before
    assert model["normal_reference"] == reference_before
    assert "daily" not in model["calibration_run_ids"] + model["training_run_ids"]

    assert service.ensure_latest_result("daily") == refreshed
    assert len(algorithm["evaluate"]) == call_count + 1
    assert len(service.history()) == len(history_before) + 1
    restored = SpindleMonitoringService(service.root, config)
    assert restored.ensure_latest_result("daily") == refreshed
    assert len(algorithm["evaluate"]) == call_count + 1
    assert restored.runs == labels_before
    assert restored.current_model["normal_reference"] == reference_before


def test_selection_returns_current_invalid_result_without_repeated_evaluation(service, tmp_path, config, algorithm):
    service.import_packages([package(tmp_path, config)])
    invalid = service.latest_result("run1")
    assert invalid["score_kind"] == "normal_compatibility_v1"
    assert invalid["score"] is None
    assert invalid["assessment"]["status"] == "unavailable"
    history_before = service.history()
    call_count = len(algorithm["evaluate"])
    assert service.ensure_latest_result("run1") == invalid
    assert service.ensure_latest_result("run1") == invalid
    assert len(algorithm["evaluate"]) == call_count
    assert service.history() == history_before


def test_switching_to_unconfirmed_daily_sample_uses_its_own_result_and_keeps_reference_frozen(
        service, tmp_path, config, algorithm):
    service.import_batch([package(tmp_path, config, tuple(f"training{i}" for i in range(5)))], label="healthy")
    model = service.train()
    reference_before = deepcopy(model["normal_reference"])
    service.import_packages([package(tmp_path, config, ("reviewed",), name="reviewed.zip")])
    service.set_label("reviewed", "healthy", "已人工复核")
    reviewed = service.evaluate("reviewed")
    assert reviewed["score"] == 0.3
    assert reviewed["manual_label_at_evaluation"] == "healthy"

    def different_signal(manifest, files):
        data_file = manifest["runs"][0]["data_file"]
        signal = json.loads(files[data_file])
        signal["score"] = 0.08
        files[data_file] = json.dumps(signal)

    service.import_packages([package(tmp_path, config, ("new_daily",), name="new_daily.zip", extra=different_signal)])
    calls_after_import = len(algorithm["evaluate"])
    assert service.ensure_latest_result("reviewed")["id"] == reviewed["id"]
    selected = service.ensure_latest_result("new_daily")
    assert selected["run_id"] == "new_daily"
    assert selected["id"] != reviewed["id"]
    assert selected["score"] == 0.08
    assert selected["manual_label_at_evaluation"] == "unconfirmed"
    assert selected["assessment"]["message"] == "偏离正常参考 · 未设置报警阈值"
    assert service.runs["new_daily"]["manual_label"] == "unconfirmed"
    assert not service.runs["new_daily"]["training_eligible"]
    assert model["normal_reference"] == reference_before
    assert "new_daily" not in model["calibration_run_ids"] + model["training_run_ids"]
    assert len(algorithm["evaluate"]) == calls_after_import


def test_retraining_keeps_threshold_values_but_requires_model_review(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    first = service.train()
    service.set_thresholds(warning=0.5, fault=0.3)
    old = service.evaluate("run1")
    assert old["assessment"]["status"] == "fault"
    assert old["thresholds_model_version"] == first["version"]
    assert not old["thresholds_review_required"]

    second = service.train()
    pending = service.latest_result("run1")
    assert service.settings["thresholds"] == {"warning": 0.5, "fault": 0.3}
    assert service.thresholds_review_required
    assert pending["score"] == 0.3
    assert pending["thresholds_model_version"] == first["version"]
    assert pending["thresholds_review_required"]
    assert pending["assessment"] == {"status": "review_required", "message": "新模型阈值待复核"}
    assert service.latest_result("run1", first["version"]) == old

    restored = SpindleMonitoringService(service.root, config)
    assert restored.thresholds_review_required
    restored.set_thresholds(warning=0.5, fault=0.3)
    assert not restored.thresholds_review_required
    confirmed = restored.evaluate("run1")
    assert confirmed["thresholds_model_version"] == second["version"]
    assert not confirmed["thresholds_review_required"]
    assert confirmed["assessment"]["status"] == "fault"
    assert restored.history("run1")[-2] == pending


def test_thresholds_set_before_model_need_review_after_training(service, tmp_path, config):
    service.set_thresholds(warning=0.5, fault=0.3)
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    assert service.latest_result("run1")["assessment"]["status"] == "unavailable"
    service.train()
    result = service.latest_result("run1")
    assert result["thresholds_review_required"]
    assert result["assessment"]["status"] == "review_required"


def test_duplicate_import_retries_only_missing_evaluations(service, tmp_path, config, monkeypatch):
    path = package(tmp_path, config, ("run1", "run2", "run3"))
    evaluate_run = service_module.algorithm.evaluate_run

    def interrupted(run, **kwargs):
        if run["run_id"] == "run2":
            raise ValueError("分析临时失败")
        return evaluate_run(run, **kwargs)

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", interrupted)
    with pytest.raises(ValueError, match="分析临时失败"):
        service.import_packages([path], purpose="initial")
    completed = service.latest_result("run1")
    assert completed is not None
    assert service.latest_result("run2") is None
    assert len(service.runs) == 3

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", evaluate_run)
    restored = SpindleMonitoringService(service.root, config)
    assert len(restored.import_packages([path], purpose="initial")) == 3
    assert len(restored.history()) == 3
    assert len(restored.state["packages"]) == 1
    assert restored.latest_result("run1") == completed
    assert restored.latest_result("run2") is not None
    assert restored.latest_result("run3") is not None
    restored.import_packages([path], purpose="initial")
    assert len(restored.history()) == 3


def test_failed_label_save_restores_memory_and_training_candidates(service, tmp_path, config, monkeypatch):
    service.import_packages([package(tmp_path, config)])
    record = service.runs["run1"]
    previous = deepcopy(record)

    def failed_save():
        raise OSError("写盘失败")

    monkeypatch.setattr(service, "_save", failed_save)
    with pytest.raises(OSError, match="写盘失败"):
        service.set_label("run1", "healthy", "不应残留", include_in_training=True)
    assert record == previous
    assert service.runs["run1"] is record
    assert service.training_candidates() == []
    assert SpindleMonitoringService(service.root, config).runs["run1"] == previous


def test_failed_threshold_confirmation_keeps_previous_model_binding(service, tmp_path, config, monkeypatch):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    service.set_thresholds(warning=0.5, fault=0.3)
    service.train()
    settings = service.settings
    previous = deepcopy(settings)
    assert service.thresholds_review_required

    def failed_save():
        raise OSError("写盘失败")

    monkeypatch.setattr(service, "_save", failed_save)
    with pytest.raises(OSError, match="写盘失败"):
        service.set_thresholds(warning=0.8, fault=0.6)
    assert service.settings is settings
    assert service.settings == previous
    assert service.thresholds_review_required
    assert SpindleMonitoringService(service.root, config).settings == previous


def test_history_retry_uses_existing_model_and_preserves_completed_results(service, tmp_path, config,
                                                                          algorithm, monkeypatch):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    evaluate_run = service_module.algorithm.evaluate_run

    def interrupted(run, **kwargs):
        if run["run_id"] == "run2":
            raise ValueError("分析临时失败")
        return evaluate_run(run, **kwargs)

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", interrupted)
    with pytest.raises(RuntimeError, match="历史重算未全部完成"):
        service.train()
    completed = service.latest_result("run1")
    version = service.current_model["version"]
    monkeypatch.setattr(service_module.algorithm, "evaluate_run", evaluate_run)
    restored = SpindleMonitoringService(service.root, config)
    progress = []
    model = restored.reanalyze_history(progress=lambda percent, message: progress.append(percent))
    assert model["version"] == version
    assert model["reanalysis_status"] == "complete"
    assert "reanalysis_error" not in model
    assert len(algorithm["train"]) == 1
    assert restored.latest_result("run1") == completed
    assert len(restored.history(model_version=version)) == 5
    assert progress[0] == 0 and progress[-1] == 100
    history = restored.history()
    restored.reanalyze_history()
    assert restored.history() == history
    assert len(algorithm["train"]) == 1
    model_file = restored.root / Path(model["model_path"]).parent / "model.json"
    assert json.loads(model_file.read_text(encoding="utf-8"))["reanalysis_status"] == "complete"


def test_history_retry_requires_model(service):
    with pytest.raises(ValueError, match="尚未建立模型"):
        service.reanalyze_history()


def test_failed_evaluation_index_save_does_not_hide_missing_result(service, tmp_path, config, monkeypatch):
    service.import_packages([package(tmp_path, config)])
    previous = deepcopy(service.state["results"])

    def failed_save():
        raise OSError("写盘失败")

    monkeypatch.setattr(service, "_save", failed_save)
    with pytest.raises(OSError, match="写盘失败"):
        service.evaluate("run1")
    assert service.state["results"] == previous
    assert SpindleMonitoringService(service.root, config).state["results"] == previous


def test_duplicate_import_fills_current_model_results_instead_of_accepting_old_results(
        service, tmp_path, config, monkeypatch):
    path = package(tmp_path, config, ("run1", "run2", "run3", "run4", "run5"))
    service.import_packages([path], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    evaluate_run = service_module.algorithm.evaluate_run

    def interrupted(run, **kwargs):
        if run["run_id"] == "run2":
            raise ValueError("分析临时失败")
        return evaluate_run(run, **kwargs)

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", interrupted)
    with pytest.raises(RuntimeError, match="历史重算未全部完成"):
        service.train()
    completed = service.latest_result("run1")
    assert service.latest_result("run2") is None
    monkeypatch.setattr(service_module.algorithm, "evaluate_run", evaluate_run)
    service.import_packages([path], purpose="daily")
    assert service.latest_result("run1") == completed
    assert service.latest_result("run2")["model_version"] == service.current_model["version"]
    assert len(service.history()) == 10
    assert all(run["purpose"] == "initial" for run in service.runs.values())
    service.import_packages([path], purpose="daily")
    assert len(service.history()) == 10
