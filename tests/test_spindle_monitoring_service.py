"""包边界、入集权限、分 run 校准及模型历史的业务回归。"""

from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from core.services import spindle_monitoring_service as service_module
from core.services.spindle_monitoring_service import SpindleMonitoringService, assess_score


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
        return {"healthy_reference_mse": 0.25, "best_epoch": 1}

    def evaluate_run(run, model_path=None, config=None, progress=None):
        if model_path is not None:
            assert Path(model_path).is_file()
        calls["evaluate"].append((run["run_id"], str(model_path)))
        return {
            "score": float(run.get("score", 3)) if model_path else None,
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
        files[directory + "/signals.h5"] = json.dumps({"run_id": run_id, "target_speed_rpm": 7000})
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
    with pytest.raises(ValueError, match="3 个"):
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
    first = package(tmp_path, config, ("one", "two", "three"), name="first.zip")
    second = package(tmp_path, config, ("other",), name="second.zip")
    service.import_batch([first], label="healthy")
    service.import_batch([second], label="abnormal")
    assert {run["run_id"] for run in service.training_candidates()} == {"one", "two", "three"}
    assert len({service.runs[key]["import_batch_id"] for key in ("one", "two", "three")}) == 1
    service.set_label("two", "abnormal")
    assert all(service.runs[key]["manual_label"] == "abnormal" for key in ("one", "two", "three"))
    assert not service.training_candidates()
    service.set_label("one", "unconfirmed", include_in_training=True)
    assert not service.training_candidates()
    assert all(not service.runs[key]["training_eligible"] for key in ("one", "two", "three"))
    service.set_label("three", "healthy")
    assert len(service.training_candidates()) == 3
    assert service.runs["other"]["manual_label"] == "abnormal"
    model = service.train()
    assert all(run["manual_label"] == "healthy" for run in model["training_samples"])
    assert set(algorithm["train"][0]["training"] + algorithm["train"][0]["validation"]) == {"one", "two", "three"}
    with pytest.raises(ValueError, match="不能修改人工判定"):
        service.set_label("two", "abnormal")
    assert len(service.training_candidates()) == 3


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
    with pytest.raises(ValueError, match="3 个"):
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
    with pytest.raises(ValueError, match="3 个"):
        service.train()


def test_training_split_versions_and_history_reanalysis(service, tmp_path, config, algorithm):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip", condition="unbalance1")])
    service.set_label("daily", "abnormal", "检查轴承")
    first = service.train()
    assert first["version"].startswith("tcn_1s_2ep_")
    assert first["model_path"] == f"models/{first['version']}/model.pt"
    call = algorithm["train"][0]
    assert len(call["training"]) == 2 and len(call["validation"]) == 1
    assert set(call["training"]).isdisjoint(call["validation"])
    assert "daily" not in call["training"] + call["validation"]
    assert len(service.history(model_version=first["version"])) == 4
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
    assert len(service.history(model_version=second["version"])) == 4
    assert len(service.history()) == 12
    assert service.runs["daily"]["manual_label"] == "abnormal"
    restored = SpindleMonitoringService(service.root, config)
    assert restored.current_model["version"] == second["version"]
    assert restored.current_model["reanalysis_status"] == "complete"
    assert len(restored.history()) == 12


def test_analysis_ratio_uses_model_normal_mean_and_legacy_history_without_contamination(
        service, tmp_path, config, monkeypatch):
    evaluate = service_module.algorithm.evaluate_run
    rms_values = {"run1": 1.0, "run2": 2.0, "run3": 3.0, "daily": 100.0}

    def evaluate_with_rms(run, **kwargs):
        result = evaluate(run, **kwargs)
        result["xy_rms_mm_s"] = rms_values[run["run_id"]]
        return result

    monkeypatch.setattr(service_module.algorithm, "evaluate_run", evaluate_with_rms)
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], "initial")
    service.set_label("run1", "healthy")
    service.import_packages([package(tmp_path, config, ("daily",), name="daily.zip")])
    first = service.train()
    assert service.analysis_reference(first["version"]) == 2
    result = service.latest_result("daily")
    assert result["analysis_score"] == 50
    assert result["analysis_reference_rms_mm_s"] == 2
    original_history = deepcopy(service.history(model_version=first["version"]))
    service.set_label("daily", "healthy")
    assert service.analysis_metrics(result) == (50, 2)
    assert service.evaluate("daily")["analysis_score"] == 50
    second = service.train()
    assert service.analysis_reference(second["version"]) == 26.5
    assert service.latest_result("daily")["analysis_score"] == pytest.approx(100 / 26.5)
    restored = SpindleMonitoringService(service.root, config)
    assert restored.analysis_reference(first["version"]) == 2
    assert restored.analysis_metrics(result) == (50, 2)
    assert restored.history(model_version=first["version"])[:len(original_history)] == original_history


def test_initial_import_is_closed_after_first_model(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    path = package(tmp_path, config, ("run4",), name="late.zip")
    with pytest.raises(ValueError, match="日常导入"):
        service.import_packages([path], purpose="initial")
    service.import_packages([path])
    result = service.latest_result("run4")
    assert result["score"] == 3
    assert result["reason"] == "import"
    assert result["model_version"] == service.current_model["version"]
    assert service.runs["run4"]["manual_label"] == "unconfirmed"


def test_trained_and_calibration_labels_stay_locked_after_restart(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
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
    service.import_packages([package(tmp_path, config, ("one", "two", "three"))], "initial")
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
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    old = service.latest_result("run1")
    assert old["assessment"]["status"] == "unconfigured"
    service.set_thresholds(warning=2, fault=3)
    new = service.evaluate("run1")
    assert new["assessment"]["status"] == "fault"
    assert "建议检修" in new["assessment"]["message"]
    assert service.history("run1")[-2] == old
    assert old["thresholds"] == {"warning": None, "fault": None}


@pytest.mark.parametrize(("score", "thresholds", "status"), [
    (None, {"warning": 2, "fault": 5}, "unavailable"),
    (1, {"warning": None, "fault": None}, "unconfigured"),
    (1, {"warning": 2, "fault": 5}, "normal"),
    (2, {"warning": 2, "fault": 5}, "warning"),
    (5, {"warning": 2, "fault": 5}, "fault"),
    (5, {"warning": None, "fault": 5}, "fault"),
])
def test_threshold_priority(score, thresholds, status):
    assert assess_score(score, thresholds)["status"] == status


@pytest.mark.parametrize(("warning", "fault"), [(3, 2), (3, 3), (-1, None), (float("nan"), None)])
def test_invalid_thresholds_do_not_change_settings(service, warning, fault):
    with pytest.raises(ValueError):
        service.set_thresholds(warning, fault)
    assert service.settings["thresholds"] == {"warning": None, "fault": None}


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
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
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


def test_retraining_keeps_threshold_values_but_requires_model_review(service, tmp_path, config):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    first = service.train()
    service.set_thresholds(warning=2, fault=3)
    old = service.evaluate("run1")
    assert old["assessment"]["status"] == "fault"
    assert old["thresholds_model_version"] == first["version"]
    assert not old["thresholds_review_required"]

    second = service.train()
    pending = service.latest_result("run1")
    assert service.settings["thresholds"] == {"warning": 2, "fault": 3}
    assert service.thresholds_review_required
    assert pending["score"] == 3
    assert pending["thresholds_model_version"] == first["version"]
    assert pending["thresholds_review_required"]
    assert pending["assessment"] == {"status": "review_required", "message": "新模型阈值待复核"}
    assert service.latest_result("run1", first["version"]) == old

    restored = SpindleMonitoringService(service.root, config)
    assert restored.thresholds_review_required
    restored.set_thresholds(warning=2, fault=3)
    assert not restored.thresholds_review_required
    confirmed = restored.evaluate("run1")
    assert confirmed["thresholds_model_version"] == second["version"]
    assert not confirmed["thresholds_review_required"]
    assert confirmed["assessment"]["status"] == "fault"
    assert restored.history("run1")[-2] == pending


def test_thresholds_set_before_model_need_review_after_training(service, tmp_path, config):
    service.set_thresholds(warning=2, fault=3)
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
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
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
    service.set_label("run1", "healthy", "整批确认正常")
    service.train()
    service.set_thresholds(warning=2, fault=3)
    service.train()
    settings = service.settings
    previous = deepcopy(settings)
    assert service.thresholds_review_required

    def failed_save():
        raise OSError("写盘失败")

    monkeypatch.setattr(service, "_save", failed_save)
    with pytest.raises(OSError, match="写盘失败"):
        service.set_thresholds(warning=4, fault=5)
    assert service.settings is settings
    assert service.settings == previous
    assert service.thresholds_review_required
    assert SpindleMonitoringService(service.root, config).settings == previous


def test_history_retry_uses_existing_model_and_preserves_completed_results(service, tmp_path, config,
                                                                          algorithm, monkeypatch):
    service.import_packages([package(tmp_path, config, ("run1", "run2", "run3"))], purpose="initial")
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
    assert len(restored.history(model_version=version)) == 3
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
    path = package(tmp_path, config, ("run1", "run2", "run3"))
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
    assert len(service.history()) == 6
    assert all(run["purpose"] == "initial" for run in service.runs.values())
    service.import_packages([path], purpose="daily")
    assert len(service.history()) == 6
