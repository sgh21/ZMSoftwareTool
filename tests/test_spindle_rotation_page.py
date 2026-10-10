"""主轴页面回归：空启动、后台导入、人工标签与历史模型隔离。"""

from copy import deepcopy
import json
from math import exp, sqrt
from pathlib import Path
from threading import Event, get_ident
from time import monotonic
from zipfile import ZipFile

import h5py
import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QInputDialog, QLabel, QMessageBox,
    QPushButton, QTableWidget, QLineEdit, QDialogButtonBox,
)

from app.pages.spindle_rotation_page import SpindleRotationPage
from app.resources import DISPLAY, UiScale, load_stylesheet
from core.algorithms import spindle_monitoring as algorithm
from core.services.spindle_monitoring_service import SpindleMonitoringService
from test_spindle_algorithms import make_run
from test_spindle_monitoring_service import h5_package


pytestmark = pytest.mark.usefixtures("styled_application")


@pytest.fixture
def service(tmp_path):
    return SpindleMonitoringService(tmp_path / "spindle_store")


@pytest.fixture
def page_factory(application):
    pages = []

    def create(service):
        page = SpindleRotationPage(service)
        page.ui_scale = 1.0
        page.resize(1500, 1000)
        page.show()
        QApplication.processEvents()
        pages.append(page)
        return page

    yield create
    for page in pages:
        wait_until(lambda: page.task is None and not page.selection_timer.isActive())
        wait_until(lambda: page.energy_task is None)
        page.close()
        page.deleteLater()
    QApplication.processEvents()


def wait_until(condition, timeout=5):
    deadline = monotonic() + timeout
    while not condition() and monotonic() < deadline:
        QTest.qWait(10)
    assert condition(), "主轴页面异步任务没有在限时内完成"
    QApplication.processEvents()


def data_package(tmp_path, service, run_ids, name="samples.zip"):
    """使用小型真实 H5/CSV 输入；不读取主应用的数据或模型。"""
    run = make_run()
    h5_path = tmp_path / "ui_signals.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs.update({
            "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
            "velocity_frequency_band_hz": [10, 900],
            "velocity_channels": ["ACC1", "ACC2", "ACC3"],
            "run_name": "ui_fixture", "run_date": "20260929", "target_speed_rpm": 7000,
        })
        group = h5.create_group("windows_10s")
        for key in ("velocity", "start_time_s", "temperature", "temperature_valid",
                    "sample_valid", "source_segment_index"):
            group[key] = run[key]
    manifest = {
        "schema_version": 1, "package_id": name, "captured_date": "2026-09-29",
        "source_type": "historical_replay",
        "preprocessing": deepcopy(service.config["preprocessing"]), "runs": [],
    }
    package = tmp_path / name
    with ZipFile(package, "w") as archive:
        for index, run_id in enumerate(run_ids):
            folder = f"runs/{run_id}"
            record = {
                "run_id": run_id, "run_name": f"7000rpm_{run_id}",
                "captured_at": f"2026-09-29T12:{index:02d}:00+08:00",
                "speed_rpm": 7000, "operation": "idle",
                "condition": {"tool_remounted": True, "seal_pressure_mpa": 0.2},
                "data_file": folder + "/signals.h5",
                "telemetry_file": folder + "/telemetry.csv",
                "source_metadata_file": folder + "/metadata.json",
            }
            manifest["runs"].append(record)
            with h5py.File(h5_path, "r+") as h5:
                h5["windows_10s/velocity"][:] = run["velocity"] * (1 + .05 * index)
            archive.write(h5_path, record["data_file"])
            archive.writestr(record["telemetry_file"],
                             "time_s,actual_speed_rpm,current_a\n610,7000,0.4\n611,7000,0.6\n")
            archive.writestr(record["source_metadata_file"], "{}")
        archive.writestr("manifest.json", json.dumps(manifest))
    return package


@pytest.mark.parametrize(("label", "stored", "eligible"), [("正常", "healthy", True), ("异常", "abnormal", False), ("未判定", "unconfirmed", False)])
def test_batch_button_selects_one_outer_zip_and_reuses_progress(service, tmp_path, page_factory, monkeypatch, label, stored, eligible):
    sample = h5_package(tmp_path)
    outer = tmp_path / "batch.zip"
    with ZipFile(outer, "w") as archive:
        archive.write(sample, "23/工况/采集.zip")
        archive.writestr("24/损坏.zip", b"invalid")
    selected = []

    def choose(*args):
        selected.append(args[1])
        return str(outer), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: (label, True))
    page = page_factory(service)
    original_bar = page.progress
    QTest.mouseClick(page.initial_button, Qt.MouseButton.LeftButton)
    assert page.task is not None
    assert not page.initial_button.isEnabled()
    assert page.status_lights["acquisition"].property("state") == "partial"
    wait_until(lambda: page.task is None)
    assert len(selected) == 1
    assert page.run_select.currentData() == "run1"
    assert page.result["window_count"] == 10
    assert page.progress is original_bar and page.progress.value() == 100
    assert (page.progress.minimum(), page.progress.maximum()) == (0, 100)
    assert "新增 1 条采集，重复 0 条，失败 1 个包" in page.task_progress_note.text()
    assert "batch.zip!/24/损坏.zip" in page.process_log.toPlainText()
    assert page.status_lights["acquisition"].property("state") == "ready"
    assert page.initial_button.isEnabled()
    assert service.runs["run1"]["manual_label"] == stored
    assert service.runs["run1"]["training_eligible"] == eligible


def test_cancel_batch_label_does_not_start_import(service, page_factory, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("samples.zip", ""))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("正常", False))
    page = page_factory(service)
    page.initial_button.click()
    assert page.task is None and not service.runs


@pytest.fixture
def model_service(service, tmp_path, monkeypatch):
    """模型输出由固定桩提供，存储、导入、DSP 与 UI 仍使用真实代码。"""
    model_number = 0
    evaluate = algorithm.evaluate_run

    def train(training, validation, output_path, config, progress=None):
        nonlocal model_number
        model_number += 1
        Path(output_path).write_text(str(model_number), encoding="utf-8")
        return {"normal_reference": {"groups": []}, "score_kind": algorithm.SCORE_KIND}

    def infer(run, model_path=None, config=None, progress=None):
        result = evaluate(run, config=config, progress=progress)
        if model_path:
            from scipy.stats import t

            number = float(Path(model_path).read_text(encoding="utf-8"))
            alpha = config["alpha"]
            rms = result["xy_rms_mm_s"]
            vibration_reference = algorithm.fit_normal_reference([rms * .9, rms, rms * 1.1])
            network_reference = algorithm.fit_normal_reference([.18, .2, .22])
            error = exp(network_reference["mu"] + t.isf(.04 / number, 2) * network_reference["s"] * sqrt(1 + 1 / 3))
            vibration = algorithm.normal_compatibility(rms, vibration_reference, alpha, two_sided=True)
            network = algorithm.normal_compatibility(error, network_reference, alpha)
            result.update({
                "score": network["score"], "analysis_score": vibration["score"],
                "score_kind": algorithm.SCORE_KIND, "alpha": alpha,
                "network_p_value": network["p_value"], "vibration_p_value": vibration["p_value"],
                "reconstruction_error_p95": error, "window_errors": [error] * result["window_count"],
                "reference_features": {"network": network_reference["values"], "vibration": vibration_reference["values"]},
                "compatibility": {"network": network, "vibration": vibration},
                "reconstruction": {
                    "time_s": result["waveform"]["time_s"],
                    "original": result["waveform"]["values"],
                    "reconstructed": [[value * 0.8 for value in axis]
                                      for axis in result["waveform"]["values"]],
                },
            })
        return result

    monkeypatch.setattr(algorithm, "train_model", train)
    monkeypatch.setattr(algorithm, "evaluate_run", infer)
    service.import_packages([data_package(tmp_path, service, ("initial1", "initial2", "initial3", "initial4", "initial5"))], "initial")
    service.set_label("initial1", "healthy", "整批确认正常")
    service.import_packages([data_package(tmp_path, service, ("daily",), "daily.zip")])
    service.set_label("daily", "abnormal", "人工确认异常")
    service.train()
    service.train()
    return service


def test_empty_start_has_no_fabricated_model_score_or_curves(service, page_factory):
    from ui_helpers import assert_disabled_tooltip

    page = page_factory(service)
    assert service.current_model is None
    assert service.runs == {}
    assert service.state["results"] == []
    assert page.result is None
    assert all(value.text() == "—" for value in page.result_values.values())
    assert all(plot.series == [] for plot in page.plots.values())
    assert page.initial_button.isEnabled()
    assert not page.train_button.isEnabled()
    assert not page.evaluate_button.isEnabled()
    assert page.evaluate_button.text() == "评估精度"
    assert "导入" in page.evaluate_button.toolTip()
    assert_disabled_tooltip(page.evaluate_button)
    assert not page.label_button.isEnabled()
    assert "示意" not in page.source_badge.text()
    assert all(light.property("state") == "missing" for light in page.status_lights.values())
    assert page.process_log.minimumHeight() == 110
    assert page.process_log.placeholderText() == "操作过程、提示与错误将在这里显示"
    assert page.progress.isVisible() and page.progress.value() == 0


def test_status_lights_follow_model_history_and_threshold_review(model_service, page_factory):
    page = page_factory(model_service)
    assert page.status_lights["acquisition"].property("state") == "ready"
    assert page.status_lights["model"].property("state") == "ready"
    assert page.status_lights["thresholds"].property("state") == "missing"
    assert "7000 r/min" in page.status_lights["acquisition"].toolTip()
    assert model_service.current_model["version"] in page.status_lights["model"].toolTip()

    model_service.set_thresholds(.5, .2)
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "ready"
    model_service.settings["thresholds_model_version"] = "old-model"
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "partial"
    assert "新模型阈值待复核" in page.status_lights["thresholds"].toolTip()
    model_service.set_thresholds(.5, None)
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "partial"
    model_service.current_model["reanalysis_status"] = "failed"
    page._refresh()
    assert page.status_lights["model"].property("state") == "partial"
    assert "补算" in page.status_lights["model"].toolTip()


def test_narrow_layout_keeps_log_and_progress_separate_after_resizing(service, page_factory):
    page = page_factory(service)
    metrics = UiScale(page)
    try:
        for width, height, scale in ((640, 400, .75), (1280, 800, 1), (640, 400, .75)):
            page.resize(width, height)
            metrics.apply(scale)
            QApplication.instance().setStyleSheet(load_stylesheet(scale))
            QTest.qWait(40)
            assert page.process_log.geometry().bottom() < page.progress.geometry().top()
            assert page.process_log.parentWidget().rect().contains(page.process_log.geometry())
            assert page.progress.geometry().bottom() < page.task_progress_note.geometry().top()
    finally:
        QApplication.instance().setStyleSheet(load_stylesheet())


def test_model_work_lights_only_model_group_and_finishes_progress_after_refresh(model_service, page_factory):
    model_service.set_thresholds(.5, .2)
    page = page_factory(model_service)
    release = Event()

    def operation(progress):
        for epoch in range(30):
            progress(epoch, f"训练 {epoch + 1}/30，训练误差 0.001")
        progress(100, "训练完成，准备刷新")
        assert release.wait(5)

    try:
        page._run_task("模型任务", operation, status_key="model")
        wait_until(lambda: page.progress.value() == 99)
        assert page.status_lights["model"].property("state") == "partial"
        assert page.status_lights["acquisition"].property("state") == "ready"
        assert page.status_lights["thresholds"].property("state") == "ready"
        assert page.task_progress_note.text() == "训练完成，准备刷新"
        assert page.process_log.document().blockCount() == 1
        assert "训练误差" not in page.process_log.toPlainText()
    finally:
        release.set()
        wait_until(lambda: page.task is None)
    assert page.progress.value() == 100
    assert page.status_lights["model"].property("state") == "ready"
    assert page.process_log.document().blockCount() == 2
    assert "处理完成" in page.process_log.toPlainText()


def test_import_runs_in_worker_and_updates_real_dsp_without_blocking_gui(
        service, tmp_path, page_factory, monkeypatch):
    package = data_package(tmp_path, service, ("daily",))
    page = page_factory(service)
    entered, release = Event(), Event()
    original_import = service.import_packages
    worker_threads = []

    def import_with_gate(paths, purpose, progress):
        worker_threads.append(get_ident())
        progress(10, "准备读取实际 H5")
        entered.set()
        if not release.wait(4):
            raise RuntimeError("测试导入门未释放")
        return original_import(paths, purpose, progress)

    monkeypatch.setattr(service, "import_packages", import_with_gate)
    ticks = []
    timer = QTimer(page)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(monotonic()))
    timer.start()
    try:
        page.import_packages([package], "daily")
        wait_until(entered.is_set)
        wait_until(lambda: len(ticks) >= 3)
        assert page.task is not None
        assert len(worker_threads) == 1
        assert worker_threads[0] != get_ident()
        assert not page.daily_button.isEnabled()
        assert not page.run_select.isEnabled()
        assert page.progress.isVisible()
        assert page.status_lights["acquisition"].property("state") == "partial"
    finally:
        release.set()
        wait_until(lambda: page.task is None)
        timer.stop()
    assert page.status_lights["acquisition"].property("state") == "ready"
    assert page.status_lights["model"].property("state") == "partial"
    assert page.progress.value() == 100
    assert page.task_progress_note.text() == "处理完成"
    assert page.run_select.currentData() == "daily"
    assert "2026-09-29" in page.run_select.currentText()
    assert page.result["run_id"] == "daily"
    assert page.result_values["network"].text() == "—"
    assert page.result_values["analysis"].text() == "—"
    assert "本次径向 RMS" in page.result_values["analysis"].toolTip()
    assert page.result_values["current"].text() == "0.5"
    assert page.plots["waveform"].series
    assert page.plots["spectrum"].series
    assert page.plots["distribution"].series == []
    assert page.daily_button.isEnabled()
    assert page.task is None


def test_invalid_zip_recovers_controls_without_stretching_window(service, tmp_path, page_factory, monkeypatch):
    package = tmp_path / ("invalid_input_" * 5 + ".zip")
    package.write_text("not a ZIP", encoding="utf-8")
    page = page_factory(service)
    page.resize(1100, 800)
    QApplication.processEvents()
    width = page.width()
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    page.import_packages([package])
    wait_until(lambda: page.task is None)
    QTest.qWait(30)
    assert errors and not service.runs
    assert page.daily_button.isEnabled()
    assert page.width() == width
    assert "处理失败" in page.task_progress_note.text()
    assert page.progress.toolTip() == page.task_progress_note.text()


def test_daily_training_requires_manual_healthy_label_and_opt_in(service, tmp_path, page_factory):
    service.import_packages([data_package(tmp_path, service, ("daily",))])
    page = page_factory(service)
    original_result = deepcopy(service.latest_result("daily"))
    assert page.label_select.currentData() == "unconfirmed"
    assert not page.training_check.isEnabled()
    assert not page.training_check.isChecked()
    page.label_select.setCurrentIndex(page.label_select.findData("healthy"))
    assert page.training_check.isEnabled()
    page.training_check.setChecked(False)
    page._save_label()
    assert service.training_candidates() == []
    page.training_check.setChecked(True)
    page.label_note.setText("人工检查正常")
    page._save_label()
    assert [run["run_id"] for run in service.training_candidates()] == ["daily"]
    assert service.runs["daily"]["label_note"] == "人工检查正常"
    page.label_select.setCurrentIndex(page.label_select.findData("abnormal"))
    assert not page.training_check.isChecked()
    assert not page.training_check.isEnabled()
    page._save_label()
    assert service.training_candidates() == []
    assert service.runs["daily"]["manual_label"] == "abnormal"
    assert service.latest_result("daily") == original_result


def test_initial_data_requires_healthy_batch_label_before_training(service, tmp_path, page_factory):
    service.import_packages([data_package(tmp_path, service, ("initial1", "initial2", "initial3", "initial4", "initial5"))], "initial")
    page = page_factory(service)
    assert not page.train_button.isEnabled()
    assert not service.training_candidates()
    assert not page.training_check.isChecked()
    assert page.label_select.currentData() == "unconfirmed"
    assert all(run["manual_label"] == "unconfirmed" and not run["label_history"]
               for run in service.runs.values())

    def confirm():
        dialog = QApplication.activeModalWidget()
        assert len(dialog.findChild(QComboBox, "spindle_review_package").currentData()) == 5
        page.label_select.setCurrentIndex(page.label_select.findData("healthy"))
        assert page.training_check.isChecked()
        assert "5 条" in page.label_button.text()
        page.label_button.click()

    QTimer.singleShot(30, confirm)
    page._show_review()
    assert all(run["manual_label"] == "healthy" for run in service.runs.values())
    assert len(service.training_candidates()) == 5 and page.train_button.isEnabled()

    def reject_batch():
        page.label_select.setCurrentIndex(page.label_select.findData("abnormal"))
        assert not page.training_check.isEnabled() and not page.training_check.isChecked()
        page.label_button.click()

    QTimer.singleShot(30, reject_batch)
    page._show_review()
    assert all(run["manual_label"] == "abnormal" for run in service.runs.values())
    assert not service.training_candidates() and not page.train_button.isEnabled()


def test_run_and_model_selection_show_matching_results_and_preserve_labels(model_service, page_factory):
    service = model_service
    page = page_factory(service)
    first, second = service.models
    originals = deepcopy(service.state["results"])
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    for model, expected in ((first, "0.8"), (second, "0.4")):
        page.model_select.setCurrentIndex(page.model_select.findData(model["version"]))
        assert page.result["run_id"] == "daily"
        assert page.result["model_version"] == model["version"]
        assert page.result_values["network"].text() == expected
        assert page.label_select.currentData() == "abnormal"
        assert not page.training_check.isChecked()
    page.run_select.setCurrentIndex(page.run_select.findData("initial2"))
    assert page.result["run_id"] == "initial2"
    assert page.result["model_version"] == second["version"]
    assert page.label_select.currentData() == "healthy"
    assert service.state["results"] == originals
    assert not page.initial_button.isEnabled()


def test_selecting_current_samples_refreshes_both_scores_without_inference(model_service, page_factory, monkeypatch):
    from core.services.position_persistence import write_document

    result = model_service.latest_result("daily")
    result.update(vibration_p_value=.03, network_p_value=.035)
    entry = next(item for item in model_service.state["results"] if item["id"] == result["id"])
    write_document(model_service.root / entry["result_file"], result)
    calls = []
    monkeypatch.setattr(model_service, "ensure_latest_result", lambda *args: calls.append(args))
    page = page_factory(model_service)
    original = deepcopy(model_service.state["results"])
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    assert page.result_values["analysis"].text() == "0.6"
    assert page.result_values["network"].text() == "0.7"
    assert "当前设置" in page.current_run_button.text()
    assert page.result_values["network"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["warning"]
    assert "偏离" in page.source_badge.text()
    page.run_select.setCurrentIndex(page.run_select.findData("initial1"))
    assert page.result_values["analysis"].text() == "1"
    assert page.result_values["network"].text() == "0.4"
    QApplication.processEvents()
    assert not calls and model_service.state["results"] == original
    result["compatibility"]["vibration"].update(status="invalid", message="正常参考无效")
    write_document(model_service.root / entry["result_file"], result)
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    QApplication.processEvents()
    assert page.result_values["analysis"].text() == "—"
    assert not calls


@pytest.mark.parametrize("stale", ["missing", "legacy"])
@pytest.mark.parametrize("first_show", [False, True])
def test_current_selection_automatically_updates_missing_or_legacy_evaluation_once(
        model_service, page_factory, monkeypatch, stale, first_show):
    from core.services.position_persistence import write_document

    page = None if first_show else page_factory(model_service)
    target = model_service.list_runs()[-1]["run_id"] if first_show else "daily"
    version = model_service.current_model["version"]
    result = model_service.latest_result(target)
    entry = next(item for item in model_service.state["results"] if item["id"] == result["id"])
    if stale == "missing":
        model_service.state["results"].remove(entry)
    else:
        result["score_kind"] = entry["score_kind"] = "legacy_ratio"
        result["score"] = 3.0
        write_document(model_service.root / entry["result_file"], result)
    count = len(model_service.state["results"])
    ensure = model_service.ensure_latest_result
    entered, release = Event(), Event()
    calls = []

    def gated(run_id, progress=None):
        calls.append((run_id, get_ident()))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("自动补算测试门未释放")
        return ensure(run_id, progress)

    monkeypatch.setattr(model_service, "ensure_latest_result", gated)
    try:
        if first_show:
            page = page_factory(model_service)
        else:
            page.run_select.setCurrentIndex(page.run_select.findData(target))
        page.run_select.activated.emit(page.run_select.currentIndex())
        page.run_select.activated.emit(page.run_select.currentIndex())
        wait_until(entered.is_set)
        assert page.task is not None
        assert page.result_values["network"].text() == "—"
    finally:
        release.set()
        if page is not None:
            wait_until(lambda: page.task is None)
    assert len(calls) == 1 and calls[0][0] == target and calls[0][1] != get_ident()
    assert page.result["model_version"] == version
    assert page.result["score_kind"] == algorithm.SCORE_KIND
    assert page.result_values["analysis"].text() == "1"
    assert page.result_values["network"].text() == "0.4"
    assert len(model_service.state["results"]) == count + 1


def test_failed_automatic_update_does_not_retry_during_refresh(model_service, page_factory, monkeypatch):
    target = model_service.list_runs()[-1]["run_id"]
    result = model_service.latest_result(target)
    model_service.state["results"][:] = [item for item in model_service.state["results"] if item["id"] != result["id"]]
    calls, errors = [], []

    def fail(run_id, progress=None):
        calls.append(run_id)
        raise OSError("无法读取采集")

    monkeypatch.setattr(model_service, "ensure_latest_result", fail)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    page = page_factory(model_service)
    wait_until(lambda: bool(errors) and page.task is None)
    page._refresh()
    page.run_select.activated.emit(page.run_select.currentIndex())
    QApplication.processEvents()
    assert calls == [target]
    assert page.task is None and page.daily_button.isEnabled()
    assert page.result_values["network"].text() == "—"


def test_historical_selection_never_starts_automatic_evaluation(model_service, page_factory, monkeypatch):
    from core.services.position_persistence import write_document

    page = page_factory(model_service)
    original = model_service.latest_result("daily", model_service.models[0]["version"])
    entry = next(item for item in model_service.state["results"] if item["id"] == original["id"])
    original["score_kind"] = entry["score_kind"] = "legacy_ratio"
    write_document(model_service.root / entry["result_file"], original)
    calls = []
    monkeypatch.setattr(model_service, "ensure_latest_result", lambda *args: calls.append(args))
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    page.model_select.setCurrentIndex(page.model_select.findData(original["model_version"]))
    page.run_select.activated.emit(page.run_select.currentIndex())
    QApplication.processEvents()
    assert not calls
    assert page.result["id"] == original["id"]
    assert page.result_values["network"].text() == "—"
    assert "历史设置" in page.current_run_button.text()


@pytest.mark.parametrize("history", [False, True])
def test_automatic_update_completion_preserves_the_current_selection(model_service, page_factory, monkeypatch, history):
    page = page_factory(model_service)
    result = model_service.latest_result("daily")
    model_service.state["results"][:] = [item for item in model_service.state["results"] if item["id"] != result["id"]]
    ensure = model_service.ensure_latest_result
    entered, release = Event(), Event()

    def gated(run_id, progress=None):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("自动补算测试门未释放")
        return ensure(run_id, progress)

    monkeypatch.setattr(model_service, "ensure_latest_result", gated)
    try:
        page.run_select.setCurrentIndex(page.run_select.findData("daily"))
        wait_until(entered.is_set)
        page.run_select.setCurrentIndex(page.run_select.findData("initial1"))
        if history:
            page.model_select.setCurrentIndex(page.model_select.findData(model_service.models[0]["version"]))
    finally:
        release.set()
        wait_until(lambda: page.task is None)
    assert page.result["run_id"] == "initial1"
    assert page.result["model_version"] == model_service.models[0 if history else -1]["version"]
    assert page.result_values["network"].text() == ("0.8" if history else "0.4")
    assert page.uses_current_thresholds is not history


def test_automatic_update_preserves_exact_historical_record_with_same_model(model_service, page_factory, monkeypatch):
    historical = model_service.latest_result("initial1")
    newer = model_service.evaluate("initial1")
    assert historical["id"] != newer["id"]
    page = page_factory(model_service)
    removed = model_service.latest_result("daily")
    model_service.state["results"][:] = [item for item in model_service.state["results"] if item["id"] != removed["id"]]
    ensure = model_service.ensure_latest_result
    entered, release = Event(), Event()

    def gated(run_id, progress=None):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("历史竞态测试门未释放")
        return ensure(run_id, progress)

    monkeypatch.setattr(model_service, "ensure_latest_result", gated)
    try:
        page.run_select.setCurrentIndex(page.run_select.findData("daily"))
        wait_until(entered.is_set)
        entries = list(reversed(model_service.state["results"]))
        row = next(index for index, item in enumerate(entries) if item["id"] == historical["id"])

        def open_history():
            QApplication.activeModalWidget().findChild(QTableWidget).cellActivated.emit(row, 0)

        QTimer.singleShot(20, open_history)
        page._show_history()
        assert page.result["id"] == historical["id"]
    finally:
        release.set()
        wait_until(lambda: page.task is None)
    assert page.result["id"] == historical["id"]
    assert not page.uses_current_thresholds
    assert "历史设置" in page.current_run_button.text()


def test_closing_review_keeps_result_completed_while_review_was_open(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)
    removed = model_service.latest_result("daily")
    model_service.state["results"][:] = [item for item in model_service.state["results"] if item["id"] != removed["id"]]
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    assert page.result is None

    def review(dialog):
        wait_until(lambda: page.result is not None and page.task is None)
        assert page.result["score_kind"] == algorithm.SCORE_KIND
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", review)
    page._show_review()
    assert page.result["id"] == model_service.latest_result("daily")["id"]
    assert page.result_values["network"].text() == "0.4"
    assert page.uses_current_thresholds


def test_sample_dialog_actions_follow_automatic_evaluation_busy_state(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)
    removed = model_service.latest_result("daily")
    model_service.state["results"][:] = [item for item in model_service.state["results"] if item["id"] != removed["id"]]
    model_service.current_model["reanalysis_status"] = "failed"
    ensure = model_service.ensure_latest_result
    entered, release = Event(), Event()

    def gated(run_id, progress=None):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("样本窗口测试门未释放")
        return ensure(run_id, progress)

    monkeypatch.setattr(model_service, "ensure_latest_result", gated)

    def inspect(dialog):
        review = next(button for button in dialog.findChildren(QPushButton) if button.text() == "人工判定")
        retry = next(button for button in dialog.findChildren(QPushButton) if button.text() == "补算缺失评价")
        assert retry.isEnabled()
        try:
            page.run_select.setCurrentIndex(page.run_select.findData("daily"))
            wait_until(entered.is_set)
            assert not review.isEnabled() and not retry.isEnabled()
            retry.click()
        finally:
            release.set()
            wait_until(lambda: page.task is None)
        assert review.isEnabled() and retry.isEnabled()
        assert model_service.current_model["reanalysis_status"] == "failed"
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    page._show_samples()
    assert page.result["id"] == model_service.latest_result("daily")["id"]


def test_selecting_model_without_run_evaluation_clears_scores_and_plots(
        model_service, tmp_path, page_factory):
    service = model_service
    service.import_packages([data_package(tmp_path, service, ("late",), "late.zip")])
    page = page_factory(service)
    page.run_select.setCurrentIndex(page.run_select.findData("late"))
    assert page.result["model_version"] == service.models[-1]["version"]
    page.model_select.setCurrentIndex(page.model_select.findData(service.models[0]["version"]))
    assert page.result is None
    assert all(value.text() == "—" for value in page.result_values.values())
    assert all(plot.series == [] for plot in page.plots.values())
    assert service.runs["late"]["manual_label"] == "unconfirmed"


def test_history_opens_exact_saved_evaluation_without_changing_manual_label(model_service, page_factory):
    service = model_service
    original = service.latest_result("daily", service.models[0]["version"])
    service.set_thresholds(warning=0.8, fault=0.5)
    service.evaluate("daily")
    page = page_factory(service)
    entries = list(reversed(service.state["results"]))
    row = next(index for index, item in enumerate(entries) if item["id"] == original["id"])
    callbacks = []

    def choose_record():
        dialog = QApplication.activeModalWidget()
        table = dialog.findChild(QTableWidget)
        callbacks.append(table.rowCount())
        table.cellActivated.emit(row, 0)

    QTimer.singleShot(30, choose_record)
    page._show_history()
    assert callbacks == [len(entries)]
    assert page.run_select.currentData() == "daily"
    assert page.model_select.currentData() == original["model_version"]
    assert page.result["id"] == original["id"]
    assert page.result["thresholds"] == {"warning": None, "fault": None}
    assert "未设置报警阈值" in page.source_badge.text()
    assert "当时阈值" in page.source_badge.toolTip()
    assert page.label_select.currentData() == "abnormal"
    assert service.runs["daily"]["manual_label"] == "abnormal"
    assert len(service.state["results"]) == len(entries)


def test_unmodeled_history_and_selector_never_display_a_trained_result(model_service, page_factory):
    service = model_service
    original = next(result for result in service.history("daily") if result["model_version"] is None)
    page = page_factory(service)
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    page.model_select.setCurrentIndex(page.model_select.findData("unmodeled"))
    assert page.model_select.currentText() == "未建模"
    assert page.result["id"] == original["id"]
    assert page.result_values["network"].text() == "—"
    page._refresh()
    assert page.model_select.currentData() == "unmodeled"
    assert page.result["model_version"] is None
    page.model_select.setCurrentIndex(0)
    assert page.result["model_version"] == service.current_model["version"]
    entries = list(reversed(service.state["results"]))
    row = next(index for index, item in enumerate(entries) if item["id"] == original["id"])

    def choose_record():
        QApplication.activeModalWidget().findChild(QTableWidget).cellActivated.emit(row, 0)

    QTimer.singleShot(30, choose_record)
    page._show_history()
    assert page.model_select.currentData() == "unmodeled"
    assert page.model_select.currentText() == "未建模"
    assert page.run_select.currentData() == "daily"
    assert page.result["id"] == original["id"]
    assert page.label_select.currentData() == "abnormal"


def test_failed_import_preserves_exact_historical_result(model_service, tmp_path, page_factory, monkeypatch):
    original = model_service.latest_result("daily")
    model_service.set_thresholds(0.8, 0.5)
    model_service.evaluate("daily")
    page = page_factory(model_service)
    entries = list(reversed(model_service.state["results"]))
    row = next(index for index, entry in enumerate(entries) if entry["id"] == original["id"])

    def choose_record():
        QApplication.activeModalWidget().findChild(QTableWidget).cellActivated.emit(row, 0)

    QTimer.singleShot(30, choose_record)
    page._show_history()
    package = tmp_path / "invalid.zip"
    package.write_text("not a ZIP", encoding="utf-8")
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[2]))
    page.import_packages([package])
    wait_until(lambda: page.task is None)
    assert errors
    assert page.result == original
    assert not page.uses_current_thresholds
    assert "未设置报警阈值" in page.source_badge.text()
    assert "当时阈值" in page.source_badge.toolTip()
    assert len(model_service.state["results"]) == len(entries)
    assert page.daily_button.isEnabled()


def test_missing_result_clears_previous_fault_color(model_service, tmp_path, page_factory):
    service = model_service
    service.set_thresholds(warning=0.8, fault=0.5)
    service.import_packages([data_package(tmp_path, service, ("late",), "late.zip")])
    page = page_factory(service)
    page.run_select.setCurrentIndex(page.run_select.findData("late"))
    assert page.result["assessment"]["status"] == "fault"
    assert page.result_values["network"].styleSheet()
    page.model_select.setCurrentIndex(page.model_select.findData(service.models[0]["version"]))
    assert page.result is None
    assert all(value.text() == "—" for value in page.result_values.values())
    assert all(value.styleSheet() == "" for value in page.result_values.values())


@pytest.mark.parametrize(("warning", "fault"), [(.5, .2), (.8, .5)])
def test_alarm_colors_only_the_triggering_score_and_clears_when_normal_or_pending_review(
        model_service, page_factory, warning, fault):
    service = model_service
    service.set_thresholds(warning, fault)
    page = page_factory(service)
    assert ("预警" if fault == .2 else "故障") in page.source_badge.text()
    assert page.result_values["analysis"].text() == "1"
    assert service.analysis_metrics(page.result)[0] == pytest.approx(1)
    assert "对数参考 μ" in page.result_values["analysis"].toolTip()
    assert page.result_values["network"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["error"]
    assert page.result_values["analysis"].styleSheet() == ""
    assert page.result_values["temperature"].styleSheet() == page.result_values["current"].styleSheet() == ""
    service.settings["thresholds_model_version"] = "previous-model"
    page._refresh()
    assert "新模型阈值待复核" in page.status_lights["thresholds"].toolTip()
    assert "新模型阈值待复核" in page.source_badge.text()
    assert page.source_badge.styleSheet() == ""
    assert all(value.styleSheet() == "" for value in page.result_values.values())
    service.set_thresholds(.3, .1)
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "ready"
    assert "偏离参考" in page.source_badge.text()
    assert page.result_values["network"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["warning"]
    assert page.result_values["analysis"].styleSheet() == ""


def test_vibration_alarm_colors_vibration_without_coloring_normal_network(model_service, page_factory):
    model_service.set_thresholds(.5, .2)
    page = page_factory(model_service)
    result = model_service.latest_result("daily")
    result.update(vibration_p_value=.005, network_p_value=.3)
    page._show_result(result, current_thresholds=True)
    assert page.result["assessment"]["status"] == "fault"
    assert page.result_values["analysis"].text() == "0.1"
    assert page.result_values["network"].text() == "1"
    assert page.result_values["analysis"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["error"]
    assert page.result_values["network"].styleSheet() == ""


def test_invalid_compatibility_and_legacy_ratio_never_display_full_compatibility(model_service, page_factory):
    model_service.set_thresholds(.3, .2)
    page = page_factory(model_service)
    result = model_service.latest_result("daily")
    result["compatibility"]["vibration"].update(status="invalid", score=None, p_value=None,
                                                  message="正常参考对数样本标准差为 0")
    result["analysis_score"] = result["vibration_p_value"] = None
    page._show_result(result, current_thresholds=True)
    assert page.result_values["analysis"].text() == "—"
    assert page.result["assessment"]["status"] == "unavailable"
    assert "无效" in page.source_badge.text()
    assert "标准差为 0" in page.result_values["analysis"].toolTip()
    assert "标准差为 0" in page.source_badge.toolTip()
    assert page.result_values["analysis"].styleSheet() == ""

    legacy = deepcopy(result)
    legacy.pop("score_kind")
    legacy["score"] = legacy["analysis_score"] = 5.0
    page._show_result(legacy)
    assert all(page.result_values[key].text() == "—" for key in ("analysis", "network"))
    assert "旧倍率" in page.source_badge.text()
    assert "重新评估" in page.result_values["network"].toolTip()


@pytest.mark.parametrize("review_required", [False, True])
def test_invalid_vibration_preserves_network_alarm_only_after_threshold_review(model_service, page_factory, review_required):
    model_service.set_thresholds(.8, .5)
    if review_required:
        model_service.settings["thresholds_model_version"] = "previous-model"
    page = page_factory(model_service)
    result = model_service.latest_result("daily")
    reason = "正常参考对数样本标准差为 0，不能计算振动正常相容度"
    result["compatibility"]["vibration"].update(status="invalid", score=None, p_value=None, message=reason)
    result["analysis_score"] = result["vibration_p_value"] = None
    page._show_result(result, current_thresholds=True)
    assert page.result_values["analysis"].text() == "—"
    assert page.result_values["network"].text() == "0.4"
    assert page.result_values["analysis"].styleSheet() == ""
    assert reason in page.source_badge.toolTip()
    assert reason not in page.source_badge.text()
    if review_required:
        assert page.result["assessment"]["status"] == "unavailable"
        assert all(value.styleSheet() == "" for value in page.result_values.values())
        assert page.source_badge.styleSheet() == ""
    else:
        assert page.result["assessment"]["status"] == "fault"
        assert page.result_values["network"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["error"]
        assert page.source_badge.styleSheet()


def test_network_distribution_uses_one_value_per_reference_acquisition(model_service, page_factory):
    page = page_factory(model_service)
    result = page.result
    plot = page.plots["distribution"]
    assert plot.xlabel == "采集级重建误差 P95"
    assert plot.ylabel == "采集比例"
    assert plot._y_limits(0, 1) == (0, 1)
    assert sum(plot.series[0][1]) == pytest.approx(1)
    assert plot.series[1][0] == [result["reconstruction_error_p95"]] * 2
    assert "初步校准" in page.source_badge.text()
    assert "p：" in page.result_values["network"].toolTip()
    assert "正常概率" not in " ".join(label.text() for label in page.findChildren(QLabel))


def test_threshold_form_previews_and_saves_independent_settings_without_inference(model_service, page_factory, monkeypatch):
    from core.services.position_persistence import write_document

    result = model_service.latest_result("daily")
    result["vibration_p_value"] = .03
    entry = next(item for item in model_service.state["results"] if item["id"] == result["id"])
    write_document(model_service.root / entry["result_file"], result)
    page = page_factory(model_service)
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    stored = deepcopy(model_service.state["results"])
    original_settings = deepcopy(model_service.settings)
    evaluations = []
    monkeypatch.setattr(model_service, "evaluate", lambda *args, **kwargs: evaluations.append(args))

    def save(dialog):
        preview = dialog.findChild(QLabel, "spindle_threshold_preview")
        assert "振动 0.6，网络 0.4" in preview.text()
        dialog.findChild(QLineEdit, "spindle_vibration_alpha").setText("0.1")
        assert "振动 0.3，网络 0.4" in preview.text()
        dialog.findChild(QLineEdit, "spindle_vibration_warning_threshold").setText("0.2")
        dialog.findChild(QLineEdit, "spindle_vibration_fault_threshold").setText("0.1")
        assert "振动 0.3，网络 0.4" in preview.text()
        dialog.findChild(QLineEdit, "spindle_network_alpha").setText("0.1")
        assert "振动 0.3，网络 0.2" in preview.text()
        dialog.findChild(QLineEdit, "spindle_network_warning_threshold").setText("0.5")
        dialog.findChild(QLineEdit, "spindle_network_fault_threshold").setText("0.25")
        assert "振动 0.3，网络 0.2" in preview.text() and "故障" in preview.text()
        assert model_service.settings == original_settings
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).click()
        return dialog.result()

    monkeypatch.setattr(QDialog, "exec", save)
    page._show_thresholds()
    QApplication.processEvents()
    assert model_service.metric_settings == {
        "vibration": {"alpha": .1, "warning": .2, "fault": .1},
        "network": {"alpha": .1, "warning": .5, "fault": .25},
    }
    assert float(page.result_values["analysis"].text()) == pytest.approx(.3)
    assert float(page.result_values["network"].text()) == pytest.approx(.2)
    assert page.result["assessment"]["status"] == "fault"
    assert page.result_values["analysis"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["warning"]
    assert page.result_values["network"].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["error"]
    assert not evaluations
    assert model_service.state["results"] == stored
    page.model_select.setCurrentIndex(page.model_select.findData(model_service.current_model["version"]))
    assert float(page.result_values["network"].text()) == pytest.approx(.4)
    assert page.result["alpha"] == .05


def test_cancelling_independent_setting_preview_does_not_save_or_evaluate(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)
    original = deepcopy(model_service.state)
    values = {key: label.text() for key, label in page.result_values.items()}

    def cancel(dialog):
        field = dialog.findChild(QLineEdit, "spindle_network_alpha")
        field.setText("0.2")
        preview = dialog.findChild(QLabel, "spindle_threshold_preview")
        assert "网络 0.1" in preview.text()
        field.setText("0")
        assert "有效" in preview.text()
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", cancel)
    page._show_thresholds()
    QApplication.processEvents()
    assert model_service.state == original
    assert {key: label.text() for key, label in page.result_values.items()} == values


def test_historical_departure_is_explicit_and_can_return_to_current_without_overwriting(model_service, page_factory):
    from core.services.position_persistence import write_document

    historical = model_service.latest_result("daily", model_service.models[0]["version"])
    historical.update(analysis_score=.9689, score=.04338,
                      vibration_p_value=.048445, network_p_value=.002169,
                      assessment={"status": "unconfigured", "message": "未设置阈值"})
    entry = next(item for item in model_service.state["results"] if item["id"] == historical["id"])
    write_document(model_service.root / entry["result_file"], historical)
    stored = deepcopy(model_service.state["results"])
    page = page_factory(model_service)
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    page.model_select.setCurrentIndex(page.model_select.findData(historical["model_version"]))
    assert page.result == historical
    assert "历史快照" in page.result_context.text()
    assert "偏离正常参考" in page.source_badge.text()
    assert "偏离正常参考" in page.result_notes["analysis"].text()
    assert "偏离正常参考" in page.result_notes["network"].text()
    assert "径向 RMS" in page.result_notes["analysis"].text()
    assert "重建误差 P95" in page.result_notes["network"].text()
    assert page.latest_button.isVisible()
    page.latest_button.click()
    assert page.uses_current_thresholds
    assert page.result["model_version"] == model_service.current_model["version"]
    assert not page.latest_button.isVisible()
    assert model_service.state["results"] == stored
    assert model_service.latest_result("daily", historical["model_version"]) == historical


def test_training_and_calibration_reevaluations_are_not_presented_as_independent(model_service, page_factory):
    page = page_factory(model_service)
    for run_id in model_service.current_model["training_run_ids"] + model_service.current_model["calibration_run_ids"]:
        page.run_select.setCurrentIndex(page.run_select.findData(run_id))
        assert "不能作为独立验证" in page.result_context.text()
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    assert "未参与此模型建模" in page.result_context.text()


def test_historical_threshold_form_identifies_current_model_and_coverage(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)
    page.model_select.setCurrentIndex(page.model_select.findData(model_service.models[0]["version"]))
    original = deepcopy(page.result)

    def inspect(dialog):
        preview = dialog.findChild(QLabel, "spindle_threshold_preview")
        assert "当前模型最新评价" in preview.text()
        assert "网络 0.4" in preview.text()
        dialog.findChild(QLineEdit, "spindle_vibration_alpha").setText("0.25")
        assert "振动 75%" in preview.text() and "网络 95%" in preview.text()
        assert any("历史分数及当时设置保留" in label.text() for label in dialog.findChildren(QLabel))
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    page._show_thresholds()
    assert page.result == original
    assert model_service.metric_settings["vibration"]["alpha"] == .05


def test_condition_details_distinguish_recorded_and_missing_air_pressure(service, tmp_path, page_factory):
    service.import_packages([data_package(tmp_path, service, ("daily",))])
    page = page_factory(service)
    assert "7000 rpm · 空转 · 刀具重装 · 密封气压 0.2 MPa" in page.current_run_button.toolTip()
    assert "预警 未设置 / 故障 未设置" in page.current_run_button.toolTip()
    assert "None" not in page.current_run_button.toolTip()
    service.runs["daily"]["condition"] = {"tool_remounted": False, "seal_pressure_mpa": None}
    page._selection_changed()
    assert "未标记重装 · 密封气压未记录" in page.current_run_button.toolTip()
    assert "0.2 MPa" not in page.current_run_button.toolTip()


def test_sample_dialog_can_reopen_without_losing_controls_or_selection(model_service, page_factory):
    page = page_factory(model_service)
    for _ in range(2):
        def select_sample():
            dialog = QApplication.activeModalWidget()
            assert page.run_select.isVisible()
            assert page.model_select.isVisible()
            table = dialog.findChild(QTableWidget)
            table.cellClicked.emit(0, 0)
            assert page.run_select.currentData() == table.item(0, 0).data(Qt.ItemDataRole.UserRole)
            page.run_select.setCurrentIndex(page.run_select.findData("daily"))
            assert table.item(table.currentRow(), 0).data(Qt.ItemDataRole.UserRole) == "daily"
            page.model_select.setCurrentIndex(page.model_select.findData(model_service.models[0]["version"]))
            dialog.accept()
        QTimer.singleShot(30, select_sample)
        page._show_samples()
        QApplication.processEvents()
        page._refresh()
        assert page.run_select.currentData() == "daily"
        assert page.result["model_version"] == model_service.models[0]["version"]
        assert not page.run_select.isVisible()
        assert not page.model_select.isVisible()
    page._set_busy(True)
    page._set_busy(False)
    assert page.daily_button.isEnabled()


def test_signal_type_menu_uses_real_temperature_and_preserves_vibration_spectrum(model_service, page_factory):
    page = page_factory(model_service)
    spectrum = deepcopy(page.plots["spectrum"].series)
    page.signal_actions.actions()[3].trigger()
    assert page.signal_select.currentIndex() == 3
    assert page.plots["waveform"].ylabel == "°C"
    assert [values for _, values, _ in page.plots["waveform"].series] == page.result["temperature"]["values"]
    assert [name for _, _, name in page.plots["waveform"].series] == ["NTC1", "RTD1"]
    low, high = page.plots["waveform"]._y_limits(27.5, 27.7)
    assert low <= 27.5 < 27.7 <= high and high - low >= 5
    assert page.plots["spectrum"].series == spectrum
    energy_plot = page.plots["energy"]
    assert energy_plot.ylabel == "能量占比 / %"
    assert sum(energy_plot.series[0][1]) == pytest.approx(100)
    assert energy_plot.series[0][1][1] == pytest.approx(96.153846, rel=1e-5)
    assert energy_plot._y_limits(0, 96.153846) == (0, 100)
    page.signal_actions.actions()[4].trigger()
    assert page.plots["waveform"].ylabel == "A"
    assert page.plots["waveform"].series[0][1] == page.result["telemetry"]["current_a"]


def test_model_names_and_marker_follow_actual_current_model(model_service, page_factory):
    service = model_service
    current, other = service.models[0], service.models[1]
    # 当前标记以持久化配置为准，不用列表排序推断。
    service.state["current_model_version"] = current["version"]
    page = page_factory(service)
    index = page.model_select.findData(current["version"])
    name = page.model_select.itemText(index)
    assert name == page._model_name(current["version"])
    assert page.model_select.itemData(index, Qt.ItemDataRole.UserRole + 1)
    assert not page.model_select.itemData(page.model_select.findData(other["version"]), Qt.ItemDataRole.UserRole + 1)
    assert page.model_select.labelDrawingMode() == QComboBox.LabelDrawingMode.UseDelegate
    assert page.model_select.itemText(page.model_select.findData(other["version"])) == page._model_name(other["version"])
    assert page.model_select.itemText(0) == f"最新评价 · {page._model_name(current['version'])}"
    assert f"当前模型：{page._model_name(current['version'])}" in page.status_lights["model"].toolTip()
    assert page.result["model_version"] == current["version"]


def test_legacy_energy_is_recomputed_in_worker_without_rewriting_saved_evaluation(
        service, tmp_path, page_factory, monkeypatch):
    service.import_packages([data_package(tmp_path, service, ("daily",))])
    result_path = service.root / service.state["results"][-1]["result_file"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["band_energy"] = {"labels": ["10–100 Hz", "100–300 Hz", "300–600 Hz", "600–900 Hz"],
                             "values": [[0, 1, 0, 0]] * 3}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    original = result_path.read_bytes()
    original_state = deepcopy(service.state)
    threads = []
    compute = service.order_band_energy

    def recorded(run_id):
        threads.append(get_ident())
        return compute(run_id)

    monkeypatch.setattr(service, "order_band_energy", recorded)
    page = page_factory(service)
    wait_until(lambda: page.energy_task is None)
    assert threads and threads[0] != get_ident()
    assert page.plots["energy"].bars == ["异步", *[f"{order}×" for order in range(1, 8)]]
    assert page.plots["energy"].ylabel == "能量占比 / %"
    assert page.plots["energy"].series[0][1][1] == pytest.approx(96.153846, rel=1e-5)
    assert sum(page.plots["energy"].series[0][1]) == pytest.approx(100)
    assert result_path.read_bytes() == original and service.state == original_state
    page._refresh()
    assert len(threads) == 1


def test_review_dialog_save_and_cancel_preserve_exact_history_view(model_service, page_factory):
    original = model_service.latest_result("daily")
    model_service.set_thresholds(warning=0.8, fault=0.5)
    model_service.evaluate("daily")
    page = page_factory(model_service)
    page.run_select.setCurrentIndex(page.run_select.findData("daily"))
    page._show_result(original)
    for save in (False, True):
        def review():
            dialog = QApplication.activeModalWidget()
            assert page.label_select.isVisible()
            page.label_select.setCurrentIndex(page.label_select.findData("healthy"))
            page.training_check.setChecked(True)
            if save:
                page.label_button.click()
            dialog.accept()
        QTimer.singleShot(30, review)
        page._show_review()
        assert page.result["id"] == original["id"]
        assert model_service.runs["daily"]["manual_label"] == ("healthy" if save else "abnormal")
        assert not page.label_select.isVisible()
    assert model_service.runs["daily"]["training_eligible"]
    page._refresh()
    page._set_busy(True)
    page._set_busy(False)


def test_training_reads_epoch_number_before_dialog_widgets_are_destroyed(model_service, page_factory, monkeypatch):
    from PyQt6 import sip
    from PyQt6.QtWidgets import QSpinBox
    page = page_factory(model_service)
    captured = {}
    def start(title, operation, completed=None, *, status_key):
        captured['operation'] = operation
        captured['status_key'] = status_key
    monkeypatch.setattr(page, '_run_task', start)
    monkeypatch.setattr(model_service, 'train', lambda progress, config: config)
    def accept():
        dialog = QApplication.activeModalWidget()
        captured['dialog'] = dialog
        dialog.findChild(QSpinBox).setValue(7)
        dialog.accept()
    QTimer.singleShot(30, accept)
    page._train()
    sip.delete(captured['dialog'])
    assert captured['operation'](lambda *_: None) == {'epochs': 7}
    assert captured['status_key'] == 'model'


def test_successful_review_save_closes_dialog_without_extra_click(service, tmp_path, page_factory):
    service.import_packages([data_package(tmp_path, service, ('daily',))])
    page = page_factory(service)
    visible = []
    def save():
        dialog = QApplication.activeModalWidget()
        page.label_select.setCurrentIndex(page.label_select.findData('healthy'))
        page.training_check.setChecked(True)
        page.label_button.click()
        visible.append(dialog.isVisible())
        if dialog.isVisible():
            dialog.reject()
    QTimer.singleShot(30, save)
    page._show_review()
    assert visible == [False]
    assert service.runs['daily']['manual_label'] == 'healthy'


def test_review_save_failure_keeps_dialog_and_unsaved_input_for_retry(service, tmp_path, page_factory, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    service.import_packages([data_package(tmp_path, service, ('daily',))])
    page = page_factory(service)
    original = deepcopy(service.runs['daily'])
    saved = service._save
    errors, observed = [], []
    def fail():
        raise OSError('模拟磁盘写入失败')
    monkeypatch.setattr(service, '_save', fail)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args[2]))
    def save():
        dialog = QApplication.activeModalWidget()
        page.label_select.setCurrentIndex(page.label_select.findData('healthy'))
        page.label_note.setText('正常，纳入训练')
        page.training_check.setChecked(True)
        page.label_button.click()
        observed.append((dialog.isVisible(), deepcopy(service.runs['daily']), page.label_note.text()))
        monkeypatch.setattr(service, '_save', saved)
        page.label_button.click()
        observed.append(dialog.isVisible())
        if dialog.isVisible():
            dialog.reject()
    QTimer.singleShot(30, save)
    page._show_review()
    assert errors == ['模拟磁盘写入失败']
    assert observed == [(True, original, '正常，纳入训练'), False]
    assert service.runs['daily']['manual_label'] == 'healthy'
    assert len(service.runs['daily']['label_history']) == 1


def test_review_defaults_to_latest_import_and_can_relabel_older_zip_without_changing_view(
        service, tmp_path, page_factory, monkeypatch):
    service.import_packages([data_package(tmp_path, service, ("old1", "old2"), "old.zip")])
    service.set_label("old1", "healthy", "原备注")
    service.import_packages([data_package(tmp_path, service, ("latest",), "latest.zip")])
    page = page_factory(service)
    page.run_select.setCurrentIndex(page.run_select.findData("old2"))
    original_result = deepcopy(page.result)

    def review(dialog):
        dialog.show()
        QApplication.processEvents()
        choice = dialog.findChild(QComboBox, "spindle_review_package")
        assert choice.currentData() == ["latest"]
        assert "latest.zip" in choice.currentText()
        assert dialog.findChild(QLabel, "spindle_review_scope") is None
        assert choice.geometry().left() == page.label_select.geometry().left()
        assert choice.geometry().right() == page.label_select.geometry().right()
        choice.setCurrentIndex(next(index for index in range(choice.count()) if choice.itemData(index) == ["old1"]))
        assert "old.zip" in choice.currentText()
        assert page.label_select.currentData() == "healthy" and page.label_note.text() == "原备注"
        page.label_select.setCurrentIndex(page.label_select.findData("abnormal"))
        page.label_note.setText("历史样本改判")
        page.label_button.click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", review)
    page._show_review()
    assert service.runs["old1"]["manual_label"] == "abnormal"
    assert service.runs["old2"]["manual_label"] == service.runs["latest"]["manual_label"] == "unconfirmed"
    assert page.run_select.currentData() == "old2" and page.result == original_result


def test_sample_dialog_review_saves_only_the_selected_older_sample(
        service, tmp_path, page_factory, monkeypatch):
    service.import_packages([data_package(tmp_path, service, ("older", "latest"))])
    page = page_factory(service)

    def interact(dialog):
        if dialog.windowTitle() == "正常样本与网络模型":
            page.run_select.setCurrentIndex(page.run_select.findData("older"))
            button = next(button for button in dialog.findChildren(QPushButton) if button.text() == "人工判定")
            button.click()
        else:
            choice = dialog.findChild(QComboBox, "spindle_review_package")
            assert choice.currentData() == ["older"]
            page.label_select.setCurrentIndex(page.label_select.findData("abnormal"))
            page.label_note.setText("复核所选历史采集")
            page.label_button.click()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", interact)
    page._show_samples()
    assert service.runs["older"]["manual_label"] == "abnormal"
    assert service.runs["older"]["label_note"] == "复核所选历史采集"
    assert service.runs["latest"]["manual_label"] == "unconfirmed"
    assert page.run_select.currentData() == "older"


def test_sample_dialog_disables_review_for_a_selected_training_sample(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)

    def interact(dialog):
        button = next(button for button in dialog.findChildren(QPushButton) if button.text() == "人工判定")
        page.run_select.setCurrentIndex(page.run_select.findData(model_service.current_model["training_run_ids"][0]))
        assert not button.isEnabled()
        page.run_select.setCurrentIndex(page.run_select.findData("daily"))
        assert button.isEnabled()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", interact)
    page._show_samples()


def test_review_hides_trained_batch_but_allows_untrained_daily_history(model_service, page_factory, monkeypatch):
    page = page_factory(model_service)
    before = deepcopy(model_service.runs)

    def review(dialog):
        choice = dialog.findChild(QComboBox, "spindle_review_package")
        assert choice.currentData() == ["daily"]
        assert choice.count() == 1 and "已训练" not in choice.currentText()
        assert page.label_select.isEnabled() and page.label_button.isEnabled()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", review)
    page._show_review()
    assert model_service.runs == before


def test_review_is_empty_and_cannot_save_when_all_samples_are_trained(model_service, page_factory, monkeypatch):
    model_service.set_label("daily", "healthy")
    model_service.train()
    page = page_factory(model_service)
    before = deepcopy(model_service.runs)

    def review(dialog):
        choice = dialog.findChild(QComboBox, "spindle_review_package")
        assert choice.count() == 0 and choice.currentData() is None
        assert not any(widget.isEnabled() for widget in
                       (page.label_select, page.label_note, page.label_button, page.training_check))
        page._save_label()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", review)
    page._show_review()
    assert model_service.runs == before


def test_review_uses_latest_untrained_zip_when_newest_import_was_trained(
        model_service, tmp_path, page_factory, monkeypatch):
    model_service.import_packages([data_package(tmp_path, model_service, ("older",), "older.zip")])
    model_service.import_packages([data_package(tmp_path, model_service, ("recent",), "recent.zip")])
    model_service.set_label("recent", "healthy")
    model_service.train()
    page = page_factory(model_service)

    def review(dialog):
        choice = dialog.findChild(QComboBox, "spindle_review_package")
        assert choice.currentData() == ["older"]
        assert choice.count() == 2
        assert all(not model_service.trained_run_ids().intersection(choice.itemData(index))
                   for index in range(choice.count()))
        assert all("recent.zip" not in choice.itemText(index) for index in range(choice.count()))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", review)
    page._show_review()


@pytest.mark.parametrize("scale", [0.75, 1.0, 2.0])
def test_form_dialogs_fit_content_at_supported_scales(service, tmp_path, page_factory, monkeypatch, scale):
    service.import_packages([data_package(tmp_path, service, ("daily",))])
    page = page_factory(service)
    page.ui_scale = scale
    QApplication.instance().setStyleSheet(load_stylesheet(scale))
    observed = []

    def inspect(dialog):
        dialog.show()
        QApplication.processEvents()
        layout = dialog.layout()
        required = layout.totalHeightForWidth(dialog.width()) if layout.hasHeightForWidth() else dialog.sizeHint().height()
        assert required <= dialog.height() <= required + 2
        assert dialog.height() >= layout.minimumSize().height()
        observed.append(dialog.windowTitle())
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    try:
        page._show_thresholds()
        page._show_acquisition()
        page._train()
        page._show_review()
    finally:
        QApplication.instance().setStyleSheet(load_stylesheet())
    assert len(observed) == 4


def test_current_threshold_change_applies_to_other_runs_without_rewriting_history(model_service, page_factory):
    service = model_service
    page = page_factory(service)
    original = deepcopy(service.state['results'])
    service.set_thresholds(0.8, 0.5)
    page._refresh('daily')
    assert page.result_values['network'].styleSheet()
    assert page.result['assessment']['status'] == 'fault'
    page.run_select.setCurrentIndex(page.run_select.findData('initial1'))
    assert page.result_values['network'].styleSheet()
    assert '当前阈值（网络）：预警 0.8 / 故障 0.5' in page.current_run_button.toolTip()
    page.model_select.setCurrentIndex(page.model_select.findData(service.current_model['version']))
    assert page.result_values['network'].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["warning"]
    assert '当时阈值（网络）：预警 未设置 / 故障 未设置' in page.current_run_button.toolTip()
    assert service.state['results'] == original


def test_threshold_confirmation_clears_review_for_all_current_views_but_not_history(model_service, page_factory):
    service = model_service
    service.set_thresholds(0.8, 0.5)
    service.train()
    page = page_factory(service)
    assert '新模型阈值待复核' in page.status_lights['thresholds'].toolTip()
    assert page.result_values['network'].styleSheet() == ''
    historical = deepcopy(service.state['results'])
    service.set_thresholds(0.8, 0.5)
    page._refresh()
    for run_id in service.runs:
        page.run_select.setCurrentIndex(page.run_select.findData(run_id))
        assert page.result_values['network'].styleSheet()
    page.model_select.setCurrentIndex(page.model_select.findData(service.current_model['version']))
    assert page.result['assessment']['status'] == 'review_required'
    assert page.result_values['network'].styleSheet() == ''
    assert service.state['results'] == historical


def test_sample_dialog_retry_completes_missing_results_without_retraining(model_service, page_factory):
    from PyQt6.QtWidgets import QPushButton
    service = model_service
    version = service.current_model['version']
    service.state['results'][:] = [entry for entry in service.state['results']
                                 if not (entry['model_version'] == version and entry['run_id'] == 'daily')]
    service.current_model['reanalysis_status'] = 'failed'
    service._save()
    count = len(service.state['results'])
    page = page_factory(service)
    clicked = []
    def retry():
        dialog = QApplication.activeModalWidget()
        button = next(button for button in dialog.findChildren(QPushButton) if button.text() == '补算缺失评价')
        clicked.append(button.isEnabled())
        button.click()
    QTimer.singleShot(30, retry)
    page._show_samples()
    wait_until(lambda: page.task is None)
    assert clicked == [True]
    assert service.current_model['version'] == version
    assert len(service.models) == 2
    assert service.current_model['reanalysis_status'] == 'complete'
    assert len(service.state['results']) == count + 1
    assert service.latest_result('daily')['model_version'] == version
    assert not page.run_select.isVisible()


def test_sample_keyboard_navigation_updates_selected_run(model_service, page_factory):
    page = page_factory(model_service)
    selected = []
    def navigate():
        dialog = QApplication.activeModalWidget()
        table = dialog.findChild(QTableWidget)
        table.setCurrentCell(0, 0)
        QTest.keyClick(table, Qt.Key.Key_Down)
        selected.append((table.currentRow(), page.run_select.currentData(),
                         table.item(1, 0).data(Qt.ItemDataRole.UserRole)))
        dialog.reject()
    QTimer.singleShot(30, navigate)
    page._show_samples()
    assert selected[0][0] == 1
    assert selected[0][1] == selected[0][2]


@pytest.mark.parametrize('keyboard', [False, True])
def test_history_opens_selected_record_with_mouse_or_enter(model_service, page_factory, keyboard):
    page = page_factory(model_service)
    expected = model_service.history()[0]
    entries = list(reversed(model_service.state['results']))
    row = next(index for index, entry in enumerate(entries) if entry['id'] == expected['id'])
    def activate():
        dialog = QApplication.activeModalWidget()
        table = dialog.findChild(QTableWidget)
        item = table.item(row, 0)
        table.setCurrentCell(row, 0)
        table.scrollToItem(item)
        table.setFocus()
        if keyboard:
            QTest.keyClick(table, Qt.Key.Key_Return)
        else:
            position = table.visualItemRect(item).center()
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
            QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
        if dialog.isVisible():
            dialog.reject()
    QTimer.singleShot(30, activate)
    page._show_history()
    assert page.result['id'] == expected['id']


def test_real_cuda_training_can_be_followed_by_another_inference_task(service, tmp_path, page_factory):
    import torch
    from PyQt6.QtWidgets import QSpinBox
    if not torch.cuda.is_available():
        pytest.skip('CUDA 任务切换回归需要 GPU')
    service.import_packages([data_package(tmp_path, service, ('first', 'second', 'third', 'fourth', 'fifth'))], 'initial')
    service.set_label('first', 'healthy', '整批确认正常')
    service.config['training'].update({'epochs': 1, 'device': 'cuda', 'batch_size': 8})
    page = page_factory(service)
    def accept():
        dialog = QApplication.activeModalWidget()
        dialog.findChild(QSpinBox).setValue(1)
        dialog.accept()
    QTimer.singleShot(30, accept)
    page._train()
    wait_until(lambda: page.task is None, timeout=40)
    assert service.current_model['device'] == 'cuda'
    previous = page.result['id']
    page._evaluate()
    wait_until(lambda: page.task is None, timeout=20)
    assert page.result['id'] != previous
    assert page.result['score'] > 0
    assert len(service.models) == 1
    assert page.daily_button.isEnabled()


def test_finishing_background_work_after_page_deletion_does_not_access_widgets(application, service, monkeypatch):
    import sys
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    entered, release = Event(), Event()
    errors, callbacks = [], []
    monkeypatch.setattr(sys, 'excepthook', lambda kind, value, trace: errors.append(str(value)))
    page = SpindleRotationPage(service)
    def operation(progress):
        entered.set()
        if not release.wait(5):
            raise TimeoutError('测试任务未释放')
        progress(100, '完成')
        return 'finished'
    try:
        page._run_task('页面销毁测试', operation, completed=callbacks.append)
        wait_until(entered.is_set)
        page.deleteLater()
        QCoreApplication.sendPostedEvents(page, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(page)
    finally:
        release.set()
        page.worker_pool.shutdown(wait=True)
    QTest.qWait(80)
    assert not errors
    assert not callbacks
