"""主轴页面回归：空启动、后台导入、人工标签与历史模型隔离。"""

from copy import deepcopy
import json
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
    QApplication, QComboBox, QDialog, QFileDialog, QInputDialog, QLabel,
    QTableWidget,
)

from app.pages.spindle_rotation_page import SpindleRotationPage
from app.resources import DISPLAY, load_stylesheet
from core.algorithms import spindle_monitoring as algorithm
from core.services.spindle_monitoring_service import SpindleMonitoringService
from test_spindle_algorithms import make_run
from test_spindle_monitoring_service import h5_package


@pytest.fixture(scope="module")
def application():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(load_stylesheet())
    return application


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
        wait_until(lambda: page.task is None)
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
        return {"healthy_reference_mse": 0.2}

    def infer(run, model_path=None, config=None, progress=None):
        result = evaluate(run, config=config, progress=progress)
        if model_path:
            score = float(Path(model_path).read_text(encoding="utf-8"))
            result.update({
                "score": score, "window_scores": [score] * result["window_count"],
                "reference_scores": [0.8, 1.0, 1.2],
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
    service.import_packages([data_package(tmp_path, service, ("initial1", "initial2", "initial3"))], "initial")
    service.set_label("initial1", "healthy", "整批确认正常")
    service.import_packages([data_package(tmp_path, service, ("daily",), "daily.zip")])
    service.set_label("daily", "abnormal", "人工确认异常")
    service.train()
    service.train()
    return service


def test_empty_start_has_no_fabricated_model_score_or_curves(service, page_factory):
    page = page_factory(service)
    assert service.current_model is None
    assert service.runs == {}
    assert service.state["results"] == []
    assert page.result is None
    assert not page.conclusion.isVisible()
    assert all(value.text() == "—" for value in page.result_values.values())
    assert all(plot.series == [] for plot in page.plots.values())
    assert page.initial_button.isEnabled()
    assert not page.train_button.isEnabled()
    assert not page.evaluate_button.isEnabled()
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

    model_service.set_thresholds(2, 4)
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "ready"
    model_service.settings["thresholds_model_version"] = "old-model"
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "partial"
    assert "新模型阈值待复核" in page.status_lights["thresholds"].toolTip()
    model_service.set_thresholds(2, None)
    page._refresh()
    assert page.status_lights["thresholds"].property("state") == "partial"
    model_service.current_model["reanalysis_status"] = "failed"
    page._refresh()
    assert page.status_lights["model"].property("state") == "partial"
    assert "补算" in page.status_lights["model"].toolTip()


def test_model_work_lights_only_model_group_and_finishes_progress_after_refresh(model_service, page_factory):
    model_service.set_thresholds(2, 4)
    page = page_factory(model_service)
    release = Event()

    def operation(progress):
        progress(100, "训练完成，准备刷新")
        assert release.wait(5)

    try:
        page._run_task("模型任务", operation, status_key="model")
        wait_until(lambda: page.progress.value() == 99)
        assert page.status_lights["model"].property("state") == "partial"
        assert page.status_lights["acquisition"].property("state") == "ready"
        assert page.status_lights["thresholds"].property("state") == "ready"
        assert page.task_progress_note.text() == "训练完成，准备刷新"
    finally:
        release.set()
        wait_until(lambda: page.task is None)
    assert page.progress.value() == 100
    assert page.status_lights["model"].property("state") == "ready"


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
    service.import_packages([data_package(tmp_path, service, ("initial1", "initial2", "initial3"))], "initial")
    page = page_factory(service)
    assert not page.train_button.isEnabled()
    assert not service.training_candidates()
    assert not page.training_check.isChecked()
    assert page.label_select.currentData() == "unconfirmed"
    assert all(run["manual_label"] == "unconfirmed" and not run["label_history"]
               for run in service.runs.values())

    def confirm():
        dialog = QApplication.activeModalWidget()
        assert len(dialog.findChild(QComboBox, "spindle_review_package").currentData()) == 3
        page.label_select.setCurrentIndex(page.label_select.findData("healthy"))
        assert page.training_check.isChecked()
        assert "3 条" in page.label_button.text()
        page.label_button.click()

    QTimer.singleShot(30, confirm)
    page._show_review()
    assert all(run["manual_label"] == "healthy" for run in service.runs.values())
    assert len(service.training_candidates()) == 3 and page.train_button.isEnabled()

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
    for model, expected in ((first, "1"), (second, "2")):
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
    service.set_thresholds(warning=0.5, fault=1.5)
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


def test_missing_result_clears_previous_fault_color(model_service, tmp_path, page_factory):
    service = model_service
    service.set_thresholds(warning=0.5, fault=1.5)
    service.import_packages([data_package(tmp_path, service, ("late",), "late.zip")])
    page = page_factory(service)
    page.run_select.setCurrentIndex(page.run_select.findData("late"))
    assert page.result["assessment"]["status"] == "fault"
    assert page.conclusion.styleSheet()
    assert page.result_values["network"].styleSheet()
    page.model_select.setCurrentIndex(page.model_select.findData(service.models[0]["version"]))
    assert page.result is None
    assert page.conclusion.styleSheet() == ""
    assert all(value.text() == "—" for value in page.result_values.values())
    assert all(value.styleSheet() == "" for value in page.result_values.values())


@pytest.mark.parametrize(("warning", "fault", "status"), [(1.5, 3, "warning"), (.5, 1.5, "fault")])
def test_alarm_turns_score_numbers_red_and_clears_when_normal_or_pending_review(
        model_service, page_factory, warning, fault, status):
    service = model_service
    service.set_thresholds(warning, fault)
    page = page_factory(service)
    assert page.result_values["analysis"].text() == "1"
    assert service.analysis_metrics(page.result)[0] == pytest.approx(1)
    assert "正常参考均值" in page.result_values["analysis"].toolTip()
    assert page.conclusion.text() == {"warning": "预警 / 建议复测", "fault": "故障 / 建议检修"}[status]
    for key in ("analysis", "network"):
        assert page.result_values[key].palette().color(QPalette.ColorRole.WindowText).name() == DISPLAY["colors"]["error"]
    assert page.result_values["temperature"].styleSheet() == page.result_values["current"].styleSheet() == ""
    service.settings["thresholds_model_version"] = "previous-model"
    page._refresh()
    assert page.conclusion.text() == "新模型阈值待复核"
    assert all(value.styleSheet() == "" for value in page.result_values.values())
    service.set_thresholds(3, 4)
    page._refresh()
    assert page.conclusion.text() == "阈值内"
    assert all(value.styleSheet() == "" for value in page.result_values.values())


def test_condition_details_distinguish_recorded_and_missing_air_pressure(service, tmp_path, page_factory):
    service.import_packages([data_package(tmp_path, service, ("daily",))])
    page = page_factory(service)
    assert "7000 rpm · 空转 · 刀具重装 · 密封气压 0.2 MPa" in page.result_detail.text()
    assert "预警 未设置 / 故障 未设置" in page.result_detail.text()
    assert "None" not in page.result_detail.text()
    service.runs["daily"]["condition"] = {"tool_remounted": False, "seal_pressure_mpa": None}
    page._selection_changed()
    assert "未标记重装 · 密封气压未记录" in page.result_detail.text()
    assert "0.2 MPa" not in page.result_detail.text()


def test_sample_dialog_can_reopen_without_losing_controls_or_selection(model_service, page_factory):
    page = page_factory(model_service)
    for _ in range(2):
        def select_sample():
            dialog = QApplication.activeModalWidget()
            assert page.run_select.isVisible()
            assert page.model_select.isVisible()
            assert not any(widget.isVisible() for widget in (page.initial_status, page.conclusion, page.result_detail))
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


def test_model_names_and_pointer_follow_actual_current_model(model_service, page_factory):
    service = model_service
    current, other = service.models[0], service.models[1]
    # 当前指针以持久化配置为准，不用列表排序推断。
    service.state["current_model_version"] = current["version"]
    page = page_factory(service)
    index = page.model_select.findData(current["version"])
    name = page.model_select.itemText(index)
    assert name == page._model_name(current["version"])
    assert page.model_select.itemData(index, Qt.ItemDataRole.UserRole + 1)
    assert "当前模型" not in page.model_select.itemText(page.model_select.findData(other["version"]))
    assert not page.model_select.itemData(page.model_select.findData(other["version"]), Qt.ItemDataRole.UserRole + 1)
    assert page.model_select.itemText(0) == f"最新评价 → {page._model_name(current['version'])}"
    assert page.initial_status.text().startswith(f"当前模型 → {page._model_name(current['version'])}")
    assert page.result["model_version"] == current["version"]
    assert not page.conclusion.isVisible()


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
    model_service.set_thresholds(warning=0.5, fault=1.5)
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
    service.set_thresholds(0.5, 1.5)
    page._refresh('daily')
    assert page.conclusion.text() == '故障 / 建议检修'
    assert page.result['assessment']['status'] == 'unconfigured'
    page.run_select.setCurrentIndex(page.run_select.findData('initial1'))
    assert page.conclusion.text() == '故障 / 建议检修'
    assert '当前阈值：预警 0.5 倍 / 故障 1.5 倍' in page.result_detail.text()
    page.model_select.setCurrentIndex(page.model_select.findData(service.current_model['version']))
    assert page.conclusion.text() == '未设置阈值'
    assert '当时阈值：预警 未设置 / 故障 未设置' in page.result_detail.text()
    assert service.state['results'] == original


def test_threshold_confirmation_clears_review_for_all_current_views_but_not_history(model_service, page_factory):
    service = model_service
    service.set_thresholds(0.5, 1.5)
    service.train()
    page = page_factory(service)
    assert page.conclusion.text() == '新模型阈值待复核'
    historical = deepcopy(service.state['results'])
    service.set_thresholds(1, 5)
    page._refresh()
    for run_id in service.runs:
        page.run_select.setCurrentIndex(page.run_select.findData(run_id))
        assert page.conclusion.text() == '预警 / 建议复测'
    page.model_select.setCurrentIndex(page.model_select.findData(service.current_model['version']))
    assert page.conclusion.text() == '新模型阈值待复核'
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
    service.import_packages([data_package(tmp_path, service, ('first', 'second', 'third'))], 'initial')
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
