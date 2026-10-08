"""Exercise the real spindle GUI in an isolated empty store (no algorithm mocks)."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from unittest.mock import patch
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QTimer, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QDialogButtonBox, QFileDialog, QInputDialog, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTableWidget,
)

from main import create_application  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.pages.robot_position_page import RobotPositionPage  # noqa: E402
from app.pages.spindle_rotation_page import SpindleRotationPage  # noqa: E402
from core.services.position_monitoring_service import PositionMonitoringService  # noqa: E402
from core.services.spindle_monitoring_service import SpindleMonitoringService, assess_score  # noqa: E402


class GuiExercise:
    def __init__(self, args):
        self.args = args
        self.main_state_path = PROJECT / "storage/spindle_monitoring/state.json"
        self.main_state_before = self.main_state_path.read_bytes() if self.main_state_path.exists() else None
        self.application = create_application()
        self.application.setQuitOnLastWindowClosed(False)
        if args.scenario == "single-sample":
            config = json.loads((PROJECT / "config/spindle_monitoring.json").read_text(encoding="utf-8"))
            config_path = args.output / "config/spindle_monitoring.json"
            config_path.parent.mkdir()
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            self.service = SpindleMonitoringService(args.output / config["storage_root"], config)
        else:
            self.service = SpindleMonitoringService(args.output / "store")
        self.page = self.create_page()
        self.started = time.monotonic()
        self.phase = "startup"
        self.failures = []
        self.expected_warning = False
        self.warnings = []
        self.ticks = []
        self.progress_seen = set()
        self.report = {"status": "running", "started_at": datetime.now().isoformat(),
                       "output": str(args.output), "epochs_per_training": args.epochs,
                       "checks": [], "tasks": [], "screenshots": [],
                       "threshold_purpose": "Unconfigured; report actual scores without an engineering threshold"
                       if args.scenario == "single-sample" else "GUI exercise only; not engineering limits"}
        self.training_screenshot_taken = False
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.monitor)
        self.timer.start()
        self.old_excepthook = sys.excepthook
        sys.excepthook = self.qt_exception

    def create_page(self):
        if self.args.scenario == "single-sample":
            position = PositionMonitoringService(self.args.output)
            with patch("app.main_window.RobotPositionPage", lambda: RobotPositionPage(position)), \
                    patch("app.main_window.SpindleRotationPage", lambda: SpindleRotationPage(self.service)):
                self.window = MainWindow()
            self.window.setWindowTitle(self.window.windowTitle() + " [独立主轴测试]")
            self.window.resize(1600, 1000)
            self.window.show()
            self.window.precision_page.tab_bar.setCurrentIndex(1)
            QTest.qWait(200)
            return self.window.precision_page.content_stack.widget(1)
        page = SpindleRotationPage(self.service)
        page.ui_scale = 1.0
        page.resize(1400, 900)
        page.setWindowTitle("主轴流程验证（独立测试数据）")
        page.show()
        self.window = page
        QTest.qWait(100)
        return page

    def save_json(self, name, value):
        (self.args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def qt_exception(self, kind, value, tb):
        self.failures.append("".join(traceback.format_exception(kind, value, tb)))
        dialog = QApplication.activeModalWidget()
        if dialog is not None:
            dialog.reject()

    def monitor(self):
        try:
            self.ticks.append(time.monotonic())
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QMessageBox):
                message = modal.text()
                self.warnings.append({"phase": self.phase, "title": modal.windowTitle(), "text": message})
                if not self.expected_warning:
                    self.failures.append("Unexpected GUI warning: " + message)
                modal.accept()
            if self.page.task is not None:
                message = self.page.progress.toolTip()
                match = re.search(r"训练 (\d+)/(\d+)", message)
                if match and (int(match[1]) % 10 == 0 or match[1] in ("1", match[2])):
                    key = (self.phase, message)
                    if key not in self.progress_seen:
                        self.progress_seen.add(key)
                        print(f"[{self.phase}] {message}", flush=True)
                        if self.args.scenario == "single-sample" and not self.training_screenshot_taken:
                            self.training_screenshot_taken = True
                            self.save_view("03_training_in_progress.png")
        except BaseException:
            self.failures.append(traceback.format_exc())
            self.timer.stop()

    def check_errors(self):
        if self.failures:
            raise AssertionError(self.failures[0])

    def check(self, condition, message):
        if not condition:
            raise AssertionError(message)

    def click(self, button):
        self.check(button.isEnabled(), f"Button disabled: {button.text()}")
        if button.window() is self.window:
            self.page.ensureWidgetVisible(button)
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.check_errors()

    @staticmethod
    def button(parent, text):
        return next(button for button in parent.findChildren(QPushButton) if button.text() == text)

    def modal(self, button, handler):
        handled = []

        def callback():
            dialog = QApplication.activeModalWidget()
            try:
                if dialog is None or isinstance(dialog, QMessageBox):
                    raise AssertionError("Expected operation dialog did not open")
                handler(dialog)
                handled.append(True)
            except BaseException:
                self.failures.append(traceback.format_exc())
                current = QApplication.activeModalWidget()
                if current is not None:
                    current.reject()
                if dialog is not None:
                    dialog.reject()

        QTimer.singleShot(80, callback)
        self.click(button)
        self.check_errors()
        self.check(handled == [True], "Dialog callback did not finish")

    def wait_task(self):
        started = time.monotonic()
        first_tick = len(self.ticks)
        busy = self.page.task is not None
        if busy:
            self.check(not self.page.daily_button.isEnabled(), "Import remained enabled while busy")
        while self.page.task is not None or QApplication.activeModalWidget() is not None:
            QTest.qWait(50)
            self.check_errors()
            if time.monotonic() - started > self.args.timeout:
                raise TimeoutError(f"GUI task exceeded {self.args.timeout} seconds")
        QTest.qWait(80)
        self.check_errors()
        ticks = self.ticks[first_tick:]
        self.report["tasks"].append({"phase": self.phase, "elapsed_seconds": time.monotonic() - started,
                                     "heartbeat_count": len(ticks),
                                     "max_heartbeat_gap_seconds": max(np.diff(ticks), default=0.0)})
        self.check(self.page.daily_button.isEnabled(), "Controls did not recover after task")
        if busy and time.monotonic() - started > 1:
            self.check(len(ticks) >= 3, "Qt heartbeat stopped while background work was running")

    def stage(self, name, action):
        self.phase = name
        print("START " + name, flush=True)
        started = time.monotonic()
        action()
        self.check_errors()
        self.report["checks"].append({"name": name, "status": "passed",
                                      "seconds": time.monotonic() - started})
        self.save_json("gui_report.json", self.report)

    def import_packages(self, paths, initial=False):
        if initial:
            with TemporaryDirectory(prefix="spindle_gui_batch_") as temporary:
                outer = Path(temporary) / "initial_samples.zip"
                with ZipFile(outer, "w") as archive:
                    for path in paths:
                        archive.write(path, f"{path.parent.name}/{path.name}")
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(outer), "")), \
                        patch.object(QInputDialog, "getItem", return_value=("正常", True)):
                    self.click(self.page.initial_button)
                self.wait_task()
        else:
            self.check(len(paths) == 1, "Daily import expects one ZIP")
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(paths[0]), "")):
                self.click(self.page.daily_button)
            self.wait_task()

    def choose(self, run_id, model=None):
        def select(dialog):
            table = dialog.findChild(QTableWidget)
            row = next(row for row in range(table.rowCount())
                       if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == run_id)
            item = table.item(row, 0)
            table.scrollToItem(item)
            QTest.qWait(30)
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualItemRect(item).center())
            self.check(self.page.run_select.currentData() == run_id, "Clicked sample differs from active sample")
            index = self.page.model_select.findData(model)
            self.check(index >= 0, "Requested model is absent from the selector")
            self.page.model_select.setCurrentIndex(index)
            self.click(self.button(dialog, "关闭"))
        self.modal(self.button(self.page, "样本与模型"), select)

    def train(self, cancel=False):
        count = len(self.service.models)
        def fill(dialog):
            spin = dialog.findChild(QSpinBox)
            self.check(spin.value() == self.service.config["training"]["epochs"], "Unexpected default epoch count")
            spin.setValue(self.args.epochs)
            if self.args.scenario == "single-sample" and not cancel:
                self.save_view("02_training_parameters.png", dialog)
            box = dialog.findChild(QDialogButtonBox)
            standard = QDialogButtonBox.StandardButton.Cancel if cancel else QDialogButtonBox.StandardButton.Ok
            self.click(box.button(standard))
        self.modal(self.page.train_button, fill)
        self.wait_task()
        self.check(len(self.service.models) == count + (not cancel), "Training cancellation/version count mismatch")
        if not cancel:
            self.check(self.service.current_model["trained_epochs"] == self.args.epochs, "Wrong epoch count used by worker")
            self.check(self.service.current_model["initialization"] == "random", "Training did not start from random weights")
            self.check(self.service.current_model["reanalysis_status"] == "complete", "History reanalysis incomplete")

    def thresholds(self, warning, fault):
        def fill(dialog):
            for key, value in (("warning", warning), ("fault", fault)):
                dialog.findChild(QLineEdit, "spindle_" + key + "_threshold").setText(str(value))
            self.click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save))
        self.modal(self.button(self.page, "阈值设置"), fill)
        self.wait_task()

    def label(self, run_id, label, include, cancel=False, note=None):
        self.choose(run_id)
        previous = deepcopy(self.service.runs[run_id])
        shown_id = self.page.result["id"] if self.page.result else None
        def fill(dialog):
            self.page.label_select.setCurrentIndex(self.page.label_select.findData(label))
            self.page.training_check.setChecked(include)
            self.page.label_note.setText(note or "真实GUI流程演练：模拟人工检查，不作为故障真值")
            self.click(self.button(dialog, "关闭") if cancel else self.page.label_button)
            self.check(not dialog.isVisible(), "Successful label save did not close the dialog")
        self.modal(self.page.review_button, fill)
        if cancel:
            self.check(self.service.runs[run_id] == previous, "Cancel changed the saved manual label")
        else:
            record = self.service.runs[run_id]
            self.check(record["manual_label"] == label, "Manual label was not saved")
            self.check(record["training_eligible"] == (label == "healthy" and include), "Training opt-in mismatch")
        self.check((self.page.result["id"] if self.page.result else None) == shown_id, "Label operation changed displayed evaluation")

    def screenshot(self, name, width, height):
        self.page.resize(width, height)
        self.page.verticalScrollBar().setValue(0)
        QTest.qWait(150)
        target = self.args.output / name
        self.check(self.page.grab().save(str(target)), "Screenshot save failed")
        self.report["screenshots"].append(str(target))

    def save_view(self, name, widget=None):
        target = self.args.output / name
        self.check((widget or self.window).grab().save(str(target)), "Screenshot save failed")
        self.report["screenshots"].append(str(target))

    def screenshot_dialog(self, button, name):
        def capture(dialog):
            QTest.qWait(100)
            self.save_view(name, dialog)
            self.click(self.button(dialog, "关闭"))
        self.modal(button, capture)

    def channels(self):
        for index in range(6):
            def select(index=index):
                try:
                    menu = self.page.signal_menu
                    category = 0 if index < 3 else (1 if index < 5 else 2)
                    action = menu.actions()[category]
                    submenu = action.menu()
                    submenu.popup(menu.mapToGlobal(menu.actionGeometry(action).topRight()))
                    QTest.qWait(60)
                    subindex = index if index < 3 else (index - 3 if index < 5 else 0)
                    target = submenu.actions()[subindex]
                    QTest.mouseClick(submenu, Qt.MouseButton.LeftButton, pos=submenu.actionGeometry(target).center())
                    menu.close()
                except BaseException:
                    self.failures.append(traceback.format_exc())
                    self.page.signal_menu.close()
            QTimer.singleShot(80, select)
            self.click(self.page.signal_button)
            QTest.qWait(50)
            self.check_errors()
            self.check(self.page.signal_select.currentIndex() == index, "Signal menu selected the wrong channel")
            expected = "mm/s" if index < 3 else ("°C" if index < 5 else "A")
            self.check(self.page.plots["waveform"].ylabel == expected, "Signal channel unit mismatch")
            self.check(bool(self.page.plots["spectrum"].series), "Temperature/current selection cleared vibration spectrum")

    def history(self, original):
        count = len(self.service.state["results"])
        def select(dialog):
            entries = list(reversed(self.service.state["results"]))
            row = next(i for i, entry in enumerate(entries) if entry["id"] == original["id"])
            table = dialog.findChild(QTableWidget)
            item = table.item(row, 0)
            table.scrollToItem(item)
            QTest.qWait(40)
            position = table.visualItemRect(item).center()
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
            QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
            self.check(not dialog.isVisible(), "History double-click did not close the picker")
        self.modal(self.button(self.page, "检测历史"), select)
        self.check(self.page.result == original, "History did not restore the exact saved evaluation")
        self.check(len(self.service.state["results"]) == count, "History browsing appended evaluations")

    def collect_metrics(self):
        rows = []
        for run in self.service.list_runs():
            result = self.service.latest_result(run["run_id"])
            self.check(result is not None, "Missing current-model evaluation")
            score = float(np.quantile(result["window_scores"], 0.95))
            self.check(np.isclose(score, result["score"], rtol=2 * np.finfo(np.float32).eps, atol=0),
                       "P95 score differs beyond float32 precision from saved window scores")
            rms = np.asarray(result["rms_mm_s"])
            self.check(np.isclose(np.linalg.norm(rms[:2]), result["xy_rms_mm_s"]), "Radial RMS calculation mismatch")
            energy = np.asarray(result["band_energy"]["values"]).sum(axis=1)
            self.check(np.allclose(energy, rms ** 2, rtol=1e-4, atol=1e-9), "Band energies do not sum to velocity mean square")
            keys = ("run_id", "model_version", "role", "score", "normalized_mse", "channel_mse", "rms_mm_s",
                    "xy_rms_mm_s", "temperature_c", "actual_speed_rpm", "current_a", "window_count",
                    "thresholds", "thresholds_review_required", "assessment")
            row = {key: result[key] for key in keys}
            row.update({"captured_date": run["captured_date"], "condition": run.get("experiment_condition"),
                        "manual_label": run["manual_label"], "training_eligible": run["training_eligible"],
                        "assessment_with_current_thresholds": assess_score(result["score"], self.service.settings["thresholds"])})
            rows.append(row)
        self.save_json("current_model_metrics.json", rows)
        model_keys = ("version", "created_at", "initialization", "trained_epochs", "device", "train_window_count",
                      "validation_window_count", "training_run_ids", "calibration_run_ids", "healthy_reference_mse",
                      "history", "score_definition", "reanalysis_status")
        self.save_json("model_summaries.json", [{key: model[key] for key in model_keys} for model in self.service.models])

    def single_sample_workflow(self):
        groups = {"initial": [], "normal_daily": [], "august_daily": []}
        allocation = []
        for path in sorted(self.args.packages.rglob("*.zip")):
            with ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8-sig"))
            date = manifest["captured_date"]
            if date in ("2026-07-23", "2026-07-24", "2026-07-25"):
                group = "initial"
            elif date == "2026-07-26":
                group = "normal_daily"
            elif date == "2026-08-23":
                group = "august_daily"
            else:
                raise ValueError(f"单采集测试目录中出现未授权日期：{date}")
            self.check(len(manifest["runs"]) == 1, f"单采集 ZIP 必须只有一个 run：{path.name}")
            run = manifest["runs"][0]
            groups[group].append(path)
            allocation.append({"group": group, "zip": str(path), "date": date,
                               "run_id": run["run_id"], "condition": run.get("experiment_condition")})
        self.check(all(groups.values()), "建模、正常日常和八月日常三个分组都必须有数据")
        self.check({group: len(paths) for group, paths in groups.items()}
                   == {"initial": 28, "normal_daily": 8, "august_daily": 7}, "授权采集包数量应为 28 + 8 + 7")
        self.check(len({row["run_id"] for row in allocation}) == len(allocation), "单采集 ZIP 的 run_id 重复")
        self.report["allocation"] = allocation
        self.report["package_counts"] = {group: len(paths) for group, paths in groups.items()}
        self.save_json("data_allocation.json", allocation)
        self.stage("empty_start", lambda: self.check(not self.service.runs and not self.service.models
                                                     and self.page.result is None, "Store was not empty"))
        self.save_view("00_empty_start.png")
        self.stage("initial_import_july_23_to_25", lambda: self.import_packages(groups["initial"], initial=True))
        initial_ids = set(self.service.runs)
        self.check(len(initial_ids) == len(groups["initial"]), "建模 ZIP 与采集数量不一致")
        self.check(len(self.service.training_candidates()) == len(initial_ids), "初始采集未全部进入训练候选")
        self.check(all(run["manual_label"] == "healthy" for run in self.service.runs.values()),
                   "初始整批正常判定未覆盖全部样本")
        self.save_view("01_initial_import_complete.png")
        self.stage("random_initialization_training", self.train)
        model = self.service.current_model
        train_ids, calibration_ids = set(model["training_run_ids"]), set(model["calibration_run_ids"])
        self.check(not train_ids.intersection(calibration_ids), "训练与健康校准采集泄漏")
        self.check(train_ids | calibration_ids == initial_ids, "模型使用了分配外数据或漏掉初始采集")
        self.check(all(value is None for value in self.service.settings["thresholds"].values()),
                   "本次测试应保留空阈值")
        self.save_view("04_training_complete.png")
        self.screenshot_dialog(self.button(self.page, "样本与模型"), "05_training_samples_and_model.png")
        for group in ("normal_daily", "august_daily"):
            for path in groups[group]:
                self.stage(group + "_" + path.stem, lambda path=path: self.import_packages([path]))
            daily = [run for run in self.service.list_runs()
                     if run["captured_date"] == ("2026-07-26" if group == "normal_daily" else "2026-08-23")]
            for run in daily:
                condition = run.get("experiment_condition", "")
                if group == "normal_daily" or condition == "normal":
                    label = "healthy"
                elif condition.startswith("unbalance"):
                    label = "abnormal"
                else:
                    label = "unconfirmed"
                note = ("用户指定 7 月 26 日为正常日常测试；不参与建模" if group == "normal_daily" else
                        f"按原始实验条件 {condition} 记录测试组；不参与建模，不作为实机检修真值")
                self.stage("manual_review_" + run["run_id"],
                           lambda run=run, label=label, note=note:
                           self.label(run["run_id"], label, False, note=note))
            self.check(len(self.service.training_candidates()) == len(initial_ids), "日常测试数据混入建模候选")
            if group == "normal_daily":
                representative = max(daily, key=lambda run: self.service.latest_result(run["run_id"])["score"])
                self.choose(representative["run_id"])
                QTest.qWait(100)
                self.save_view("06_normal_daily_highest_score.png")
            else:
                for index, condition in enumerate(sorted({run.get("experiment_condition") for run in daily}), 7):
                    same_condition = [run for run in daily if run.get("experiment_condition") == condition]
                    representative = max(same_condition, key=lambda run: self.service.latest_result(run["run_id"])["score"])
                    self.choose(representative["run_id"])
                    QTest.qWait(100)
                    self.save_view(f"{index:02d}_august_{condition}.png")
        self.check(len(self.service.runs) == len(allocation), "最终采集数与 ZIP 总数不一致")
        self.check(len(self.service.models) == 1, "本次测试应只有一个固定模型")
        daily_ids = {run["run_id"] for run in self.service.list_runs() if run["purpose"] == "daily"}
        self.check(len(daily_ids) == 15 and not daily_ids.intersection(train_ids | calibration_ids),
                   "日常测试 15 次采集进入了训练或校准")
        self.check(all(self.service.latest_result(run_id)["window_count"] == 590 for run_id in self.service.runs),
                   "每次采集应有 59 段完整 10 秒数据、590 个 1 秒窗口")
        self.check(all(self.service.latest_result(run_id)["model_version"] == model["version"]
                       for run_id in self.service.runs), "日常样本未全部使用固定模型评分")
        self.check(all(self.service.latest_result(run_id)["assessment"]["status"] == "unconfigured"
                       for run_id in self.service.runs), "空阈值测试产生了自动健康或故障判定")
        self.stage("metric_consistency", self.collect_metrics)
        self.screenshot_dialog(self.button(self.page, "样本与模型"), "13_all_samples_and_labels.png")
        self.screenshot_dialog(self.button(self.page, "检测历史"), "14_detection_history.png")
        self.report.update({"status": "passed", "runs": len(self.service.runs), "models": 1,
                            "evaluations": len(self.service.state["results"]),
                            "training_candidates": len(initial_ids), "model_versions": [model["version"]],
                            "limitations": ["Recorded experiment replay; no real device acceptance claim",
                                            "August normal is the healthy control, balance remains unconfirmed, unbalance is the abnormal group",
                                            "No engineering thresholds supplied; results report scores, not validated fault decisions",
                                            "Training completion does not certify convergence or fault separability"]})
        self.save_json("gui_report.json", self.report)
        self.page.worker_pool.shutdown(wait=True)
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(100)
        from debug.reset import reset_usage_data
        reset_usage_data(self.args.output)
        self.service = SpindleMonitoringService(self.args.output / "storage/spindle_monitoring")
        self.page = self.create_page()
        self.check(not self.service.runs and not self.service.models and not self.service.state["results"],
                   "测试结束 reset 未清空使用数据")
        self.check(self.page.result is None and not self.page.process_log.toPlainText(), "reset 后主轴显示未清空")
        self.save_view("15_reset_empty.png")
        self.report["isolated_test_store_reset"] = True

    def workflow(self):
        if self.args.scenario == "single-sample":
            self.single_sample_workflow()
            return
        self.stage("empty_start", lambda: self.check(not self.service.runs and not self.service.models and self.page.result is None,
                                                    "Store was not empty"))
        invalid = self.args.output / "invalid_input.zip"
        invalid.write_text("not a ZIP; intentional GUI failure exercise", encoding="utf-8")
        def invalid_import():
            before = len(self.warnings)
            self.expected_warning = True
            try:
                self.import_packages([invalid])
            finally:
                self.expected_warning = False
            self.check(len(self.warnings) == before + 1 and not self.service.runs, "Invalid ZIP was not rejected cleanly")
        self.stage("invalid_zip_and_control_recovery", invalid_import)
        initial = [self.args.packages / f"spindle_202607{day}.zip" for day in (22, 23, 24, 25)]
        self.stage("initial_import_31_runs", lambda: self.import_packages(initial, initial=True))
        self.check(len(self.service.runs) == len(self.service.training_candidates()) == 31, "Initial sample count mismatch")
        self.check(all(run["manual_label"] == "healthy" for run in self.service.runs.values()), "Batch normal label did not cover all samples")
        self.stage("cancel_training", lambda: self.train(cancel=True))
        self.stage("first_training", self.train)
        first = deepcopy(self.service.current_model)
        original = deepcopy(self.service.latest_result(self.service.list_runs()[0]["run_id"]))
        self.stage("initial_thresholds_2_4", lambda: self.thresholds(2, 4))
        for date in ("20260726", "20260823"):
            self.stage("daily_import_" + date, lambda date=date: self.import_packages([self.args.packages / f"spindle_{date}.zip"]))
        self.check(len(self.service.runs) == 49 and len(self.service.training_candidates()) == 31, "Daily samples were auto-enrolled")
        def duplicate():
            before = len(self.service.state["results"])
            self.import_packages([self.args.packages / "spindle_20260823.zip"])
            self.check(len(self.service.state["results"]) == before, "Duplicate import appended completed results")
        self.stage("duplicate_import", duplicate)
        july = [run for run in self.service.list_runs() if run["captured_date"] == "2026-07-26"]
        abnormal = [run for run in self.service.list_runs() if run.get("experiment_condition", "").startswith("unbalance")]
        self.check(len(july) == 11 and len(abnormal) == 4, "Unexpected experiment grouping")
        for index, run in enumerate(july[:3]):
            self.stage(f"healthy_label_in_training_{index + 1}", lambda run=run: self.label(run["run_id"], "healthy", True))
        self.stage("healthy_label_without_training", lambda: self.label(july[3]["run_id"], "healthy", False))
        self.stage("cancel_manual_label", lambda: self.label(july[4]["run_id"], "abnormal", False, cancel=True))
        for run in abnormal:
            self.stage("abnormal_label_" + run["experiment_condition"], lambda run=run: self.label(run["run_id"], "abnormal", False))
        self.check(len(self.service.training_candidates()) == 34, "Wrong retraining candidate count")
        before_retrain = deepcopy(self.service.state["results"])
        labels = {key: deepcopy(run["label_history"]) for key, run in self.service.runs.items()}
        self.save_json("first_model_scores.json", [row for row in before_retrain if row["model_version"] == first["version"]])
        self.stage("second_training", self.train)
        second = self.service.current_model
        self.check(second["version"] != first["version"] and self.service.thresholds_review_required, "New model did not require threshold review")
        self.check("新模型阈值待复核" in self.page.status_lights["thresholds"].toolTip(), "Current GUI did not show new-model threshold review")
        self.check(self.service.state["results"][:len(before_retrain)] == before_retrain, "Retraining modified old history")
        self.check(labels == {key: run["label_history"] for key, run in self.service.runs.items()}, "Retraining modified human labels")
        used = set(second["training_run_ids"] + second["calibration_run_ids"])
        self.check(len(used) == 34 and not used.intersection(run["run_id"] for run in self.service.list_runs()
                                                           if run["captured_date"] == "2026-08-23"), "August verification leaked into training")
        self.stage("confirm_thresholds_for_new_model", lambda: self.thresholds(2, 4))
        self.stage("switch_run_uses_current_thresholds", lambda: self.choose(july[1]["run_id"]))
        self.check(self.page.status_lights["thresholds"].property("state") == "ready" and not self.service.thresholds_review_required,
                   "Switching samples kept obsolete threshold-review status")
        self.stage("six_signal_channels", self.channels)
        score = self.page.result["score"]
        for warning, fault, status in ((score * 2, score * 3, "normal"),
                                       (score * .5, score * 2, "warning"),
                                       (score * .25, score * .5, "fault")):
            self.stage("threshold_branch_" + status, lambda warning=warning, fault=fault: self.thresholds(warning, fault))
            self.check(self.page.result["assessment"]["status"] == status
                       and bool(self.page.result_values["network"].styleSheet()) == (status != "normal"),
                       "GUI threshold branch mismatch: " + status)
        self.stage("restore_demo_thresholds", lambda: self.thresholds(2, 4))
        self.stage("exact_history_double_click", lambda: self.history(original))
        for model in (first["version"], second["version"], "unmodeled", None):
            self.stage("select_model_" + str(model), lambda model=model: self.choose(original["run_id"], model))
            expected = second["version"] if model is None else (None if model == "unmodeled" else model)
            self.check(self.page.result["model_version"] == expected, "Model selector displayed another model")
        self.screenshot("gui_1400x900.png", 1400, 900)
        self.screenshot("gui_1100x800.png", 1100, 800)
        def restart():
            persisted = (self.service.root / "state.json").read_bytes()
            self.page.close()
            self.page.deleteLater()
            QTest.qWait(80)
            self.service = SpindleMonitoringService(self.args.output / "store")
            self.page = self.create_page()
            self.check((self.service.root / "state.json").read_bytes() == persisted, "Restart rewrote persistent state")
            self.check(self.service.current_model["version"] == second["version"], "Restart lost the selected model")
            self.check(labels == {key: run["label_history"] for key, run in self.service.runs.items()}, "Restart lost human labels")
            self.check(self.service.settings["thresholds"] == {"warning": 2, "fault": 4}, "Restart lost thresholds")
        self.stage("restart_persistence", restart)
        self.stage("metric_consistency", self.collect_metrics)
        self.report.update({"status": "passed", "runs": len(self.service.runs), "models": len(self.service.models),
                            "evaluations": len(self.service.state["results"]), "training_candidates": 34,
                            "model_versions": [first["version"], second["version"]],
                            "limitations": ["Replay of recorded data, not a real device acceptance test",
                                            "Human labels and thresholds here are workflow simulation inputs",
                                            "Training completion does not certify convergence or fault separability"]})

    def finish(self):
        main_state_after = self.main_state_path.read_bytes() if self.main_state_path.exists() else None
        self.report["main_store_unchanged"] = main_state_after == self.main_state_before
        self.report["main_store_check_note"] = ("只读字节比较一致" if main_state_after == self.main_state_before
                                                else "检测到外部变化；演练仅使用隔离store，未恢复或改写主store")
        self.report.update({"finished_at": datetime.now().isoformat(), "elapsed_seconds": time.monotonic() - self.started,
                            "phase": self.phase, "warnings": self.warnings, "callback_errors": self.failures})
        self.save_json("gui_report.json", self.report)
        self.timer.stop()
        sys.excepthook = self.old_excepthook
        if self.page.task is None:
            self.page.worker_pool.shutdown(wait=True)
            self.window.close()
            if self.args.scenario == "single-sample" and not self.report.get("isolated_test_store_reset"):
                from debug.reset import reset_usage_data
                reset_usage_data(self.args.output)
                self.report["isolated_test_store_reset"] = True
            if self.args.scenario == "single-sample":
                for name in ("storage", "config"):
                    target = self.args.output / name
                    if not target.resolve().is_relative_to(self.args.output):
                        raise ValueError(f"隔离演练清理目录越界：{target}")
                    if target.exists():
                        shutil.rmtree(target)
                self.report["isolated_usage_files_removed"] = True
                self.save_json("gui_report.json", self.report)
        print(self.report["status"].upper() + " " + str(self.args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=PROJECT / "data/processed/spindle_daily_packages")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/reports/spindle_workflow" /
                        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_gui"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--scenario", choices=("daily-replay", "single-sample"), default="daily-replay",
                        help="single-sample: July 23–25 modeling, July 26 normal daily, August 23 condition tests")
    parser.add_argument("--timeout", type=float, default=5400, help="Maximum seconds per background task")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("GUI演练要求新的隔离目录，请另选 --output")
    if args.output == (PROJECT / "storage/spindle_monitoring").resolve():
        raise ValueError("不能使用主应用存储作为演练输出")
    args.output.mkdir(parents=True)
    exercise = None
    try:
        exercise = GuiExercise(args)
        exercise.workflow()
    except BaseException:
        error = traceback.format_exc()
        if exercise is None:
            (args.output / "gui_report.json").write_text(json.dumps({"status": "failed", "error": error},
                ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            exercise.report.update({"status": "failed", "error": error})
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        if exercise is not None:
            exercise.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
