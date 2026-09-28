"""操作真实 Qt 页面复现初始建模和后续复测；两阶段分进程执行以检查重启。"""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QPushButton, QTableWidget  # noqa: E402

from main import create_application  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from core.services.position_monitoring_service import read_document, write_document  # noqa: E402
from verify_results import verify_dataset  # noqa: E402


def click(page, text):
    buttons = [button for button in page.findChildren(QPushButton)
               if button.text() == text and button.isVisible()]
    if len(buttons) != 1:
        raise ValueError(f"按钮不唯一：{text}")
    QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)


def wait_task(application, page):
    started = time.monotonic()
    errors = []
    if page.task is not None:
        page.task.signals.failed.connect(errors.append)
        last_percent = [-1]

        def progress(percent, message):
            if percent // 5 != last_percent[0]:
                last_percent[0] = percent // 5
                print(f"{percent}% {message}", flush=True)

        page.task.signals.progress.connect(progress)
    while page.task is not None:
        application.processEvents()
        QTest.qWait(20)
        if time.monotonic() - started > 1800:
            raise TimeoutError("界面任务超过30分钟")
    application.processEvents()
    if errors:
        raise RuntimeError(errors[-1])


def capture(window, destination):
    QTest.qWait(250)
    if not window.grab().save(str(destination)):
        raise OSError(f"截图保存失败：{destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "initial", "current", "reopen"), required=True)
    options = parser.parse_args()
    output = options.report_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    application = create_application()
    window = MainWindow()
    window.resize(1600, 960)
    window.show()
    QTest.qWait(400)
    page = window.precision_page.content_stack.widget(0)
    service = page.service
    record = {"stage": options.stage, "service_root": str(service.root)}
    if options.stage == "calibration":
        from app.dialogs.robot_camera_calibration_dialog import RobotCameraCalibrationDialog
        from app.resources import fit_dialog

        assert service.parameters.get("hand_eye") is None and service.baseline is None
        before = {str(path): path.read_bytes() for path in service.storage.rglob("*.json")}
        dialog = RobotCameraCalibrationDialog(page)
        fit_dialog(dialog, 1050, 820)
        dialog.show()
        dialog.data_root.setText(str(options.dataset.resolve()))
        dialog._load_dataset()
        results = []
        for mode in ("calibrate", "reference"):
            dialog.hand_eye_mode.setCurrentIndex(dialog.hand_eye_mode.findData(mode))
            QTest.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton)
            assert dialog.task is not None
            dialog.task.signals.completed.connect(results.append)
            wait_task(application, dialog)
            assert dialog.report is not None
            capture(dialog, output / f"calibration_{mode}.png")
        record["calibrations"] = results
        record["monitoring_parameters_path"] = results[-1]["parameters_path"]
        after = {str(path): path.read_bytes() for path in service.storage.rglob("*.json")}
        assert before == after, "独立标定改变了主应用状态"
        dialog.close()
    elif options.stage == "initial":
        assert service.parameters.get("hand_eye") is None, "当前状态不是空白，请先备份并移走定位状态"
        assert service.parameters.get("camera_matrix") is None
        assert service.baseline is None and not service.list_history()
        capture(window, output / "01_empty.png")
        if options.parameters is None:
            raise ValueError("初始阶段需指定参数文件")
        with patch("app.pages.robot_position_page.QFileDialog.getOpenFileName",
                   return_value=(str(options.parameters.resolve()), "JSON")):
            click(page, "加载参数")
        assert service.parameters.get("hand_eye") is not None
        record["parameter_version"] = service.parameters["version"]
        capture(window, output / "02_parameters_loaded.png")
        with patch("app.pages.robot_position_page.QFileDialog.getExistingDirectory",
                   return_value=str((options.dataset / "B001").resolve())):
            page._import_image_directory()
        wait_task(application, page)
        assert len(service.current_batch["samples"]) == 600
        assert service.current_batch["batch_id"] == "B001"
        record["observation_path"] = service.current_batch["saved_path"]
        page._create_baseline()
        wait_task(application, page)
        assert service.baseline["batch"]["batch_id"] == "B001"
        assert service.current_batch is None
        record["baseline_id"] = service.baseline["id"]
        record["baseline_path"] = service.baseline["path"]
        capture(window, output / "03_baseline.png")
    elif options.stage == "current":
        previous = read_document(output / "initial.json")
        assert service.baseline["id"] == previous["baseline_id"]
        assert service.parameters["version"] == previous["parameter_version"]
        assert service.current_batch is None
        record["restart_restored_baseline"] = True
        with patch("app.pages.robot_position_page.QFileDialog.getExistingDirectory",
                   return_value=str((options.dataset / "B002").resolve())):
            page._import_image_directory()
        wait_task(application, page)
        assert len(service.current_batch["samples"]) == 600
        assert service.current_batch["batch_id"] == "B002"
        click(page, "评估精度")
        wait_task(application, page)
        assert page.result is not None and page.result["baseline_id"] == previous["baseline_id"]
        assert len(page.result["groups"]) == 30
        for index, mode in enumerate(("absolute_change", "repeatability_change", "repeatability"), 4):
            page.result_metric.setCurrentIndex(page.result_metric.findData(mode))
            capture(window, output / f"{index:02d}_{mode}.png")
        report = verify_dataset(options.dataset, service.baseline["batch"], service.current_batch, page.result)
        write_document(output / "verification.json", report)
        assert report["statistics_check"]["passed"], "独立统计与软件结果不一致"
        record.update({"result": page.result, "verification_path": str(output / "verification.json")})
        print(report["metric_comparison"], flush=True)
    else:
        previous = read_document(output / "current.json")
        histories = service.list_history()
        saved = next(item for item in histories if item["id"] == previous["result"]["id"])
        assert saved == previous["result"]
        assert service.baseline["id"] == saved["baseline_id"]
        history_index = next(index for index, item in enumerate(histories) if item["id"] == saved["id"])

        def select_history():
            dialog = application.activeModalWidget()
            table = dialog.findChild(QTableWidget)
            table.cellDoubleClicked.emit(history_index, 0)

        QTimer.singleShot(200, select_history)
        click(page, "历史记录")
        assert page.result == saved
        assert page.history_batches["current"]["batch_id"] == "B002"
        assert page.history_batches["baseline"]["batch_id"] == "B001"
        for index, batch in ((0, "B002"), (1, "B001")):
            page.observation_source.setCurrentIndex(index)
            assert page.observation_sample.count() == 600
            assert page.observation_sample.currentData()["sample_id"].startswith(batch)
        page.observation_source.setCurrentIndex(0)
        capture(window, output / "07_restored_history.png")
        record["restart_restored_history"] = True
        record["history_ui_and_images_restored"] = True
        record["history_id"] = saved["id"]
    write_document(output / f"{options.stage}.json", record)
    window.close()
    application.processEvents()
    print(f"COMPLETED {options.stage}: {output}", flush=True)


if __name__ == "__main__":
    main()
