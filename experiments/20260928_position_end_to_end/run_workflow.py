"""操作真实 Qt 页面完成 B001 基准、B002/B003 复测和自动恢复；保留最终状态。"""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QPushButton  # noqa: E402

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


def wait_task(application, page, progress_capture=None):
    started = time.monotonic()
    errors = []
    if page.task is not None:
        page.task.signals.failed.connect(errors.append)
        last_percent = [-1]

        def progress(percent, message):
            if percent // 5 != last_percent[0]:
                last_percent[0] = percent // 5
                print(f"{percent}% {message}", flush=True)
            if progress_capture is not None and 25 <= percent <= 50 and not progress_capture.exists():
                page.window().grab().save(str(progress_capture))

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
    parser.add_argument("--batch", choices=("B002", "B003"), default="B003")
    parser.add_argument("--bias-reference", type=Path,
                        help="仅事后核验用的生成偏置表，不传给测量服务")
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
        assert page.result is not None and page.result["batch_id"] == "B001"
        assert len(service.list_history()) == 1
        record["baseline_evaluation_id"] = page.result["id"]
        record["baseline_id"] = service.baseline["id"]
        record["baseline_path"] = service.baseline["path"]
        capture(window, output / "03_baseline.png")
    elif options.stage == "current":
        previous = read_document(output / "initial.json")
        assert service.baseline["id"] == previous["baseline_id"]
        assert service.parameters["version"] == previous["parameter_version"]
        assert page.result is not None, "启动未自动恢复上次结果"
        record["restart_restored_baseline"] = True
        with patch("app.pages.robot_position_page.QFileDialog.getExistingDirectory",
                   return_value=str((options.dataset / options.batch).resolve())):
            page._import_image_directory()
        wait_task(application, page, output / "image_progress.png")
        assert len(service.current_batch["samples"]) == 600
        assert service.current_batch["batch_id"] == options.batch
        # 已导入但尚未评估，也必须能从磁盘恢复。
        restored = type(service)(root=service.root)
        assert restored.current_batch["batch_id"] == options.batch
        assert len(restored.current_batch["samples"]) == 600
        click(page, "评估精度")
        wait_task(application, page)
        assert page.result is not None and page.result["baseline_id"] == previous["baseline_id"]
        assert len(page.result["groups"]) == 30
        for mode in ("absolute_change", "repeatability_change", "repeatability"):
            page.result_metric.setCurrentIndex(page.result_metric.findData(mode))
            if options.batch == "B003":
                capture(window, output / f"{mode}.png")
        bias = read_document(options.bias_reference) if options.bias_reference else None
        report = verify_dataset(options.dataset, service.baseline["batch"], service.current_batch,
                                page.result, bias_reference=bias)
        verification_path = output / f"verification_{options.batch}.json"
        write_document(verification_path, report)
        assert report["statistics_check"]["passed"], "独立统计与软件结果不一致"
        record.update({"result_id": page.result["id"], "baseline_id": page.result["baseline_id"],
                       "verification_path": str(verification_path),
                       "import_without_evaluation_restored": True})
        if options.batch == "B003":
            b002 = next(item for item in service.list_history() if item["batch_id"] == "B002")
            second = read_document(b002["current_batch_path"])
            comparison = verify_dataset(options.dataset, second, service.current_batch, bias_reference=bias)
            write_document(output / "comparison_B002_B003.json", comparison)
            page.result_metric.setCurrentIndex(page.result_metric.findData("absolute_change"))
        print(report["metric_comparison"], flush=True)
    else:
        previous = read_document(output / "current_B003.json")
        histories = service.list_history()
        saved = next(item for item in histories if item["id"] == previous["result_id"])
        assert saved["baseline_id"] == previous["baseline_id"]
        assert service.baseline["id"] == saved["baseline_id"]
        assert [item["batch_id"] for item in histories] == ["B001", "B002", "B003"]
        assert page.result == saved
        assert service.current_batch["batch_id"] == "B003"
        for index, batch in ((0, "B003"), (1, "B001")):
            page.observation_source.setCurrentIndex(index)
            assert page.observation_sample.count() == 600
            assert page.observation_sample.currentData()["sample_id"].startswith(batch)
        page.observation_source.setCurrentIndex(0)
        capture(window, output / "final_restored.png")
        record["automatic_restart_restore"] = True
        record["history_and_images_restored"] = True
        record["history_id"] = saved["id"]
    name = f"current_{options.batch}" if options.stage == "current" else options.stage
    write_document(output / f"{name}.json", record)
    window.close()
    application.processEvents()
    print(f"COMPLETED {options.stage}: {output}", flush=True)


if __name__ == "__main__":
    main()
