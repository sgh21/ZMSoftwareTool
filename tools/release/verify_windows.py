"""从外部操作最终 Windows EXE；需要已登录且无人操作的 Windows 桌面。

验证依赖留在开发/验收机器，不加入软件目录。此检查不等于干净虚拟机验收。
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback
from zipfile import ZipFile


def wait_until(predicate, description, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.3)
    raise TimeoutError(description)


def clean_environment(data_dir):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("PYTHON", "CONDA", "QT_", "VIRTUAL_ENV"))}
    windows = Path(os.environ["SystemRoot"])
    env["PATH"] = os.pathsep.join(str(windows / part) for part in ("System32", "", "System32/Wbem"))
    env["ZMSOFTWARE_DATA_DIR"] = str(data_dir)
    return env


def bundle_files(folder):
    return {path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(folder.rglob("*")) if path.is_file()}


class WindowsVerification:
    def __init__(self, exe, data, output):
        self.exe = exe
        self.data = data
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.user_data = output / "user_data"
        self.manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        self.report = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "executable": str(exe), "executable_bytes": exe.stat().st_size,
            "executable_mtime_ns": exe.stat().st_mtime_ns,
            "bundle_files": bundle_files(exe.parent),
            "platform": platform.platform(), "machine": platform.machine(),
            "environment_scope": "现有 Windows 主机，隔离工作目录、业务存储和 PATH；不是干净虚拟机",
            "test_data": "人工构造的软件验证样例，不是实际设备测量",
            "checks": [],
            "not_verified": ["干净 Windows 11 虚拟机/新机器", "麒麟 V10 实机", "真实相机/主轴设备", "模型精度或现场测量精度"],
        }
        self.process = None

    def start(self, cwd):
        from pywinauto import Application

        self.env = clean_environment(self.user_data)
        self.report["working_directory"] = str(cwd)
        self.report["launch_path"] = self.env["PATH"]
        self.log_stream = (self.output / "process.log").open("ab")
        self.process = subprocess.Popen([str(self.exe)], cwd=cwd, env=self.env,
                                        stdout=self.log_stream, stderr=subprocess.STDOUT)
        self.app = Application(backend="uia").connect(process=self.process.pid, timeout=90)
        self.native = Application(backend="win32").connect(process=self.process.pid)
        self.window = self.app.window(title="精度监控系统")

        def started():
            if self.process.poll() is not None:
                raise RuntimeError(f"软件在创建主窗口前退出：{self.process.returncode}，见 process.log")
            error = self.native.window(title="Unhandled exception in script")
            if error.exists(timeout=0.1):
                error.capture_as_image().save(self.output / "startup_failure.png")
                raise RuntimeError("打包程序启动异常：" + "\n".join(item.window_text() for item in error.descendants()))
            return self.window.exists(timeout=0.2) and self.window.is_visible()

        wait_until(started, "主窗口启动超时", timeout=90)
        self.main_handle = self.window.wrapper_object().handle
        self.window = self.app.window(handle=self.main_handle)
        wait_until(lambda: self.button("加载参数").is_enabled(), "启动初始化未完成")
        if not (self.output / "01_startup.png").exists():
            self.screenshot("01_startup")

    def stop(self):
        if self.process is None:
            return
        try:
            self.native.window(handle=self.main_handle).close()
            self.process.wait(timeout=15)
            self.report["shutdown_exit_code"] = self.process.returncode
        except Exception:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.report["shutdown_forced"] = True
        self.log_stream.close()

    def button(self, name, parent=None):
        return (parent or self.window).child_window(title=name, control_type="Button")

    def click(self, name, parent=None):
        self.button(name, parent).wait("enabled", timeout=30)
        self.button(name, parent).click_input()

    def tab(self, name):
        self.window.child_window(title=name, control_type="TabItem").select()
        time.sleep(0.3)

    def texts(self, parent=None):
        return [item.window_text() for item in (parent or self.window).descendants(control_type="Text")]

    def wait_text(self, text, timeout=90):
        wait_until(lambda: any(text in value for value in self.texts()),
                   f"界面未出现 {text}", timeout)

    def file_dialog(self, title, path):
        # Qt 的原生文件对话框在 UIA 中不可见，按本进程的 Win32 控件定位。
        dialog = self.native.window(title=title, class_name="#32770")
        dialog.wait("visible", timeout=30)
        # Windows 的打开/保存对话框使用不同控件 ID；两者都只有文件名 Edit 可见。
        fields = [field for field in dialog.descendants(class_name="Edit") if field.is_visible()]
        assert len(fields) == 1, [(field.window_text(), field.control_id()) for field in fields]
        fields[0].set_edit_text(str(path))
        dialog.child_window(control_id=1, class_name="Button").click_input()
        dialog.wait_not("visible", timeout=30)

    def dialog(self, title_re):
        native = self.native.window(title_re=title_re, class_name_re="Qt.*QWindowIcon")
        native.wait("visible", timeout=30)
        return self.app.window(handle=native.handle)

    def screenshot(self, name):
        self.window.capture_as_image().save(self.output / f"{name}.png")
        (self.output / f"{name}_ui.json").write_text(json.dumps([
            {"text": item.window_text(), "type": item.element_info.control_type,
             "id": item.element_info.automation_id, "enabled": item.is_enabled()}
            for item in self.window.descendants()
        ], ensure_ascii=False, indent=2), encoding="utf-8")

    def check(self, name, operation):
        started = time.monotonic()
        print(name, flush=True)
        result = {"name": name}
        try:
            result["evidence"] = operation()
            result["status"] = "passed"
        except Exception:
            result["status"] = "failed"
            result["error"] = traceback.format_exc()
            try:
                self.screenshot(f"failed_{len(self.report['checks'])}")
            except Exception:
                pass
            # 清掉失败步骤遗留的本进程弹窗，避免后续检查被同一个模态窗口阻塞。
            for window in self.native.windows():
                if window.is_visible() and window.handle != self.main_handle:
                    try:
                        window.close()
                    except Exception:
                        pass
        result["seconds"] = round(time.monotonic() - started, 2)
        self.report["checks"].append(result)
        self.save_report()
        print(f"{name}: {result['status']}", flush=True)
        return result["status"] == "passed"

    def save_report(self):
        (self.output / "report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")

    def startup(self):
        from pywinauto import mouse

        expected = ("机器人末端定位精度", "主轴回转精度", "主轴轴向进给精度")
        states = []
        for index, name in enumerate(expected):
            self.tab(name)
            button = self.button("评估精度").wrapper_object()
            assert not button.is_enabled(), f"{name}空数据时评估未禁用"
            header = self.window.rectangle()
            mouse.move(coords=(header.left + 100, header.top + 20))
            time.sleep(0.2)
            point = button.rectangle().mid_point()
            mouse.move(coords=(point.x, point.y))
            tip = self.app.window(class_name="QTipLabel")
            tip.wait("visible", timeout=10)
            tooltip = tip.window_text()
            assert "先" in tooltip, f"{name}缺少操作提示：{tooltip}"
            self.window.capture_as_image().save(self.output / f"00_tooltip_{index}.png")
            if name == expected[2]:
                assert self.button("导入数据").is_enabled()
            states.append({"page": name, "evaluate_enabled": False, "help": tooltip})
        self.tab(expected[0])
        bundle = self.exe.parent / "_internal"
        for relative in ("config/display.json", "resources/styles/light.qss",
                         "resources/icons/branding/reference-interface.png",
                         "resources/icons/precision/robot-position.svg"):
            assert (bundle / relative).is_file(), f"缺失资源 {relative}"
        return {"pages": states, "resources": "配置、QSS、PNG、SVG存在；界面截图另附"}

    def import_observation(self, path):
        self.click("导入观测")
        menu = self.app.window(class_name="QMenu")
        menu.child_window(title="加载观测结果或记录文件", control_type="MenuItem").click_input()
        self.file_dialog("加载观测结果或记录文件", path)
        self.wait_text("已导入 ")
        wait_until(lambda: self.button("加载参数").is_enabled(), "观测导入未完成")

    def export_robot(self, destination):
        self.click("展开明细")
        dialog = self.dialog(".*当前比较明细")
        self.click("导出结果", dialog)
        self.file_dialog("导出定位结果", destination)
        wait_until(destination.is_file, "未生成导出 JSON")
        result = json.loads(destination.read_text(encoding="utf-8"))
        self.click("关闭", dialog)
        return result

    def robot(self):
        spec = self.manifest["robot"]
        self.tab("机器人末端定位精度")
        self.click("加载参数")
        self.file_dialog("加载视觉与手眼参数", self.data / spec["parameters"])
        wait_until(lambda: (self.user_data / "storage/position_monitoring/parameters/current.json").is_file(),
                   "参数未保存")
        wait_until(lambda: self.button("加载参数").is_enabled(), "参数加载未完成")
        self.import_observation(self.data / spec["baseline"])
        self.click("建立基准")
        self.wait_text("已建立，请导入复测观测")
        self.import_observation(self.data / spec["current"])
        self.click("评估精度")
        wait_until(lambda: self.button("评估精度").is_enabled(), "评估未完成")
        wait_until(lambda: len(self.robot_values()) == 4 and all(text != "—" for text in self.robot_values()),
                   "机器人指标未显示")
        actual = [float(value) for value in self.robot_values()]
        for value, expected in zip(actual, spec["expected"]["repeatability"]):
            assert math.isclose(value, expected, abs_tol=0.0001), (actual, spec["expected"])
        result = self.export_robot(self.output / "robot_result.json")
        assert result["batch_id"] == "B002" and result.get("groups")
        self.screenshot("02_robot_evaluated")
        return {"display_values_mm": actual, "export": "robot_result.json", "batch_id": result["batch_id"]}

    def robot_values(self):
        return [item.window_text() for item in self.window.descendants(control_type="Text")
                if item.element_info.automation_id.endswith(".MetricValueLabel")]

    def robot_images(self):
        spec = self.manifest["robot"]
        self.import_observation(self.data / spec["image_baseline"])
        self.click("建立基准")
        self.wait_text("已建立，请导入复测观测")
        self.import_observation(self.data / spec["image_current"])
        self.click("评估精度")
        wait_until(lambda: self.button("评估精度").is_enabled(), "图片评估未完成")
        result = self.export_robot(self.output / "robot_image_result.json")
        observed = list((self.user_data / "storage/position_monitoring/observations").glob("*.json"))
        batches = [json.loads(path.read_text(encoding="utf-8")) for path in observed]
        samples = [sample for batch in batches for sample in batch.get("samples", []) if sample.get("corner_count")]
        assert len(samples) == 6 and all(row["corner_count"] == 48 for row in samples)
        assert all(row["reprojection_error_px"] < 0.15 for row in samples)
        self.screenshot("03_robot_images")
        return {"detected_images": len(samples), "export": "robot_image_result.json", "batch_id": result["batch_id"]}

    def spindle_results(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in
                (self.user_data / "storage/spindle_monitoring/evaluations").glob("*.json")]

    def spindle(self):
        spec = self.manifest["spindle"]
        self.tab("主轴回转精度")
        for index, relative in enumerate(spec["packages"], 1):
            self.click("日常导入")
            self.file_dialog("导入一天检测数据", self.data / relative)
            wait_until(lambda: len(self.spindle_results()) >= index, "主轴导入未保存评价")
            wait_until(lambda: self.button("日常导入").is_enabled(), "主轴导入未完成")
        results = sorted(self.spindle_results(), key=lambda row: row["run_name"])
        for row, rms in zip(results, spec["expected"]["rms_acc1_mm_s"]):
            assert row["window_count"] == 10 and row["score"] is None
            assert math.isclose(row["rms_mm_s"][0], rms, rel_tol=1e-5)
            spectrum = row["spectrum"]
            peak = max(range(len(spectrum["amplitude_mm_s"][0])), key=spectrum["amplitude_mm_s"][0].__getitem__)
            assert spectrum["frequency_hz"][peak] == spec["expected"]["spectrum_peak_hz"]
        self.click("评估精度")
        wait_until(lambda: len(self.spindle_results()) == 3, "主轴重新评估未保存")
        wait_until(lambda: self.button("评估精度").is_enabled(), "主轴评估未完成")
        self.screenshot("04_spindle_imported")
        return {"imported_runs": len(results), "rms_mm_s": [row["rms_mm_s"][0] for row in results], "peak_hz": 120}

    def spindle_training(self):
        spec = self.manifest["spindle"]
        self.click("批量导入")
        self.file_dialog("批量导入建模数据（递归检索 ZIP）", self.data / spec["training_package"])
        label_dialog = self.dialog("整批数据判定")
        # 默认选项为正常；只标记本次隔离验收用的模拟样例。
        label_dialog.type_keys("{ENTER}")
        wait_until(lambda: self.button("训练网络").is_enabled(), "训练样本尚未准备完毕")
        self.click("训练网络")
        train_dialog = self.dialog("从头训练网络")
        edit = train_dialog.descendants(control_type="Edit")[0]
        edit.set_edit_text("1")
        self.click("开始训练", train_dialog)
        model_folder = self.user_data / "storage/spindle_monitoring/models"
        wait_until(lambda: bool(list(model_folder.rglob("*.pt"))), "未保存训练模型", timeout=300)
        wait_until(lambda: self.button("日常导入").is_enabled(), "训练后历史重算未完成", timeout=300)
        results = [row for row in self.spindle_results() if row.get("model_version")]
        assert results and all(row.get("score") is not None for row in results)
        previous_count = len(self.spindle_results())
        self.click("评估精度")
        wait_until(lambda: len(self.spindle_results()) > previous_count, "训练后推理未保存", timeout=180)
        wait_until(lambda: self.button("评估精度").is_enabled(), "训练后推理未完成")
        self.screenshot("05_spindle_trained")
        return {"epochs": 1, "models": [str(p.relative_to(self.user_data)) for p in model_folder.rglob("*.pt")],
                "evaluations_using_model": len(results), "scope": "只验证训练/保存/推理可运行，不验证模型精度"}

    def feed(self):
        self.tab("主轴轴向进给精度")
        self.click("理论窝深与报警设置")
        dialog = self.dialog("理论窝深与报警设置")
        for field in dialog.descendants(control_type="Edit"):
            if field.element_info.automation_id.endswith(".feedDepthErrorLower"):
                field.set_edit_text("-0.1")
            elif field.element_info.automation_id.endswith(".feedDepthErrorUpper"):
                field.set_edit_text("0.1")
        self.click("应用设置", dialog)
        self.wait_text("设置已保存")
        imported = []
        for name in ("normal", "out_of_limit"):
            for relative in self.manifest["feed"][name]:
                self.click("导入数据")
                self.file_dialog("选择窝深数据", self.data / relative)
                wait_until(lambda: self.button("评估精度").is_enabled(), "窝深数据未载入")
                self.click("评估精度")
                wait_until(lambda: self.button("导入数据").is_enabled(), "窝深评估未完成")
                texts = self.texts()
                assert "5" in texts and "1.500" in texts
                variance = self.manifest["feed"]["expected"][name]["variance_mm2"]
                assert f"{variance:.6f}" in texts, texts
                self.wait_text("当前批次未超限" if name == "normal" else "2 孔超限")
                imported.append(relative)
        self.click("孔位明细")
        details = self.dialog("孔位窝深明细")
        assert any("缺测" in item.window_text() for item in details.descendants())
        self.click("关闭", details)
        self.screenshot("06_feed_evaluated")
        return {"imported": imported, "valid_count": 5, "mean_mm": 1.5, "missing_preserved": True}

    def dependencies(self):
        import win32api
        import win32con
        import win32process

        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                                      False, self.process.pid)
        try:
            loaded = sorted(win32process.GetModuleFileNameEx(handle, module)
                            for module in win32process.EnumProcessModules(handle))
        finally:
            handle.Close()
        (self.output / "loaded_modules.json").write_text(json.dumps(loaded, ensure_ascii=False, indent=2), encoding="utf-8")
        python_dlls = [Path(name) for name in loaded if Path(name).name.lower().startswith("python")]
        assert python_dlls and all(path.is_relative_to(self.exe.parent) for path in python_dlls)
        escaped = [name for name in loaded if "miniconda" in name.lower() or "anaconda" in name.lower()]
        assert not escaped, escaped
        return {"loaded_module_count": len(loaded), "python_runtime": [str(path) for path in python_dlls],
                "details": "loaded_modules.json", "scope": "清理PATH后的本机加载检查，不等于干净系统兼容性"}

    def restart(self, cwd):
        self.stop()
        assert not self.report.get("shutdown_forced") and self.report["shutdown_exit_code"] == 0
        self.start(cwd)
        wait_until(lambda: len(self.robot_values()) == 4 and all(value != "—" for value in self.robot_values()),
                   "重启未恢复机器人指标")
        self.tab("主轴回转精度")
        assert self.button("评估精度").is_enabled()
        self.tab("主轴轴向进给精度")
        assert self.button("导入数据").is_enabled() and not self.button("评估精度").is_enabled()
        settings = json.loads((self.user_data / "storage/feed_depth/settings.json").read_text(encoding="utf-8"))
        assert settings["error_lower_mm"] == -0.1 and settings["error_upper_mm"] == 0.1
        self.screenshot("07_restarted")
        return {"robot_restored": True, "spindle_restored": True, "feed_settings_restored": True,
                "feed_records": "会话内记录，重启无载入数据符合现状"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("此脚本只在 Windows 桌面运行；麒麟按人工验收清单执行。")
    exe, data, output = args.exe.resolve(), args.test_data.resolve(), args.output.resolve()
    if not exe.is_file() or exe.suffix.lower() != ".exe":
        parser.error("--exe 必须指向已构建的最终 EXE")
    if (output / "user_data").exists():
        parser.error("输出目录已有验收数据；请换一个新目录，避免混入上次结果。")
    output.mkdir(parents=True, exist_ok=True)
    if data.suffix.lower() == ".zip":
        with ZipFile(data) as archive:
            archive.extractall(output / "test_data")
        data = output / "test_data"
    verification = WindowsVerification(exe, data, output)
    with tempfile.TemporaryDirectory(prefix="ZM_发布验收_") as cwd:
        try:
            verification.start(Path(cwd))
            verification.check("启动、三页按钮和运行资源", verification.startup)
            if verification.check("机器人参数、JSON导入、基准、评估与导出", verification.robot):
                verification.check("机器人PNG检测、图片基准与评估导出", verification.robot_images)
            if verification.check("主轴ZIP、HDF5读取、频谱、RMS及重新评估", verification.spindle):
                verification.check("CPU网络1轮训练、模型保存和推理", verification.spindle_training)
            verification.check("窝深CSV/Excel导入、评估、边界与缺测", verification.feed)
            verification.check("Python运行库随包加载、无Conda DLL依赖", verification.dependencies)
            verification.check("正常关闭、重启与已存数据恢复", lambda: verification.restart(Path(cwd)))
        except Exception:
            verification.report["fatal_error"] = traceback.format_exc()
        finally:
            verification.stop()
            verification.report["bundle_unchanged"] = (
                verification.report["bundle_files"] == bundle_files(exe.parent)
            )
            verification.report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            passed = (not verification.report.get("fatal_error") and verification.report["checks"]
                      and all(row["status"] == "passed" for row in verification.report["checks"])
                      and not verification.report.get("shutdown_forced")
                      and verification.report.get("shutdown_exit_code") == 0
                      and verification.report["bundle_unchanged"])
            verification.report["status"] = "passed" if passed else "failed"
            verification.save_report()
    print(str(output / "report.json"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
