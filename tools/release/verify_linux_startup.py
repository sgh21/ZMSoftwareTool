"""外部检查最终 Linux 可执行文件；不向软件内嵌任何测试入口。"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def inspect_elf(executable, output):
    files = []
    for path in executable.parent.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                files.append(path)
    requirements = {}
    for path in files:
        result = subprocess.run(["readelf", "--version-info", str(path)], check=True,
                                text=True, encoding="utf-8", capture_output=True)
        versions = sorted(set(re.findall(r"GLIBC_(\d+(?:\.\d+)+)", result.stdout)),
                          key=lambda value: tuple(map(int, value.split("."))))
        if versions:
            requirements[str(path.relative_to(executable.parent))] = versions[-1]
    highest = max(requirements.values(), key=lambda value: tuple(map(int, value.split("."))))
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(sorted({str(path.parent) for path in files}))
    required = [executable, *executable.parent.rglob("libqxcb.so")]
    missing = []
    with (output / "ldd-required.txt").open("w", encoding="utf-8") as stream:
        for path in required:
            result = subprocess.run(["ldd", str(path)], env=environment,
                                    text=True, encoding="utf-8", capture_output=True)
            stream.write(f"{path.relative_to(executable.parent)}\n{result.stdout}{result.stderr}\n")
            missing.extend(line.strip() for line in result.stdout.splitlines() if "not found" in line)
    report = {"elf_count": len(files), "glibc_maximum_required": highest,
              "glibc_by_file": requirements, "required_libraries_missing": missing}
    (output / "elf-requirements.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if tuple(map(int, highest.split("."))) > (2, 31):
        raise RuntimeError(f"Bundle requires GLIBC {highest}, above the Kylin V10 Desktop 2.31 baseline.")
    if missing:
        raise RuntimeError(f"Missing executable/X11 dependencies: {missing}")
    if len(required) < 2:
        raise RuntimeError("The bundle has no Qt xcb platform plugin.")
    return report


def launch_with_xvfb(executable, output):
    for command in ("Xvfb", "xwininfo"):
        if not shutil.which(command):
            raise RuntimeError(f"Install {command} to perform the external X11 startup check.")
    with TemporaryDirectory(prefix="softwaretools-startup-") as temporary:
        isolated = Path(temporary)
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "QT_PLUGIN_PATH",
                     "QT_QPA_PLATFORM_PLUGIN_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            environment.pop(name, None)
        environment.update({
            "HOME": str(isolated), "PATH": "/usr/bin:/bin",
            "ZMSOFTWARE_DATA_DIR": str(isolated / "user-data"),
            "QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99",
            "XDG_RUNTIME_DIR": str(isolated / "runtime"),
        })
        (isolated / "runtime").mkdir(mode=0o700)
        with (output / "xvfb.log").open("w", encoding="utf-8") as xvfb_log:
            server = subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1600x1000x24", "-nolisten", "tcp"],
                                      env=environment, stdout=xvfb_log, stderr=subprocess.STDOUT)
            application = None
            try:
                for _ in range(50):
                    probe = subprocess.run(["xwininfo", "-root"], env=environment,
                                           capture_output=True)
                    if probe.returncode == 0:
                        break
                    if server.poll() is not None:
                        raise RuntimeError("Xvfb exited before the display became available.")
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Xvfb did not become ready within 5 seconds.")
                with (output / "application.log").open("w", encoding="utf-8") as app_log:
                    application = subprocess.Popen([str(executable)], cwd=isolated, env=environment,
                                                   stdout=app_log, stderr=subprocess.STDOUT)
                    parameter_path = isolated / "user-data/storage/position_monitoring/parameters/current.json"
                    title_found = False
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if application.poll() is not None:
                            raise RuntimeError(f"Final executable exited during startup: {application.returncode}")
                        tree = subprocess.run(["xwininfo", "-root", "-tree"], env=environment, check=True,
                                              text=True, encoding="utf-8", errors="replace", capture_output=True)
                        (output / "window-tree.txt").write_text(tree.stdout, encoding="utf-8")
                        title_found = "精度监控系统" in tree.stdout
                        if title_found and parameter_path.is_file():
                            break
                        time.sleep(0.5)
                    if not title_found or not parameter_path.is_file():
                        raise RuntimeError("The main window and isolated user-data initialization were not both observed.")
                    # Catch delayed startup-thread errors before declaring the startup check complete.
                    time.sleep(3)
                    if application.poll() is not None:
                        raise RuntimeError(f"Final executable exited after showing its window: {application.returncode}")
                    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
                    return {"main_window_observed": title_found, "user_parameters_initialized": bool(parameters),
                            "qpa_platform": "xcb", "display": "Xvfb", "working_directory": "isolated temporary directory",
                            "development_python_environment_removed": True}
            finally:
                if application is not None and application.poll() is None:
                    application.terminate()
                    application.wait(timeout=10)
                server.terminate()
                server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this check on Linux.")
    executable, output = args.exe.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"executable": str(executable), "platform": platform.platform(),
              "machine": platform.machine(), "kylin_v10_hardware_tested": False,
              "scope": "Final executable X11 startup and ELF audit in the build container; no business-operation acceptance."}
    try:
        report["elf"] = inspect_elf(executable, output)
        report["startup"] = launch_with_xvfb(executable, output)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "startup-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Final Linux executable started under Xvfb. Kylin V10 hardware acceptance remains pending.")


if __name__ == "__main__":
    main()
