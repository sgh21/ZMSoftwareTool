"""交付只收实际产物与明确证据，不能把缺失/过期验证包装成成功。"""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.release.assemble_delivery import add_files, assemble_kylin, assemble_windows, write_delivery


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evidence_collection_does_not_copy_test_or_user_data(tmp_path):
    for name in ("report.json", "01_startup.png", "user_data/private.json", "test_data/simulated.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    files = {}
    add_files(files, tmp_path, "evidence", ("*.json", "*.png"))
    assert set(files) == {"evidence/report.json", "evidence/01_startup.png"}


@pytest.mark.parametrize("problem", ["failed", "stale", "dll_changed"])
def test_windows_requires_passed_report_for_the_same_executable(tmp_path, problem):
    executable = tmp_path / "dist/SoftwareTools/SoftwareTools.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"archive fixture")
    report = {
        "status": "failed" if problem == "failed" else "passed",
        "checks": [{"name": "fixture", "status": "passed"}],
        "executable_bytes": executable.stat().st_size + (problem == "stale"),
        "executable_mtime_ns": executable.stat().st_mtime_ns,
    }
    if problem == "dll_changed":
        report["bundle_files"] = {"SoftwareTools.exe": {"bytes": executable.stat().st_size,
                                                       "mtime_ns": executable.stat().st_mtime_ns}}
        report["bundle_unchanged"] = True
        library = executable.parent / "_internal" / "dependency.dll"
        library.parent.mkdir()
        library.write_bytes(b"changed dependency fixture")
    write_json(tmp_path / "dist/verification/windows_gui_final/report.json", report)
    write_json(tmp_path / "dist/verification/build-windows.json", {"build_succeeded": True})
    with pytest.raises(RuntimeError, match="未通过|不同于|完整发布目录已改变"):
        assemble_windows(tmp_path, tmp_path / "delivery")
    assert not (tmp_path / "delivery/Windows版.zip").exists()


def test_missing_linux_build_does_not_create_placeholder_delivery(tmp_path):
    with pytest.raises(FileNotFoundError):
        assemble_kylin(tmp_path, tmp_path / "delivery")
    assert not (tmp_path / "delivery/麒麟V10版.zip").exists()


def test_zip_contains_only_explicit_inputs_and_readable_chinese_text(tmp_path):
    payload = tmp_path / "runtime.bin"
    payload.write_bytes(b"software fixture")
    destination = write_delivery(tmp_path / "delivery/Windows版.zip", {"SoftwareTools/runtime.bin": payload},
                                 {"运行说明.txt": "验收说明：尚未验证目标机。"})
    with ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"SoftwareTools/runtime.bin", "运行说明.txt"}
        assert archive.read("SoftwareTools/runtime.bin") == payload.read_bytes()
        assert "尚未验证" in archive.read("运行说明.txt").decode("utf-8-sig")
    assert not Path(str(destination) + ".tmp").exists()
