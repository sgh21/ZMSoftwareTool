"""发布审计必须读取最终可执行文件内部，且不误判第三方运行时模块。"""

from pathlib import Path

import pytest

from tools.release.audit_bundle import RUNTIME_MODULES, audit, forbidden_file, forbidden_module, forbidden_torch_binary


writers = pytest.importorskip("PyInstaller.archive.writers")


def make_bundle(tmp_path, modules, *, embedded_file=None):
    bundle = tmp_path / "SoftwareTools"
    bundle.mkdir()
    pyz_path = tmp_path / "PYZ.pyz"
    entries = [(name, str(tmp_path / (name + ".py")), "PYMODULE") for name in modules]
    code = {name: compile("value = 1", name + ".py", "exec") for name in modules}
    writers.ZlibArchiveWriter(str(pyz_path), entries, code_dict=code)
    archive_entries = [("PYZ.pyz", str(pyz_path), False, "z")]
    if embedded_file:
        data = tmp_path / "archive-data.txt"
        data.write_text("software verification input", encoding="utf-8")
        archive_entries.append((embedded_file, str(data), False, "x"))
    # 真实 PyInstaller 归档格式；这里只需其目录，无须构建/启动一个额外程序。
    writers.CArchiveWriter(str(bundle / "SoftwareTools.exe"), archive_entries, "python312.dll")
    return bundle


def test_embedded_project_tests_and_development_modules_are_rejected(tmp_path):
    modules = ["app.main_window", "tests.test_main_window", "debug.reset", "experiments.replay",
               "tools.release.build", "_pytest.config", "pytest", "numpy.tests.test_numeric",
               "py", "pylab", "mpl_toolkits.mplot3d"]
    report = audit(make_bundle(tmp_path, modules))
    assert report["forbidden_files"] == []  # 普通目录检查无法发现这些模块。
    assert report["forbidden_modules"] == sorted(modules[1:])
    assert report["module_count"] == len(modules)
    assert report["pyz_archives"] == ["PYZ.pyz"] and not report["passed"]


def test_runtime_debug_and_test_compatibility_helpers_are_kept(tmp_path):
    modules = ["app.main_window", "core.services.position_simulation_debug",
               "torch.distributed.debug", "numpy._pytesttester", "scipy._lib._testutils", *sorted(RUNTIME_MODULES)]
    bundle = make_bundle(tmp_path, modules)
    for name in ("torch/testing/__init__.py", "torch/testing/_internal/__init__.py",
                 "torch/testing/_internal/logging_tensor.py", "torch/testing/_comparison.py"):
        source = bundle / "_internal" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("runtime source fixture", encoding="utf-8")
    report = audit(bundle)
    assert report["passed"] and report["forbidden_modules"] == []
    assert (tmp_path / "verification/bundle-audit.json").is_file()


@pytest.mark.parametrize("name", ["numpy.testing.tests.test_utils", "numpy.testing._private.new_helper",
                                  "torch.testing._internal.common_utils", "torch.testing._internal.opinfo",
                                  "scipy._lib.array_api_extra.testing.tests", "jinja2.tests.test_extra"])
def test_helper_allowlist_does_not_include_suites_or_unlisted_internal_modules(name):
    assert forbidden_module(name)
    assert forbidden_file(Path("_internal") / (name.replace(".", "/") + ".py"))


@pytest.mark.parametrize("name", ["torch/bin/HashStoreTest", "torch/bin/FileStoreTest", "torch/bin/TCPStoreTest",
                                  "torch/bin/protoc", "torch/bin/protoc-3.13.0.0", "torch/bin/protoc.exe",
                                  "torch/lib/libtorchbind_test.so", "torch/lib/libjitbackend_test.so"])
def test_torch_test_binaries_and_compiler_are_rejected(name):
    assert forbidden_torch_binary(name)
    assert forbidden_file(Path("_internal") / name)


@pytest.mark.parametrize("name", ["torch/bin/torch_shm_manager", "torch/lib/libtorch_cpu.so",
                                  "torch/lib/torch_cpu.dll", "torch/lib/libtorch_python.so",
                                  "numpy/_core/_multiarray_tests.cpython-312-x86_64-linux-gnu.so"])
def test_required_native_libraries_are_not_classified_by_substring(name):
    assert not forbidden_torch_binary(name)
    assert not forbidden_file(Path("_internal") / name)


def test_development_sample_is_rejected_inside_or_outside_executable(tmp_path):
    bundle = make_bundle(tmp_path, ["app.main_window"], embedded_file="config/examples/sample.json")
    extra = bundle / "_internal" / "tests" / "fixture.csv"
    extra.parent.mkdir(parents=True)
    extra.write_text("a,b\n1,2", encoding="utf-8")
    report = audit(bundle)
    assert report["forbidden_archive_files"] == [str(Path("config/examples/sample.json"))]
    assert report["forbidden_files"] == [str(Path("_internal/tests/fixture.csv"))]
    assert not report["passed"]


def test_empty_output_cannot_pass_without_an_executable(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)
