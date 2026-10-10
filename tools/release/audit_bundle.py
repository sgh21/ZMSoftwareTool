"""Reject project tests, sample data and development tools in the runtime bundle."""
import argparse
import json
from pathlib import Path


DEVELOPMENT_ROOTS = {
    "tests", "test", "debug", "experiments", "diagnostics", "tools", "pytest", "_pytest",
    "pywinauto", "comtypes", "notebook", "ipython", "jupyter", "jupyter_client", "jupyter_core",
    "ipykernel", "pyinstaller", "py", "pylab", "mpl_toolkits",
}
CACHE_PARTS = {".git", ".vscode", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
TEST_PARTS = {"tests", "test", "testing"}
# 精确保留正常运行导入链需要的帮助模块；不按 testing 包前缀整包放行。
RUNTIME_MODULES = {
    "jinja2.tests",
    "numpy.testing", "numpy.testing._private", "numpy.testing.overrides",
    "numpy.testing._private.extbuild", "numpy.testing._private.utils",
    "scipy._lib.array_api_extra.testing",
    "torch.testing", "torch.testing._utils", "torch.testing._comparison", "torch.testing._creation",
    "torch.testing._internal", "torch.testing._internal.logging_tensor",
}


def forbidden_torch_binary(relative):
    parts = [part.lower() for part in Path(str(relative).replace("\\", "/")).parts]
    if parts and parts[0] == "_internal":
        parts = parts[1:]
    name = parts[-1]
    if parts[:2] == ["torch", "bin"]:
        return name.removesuffix(".exe") in {"hashstoretest", "filestoretest", "tcpstoretest"} or name.startswith("protoc")
    return parts[:2] == ["torch", "lib"] and name in {
        "libtorchbind_test.so", "libjitbackend_test.so", "torchbind_test.dll", "jitbackend_test.dll",
    }


def runtime_helper_file(relative):
    parts = [part.lower() for part in Path(str(relative).replace("\\", "/")).parts]
    if parts and parts[0] == "_internal":
        parts = parts[1:]
    if parts[-1] == "__init__.py":
        parts.pop()
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    else:
        return False
    return ".".join(parts) in RUNTIME_MODULES


def forbidden_module(name):
    name = name.replace("\\", ".").replace("/", ".").lower()
    if name in RUNTIME_MODULES:
        return False
    parts = name.split(".")
    return (parts[0] in DEVELOPMENT_ROOTS or bool(set(parts) & TEST_PARTS)
            or parts[-1].startswith("test_")
            or name in {"torch.utils.benchmark", "torch.utils.tensorboard"})


def forbidden_file(relative):
    if forbidden_torch_binary(relative):
        return True
    if runtime_helper_file(relative):
        return False
    parts = [part.lower() for part in relative.parts]
    if parts and parts[0] == "_internal":
        parts = parts[1:]
    return (parts[0] in DEVELOPMENT_ROOTS or bool(set(parts) & (CACHE_PARTS | TEST_PARTS))
            or parts[:2] == ["config", "examples"]
            or relative.name.startswith("test_")
            or relative.suffix.lower() in {".h5", ".xlsx", ".csv", ".log"})


def audit_executable(executable):
    """只读取已构建可执行文件的 CArchive/PYZ 目录，不执行其中的模块。"""
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(executable))
    modules, pyz_archives, files = [], [], []
    for name, entry in archive.toc.items():
        typecode = entry[-1]
        if typecode == "z":
            pyz = archive.open_embedded_archive(name)
            pyz_archives.append(name)
            modules.extend(pyz.toc)
        elif typecode in {"s", "m", "M"}:
            modules.append(name.removesuffix(".py"))
        elif typecode in {"x", "b"} and forbidden_file(Path(name.replace("\\", "/"))):
            files.append(name)
    return {
        "executable": executable.name, "pyz_archives": pyz_archives, "module_count": len(modules),
        "forbidden_modules": sorted(name for name in modules if forbidden_module(name)),
        "forbidden_archive_files": sorted(files),
    }


def audit(path):
    path = Path(path).resolve()
    problems, count, size = [], 0, 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        count += 1
        size += item.stat().st_size
        if forbidden_file(relative):
            problems.append(str(relative))
    executable = path / ("SoftwareTools.exe" if (path / "SoftwareTools.exe").is_file() else "SoftwareTools")
    embedded = audit_executable(executable)
    result = {
        "bundle": path.name, "files": count, "bytes": size, "forbidden_files": sorted(problems), **embedded,
        "passed": not (problems or embedded["forbidden_modules"] or embedded["forbidden_archive_files"])
                  and bool(embedded["pyz_archives"]),
    }
    report = path.parent / "verification/bundle-audit.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    args = parser.parse_args()
    result = audit(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
