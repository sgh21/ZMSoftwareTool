"""Keep PyTorch runtime/source inspection; omit its test suites and C++ SDK."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
from tools.release.audit_bundle import RUNTIME_MODULES, forbidden_torch_binary

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False


def runtime_module(name):
    if name in RUNTIME_MODULES:
        return True
    parts = name.split(".")
    return (not set(parts) & {"tests", "test", "testing", "benchmark", "tensorboard"}
            and not parts[-1].startswith("test_"))


hiddenimports = collect_submodules("torch", filter=runtime_module)
datas = collect_data_files("torch", excludes=[
    "**/tests/**", "**/test/**", "**/testing/**", "**/include/**",
    "**/*.h", "**/*.hpp", "**/*.cuh", "**/*.cpp", "**/*.lib", "**/*.pyi", "**/*.cmake",
])
binaries = collect_dynamic_libs("torch")
# 在二进制依赖分析前移除已确认的测试程序与 protobuf 编译器；保留 shm 管理器。
datas = [entry for entry in datas if not forbidden_torch_binary(Path(entry[1]) / Path(entry[0]).name)]
binaries = [entry for entry in binaries if not forbidden_torch_binary(Path(entry[1]) / Path(entry[0]).name)]
bindepend_symlink_suppression = ["**/torch/lib/*.so*"]
