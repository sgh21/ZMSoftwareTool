"""Keep PyTorch runtime/source inspection; omit its test suites and C++ SDK."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False


def runtime_module(name):
    parts = name.split(".")
    return (not set(parts) & {"tests", "test", "testing", "benchmark", "tensorboard"}
            and not parts[-1].startswith("test_"))


hiddenimports = collect_submodules("torch", filter=runtime_module)
datas = collect_data_files("torch", excludes=[
    "**/tests/**", "**/test/**", "**/testing/**", "**/include/**",
    "**/*.h", "**/*.hpp", "**/*.cuh", "**/*.cpp", "**/*.lib", "**/*.pyi", "**/*.cmake",
])
binaries = collect_dynamic_libs("torch")
bindepend_symlink_suppression = ["**/torch/lib/*.so*"]
