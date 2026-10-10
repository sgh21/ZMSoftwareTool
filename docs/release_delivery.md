# Windows / 麒麟 V10 发布交付

发布前的界面及导入功能已提交并推送：`3beaf04`。本次整理在分支 `codex/release-windows-kylin-v10` 进行，不改变页面布局或统计协议。

## 目标与验证边界

| 目标 | 架构与条件 | 验证环境 |
| --- | --- | --- |
| Windows 11，用户指定 Intel i5-13500K | x86_64，普通桌面用户；CPU 版 PyTorch，不需要 Python、Conda 或 CUDA | 本地 Windows 11 家庭中文版 10.0.26300，i5-14600KF；可做最终 EXE 操作测试，不能视为干净新机器或指定处理器验收 |
| 麒麟 V10 桌面版，海光 3550 | x86_64；已构建二进制要求 glibc ≥ 2.28、系统图形库及中文字体；具体 SP 版本未提供 | Linux manylinux 2.28 容器构建、Xvfb 窗口启动；无麒麟桌面实机，不能认定目标系统业务验收通过 |

本机没有麒麟、WSL Linux 发行版或干净 Windows 虚拟机。Linux 构建通过仓库 GitHub Actions 执行。每份交付中的测试报告以实际文件为依据，区分源码测试、最终程序检查与未执行项目。

PyInstaller 必须分别在 Windows / Linux 构建；Linux 的 glibc 不随程序打包，构建使用较旧运行库环境，实际兼容性仍须现场核对。依据：[PyInstaller 平台说明](https://www.pyinstaller.org/en/stable/)及[Linux 兼容限制](https://pyinstaller.org/en/stable/usage.html)。麒麟公开桌面 V10 资料列出的 glibc 为 2.31，实际安装版本以 `ldd --version` 为准：[麒麟产品资料](https://kylinos.cn/upload/product/20230509/0dff0c074eee307520f38eb792a2d168.pdf)。

## 软件与数据

- 软件本体仅包含入口、业务实现、运行依赖、配置和界面资源。测试、调试 CLI、实验脚本、开发样例、自动化工具与缓存均排除；审计还检查可执行文件内嵌 PYZ，不能仅检查外层目录。
- `测试数据.zip` 独立存放，含目录结构与简短操作说明。全部为人工构造的软件验证样例，没有用户原始采集、历史、权重或测试代码。
- 源码运行沿用仓库数据目录。发布版默认写入 Windows 的 `%LOCALAPPDATA%\SoftwareTools_PyQt`，Linux 的 `${XDG_DATA_HOME:-$HOME/.local/share}/SoftwareTools_PyQt`。业务服务在其下保存 `storage/` 等目录。
- 可用 `ZMSOFTWARE_DATA_DIR` 指定独立可写数据根目录；发布版不写安装目录，不依赖启动时工作目录。旧数据不会自动迁移，需由用户按实际需要导入或备份后复制对应存储目录。
- 标定实现已从开发调试目录迁至 `core/services/camera_calibration.py`，原调试 CLI 保留兼容入口，生产界面不再依赖 `debug/`。

## 启动

Windows：完整解压 `Windows版.zip`，双击 `SoftwareTools/SoftwareTools.exe`；必须保留同级 `_internal`，不要单独复制 EXE。系统需有正常桌面图形环境；如系统提示缺少 MSVC 运行库，可按[微软说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)安装 Visual C++ v14 Redistributable x64。包内包含 Python 与 Qt 等依赖。无需管理员权限。

麒麟：解压 `麒麟V10版.zip` 后，执行 `tar -xzf SoftwareTools-linux-x86_64.tar.gz`，再运行 `./SoftwareTools/SoftwareTools`。内层 tar 保留执行权限与符号链接。可从其他目录用绝对路径启动；Wayland 桌面如插件不可用，可使用 `QT_QPA_PLATFORM=xcb ./SoftwareTools/SoftwareTools`（需系统 XWayland）。

麒麟需要系统已有 X11/XCB、xkbcommon、OpenGL/EGL、fontconfig、FreeType、D-Bus 运行库及中文字体。可用 `ldd SoftwareTools/SoftwareTools` 及 `ldd SoftwareTools/_internal/PyQt6/Qt6/plugins/platforms/libqxcb.so` 检查缺失项。常见 Debian 系包名包括 `libxcb-cursor0`、`libxkbcommon-x11-0`、`libxcb-icccm4`、`libxcb-image0`、`libxcb-keysyms1`、`libxcb-render-util0`、`libxcb-xinerama0`、`libgl1`、`libegl1`；实际名称及安装方式以麒麟软件源为准。Windows 使用现有微软雅黑设置，麒麟由 Qt 选择已安装中文字体，不分发微软字体。

## 重建

在现有 Python 3.12 x64 构建环境运行，不在项目中新建虚拟环境。完整锁定版本分别见 `tools/release/requirements-windows.lock` 和 `requirements-kylin.lock`；构建输出另记录实际依赖。

Windows 实际使用现有 `ZMSoftware` 环境中的 CPython 3.12.13。3.12.0 在冻结 SciPy 时存在已确认的代码对象缺陷，构建脚本会拒绝该版本；本次通过 `conda install -p D:/Softwares/miniconda3/envs/ZMSoftware python=3.12.13 --freeze-installed` 修复原环境，并保留升级计划与日志。无需另建环境。[上游修复说明](https://github.com/pyinstaller/pyinstaller/issues/8186)

Windows PowerShell：

```powershell
./tools/release/build_windows.ps1 -Python 'D:/Softwares/miniconda3/envs/ZMSoftware/python.exe' -InstallDependencies
python -B -X utf8 tools/release/prepare_test_data.py
python -m pip install -r tools/release/verification-requirements-windows.txt
python -B tools/release/verify_windows.py --exe dist/SoftwareTools/SoftwareTools.exe --test-data dist/release_inputs/测试数据.zip --output dist/verification/windows_gui_final
```

`-Python` 可指定其他已有 Python 3.12。构建先运行正式测试再打包；`-SkipTests` 只用于同一源码已验证后的重打包，不能代替发布测试。CPU PyTorch 2.9.0 安装到 Git 忽略的 `build/release_deps/windows`，不替换开发环境的 CUDA PyTorch。

Linux：推送构建相关变更到 `codex/release-windows-kylin-v10` 即触发 `.github/workflows/build-kylin-v10.yml`；也可在已有 Linux x86_64 / CPython 3.12 环境执行：

```bash
PYTHON=/opt/softwaretools-python/bin/python3.12 \
LD_LIBRARY_PATH=/opt/softwaretools-python/lib \
bash tools/release/build_kylin.sh
```

完整容器图形库安装命令在 workflow 中。manylinux 自带解释器缺少冻结所需的 Python 共享库；workflow 先通过 `build_shared_python.sh` 编译带共享库的 CPython 3.12.15，并缓存该构建环境。编译配置和实际依赖均记录在报告中，程序自身仍携带运行所需 Python，不要求目标机安装它。

Qt 固定 6.7、CPU PyTorch 固定 2.9，避免新 Qt 的 glibc 要求高于麒麟 V10。产物 `dist/SoftwareTools-linux-x86_64.tar.gz` 和 `dist/linux-build-report/` 由 Actions 上传；下载后按原目录放入本地 `dist/`，再运行交付组装脚本：

```powershell
python -B -X utf8 tools/release/assemble_delivery.py
```

最终输出 `dist/delivery/Windows版.zip` 与 `dist/delivery/麒麟V10版.zip`。构建及验证脚本留在源码仓库，不放入软件本体。重新打包后须对新产物执行对应验收；不能沿用旧 EXE 的通过记录。

## 已知限制与现场验收

完整清单见 [最终软件包验收](../tools/release/验收步骤.md)。必须在未安装开发环境的目标系统完成启动、资源、三页导入/评估、机器人导出、主轴训练/推理、关闭及重启检查。

当前未完成麒麟 V10 / 海光实机和干净 Windows 11 环境验证；Linux 容器窗口启动不覆盖三页业务 GUI。现场系统 SP、驱动、字体、屏幕缩放仍需记录。

本次 Windows 发布为未做代码签名的便携目录包；首次启动若遇系统信誉提示，按单位的软件安装策略处理。

轴向窝深导入历史只在本次会话保留，设置跨启动保存；相机在线采集、机器人/主轴设备控制、加工点位预测接口尚未接入。本次保留这些现有行为。模拟样例的一轮训练只验证训练及推理可运行，不证明模型或设备测量精度。源码整理保留范围及未完成的本地缓存清理见 [整理审计](release_source_audit.md)。
