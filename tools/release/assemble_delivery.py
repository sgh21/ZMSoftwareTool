"""按明确清单组装两平台交付 ZIP；缺失产物或验证失败时不生成该平台交付包。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_test_summary(path):
    tree = ET.parse(path)
    suites = list(tree.getroot().iter("testsuite"))
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    return (f"{totals['tests']} 项；失败 {totals['failures']}，错误 {totals['errors']}，"
            f"跳过 {totals['skipped']}（以原始 XML 为准）")


def add_files(files, folder, prefix, patterns):
    """只收指定目录的直接文件，禁止递归带入验收 user_data / test_data。"""
    for pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            if path.is_file():
                files[f"{prefix}/{path.name}"] = path


def write_delivery(destination, files, texts):
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少交付文件：" + ", ".join(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        for name, text in sorted(texts.items()):
            archive.writestr(name, text.encode("utf-8-sig"))
    with ZipFile(temporary) as archive:
        failed = archive.testzip()
        if failed:
            raise RuntimeError(f"交付 ZIP 校验失败：{failed}")
    temporary.replace(destination)
    return destination


ACCEPTANCE = """目标机验收步骤

本文件是待执行步骤，不是已完成报告。请在目标系统干净账户/新机器执行，并记录系统版本、CPU架构和结果。
1. 不安装开发 Python/Conda，不加入开发目录 PATH；解压完整交付包到普通用户可读目录。
2. 使用独立用户或 ZMSOFTWARE_DATA_DIR 指定空目录，从与程序目录不同的工作目录启动。
3. 确认三页可打开，中文、品牌图、图标、样式、图表可见；无数据时评估置灰并有悬停提示。
4. 解压独立“测试数据.zip”，按其中“使用说明.txt”完成机器人JSON基准/复测与PNG导入。
5. 日常导入两个主轴ZIP，核对120 Hz频谱及RMS；在独立模型存储中批量导入“小型建模.zip”，
   按说明确认模拟标签并进行1轮CPU训练，再导入日常包检查推理和模型/历史恢复。
6. 分别导入窝深CSV与XLSX，核对正常、边界、超限和缺测结果；取消/错误导入应保留原结果。
7. 在各页可用导出入口保存结果并重新打开；核对参数、基准、模型和历史的重启恢复。
   窝深导入记录只保留本次会话，设置跨启动保存，重启后窝深数据为空属于当前已知限制。
8. 关闭程序，保留验收记录/截图与真实错误；模拟评分不等于现场设备精度或模型有效性。

出现错误请记录启动目录、系统版本、日志和复现步骤，不要把开发机测试结果替代目标机验收。
"""


def assemble_windows(root, output, report_directory=None):
    bundle = root / "dist/SoftwareTools"
    executable = bundle / "SoftwareTools.exe"
    verification = root / "dist/verification"
    report_directory = report_directory or verification / "windows_gui_final"
    report = read_json(report_directory / "report.json")
    build = read_json(verification / "build-windows.json")
    if not build.get("build_succeeded") or report.get("status") != "passed":
        raise RuntimeError("Windows 构建或最终程序验证未通过，不生成 Windows版.zip。")
    if any(check.get("status") != "passed" for check in report.get("checks", [])) or not report.get("checks"):
        raise RuntimeError("Windows GUI 报告存在未通过项或没有检查记录。")
    stat = executable.stat()
    if stat.st_size != report["executable_bytes"] or stat.st_mtime_ns != report["executable_mtime_ns"]:
        raise RuntimeError("Windows EXE 已不同于 GUI 报告中的测试对象，请重新验证最终 EXE。")
    current_files = {}
    for path in bundle.rglob("*"):
        if path.is_file():
            info = path.stat()
            current_files[path.relative_to(bundle).as_posix()] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if report.get("bundle_files") != current_files or report.get("bundle_unchanged") is not True:
        raise RuntimeError("Windows 完整发布目录已改变或缺少验证清单，请重新验证包含 _internal 的最终程序。")
    subprocess.run([sys.executable, "-B", str(root / "tools/release/audit_bundle.py"), str(bundle)],
                   check=True, capture_output=True, text=True, encoding="utf-8")
    audit = read_json(verification / "bundle-audit.json")
    files = {"SoftwareTools/" + path.relative_to(bundle).as_posix(): path
             for path in bundle.rglob("*") if path.is_file()}
    files.update({
        "测试数据.zip": root / "dist/release_inputs/测试数据.zip",
        "依赖版本/requirements-windows.lock": root / "tools/release/requirements-windows.lock",
        "依赖版本/build-windows.json": verification / "build-windows.json",
        "验收证据/bundle-audit.json": verification / "bundle-audit.json",
        "验收证据/source-tests.xml": verification / "source-tests.xml",
        "验收证据/source-tests.log": verification / "source-tests.log",
    })
    add_files(files, report_directory, "验收证据/Windows", ("*.json", "*.png", "process.log"))
    checks = "\n".join(f"- {check['name']}：{check['status']}" for check in report["checks"])
    pending = "\n".join(f"- {value}" for value in report.get("not_verified", []))
    testing = f"""# Windows 发布测试报告

构建记录：{build['platform']} / {build['architecture']}，提交 `{build['source_commit']}`。
构建时工作区另有修改：{build.get('working_tree_modified', '未记录')}。精确依赖见“依赖版本”。
{build.get('working_tree_note', '')}
源码测试：{source_test_summary(verification / 'source-tests.xml')}。

最终程序验证状态：{report['status']}。
环境范围：{report['environment_scope']}。
测试对象：SoftwareTools/SoftwareTools.exe，{stat.st_size} 字节；组装前核对完整发布目录共 {len(current_files)} 个文件，
相对路径、大小、修改时间均与GUI验证启动前清单一致，验证结束时也确认目录未变。

{checks}

发布排除审计：{audit['files']} 个文件，{audit['module_count']} 个归档模块，passed={audit['passed']}。

尚未验证：
{pending}

截图和原始 JSON 位于“验收证据”。其中绝对路径对应验证机器，仅作追溯，不是目标机运行路径。
模拟样例仅检查软件行为，不能证明现场设备或模型有效性；干净目标机验收仍需按“验收步骤.txt”执行。
"""
    instructions = f"""Windows 版运行说明

目标系统：Windows 11，x86_64（Intel/AMD 64位）。当前实际验证环境：{report['platform']} / {report['machine']}。
这是便携目录版，无需安装 Python、Conda 或开发工具；需要图形桌面和可用中文字体。
本包未做代码签名；首次启动若遇系统信誉提示，按单位的软件安装策略处理。
完整解压后双击 SoftwareTools/SoftwareTools.exe，不要单独移动 EXE 或删除 _internal。
仅当系统明确提示缺少MSVC运行库时，从微软官方安装适用于x64的Visual C++ v14 Redistributable；
不要从第三方DLL站点下载文件覆盖软件依赖。[微软官方下载说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
普通用户默认数据位置：%LOCALAPPDATA%/SoftwareTools_PyQt。
可在启动前设置 ZMSOFTWARE_DATA_DIR 指向独立可写目录；不使用开发机绝对路径。

测试数据.zip 独立放置，按内附使用说明导入；不要把样例或验收模型混入生产记录。
当前窝深导入历史只保留本次会话，理论和报警设置跨启动保存。
本包在现有Windows主机隔离环境验证，未完成干净虚拟机/目标机验收，详见“测试报告.md”。
已实现：机器人观测/基准/精度评估，主轴信号分析与CPU模型训练推理，窝深CSV/XLSX统计和报警。
未接入：相机在线采集、机器人/主轴控制、加工点位预测接口；设备动作仍由设备集控负责。

可复现构建入口：仓库 tools/release/build_windows.ps1；交付组装：tools/release/assemble_delivery.py。
"""
    return write_delivery(output / "Windows版.zip", files, {
        "运行说明.txt": instructions, "测试报告.md": testing, "验收步骤.txt": ACCEPTANCE,
    })


def assemble_kylin(root, output):
    package = root / "dist/SoftwareTools-linux-x86_64.tar.gz"
    reports = root / "dist/linux-build-report"
    # 文件不存在时直接失败；不创建装有占位说明的“麒麟版”冒充构建产物。
    with tarfile.open(package, "r:gz") as archive:
        executable = archive.getmember("SoftwareTools/SoftwareTools")
        if not executable.isfile() or not executable.mode & 0o111:
            raise RuntimeError("Linux 发布 tar.gz 缺少可执行入口或执行权限。")
    startup = read_json(reports / "final-executable/startup-report.json")
    audit = read_json(reports / "bundle-audit.json")
    if startup.get("status") != "passed" or not audit.get("passed"):
        raise RuntimeError("Linux 最终程序启动或发布内容审计未通过，不生成 麒麟V10版.zip。")
    glibc = startup["elf"]["glibc_maximum_required"]
    files = {
        "SoftwareTools-linux-x86_64.tar.gz": package,
        "测试数据.zip": root / "dist/release_inputs/测试数据.zip",
        "依赖版本/requirements-kylin.lock": root / "tools/release/requirements-kylin.lock",
        "依赖版本/python-packages.txt": reports / "python-packages.txt",
        "依赖版本/build-environment.txt": reports / "build-environment.txt",
    }
    add_files(files, reports, "验收证据/Linux构建", ("*.json", "*.xml", "*.txt", "*.log"))
    add_files(files, reports / "final-executable", "验收证据/Linux最终程序", ("*.json", "*.txt", "*.log", "*.png"))
    testing = f"""# 麒麟 V10 目标包测试报告

这是为麒麟 V10 x86_64 准备的 Linux 构建产物，麒麟 V10 海光 3550 实机验收尚未完成。
实际构建/启动平台：{startup['platform']} / {startup['machine']}。
验证范围：{startup['scope']}。
源码测试：{source_test_summary(reports / 'source-tests.xml')}。

最终程序 X11 启动：{startup['status']}；主窗口观察结果：{startup['startup']['main_window_observed']}；
独立用户目录初始化：{startup['startup']['user_parameters_initialized']}。
ELF要求最高 GLIBC_{glibc}；缺失依赖：{startup['elf']['required_libraries_missing']}。
内容审计：{audit['files']} 个文件，{audit['module_count']} 个归档模块，passed={audit['passed']}。
tar.gz 内已核对 SoftwareTools/SoftwareTools 存在且保留执行权限。

未完成：麒麟 V10 具体子版本/桌面环境与海光3550实机启动、三页业务操作、数据导入导出、CPU训练推理、字体和图形兼容。
容器中测试不等于麒麟实机成功；应按“验收步骤.txt”验证并记录。软件样例不证明现场设备精度。
完整证据及构建系统版本位于“验收证据”和“依赖版本”；报告中的绝对路径仅属于构建环境。
"""
    instructions = f"""麒麟 V10 目标包运行说明（实机待验收）

目标：麒麟 V10 桌面系统，x86_64，海光3550 CPU。尚未取得目标机具体V10子版本与现场桌面信息。
该包已在报告记录的 Linux 容器构建并检查最终程序窗口启动；这不代表麒麟实机验证通过。
最低已测得二进制要求：glibc >= {glibc}。目标机还需图形桌面（X11或XWayland）、中文字体、
X11/XCB、xkbcommon、OpenGL/EGL、fontconfig、FreeType、D-Bus系统运行库。
常见Debian系包名如libxcb-cursor0、libxkbcommon-x11-0、libxcb-icccm4、libxcb-image0、libxcb-keysyms1、
libxcb-render-util0、libxcb-xinerama0、libgl1、libegl1；安装名称以实际麒麟软件源和仓库docs/release_delivery.md为准。
先在目标机执行 uname -m、cat /etc/os-release、getconf GNU_LIBC_VERSION，记录实际系统与架构。

解压外层ZIP后，在终端进入交付目录：
tar -xzf SoftwareTools-linux-x86_64.tar.gz
./SoftwareTools/SoftwareTools

tar.gz 保留Linux执行权限和符号链接，勿在Windows中解成文件再拷贝替代它。
软件包含Python、Qt与CPU PyTorch运行库，不依赖开发Python环境。
默认业务数据：$XDG_DATA_HOME/SoftwareTools_PyQt，未设置XDG_DATA_HOME时为 ~/.local/share/SoftwareTools_PyQt。
可在启动前设置 ZMSOFTWARE_DATA_DIR 指定独立可写目录。
先用 ldd SoftwareTools/SoftwareTools 和 ldd SoftwareTools/_internal/PyQt6/Qt6/plugins/platforms/libqxcb.so 检查系统缺库，
如显示 not found，按麒麟当前软件源安装对应系统库后再验收，不能据此直接宣称支持所有V10版本。

测试数据.zip 独立解压，按内附说明导入；窝深导入历史只保留本次会话，设置跨启动保存。
软件已实现机器人观测评估、主轴分析与CPU训练推理、窝深导入统计；这些业务尚未在麒麟实机验收。
相机在线采集、机器人/主轴控制和加工点位预测接口尚未接入，设备动作仍由设备集控负责。
尚未验证项目详见“测试报告.md”；复现构建入口为仓库 tools/release/build_kylin.sh，目标机步骤见“验收步骤.txt”。
"""
    return write_delivery(output / "麒麟V10版.zip", files, {
        "运行说明.txt": instructions, "测试报告.md": testing, "验收步骤.txt": ACCEPTANCE,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "kylin", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/delivery")
    parser.add_argument("--windows-report", type=Path, default=ROOT / "dist/verification/windows_gui_final")
    args = parser.parse_args()
    if args.platform in {"windows", "all"}:
        print(assemble_windows(ROOT, args.output, args.windows_report.resolve()))
    if args.platform in {"kylin", "all"}:
        print(assemble_kylin(ROOT, args.output))


if __name__ == "__main__":
    main()
