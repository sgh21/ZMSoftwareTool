# 发布整理审计

2026-10-10：在已推送版本 `3beaf04` 上检查入口、生产调用、命令行入口、文档与正式测试引用。此记录不代表目标平台验收通过。

## 目录边界

| 内容 | 处理 |
| --- | --- |
| `main.py`、`app/`、`core/` | 软件入口、界面与业务实现，构建按运行依赖收集。 |
| `config/`、`resources/` | 保留运行配置、图标、样式和许可证；`config/examples/` 是开发样例，不放入软件本体。 |
| `tests/` | 正式回归留在源码仓库，全部排除发布软件。新交付样例测试也位于这里。 |
| `experiments/` | 保留有文档引用的定位全流程复现与主轴数据转换脚本，不打包。 |
| `debug/` | 独立重置入口与 UR10 仿真工具保留源码，不打包；原标定实现被生产弹窗调用，必须先迁至业务层，不能直接删掉。 |
| `data/`、`storage/`、仿真 `assets/` | 原始采集、图片、参数、模型、标签、历史及本地报告原地保留，全部不提交、不打包。 |
| `tools/release/` | 构建与交付验证工具留在源码仓库，不进入软件本体。 |
| `dist/`、`build/` | 可再生成的构建交付产物，由 Git 忽略。 |

`resources/icons/branding/reference-interface.png` 当前仍用于顶部品牌图，不能当作无用截图删除。页面中的仿真核验、窝深模拟与相机标定入口仍有 Qt 回调，不能因名称含 debug/simulation 就移除这些现有功能。

## 本地缓存清理状态

已核对绝对路径均在当前仓库内，缓存目录只包含 `.pyc`，以下内容可再生成：

- `__pycache__/`、`app/__pycache__/`、`app/dialogs/__pycache__/`、`app/pages/__pycache__/`、`core/algorithms/__pycache__/`、`core/services/__pycache__/`、`debug/__pycache__/`。
- `experiments/20261005_launch/165337.stdout.log`、`165337.stderr.log` 均为 0 字节旧启动日志。

限定范围的 PowerShell 清理请求被自动审批以 `blocked by policy` 拒绝，工具未给出更具体理由；未执行删除，也未换工具绕过限制。因此这些缓存和空日志仍留在本地，属于整理未完成项；构建排除它们，最终发布审计检查不进入软件包。`data/`、`storage/` 和任何历史报告未进行删除操作。

未发现可仅凭“没有静态 import”就删除的已跟踪业务或复现代码。`experiments/feed_depth_ui/outputs/` 及其他报告保留，不能把验收截图与历史报告当作临时文件批量清空。

## 独立交付测试数据

用开发环境运行：

```powershell
python -B -X utf8 tools/release/prepare_test_data.py
python -B -m pytest -q -p no:cacheprovider tests/test_release_test_data.py
```

产物为 `dist/release_inputs/测试数据.zip`，包含三页所需 JSON、ChArUco PNG、主轴 H5/CSV 内层 ZIP、窝深 CSV/XLSX，以及 `使用说明.txt` 和供验收读取的 `manifest.json`。不读取原始业务资产，不包含测试代码或权重；所有数据均明确标为人工模拟。主轴固定模拟时间仅满足协议，不充当真实采集日期。

主轴另有 `02_主轴/小型建模.zip`，内含五份独立随机种子的人工信号，供最终程序在隔离存储中验证一轮训练与推理。该包不含预训练权重，不证明模型或设备有效性。

2026-10-10 在隔离临时目录中运行新增 8 项正式回归，全部通过：解析观测与历史恢复、实际 ChArUco 检测、主轴实际 H5/CSV 导入/频谱计算/重复导入、五份可分组训练候选，以及 CSV/XLSX 的缺测、边界与超限统计。Ruff 通过。这是源码服务与样例格式验证，最终发布包和目标系统验证另见发布测试报告。

## 最终程序内部审计

`tools/release/audit_bundle.py` 同时检查发布目录和最终 EXE/ELF 中的 CArchive、内嵌 PYZ 模块目录。排除项目测试、调试、实验、构建工具、pytest、界面自动化依赖及测试数据；空目录或缺失可执行文件不能判为成功。

第三方运行依赖按实际用途区分：`torch/distributed/tensor/debug` 是 PyTorch 运行库；`jinja2.tests` 是模板表达式判断实现；`numpy._pytesttester` 与 `scipy._lib._testutils` 是导入兼容帮助模块。不能因名字含 debug/test 就把它们当作本项目测试集删除。审计的 4 项正式回归使用真实 PyInstaller 归档，覆盖隐藏在 PYZ 的开发代码、合法运行模块及内外层样例文件。

首个 Windows 包实际读取到 4,971 个 CArchive/PYZ 模块，没有禁止的项目或开发模块；发现 `config/examples/` 两个开发 JSON，按失败报告并要求构建端排除后重建。这条记录仅描述首包问题，最终审计结论以 `dist/verification/bundle-audit.json` 和发布测试报告为准。

最终 EXE 的主轴导入实际暴露出按名字裁剪第三方 `testing` 帮助模块的问题：SciPy `signal/stats` 的运行导入链需要 NumPy 的通用数组比较工具；PyTorch 2.9 的 `autograd/gradcheck`、`utils/checkpoint` 和优化器导入需要比较、张量创建及日志张量工具。它们不是独立测试用例集，不能只因包名含 testing 就删掉。

构建与审计共用 `tools/release/audit_bundle.py` 中的精确允许名单：`numpy.testing`、`numpy.testing._private`、`numpy.testing.overrides`、`numpy.testing._private.extbuild`、`numpy.testing._private.utils`、`scipy._lib.array_api_extra.testing`，以及 `torch.testing`、`torch.testing._utils`、`torch.testing._comparison`、`torch.testing._creation`、`torch.testing._internal`、`torch.testing._internal.logging_tensor`；另保留既有 `jinja2.tests`。Torch 必需的同名 `.py` 也按这份名单保留，满足其运行时源码检查。

该名单按完整模块名匹配，不允许任何实际 `tests/` 子目录、未列出的 `torch.testing._internal` 模块或 pytest；正式回归验证这些边界。最终软件行为仍由重建后的 EXE 验证，源码导入试验不替代最终产物验收。

最终 EXE 的 CPU 训练继续发现 `torch._inductor.test_operators` 被误删：PyTorch 2.9 的优化器经 `torch._dynamo` 导入 `trace_rules`，后者无条件导入此模块。文件只注册 `realize` 算子及其自动微分实现，没有测试用例。因此将该完整模块名及同名 `.py` 加入运行允许名单，并移除 spec 的单项排除；相邻 `test_case`、其他 `test_*` 和实际测试包仍排除。两平台必须重建，旧包启动通过不能覆盖此训练问题。

随后使用 Python 3.12.13 与构建用 CPU Torch 2.9.0，在首次导入 SciPy/Torch 前替换磁盘模块查找器，按完整审计规则和 spec 排除名单模拟模块缺失。SciPy 频谱计算及 AdamW 初始化、前向、反向与参数更新通过，`pandas`、`sklearn` 和 `torch.testing._internal.distributed` 保持不可用；未再发现必需模块。此试验通过标准输入运行，无临时脚本，记录在 `dist/verification/dependency-pruning-check.json`；仍需最终程序实测训练。

Linux 实际 ELF 检查还发现 PyTorch wheel 自带 `torch/bin/HashStoreTest`、`FileStoreTest`、`TCPStoreTest`、`protoc*` 和 `torch/lib/libtorchbind_test.so`、`libjitbackend_test.so`。构建 hook 在分析二进制依赖前按这些确切名称剔除，spec 与最终审计共用同一规则。`torch_shm_manager`、正常 CPU/Python 运行库及 NumPy 的 `_multiarray_tests` 扩展不按名称子串误删；后者可能被运行帮助模块导入。
