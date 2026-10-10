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

## 清理候选

仅确认以下内容可再生成；审计本身未删除：

- 根目录及 `app/`、`core/`、`debug/` 中的 `__pycache__`、`.pyc`；不属于业务输入或持久化数据。
- `experiments/20261005_launch/165337.stdout.log`、`165337.stderr.log` 均为 0 字节旧启动日志。

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
