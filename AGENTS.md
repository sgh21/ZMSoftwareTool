# AGENTS.md

面向 Codex 或其他代码代理的项目入口说明。任何新任务开始前，先阅读本文件，再扫描、阅读、修改或运行命令。

## 项目简介

本仓库是本地运行的 UR 机器人数字孪生工具软件，使用 PySide6 桌面界面承载三维仿真、误差参数加载、S1 参数辨识、精度状态评估、健康状态评估、校验与报告输出。项目定位是本地单进程桌面工具链，不是 Web 前后端系统。

## 当前目录结构

```text
SoftwareTools/
├── main.py                         # 桌面软件入口
├── pyproject.toml                  # 工程元数据、可编辑安装、pytest/ruff 配置
├── requirements.txt                # 运行时依赖
├── requirements-dev.txt            # 测试和开发依赖
├── environment.yml                 # Conda 环境复现
├── app/                            # PySide6 UI 层
│   ├── main_window.py
│   ├── app_context.py
│   ├── pages/
│   └── widgets/
├── core/                           # 业务逻辑、数据处理和算法适配层
│   ├── calibration_service.py
│   ├── calibration_persistence.py
│   └── calibration/bayesian_calibration_pipeline/
├── config/                         # 项目、机器人、阈值、主题、名义模型和辨识结果配置
├── models/                         # URDF 与 Mesh 模型
├── data/                           # 标定数据、校验数据、处理结果和报告输出
├── storage/                        # 模型版本和 SQLite 历史记录
├── docs/                           # 维护文档、设计材料和历史报告
├── experiments/                    # 调试、实验脚本、配置和结果归档
└── tests/                          # pytest 测试
```

## 运行入口与常用命令

推荐环境来自现有文档和配置：

```powershell
conda activate ZMSoftware
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

常用命令：

```powershell
python main.py
python -m pytest
python -m pytest tests/test_initialization_page.py
python -m pytest tests/test_calibration.py
python -m ruff check .
```

可安装脚本入口在 `pyproject.toml` 中声明为 `zmsoftware = "main:main"`，是否已在当前环境安装需按实际环境确认。

算法包内独立实验入口见 `core/calibration/bayesian_calibration_pipeline/README.md`，其中 `bayesian_real_experiment` 属于较重实验；日常维护优先使用小规模 pytest 或 quick-smoke。

## 重要配置文件

```text
config/project.yaml             项目默认目录配置
config/robot.yaml               UR10 模型路径、mesh 路径和坐标系
config/nominal_robot.yaml       名义 MD-H、基座、工具和关节限制
config/thresholds.yaml          精度阈值、最大误差、超差率和置信度阈值
config/theme.yaml               UI 状态颜色和机器人 mesh 渲染配色
config/verification.yaml        校验输出目录、拍摄数量和位姿随机化配置
config/calibration_result.yaml  当前默认加载的辨识结果
```

所有路径、阈值、机器人模型、位姿和拍摄数量优先从 `config/` 读取。不要把硬编码配置散落到 UI 或算法模块。

## 核心模块

- `main.py`：创建 `QApplication`，实例化 `app.main_window.MainWindow`。
- `app/main_window.py`：主窗口，仅承载初始化页面，并暴露参数文件加载入口。
- `app/pages/initialization_page.py`：当前主要 UI 页面，包含模型/参数加载、仿真视图、实时精度状态、健康状态、S1 参数辨识入口和调试窗口入口。
- `app/pages/calibration_page.py`：独立参数辨识与精度报告页面，当前与初始化页存在部分功能重叠。
- `app/widgets/robot_simulation_widget.py`：基于 yourdfpy 和 PyVista off-screen 渲染 URDF，读取 `config/theme.yaml` 配色。
- `core/calibration_service.py`：参数辨识服务层入口。UI 必须通过该服务层调用算法，不直接调用 `core/calibration/bayesian_calibration_pipeline/` 内部模块。
- `core/calibration_persistence.py`：辨识结果 YAML 保存/加载与 SQLite 历史记录。
- `core/accuracy_evaluator.py`、`core/health_evaluator.py`：精度和健康状态评估。
- `core/calibration/bayesian_calibration_pipeline/`：已归档的机器人误差建模与贝叶斯标定核心算法包。

## 参数辨识约定

1. 当前默认辨识方法为 S1：交叉验证选择正则化参数、可辨识性参数加权、按位姿可辨识性划分子空间，再执行子空间顺序 LM 拟合。
2. 辨识数据支持单个或多个 `.pkl/.pickle` 文件，由 `CalibrationService.load_identification_data()` 合并。
3. 定位误差固定定义为 `p_identified(q) - p_nominal(q)`。
4. 拟合残差固定定义为 `p_identified(q) - p_measured(q)`。
5. 名义位置优先由 `config/nominal_robot.yaml` 的名义 MD-H/基座/工具参数计算；缺失时回退到算法包默认名义模型。
6. 辨识结果默认保存到 `config/calibration_result.yaml`，历史写入 `storage/records/identification_history.sqlite` 的 `identification_runs` 表。
7. 参数辨识页面运行耗时算法时使用进度弹窗；完成后自动保存 YAML 并写入 SQLite 历史。
8. UR10 关节调试窗口属于临时调试入口，通过顶部“调试”入口打开，不放在右侧常用设置区。

## 修改前优先阅读

按任务类型先读这些文件：

- 桌面入口或主界面：`main.py`、`app/main_window.py`、`app/pages/initialization_page.py`
- 机器人渲染：`app/widgets/robot_simulation_widget.py`、`config/theme.yaml`、`models/urdf/ur10.urdf`
- 参数辨识：`core/calibration_service.py`、`core/calibration_persistence.py`、`tests/test_calibration.py`
- 配置路径或阈值：`config/*.yaml`、`core/config_manager.py`
- 算法实验：`core/calibration/bayesian_calibration_pipeline/README.md`
- 项目维护背景：`docs/PROJECT_OVERVIEW.md`、`docs/TASK_LOG.md`

## 代码修改原则

1. 修改前先理解相关目录、调用链、测试和现有风格。
2. 只做与当前任务直接相关的最小修改，不重写无关模块。
3. UI 与核心算法分离；UI 不直接耦合算法包内部实现。
4. 不改变核心算法逻辑，除非任务明确要求且有小规模验证。
5. 模型更新前必须备份旧参数；标定和校验结果必须可追溯。
6. 保持本地单进程桌面工具链形态，不引入 Web 前后端架构。
7. 生成文件优先写入 `data/reports/`、`data/processed/`、`storage/records/` 或 `experiments/`，不要散落在根目录。
8. 遇到不确定模块作用时，在文档或汇报中标注“待确认”，不要猜测。

## 实验与调试规范

1. 仓库级实验、调试脚本和临时分析代码统一放入 `experiments/`。
2. 实验命名建议使用 `YYYYMMDD_short_description`。
3. 可复现实验脚本、轻量配置和小型结果摘要可以提交；大文件、日志、checkpoint、模型权重和批量输出默认不提交。
4. 算法包历史实验仍保留在 `core/calibration/bayesian_calibration_pipeline/experiments/`，新增仓库级实验优先使用根目录 `experiments/`。
5. 每次进行试验验证后，需要询问用户是否将试验结果更新进 `AGENTS.md`；用户确认后再维护更新。
6. 对于临时测试代码，测试结束后询问用户是否保留；不保留时及时清理。

## 测试与验证方式

优先使用轻量验证：

```powershell
python -m pytest tests/test_initialization_page.py
python -m pytest tests/test_calibration.py
python -m pytest
python -m ruff check .
```

涉及完整训练、长时间实验或高成本计算时，只运行小规模 sanity check，并在汇报中说明未运行完整实验的原因。

## 每次任务完成后的汇报格式

任务结束时用中文汇报：

1. 修改了什么
2. 为什么这样修改
3. 修改了哪些文件
4. 执行了哪些命令
5. 测试或验证结果
6. 当前仍不确定的信息
7. 下一步建议
