# PROJECT_OVERVIEW.md

## 项目目标

本项目是一款本地运行的 UR 机器人数字孪生工具软件，用于显示机器人三维仿真、加载误差参数、导入并分析标定数据、评估当前精度和健康状态，并为后续精度校验和模型更新提供本地工具链。

当前项目形态是 Python 3.12 + PySide6 的单进程桌面应用。核心标定算法包已归档到 `core/calibration/bayesian_calibration_pipeline/`，桌面 UI 通过服务层调用，不直接耦合算法内部模块。

## 主要功能模块

- 机器人实时仿真：`app/widgets/robot_simulation_widget.py` 读取 `models/urdf/ur10.urdf` 和 mesh，使用 PyVista off-screen 渲染。
- 初始化与主界面：`app/pages/initialization_page.py` 负责默认模型/参数加载、状态卡片、健康状态、右侧设置、参数辨识入口和调试入口。
- 参数辨识：`core/calibration_service.py` 提供 S1 参数辨识服务，支持多个 `.pkl/.pickle` 数据集合并。
- 辨识结果持久化：`core/calibration_persistence.py` 保存 YAML，并写入 SQLite 历史。
- 精度与健康评估：`core/accuracy_evaluator.py` 和 `core/health_evaluator.py` 提供指标计算。
- 报告输出：当前支持 HTML/YAML/JSON 等轻量输出，算法包也包含 HTML 报告生成模块。
- 精度校验：`core/verification_manager.py`、`core/image_comparator.py`、`config/verification.yaml` 已有基础结构，完整闭环仍待确认。

## 总体架构

```text
UI 层
  main.py -> app.main_window.MainWindow -> app.pages.initialization_page.InitializationPage

可视化层
  app.widgets.robot_simulation_widget.RobotSimulationWidget

服务层
  core.calibration_service.CalibrationService
  core.calibration_persistence

算法层
  core.calibration.bayesian_calibration_pipeline

数据与配置层
  config/*.yaml
  data/calibration/
  data/reports/
  storage/records/
```

架构约束：本项目优先保持本地单进程桌面工具链，不做 Web 前后端拆分。

## 数据流

启动流程：

```text
python main.py
  -> 创建 QApplication
  -> 打开 MainWindow
  -> 初始化 InitializationPage
  -> 尝试加载 models/urdf/ur10.urdf
  -> 尝试加载 config/calibration_result.yaml 或 storage/model_versions/active_calib_params.yaml
  -> 刷新模型状态、精度指标和健康状态
```

参数辨识流程：

```text
选择一个或多个 .pkl/.pickle 数据文件
  -> CalibrationService.load_identification_data()
  -> 合并 joints / measured_positions / payloads / directions
  -> CalibrationService.run_identification()
  -> S1 子空间可辨识性加权 + 交叉验证正则化 + LM 拟合
  -> 计算定位误差 p_identified - p_nominal
  -> 计算拟合残差 p_identified - p_measured
  -> 保存 config/calibration_result.yaml
  -> 写入 storage/records/identification_history.sqlite
```

实时精度状态：

```text
当前关节角
  -> 名义模型计算 p_nominal(q)
  -> 当前激活辨识参数计算 p_identified(q)
  -> 定位误差、RMS、最大误差、超差结论、健康状态
```

## 关键文件说明

- `main.py`：桌面应用启动入口。
- `pyproject.toml`：项目名称、依赖范围、`zmsoftware` 脚本入口、pytest 和 ruff 基础配置。
- `README.md`：简要说明项目定位、安装和启动命令。
- `AGENTS.md`：面向代码代理的任务入口和协作规则。
- `app/main_window.py`：主窗口和页面挂载。
- `app/pages/initialization_page.py`：当前主工作台页面。
- `app/pages/calibration_page.py`：独立参数辨识页面。
- `app/widgets/robot_simulation_widget.py`：URDF/PyVista 渲染。
- `core/calibration_service.py`：参数辨识和实时精度状态服务入口。
- `core/calibration_persistence.py`：YAML/SQLite 持久化。
- `core/calibration/bayesian_calibration_pipeline/README.md`：算法包结构与重实验命令说明。
- `tests/test_initialization_page.py`：初始化页、默认模型加载、调试窗口和 UI 行为测试。
- `tests/test_calibration.py`：S1 参数辨识、YAML/SQLite 持久化和 UI 集成测试。

## 配置文件说明

- `config/project.yaml`：默认模型、数据和存储目录。
- `config/robot.yaml`：UR10 模型、mesh 路径、base/tool frame。
- `config/nominal_robot.yaml`：名义基座、工具、MD-H 参数和关节限制。
- `config/thresholds.yaml`：RMS、最大误差、超差率和最低置信度阈值。
- `config/theme.yaml`：状态颜色和机器人 mesh 分类配色。
- `config/verification.yaml`：校验拍照数量、随机位姿和输出目录。
- `config/calib_params.yaml`：旧/基础标定参数配置，具体格式需按调用点确认。
- `config/calibration_result.yaml`：当前默认辨识结果。

## 如何运行

安装依赖：

```powershell
conda activate ZMSoftware
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

启动桌面软件：

```powershell
python main.py
```

可安装入口：

```powershell
zmsoftware
```

`zmsoftware` 是否可用取决于当前环境是否已执行可编辑安装。

## 如何调试

- UI 调试优先运行 `python main.py`，观察初始化页、模型加载和参数文件状态。
- Qt 测试使用 offscreen 环境，现有测试文件已设置 `QT_QPA_PLATFORM=offscreen`。
- UR10 关节角调试入口在主界面顶部“调试”按钮。
- 参数辨识调试优先使用 `tests/test_calibration.py` 中的小规模合成数据。
- 算法包历史实验命令见 `core/calibration/bayesian_calibration_pipeline/README.md`；较重实验不要作为常规验证。
- 仓库级临时实验统一放在 `experiments/`，不要放到根目录或核心源码目录。

## 如何添加新功能

1. 先确认功能归属：UI 放 `app/`，业务逻辑放 `core/`，配置放 `config/`，实验放 `experiments/`。
2. UI 新功能应通过服务层调用业务逻辑，不直接调用算法包内部模块。
3. 新增路径、阈值、位姿、拍摄数量等参数优先放入 YAML 配置。
4. 涉及模型更新时，先备份旧参数，再保存新结果和历史记录。
5. 新增可追溯流程时，同步补充 `docs/TASK_LOG.md` 和必要测试。
6. 对算法包的改动要保持导入路径为 `core.calibration.bayesian_calibration_pipeline.*`。

## 当前不确定点和技术债

- `app/pages/initialization_page.py` 与 `app/pages/calibration_page.py` 存在参数辨识 UI 逻辑重叠，后续需要明确保留关系。
- `config/calib_params.yaml` 与 `config/calibration_result.yaml` 的职责边界仍需进一步统一。
- `core/verification_manager.py` 和图像校验流程已有基础模块，但完整校验闭环是否可运行待确认。
- `zmsoftware` 命令入口已在 `pyproject.toml` 声明，但当前环境是否安装待确认。
- PyVista/VTK/yourdfpy 渲染链对本机图形和依赖环境敏感，CI 或无显示环境需要继续验证。
- `data/calibration/bayesian_calibration_pipeline/` 中包含真实/样例 `.pkl` 数据，数据体积和版本管理策略后续可再细化。
