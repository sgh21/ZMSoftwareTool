# AGENTS.md

## 2026-05-28 参数辨识实现逻辑更新

1. 参数辨识服务层入口为 `core/calibration_service.py`，UI 不直接调用 `core/calibration/bayesian_calibration_pipeline/` 内部算法模块。
2. 当前默认辨识方法为 S1：交叉验证选择正则化参数、可辨识性参数加权、按位姿可辨识性进行子空间划分，再执行子空间顺序 LM 拟合。
3. 辨识数据支持单个或多个 `.pkl/.pickle` 文件，服务层通过 `CalibrationService.load_identification_data()` 合并为统一数据结构。
4. 定位误差定义固定为 `p_identified(q) - p_nominal(q)`，即辨识所得预测模型位置与名义模型位置的差；拟合残差另行定义为 `p_identified(q) - p_measured(q)`，两者不得混用。
5. 名义位置由 `config/nominal_robot.yaml` 中的名义 MD-H/基座/工具参数前向运动学计算；如果项目目录没有该配置，服务层回退到算法包内置默认名义模型。
6. 辨识结果默认保存到 `config/calibration_result.yaml`，保存内容包括方法名、时间戳、样本数、S1 选择的 lambda、定位误差指标、拟合残差指标、参数字典、交叉验证分数和子空间摘要。
7. 每次辨识历史写入 SQLite：`storage/records/identification_history.sqlite`，表名为 `identification_runs`。
8. 主界面实时精度状态从当前关节角计算：先用名义参数计算 `p_nominal(q)`，再用当前激活辨识参数计算 `p_identified(q)`，最后基于定位误差刷新 RMS、最大误差、超差结论和健康状态。
9. 参数辨识页面运行耗时算法时使用进度弹窗提示当前处于 S1 参数辨识流程；辨识完成后自动保存 YAML 并写入 SQLite 历史。
10. UR10 关节调试窗口属于临时调试入口，不放在右侧常用设置区，通过顶部“调试”入口打开。

## 协作约定

1. 在任何代码仓库中开启新会话时，先检查并阅读仓库根目录的 `AGENTS.md` / `AGENT.md`，再做文件扫描、代码阅读、修改或运行命令。
2. 每次进行试验验证后，需要询问用户是否将试验结果更新进 `AGENTS.md`；如果用户确认，则自动维护更新。
3. 对于临时测试代码，测试结束后询问用户是否保留；如果不保留，需要清理，保证项目简洁、可读、易维护。
4. 本项目优先保持本地单进程桌面工具链形态，不做 Web 前后端架构。

---

## 项目目标

本项目是一款本地运行的机器人数字孪生工具软件，用于显示 UR 机器人三维仿真、加载误差参数、分析标定数据、评估当前精度和健康状态，并支持精度校验与模型更新。

软件定位是**本地工具链验证软件**，不是 Web 系统，也不是完整工业平台。要求开发快、架构简单、便于后续集成，支持 Windows 开发和 Linux 运行。

推荐技术栈：

```text
PySide6 + PyQtGraph / OpenGL / VTK / PyVista
Pandas + NumPy
YAML / JSON / SQLite
```

---

## 当前 Python 环境

已按本地桌面工具链创建 `ZMSoftware` Conda 环境，当前环境 Python 版本为 `3.12.0`。

环境配置文件：

```text
requirements.txt       运行时依赖
requirements-dev.txt   开发和测试依赖
environment.yml        Conda 环境复现文件
pyproject.toml         工程元数据、可编辑安装和工具配置
```

推荐安装命令：

```powershell
conda activate ZMSoftware
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

主要依赖分工：

```text
PySide6                桌面主窗口、菜单栏、弹窗、控件
PyQtGraph              曲线、状态报表、Qt 内嵌可视化
PyOpenGL               OpenGL 基础支持
VTK / PyVista          三维机器人、工作站和网格渲染
PyVistaQt              PyVista 与 Qt 主窗口集成
yourdfpy / xacro       URDF / Xacro 模型读取与预处理
trimesh / lxml         Mesh 与 XML/URDF 解析辅助
NumPy / SciPy          数值计算和优化
Pandas                 标定数据、历史记录和报表数据处理
scikit-learn           现有贝叶斯标定算法依赖
Matplotlib             算法报告图表
PyYAML / jsonschema    配置文件读取和校验
OpenCV                 后续精度校验图像对比
Jinja2                 HTML 报告模板
```

---

## 当前实际目录结构与职责

```text
SoftwareTools/
├── AGENTS.md
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
│
├── app/
│   ├── main_window.py
│   ├── app_context.py
│   ├── menus/
│   ├── pages/
│   ├── dialogs/
│   ├── widgets/
│   └── resources/
│
├── core/
│   ├── calibration/
│   │   └── bayesian_calibration_pipeline/
│   ├── config_manager.py
│   ├── project_manager.py
│   ├── robot_model_loader.py
│   ├── error_param_manager.py
│   ├── calibration_data_loader.py
│   ├── calibration_analyzer.py
│   ├── accuracy_evaluator.py
│   ├── health_evaluator.py
│   ├── verification_manager.py
│   ├── image_comparator.py
│   └── report_generator.py
│
├── config/
│   ├── project.yaml
│   ├── robot.yaml
│   ├── calib_params.yaml
│   ├── thresholds.yaml
│   └── verification.yaml
│
├── models/
│   ├── urdf/
│   ├── meshes/
│   └── workcell/
│
├── data/
│   ├── raw/
│   ├── calibration/
│   │   └── bayesian_calibration_pipeline/
│   ├── verification/
│   ├── processed/
│   └── reports/
│
├── storage/
│   ├── model_versions/
│   └── records/
│
├── docs/
│   └── design/
└── tests/
```

目录职责：

```text
app/
  PySide6 桌面 UI 层。负责主窗口、菜单、页面、弹窗、控件和静态资源。

core/
  业务逻辑层与数据处理层。负责配置管理、项目路径、模型加载、误差参数、标定数据、精度评估、健康评估、精度校验、图像对比和报告生成。

core/calibration/bayesian_calibration_pipeline/
  已完成的机器人误差建模与贝叶斯标定核心算法包。当前导入路径为：
  core.calibration.bayesian_calibration_pipeline

config/
  本地工具链配置。所有路径、阈值、机器人模型、校验位姿和拍摄数量优先从这里读取。

models/
  机器人和工位三维模型。`urdf/` 放 URDF/Xacro，`meshes/` 放 STL/DAE/OBJ 等网格，`workcell/` 放工作台、夹具、相机、校验工位模型。

data/raw/
  原始采集数据，原则上不在算法中直接覆盖。

data/calibration/
  标定输入数据。现有算法样本数据已归档到：
  data/calibration/bayesian_calibration_pipeline/

data/verification/
  精度校验图像、位姿、过程数据和校验输入。

data/processed/
  中间处理结果、清洗后的数据、算法过程缓存。

data/reports/
  HTML / CSV / YAML / JSON 报告输出。贝叶斯标定算法默认报告输出到：
  data/reports/bayesian_calibration_pipeline/

storage/model_versions/
  误差参数模型版本和模型更新前备份。模型更新前必须先备份旧参数。

storage/records/
  历史记录、标定记录、校验记录和可追溯元数据。

docs/
  工程文档、设计说明和后续接口文档。当前界面设计 PPT 和界面截图已归档到 `docs/design/`。

tests/
  单元测试、集成测试和算法烟雾测试。
```

---

## 核心算法归档规则

1. `bayesian_calibration_pipeline` 不再作为仓库顶层目录存在，已归档到 `core/calibration/bayesian_calibration_pipeline/`。
2. 算法源码保持为独立包，只做导入路径迁移，不改动核心算法逻辑。
3. 原算法包内 `dataset/` 数据已迁移到 `data/calibration/bayesian_calibration_pipeline/`。
4. 算法默认报告路径从 `outputs/...` 调整为 `data/reports/...`。
5. 软件 UI 和业务层通过 `core/calibration_analyzer.py` 或新的服务层适配器调用算法，不在 UI 层直接耦合算法内部模块。

---

## 核心工作流程

```text
启动软件
  ↓
读取默认项目目录
  ↓
加载机器人三维模型和误差参数文件
  ↓
若文件缺失，提示用户手动加载
  ↓
显示机器人实时仿真、精度指标和健康状态
  ↓
用户导入标定数据
  ↓
弹窗执行标定数据分析
  ↓
计算 RMS 误差、最大误差、超差率、模型置信度
  ↓
判断是否建议更新精度模型
  ↓
用户确认后更新并持久化误差参数
  ↓
刷新主界面精度状态和健康状态
  ↓
执行精度校验
  ↓
机器人移动到校验工位并拍摄图像
  ↓
对比标定前后图像和精度结果
  ↓
输出模型是否需要更新、当前精度不确定度和校验报告
```

---

## 主要功能

### 1. 机器人实时仿真

* 加载 URDF / Xacro / Mesh 模型
* 显示当前 UR 机器人构型
* 显示坐标系、工作台、夹具、相机、校验工位
* 支持旋转、缩放、平移视角
* 在仿真窗口右下角显示当前精度和健康状态

### 2. 精度状态显示

显示内容包括：

```text
位置误差 RMS
最大误差
超差阈值
是否超差
模型置信度
当前精度不确定度
更新时间
```

超差阈值来自配置文件，可通过菜单或右侧设置面板修改。

### 3. 标定数据分析

用户通过文件菜单导入标定数据后，软件弹窗分析：

```text
采样点数
有效点数
RMS 误差
最大误差
超差率
模型置信度
是否建议更新模型
```

如果用户确认更新，需要备份旧参数，并将新误差参数持久化到本地。

### 4. 健康状态评估

根据精度误差、超差率、模型置信度和历史趋势给出健康状态：

```text
健康分
健康等级
当前状态
异常提示
维护建议
```

### 5. 精度校验

精度校验流程：

```text
读取校验位姿配置
移动到校验工位
按配置随机化位姿
拍摄指定数量图片
对比标定前后图像
计算当前精度不确定度
判断是否需要更新模型
生成校验报告
```

### 6. 报告与历史记录

保存内容包括：

```text
标定记录
模型版本
精度状态
校验结果
历史趋势
报告文件
```

优先支持：

```text
HTML
CSV
YAML
JSON
```

PDF 后续实现。

---

## 软件架构

采用本地单进程桌面架构。

```text
UI 层
  PySide6 主窗口、菜单栏、弹窗、页面布局

可视化层
  机器人三维仿真、曲线、状态卡片、报表

业务逻辑层
  标定分析、精度评估、健康评估、精度校验

数据层
  YAML / JSON / SQLite / CSV / 历史记录

外部接口层
  模型文件、标定数据、相机数据、机器人控制接口
```

第一阶段不使用 Web 前后端，不引入复杂服务架构。

---

## 主界面设计

主页面采用 16:9 横向布局。

```text
顶部菜单栏：
文件 / 编辑 / 查看 / 视图 / 校验

顶部状态栏：
当前项目 / 连接状态 / 配置状态 / 告警状态

左侧主区域：
机器人实时仿真三维视图

仿真区域右下角：
当前精度指标卡片 + 健康状态卡片

右侧上方：
常用设置

右侧下方：
精度状态报表 / 历史精度 / 最近报告 / 目标点位功能

底部状态栏：
机器人型号 / 控制器 / 仿真频率 / 时间 / 坐标系 / 日志
```

---

## 菜单设计

```text
文件：
新建项目、打开项目、加载三维模型、加载误差参数、导入标定数据、导出报告

编辑：
编辑机器人配置、编辑误差参数、编辑阈值、编辑校验位姿

查看：
显示日志、显示历史记录、显示标定详情、打开报告目录

视图：
重置视角、显示坐标系、显示目标点位、显示误差向量、显示校验工位

校验：
开始精度校验、暂停校验、重新拍摄、查看校验报告、校验配置
```

---

## 界面状态

软件至少需要支持以下状态：

```text
未初始化
配置缺失
模型已加载
参数已加载
运行正常
标定分析中
等待确认更新
模型更新完成
精度校验中
校验完成
超差告警
错误状态
```

不同状态下，主界面显示不同内容：

```text
未初始化：
显示空三维视图和加载提示

运行正常：
显示机器人模型、精度指标和健康状态

标定分析中：
显示分析弹窗和进度

校验中：
显示校验工位、拍摄进度和校验结果
```

---

## 开发优先级

### 第一阶段

完成主页面和基础配置闭环：

```text
PySide6 主窗口
菜单栏
三维视图占位
模型和参数加载
未初始化提示
精度指标卡片
健康状态卡片
右侧设置和报表面板
```

### 第二阶段

完成标定数据分析闭环：

```text
导入标定数据
计算基础精度指标
分析是否建议更新模型
更新并持久化参数
刷新主界面
```

### 第三阶段

完成精度校验闭环：

```text
读取校验配置
显示校验工位
管理拍摄进度
图像对比
输出精度不确定度
生成校验报告
```

### 第四阶段

完善集成能力：

```text
历史记录
报告导出
CLI 接口
Linux 运行适配
打包发布
```

---

## 开发原则

1. 优先完成本地功能闭环。
2. 不做 Web 前后端架构。
3. UI 与核心算法分离。
4. 所有路径、阈值、位姿、拍摄数量都从配置文件读取。
5. 模型更新前必须备份旧参数。
6. 标定和校验结果必须可追溯。
7. 第一版允许三维视图先用占位或简化模型。
8. 优先实现 HTML / CSV / YAML / JSON 报告，PDF 后置。
9. 代码结构要便于后续接入机器人控制器、相机、ROS2 或其他接口。
