# SoftwareTools · Python / PyQt

本地桌面软件的重新设计工程。Python 负责核心算法和业务逻辑，PyQt 负责界面展示与交互。

当前为 PyQt6 桌面软件：左侧仅保留“精度监控”，首页包含“机器人末端定位精度”“主轴回转精度”“主轴轴向进给精度”三个选项卡。机器人页已接入参数设置、观测导入、基准管理、精度计算和历史保存；主轴回转页已接入批量建模、每日检测、人工标签、阈值和重训，轴向进给页仍为“待开发”。

机器人页采用“左侧结果、右侧设置”：左侧展示 XYZ＋距离四列指标、可切换指标的历史趋势和加工点位预测，结果区延伸至页面底部；“历史记录”与“展开明细”在左侧底部并排。右侧为机器人手眼参数、基准、阈值、加工点位设置和运行日志。采集靶标、导入观测、清空日志、评估精度在右侧日志下方同排，不另占整页底栏。主页面业务操作按钮统一蓝底白字，设置类按钮保留白底。指标与曲线使用对应颜色，图例位于下拉框左侧。区域 1—8 仅为沟通约定，不在界面显示编号。

视觉相对监控使用相机到被评估末端（TCP）的手眼参数，不依赖完整机器人运动学模型。基准与复测按相同程序、固定位姿拍摄靶标；XYZ 使用机器人基座系。页面固定提供“绝对定位精度退化”“重复定位精度退化”“当前重复定位精度”三项指标，各指标分别设置阈值。实际统计口径取决于采样协议，说明中区分同方向重复与多方向到位散布。缺少坐标或绝对误差依据、同组样本不足时，相应指标留空。输入格式、操作及数学定义见 [机器人定位监控使用说明](docs/robot_position_monitoring.md)。

主轴回转页支持从空存储开始：批量导入 ZIP 并确认整批正常 → 随机初始化训练 → 每日导入与频谱/重建分析 → 人工标注并选择是否入训 → 人工重训并自动重算历史。初次训练和重训都只接收已判定正常且允许入训的数据，批量数据统一按导入批次判定。预警和故障检修阈值由人员设置；模型更新后提示复核阈值。结果来自真实数据，不加载研究仓库权重。操作、数据包协议和验证命令见 [主轴回转监控说明](docs/spindle_rotation_page.md)。

## 运行

使用已有 Conda 环境 `ZMSoftware`（Python 3.12），界面依赖 [PyQt6](https://pypi.org/project/PyQt6/)。

```powershell
conda activate ZMSoftware
cd D:\WorkSpace\ZMProject\SoftwareTools_PyQt
python -m pip install -r requirements.txt
python main.py
```

新机器可用 `conda env create -f environment.yml` 创建环境。已有 `ZMSoftware` 时直接安装本工程依赖即可。

开发调试时可执行 `python -m debug reset` 清空使用数据并刷新调试窗口，保留参数和阈值。独立入口、清理范围和使用方式见 [调试说明](debug/README.md)。

## 目录结构

```text
SoftwareTools_PyQt/
├── README.md                # 项目与目录说明
├── AGENTS.md                # 代码代理工作约定
├── .gitignore               # 缓存、环境和运行数据忽略规则
├── main.py                  # 桌面程序入口
├── requirements.txt         # PyQt6、NumPy、OpenCV、PyYAML、h5py、PyTorch
├── environment.yml          # Conda 环境定义
├── core/                    # Python 核心逻辑，可脱离界面使用
│   ├── algorithms/          # 算法、计算和数据处理
│   └── services/            # 业务流程，向界面提供调用入口
├── app/                     # PyQt 界面层
│   ├── main_window.py       # 主窗口、精度页面和选项卡
│   ├── resources.py         # 配置、样式和图标读取
│   ├── pages/               # 机器人定位页、主轴回转页及各自小弹窗
│   └── dialogs/             # 参数设置、选择和确认弹窗
├── resources/               # 随软件发布的静态资源
│   ├── icons/               # 按 branding/navigation/precision/window/common 分类
│   └── styles/              # QSS 样式与主题
├── config/                  # display.json：窗口尺寸、标题和状态颜色
├── data/                    # 输入数据与计算输出
│   ├── robot_error/         # 用户提供的原始图像与位姿记录
│   ├── processed/           # 处理后的数据
│   └── reports/             # 导出的报告和图表
├── storage/                 # 本地持久化内容
│   └── position_monitoring/ # 参数、图像、基准、观测、每日评估及日志
├── debug/                   # 标定、使用数据重置和 UR10 仿真工具
├── docs/                    # 需求、页面设计、接口与使用说明
├── tests/                   # 算法、服务和界面正式测试
└── experiments/             # 保留的流程复现、独立核验和数据打包工具
```

原始数据、调试场景资产和运行状态留在本地，不随 Git 提交。

## 开发边界

调用关系为 `app → core/services → core/algorithms`。

- **算法层**：接收明确的输入，返回计算结果，不读取界面控件，不依赖 PyQt。可先通过独立测试或实验验证。
- **服务层**：组织数据准备、算法调用和结果保存，提供界面可调用的业务接口。
- **界面层**：负责展示、用户输入和任务状态；耗时计算接入后台任务，避免阻塞界面。
- **配置与资源**：业务参数和路径放在 `config/`，图标、样式和 Designer 文件放在 `resources/`。

机器人定位的纯算法位于 `core/algorithms/position_monitoring.py` 和 `board_pose.py`；服务入口为 `core/services/position_monitoring_service.py`；页面调用服务，棋盘/ChArUco PnP 与评估在后台执行。仿真核验窗口位于 `app/dialogs/robot_position_simulation_dialog.py`，其中“相机与手眼标定”使用 `debug/diagnostics/camera_calibration.py` 处理配对图像和机器人位姿。完整流程的复现与独立核验脚本保存在 `experiments/20260928_position_end_to_end/`。

机器人默认参数、基准、观测、阈值、加工点位配置和评估历史保存在 `storage/position_monitoring/`。参数采用 `parameters/current.json` 和一份 `previous.json` 备份；支持图片、批次目录及已解算观测文件，导入图片后自动保存六维观测结果。每点不同方向各一次采用多方向工程散布，旧同方向重复协议继续兼容，详见 [使用说明](docs/robot_position_monitoring.md)。点位预测、相机在线采集及设备运动接口尚未接入；主轴数据通过每日ZIP导入，界面不控制机器人或主轴运动。

建立基准时保存一次实际评估，复测继续与所选基准比较。重启自动恢复最新观测、结果、趋势、图片和日志。真实观测按采集日期写入 `daily/YYYY-MM-DD.json`，缺少采集时间时明确使用导入时间；同日多次评估追加到该文件。仿真 B001/B002/B003 按第 0/1/2 天写入 `debug/day_0000.json` 等文件，保留真实操作时间。图像托管在 `images/`，观测在 `observations/`，基准引用已保存观测；重复评估不会复制整批图像和观测。

## 界面与图标

界面采用浅灰背景、白色面板和细边框。侧栏菜单不带外边框或圆角，使用深色线条图标和普通字重；仅选中时显示浅灰填充与左侧短竖线。精度选项卡选中时图标和底部指示条变绿。

返回按钮、软件标识、本机时间、最小化、最大化/还原和关闭按钮位于同一条顶部栏。左上角按访问历史返回上一精度选项卡，没有可返回记录时置灰。拖动顶部空白处移动窗口，双击顶部空白处切换最大化/还原，右下角可拖动调整窗口大小。

顶部标识暂时直接显示用户参考截图中的品牌区域，原始截图保存在 `resources/icons/branding/reference-interface.png`。显示区域通过 `config/display.json` 的 `branding.source_rect` 配置，后续可换成正式标识。

文字参考用户提供的上位机截图，采用深色微软雅黑：选项卡 20px 粗体、面板标题 18px、侧栏及正文 16px。“待开发”提示为 22px，未选中的标签也保持清晰可读。

- `config/display.json`：修改窗口标题、尺寸、侧栏宽度、统一字体和界面配色。
- `resources/styles/light.qss`：修改页面布局样式、字体和边框。
- `resources/icons/`：精度选项卡采用机械臂、主轴回转、主轴向工件进给的 SVG 图标；其他图标和参考图片也在此分类管理。来源与替换方法见该目录的 [说明](resources/icons/README.md)。

三个选项卡共用一套样式，支持鼠标点击和键盘方向键切换。底部绿色条表示当前选中页，不表示测量进度。

窗口支持整体缩放：以 1600 × 1000 为设计基准，按当前窗口宽、高计算同一个比例，统一调整字体、图标、标题栏、控件尺寸和间距。左右结果与设置区按约 65:35 分配。窗口可以自由调整长宽，缩放不会重建页面或清除选中状态；启动及换屏时限制窗口不超过屏幕可用区域。

缩放参数集中在 `config/display.json` 的 `scaling`：`reference_size` 为设计基准，`minimum`、`maximum` 为缩放范围，当前为 0.75–2.0。`window_size` 是初始窗口尺寸，不是固定分辨率。窗口很小时保留纵向滚动，避免文字无限缩小。`resources/styles/light.qss` 和布局代码中的数值均按设计尺寸填写，不要手工预乘缩放比例；弹窗调用 `app.resources.fit_dialog()` 沿用主窗口比例。

Windows 系统 DPI 继续使用 [Qt 6 原生支持](https://doc.qt.io/qt-6/highdpi.html#configuring-windows)，程序不额外设置全局 `QT_SCALE_FACTOR`；测试中才使用该变量模拟系统缩放。

## 数据与实验

`data/` 和 `storage/` 中的运行内容默认不提交。需要随工程发布的小型测试样例可放在 `tests/` 中。

复现工具按 `experiments/YYYYMMDD_用途/` 组织。保留有效复现脚本和正式测试，清理已无用途的临时逻辑；批量输出、日志、模型权重和大文件留在本地。每轮试验后询问是否更新 `AGENTS.md`，确认后仅写入稳定的全局结论；过程、数值和截图放在对应文档或本地报告，已确认记录见 [维护记录](docs/validation_history.md)。新增临时测试代码完成后询问是否保留。

## 本地目录与分支

本地开发目录为 `D:\WorkSpace\ZMProject\SoftwareTools_PyQt`，对应同一 Git 仓库的独立工作树。机器人定位功能分支为 `codex/robot-position-monitoring`，远程为 `ZMS`；基础框架保留在 `develop`。

```powershell
cd D:\WorkSpace\ZMProject\SoftwareTools_PyQt
git status
git push ZMS codex/robot-position-monitoring
```

`develop` 从空白建立，以当前骨架作为新历史的首次提交。原工程继续保存在旁边 `SoftwareTools` 工作树的 `master` 中，后续需要复用的功能再按任务迁入。

业务页填充前的框架已推送：提交 `3d25535`，标签 `础框架`（按用户指定名称）。该标签不包含后续机器人业务页布局。

## 维护检查

```powershell
python -B -m pytest -q -p no:cacheprovider tests
python -B -m ruff check --no-cache main.py app core debug tests experiments
git diff --check
```

测试使用隔离数据，不改变主应用的参数、模型和历史。稳定的全局约定与架构见 [AGENTS.md](AGENTS.md)，业务边界见上文对应模块说明。
