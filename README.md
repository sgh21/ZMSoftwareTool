# SoftwareTools · Python / PyQt

本地桌面软件的重新设计工程。Python 负责核心算法和业务逻辑，PyQt 负责界面展示与交互。

当前已搭建 PyQt6 桌面框架：左侧仅保留“精度监控”，首页包含“机器人末端定位精度”“主轴回转精度”“主轴轴向进给精度”三个选项卡。选中时图标和底部指示条变为绿色。机器人页和主轴回转页已填充布局预览，轴向进给页仍为“待开发”，底层逻辑尚未接入。

机器人页采用“左侧结果、右侧设置”：左侧展示 XYZ＋距离四列指标、可切换指标的历史趋势和加工点位预测，结果区延伸至页面底部；“历史记录”与“展开明细”在左侧底部并排。右侧为机器人手眼参数、基准、阈值、加工点位设置和运行日志。采集靶标、导入观测、清空日志、评估精度在右侧日志下方同排，不另占整页底栏。主页面业务操作按钮统一蓝底白字，设置类按钮保留白底。指标与曲线使用对应颜色，图例位于下拉框左侧。区域 1—8 仅为沟通约定，不在界面显示编号。

视觉相对监控使用相机到被评估末端（TCP）的手眼参数，不依赖完整机器人运动学模型。基准与复测按相同程序、固定位姿拍摄靶标；页面只显示 XYZ 与距离，不展示姿态指标。XYZ 默认采用机器人基坐标系，阈值弹窗仅有 X、Y、Z 和距离四项，无坐标系选择。转换到基坐标还需每个测量点的初始末端朝向，后续由基准关联数据提供。公式、前提和待确认接口见 [页面设计](docs/robot_position_page.md)。

主轴回转页沿用左侧结果、右侧设置布局：左侧先显示解析评价、网络评分、温度、电流及综合结论，下面为采集信号、振动频谱、频带能量和网络分布四幅图；右侧放采集方案、人工确认正常样本、训练入口、阈值和日志。阈值仅有网络评分、综合评价两类，解析评价不设独立阈值。采集、导入、清空日志与评估在右侧底部同排。按用户要求绘制了明确标注的示意曲线，便于审阅排版；真实结果仍为空，不执行采集、FFT、能量计算或网络训练。详见 [主轴回转页设计](docs/spindle_rotation_page.md)。

主轴信号通过“类型 → 通道”的二级菜单选择，振动 X/Y/Z 联动频谱与能量图，温度支持多个测点。占位通道名称和数量在 `config/display.json` 的 `spindle_preview_channels` 中修改。页面共用一组图例与一个示意标识，补充说明放在悬停提示中。

## 运行

使用已有 Conda 环境 `ZMSoftware`（Python 3.12），界面依赖 [PyQt6](https://pypi.org/project/PyQt6/)。

```powershell
conda activate ZMSoftware
cd D:\WorkSpace\ZMProject\SoftwareTools_PyQt
python -m pip install -r requirements.txt
python main.py
```

新机器可用 `conda env create -f environment.yml` 创建环境。已有 `ZMSoftware` 时直接安装本工程依赖即可。

## 目录结构

```text
SoftwareTools_PyQt/
├── README.md                # 项目与目录说明
├── AGENTS.md                # 代码代理工作约定
├── .gitignore               # 缓存、环境和运行数据忽略规则
├── main.py                  # 桌面程序入口
├── requirements.txt         # PyQt6 界面依赖
├── environment.yml          # Conda 环境定义
├── core/                    # Python 核心逻辑，可脱离界面使用
│   ├── algorithms/          # 算法、计算和数据处理
│   └── services/            # 业务流程，向界面提供调用入口
├── app/                     # PyQt 界面层
│   ├── main_window.py       # 主窗口、精度页面和选项卡
│   ├── resources.py         # 配置、样式和图标读取
│   ├── pages/               # 机器人定位页、主轴回转页及各自小弹窗
│   ├── widgets/             # 预留，出现实际复用后再拆分
│   └── dialogs/             # 参数设置、选择和确认弹窗
├── resources/               # 随软件发布的静态资源
│   ├── ui/                  # 可选的 Qt Designer .ui 文件
│   ├── icons/               # 按 branding/navigation/precision/window/common/robot 分类
│   └── styles/              # QSS 样式与主题
├── config/                  # display.json：窗口尺寸、标题和状态颜色
├── data/                    # 输入数据与计算输出
│   ├── raw/                 # 原始采集或导入数据
│   ├── processed/           # 处理后的数据
│   └── reports/             # 导出的报告和图表
├── storage/                 # 本地持久化内容
│   ├── models/              # 模型、权重和参数版本
│   └── records/             # 历史记录和数据库
├── docs/                    # 需求、页面设计、接口与使用说明
├── tests/                   # 正式测试
│   ├── core/                # 算法与服务测试
│   └── app/                 # 界面与交互测试
└── experiments/             # 算法试验和临时调试
```

上面也包含规划目录；空目录不随 Git 保存，后续按实际功能添加文件。

## 开发边界

调用关系为 `app → core/services → core/algorithms`。

- **算法层**：接收明确的输入，返回计算结果，不读取界面控件，不依赖 PyQt。可先通过独立测试或实验验证。
- **服务层**：组织数据准备、算法调用和结果保存，提供界面可调用的业务接口。
- **界面层**：负责展示、用户输入和任务状态；耗时计算接入后台任务，避免阻塞界面。
- **配置与资源**：业务参数和路径放在 `config/`，图标、样式和 Designer 文件放在 `resources/`。

目前实际代码只有五个实现文件：`main.py`、`app/main_window.py`、`app/resources.py`、`app/pages/robot_position_page.py` 和 `app/pages/spindle_rotation_page.py`。两个业务页各放一个文件，各页的小控件、绘图及弹窗集中在对应文件内，未增加绘图库或其他依赖。

目前界面没有调用核心层或连接设备，主轴示意图不参与评估或保存到历史。机器人区域 7 的“点位管理”支持增删、编辑编号与理论坐标、上下移及顺序编号，点击“应用”更新区域 3，仅保留在当前会话；取消会丢弃本次编辑。预测指标始终留空，阈值设置仍为不保存的预览。区域 8 显示操作与输入错误，采集、导入和评估按钮会说明尚未接入，未执行实际任务。小窗口可纵向滚动。开发以简单直接、便于维护为原则，具体约定见 `AGENTS.md`。

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

实验按 `experiments/YYYYMMDD_用途/` 组织。可提交可复现脚本和简短说明；批量输出、日志、模型权重和大文件留在本地。实验结果写入 `AGENTS.md` 前需用户确认，临时测试代码是否保留也需询问用户。

## 本地目录与分支

本地开发目录为 `D:\WorkSpace\ZMProject\SoftwareTools_PyQt`，对应同一 Git 仓库的独立工作树。开发分支为 `develop`，远程为 `ZMS`，对应 `ZMS/develop`。

```powershell
cd D:\WorkSpace\ZMProject\SoftwareTools_PyQt
git status
git push ZMS develop
```

`develop` 从空白建立，以当前骨架作为新历史的首次提交。原工程继续保存在旁边 `SoftwareTools` 工作树的 `master` 中，后续需要复用的功能再按任务迁入。

业务页填充前的框架已推送：提交 `3d25535`，标签 `础框架`（按用户指定名称）。该标签不包含后续机器人业务页布局。

## 建议开发顺序

1. 确认当前框架布局和正式软件标识。
2. 在 `docs/` 明确各精度页面的业务流程、指标和数据输入输出。
3. 在 `core/` 独立实现并验证算法，再通过服务接口接入界面。
