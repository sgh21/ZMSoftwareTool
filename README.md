# SoftwareTools · Python / PyQt

本地桌面软件的重新设计工程。Python 负责核心算法和业务逻辑，PyQt 负责界面展示与交互。

当前已搭建 PyQt6 桌面框架：左侧仅保留“精度监控”，首页包含“机器人末端定位精度”“主轴回转精度”“主轴轴向进给精度”三个选项卡。选中时图标和底部指示条变为绿色，各页显示“待开发”。核心算法和业务内容尚未实现。

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
│   ├── pages/               # 预留，出现独立页面职责后再拆分
│   ├── widgets/             # 预留，出现实际复用后再拆分
│   └── dialogs/             # 参数设置、选择和确认弹窗
├── resources/               # 随软件发布的静态资源
│   ├── ui/                  # 可选的 Qt Designer .ui 文件
│   ├── icons/               # 按 branding/navigation/precision/window/common 分类
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

空目录内的 `.gitkeep` 只用于让 Git 保留目录，没有运行逻辑。后续按实际功能逐步添加文件。

## 开发边界

调用关系为 `app → core/services → core/algorithms`。

- **算法层**：接收明确的输入，返回计算结果，不读取界面控件，不依赖 PyQt。可先通过独立测试或实验验证。
- **服务层**：组织数据准备、算法调用和结果保存，提供界面可调用的业务接口。
- **界面层**：负责展示、用户输入和任务状态；耗时计算接入后台任务，避免阻塞界面。
- **配置与资源**：业务参数和路径放在 `config/`，图标、样式和 Designer 文件放在 `resources/`。

目前实际代码只有三个实现文件：`main.py`、`app/main_window.py`、`app/resources.py`。主窗口、精度页面和选项卡放在同一个界面文件中，方便定位和修改；后续有实际复用或独立业务职责再拆分。

目前界面没有调用核心层，也没有模拟测量值或连接设备。后续从 `app/main_window.py` 中为各选项卡接入实际页面。开发以简单直接、便于维护为原则，不为假设的罕见情况堆叠防御代码；具体约定见 `AGENTS.md`。

## 界面与图标

界面采用浅灰背景、白色面板和细边框。侧栏菜单不带外边框或圆角，使用深色线条图标和普通字重；仅选中时显示浅灰填充与左侧短竖线。精度选项卡选中时图标和底部指示条变绿。

返回按钮、软件标识、本机时间、最小化、最大化/还原和关闭按钮位于同一条顶部栏。左上角按访问历史返回上一精度选项卡，没有可返回记录时置灰。拖动顶部空白处移动窗口，双击顶部空白处切换最大化/还原，右下角可拖动调整窗口大小。

顶部标识暂时直接显示用户参考截图中的品牌区域，原始截图保存在 `resources/icons/branding/reference-interface.png`。显示区域通过 `config/display.json` 的 `branding.source_rect` 配置，后续可换成正式标识。

文字参考用户提供的上位机截图，采用深色微软雅黑：选项卡 20px 粗体、面板标题 18px、侧栏及正文 16px。“待开发”提示为 22px，未选中的标签也保持清晰可读。

- `config/display.json`：修改窗口标题、尺寸、侧栏宽度、统一字体和界面配色。
- `resources/styles/light.qss`：修改页面布局样式、字体和边框。
- `resources/icons/`：精度选项卡采用机械臂、主轴回转、主轴向工件进给的 SVG 图标；其他图标和参考图片也在此分类管理。来源与替换方法见该目录的 [说明](resources/icons/README.md)。

三个选项卡共用一套样式，支持鼠标点击和键盘方向键切换。底部绿色条表示当前选中页，不表示测量进度。

## 数据与实验

`data/` 和 `storage/` 中的运行内容默认不提交，仅跟踪目录占位文件。需要随工程发布的小型测试样例可放在 `tests/` 中。

实验按 `experiments/YYYYMMDD_用途/` 组织。可提交可复现脚本和简短说明；批量输出、日志、模型权重和大文件留在本地。实验结果写入 `AGENTS.md` 前需用户确认，临时测试代码是否保留也需询问用户。

## 本地目录与分支

本地开发目录为 `D:\WorkSpace\ZMProject\SoftwareTools_PyQt`，对应同一 Git 仓库的独立工作树。开发分支为 `develop`，远程为 `ZMS`，对应 `ZMS/develop`。

```powershell
cd D:\WorkSpace\ZMProject\SoftwareTools_PyQt
git status
git push ZMS develop
```

`develop` 从空白建立，以当前骨架作为新历史的首次提交。原工程继续保存在旁边 `SoftwareTools` 工作树的 `master` 中，后续需要复用的功能再按任务迁入。

## 建议开发顺序

1. 确认当前框架布局和正式软件标识。
2. 在 `docs/` 明确各精度页面的业务流程、指标和数据输入输出。
3. 在 `core/` 独立实现并验证算法，再通过服务接口接入界面。
