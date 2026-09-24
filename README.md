# SoftwareTools · Python / PyQt

本地桌面软件的重新设计工程。Python 负责核心算法和业务逻辑，PyQt 负责界面展示与交互。

**当前仅有目录骨架和说明文档，尚未实现算法、界面或启动入口。** Python、PyQt 的具体版本及第三方依赖将在开始实现时确定。

## 目录结构

```text
SoftwareTools_PyQt/
├── README.md                # 项目与目录说明
├── AGENTS.md                # 代码代理工作约定
├── .gitignore               # 缓存、环境和运行数据忽略规则
├── core/                    # Python 核心逻辑，可脱离界面使用
│   ├── algorithms/          # 算法、计算和数据处理
│   └── services/            # 业务流程，向界面提供调用入口
├── app/                     # PyQt 界面层
│   ├── pages/               # 业务页面
│   ├── widgets/             # 可复用界面控件
│   └── dialogs/             # 参数设置、选择和确认弹窗
├── resources/               # 随软件发布的静态资源
│   ├── ui/                  # 可选的 Qt Designer .ui 文件
│   ├── icons/               # 图标和界面图片
│   └── styles/              # QSS 样式与主题
├── config/                  # 路径、设备、算法及界面配置
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

当前只约定上述职责，具体算法模块、页面结构和接口由后续设计确定。

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

目前没有安装或启动命令；开始实现时再补充依赖清单、环境配置和程序入口。

## 建议开发顺序

1. 在 `docs/` 明确业务流程、页面结构和数据输入输出。
2. 确认 Python / PyQt 版本，补充依赖和最小可运行界面。
3. 在 `core/` 独立实现并验证算法，再通过服务接口接入界面。
