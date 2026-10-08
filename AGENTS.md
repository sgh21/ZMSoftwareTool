# AGENTS.md

新会话先阅读根目录的 AGENTS.md / AGENT.md，再扫描、阅读或修改工程。本文件只保留稳定的全局规则、架构和文档入口；功能细节、当前状态、待办和验收记录放在对应文档或本地报告。

## 工作约定

- 开发目录为 `D:\WorkSpace\ZMProject\SoftwareTools_PyQt`；相邻 `SoftwareTools` 是旧工程，不向其写入新代码。先看 `git status`，保留已有未提交工作，分支与运行状态以实际结果为准。
- 使用 `D:\Softwares\miniconda3\envs\ZMSoftware\python.exe`（Python 3.12）。入口为 `main.py`，依赖见 `requirements.txt` 和 `environment.yml`，不另建虚拟环境。
- 优先复用现有功能和 Qt 原生控件，保持直接的数据流。只保留真实输入、文件操作和业务边界所需检查，不为假设的扩展增加框架、抽象、重试或兼容分支，不使用 SHA-256。
- 清理前核对生产调用、Qt 回调、命令行入口及测试引用。保留有效复现工具和正式回归测试，不为减少代码改变功能、输入协议或历史兼容。
- 整理代码不得删除或改写原始图像、采集记录、参数、基准、标签、模型和历史。`data/`、`storage/`、仿真资产与批量输出留在本地，不提交。
- 验证与改动相称，使用隔离目录，不改主业务存储，不用虚构结果充当测量。批量渲染、真实数据全流程和网络重训仅在任务需要时运行。
- 每轮试验验证后询问是否更新 AGENTS.md；经确认后只写入稳定的全局结论，过程、数值和截图放在对应文档或本地报告。新增临时测试代码完成后询问是否保留，不保留则及时清理；一次性检查优先通过标准输入执行。
- 中文简洁沟通，文档直接撰写。默认不使用 Superpowers，除非用户明确调用；图片生成优先使用 ChatGPT image2，不用本地 PIL 绘图库生成。

## 架构与边界

本工程为 PyQt6 Widgets 本地桌面软件，负责数据接收、分析、评价、追溯和报警；机器人运动、主轴运转及拍照由设备集控负责。调用关系为 `app → core/services → core/algorithms`。

| 位置 | 职责 |
| --- | --- |
| `main.py`、`app/main_window.py` | 启动、窗口与导航 |
| `app/pages/`、`app/dialogs/` | 展示、输入与任务状态，通过服务调用算法 |
| `core/services/` | 组织业务流程、数据准备与持久化 |
| `core/algorithms/` | 纯算法，不依赖 Qt，不读取界面控件或调试路径 |
| `app/resources.py`、`config/`、`resources/` | 集中管理配置、样式、图标与缩放 |
| `diagnostics/`、`debug/`、`experiments/` | 标定、独立调试及保留的流程复现工具 |
| `tests/` | 算法、服务与 Qt 正式回归 |
| `data/`、`storage/` | 本地输入、输出与持久化状态 |

- 耗时导入、图像解算、标定、训练和历史计算在后台执行；经 Qt 信号回主线程更新界面，处理、保存及刷新完成才到 100%。
- 沿用浅灰背景、白色面板、深色微软雅黑；业务按钮蓝底白字，设置按钮白底。窗口移动用 `startSystemMove()`，调整大小用 `QSizeGrip`，不引入窗口框架。
- 尺寸按 1600×1000 设计值填写，由 `UiScale` 在 0.75–2.0 内统一缩放；系统 DPI 交给 Qt，不重复乘比例或设备像素比，弹窗使用 `fit_dialog()`。

## 按任务阅读

- 机器人协议、保存与指标定义：[机器人定位监控](docs/robot_position_monitoring.md)；布局与交互：[机器人页面](docs/robot_position_page.md)。
- 标定：[相机与手眼标定](docs/camera_hand_eye_debug.md)；完整复现：[机器人全流程](docs/robot_position_end_to_end.md)。
- 主轴导入、标签、训练、检测、线程约束与复现：[主轴回转](docs/spindle_rotation_page.md)。
- 调试入口：[调试说明](debug/README.md)；仿真坐标、固定计划、批次与补采：[UR10 场景](debug/ur10_charuco_scene/README.md)。
- 已确认的历史验收与报告入口：[维护记录](docs/validation_history.md)。记录描述对应时点，接续任务先读实际参数、record 和验收索引。

常用检查（使用上述开发解释器，按改动范围选择）：

```powershell
python -B -m pytest -q -p no:cacheprovider tests
python -B -m ruff check --no-cache main.py app core diagnostics tests experiments
git diff --check
```
