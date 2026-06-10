# UI_COMPLETION_STATUS.md

## 当前 UI 完成情况

更新时间：2026-05-31

## 已完成

- 主窗口入口：`main.py` 创建 `QApplication` 并打开 `MainWindow`。
- 主窗口基础尺寸：默认 `1440 x 810`，最小尺寸 `1180 x 700`。
- 初始化主页面：`InitializationPage` 已作为主窗口中心页面。
- 顶部导航区：包含文件、编辑、查看、视图、校验、调试入口。
- 顶部状态区：显示当前项目、连接状态、配置状态。
- 窗口控制按钮：顶部 `−`、`□`、`×` 已分别接入最小化、最大化/还原、关闭动作。
- 默认模型加载：启动时尝试加载 `models/urdf/ur10.urdf`。
- 默认参数加载：启动时尝试加载 `config/calibration_result.yaml`，其次加载 `storage/model_versions/active_calib_params.yaml`。
- 机器人仿真视图：`RobotSimulationWidget` 基于 yourdfpy + PyVista off-screen 渲染 URDF/Mesh。
- 机器人配色：从 `config/theme.yaml` 读取状态颜色与 mesh 配色映射。
- 精度状态卡片：显示位置误差 RMS、最大误差、超差阈值和当前结论。
- 健康状态卡片：显示健康环、模型置信度、当前状态和更新时间。
- 参数文件加载：支持 YAML/JSON 参数文件基础解析和状态刷新。
- 三维模型加载：支持 `.urdf/.xacro/.stl/.dae/.obj` 后缀选择，当前渲染器主要支持 `.urdf`。
- UR10 关节调试：通过顶部“调试”按钮打开独立窗口，支持应用关节角并刷新仿真。
- 参数辨识入口：初始化页内置 S1 参数辨识数据加载、运行、保存和 HTML 报告生成入口。
- 参数辨识进度：耗时辨识运行时显示进度弹窗。
- 辨识结果持久化：保存到 `config/calibration_result.yaml` 并写入 SQLite 历史。

## 部分完成

- 顶部“文件/编辑/查看/视图/校验”目前主要是导航占位，未形成完整菜单动作。
- “通知/帮助/设置”按钮已有 UI 占位，具体功能待实现。
- `app/pages/calibration_page.py` 仍保留独立辨识页面，与初始化页内置辨识区域存在功能重叠，后续需要明确整合方式。
- 精度校验相关配置和基础模块存在，但完整 UI 流程和真实校验闭环待确认。
- 报告输出已有 HTML 生成功能，统一报告管理和历史查看入口待完善。

## 待实现或待确认

- 完整菜单栏动作：项目创建/打开、报告目录、历史记录、校验配置等。
- 主窗口状态栏与日志面板的完整实现。
- 精度校验页面：位姿读取、拍摄流程、图像对比、校验报告和异常处理。
- 模型版本管理 UI：模型更新前备份、版本切换、历史追溯。
- `config/calib_params.yaml` 与 `config/calibration_result.yaml` 的职责边界。
- Linux 图形环境下 PyVista/VTK 渲染兼容性。

## 本次 UI 基础功能更新

- 将顶部窗口控制按钮设置为独立 objectName，便于测试和维护。
- `window_minimize_button` 调用 `showMinimized()`。
- `window_maximize_button` 在 `showMaximized()` 与 `showNormal()` 之间切换。
- `window_close_button` 调用 `close()`。
- 新增测试覆盖窗口控制按钮基础行为。
