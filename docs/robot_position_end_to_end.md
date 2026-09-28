# 机器人定位全流程复现

运行环境为已有 `ZMSoftware`，在 `SoftwareTools_PyQt` 根目录执行。先关闭软件，备份并移出 `storage/position_monitoring/`，保留原始数据和 `config/robot_position.json` 的未配置默认值。已有基线及历史不要直接删除。

日常使用可直接运行 `python main.py`，进入“调试 → 相机与手眼标定”，用 B000 准备参数，然后在主页面加载导出文件，导入 B001、建立基准，重启后导入 B002、评估。标定窗口用法见 [相机与手眼标定调试](camera_hand_eye_debug.md)。

复现脚本操作真实 Qt 页面，文件选择器由脚本提供路径，图像解算、后台服务及持久化均使用生产代码。

```powershell
$dataset = 'data/robot_error/approved_30poses_three_inertias'
$report = 'data/reports/robot_position_e2e_new_run'
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage calibration
$calibration = Get-Content -LiteralPath "$report/calibration.json" -Raw | ConvertFrom-Json
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --parameters $calibration.monitoring_parameters_path --report-root $report --stage initial
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage current
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage reopen
```

每阶段为独立进程，检查参数、基线及历史重启恢复。`initial` 要求空白状态；脚本不自动删除已有配置。输出截图、操作记录和独立核验 JSON。B 编号只代表批次，脚本不伪造采集日期。

独立核验直接使用 actual 的基座位置计算多方向散布，使用 actual−ideal 算绝对误差，并用已测末端位置重算软件指标，不调用生产统计函数构造期望值。`statistics_check.passed` 只表示计算一致；测量是否准确还需查看 `metric_comparison` 和 `position_measurement_errors`。

本次已完成的报告为 [2026-09-28 测试流程与结果](../data/reports/robot_position_e2e_20260928/测试流程与结果.md)（本地报告，不提交批量数据）。主流程和真值参数对照均完成 B001/B002 共 1200 张测量；软件统计一致，但当前空间散布约 0.609 mm，真值约 0.407 mm，视觉测量偏差仍需研究。

模板：`config/examples/robot_hand_eye.json` 仅包含手眼；`config/examples/robot_position_simulation.json` 包含固定仿真真值系统参数。实际运行使用本次标定导出的内参时，应加载标定输出的完整 `parameters.json`。所有模板均须按实际设备的坐标及参数替换，不能把仿真手眼当作实机标定。
