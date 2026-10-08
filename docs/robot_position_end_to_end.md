# 机器人定位全流程复现

运行环境为已有 `ZMSoftware`，在 `SoftwareTools_PyQt` 根目录执行。先关闭软件，备份并移出 `storage/position_monitoring/`，保留原始数据和 `config/robot_position.json` 的未配置默认值。已有基线及历史不要直接删除。

日常使用可直接运行 `python main.py`，进入“调试 → 相机与手眼标定”，用 B000 准备参数，然后在主页面加载导出文件，导入 B001、建立基准，再依次导入 B002、B003 并评估。基准建立时保存一次基准评估；重启自动恢复最后的 B003 结果、B001 基准、图片和三次历史。标定窗口用法见 [相机与手眼标定调试](camera_hand_eye_debug.md)。

复现脚本操作真实 Qt 页面，文件选择器由脚本提供路径，图像解算、后台服务及持久化均使用生产代码。

```powershell
$dataset = 'data/robot_error/approved_30poses_three_inertias'
$report = 'data/reports/robot_position_e2e_new_run'
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage calibration
$calibration = Get-Content -LiteralPath "$report/calibration.json" -Raw | ConvertFrom-Json
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --parameters $calibration.monitoring_parameters_path --report-root $report --stage initial
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage current --batch B002
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage current --batch B003
python -B -X utf8 experiments/20260928_position_end_to_end/run_workflow.py --dataset $dataset --report-root $report --stage reopen
```

每阶段为独立进程，检查参数、基线及历史自动恢复；还检查已导入但未评估时的观测恢复。`initial` 要求空白状态；脚本不自动删除已有配置。输出截图、操作记录和独立核验 JSON，最后保留软件状态。仿真批次按用户约定 B001/B002/B003 显示为第 0/1/2 天；真实导入和评估时间另存，不能把模拟日解释成实际采集日期。

若有生成时冻结的 `target_bias.json`，在 B003 命令追加 `--bias-reference <路径>`。该文件只传给事后独立核验，测量服务仍只使用图像和固定系统参数；缺少偏置表时，报告会明确标记偏置是从 actual 残差估计，不能称为独立验证了设定向量。

独立核验直接使用 actual 的基座位置计算多方向散布，并用已测末端位置重算软件指标，不调用生产统计函数构造期望值。新结果按专利 V6 计算相对同期中心的 RMS、RMS 跨期变化与中心漂移模长，保存在 `patent_v6` 中；旧平均半径加 3 倍标准差、actual−ideal 绝对误差仍保留作历史诊断。对照时依据评估的 `metric_definition` 选择同一统计定义。`statistics_check.passed` 只表示计算一致；测量是否准确还需查看 `metric_comparison` 和 `position_measurement_errors`。

最终报告为 [测试流程与结果](../data/reports/robot_position_final_20260928/测试流程与结果.md)（本地报告，不提交批量数据）。`verification_B002.json`、`verification_B003.json` 分别核验与 B001 的比较；`comparison_B002_B003.json` 只用于报告，按保存的测量位置重算同惯性条件下的差异，不改变软件基准、不生成额外评估历史。

最终运行状态在 `storage/position_monitoring/`：参数、基准、观测及图像由软件托管；每天的评估集中在一个 JSON 中，同一天多次评估追加保存。软件结果和独立核验的差异验证统计实现，图像结果与 actual 的差异验证测量能力，两者须分别报告。

模板：`config/examples/robot_hand_eye.json` 仅包含手眼；`config/examples/robot_position_simulation.json` 包含固定仿真真值系统参数。实际运行使用本次标定导出的内参时，应加载标定输出的完整 `parameters.json`。所有模板均须按实际设备的坐标及参数替换，不能把仿真手眼当作实机标定。
