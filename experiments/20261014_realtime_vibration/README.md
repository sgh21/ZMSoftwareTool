# 真实时间连续回放验证

工具启动真实 Qt 实时窗口，使用隔离服务目录，不连接硬件、不修改原始记录或主业务存储。默认以 1× 流速播放所选完整记录，不能用加速播放代替长期稳定性验证。

```powershell
& 'D:\Softwares\miniconda3\envs\ZMSoftware\python.exe' -B experiments/20261014_realtime_vibration/validate_replay.py --run-dir data/spindle_error/20260723/run_20260723_095732_7000rpm_20min_0723_morning2 --report-dir data/reports/realtime_vibration_20261014/full_final
```

该记录名称标称20分钟，实际1329.76秒。完整验证约需22分10秒，另有后台预扫时间。`--duration-seconds 15` 可做短测，报告会明确 `complete=false`，不能用于完整20分钟结论。

产出 `frames.csv`、每分钟 `progress.json`、最终 `validation.json` 和原生 Qt 截图。指标分别记录真实帧间隔、界面数据更新时间、Qt paint 耗时、原始样本消费、固定纵轴、显示点数及进程 RSS；paint 包装只在验证进程内生效。

工具依赖项目开发环境中已安装的 `psutil`；它仅用于这个验证进程的内存采样，不参与应用回放。完成状态、每通道样本总数、帧间隔、固定纵轴及有界缓存由报告直接核验，不能只看运行时间作结论。
