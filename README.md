# ZMSoftware Robot Digital Twin Tool

本仓库是本地运行的机器人数字孪生工具软件工程，用于承载 UR 机器人三维仿真、误差参数加载、标定数据分析、精度状态评估、健康状态评估和精度校验流程。

当前阶段重点是软件开发环境、目录骨架和算法包归档。核心贝叶斯标定算法已归档到：

```text
core/calibration/bayesian_calibration_pipeline/
```

算法样本数据归档到：

```text
data/calibration/bayesian_calibration_pipeline/
```

推荐使用已创建的 Conda 环境：

```powershell
conda activate ZMSoftware
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

启动桌面软件入口：

```powershell
python main.py
```
