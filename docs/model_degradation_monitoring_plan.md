# 机器人长期退化监控任务理解与改动计划

## 任务理解

当前软件已经通过 `config/calibration_result.yaml` 维护一份辨识后的机器人误差模型。该模型能在给定关节角时计算末端相对于机器人基坐标系的预测位姿或位置误差。长期运行后，机器人真实到位状态会因为磨损、间隙、热变形或刚度变化逐渐偏离初始健康状态，因此只依赖初始辨识模型会低估当前定位风险。

这次需求的目标不是重新辨识完整模型参数，而是先引入一个固定标定板长期观测闭环：

1. 机器人末端安装相机，工作台固定棋盘格或标定板。
2. 初始健康状态下，机器人运动到一组固定监控关节位姿，采集标定板图像并通过 PnP 解算 `C_T_B`，形成参考观测。
3. 后续定期运动到相同关节位姿，再次采集图像或加载已解算的 PnP 位姿。
4. 比较参考 PnP 和当前 PnP 的相对变化，推导末端真实位姿相对于初始健康状态的漂移。
5. 将漂移写入评估日志，并让用户决定是否把本次退化评估结果写回当前模型的置信度和定位精度评估不确定性。

## 数学落地

PDF 中采用坐标变换记号 `A_T_B`，表示从坐标系 `B` 到坐标系 `A` 的齐次变换。本项目首版实现使用以下约定：

- `E_T_C`：手眼参数，相机坐标系到末端坐标系的齐次变换，对应需求中给出的 `T_C2T` 和数据文件中的 `T_tool_cam`。
- `C_T_E = inv(E_T_C)`。
- `C_T_B(i,0)`：第 `i` 个固定监控位姿在初始健康状态下的 PnP 结果。
- `C_T_B(i,k)`：第 `i` 个固定监控位姿在第 `k` 次监控时的 PnP 结果。

在标定板固定、相机刚性固定的假设下，末端相对漂移为：

```text
Delta_T_i(k) = E_T_C * C_T_B(i,0) * inv(C_T_B(i,k)) * C_T_E
```

物理意义是：

```text
Delta_T_i(k) ~= inv(W_T_E(i,0)) * W_T_E(i,k)
```

也就是相同关节指令下，当前真实末端位姿相对于初始健康状态真实末端位姿的变化。

对每个位姿计算 `se(3)` 李代数误差向量：

```text
xi_i(k) = Log(Delta_T_i(k))^vee = [rho_i(k), phi_i(k)]
```

其中 `rho` 用米表达等效平移漂移，`phi` 用弧度表达姿态漂移。首版输出：

- 单个位姿位置漂移：`||rho_i|| * 1000`，单位 mm。
- 单个位姿姿态漂移：`||phi_i||` 转为 deg。
- 整体位置漂移 RMS：所有位姿位置漂移的 RMS。
- 整体姿态漂移 RMS：所有位姿姿态漂移的 RMS。
- 若存在健康基线均值和协方差，则计算 Mahalanobis 统计量；首版允许缺省，不阻塞评估。

PDF 第 10.4 节的参数漂移反演需要 `J_phi(q)` 参数灵敏度矩阵和噪声协方差建模，属于后续增强。本次首版只更新模型置信度和定位精度评估不确定性，不直接改写 MD-H 或误差参数。

## YAML 字段设计

标定结果和监控硬件参数需要分开管理。`config/calibration_result.yaml` 会被“模型标定分析”完整重写，因此只保存当前机器人误差模型、模型置信度和随模型状态变化的定位不确定性。手眼、相机内参、棋盘格规格和评估权重放入独立的 `config/model_monitoring.yaml`，避免重新标定机器人误差模型时覆盖外部传感器配置。

```yaml
# config/calibration_result.yaml
identification:
  confidence: 97.16
  metrics:
    position_error_rmse_mm: 9.37
    position_uncertainty_rmse_mm: 0.0
  monitoring:
    last_degradation_evaluation:
      timestamp: ...
      reference_source: ...
      current_source: ...
      sample_count: ...
      position_drift_rms_mm: ...
      orientation_drift_rms_deg: ...
      confidence_before: ...
      confidence_after: ...
      position_uncertainty_before_mm: ...
      position_uncertainty_after_mm: ...

# config/model_monitoring.yaml
model_monitoring:
  hand_eye:
    convention: E_T_C
    T_tool_camera: [[...], [...], [...], [...]]
  camera:
    K: [[...], [...], [...]]
    D: [...]
    board_grid: [39, 34]
    square_size_mm: 0.5
  evaluation:
    orientation_weight_mm_per_rad: 100.0
```

服务层会优先读取 `config/model_monitoring.yaml`，同时保留对旧版 `identification.monitoring.hand_eye/camera/evaluation` 的兼容读取。写回退化评估时只更新 `confidence`、`metrics.position_uncertainty_rmse_mm` 和 `monitoring.last_degradation_evaluation`，不会再把手眼或相机内参写入模型结果文件。

给定参数采用用户提供的数值：

- 手眼 `E_T_C`：
  `[[0.999934, 0.003458, -0.01094, 0.000049], [-0.003407, 0.999983, 0.004657, -0.087182], [0.010956, -0.00462, 0.999929, 0.072556], [0, 0, 0, 1]]`
- 相机内参 `K`：
  `[[3675.2707, 0, 1231.6813], [0, 3674.6618, 1073.9725], [0, 0, 1]]`
- 畸变 `D`：
  `[-0.11570, 0.32350, 0.00100, -0.00060, -2.9623]`
- 棋盘格：`board_grid = [39, 34]`，`square_size_mm = 0.5`，来自当前 `SystemConfig.py`。

## 软件业务逻辑分配

### 服务层

新增 `core/model_degradation_monitoring.py`，职责如下：

1. 读取监控配置：优先从 `config/model_monitoring.yaml` 读取手眼、相机、棋盘格和评估参数；缺省时使用本次给定默认值，并兼容旧版模型 YAML 内嵌字段。
2. 加载观测数据：
   - `.npz`：读取 `T_cam_board`、`filenames`、`reproj_mean_px`。
   - `.yaml/.yml/.json`：支持显式观测列表；若 JSON 只是摘要，则自动查找同目录下的 `camera_board_poses.npz` 或 `camera_board_poses_none.npz`。
   - 图片目录：按文件名排序读取图片，用 `core/vision/pnp.py` 的棋盘格 PnP 核心函数解算。
3. 对齐参考和当前观测：优先按文件名求交集，缺少文件名时按顺序截断到共同长度。
4. 计算末端漂移、李代数残差、RMS 位置/姿态漂移和可选 Mahalanobis 统计量。
5. 给出推荐模型状态更新：
   - 新的不确定性：`sqrt(old_uncertainty^2 + position_drift_rms_mm^2)`。
   - 新的置信度：按位置漂移相对阈值衰减，首版使用与现有 CV 置信度相似的比例型公式。
6. 用户确认后写回当前模型 YAML 的置信度、不确定性和最近一次评估记录，并在 `storage/model_versions/` 下备份旧文件。

### PnP 核心

`core/vision/pnp.py` 不再依赖外部 `controller.Transforms`。首版会将 `makeT`、`invT` 和 `rotvec2rot` 改为文件内实现，并把 `cv2` 依赖延迟到实际图像解算时检查。这样没有 OpenCV 的环境仍然可以运行非图像路径的单元测试，安装 `opencv-python` 后即可直接从图片目录解算 PnP。

### UI 层

在主界面顶部 `校验` 菜单中新增 `模型评估`：

1. 点击后打开非模态对话框。
2. 对话框提供参考观测和当前观测路径选择，支持 PnP 结果文件或图片目录。
3. 点击评估后在后台线程执行 PnP 与漂移计算，进度条显示当前处理到第几张图片或哪个 PnP 结果文件，主界面不被阻塞。
4. 点击写回模型前弹出确认框；确认后调用服务层备份并更新当前 YAML，然后刷新主界面的模型置信度和实时精度状态。

## 测试计划

1. 数学单元测试：构造已知 `Delta_T`，生成对应参考/当前 `C_T_B`，确认评估结果能恢复 1 mm 级平移和指定姿态漂移。
2. 持久化测试：对临时 `calibration_result.yaml` 执行写回，确认旧文件已备份、`confidence` 和 `metrics.position_uncertainty_rmse_mm` 被更新、`last_degradation_evaluation` 被记录，同时确认不会把手眼和相机内参写回模型结果文件。
3. UI 测试：确认 `校验` 菜单包含 `模型评估`，打开后出现对应对话框。
4. PnP 导入测试：确认 `core/vision/pnp.py` 不再依赖外部 `controller.Transforms`。
5. 图片验证：当前环境可用 OpenCV 时，对 `data/calibration/.../calibration_images` 运行小规模 PnP smoke；若当前 Python 缺少 `cv2`，记录为依赖未满足，不把数学和 YAML 测试与图像依赖绑定。

## 首版边界和待确认点

- 本次不做自动定时请求机器人运动或相机采图，只实现加载参考/当前观测并评估。
- 本次不反演和改写机器人 MD-H 误差参数；参数漂移估计留到拿到 `J_phi(q)` 和噪声协方差后再做。
- Mahalanobis `D^2` 需要健康阶段多轮基线残差均值和协方差；首版在无基线时只输出 RMS 漂移。
- 图片 PnP 的棋盘格规格按 `config/model_monitoring.yaml` 取 `[39, 34]` 和 `0.5 mm`，若未来标定板更换，应只更新该监控配置，不改代码。
