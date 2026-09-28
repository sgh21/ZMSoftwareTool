# 机器人末端定位监控

本模块通过固定靶标的视觉观测比较初始与复测时期。算法位于 `core/algorithms/`，参数、基准和历史由 `PositionMonitoringService` 管理，机器人页面通过服务调用；旧数据适配单独放在 `experiments/`。公式与指标依据见根目录 `AGENTS.md`。

## 使用流程

1. 在机器人页“加载参数”或“参数设置”中配置相机、棋盘和相机到被监测末端的手眼变换。末端应为实际评价的 TCP；相机到法兰的变换须先合入工具变换。
2. 导入初始观测清单，点击“建立基准”。清单按测点、接近方向和每次重新到达组织；同一次停留连拍不能填成多次到达。
3. 导入同程序、同靶标、同测点和同方向的复测清单，点击“评估精度”。图像解算在后台执行。
4. 页面标题右侧的下拉菜单仅有“绝对定位精度退化”“重复定位精度退化”“当前重复定位精度”，默认第三项。切换后卡片、趋势、逐点表、判定及报警统一显示该指标；原始观测图像及右侧参数不变。四列为 X、Y、Z、空间指标，单位 mm。
5. “阈值设置”分别保存三项指标的 X、Y、Z、空间上限，逐测点、接近方向判定。留空不判定；退化量直接与上限比较，负值表示改善，不取绝对值报警。旧定位漂移阈值不会转用。

未建立基准，或现有基准与本次参数、程序、靶标、点位组合不匹配时，也可选择“当前重复定位精度”评估并保存单期结果；两项退化保持空白并提示需要可比较的基准。基准可比时一次评估计算三项，切换指标直接查看。单期评估的 Q 对应该批次每点第一条观测，不能借用其他批次的参考朝向。切换指标本身不重新测量或保存评估记录。

机器人运动与实际相机采集仍由设备软件负责；“采集靶标”不连接硬件。加工点位预测模型尚未实现，不能把点位表当作预测结果。

## 参数 JSON

参数字段可在 `config/robot_position.json` 查看；界面保存后写入独立版本，不覆盖原始输入文件。

| 字段 | 含义 |
| --- | --- |
| `camera_matrix` | 3×3 相机内参，焦距、主点单位为像素；输入图像时必须提供 |
| `dist_coeffs` | OpenCV 畸变系数，数量为 0、4、5、8、12 或 14 |
| `board_grid` | `[内角点列数, 内角点行数]`，不是方格数 |
| `square_size_mm` | 棋盘相邻角点距离，始终以 mm 表示 |
| `hand_eye` | 4×4 `E_T_C`，将相机坐标变换到被监测末端；建立基准前必须提供 |
| `length_unit` | 手眼平移的输入单位：`mm` 或 `m`；内部统一换算为 mm |
| `transform_convention` | 固定为 `E_T_C` |
| `reference_rotation` | 可选 3×3 棋盘视觉参考旋转，用于约束对称棋盘的 180°角点顺序；不是基座朝向 Q |
| `max_reprojection_error_px` | 可选重投影 RMS 上限；`null` 表示未设置，不预置工程阈值 |
| `version` | 服务保存时生成新版本；输入文件中的旧版本号不会覆盖此机制 |

直接输入已解算的 `vision_pose` 时不要求相机内参。图像路径输入采用居中棋盘模型、亚像素角点、IPPE 与 LM 优化，输出完整 `C_T_M`。复测优先使用已选基准该测点的参考旋转对齐角点顺序；普通对称棋盘没有天然物理角点编号，大幅改变观测方向时须确认对应关系。

服务与主页面均支持读取 JSON/YAML。旧手眼结果中的 `T_tool_cam` 可导入，服务按旧格式的米转换为毫米。

## 观测清单

下面是结构模板，路径须替换为真实的独立到达图像。空的 `base_rotations` 与 `initial_errors` 表示尚未提供相应依据。

```json
{
  "schema_version": 1,
  "batch_id": "initial_batch_01",
  "label": "初始测量",
  "program_id": "inspection_program_v1",
  "target_id": "fixed_board_01",
  "length_unit": "mm",
  "orientation_source": "unspecified",
  "base_rotations": {},
  "initial_errors": {},
  "samples": [
    {
      "point_id": "P01",
      "direction_id": "approach_01",
      "sample_id": "arrival_001",
      "image_path": "images/arrival_001.jpg"
    },
    {
      "point_id": "P01",
      "direction_id": "approach_01",
      "sample_id": "arrival_002",
      "image_path": "images/arrival_002.jpg"
    }
  ]
}
```

`program_id` 与 `target_id` 跨期必须相同，测点和方向组合也必须相同。程序编号本身不能证明工况可比，需由采集流程保证负载、速度、热状态、靶标固定关系等条件。

每条样本必须有 `point_id、direction_id、sample_id`；三者组成的编号在同一批次内不能重复。`direction_id` 表示接近方向，与基座 XYZ 分量不同。

样本可提供以下数据：

| 字段 | 含义 |
| --- | --- |
| `image_path` | 绝对路径，或相对清单所在目录的路径 |
| `vision_pose` | 可替代图像的 4×4 `C_T_M`；同时存在时使用该矩阵，不重算图像 |
| `robot_pose` | 4×4 `B_T_E`，仅手眼标定必须；视觉定位监控不要求机器人运动学或逐条机器人位姿 |

矩阵平移与 `initial_errors` 共用清单的 `length_unit`；服务接受 `mm/m`，内部保存为 mm。不要对已转换的矩阵再次声明为米。

`base_rotations` 的结构为 `{point_id: Q_i}`，每个 Q 为 3×3 `B_R_Er`，必须对应**该点初始清单中第一条观测的末端朝向**，不按方向单独更换。服务使用基准里的 Q，不用复测朝向替换它。若来自控制器估计，应设置 `orientation_source: "controller_approximation"` 并保留相应说明。

`initial_errors` 的结构为 `{point_id: {direction_id: [ex, ey, ez]}}`，表示初始到达重心相对指令位置的有符号误差向量，沿基座三轴表达。它必须来自有依据的初始精度测量，不能用零或视觉基准自身代替。

## 指标边界与追溯

- 缺 Q：保留固定参考末端系结果与空间漂移距离；基座 XYZ 留空。Q 只能转换坐标方向，不能提供绝对误差。
- 缺初始误差向量：仍可计算定位漂移；当前绝对 AP 及其退化量留空。空间漂移距离不是 AP 增量。
- 同点同方向每期少于 2 次重新到达：RP 留空。少于 30 次提示样本不足；达到 30 次也不代表完成全部国标试验条件。
- 空间 RP 使用 `平均到重心距离 + 3 × 距离样本标准差`；XYZ 显示轴向 `3σ` 半宽，明确属于补充口径。不同接近方向的重心差单列，不混入同方向 RP。
- 参数变化后需重新导入观测并建立匹配基准，或“选择基准”恢复原参数。不同参数版本不直接混算。

参数版本、基准、手眼结果、观测快照和评估历史保存于 `storage/position_monitoring/`。阈值与加工点位设置同时持久化；原始输入文件不修改。三项阈值保存于 `metric_thresholds`，键为 `absolute_change`、`repeatability_change`、`repeatability`，各含 `X/Y/Z/distance`，空值为 `null`。历史保留当时阈值，旧记录没有这些阈值时显示未设置，不用当前配置补判历史。趋势核对基准、参数版本、程序、靶标及测点/方向组合。

## 旧数据调试

界面“调试”中依次执行：生成旧数据输入 → 载入视觉参数 → 载入并标定手眼 → 查看固定靶标一致性残差 → 应用标定手眼 → 建立演示基准 → 运行演示复测。PARK、TSAI 为当前支持的方法；残差反映手眼数据的一致性，不是机器人绝对定位误差。

也可从项目根目录运行独立输入生成入口：

```powershell
python experiments/20260928_robot_position/run_debug.py --data-root data/robot_error --output-dir data/processed/robot_position_debug
```

默认只生成参数与三份观测清单。加 `--run` 可复用服务完成手眼标定、建立基准和复测评估；`--method` 选择 PARK（默认）或 TSAI：

```powershell
python experiments/20260928_robot_position/run_debug.py --output-dir data/processed/robot_position_debug_run --run --method PARK
```

运行时在输出目录的 `session/` 下建立独立服务状态，不使用主应用的参数、基准和历史；完整报告保存为 `debug_result.json`。终端仅显示调试警告、手眼一致性残差、状态和指标摘要。此入口会处理真实图像，耗时取决于图像数量与尺寸。

`calib_data20` 用于手眼标定，`camera_pos_normal50` 与 `camera_pos_x05y-05_50` 仅各取 `calib_00` 作流程演示；其余同名图像的机器人记录位姿不同，不能直接配成跨期监控点。

调试清单从 `robot_poses.npz` 读取控制器记录，将米转换为毫米；不混用旧 PnP 缓存或激光替换位姿。基准 Q 是控制器朝向近似。手眼清单标记 `comparison_status: "calibration_only"`，服务禁止将其直接用于监控；演示两期标记 `"debug_unverified"`，结果明确提示采样对应待确认。

仅凭现有文件不能确认 `calib_00` 的同指令位姿、接近方向及安装稳定性，因此演示结果不能认定为真实精度退化；RP 与缺乏初始误差依据的 AP 应留空。产物必须位于原始 `robot_error` 目录之外，原始图像、位姿与缓存保持不动。
