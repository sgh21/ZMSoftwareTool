# UR10 单板仿真与图像采集

当前交付目录为 `output/approved_30poses_three_inertias/`，B000～B008共九批4830张已完成并通过验收。各批沿用用户已批准的同一套30个目标位置与姿态。原B000～B002的1230张核验保留在 `validation.json`；B003～B008各600张的核验分别为 `validation_Bxxx.json`，汇总入口为 `validation_index.json`。补采保留原批次图片和记录。

旧小范围数据、旧命名备份、旧预览冗余文件和临时渲染结果约6.1 GiB已移入Windows回收站。必要的原30张图及参数已迁入正式数据 `_work/reference`，清理后重新核对全部图片、记录与参数路径有效。保留 `output/before_front_mount/` 和 `output/before_smaller_board/`，因为其中含独有未保存场景状态。

补充清理了旧点位生成/预览脚本、顶层单方向预览点表副本、缓存和已完成渲染日志。现有模块用于采集、追加批次、导出校验、运动学及场景/资产复现。正式数据 `_work/fixed_plan_audit.json` 记录此前准备任务与原任务1200条指令逐项一致的核对结果；那次核对只准备任务，没有重新拍图。

| 批次 | 目标偏移系数 | 惯性系数 | 图像数量 | 内容 |
| --- | ---: | ---: | ---: | --- |
| `B000` | 0.0 | 0.0 | 30 | 保留已批准预览的原始 PNG，每点一张 `D001` |
| `B001` | 0.0 | 0.1 | 600 | 30 点 × 20 方向，每方向一次到位拍照 |
| `B002` | 0.0 | 0.2 | 600 | 与 B001 相同的目标、方向和接近起点 |
| `B003` | 0.1 | 0.2 | 600 | 同一理想目标及方向；每点另有固定随机平移，距离不超过0.5 mm |
| `B004` | 0.01 | 0.03 | 600 | 固定偏移上限0.05 mm |
| `B005` | 0.02 | 0.05 | 600 | 固定偏移上限0.10 mm |
| `B006` | 0.03 | 0.10 | 600 | 固定偏移上限0.15 mm |
| `B007` | 0.05 | 0.13 | 600 | 固定偏移上限0.25 mm |
| `B008` | 0.07 | 0.17 | 600 | 固定偏移上限0.35 mm |

九组合计4830张，验收均通过。图片仅用 `B000_P001_D001.png` 形式命名：B为批次，P为固定目标位姿，D为接近方向，均为三位数字，不添加重复号或帧号。

## 读取数据

```text
output/approved_30poses_three_inertias/
├── parameters.json          相机参数、手眼、固定坐标关系及逐图外参
├── validation.json          原 B000～B002 的1230张检查结果
├── validation_B003.json     B003 的600张检查结果
├── validation_index.json    各次检查的索引与数量汇总
├── B000/                    惯性 0.0，30 张原始预览
├── B001/                    惯性 0.1，600 张
├── B002/                    惯性 0.2，600 张
├── B003/                    目标偏移 0.1、惯性 0.2，600 张
│   ├── record.json          精简位姿记录
│   └── calibration_images/
├── B004/ ... B008/          新增五组，各600张，同样的记录格式
└── _work/                   固定计划、场景快照、断点任务和详细检查日志
    └── reference/           已批准的 30 个目标及零惯性原图，用于后续复用
```

每组 `record.json` 包含 `batch`、`inertia`、`frames`；B003～B008另有顶层 `target_bias` 系数。每帧仍仅有 `image`、`ideal`、`actual`：

- `image`：相对于该批目录的图片路径，文件名已包含点号和方向号。
- `ideal`：原始理想末端位姿 `B_T_E`，4×4齐次矩阵，B003仍与原三批相同。
- `actual`：拍摄时的实际停止位姿 `B_T_E`，包含目标固定偏移与惯性偏移。

矩阵按四行排版，保留原始数值精度。平移统一为 **mm**；`E` 是 `tool0`，未附加 TCP 偏置。详细运动信息放在 `_work`，日常读取无需打开。

B000 每点只有一张原始零偏移图，可按 P 号作为参考；没有复制成 600 次拍摄。B001 与 B002 可按相同 P/D 逐项比较。20 个不同方向各到达一次，不等于同方向重复到达 20 次。公开新图片与 `_work` 的原始渲染文件使用 NTFS 硬链接，共用文件内容，修改任意一处会影响另一处。

## 重新采集与继续任务

### 2026-09-29 五组补采队列

五组均沿用原30点、每点20方向，各600张，现已全部通过验收。完成状态见正式目录 `_work/series_B004_B008/status.json`，各批结果见 `validation_Bxxx.json`。

| 批次 | 惯性系数 | 目标偏移系数 | 惯性偏移/mm | 目标偏移上限/mm |
| --- | ---: | ---: | ---: | ---: |
| B004 | 0.03 | 0.01 | 0.06 | 0.05 |
| B005 | 0.05 | 0.02 | 0.10 | 0.10 |
| B006 | 0.10 | 0.03 | 0.20 | 0.15 |
| B007 | 0.13 | 0.05 | 0.26 | 0.25 |
| B008 | 0.17 | 0.07 | 0.34 | 0.35 |

复用B003已保存的30个随机偏移向量，按 `新系数 / 0.1` 缩放，不重新随机方向或相对距离。每点20个方向共用该点缩放后的向量，原理想位姿、20方向及接近距离保持不变。原始参数在 `capture_series.json`，本次冻结配置为队列内 `queue.json`；开始采集后以冻结配置为准。

`collect_series.py` 依次调用已有采集和追加验收入口：一组通过后继续下一组，失败则停止并保存错误；完成组恢复时跳过，未完成组从原日志补采。本次已全部通过，正式目录合计4830张。公开参数、精简记录及图片格式与B003一致，不复制B000作为新采样。

```powershell
# 仅首次准备；本次已经完成，不重复执行
python -B collect_series.py prepare --config capture_series.json
# 首次运行或中断后恢复同一队列；运行中的队列有进程锁，不会重复启动
python -B collect_series.py run --queue output/approved_30poses_three_inertias/_work/series_B004_B008/queue.json
# 读取最近一次保存的状态，不触发重新采集
python -B collect_series.py status --queue output/approved_30poses_three_inertias/_work/series_B004_B008/queue.json
```

后台每30分钟读取一次进度；开始、阶段切换及完成也保存状态，详细时间点见队列目录 `progress.jsonl`。采集器正常逐张写入自身断点进度，这与监控轮询频率不同。验收成功后的冗余渲染日志移入Windows回收站，保留固定偏移、任务、详细记录和验收明细。队列不会连接或改写主应用的观测、基准、阈值与历史。

本次新增3000张全部识别31个标记和48个角点，原理想位姿差为0，五组白板最小余量分别为91.081、90.544、89.383、88.516、87.439 px。与B002对比并扣除惯性变化后，目标偏移向量最大数值差0.000584 mm；这是仿真浮点一致性检查，不是图像测量精度。原记录、固定表及原外参已核对保留，五组冗余渲染日志均已移入回收站；没有留下临时测试脚本。

### 原三组及单批采集

在本目录运行，使用已有 `ZMSoftware` 环境。推荐入口直接读取根目录 `fixed_observation_plan.json` 的30个固定目标及每点20个固定方向，并从当前交付的 `_work/reference/` 复制零惯性原图，再采集惯性 0.1 和 0.2。固定表缺失时直接报错，不重新生成点位或方向。

```powershell
conda activate ZMSoftware
# 新输出目录必须尚不存在
python collect_approved.py --dataset output/new_name
# 中断后继续原任务，不改变目标、方向或惯性
python collect_approved.py --dataset output/new_name --resume
```

`--prepare-only` 仅准备任务。采集使用 Blender 后台进程，不另开可见窗口；完成后自动调用 `export_grouped.py` 整理三批并逐图验证。单独重新验证：

```powershell
python export_grouped.py validate --dataset output/new_name
```

需要其他惯性系数时，使用通用入口并明确指定已冻结的计划和场景：

```powershell
python collect_images.py --dataset output/inertia_0.3 --inertia 0.3 --points 30 --directions 20 --samples 32 --plan output/approved_30poses_three_inertias/_work/fixed_observation_plan.json --scene-source output/approved_30poses_three_inertias/_work/scene_snapshot.blend
```

通用入口保留详细运行记录。`collect_approved.py` / `export_grouped.py` 只管理原三组协议；已有额外批次时，旧导出入口会拒绝重写公共参数，应使用 `append_batch.py`。

追加本次B003使用以下命令。首次准备随机偏移一次；已有任务只用 `--resume`，不重新抽样：

```powershell
python collect_images.py --dataset output/approved_30poses_three_inertias/_work/B003 --inertia 0.2 --target-bias 0.1 --bias-seed 20260928 --batch-start 3 --points 30 --directions 20 --samples 32 --plan output/approved_30poses_three_inertias/fixed_observation_plan.json --scene-source output/approved_30poses_three_inertias/_work/scene_snapshot.blend
# 已有任务时，用这一条继续
python collect_images.py --dataset output/approved_30poses_three_inertias/_work/B003 --resume
# 完成渲染后追加公开图片、精简记录和外参，并核验600张
python append_batch.py --dataset output/approved_30poses_three_inertias --source output/approved_30poses_three_inertias/_work/B003
```

只重新核验B003时加动作 `validate`；追加导出会复用已有硬链接，不修改旧三组文件。B003的断点任务和完整记录在 `_work/B003`。

## 固定目标与停止模型

已批准 30 个目标的末端基座系 XYZ 范围约为 **331～1142、−217～854、405～771 mm**；光轴相对板法线倾角约 **4.1°～50.1°**，光心距板中心约 **472～717 mm**。这是离散样本的范围，不表示范围内任意位姿都可达或入镜。程序复用实际点表，不重新随机这些目标。

每点 `D001` 保留原预览的光轴接近方向；其余19个方向只在首次准备时按固定球面分布确定，已转换到基座系保存，`D020` 为反向。以后批次直接读取这些数值。一般从目标前5 mm开始接近；P027/D005使用2.5 mm以避开逆运动学边界，目标和方向保持不变，路径采样间隔为0.5 mm。

固定表的 `points` 中，`point_id` 是点号，`B_T_E_ideal_mm` 是完整目标位置和姿态，`q_ideal_rad` 是对应六关节角；每点 `directions` 中的 `direction_id` 是方向编号，`direction_B` 是基座系三维单位向量，`distance_mm` 是接近距离。相同P/D在不同批次保持一致；不同P点各有自己的20个方向，不要求它们的基座向量相同。

每次采集在数据目录保存同名固定表，并在当次 `capture_job.json` 冻结实际使用值。`--resume` 只读取原任务。B001/B002仅改变惯性系数；B003在原理想目标上增加每点固定随机平移，接近起点也随之平移。所有批次的原目标位姿、20个方向和接近距离不变。

```text
p_target = p_ideal + point_bias_B
p_start  = p_target − approach_distance × unit_direction_B
p_stop   = p_target + (2 mm × inertia) × unit_direction_B
R_stop  = R_ideal
```

惯性系数在 `[0,1]`：0 对应零惯性偏移，0.1/0.2 对应沿接近方向偏移 0.2/0.4 mm，最大 2 mm；目标点偏移单独计算。新图片逐次执行离开、接近、停止和独立渲染。采集为 3072×2048、RGB 8 位原生 PNG，Cycles OptiX 32 samples，使用 OptiX 降噪。

目标偏移系数也在 `[0,1]`，最大距离为 `5 × 系数 mm`。B003的每个点独立抽取球面均匀方向和 `[0,0.5] mm` 上均匀分布的距离；不是固定0.5 mm，也不是球体体积均匀采样。随机种子为20260928，30个基座系偏移向量保存后固定；同一点的20个方向全部复用同一个向量。公共 `parameters.json → runs` 的B003条目中，`target_bias.offsets_B_mm` 按P号给出实际向量，原始副本为 `_work/B003/target_bias.json`。补采以已保存向量和任务为准，种子只用于追溯。

B003与B002惯性均为0.2；同一P/D的实际位置差应等于该P的固定偏移（允许仿真浮点误差）。`actual−ideal` 是两种偏移的向量和，大小可能相互抵消，不能将其直接等同于0.4 mm或0.5 mm。

检查内容包括文件与记录对应、固定目标/方向/接近起点、停止偏移、坐标变换链、31 个标记和 48 个 ChArUco 角点，以及完整白板至少 60 px 的画面余量。原三批独立 PnP 结果在 `_work/per_image_validation.jsonl`，B003在 `_work/B003/per_image_validation.jsonl`，均与仿真真值分开。

## 相机参数与坐标

`parameters.json` 使用 `A_T_B` 表示把 B 系坐标转换到 A 系。内参、畸变和手眼固定；相机随末端运动，外参按图片保存。

| 坐标系 | 定义 |
| --- | --- |
| `W` | Blender 世界坐标系 |
| `B` | UR 控制器基座系；与 ROS `base_link` 相差绕 Z 轴 180° |
| `E` | UR `tool0` |
| `C` | OpenCV 相机系：X 向右、Y 向下、Z 向前 |
| `M` | ChArUco 图案左上角：X 向右、Y 向下、Z 指向板内 |
| `target_center` | 印刷图案表面中心，轴方向与 M 相同 |

相机分辨率 3072×2048、像元 2.4 µm、12 mm 镜头，对应有效传感器 7.3728×4.9152 mm，`fx=fy=5000 px`、`cx=1536`、`cy=1024`，畸变 `[k1,k2,p1,p2,k3]=[0,0,0,0,0]`。

手眼 `E_T_C` 的旋转为 `diag(-1,-1,1)`、平移约 `(-75,0,70) mm`；靶心 `B_T_target_center` 的旋转为 `diag(-1,1,-1)`、平移 `(960,370,26.1) mm`。从靶心到图案左上角的平移为 `(-90,-70,0) mm`。每图满足：

```text
B_T_E_actual × E_T_C × C_T_M = B_T_M
```

这些是仿真真值，不是实机标定结果。

## 场景查看与构建

打开 `output/ur10_charuco_scene.blend`。调试只保留 **一个可见 Blender 窗口**；文件在后台重建后，应在原窗口重新载入。文件中的三个 Scene 共享同一套物体：总览、3072×2048 末端相机、法兰安装近景，始终只有一块实体标定板。按小键盘 0 查看所选相机，F12 渲染。

安装结构为圆形过渡法兰加一体 L 形板；整个相机安装组件绕 `tool0` 本地 Z 轴转 180°。UR10 关节为磨砂暗灰，连杆银色、端盖蓝色。初始六关节角保存在 `scene_config.json`，构建时不通过抬高机器人扩大视野；采集过程中才运动到固定目标位姿。

单板图案为 180×140 mm，白板 200×160×5 mm，背板 206×166×12 mm，底座 215×175×27 mm；板中心世界坐标为 `(-430,-120,851) mm`，印刷面另高 0.1 mm。静态初始视图中图案面积约占画面 32%、完整白板约占 40.6%；采集位姿改变后比例会变化。

场景配置集中在 `scene_config.json`。已有资产可直接使用；需要重新构建时：

```powershell
python prepare_robot_assets.py
python prepare_charuco.py
& 'D:/Softwares/Blender/blender.exe' --background --factory-startup --python build_scene.py
python verify_view.py --annotate
```

`build_scene.py -- --build-only` 只构建，`-- --preview` 使用低采样预览。静态场景检查在 `output/geometry.json`、`output/validation.json`；正式查看图片为 `overview.png`、`wrist_camera.png`、`mount_detail.png`。修改前的未保存状态备份位于 `output/before_front_mount/` 和 `output/before_smaller_board/`。

## 模型与图案来源

| 对象 | 来源与使用方式 |
| --- | --- |
| UR10 CB3 | [Universal Robots 官方仓库固定提交 89bbe795f38a7ab00fb66fe8831dfff79dc99edf](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/89bbe795f38a7ab00fb66fe8831dfff79dc99edf)，七个官方网格及名义运动学，BSD-3-Clause，文件与许可在 `assets/ur10/` |
| MV-CS060-10GC | 用户提供可能型号；机身 29×29×42 mm，依据[海康原厂规格书，由 EQ Vision 转载](https://eq-vision.de/wp-content/uploads/2026/06/MV-CS060-10GC-datasheet.pdf)简化建模，非官方 CAD |
| 12 mm 镜头 | 焦距由用户提供；外形参考[海康 MVL-MF1228M-8MP 规格书](https://www.hikrobotics.com/en2/source/vision/document/2023/12/7/MVL-MF1228M-8MP_datasheet_20221024.pdf)，实际镜头型号待确认 |
| ChArUco | OpenCV 4.13 `CharucoBoard.generateImage` 生成，9×7 格、方格 20 mm、标记 14 mm、`DICT_5X5_1000`、ID 0–30，纹理 3600×2800；[OpenCV 官方说明](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html) |

UR10 资产已只读对照用户 `SoftwareTools/storage/model_assets/` 的模型；未修改原模型。法兰、安装板及工装为仿真设计，孔距和结构强度未经实物加工图确认。

## 使用边界

当前使用理想针孔、零畸变，未模拟有限景深、传感器噪声、真实相机响应或动态卷帘；曝光 −1 EV 是渲染显示设置。停止偏移是指定的几何模型，没有求解质量、速度、制动控制器等动力学。运动时检查连杆与工作台的包围盒间隙，未实现完整自碰撞检测。

图像检测与 PnP 检查用于验证仿真数据，不能当作实机测量精度。这里不连接真实设备，也未修改主应用的 ChArUco 导入接口。各批次已确认结果见本说明中的 `validation_index.json` 与逐批验收文件；其他维护验收记录见 [维护记录](../../docs/validation_history.md)。
