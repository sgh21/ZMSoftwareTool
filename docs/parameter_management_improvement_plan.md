# 参数管理系统改进建议

## 任务要求

当前参数文件以直接覆盖为主，`config/nominal_robot.yaml`、`config/calibration_result.yaml`、`config/model_monitoring.yaml` 和 SQLite 历史记录之间职责边界仍不够清晰。随着名义参数、控制器参数、辨识模型和相机监控参数都开始参与精度评估，继续覆盖单个 YAML 会带来三个问题：

1. 参数不可追溯：无法可靠回答“某次报告、某次退化评估、某次标定分析到底使用了哪一版参数”。
2. 参数生命周期混杂：机器人模型版本、示教器控制参数、相机内外参和评估结果被放入同一类文件，更新频率和责任主体不同，容易互相覆盖。
3. 参数语义不稳定：辨识算法内部以名义参数为基准优化误差值，但保存时若仍以误差增量为主，会增加前向运动学和历史对比的复杂度。

本次革新的目标是建立一个本地单机可用、可回溯、可比较、可回退的参数版本管理机制。它不必拘泥于 YAML，但应保留 YAML/JSON 这类可读文件作为参数快照或导出格式，并用 SQLite 维护索引、版本关系和审计信息。

### 参数分类

| 参数类型 | 定义 | 必含字段 | 更新来源 | 使用方式 |
| --- | --- | --- | --- | --- |
| 名义参数 `nominal_robot` | 机器人设计模型参数，也是辨识优化的基础基准 | 基坐标变换、靶球偏置、MD-H | 设计值、人工维护、历史名义模型回退 | 作为辨识初始模型和名义 FK 基准 |
| 控制参数 `controller_robot` | 示教器/控制器中实际配置的参数，用于和辨识模型对比 | MD-H、工具坐标系参数 | 示教器导出或人工录入 | 对比控制器模型与辨识模型，辅助判断软件模型是否需要同步 |
| 辨识参数 `identified_model` | 辨识算法输出的最终机器人模型 | 基坐标变换、MD-H、靶球偏置、工具坐标系、置信度、定位不确定性 | S1/后续辨识算法 | 直接用于前向运动学和实时精度评估 |
| 相机监控参数 `camera_monitoring` | 长期监控链路所需的相机和棋盘配置 | 手眼、相机内参、畸变、棋盘格尺寸和布局、角点排序策略 | 手眼标定、相机标定、监控配置更新 | PnP 解算、退化评估、长期监控报告 |

### 关键语义约束

- 名义参数是基准，不等同于控制器参数。控制器参数来自示教器，是需要被导入、比较和记录的外部现实配置。
- 辨识算法内部可以继续优化 `delta`，但持久化时必须保存最终绝对值模型，例如完整 MD-H、完整基坐标变换、完整工具/靶球偏置。`delta` 只作为算法过程元数据保留。
- 相机参数不属于机器人误差模型，不应随 `calibration_result.yaml` 被模型标定分析覆盖。
- 任何参数更新都不覆盖历史版本；更新动作创建新版本，回退动作只切换 active 指针。
- 每次评估、标定、报告都应记录一个“参数包”引用，说明当时使用的名义参数、控制参数、辨识模型和相机参数版本。

### 成熟方案参考

本项目可以借鉴以下成熟做法，但不需要引入复杂外部系统：

1. Git 式不可变对象：每个版本都是内容快照，用 `hash` 校验内容完整性；更新不是改旧对象，而是创建新对象。
2. MLflow Model Registry 式阶段管理：模型版本有 `draft/candidate/active/archived` 状态，只有被提升到 `active` 的版本参与运行。
3. 数据库审计日志：用 SQLite 记录谁在何时、基于什么来源、为什么创建了新版本，支持列表、查询、回滚和报告引用。
4. ROS 参数命名空间思想：不同参数域必须命名空间隔离，例如 `/robot/nominal`、`/robot/controller`、`/robot/identified`、`/sensor/camera_monitoring`，避免同名字段混用。
5. Schema migration：参数结构必须带 `schema_version`，未来字段变动通过迁移函数升级旧版本，而不是让业务代码到处兼容碎片格式。

## 改进进化

### 目标架构

建议新增一个参数仓库层 `ParameterRepository`，统一处理参数读写、版本创建、active 指针、校验和回退。底层采用“SQLite 元数据 + 文件快照”的混合方式。

建议目录：

```text
storage/
├── parameters/
│   ├── parameter_registry.sqlite
│   └── artifacts/
│       ├── nominal_robot/
│       ├── controller_robot/
│       ├── identified_model/
│       ├── camera_monitoring/
│       └── parameter_bundle/
config/
└── active_parameters.yaml
```

`storage/parameters/artifacts/` 保存不可变参数快照，格式可以先用 YAML，后续需要更严格校验时可以切到 JSON + JSON Schema。`parameter_registry.sqlite` 保存版本索引和审计记录。`config/active_parameters.yaml` 只保存当前生效版本指针，不保存完整参数正文。

示例 active profile：

```yaml
active_profile:
  name: default
  nominal_robot: nominal_robot:ur10_design:v0003
  controller_robot: controller_robot:ur10_teach_pendant:v0002
  identified_model: identified_model:s1_20260612:v0001
  camera_monitoring: camera_monitoring:fixed_board_c1:v0002
```

示例参数快照元数据：

```yaml
schema_version: 1
kind: identified_model
parameter_id: s1_20260612
version: v0001
created_at: "2026-06-12T15:30:00+08:00"
created_by: local_user
parent:
  nominal_robot: nominal_robot:ur10_design:v0003
  calibration_dataset: data/calibration/...
status: candidate
content_hash: sha256:...
payload:
  confidence: 97.16
  position_uncertainty_rmse_mm: 0.0
  base_transform: ...
  mdh: ...
  target_ball_offset: ...
  tool_frame: ...
  algorithm_metadata:
    method: S1
    optimized_delta: ...
```

### 数据库表建议

| 表 | 职责 |
| --- | --- |
| `parameter_sets` | 每个逻辑参数对象，例如 UR10 名义模型、示教器参数、某套相机监控配置 |
| `parameter_versions` | 每个不可变版本的路径、hash、schema、状态、父版本和创建信息 |
| `active_profiles` | 当前激活的参数组合，可以支持多个 profile |
| `parameter_bundles` | 一次运行或报告使用的完整参数版本集合 |
| `migration_history` | 记录旧 schema 升级到新 schema 的过程 |
| `audit_events` | 创建、激活、归档、回退、导入、导出等操作日志 |

### 版本状态机

```text
draft -> candidate -> active -> archived
                 \-> rejected
```

- `draft`：刚导入或正在编辑，还不能用于正式评估。
- `candidate`：通过 schema 校验和基本 FK/PnP sanity check，可人工选择激活。
- `active`：当前生效版本。每类参数在一个 profile 中只允许一个 active。
- `archived`：历史版本，仍可读取、比较和回退。
- `rejected`：校验失败或人工判定不可用，保留原因但不参与运行。

### 推荐演进路线

#### 阶段 0：统一语义和 schema

建立四类参数的字段规范和命名规范：

- `nominal_robot`：基坐标变换、靶球偏置、MD-H。
- `controller_robot`：MD-H、工具坐标系。
- `identified_model`：最终绝对值模型、置信度、不确定性、算法过程元数据。
- `camera_monitoring`：手眼、相机内参、畸变、棋盘格、角点顺序策略。

这一阶段只写文档和 schema，不改变现有 UI。目标是消除“工具坐标系、靶球偏置、末端偏置、辨识误差值、最终模型值”的混用。

#### 阶段 1：引入只追加的参数仓库

新增 `core/parameter_repository.py`，实现：

- `create_version(kind, payload, metadata)`：创建不可变版本。
- `load_version(ref)`：按版本引用读取。
- `activate(profile, kind, ref)`：切换当前生效版本。
- `list_versions(kind)`：列出历史版本。
- `compare_versions(ref_a, ref_b)`：生成结构化差异。

同时将现有文件导入为初始版本：

- `config/nominal_robot.yaml` -> `nominal_robot:v0001`
- `config/calibration_result.yaml` -> `identified_model:v0001`
- `config/model_monitoring.yaml` -> `camera_monitoring:v0001`
- 控制器参数如果暂缺，则创建空占位或等待人工导入。

#### 阶段 2：业务服务改读 active profile

让现有服务层从 `config/active_parameters.yaml` 解析当前参数版本：

- `CalibrationService` 读取 active `nominal_robot` 和 `identified_model`。
- `ModelDegradationMonitoringService` 读取 active `camera_monitoring`。
- 标定分析保存结果时不再覆盖 `config/calibration_result.yaml`，而是创建新的 `identified_model` 版本。
- 名义参数更新不再覆盖 `config/nominal_robot.yaml`，而是创建新的 `nominal_robot` 版本。

为了平滑迁移，旧 YAML 仍作为 fallback 读取，但所有新写入都进入参数仓库。

#### 阶段 3：增加 UI 参数版本管理

新增“参数管理”页面或对话框，至少包含：

- 当前 active profile 概览。
- 四类参数的版本列表。
- 参数版本详情和差异对比。
- 激活、归档、回退按钮。
- 从示教器导入控制参数。
- 从现有 YAML 导入历史参数。
- 生成参数包报告。

其中“回退”不复制旧文件覆盖当前文件，而是把 active 指针切回历史版本。

#### 阶段 4：绑定报告和实验可追溯性

所有输出报告都记录 `parameter_bundle`：

```yaml
parameter_bundle:
  nominal_robot: nominal_robot:ur10_design:v0003
  controller_robot: controller_robot:ur10_teach_pendant:v0002
  identified_model: identified_model:s1_20260612:v0001
  camera_monitoring: camera_monitoring:fixed_board_c1:v0002
```

标定报告、健康报告、退化监控报告都应能从这个 bundle 回放当时的参数状态。这样即使 active 版本已经更新，历史报告仍可复现。

#### 阶段 5：增强校验和迁移能力

为每类参数增加校验：

- MD-H 数组长度、单位、数值范围。
- 齐次变换是否为 4x4，旋转矩阵是否正交。
- 相机内参矩阵形状和畸变向量长度。
- 棋盘格尺寸、角点排序策略。
- `identified_model` 的 nominal parent 是否存在。
- `controller_robot` 与 `identified_model` 的差异是否超过人工确认阈值。

每次 schema 变更时提供迁移函数，例如 `schema_version 1 -> 2`。迁移也创建新版本，不原地修改旧版本。

### 与当前文件的关系

短期保留：

- `config/nominal_robot.yaml`：作为初始导入和旧代码 fallback。
- `config/calibration_result.yaml`：作为旧代码 fallback 和导出兼容文件。
- `config/model_monitoring.yaml`：作为相机监控参数的初始导入和 fallback。

中期调整：

- `config/*.yaml` 不再作为主存储，只保存 active 指针或导出快照。
- `storage/model_versions/` 逐步迁移到 `storage/parameters/artifacts/`。
- `storage/records/identification_history.sqlite` 可以保留，但应与 `parameter_registry.sqlite` 建立 `identified_model_version_ref` 关联。

长期目标：

- 所有参数更新都走 `ParameterRepository`。
- 所有业务计算都由 active profile 或显式 parameter bundle 驱动。
- 不再存在“覆盖当前 YAML 导致历史丢失”的路径。

### 优先落地建议

最小可行改造顺序：

1. 新增 `config/active_parameters.yaml` 和 `storage/parameters/parameter_registry.sqlite`。
2. 写一次迁移脚本，把当前三份 YAML 导入为 v0001。
3. 改 `calibration_persistence.save_identification_result()`，让它创建 `identified_model` 新版本，同时可选导出到旧 YAML。
4. 改 `NominalParameterService.update_direct()`，让它创建 `nominal_robot` 新版本。
5. 改模型评估写回，让它创建新的 `identified_model` 状态版本，记录更新后的置信度和不确定性。
6. 最后补 UI 版本管理和参数对比。

这个路线可以逐步兼容现有代码，不需要一次性重写桌面软件。
