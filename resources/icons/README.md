# 图标资源

图标按用途分类。精度图标采用机械设备轮廓，避免用通用刷新、上下箭头表达具体业务；均为透明背景、单色线条，统一由程序着色。

| 目录 | 文件 | 用途 |
| --- | --- | --- |
| `branding/` | `reference-interface.png` | 用户参考截图，顶部显示其中的品牌区域 |
| `branding/` | `application.svg`（scan-line） | 任务栏占位图标 |
| `navigation/` | `precision-monitor.svg` | 参考图风格的面板菜单图标 |
| `navigation/` | `back.svg` | 返回上一精度选项卡 |
| `precision/` | `robot-position.svg` | 机械臂与末端工具 |
| `precision/` | `spindle-rotation.svg` | 主轴、钻头和回转箭头 |
| `precision/` | `axial-feed.svg` | 主轴、工件和轴向进给箭头 |
| `window/` | `minimize.svg`、`maximize.svg`、`restore.svg`、`close.svg` | 标题栏窗口控制 |
| `common/` | `pending.svg`（panel-top） | 待开发提示 |
| `common/` | `chevron-down.svg` | 下拉选择箭头，本工程绘制 |
| `robot/` | `camera.svg` | 机器人页标志点观测图像占位，相机与靶心线条图标，本工程绘制 |

替换时保持文件名即可，无需修改页面代码。新增按钮图标按业务用途放进相应目录，跨页面通用操作放在 `common/`。

`app/resources.py` 统一读取并着色，精度选项卡选中时为绿色，未选中为深灰色；侧栏和窗口控制为深色。颜色和字体配置见 `config/display.json`。

顶部标识来自用户提供的 `微信图片_20260710114555_730_46.png`（2400 × 1350）。`reference-interface.png` 是原图副本，界面只显示 `branding.source_rect` 指定的区域，不重新绘制其中的标识与文字。图片路径、显示区域和尺寸在 `config/display.json` 中调整，后续可换成正式软件标识。

`application.svg`、`pending.svg` 取自 [Lucide 源码](https://github.com/lucide-icons/lucide/tree/f06ac67e33d645c40b8ce19a0419c85c5d7dd751/icons)，对应 scan-line、panel-top。完整许可见本目录 `LICENSE.txt`。菜单、精度和窗口控制 SVG 为本工程绘制。之后增加外部资源时同样保留来源和许可。
