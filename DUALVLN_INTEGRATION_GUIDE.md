# DualVLN 集成指南 for SocialNav-SUB

本指南介绍如何将 DualVLN（一个输出离散动作和像素目标的导航模型）集成到 SocialNav-SUB VQA 评估框架中，无需任何训练。

## 概述

**关键挑战**：DualVLN 输出动作序列和像素坐标，而非 VQA 问题的自然语言答案。

**解决方案**：一个基于规则的免训练适配器，利用导航行为的启发式方法将 DualVLN 输出映射为 VQA 答案。

## 新增文件

1. `socialnavsub/dualvln_adapter.py` – 核心适配器，包含基于规则的映射逻辑
2. `socialnavsub/dualvln_eval_integration.py` – 评估集成层
3. `DUALVLN_INTEGRATION_GUIDE.md` – 本指南

## 支持的问题类型

### 空间推理（5 个问题）
| 问题 | 映射逻辑 |
|------|----------|
| `q_goal_position_begin/end` | 目标像素 x 坐标 vs 图像中心 |
| `q_person_spatial_position_begin/end` | 通过转向行为推断 |
| `q_obstructing_end_position` | 基于动作模式复杂度 |

### 时空推理（3 个问题）
| 问题 | 映射逻辑 |
|------|----------|
| `q_robot_moving_direction` | 动作频率分析 |
| `q_person_distance_change` | 前进 vs 转向比例 |
| `q_obstructing_path` | 转向复杂度启发式 |

### 不支持的问题（6 个 - 社交推理）
- `q_robot_affected`
- `q_robot_action`
- `q_person_affected`
- `q_person_action`
- `q_robot_suggested_affected`
- `q_robot_suggested_action`
- `q_human_future_action_prediction`

这些问题需要理解社交意图，无法仅通过低层动作可靠推断（无需训练）。

## 集成步骤

### 步骤 1：准备 DualVLN 输出

对于数据集中的每个样本，运行 DualVLN 并保存其输出日志：

```
dualvln_outputs/
├── 101_Spot_1_155.txt
├── 101_Spot_1_22.txt
├── 101_Spot_1_40.txt
└── ...
```

每个文件应包含原始的 DualVLN 日志（如 `example_dualvln.txt` 所示）。

### 步骤 2：修改 evaluate_vlm.py

添加以下代码片段：

**在文件顶部（导入部分）：**
```python
from dualvln_eval_integration import DualVLNEvaluator
```

**大约第 87 行（模型加载处）：**
将：
```python
model = load_model_class(baseline_model, model_to_api_key, config=config)
```
替换为：
```python
if baseline_model.startswith('dualvln'):
    dualvln_outputs_dir = config.get('dualvln_outputs_dir', 'dualvln_outputs/')
    model = DualVLNEvaluator(dualvln_outputs_dir, config=config)
    model.current_sample_id = None  # 将在循环中设置
else:
    model = load_model_class(baseline_model, model_to_api_key, config=config)
```

**大约第 214 行（评估循环内）：**
在调用 `model.generate_text()` 之前，添加：
```python
if baseline_model.startswith('dualvln'):
    model.current_sample_id = sample_id
```

### 步骤 3：更新 config.yaml

添加 DualVLN 配置：

```yaml
# 用于 DualVLN 评估
baseline_model: "dualvln"  # 或者保留当前模型用于其他测试

# 包含 DualVLN 输出日志的目录路径
dualvln_outputs_dir: "dualvln_outputs/"

# 图像尺寸（用于像素到位置的映射）
image_height: 480
image_width: 640
```

### 步骤 4：运行评估

```bash
python socialnavsub/evaluate_vlm.py --cfg_path config.yaml
```

## 映射原理

### 1. 目标位置 (`q_goal_position_*`)

```python
goal_pixel = (y, x)  # 来自 DualVLN 输出
center_x = image_width / 2

if x < center_x - 15%:  "左边"
elif x > center_x + 15%: "右边"
else: "前方"
```

### 2. 机器人移动方向 (`q_robot_moving_direction`)

分析动作频率：
```
FORWARD = 1, LEFT = 2, RIGHT = 3

if forward_ratio > 60%: 包含 "向前移动"
if left_ratio > 30%: 包含 "向左转"
if right_ratio > 30%: 包含 "向右转"
```

### 3. 人物空间位置 (`q_person_spatial_position_*`)

使用避让行为作为代理：
```
如果机器人 LEFT 转向多于 RIGHT：
    人物可能在 RIGHT 侧（避让）
如果机器人 RIGHT 转向多于 LEFT：
    人物可能在 LEFT 侧（避让）
否则：
    从目标方向推断
```

### 4. 距离变化 (`q_person_distance_change`)

```
if forward_ratio > 50%: "靠近"
elif backward_ratio > 30%: "远离"
else: "距离保持大约相同"
```

### 5. 障碍物 (`q_obstructing_path`, `q_obstructing_end_position`)

```
if turn_ratio > 40% and forward_ratio < 60%:
    "是"  # 复杂机动表示存在障碍
else:
    "否"
```

## 局限性

1. **无人检测**：DualVLN 不输出人物位置，因此与人物相关的问题仅使用行为启发式。

2. **无社交推理**：关于“避开”、“让行”、“超车”等问题无法在没有训练分类器的情况下可靠回答。

3. **单帧目标**：DualVLN 只输出一个目标，因此 `begin` 和 `end` 问题得到相同答案。

4. **可调启发式**：阈值（30%、40%、60%）是手动调整的，可能需要根据 DualVLN 的具体行为进行调整。

## 扩展适配器

### 增加社交推理（需要训练）

要支持社交推理问题，可以：

1. 收集带有社交动作标签的动作序列
2. 基于动作统计训练一个小型分类器（如随机森林）
3. 添加到 `dualvln_adapter.py`：

```python
def infer_social_action(self, actions: List[int]) -> str:
    features = self.extract_action_features(actions)
    return self.social_classifier.predict(features)
```

### 改进人物位置估计

如果有人员检测器：

```python
def infer_person_position_with_detection(self,
                                         actions: List[int],
                                         detections: List[BoundingBox]) -> str:
    # 使用检测器得到的实际人物位置
    # 结合机器人移动推断相对位置
```

## 测试

运行适配器测试：

```bash
python socialnavsub/dualvln_eval_integration.py
```

这将解析 `example_dualvln.txt` 并显示所有支持问题的答案。

## 引用

如果使用此适配器，请同时引用 SocialNav-SUB 和原始 DualVLN 论文。

## 故障排除

### “未找到样本 {sample_id} 的 DualVLN 输出”

- 检查 `dualvln_outputs_dir` 是否正确
- 确认输出文件命名为 `{sample_id}.txt`
- 确保 DualVLN 已在所有样本上运行

### 所有答案都是“前方”或“否”

- 检查动作标签是否与预期值匹配（0-5）
- 确认图像尺寸与数据一致
- 必要时调整 `dualvln_adapter.py` 中的阈值

### 解析错误

- 检查 DualVLN 输出格式是否与 `example_dualvln.txt` 匹配
- 解析器期望的行格式：`[timestamp] step_id XXX action Y`
- 目标坐标格式应为：`output text: Y X`