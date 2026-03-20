# DualVLN 集成摘要 – 免训练方法

## 已完成工作

创建了一套完整的免训练集成方案，使 DualVLN（输出离散动作和像素目标的导航模型）无需微调即可在 SocialNav-SUB 的 VQA 任务上进行评估。

## 创建的文件

| 文件 | 用途 |
|------|------|
| `socialnavsub/dualvln_adapter.py` | 基于规则的映射逻辑核心代码（480 行） |
| `socialnavsub/dualvln_eval_integration.py` | 与评估框架的集成 |
| `config_dualvln_example.yaml` | 示例配置文件 |
| `DUALVLN_INTEGRATION_GUIDE.md` | 详细集成指南 |
| `test_dualvln_integration.py` | 端到端测试套件 |
| `evaluate_vlm_patch.py` | 代码修改参考 |

## 对现有文件的修改

### `socialnavsub/evaluate_vlm.py`
三处微小改动：
1. **第 45‑50 行**：添加 DualVLN 导入
2. **第 95‑107 行**：添加 DualVLN 评估器初始化
3. **第 184‑185 行**：为 DualVLN 添加 sample_id 传递

**总计改动**：约 15 行代码

## 工作原理

### 映射逻辑（免训练规则）

#### 1. 目标位置（空间）
```
目标像素：(y, x)
图像中心 x = 320

if x < 中心 - 15%: "在左边"
elif x > 中心 + 15%: "在右边"
else: "正前方"
```

#### 2. 机器人移动方向（时空）
```
动作：[1, 3, 1, 1, 2, 1, ...]  # 1=前进, 2=左转, 3=右转

前进比例 = count(1) / 总步数
左转比例 = count(2) / 总步数
右转比例 = count(3) / 总步数

结果：["向前移动", "向左转"]  # 若两个比例都大于阈值
```

#### 3. 人的位置（空间）
```
# 利用避让行为作为代理
if 左转 > 右转 + 2:
    人在“右边”（机器人向左避让）
elif 右转 > 左转 + 2:
    人在“左边”（机器人向右避让）
else:
    根据目标方向推断
```

#### 4. 距离变化（时空）
```
if 前进比例 > 50%: "靠近"
elif 后退比例 > 30%: "远离"
else: "距离基本不变"
```

#### 5. 障碍物（空间/时空）
```
if 转向比例 > 40% and 前进比例 < 60%:
    "是"  # 复杂操纵暗示有障碍
else:
    "否"
```

## 支持的问题

### ✅ 支持（8 个问题）

| 类别 | 问题 | 回答方式 |
|------|------|----------|
| 空间 | `q_goal_position_begin` | 目标像素 x 坐标 |
| 空间 | `q_goal_position_end` | 与开始相同（单一目标） |
| 空间 | `q_person_spatial_position_begin` | 转向行为分析 |
| 空间 | `q_person_spatial_position_end` | 转向行为分析 |
| 空间 | `q_obstructing_end_position` | 动作复杂度 |
| 时空 | `q_robot_moving_direction` | 动作频率 |
| 时空 | `q_person_distance_change` | 前进/转向比例 |
| 时空 | `q_obstructing_path` | 转向复杂度 |

### ❌ 不支持（7 个问题）

这些问题需要理解社会意图，无法仅从底层动作可靠推断，且没有训练：

- `q_robot_affected` – 机器人是否受人的影响？
- `q_robot_action` – 机器人在做什么？（避让/超越/让行/跟随）
- `q_person_affected` – 人是否受机器人的影响？
- `q_person_action` – 人在做什么？
- `q_robot_suggested_affected` – 机器人应该受影响吗？
- `q_robot_suggested_action` – 机器人应该做什么？
- `q_human_future_action_prediction` – 人接下来会做什么？

## 测试结果

```
============================================================
测试摘要
============================================================
  适配器解析：通过
  问题回答：通过
  评估器：通过
  集成：通过
  配置加载：通过
============================================================
所有测试均通过！
```

## 使用方法

### 1. 准备 DualVLN 输出

```bash
mkdir dualvln_outputs

# 对每个样本运行 DualVLN 并保存输出：
# dualvln_outputs/101_Spot_1_155.txt
# dualvln_outputs/101_Spot_1_22.txt
# ...
```

### 2. 配置

```bash
cp config_dualvln_example.yaml config.yaml
# 编辑 config.yaml 设置正确路径
```

### 3. 运行评估

```bash
python socialnavsub/evaluate_vlm.py --cfg_path config.yaml
```

## 局限性

1. **无人物检测**：人的位置通过避让行为推断，而非检测得到
2. **单一目标**：`begin` 和 `end` 目标位置相同（DualVLN 只输出一个目标）
3. **无社会推理**：15 个问题中有 7 个无法回答
4. **启发式阈值**：30%、40%、60%、15% 均为手动调参，可能需要调整
5. **假设为避让**：人的位置逻辑假设机器人避让人，而非相反

## 可能的改进（仍保持免训练）

### 1. 更好的位置估计
使用计算机视觉检测人物边界框，然后映射到空间关系：
```python
person_box = detect_person(image)  # YOLO, RT-DETR 等
position = map_box_to_relation(person_box, robot_position)
```

### 2. 社会动作分类（基于规则）
```python
def infer_social_action(actions, goal, person_position):
    if is_yielding_pattern(actions):  # 减速，让人通过
        return "让行"
    elif is_overtaking_pattern(actions):  # 偏离后回归，加速
        return "超越"
    elif is_avoiding_pattern(actions):  # 偏离路径
        return "避让"
    else:
        return "不考虑"
```

### 3. 时序目标跟踪
若 DualVLN 在视频序列上运行，跟踪目标如何变化：
```python
goals_over_time = [goal_t1, goal_t2, ...]
q_goal_position_begin = infer_from(goals_over_time[0])
q_goal_position_end = infer_from(goals_over_time[-1])
```

### 4. 多模态融合
将 DualVLN 与现成模型结合：
```python
person_location = owlvit.detect(image, "person")
social_action = clip.classify(image, ["避让", "让行", ...])
```

## 性能预期

### 很可能效果良好
- 目标位置（直接来自像素坐标）
- 机器人移动方向（直接来自动作）

### 可能有效（取决于 DualVLN 的行为）
- 人的位置（若 DualVLN 会避让人）
- 距离变化（若前进进度相关）

### 很可能失败
- 所有社会推理问题（受影响、动作、建议）

## 结论

本集成表明，仅使用 DualVLN 的导航输出并通过基于规则的映射，**15 个 SocialNav-SUB 问题中的 8 个**无需任何训练即可回答。剩余的 7 个问题需要理解社会意图，而这些意图无法从底层动作中可靠提取。

要在不训练的情况下回答全部 15 个问题，需要添加：
1. 人物检测（使用现成检测器）
2. 社会动作启发式（基于动作的规则分类器）
3. 或接受部分问题无法回答

## 代码统计

- **新增代码**：约 700 行（适配器 + 集成 + 测试）
- **修改代码**：`evaluate_vlm.py` 中约 15 行
- **新增依赖**：0（纯 Python + NumPy）
- **所需训练**：无
- **所需模型权重**：无