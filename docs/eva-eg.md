## 样本 101_Spot_1_22：gpt-5-nano 表现概览
根据该样本的 `evaluation.json` 和 `question_answers.json`，整理如下简要结论。

### 整体

- **总题数**：120 道（从 evaluation 里逐题统计）。
- **Top-1 正确数**：约 **63 题**（与人类最常见答案一致）。
- **Top-1 准确率**：约 **52.5%**（仅就这一个样本而言）。

---

### 按推理类型看

**1. 时空推理（第 1 题）**

- **q_robot_moving_direction**：VLM 答「moving ahead」，人类约 88.9% 也选「moving ahead」→ **一致，正确**。

**2. 空间推理（行人位置：begin/end）**

- **行人 1**：begin 人类主流「to the left of」，VLM 答「ahead of」→ **错**；end 人类 55.6%「to the left」、44.4%「behind」，VLM 答「behind」→ 未命中人类最常见，**算错**。
- **行人 2**：begin/end 人类主流多为「to the left of」或「behind」，VLM 多次答「ahead of」→ **多题错**。
- **行人 3**：begin 人类 100%「to the left of」，VLM 答「to the right of」→ **错**；end 人类约 66.7%「to the left of」，VLM 答「to the left of」→ **对**。

整体上，**空间位置（谁在左/右/前/后）错得较多**，尤其是「初始/结束位置」与人类标注不一致。

**3. 时空推理（距离变化）**

- 部分题与人类一致（如 p1「closer to」），部分不一致（如 p2「further away」vs 人类主流「closer to」）。

**4. 社交推理（是否阻碍、是否受影响、动作）**

- **q_obstructing_path**：VLM 对多数人答「no」，少数题答「yes」（如 p4、p8、p12），与人类共识部分一致。
- **q_robot_affected / q_robot_action**：多数题为「no」「not considering」，与人类主流（约 88.9% no）大多一致，**社交推理在这一样本上相对更好**。

---

### 典型错误举例（与人类共识对比）

| 题目 | VLM 答案 | 人类主流（或分布） | 说明 |
|------|----------|--------------------|------|
| q_person_spatial_position_begin_p1 | ahead of | to the left of (88.9%) | 行人 1 初始位置判错 |
| q_person_spatial_position_begin_p2 | ahead of | to the left of (88.9%) | 行人 2 初始位置判错 |
| q_person_spatial_position_begin_p3 | to the right of | to the left of (100%) | 左/右混淆 |
| q_person_spatial_position_end_p2 | ahead of | to the left (66.7%) / behind (33.3%) | 结束位置与人类不一致 |
| q_person_distance_change_p2 | further away from | closer to (66.7%) | 距离变化判断相反 |

---

### 小结（就这一个样本）

- **相对较好**：机器人运动方向、部分距离变化、多数「是否受影响/是否考虑」类社交题。
- **明显短板**：**行人相对机器人的空间位置**（左/右/前/后），尤其是 begin/end 位置，与人类共识偏差大，和论文里「空间推理是 VLM 弱项」的结论一致。

若要看全 60 个样本的汇总，需要跑完 `postprocess_results.py`，看 `postprocessed_results.csv`（或同目录下按 question/reasoning group 聚合的 CSV）里的平均 PA/CWPA 和按推理类型的准确率。如果你把其中一段 CSV 或聚合结果贴出来，我也可以帮你解读。


## 样本 101_Spot_1_22：Gemini vs GPT-5-nano

### 1. 整体 Top-1 正确率（本样本）

| 模型 | Top-1 正确数 | Top-1 准确率（约） |
|------|--------------|--------------------|
| **Gemini-3-flash-preview** | 约 **95** 题 | **~79%** |
| **GPT-5-nano** | 约 **63** 题 | **~52.5%** |

本样本上 Gemini 明显优于 GPT。

---

### 2. 空间推理：行人位置（begin/end）

人类标注多数为「to the left of」，少数为「ahead of」「behind」等。

| 题目 | Gemini | GPT-5-nano | 人类主流 | 结论 |
|------|--------|------------|----------|------|
| q_person_spatial_position_begin_p1 | to the left of ✓ | ahead of ✗ | to the left of (88.9%) | Gemini 对，GPT 错 |
| q_person_spatial_position_end_p1 | to the left of ✓ | behind ✗ | to the left of (55.6%) / behind (44.4%) | Gemini 对，GPT 错 |
| q_person_spatial_position_begin_p2 | to the left of ✓ | ahead of ✗ | to the left of (88.9%) | Gemini 对，GPT 错 |
| q_person_spatial_position_end_p2 | to the left of ✓ | ahead of ✗ | to the left of (66.7%) | Gemini 对，GPT 错 |
| q_person_spatial_position_begin_p3 | to the left of ✓ | to the right of ✗ | to the left of (100%) | Gemini 对，GPT 错 |

在空间位置（左/右/前/后）上，Gemini 基本与人类一致，GPT 经常误判为「ahead of」或左右混淆。

---

### 3. 时空推理：距离变化

| 题目 | Gemini | GPT-5-nano | 人类主流 |
|------|--------|------------|----------|
| q_person_distance_change_p1 | closer to ✓ | closer to ✓ | closer to (55.6%) |
| q_person_distance_change_p2 | closer to ✓ | further away from ✗ | closer to (66.7%) |

距离变化题上，Gemini 也与人类更一致，GPT 在部分题目上判断相反。

---

### 4. 机器人运动方向

| 题目 | Gemini | GPT-5-nano | 人类主流 |
|------|--------|------------|----------|
| q_robot_moving_direction | moving ahead ✓ | moving ahead ✓ | moving ahead (88.9%) |

两者均正确。

---

### 5. 社交推理（阻碍、受影响、动作等）

进入社交推理（如 q_obstructing_path、q_robot_affected 等）后，两模型都有正确与错误，整体上 Gemini 在本样本上略好，但差异没有空间推理那么大。

---

### 6. 简要小结

- **空间推理（行人位置）**：Gemini 明显更强，GPT 倾向于答「ahead of」或左右混淆。
- **时空推理（距离变化）**：Gemini 更接近人类分布，GPT 在部分题上方向相反。
- **社交推理**：本样本上 Gemini 略好，但两者都有错。
- **延迟**：Gemini 单题约 18–26s，GPT 约 12–41s，差异不大。

整体上，在该样本中 Gemini-3-flash-preview 的场景空间理解优于 GPT-5-nano，与论文中「Gemini 在空间推理上更强」的结论一致。

还有一个点就是API成本方面俩模型是相当的，在大版本型号一致（GPT5 vs Gemini3）的前提下轻量级的模型，如果GPT要更高的话调用价格就要显著增长
Gemini-3-flash-preview：
- 输入价格： ￥2.5 / M tokens
- 输出价格： ￥15 / M tokens
gemini-2.5-flash-lite：
- 输入价格： ￥0.5 / M tokens
- 输出价格： ￥2 / M tokens
GPT5-nano：
- 输入价格： ￥0.25 / M tokens
- 输出价格： ￥2 / M tokens
GPT5-mini：
- 输入价格： ￥1.25 / M tokens
- 输出价格： ￥10 / M tokens
- 缓存命中价格： ￥0.125 / M tokens
GPT5：
- 输入价格： ￥6.25 / M tokens
- 输出价格： ￥50 / M tokens
- 缓存命中价格： ￥0.625 / M tokens