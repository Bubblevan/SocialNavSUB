# CoT / BEV 消融说明 + 本地 LLaVA-NeXT-Video 评测

## 1. 主实验里 CoT 和 BEV 消融是怎么做的

消融**只改 config 两项**，不改代码；每次跑完会生成新的 experiment 目录，便于对比。

### 1.1 CoT 消融（思维链 vs 独立问答）

- **含义**：同一套题目，要么「按顺序问且把前面答案当上下文」（CoT），要么「每题独立、不给前面答案」（No CoT）。
- **实现**：config 里改 **`method`** 即可。
  - **CoT（主实验）**：`method: "cot"`
  - **No CoT（消融）**：`method: "independent"`
- **结果区分**：实验目录名里会带 method，例如：
  - `experiment_X_gpt-5-nano_cot`
  - `experiment_Y_gpt-5-nano_independent`
  后处理会按 `method` 区分并写入 CSV。

**论文对应**：表 13 等，去掉 CoT 后社交推理明显掉（例如 GPT-4o 0.47→0.35）。

### 1.2 BEV 消融（有无鸟瞰图）

- **含义**：给模型的图像要么「前视图 + BEV」（完整），要么「只有前视图、没有 BEV」。
- **实现**：config 里改 **`prompt_img_type`** 即可。
  - **有 BEV（主实验）**：`prompt_img_type: "img_with_bev"`
  - **No BEV（消融）**：`prompt_img_type: "img"`
- **可选**：还有 `"bev"`（仅 BEV）、`"grid"`、`"grid_with_bev"` 等，按需选用。
- **结果区分**：每个 experiment 目录里会复制当时的 config，`postprocess_results.py` 会读其中的 `prompt_img_type` 并写入 CSV，因此不同 BEV 设置会分开统计。

**论文对应**：去掉 BEV 后，对 GPT-4o 空间/时空推理影响大，对 Gemini、LLaVA-NeXT 影响较小。

### 1.3 跑齐消融的推荐方式

对**同一个模型**（如 gpt-5-nano）跑 4 种配置，得到 4 个 experiment：

| 配置 | method | prompt_img_type | 对应论文 |
|------|--------|------------------|----------|
| 主实验 | cot | img_with_bev | 表 1 主结果 |
| No CoT | independent | img_with_bev | 表 13 去 CoT |
| No BEV | cot | img | 表 13 去 BEV |
| No CoT + No BEV | independent | img | 可选 |

每次改好 config 后执行：

```bash
python socialnavsub/evaluate_vlm.py --cfg_path config.yaml
```

再统一后处理：

```bash
python socialnavsub/postprocess_results.py --cfg_path config.yaml
```

在生成的 CSV 里按 `method`、`prompt_image_type` 筛选即可得到各消融的 PA/CWPA。

---

## 2. 本地 LLaVA-NeXT-Video-7B 怎么 evaluate（不需要 vLLM）

评测用的是 **HuggingFace Transformers**（`LlavaNextVideoForConditionalGeneration`），**不需要 vLLM**。vLLM 是另一种推理加速方案，要接需要单独写一套接口；当前流程用 Transformers 即可。

### 2.1 配置

在 **config.yaml** 里：

1. 指定用 LLaVA 视频模型：`baseline_model: "llava-video"`
2. 指定本地权重路径（二选一）：
   - 相对项目根：`llava_video_path: "llava-next-video-7b"`
   - 或绝对路径：`llava_video_path: "D:/MyLab/SocialNavSUB/llava-next-video-7b"`
3. （可选）显存紧张时开量化：在 config 里增加 `quantization_bits: 4` 或 `8`（需已安装 `bitsandbytes`）。

**示例（仅保留相关项）：**

```yaml
baseline_model: "llava-video"
llava_video_path: "llava-next-video-7b"
# quantization_bits: 4
method: "cot"
prompt_img_type: "img_with_bev"
# ... 其余 survey_folder、prompts_folder 等与现有一致
```

### 2.2 代码已做的修改

- 在 **`socialnavsub/utils.py`** 的 `load_model_class` 里，当 `baseline_model == "llava-video"` 时：
  - 若 config 里存在非空的 **`llava_video_path`**，则用该路径加载（支持相对/绝对，会做 `os.path.expanduser`）；
  - 否则仍用 `"llava-hf/LLaVA-NeXT-Video-7B-hf"` 从 HuggingFace 下载。
- 本地路径会直接传给 `LlavaNextVideoProcessor.from_pretrained(...)` 和 `LlavaNextVideoForConditionalGeneration.from_pretrained(...)`，HF 支持本地目录。

### 2.3 运行方式

在项目根目录下（保证能 import `socialnavsub`）：

```bash
python socialnavsub/evaluate_vlm.py --cfg_path config.yaml
```

若 OOM，可把 config 里 `quantization_bits: 4` 打开再跑。

### 2.4 环境

- 已安装 `transformers`、`torch`、以及 LLaVA-NeXT 所需的依赖（见 `requirements.txt`）。
- 若用 `quantization_bits: 4` 或 `8`，需要：`pip install bitsandbytes`。
- **不需要** 单独装 vLLM；当前评测 pipeline 与 vLLM 无关。

---

## 3. 小结

| 问题 | 做法 |
|------|------|
| CoT 消融 | config 设 `method: "independent"`（No CoT）或 `method: "cot"`（CoT），重跑评测即可。 |
| BEV 消融 | config 设 `prompt_img_type: "img"`（No BEV）或 `"img_with_bev"`（有 BEV），重跑评测即可。 |
| 本地 LLaVA-NeXT-Video | config 设 `baseline_model: "llava-video"` 和 `llava_video_path: "llava-next-video-7b"`，不需 vLLM，直接 `evaluate_vlm.py`。 |
| 显存不够 | config 里设 `quantization_bits: 4`（或 8），并安装 `bitsandbytes`。 |
