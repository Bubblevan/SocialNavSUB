# 三款 VLM 详细对比 + NaVILA 大脑结构解析

本文档对比你本地的三套模型：**NaVILA 大脑（navila-llama3-8b-8f）**、**Qwen3-VL-8B**、**LLaVA-NeXT-Video-7B**，并解析 NaVILA 所用 VLM（NVILA 系）的架构。

---

## 一、三款模型详细对比

| 维度 | NaVILA 大脑 (navila-llama3-8b-8f) | Qwen3-VL-8B | LLaVA-NeXT-Video-7B |
|------|-----------------------------------|-------------|----------------------|
| **定位** | 机器人导航专用（NaVILA 框架的 VLM） | 通用多模态大模型（图像+视频+长上下文） | 通用开源视频 VLM（SocialNav-SUB 论文用的开源基线） |
| **根架构** | LlavaLlamaModel（VILA/NVILA 系） | Qwen3VLForConditionalGeneration（一体式） | LlavaLlamaForCausalLM（LLaVA 系） |
| **视觉编码器** | **SigLIP** (SiglipVisionModel) | 内置 ViT（SigLIP 系）：depth=27, hidden=1152, patch=16 | **CLIP ViT-L** (openai/clip-vit-large-patch14-336) |
| **视觉分辨率** | 384×384（单尺度） | 多尺度 + 视频 temporal patch | 336 + anyres 多尺度 (224~1008) |
| **视觉→语言桥接** | 2 层 MLP 下采样 (mlp_downsample)，1152→4096 | 视觉直接出 4096（out_hidden_size），与文本对齐 | MLP 2 层 GELU + **Spatial Pool**（spatial_pool_stride=2） |
| **语言模型** | **Llama** 32 层, 4096 hidden, 8K 上下文 | **Qwen3-VL Text** 36 层, 4096 hidden, **262144 (256K)** 上下文 | **Vicuna-7B** (Llama 系) 32 层, 4096 hidden, 4K 上下文 |
| **总参数量级** | ~8B（4×safetensors） | ~8B（4×safetensors） | ~7B（3×safetensors） |
| **视频能力** | 8 帧 (num_video_frames=8)，无显式 time token | 原生视频：temporal_patch_size=2, 长视频+时间戳 | 多帧 + image_grid_pinpoints，spatial pool 做时序 |
| **训练/微调目标** | **ScanQA + RXR**（VLN 真实环境，vila-long-8b-8f-scanqa-rxr-real） | 通用 Instruct（图像/视频/长文档/Agent） | 通用多模态指令（图像+视频指令数据） |
| **与 SocialNav-SUB** | 未在论文中评测；属导航专用 VLM | 未在论文中评测；可作强基线 | **论文中开源基线**（LLaVA-NeXT-Video），PA≈0.46 |
| **目录结构** | 三分支：vision_tower / mm_projector / llm | 单模型 + text_config / vision_config | 单模型 + delay_load 视觉 |

---

## 二、NaVILA 大脑（NVILA 系）结构解析

你本地的 `navila-llama3-8b-8f` 对应的是 **NaVILA 框架里的“大脑”VLM**：基于 **NVILA/VILA** 架构，在 **ScanQA + RXR** 上微调后的 **VILA-Long 8B 8-frame** 模型。下面按数据流解析结构。

### 2.1 整体数据流

```
输入图像/视频帧
    → Vision Tower (SigLIP)  →  视觉特征 [B, N_patch, 1152]
    → MM Projector (MLP)     →  语言空间特征 [B, N', 4096]
    → 与文本 token 拼接
    → LLM (Llama)            →  自回归生成
    → 输出文本（导航/问答等）
```

### 2.2 各模块详解

#### （1）Vision Tower：SigLIP

- **路径**：`navila-llama3-8b-8f/vision_tower/`
- **类型**：`SiglipVisionModel`（与 NVILA 论文一致：NVILA 用 SigLIP 做视觉编码器）
- **配置要点**：
  - `image_size`: 384（高宽 384×384）
  - `patch_size`: 14 → 每帧 patch 数 = (384/14)² ≈ 729
  - `hidden_size`: 1152, `num_hidden_layers`: 27
  - `num_attention_heads`: 16, `intermediate_size`: 4304
- **输出**：取 `mm_vision_select_layer`: -2（倒数第二层），特征维度 1152；`mm_vision_select_feature`: "cls_patch"（CLS + patch tokens）。
- **预处理器**：`preprocessor_config.json` 中 `SiglipImageProcessor`，resize 384×384，归一化 mean/std 0.5。

与 **NVILA 论文** 对应：空间上单尺度 384，未用 Dynamic-S²；时间上支持 8 帧（`num_video_frames=8`），适合导航短视频片段。

#### （2）MM Projector（视觉→语言对齐）

- **路径**：`navila-llama3-8b-8f/mm_projector/`
- **类型**：`MultimodalProjector`，`mm_projector_type`: **"mlp_downsample"**
- **作用**：将视觉特征从 1152 维映射到 LLM 的 4096 维，并做序列长度下采样（减少 token 数、控制成本）。
- **配置**：仅 `mm_hidden_size`: 1152、`model_type`: "v2l_projector"，具体层数/下采样比在实现里（典型为 2 层 MLP + 空间或通道压缩）。

与 **NVILA 论文** 对应：空间/时间令牌压缩（STC 等）在训练阶段用，当前权重已是压缩后的投影结果。

#### （3）LLM：Llama 8B 级

- **路径**：`navila-llama3-8b-8f/llm/`
- **类型**：`LlamaForCausalLM`
- **配置要点**：
  - `num_hidden_layers`: 32, `hidden_size`: 4096
  - `num_attention_heads`: 32, `num_key_value_heads`: 8（GQA）
  - `intermediate_size`: 14336, `max_position_embeddings`: 8192
  - `vocab_size`: 128259（含大量多模态/导航相关 token）
  - `rope_theta`: 500000（长上下文 RoPE）
- **权重**：4 个 shard，总约 16GB（bfloat16），标准 Llama 结构（embed_tokens, layers.*, lm_head）。

整体即 **“SigLIP → MLP 投影 → Llama”** 的经典 VILA/NVILA 三件套。

### 2.3 训练与用途

- **checkpoint 名**：`vila-long-8b-8f-scanqa-rxr-real-v1-seed10-bs10-1e4`
  - **vila-long-8b-8f**：VILA-Long 8B 参数、8 帧视频输入。
  - **scanqa**：3D 场景问答（ScanNet）。
  - **rxr-real**：RXR 数据集在**真实环境**的 VLN（Vision-and-Language Navigation）。
- **含义**：该权重是 **NVILA 系 VLM 在导航+场景理解任务上微调后的“大脑”**，用于 NaVILA 的端到端导航（指令 + 视频观测 → 动作/路径），与 SocialNav-SUB 的“场景理解 VQA”任务不同但相关（都是视觉+语言+空间）。

### 2.4 与 NVILA 论文的对应关系

| 论文组件 | 本仓库 navila-llama3-8b-8f |
|----------|----------------------------|
| 视觉编码器 | SigLIP（与论文一致） |
| 空间缩放 | 单尺度 384，未开 Dynamic-S²（可能是为导航速度做的取舍） |
| 投影器 | 2 层 MLP 下采样（论文中 STC/压缩的一种实现） |
| 语言模型 | Qwen2/Llama 系 8B（论文为 Qwen2 LLM） |
| 时间 | 8 帧，无显式 time token（num_time_tokens=0） |
| 应用 | NaVILA 机器人导航（VLN-CE SR/SPL SOTA） |

---

## 三、对比小结与在 SocialNav-SUB 中的位置

1. **NaVILA 大脑（navila-llama3-8b-8f）**
   - **优势**：导航与 3D 场景问答专用、SigLIP+Llama 高效、单卡可部署。
   - **与 SocialNav-SUB**：论文未评测；若要跑 SUB，需要接 SocialNav-SUB 的评测 pipeline（读入 SUB 的图像/序列与 prompt），输出与人类选项对齐的答案。

2. **Qwen3-VL-8B**
   - **优势**：256K 上下文、原生视频与时间戳、DeepStack + MRoPE、通用能力强。
   - **与 SocialNav-SUB**：论文未评测；可作为强基线（需实现 Qwen3-VL 的 baseline 封装，读入 SUB 的 image/video 与 survey 题目）。

3. **LLaVA-NeXT-Video-7B**
   - **优势**：SocialNav-SUB 论文**已用**的开源基线，复现结果可直接对比；CLIP ViT-L + Spatial Pool + Vicuna-7B。
   - **与 SocialNav-SUB**：论文中 PA≈0.46，空间/时空推理弱于人类与规则基线。

若你想在 SocialNav-SUB 上**同时跑这三者**：现有代码已支持 LLaVA-NeXT（`llava-video`）；NaVILA 大脑和 Qwen3-VL 需要各写一个 `baseline` 封装（类似 `llava.py`），从本地权重加载并实现 `generate_text` / `generate_text_using_past_conversations`，再在 `utils.load_model_class` 里挂上对应 `baseline_model` 名称即可。
