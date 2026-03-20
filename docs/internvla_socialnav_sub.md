# 在 SocialNav-SUB 上评估 InternVLA-N1-DualVLN

[InternVLA-N1-DualVLN](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) 是 InternNav 的双系统 VLN 模型。本说明介绍如何在本仓库的 SocialNav-SUB 评测流程中跑该模型。

## 1. 环境准备

### 1.1 安装 InternNav

模型与 processor 由 [InternNav](https://github.com/InternRobotics/InternNav) 提供，需先安装：

```bash
git clone https://github.com/InternRobotics/InternNav
cd InternNav
pip install -e .
pip install diffusers   # InternVLA-N1 依赖（System 1 轨迹用），若报 ModuleNotFoundError: diffusers 则必装
# 若需 submodules（如 LongCLIP、diffusion-policy），按仓库说明执行 git submodule update --init
```

**环境要求（重要）**：  
- **PyTorch >= 2.1**（且需可用，不能是“已禁用”状态）。当前若为 2.0.1 或 CPU-only，transformers 可能禁用 PyTorch，导致 InternNav 加载时出现 **metaclass conflict**。请先升级：`pip install 'torch>=2.1' 'torchvision>=0.16'`（有 GPU 则从 [pytorch.org](https://pytorch.org) 选对应 CUDA 版本）。  
- **transformers**：官方 notebook 使用 4.51；若升级到 4.57+ 后出现 `TypeError: metaclass conflict`，可尝试与 InternNav 对齐：`pip install transformers==4.51.0`。  
- **diffusers**（见上）、可选 flash-attn。

### 1.2 准备 checkpoint
将权重放在项目下，例如：`checkpoints/InternVLA-N1-DualVLN/`（需包含 `config.json`、`*.safetensors`、`tokenizer_config.json`、`preprocessor_config.json` 等）。

## 2. 配置与运行

### 2.1 配置文件

已提供 `config_internvla.yaml`，主要项：

- `baseline_model: "internvla"`  
  也可用 `internvla_n1` 或 `internvla-n1-dualvln`。
- `internvla_path: "checkpoints/InternVLA-N1-DualVLN"`  
  改为你的实际 checkpoint 路径（绝对路径或相对项目根目录均可）。

其余（`method`、`survey_folder`、`prompts_folder`、`evaluation_folder` 等）与 NaVILA/LLaVA 等 baseline 一致，按需修改。

### 2.2 运行评测

在项目根目录执行：

```bash
python socialnavsub/evaluate_vlm.py --cfg_path config_internvla.yaml
```

脚本会：

1. 通过 `internvla_path` 加载 InternVLA-N1（依赖已安装的 InternNav）。
2. 按 SocialNav-SUB 的 prompt + 图像列表调用模型的文本生成接口。
3. 将结果写入 `evaluation_folder` 下对应实验目录，与其它 baseline 格式一致。

## 3. 实现说明

- **Baseline 类**：`socialnavsub/internvla.py` 中的 `InternVLABaseline`。
- **加载方式**：使用 InternNav 的 `InternVLAN1ForCausalLM.from_pretrained(...)` 与 `AutoProcessor.from_pretrained(...)`，与 [官方 inference notebook](https://github.com/InternRobotics/InternNav/blob/main/scripts/notebooks/inference_only_demo.ipynb) 一致。
- **输入/输出**：`generate_text_individual(prompt, images)` 将多图 + 文本组成单轮对话，调用 `model.generate` 得到模型回复字符串，供后续解析与指标计算。
- **依赖**：未安装 InternNav 时会抛出明确错误，提示先安装并设置 `internvla_path`。

## 4. 常见问题

- **ImportError: No module named 'internnav'**  
  未安装或未激活 InternNav 所在环境，请完成 1.1 并确认 `python` 来自该环境。

- **ModuleNotFoundError: No module named 'diffusers'**  
  InternNav 的 InternVLA-N1 代码会 import `diffusers`（用于 System 1 轨迹）。在已安装 InternNav 的同一环境中执行：`pip install diffusers`。

- **TypeError: metaclass conflict ... InternVLAN1ForCausalLM** 或 **Disabling PyTorch because PyTorch >= 2.1 is required**  
  说明当前 PyTorch 版本过旧（如 2.0.1）或不可用，transformers 禁用了 PyTorch，导致 InternNav 继承的基类 metaclass 不一致。解决：升级到 **PyTorch >= 2.1** 并确保环境里 `import torch` 正常。若已用 2.1+ 仍报 metaclass conflict，可尝试与 InternNav 对齐 **transformers 版本**：`pip install transformers==4.51.0`。

- **FileNotFoundError: InternVLA checkpoint not found**  
  检查 `config_internvla.yaml` 中 `internvla_path` 是否指向包含 `config.json` 和权重文件的目录。

- **CUDA / flash_attention**  
  若未装 flash-attn，代码会回退到默认 attention；若需加速可按 InternNav 文档安装 flash-attn。

- **与官方 VLN 评测的区别**  
  官方 InternNav 侧重 VLN-CE/VLN-PE 等导航任务（动作/轨迹）；本流程将同一模型当作 VLM，用于 SocialNav-SUB 的问卷式多选/推理题，输入为 prompt + 多张图，输出为自由文本，再按题目解析为选项或 JSON。
