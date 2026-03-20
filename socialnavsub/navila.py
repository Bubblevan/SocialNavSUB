# -*- coding: utf-8 -*-
"""
NaVILA (VILA-Long) baseline: load local checkpoint and run SocialNav-SUB evaluation.
Requires checkpoint dir on Python path for trust_remote_code (e.g. checkpoints/navila-llama3-8b-8f).
Checkpoint config.json uses model_type "llava_llama" which is not in transformers CONFIG_MAPPING;
we register the checkpoint's VILA config/model under "llava_llama" before loading.
Checkpoint code imports Qwen2* from transformers; if your transformers<4.40 lacks Qwen2, we alias
Qwen2Config/Qwen2ForCausalLM/Qwen2PreTrainedModel to Llama* (this checkpoint uses Llama LLM).
"""
import os
import re
import sys
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from baseline import Baseline

logger = logging.getLogger(__name__)


def _ensure_qwen2_available() -> None:
    """If transformers has no Qwen2 (e.g. <4.40), alias Qwen2* to Llama* so checkpoint code can import."""
    import transformers
    if getattr(transformers, "Qwen2Config", None) is not None:
        return
    try:
        from transformers.models.llama import (
            LlamaConfig,
            LlamaForCausalLM,
            LlamaPreTrainedModel,
        )
    except ImportError as e:
        raise ImportError(
            "NaVILA checkpoint code requires Qwen2Config/Qwen2ForCausalLM from transformers, "
            "which are available in transformers>=4.40. Either upgrade: pip install 'transformers>=4.40', "
            "or ensure Llama models are available for fallback."
        ) from e
    setattr(transformers, "Qwen2Config", LlamaConfig)
    setattr(transformers, "Qwen2ForCausalLM", LlamaForCausalLM)
    setattr(transformers, "Qwen2PreTrainedModel", LlamaPreTrainedModel)
    # So "from transformers import Qwen2Config" resolves when package uses __all__ or direct access
    if hasattr(transformers, "__all__") and "Qwen2Config" not in transformers.__all__:
        transformers.__all__ = getattr(transformers, "__all__", []) + [
            "Qwen2Config", "Qwen2ForCausalLM", "Qwen2PreTrainedModel"
        ]


def _register_navila_auto_classes(model_path: str) -> None:
    """Register checkpoint's VILA config and model for model_type 'llava_llama' so AutoConfig/AutoModel work."""
    _ensure_qwen2_available()
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from configuration_vila import VILAConfig
    from modeling_vila import VILAForCausalLM

    class LlavaLlamaConfig(VILAConfig):
        model_type = "llava_llama"

    if "llava_llama" not in CONFIG_MAPPING:
        AutoConfig.register("llava_llama", LlavaLlamaConfig)
    if LlavaLlamaConfig not in AutoModelForCausalLM._model_mapping.keys():
        AutoModelForCausalLM.register(LlavaLlamaConfig, VILAForCausalLM, exist_ok=True)


class NaVILABaseline(Baseline):
    """Load NaVILA/VILA from local checkpoint and run generate_content for VQA."""

    def __init__(self, model_path: str):
        super().__init__(use_cot=False)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"NaVILA checkpoint not found: {self.model_path}")

        # So that trust_remote_code finds local modules (modeling_vila, etc.)
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)

        _register_navila_auto_classes(self.model_path)

        import torch
        from transformers import AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading NaVILA from %s (trust_remote_code)...", self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu" and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.baseline_type = "local"
        logger.info("NaVILA loaded on %s", self.device)

    def _images_to_prompt_list(self, images: List, text: str):
        """Build value list [img1, img2, ..., text] for generate_content."""
        out = []
        for im in images:
            if isinstance(im, np.ndarray):
                out.append(Image.fromarray(im))
            elif isinstance(im, Image.Image):
                out.append(im)
            else:
                out.append(Image.open(im).convert("RGB"))
        out.append(text)
        return out

    def extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model output.")
        return match.group()

    def generate_text_individual(self, prompt: str, images: List) -> str:
        if not images:
            raise ValueError("NaVILA requires at least one image.")
        value = self._images_to_prompt_list(images, prompt)
        try:
            out = self.model.generate_content(value)
            return self.extract_json(out)
        except Exception as e:
            logger.warning("NaVILA generate_content failed: %s", e)
            raise

    def generate_text_using_past_conversations(self, text: str, image_filepaths: List = None) -> str:
        return self.generate_text_individual(text, image_filepaths or [])
