# -*- coding: utf-8 -*-
"""
InternVLA-N1-DualVLN baseline for SocialNav-SUB.

Requires InternNav to be installed (model + processor live there):
  git clone https://github.com/InternRobotics/InternNav && cd InternNav && pip install -e .
Checkpoint: local path or HuggingFace id, e.g. checkpoints/InternVLA-N1-DualVLN.
"""
import os
import re
import logging
from typing import List, Any

import numpy as np
from PIL import Image

from baseline import Baseline

logger = logging.getLogger(__name__)


def _ensure_internnav():
    try:
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
        return InternVLAN1ForCausalLM
    except ImportError as e:
        err_msg = str(e).lower()
        hint = (
            "InternVLA baseline requires InternNav and its dependencies. Install with:\n"
            "  git clone https://github.com/InternRobotics/InternNav && cd InternNav && pip install -e .\n"
            "  pip install diffusers  # required by InternVLA-N1 (System 1 trajectory)\n"
            "Then set internvla_path in config to your checkpoint (e.g. checkpoints/InternVLA-N1-DualVLN)."
        )
        if "diffusers" in err_msg:
            hint = (
                "InternVLA-N1 depends on diffusers (used by InternNav). Install with:\n"
                "  pip install diffusers\n"
                "Then rerun. If other modules are missing, see InternNav requirements or docs/internvla_socialnav_sub.md."
            )
        raise ImportError(hint) from e


class InternVLABaseline(Baseline):
    """Load InternVLA-N1 (DualVLN) via InternNav and run VLM inference for survey-style prompts."""

    def __init__(self, model_path: str, attn_implementation: str = "flash_attention_2"):
        super().__init__(use_cot=False)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"InternVLA checkpoint not found: {self.model_path}")

        InternVLAN1ForCausalLM = _ensure_internnav()
        import torch
        if not hasattr(torch, 'xpu'):
            torch.xpu = torch.cuda  # 猴子补丁，指向CUDA
        from transformers import AutoProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading InternVLA-N1 from %s ...", self.model_path)

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": self.device} if str(self.device).startswith("cuda") else None,
        }
        if attn_implementation and attn_implementation != "none":
            try:
                __import__("flash_attn")
                kwargs["attn_implementation"] = attn_implementation
            except ImportError:
                logger.warning("flash_attn not found, using default attention")
                kwargs.pop("attn_implementation", None)

        self.model = InternVLAN1ForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        self.baseline_type = "local"
        logger.info("InternVLA-N1 loaded on %s", self.device)

    def _images_to_pil(self, images: List) -> List[Image.Image]:
        out = []
        for im in images:
            if isinstance(im, np.ndarray):
                out.append(Image.fromarray(im).convert("RGB"))
            elif isinstance(im, Image.Image):
                out.append(im.convert("RGB"))
            else:
                out.append(Image.open(im).convert("RGB"))
        return out

    def generate_text_individual(self, prompt: str, images: List) -> str:
        if not images:
            raise ValueError("InternVLA requires at least one image.")
        import torch

        pil_images = self._images_to_pil(images)
        # Build conversation: user message with images + text (same format as InternNav agent)
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=pil_images, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences
        answer = self.processor.tokenizer.decode(
            gen_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return answer.strip()

    def generate_text_using_past_conversations(self, text: str, image_filepaths: List = None) -> str:
        return self.generate_text_individual(text, image_filepaths or [])
