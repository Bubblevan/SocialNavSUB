#!/usr/bin/env python3
"""
GPT-4o API Baseline Wrapper

Provides a clean interface for calling OpenAI's GPT-4o endpoint
with support for image embedding and optional conversation context.
"""
import os
import time
import json
import logging
import argparse
from typing import List, Tuple, Optional, Dict

import base64
import requests
import yaml

from api_baseline import APIBaseline

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration from file.

    Args:
        config_path: Path to YAML config file.
    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, 'r') as config_fp:
        return yaml.safe_load(config_fp)


class GPT4o(APIBaseline):
    """
    Wrapper around OpenAI's GPT-4o chat completion API.

    Adds support for embedding images as base64-encoded URLs
    and including past conversation history optionally.
    When base_url is set (e.g. for DMXAPI 中转), uses /responses format.
    """
    def __init__(
        self,
        model_name: str,
        api_key_env_var: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize GPT-4o client.

        Args:
            model_name: Name of the GPT-4o model (e.g., 'gpt-4o').
            api_key_env_var: Environment variable name containing the API key.
            endpoint: Optional HTTP endpoint for the chat completion API.
            base_url: Optional base URL for proxy/中转 (e.g. https://www.dmxapi.cn/v1).
                     When set, endpoint becomes base_url/responses and DMXAPI request/response format is used.
        """
        super().__init__(model_name=model_name, api_key_env_var=api_key_env_var)
        self.model_name = model_name
        self.api_key = os.getenv(api_key_env_var) if api_key_env_var else None
        if not self.api_key:
            raise ValueError(f"API key not found in env var '{api_key_env_var}'")
        self._use_dmxapi = bool(base_url)
        if base_url:
            self.endpoint = base_url.rstrip("/") + "/responses"
            self.logger = logging.getLogger(f"{__name__}.GPT4o")
        else:
            self.endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
            self.logger = logging.getLogger(f"{__name__}.GPT4o")

    def encode_image(self, image_path: str) -> str:
        """
        Read and base64-encode an image for inline embedding.

        Args:
            image_path: Path to the image file.
        Returns:
            Base64-encoded string of the image.
        """
        with open(image_path, 'rb') as img_fp:
            return base64.b64encode(img_fp.read()).decode('utf-8')

    def generate_text_individual(
        self,
        text: str,
        image_filepaths: Optional[List[str]] = None
    ) -> str:
        """
        Single-turn completion without past conversation context.

        Args:
            text: Prompt text.
            image_filepaths: List of base64-encoded images.
        Returns:
            Model's response content.
        """
        return self._call_api(text, image_filepaths or [], None)

    def generate_text_using_past_conversations(
        self,
        text: str,
        image_filepaths: Optional[List[str]] = None
    ) -> str:
        """
        Multi-turn completion including past conversation history.

        Args:
            text: Prompt text.
            image_filepaths: List of base64-encoded images.
        Returns:
            Model's response content.
        """
        return self._call_api(text, image_filepaths or [], self.past_conversations)

    def _call_api(
        self,
        text: str,
        encoded_images: List[str],
        past_conversations: Optional[List[Tuple[str, str]]]
    ) -> str:
        """
        Internal method to send a chat completion request.
        Uses DMXAPI /responses format when base_url was set (中转).
        """
        if self._use_dmxapi:
            return self._call_dmxapi(text, encoded_images)
        return self._call_openai(text, encoded_images)

    def _call_dmxapi(self, text: str, encoded_images: List[str]) -> str:
        """DMXAPI 中转: POST /v1/responses, input 数组格式，Authorization 仅 key."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }
        content = [{"type": "input_text", "text": text}]
        for img_b64 in encoded_images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img_b64}"
            })
        payload = {
            "model": self.model_name,
            "input": [{"role": "user", "content": content}],
            "stream": False,
        }
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
            data = response.json()
            if response.status_code != 200:
                err = data.get("error", data.get("message", "Unknown error"))
                if isinstance(err, dict):
                    err = err.get("message", str(err))
                raise Exception(f"Status {response.status_code}: {err}")
            status = data.get("status")
            if status != "completed":
                raise Exception(f"DMXAPI status: {status}")
            output = data.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    content_out = item.get("content")
                    if isinstance(content_out, str):
                        return content_out.strip()
                    if isinstance(content_out, list):
                        for block in content_out:
                            if isinstance(block, dict) and block.get("type") == "output_text":
                                return (block.get("text") or "").strip()
                            if isinstance(block, dict) and "text" in block:
                                return (block.get("text") or "").strip()
            raise Exception("DMXAPI: no text in output")
        except Exception as e:
            self.logger.warning("DMXAPI call failed: %s. Retrying in 10s.", e)
            time.sleep(10)
            return self._call_dmxapi(text, encoded_images)

    def _call_openai(self, text: str, encoded_images: List[str]) -> str:
        """Official OpenAI chat/completions format."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        user_block = [{"type": "text", "text": text}]
        for img_b64 in encoded_images:
            user_block.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_block}],
            "response_format": {"type": "json_object"},
        }
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
            data = response.json()
            if response.status_code != 200:
                err = data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"Status {response.status_code}: {err}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.warning("API call failed: %s. Retrying in 10s.", e)
            time.sleep(10)
            return self._call_openai(text, encoded_images)


def main():
    """
    CLI entrypoint: load config, instantiate GPT4o, and perform a demo call.
    """
    parser = argparse.ArgumentParser(description="GPT-4o Baseline CLI")
    parser.add_argument(
        '--config',
        type=str,
        default='gpt4o_cfg.yaml',
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = GPT4o(
        model_name=cfg['model_name'],
        api_key_env_var=cfg.get('api_key_env_var'),
        endpoint=cfg.get('endpoint')
    )

    prompt = cfg.get('demo_prompt', 'Hello, GPT-4o!')
    demo_images = cfg.get('demo_images', [])
    result = client.generate_text_individual(prompt, demo_images)
    logger.info("Demo result: %s", result)


if __name__ == '__main__':
    main()