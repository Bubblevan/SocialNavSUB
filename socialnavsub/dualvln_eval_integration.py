"""
Integration module for DualVLN into SocialNav-SUB evaluation.

This shows how to modify evaluate_vlm.py to support DualVLN without
training any models - using only the adapter's rule-based mapping.
"""

import os
import json
from typing import Dict, Optional, List, Any
import logging

from dualvln_adapter import DualVLNSpatialAdapter, DualVLNWrapper

logger = logging.getLogger(__name__)


class DualVLNEvaluator:
    """
    Evaluator that processes pre-recorded DualVLN outputs and maps them to VQA answers.

    Instead of calling an API or loading a neural model, this reads DualVLN log files
    and applies rule-based mapping to answer survey questions.
    """

    def __init__(self, dualvln_outputs_dir: str, config: Optional[Dict] = None):
        """
        Args:
            dualvln_outputs_dir: Directory containing DualVLN output files
                                  (one per sample, named {sample_id}.txt)
            config: Additional configuration
        """
        self.dualvln_outputs_dir = dualvln_outputs_dir
        self.config = config or {}

        # Initialize adapter
        img_height = self.config.get('image_height', 480)
        img_width = self.config.get('image_width', 640)
        self.adapter = DualVLNSpatialAdapter(img_height, img_width)
        self.wrapper = DualVLNWrapper(self.adapter)

        # Cache for loaded outputs
        self._output_cache: Dict[str, str] = {}

        # Required attributes for compatibility
        self.baseline_type = 'dualvln'
        self.past_conversations = []

    def load_dualvln_output(self, sample_id: str) -> Optional[str]:
        """
        Load DualVLN output for a sample.

        Args:
            sample_id: Sample identifier (e.g., "101_Spot_1_155")

        Returns:
            Raw DualVLN output text, or None if not found
        """
        if sample_id in self._output_cache:
            return self._output_cache[sample_id]

        # Try different filename patterns
        possible_filenames = [
            f"{sample_id}.txt",
            f"{sample_id}_log.txt",
            f"{sample_id}_output.txt",
        ]

        for filename in possible_filenames:
            filepath = os.path.join(self.dualvln_outputs_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                self._output_cache[sample_id] = content
                logger.info(f"Loaded DualVLN output for {sample_id}")
                return content

        logger.warning(f"No DualVLN output found for {sample_id}")
        return None

    def generate_text(self, prompt: str, images: List[Any]) -> str:
        """
        Generate text answer for a prompt.

        This method mimics the interface of other baseline models (GPT4o, Gemini, etc.)
        so it can be used as a drop-in replacement.

        Args:
            prompt: The question prompt
            images: List of images (not used, sample_id set via current_sample_id)

        Returns:
            JSON-formatted answer string
        """
        # Use sample_id set by evaluate_vlm.py
        sample_id = getattr(self, 'current_sample_id', None)

        if sample_id is None:
            logger.error("DualVLN: current_sample_id not set before generate_text call")
            return json.dumps({"answer": "INVALID"})

        # Load DualVLN output for this sample
        dv_output = self.load_dualvln_output(sample_id)

        if dv_output is None:
            logger.error(f"No DualVLN output available for {sample_id}")
            return json.dumps({"answer": "INVALID"})

        # Generate answer using wrapper
        answer = self.wrapper.generate(prompt, images, dv_output)
        return answer

    def _extract_sample_id(self, images: List[Any], prompt: str) -> Optional[str]:
        """
        Extract sample_id from context.

        In the actual evaluate_vlm.py integration, sample_id is available
        in the outer loop and should be passed directly.

        This is a fallback heuristic.
        """
        # Try to extract from prompt (if CoT format includes it)
        # This is fragile and should be replaced with proper passing
        return None

    def add_to_conversation_history(self, conversation: tuple):
        """Compatibility method - DualVLN doesn't use conversation history."""
        pass

    def clear_conversation_history(self):
        """Compatibility method."""
        self.past_conversations = []


def integrate_into_evaluate_vlm():
    """
    Pseudocode showing how to modify evaluate_vlm.py to support DualVLN.

n    Changes needed in evaluate_vlm.py:
    """
    code = '''
    # 1. Add import at top of evaluate_vlm.py:
    from dualvln_eval_integration import DualVLNEvaluator

    # 2. In evaluate_baseline(), modify model initialization:
    if baseline_model == 'dualvln':
        # For DualVLN, we don't load a neural model
        # Instead, we create an evaluator that reads pre-recorded outputs
        dualvln_outputs_dir = config.get('dualvln_outputs_dir', 'dualvln_outputs/')
        model = DualVLNEvaluator(dualvln_outputs_dir, config=config)
    else:
        model = load_model_class(baseline_model, model_to_api_key, config=config)

    # 3. In the evaluation loop, pass sample_id to the model:
    # (Modify the existing loop to handle DualVLN specially if needed)

    # 4. In config.yaml, add:
    # dualvln_outputs_dir: "path/to/dualvln/logs"
    '''
    return code


def create_modified_evaluate_vlm_snippet():
    """
    Actual code snippet to insert into evaluate_vlm.py
    """
    snippet = '''
# ===== DUALVLN SUPPORT =====
# Add these imports at the top of evaluate_vlm.py:
try:
    from dualvln_eval_integration import DualVLNEvaluator
    from dualvln_adapter import DualVLNSpatialAdapter, DualVLNWrapper
    DUALVLN_AVAILABLE = True
except ImportError:
    DUALVLN_AVAILABLE = False
    logger.warning("DualVLN adapter not available")

# Add this to evaluate_baseline() after loading config:
# (Around line 87 where model is loaded)

if baseline_model.startswith('dualvln'):
    if not DUALVLN_AVAILABLE:
        raise ImportError("DualVLN support requested but adapter not available")
    dualvln_outputs_dir = config.get('dualvln_outputs_dir', 'dualvln_outputs/')
    if not os.path.exists(dualvln_outputs_dir):
        raise FileNotFoundError(f"DualVLN outputs directory not found: {dualvln_outputs_dir}")
    model = DualVLNEvaluator(dualvln_outputs_dir, config=config)
    logger.info(f"Initialized DualVLN evaluator with outputs from {dualvln_outputs_dir}")
else:
    model = load_model_class(baseline_model, model_to_api_key, config=config)

# In the evaluation loop (around line 214), for DualVLN we need to pass sample_id.
# The cleanest way is to modify the model call:

if baseline_model.startswith('dualvln'):
    # For DualVLN, we need to tell it which sample we're processing
    model.current_sample_id = sample_id  # Set before calling
    ans_raw = model.generate_text(prompt, images)
else:
    ans_raw = model.generate_text(prompt, images)

# ===== END DUALVLN SUPPORT =====
'''
    return snippet


# Example usage and testing
if __name__ == '__main__':
    import sys

    # Test with example_dualvln.txt
    example_output = """[19:49:06.210009] step_id 330 action 5
[19:49:07.324894] step_id: 330 output text: 130 401
[19:49:08.270133] predicted goal [401, 130]
[19:49:08.270464] step_id 330 action 1
[19:49:09.103996] step_id 331 action 3
[19:49:09.936227] step_id 332 action 1
[19:49:10.767845] step_id 333 action 1
[19:49:11.751438] local_actions [2, 1, 2, 1]
[19:49:11.751727] step_id 334 action 2
[19:49:12.541679] step_id 335 action 1
[19:49:13.315754] step_id 336 action 2
[19:49:14.115066] step_id 337 action 1
[19:49:15.055516] local_actions [2, 1, 2, 1]
[19:49:16.221731] step_id: 339 output text: ↓
[19:49:16.221788] actions [5]
[19:49:16.222106] step_id 339 action 5
[19:49:17.319034] step_id: 339 output text: 340 240
[19:49:18.236368] predicted goal [240, 340]
[19:49:18.236698] step_id 339 action 1
[19:49:19.021911] step_id 340 action 3
[19:49:19.793622] step_id 341 action 1
[19:49:20.557497] step_id 342 action 1
[19:49:21.486973] local_actions [1, 1, 1, 2]
[19:49:21.487257] step_id 343 action 1
[19:49:22.279229] step_id 344 action 1
[19:49:23.133988] step_id 345 action 1
[19:49:24.027749] step_id 346 action 2
[19:49:25.055916] local_actions [1, 1, 1, 2]
[19:49:26.288698] step_id: 348 output text: ↓
[19:49:26.288788] actions [5]
[19:49:26.289057] step_id 348 action 5
[19:49:27.407897] step_id: 348 output text: 283 376
[19:49:28.368159] predicted goal [376, 283]
[19:49:28.368477] step_id 348 action 1
[19:49:29.278137] step_id 349 action 1
[19:49:30.203169] step_id 350 action 2
[19:49:31.084851] step_id 351 action 1
[19:49:32.144982] local_actions [1, 2, 1, 0]
[19:49:32.145282] step_id 352 action 1
[19:49:33.077811] step_id 353 action 2
[19:49:33.938681] step_id 354 action 1
[19:49:36.067500] step_id: 356 output text: ↓
[19:49:36.067593] actions [5]
[19:49:36.067871] step_id 356 action 5
[19:49:37.189545] step_id: 356 output text: 485 413
[19:49:38.162386] predicted goal [413, 485]
[19:49:38.162716] step_id 356 action 3
[19:49:39.097825] step_id 357 action 3
[19:49:40.032206] step_id 358 action 1
[19:49:40.952018] step_id 359 action 1
[19:49:42.017302] local_actions [1, 1, 0, 0]
[19:49:42.017610] step_id 360 action 1
[19:49:42.921761] step_id 361 action 1
[19:49:45.088890] step_id: 363 output text: ↓
[19:49:45.088985] actions [0]
[19:49:45.089259] step_id 363 action 0"""

    # Parse and test
    adapter = DualVLNSpatialAdapter(image_height=480, image_width=640)
    output = adapter.parse_raw_output(example_output)

    print("=" * 60)
    print("DualVLN Adapter Test")
    print("=" * 60)
    print(f"Parsed {len(output.actions)} actions")
    print(f"Goal pixel: {output.goal_pixel}")
    print()

    # Test each spatial/spatiotemporal question
    test_questions = [
        ('q_goal_position_begin', 'multiple_choice'),
        ('q_goal_position_end', 'multiple_choice'),
        ('q_robot_moving_direction', 'multiple_select'),
        ('q_person_spatial_position_begin', 'multiple_choice'),
        ('q_person_spatial_position_end', 'multiple_choice'),
        ('q_person_distance_change', 'multiple_choice'),
        ('q_obstructing_path', 'multiple_choice'),
        ('q_obstructing_end_position', 'multiple_choice'),
    ]

    print("Question Answering Test:")
    print("-" * 60)

    for q_key, q_type in test_questions:
        answer = adapter.answer_question(q_key, output, person_idx=1)
        formatted = adapter.format_answer(answer, q_type)
        print(f"{q_key}:")
        print(f"  Raw: {answer}")
        print(f"  Formatted: {formatted}")
        print()

    print("=" * 60)
    print("Integration code:")
    print(create_modified_evaluate_vlm_snippet())
