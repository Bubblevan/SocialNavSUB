"""
DualVLN Adapter for SocialNav-SUB (Training-Free)

Converts DualVLN output (discrete action sequence + pixel goal) to VQA answers
for spatial and spatiotemporal reasoning questions.

Action labels (from DualVLN example):
    0: STOP
    1: Move forward
    2: Turn left
    3: Turn right
    4: Move backward (if exists)
    5: Set goal / Trigger prediction
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class DualVLNOutput:
    """Parsed DualVLN output"""
    actions: List[int]              # Action sequence
    goal_pixel: Optional[Tuple[int, int]]  # Goal in pixel coordinates (y, x) or (row, col)
    step_ids: List[int]             # Step IDs corresponding to actions


class DualVLNSpatialAdapter:
    """
    Training-free adapter that maps DualVLN navigation outputs to
    spatial/spatiotemporal reasoning answers using heuristic rules.
    """

    # Action definitions
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    BACKWARD = 4
    SET_GOAL = 5

    def __init__(self, image_height: int = 480, image_width: int = 640):
        """
        Args:
            image_height: Height of input images
            image_width: Width of input images
        """
        self.image_height = image_height
        self.image_width = image_width

    def parse_raw_output(self, raw_output: str) -> DualVLNOutput:
        """
        Parse DualVLN raw log output (like example_dualvln.txt)

        Args:
            raw_output: Raw text output from DualVLN

        Returns:
            DualVLNOutput with actions and goal
        """
        actions = []
        step_ids = []
        goal_pixel = None

        lines = raw_output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if 'action' in line and 'step_id' in line:
                # Parse: "[timestamp] step_id XXX action Y"
                try:
                    parts = line.split()
                    step_idx = parts.index('step_id')
                    action_idx = parts.index('action')
                    step_id = int(parts[step_idx + 1])
                    action = int(parts[action_idx + 1])
                    actions.append(action)
                    step_ids.append(step_id)
                except (ValueError, IndexError):
                    continue
            elif 'output text:' in line and not '↓' in line:
                # Parse goal coordinates: "output text: 340 240"
                try:
                    text_part = line.split('output text:')[1].strip()
                    coords = text_part.split()
                    if len(coords) == 2:
                        # Format is "y x" (row, col)
                        y, x = int(coords[0]), int(coords[1])
                        goal_pixel = (y, x)
                except (ValueError, IndexError):
                    continue

        return DualVLNOutput(actions=actions, goal_pixel=goal_pixel, step_ids=step_ids)

    def infer_goal_position(self, goal_pixel: Optional[Tuple[int, int]],
                           robot_position: str = "center") -> str:
        """
        Infer goal position relative to robot from pixel coordinates.

        Args:
            goal_pixel: (y, x) coordinates in image
            robot_position: Where robot is in the image (assumed center)

        Returns:
            One of: "ahead", "to the left", "to the right"
        """
        if goal_pixel is None:
            # Default to ahead if no goal
            return "ahead"

        y, x = goal_pixel
        center_x = self.image_width / 2

        # Determine horizontal position
        if x < center_x - self.image_width * 0.15:
            return "to the left"
        elif x > center_x + self.image_width * 0.15:
            return "to the right"
        else:
            return "ahead"

    def infer_robot_moving_direction(self, actions: List[int]) -> List[str]:
        """
        Infer robot moving direction from action sequence.

        Args:
            actions: List of action IDs

        Returns:
            List of directions: "moving ahead", "turning left", "turning right"
        """
        if not actions:
            return ["moving ahead"]

        # Count actions (exclude SET_GOAL and STOP for movement analysis)
        movement_actions = [a for a in actions
                           if a not in [self.SET_GOAL, self.STOP]]

        if not movement_actions:
            return ["moving ahead"]

        total = len(movement_actions)
        forward_count = movement_actions.count(self.FORWARD)
        left_count = movement_actions.count(self.LEFT)
        right_count = movement_actions.count(self.RIGHT)
        backward_count = movement_actions.count(self.BACKWARD)

        directions = []

        # Thresholds
        dominant_threshold = 0.3  # 30% to be considered dominant

        # Forward vs turning
        forward_ratio = forward_count / total
        left_ratio = left_count / total
        right_ratio = right_count / total

        # If mostly forward
        if forward_ratio > 0.6:
            directions.append("moving ahead")
        elif forward_ratio > 0.3:
            directions.append("moving ahead")

        # Check turning
        if left_ratio > dominant_threshold or (left_count > 2 and left_count > right_count):
            directions.append("turning left")
        elif right_ratio > dominant_threshold or (right_count > 2 and right_count > left_count):
            directions.append("turning right")

        # If no clear direction, default to moving ahead
        if not directions:
            directions.append("moving ahead")

        return directions

    def infer_distance_change(self, actions: List[int],
                             goal_pixel: Optional[Tuple[int, int]]) -> str:
        """
        Infer distance change based on action patterns.

        Heuristic: If robot is mostly moving forward, distance decreases.
        If turning a lot without forward movement, distance stays similar.

        Args:
            actions: Action sequence
            goal_pixel: Goal position

        Returns:
            One of: "closer to", "further away from", "about the same distance to"
        """
        if not actions:
            return "about the same distance to"

        movement_actions = [a for a in actions
                           if a not in [self.SET_GOAL, self.STOP]]

        if not movement_actions:
            return "about the same distance to"

        total = len(movement_actions)
        forward_count = movement_actions.count(self.FORWARD)
        backward_count = movement_actions.count(self.BACKWARD)

        forward_ratio = forward_count / total
        backward_ratio = backward_count / total

        # Thresholds
        if forward_ratio > 0.5:
            return "closer to"
        elif backward_ratio > 0.3:
            return "further away from"
        else:
            return "about the same distance to"

    def infer_obstruction(self, actions: List[int],
                         goal_pixel: Optional[Tuple[int, int]],
                         has_person: bool = True) -> str:
        """
        Infer if path is obstructed based on navigation patterns.

        Heuristic: If robot takes many turning actions, it may be avoiding/obstructed.

        Args:
            actions: Action sequence
            goal_pixel: Goal position
            has_person: Whether there's a person in the scene

        Returns:
            One of: "yes", "no"
        """
        if not has_person or not actions:
            return "no"

        movement_actions = [a for a in actions
                           if a not in [self.SET_GOAL, self.STOP]]

        if not movement_actions:
            return "no"

        total = len(movement_actions)
        turn_count = (movement_actions.count(self.LEFT) +
                     movement_actions.count(self.RIGHT))
        forward_count = movement_actions.count(self.FORWARD)

        turn_ratio = turn_count / total
        forward_ratio = forward_count / total

        # If lots of turning relative to forward movement, likely obstructed
        if turn_ratio > 0.4 and forward_ratio < 0.6:
            return "yes"

        return "no"

    def infer_person_spatial_position(self, person_idx: int,
                                     actions: List[int],
                                     goal_pixel: Optional[Tuple[int, int]]) -> str:
        """
        Infer person's spatial position relative to robot.

        This is challenging without person detection. We use heuristics:
        - If robot turns left a lot, person might be on the right (robot avoiding)
        - If robot turns right a lot, person might be on the left
        - Otherwise use goal direction as proxy

        Args:
            person_idx: Person index (1-based)
            actions: Action sequence
            goal_pixel: Goal position

        Returns:
            One of: "ahead of", "to the left of", "to the right of", "behind"
        """
        if not actions:
            return "ahead of"

        movement_actions = [a for a in actions
                           if a not in [self.SET_GOAL, self.STOP]]

        left_count = movement_actions.count(self.LEFT)
        right_count = movement_actions.count(self.RIGHT)
        forward_count = movement_actions.count(self.FORWARD)

        total = len(movement_actions) if movement_actions else 1

        # Simple heuristic: if robot predominantly turns one way,
        # person is likely on the opposite side (avoidance behavior)
        if left_count > right_count + 2:
            # Turning left -> person on right
            return "to the right of"
        elif right_count > left_count + 2:
            # Turning right -> person on left
            return "to the left of"
        elif forward_count / total > 0.7:
            # Mostly forward, person likely ahead
            return "ahead of"
        else:
            # Default based on goal direction
            goal_pos = self.infer_goal_position(goal_pixel)
            if goal_pos == "ahead":
                return "ahead of"
            elif goal_pos == "to the left":
                return "to the left of"
            else:
                return "to the right of"

    def answer_question(self, question_key: str,
                       dualvln_output: DualVLNOutput,
                       person_idx: int = 0) -> Any:
        """
        Answer a specific question based on DualVLN output.

        Args:
            question_key: Question identifier (e.g., 'q_goal_position_begin')
            dualvln_output: Parsed DualVLN output
            person_idx: Person index (0 for robot-only questions)

        Returns:
            Answer (str or list of str)
        """
        actions = dualvln_output.actions
        goal = dualvln_output.goal_pixel

        # Spatial reasoning questions
        if question_key == 'q_goal_position_begin':
            return self.infer_goal_position(goal, "center")

        elif question_key == 'q_goal_position_end':
            # Same as begin for now (no temporal evolution in single output)
            return self.infer_goal_position(goal, "center")

        elif question_key == 'q_person_spatial_position_begin':
            return self.infer_person_spatial_position(person_idx, actions, goal)

        elif question_key == 'q_person_spatial_position_end':
            # Infer based on movement patterns
            return self.infer_person_spatial_position(person_idx, actions, goal)

        elif question_key == 'q_obstructing_end_position':
            return self.infer_obstruction(actions, goal, has_person=True)

        # Spatiotemporal reasoning questions
        elif question_key == 'q_robot_moving_direction':
            return self.infer_robot_moving_direction(actions)

        elif question_key == 'q_person_distance_change':
            return self.infer_distance_change(actions, goal)

        elif question_key == 'q_obstructing_path':
            return self.infer_obstruction(actions, goal, has_person=True)

        else:
            # Unknown question - return default
            return "no"

    def format_answer(self, answer: Any, question_type: str) -> str:
        """
        Format answer to JSON string expected by evaluation.

        Args:
            answer: Raw answer (str or list)
            question_type: "multiple_choice" or "multiple_select"

        Returns:
            JSON-formatted answer string
        """
        if question_type == "multiple_select" and isinstance(answer, list):
            return json.dumps({"answer": answer})
        else:
            if isinstance(answer, list):
                # Take first for multiple choice
                answer = answer[0] if answer else ""
            return json.dumps({"answer": answer})


class DualVLNWrapper:
    """
    Wrapper to integrate DualVLN adapter with SocialNav-SUB evaluation.
    This replaces the standard model.generate_text() call.
    """

    def __init__(self, adapter: Optional[DualVLNSpatialAdapter] = None):
        self.adapter = adapter or DualVLNSpatialAdapter()
        self.baseline_type = 'dualvln'
        self.past_conversations = []

    def generate(self, prompt: str, images: List[Any],
                dualvln_raw_output: Optional[str] = None) -> str:
        """
        Generate answer for a prompt.

        In actual usage, dualvln_raw_output would come from running the model.
        For evaluation, we can pre-extract or simulate.

        Args:
            prompt: The question prompt
            images: List of images (not used by adapter)
            dualvln_raw_output: Pre-recorded DualVLN output log

        Returns:
            JSON-formatted answer
        """
        if dualvln_raw_output is None:
            # No output available - return invalid
            return json.dumps({"answer": "INVALID"})

        # Parse DualVLN output
        dv_output = self.adapter.parse_raw_output(dualvln_raw_output)

        # Determine question type and key from prompt
        q_key, q_type = self._extract_question_info(prompt)

        if q_key is None:
            return json.dumps({"answer": "INVALID"})

        # Extract person index from question key
        person_idx = 0
        if '_p' in q_key:
            try:
                person_idx = int(q_key.split('_p')[1].split('_')[0])
            except ValueError:
                pass

        # Get base question key (without person suffix)
        base_q_key = q_key
        if person_idx > 0:
            base_q_key = q_key.replace(f'_p{person_idx}', '')

        # Generate answer
        answer = self.adapter.answer_question(base_q_key, dv_output, person_idx)

        # Format
        return self.adapter.format_answer(answer, q_type)

    def _extract_question_info(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Extract question key and type from prompt text.

        Args:
            prompt: Full prompt text

        Returns:
            (question_key, question_type)
        """
        # Detect question type from prompt content
        q_type = "multiple_choice"
        if "Select all that apply" in prompt:
            q_type = "multiple_select"

        # Try to extract question key from CoT format or content
        # This is a heuristic based on prompt structure
        q_key = None

        # Check for known question patterns
        question_patterns = [
            ("robot is _____", "q_robot_moving_direction"),
            ("is _____ the robot", "q_person_spatial_position_begin"),
            ("At the end, person", "q_person_spatial_position_end"),
            ("ends up _____ the robot", "q_person_distance_change"),
            ("goal is ___ of the robot", "q_goal_position_begin"),
            ("At the end frame, the goal", "q_goal_position_end"),
            ("path in the way", "q_obstructing_path"),
            ("position in the way", "q_obstructing_end_position"),
            ("robot's movement affected", "q_robot_affected"),
            ("robot is most likely", "q_robot_action"),
            ("person's movement affected", "q_person_affected"),
            ("person is most likely", "q_person_action"),
            ("robot should ____", "q_robot_suggested_action"),
            ("person will most likely", "q_human_future_action_prediction"),
        ]

        prompt_lower = prompt.lower()
        for pattern, key in question_patterns:
            if pattern.lower() in prompt_lower:
                q_key = key
                break

        return q_key, q_type


def load_dualvln_output_from_file(filepath: str) -> str:
    """Load DualVLN raw output from file."""
    with open(filepath, 'r') as f:
        return f.read()


# Convenience function for evaluation integration
def create_dualvln_answer_mapper(image_height: int = 480,
                                  image_width: int = 640) -> DualVLNWrapper:
    """
    Create a DualVLN wrapper for use in evaluation.

    Usage in evaluate_vlm.py:
        if model_name == 'dualvln':
            wrapper = create_dualvln_answer_mapper()
            # Pre-load all DualVLN outputs for samples
            dualvln_outputs = load_all_dualvln_outputs(...)

            # In the loop:
            ans_raw = wrapper.generate(prompt, images,
                                      dualvln_outputs.get(sample_id))
    """
    adapter = DualVLNSpatialAdapter(image_height, image_width)
    return DualVLNWrapper(adapter)
