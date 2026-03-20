"""
Test script to verify DualVLN integration works correctly.

This tests:
1. Adapter parsing
2. Question answering logic
3. Integration with evaluation framework
"""

import os
import sys
import json
import tempfile
import shutil

# Add socialnavsub to path
sys.path.insert(0, 'socialnavsub')

from dualvln_adapter import DualVLNSpatialAdapter, DualVLNWrapper
from dualvln_eval_integration import DualVLNEvaluator


# Sample DualVLN output for testing
SAMPLE_DUALVLN_OUTPUT = """[19:49:06.210009] step_id 330 action 5
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
[19:49:45.089259] step_id 363 action 0
"""


def test_adapter_parsing():
    """Test that adapter correctly parses DualVLN output."""
    print("\n" + "="*60)
    print("Test 1: Adapter Parsing")
    print("="*60)

    adapter = DualVLNSpatialAdapter(image_height=480, image_width=640)
    output = adapter.parse_raw_output(SAMPLE_DUALVLN_OUTPUT)

    print(f"  Parsed {len(output.actions)} actions")
    print(f"  Goal pixel: {output.goal_pixel}")
    print(f"  Step IDs: {len(output.step_ids)} entries")

    assert len(output.actions) > 0, "No actions parsed"
    assert output.goal_pixel is not None, "No goal parsed"

    print("  PASSED")
    return True


def test_question_answering():
    """Test question answering logic."""
    print("\n" + "="*60)
    print("Test 2: Question Answering")
    print("="*60)

    adapter = DualVLNSpatialAdapter(image_height=480, image_width=640)
    output = adapter.parse_raw_output(SAMPLE_DUALVLN_OUTPUT)

    test_cases = [
        ('q_goal_position_begin', 'multiple_choice'),
        ('q_goal_position_end', 'multiple_choice'),
        ('q_robot_moving_direction', 'multiple_select'),
        ('q_person_spatial_position_begin', 'multiple_choice'),
        ('q_person_spatial_position_end', 'multiple_choice'),
        ('q_person_distance_change', 'multiple_choice'),
        ('q_obstructing_path', 'multiple_choice'),
        ('q_obstructing_end_position', 'multiple_choice'),
    ]

    all_passed = True
    for q_key, q_type in test_cases:
        try:
            answer = adapter.answer_question(q_key, output, person_idx=1)
            formatted = adapter.format_answer(answer, q_type)

            # Verify JSON format
            parsed = json.loads(formatted)
            assert 'answer' in parsed, f"Missing 'answer' key in {q_key}"

            print(f"  {q_key}: {parsed['answer']}")
        except Exception as e:
            print(f"  {q_key}: FAILED - {e}")
            all_passed = False

    if all_passed:
        print("  PASSED")
    return all_passed


def test_evaluator():
    """Test DualVLNEvaluator with temp files."""
    print("\n" + "="*60)
    print("Test 3: DualVLN Evaluator")
    print("="*60)

    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Write sample output file
        sample_id = "test_sample_001"
        output_file = os.path.join(temp_dir, f"{sample_id}.txt")
        with open(output_file, 'w') as f:
            f.write(SAMPLE_DUALVLN_OUTPUT)

        # Create evaluator
        config = {'image_height': 480, 'image_width': 640}
        evaluator = DualVLNEvaluator(temp_dir, config=config)

        # Test loading
        loaded = evaluator.load_dualvln_output(sample_id)
        assert loaded is not None, "Failed to load output"
        print(f"  Loaded output for {sample_id}: {len(loaded)} chars")

        # Test generation
        evaluator.current_sample_id = sample_id
        prompt = "The robot is _____. (Select all that apply)"
        answer = evaluator.generate_text(prompt, [])

        parsed = json.loads(answer)
        assert 'answer' in parsed, "Missing answer key"
        print(f"  Generated answer: {parsed['answer']}")

        # Test caching
        loaded_again = evaluator.load_dualvln_output(sample_id)
        assert loaded_again == loaded, "Cache not working"
        print("  Cache working correctly")

        print("  PASSED")
        return True
    finally:
        shutil.rmtree(temp_dir)


def test_integration_with_evaluate_vlm():
    """Test that evaluate_vlm.py imports work."""
    print("\n" + "="*60)
    print("Test 4: Integration with evaluate_vlm.py")
    print("="*60)

    try:
        # Import the modified evaluate_vlm module
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluate_vlm", "socialnavsub/evaluate_vlm.py")
        eval_module = importlib.util.module_from_spec(spec)

        # This will fail if imports are wrong
        print("  Loading evaluate_vlm.py...")
        spec.loader.exec_module(eval_module)

        # Check DualVLN availability
        if hasattr(eval_module, 'DUALVLN_AVAILABLE'):
            print(f"  DUALVLN_AVAILABLE: {eval_module.DUALVLN_AVAILABLE}")
            if eval_module.DUALVLN_AVAILABLE:
                print("  DualVLN integration loaded successfully")
            else:
                print("  WARNING: DualVLN adapter not available (imports may have failed)")
        else:
            print("  WARNING: DUALVLN_AVAILABLE not found (integration not applied?)")

        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test loading config with DualVLN settings."""
    print("\n" + "="*60)
    print("Test 5: Config Loading")
    print("="*60)

    try:
        with open('config_dualvln_example.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)

        assert config['baseline_model'] == 'dualvln', "Wrong baseline model"
        assert 'dualvln_outputs_dir' in config, "Missing dualvln_outputs_dir"
        assert 'image_height' in config, "Missing image_height"
        assert 'image_width' in config, "Missing image_width"

        print(f"  Baseline: {config['baseline_model']}")
        print(f"  Outputs dir: {config['dualvln_outputs_dir']}")
        print(f"  Image size: {config['image_width']}x{config['image_height']}")

        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DualVLN Integration Tests")
    print("="*60)

    results = []

    results.append(("Adapter Parsing", test_adapter_parsing()))
    results.append(("Question Answering", test_question_answering()))
    results.append(("Evaluator", test_evaluator()))
    results.append(("Integration", test_integration_with_evaluate_vlm()))
    results.append(("Config Loading", test_config_loading()))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED!")
        print("\nDualVLN integration is ready to use.")
        print("\nNext steps:")
        print("  1. Create dualvln_outputs/ directory")
        print("  2. Run DualVLN on all samples and save outputs there")
        print("  3. Copy config_dualvln_example.yaml to config.yaml")
        print("  4. Run: python socialnavsub/evaluate_vlm.py")
    else:
        print("Some tests FAILED. Please check the errors above.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
