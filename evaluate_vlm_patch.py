"""
Patch file showing exact modifications needed to evaluate_vlm.py for DualVLN support.

Apply these changes manually or use as reference.
"""

# =============================================================================
# CHANGE 1: Add imports (after line 27)
# =============================================================================

# Add these lines after "from utils import (" block:
DUALVLN_IMPORTS = '''
# DualVLN support
try:
    from dualvln_eval_integration import DualVLNEvaluator
    DUALVLN_AVAILABLE = True
except ImportError:
    DUALVLN_AVAILABLE = False
'''

# =============================================================================
# CHANGE 2: Modify model initialization (around line 87)
# =============================================================================

# REPLACE this block:
ORIGINAL_MODEL_LOAD = '''
    model = load_model_class(baseline_model, model_to_api_key, config=config)
'''

# WITH this block:
PATCHED_MODEL_LOAD = '''
    if baseline_model.startswith('dualvln'):
        if not DUALVLN_AVAILABLE:
            raise ImportError("DualVLN support requested but adapter not available. "
                            "Please ensure dualvln_adapter.py and dualvln_eval_integration.py are present.")
        dualvln_outputs_dir = config.get('dualvln_outputs_dir', 'dualvln_outputs/')
        if not os.path.exists(dualvln_outputs_dir):
            raise FileNotFoundError(f"DualVLN outputs directory not found: {dualvln_outputs_dir}. "
                                   f"Please run DualVLN on all samples and save outputs there.")
        model = DualVLNEvaluator(dualvln_outputs_dir, config=config)
        logger.info(f"Initialized DualVLN evaluator with outputs from {dualvln_outputs_dir}")
    else:
        model = load_model_class(baseline_model, model_to_api_key, config=config)
'''

# =============================================================================
# CHANGE 3: Pass sample_id to model (around line 160, inside the loop)
# =============================================================================

# BEFORE this line:
#   sample_id = os.path.basename(s_dir)

# ADD this block:
SAMPLE_ID_PATCH = '''
        sample_id = os.path.basename(s_dir)

        # Set sample_id for DualVLN evaluator
        if baseline_model.startswith('dualvln'):
            model.current_sample_id = sample_id
'''

# =============================================================================
# CHANGE 4: Handle DualVLN in model.generate_text() (around line 214)
# =============================================================================

# The existing code:
#   ans_raw = model.generate_text(prompt, images)

# Should work as-is since DualVLNEvaluator has the same interface.
# But we need to ensure sample_id is set before the loop:

FULL_INTEGRATION_CODE = '''
# =============================================================================
# DUALVLN INTEGRATION - Add these snippets to evaluate_vlm.py
# =============================================================================

# 1. ADD TO IMPORTS (after line 27):
try:
    from dualvln_eval_integration import DualVLNEvaluator
    DUALVLN_AVAILABLE = True
except ImportError:
    DUALVLN_AVAILABLE = False


# 2. REPLACE model initialization (around line 87):
# Original:
#     model = load_model_class(baseline_model, model_to_api_key, config=config)

# New:
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


# 3. ADD after sample_id extraction (around line 160):
# Original:
#         sample_id = os.path.basename(s_dir)

# New:
        sample_id = os.path.basename(s_dir)
        if baseline_model.startswith('dualvln'):
            model.current_sample_id = sample_id


# 4. The existing model.generate_text() call should work as-is.
'''

if __name__ == '__main__':
    print(FULL_INTEGRATION_CODE)
