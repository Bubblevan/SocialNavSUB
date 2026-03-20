# -*- coding: utf-8 -*-
"""Aggregate Top-1 accuracy and PA for two experiment folders (no config)."""
import os
import json

REASONING_GROUPS = {
    "Spatial reasoning": [
        "q_person_spatial_position_begin", "q_person_spatial_position_end",
        "q_goal_position_begin", "q_goal_position_end", "q_obstructing_end_position"
    ],
    "Spatiotemporal reasoning": [
        "q_robot_moving_direction", "q_person_distance_change", "q_obstructing_path"
    ],
    "Social reasoning": [
        "q_robot_affected", "q_robot_action", "q_person_affected", "q_person_action",
        "q_robot_suggested_affected", "q_robot_suggested_action", "q_human_future_action_prediction"
    ]
}

def get_group(q_key):
    base = q_key.rsplit("_p", 1)[0] if "_p" in q_key else q_key
    if base.startswith("q_goal_position"):
        base = "q_goal_position_begin" if "begin" in q_key else "q_goal_position_end"
    for grp, prefixes in REASONING_GROUPS.items():
        for p in prefixes:
            if base == p or base.startswith(p + "_"):
                return grp
    return None

def aggregate_experiment(exp_path):
    correct_total = 0
    total_questions = 0
    by_group = {}
    n_samples = 0
    for name in sorted(os.listdir(exp_path)):
        sample_dir = os.path.join(exp_path, name)
        if not os.path.isdir(sample_dir):
            continue
        eval_fp = os.path.join(sample_dir, "evaluation.json")
        if not os.path.isfile(eval_fp):
            continue
        n_samples += 1
        with open(eval_fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q_key, v in data.items():
            if not isinstance(v, dict) or "top_1_accuracy" not in v:
                continue
            total_questions += 1
            if v["top_1_accuracy"] == 1.0:
                correct_total += 1
            grp = get_group(q_key)
            if grp:
                by_group.setdefault(grp, {"correct": 0, "total": 0})
                by_group[grp]["total"] += 1
                if v["top_1_accuracy"] == 1.0:
                    by_group[grp]["correct"] += 1
    return {
        "n_samples": n_samples,
        "total_correct": correct_total,
        "total_questions": total_questions,
        "top1_acc": correct_total / total_questions if total_questions else 0,
        "by_group": by_group,
    }

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    experiments = os.path.join(base, "experiments")
    gemini_path = os.path.join(experiments, "experiment_3_gemini-2.5-flash-lite_cot")
    gpt_path = os.path.join(experiments, "experiment_3_gpt-5-nano_cot")
    if not os.path.isdir(gemini_path) or not os.path.isdir(gpt_path):
        print("Missing experiment folders")
        return
    gemini = aggregate_experiment(gemini_path)
    gpt = aggregate_experiment(gpt_path)
    print("=" * 60)
    print("Final results (CoT, img_with_bev)")
    print("=" * 60)
    print(f"\n{'Model':<35} {'Samples':<10} {'Top-1 Acc':<12} {'Correct/Total'}")
    print("-" * 60)
    print(f"{'Gemini-2.5-flash-lite':<35} {gemini['n_samples']:<10} {gemini['top1_acc']:.4f}       {gemini['total_correct']}/{gemini['total_questions']}")
    print(f"{'GPT-5-nano':<35} {gpt['n_samples']:<10} {gpt['top1_acc']:.4f}       {gpt['total_correct']}/{gpt['total_questions']}")
    print("\nBy reasoning group (Top-1 accuracy):")
    print("-" * 60)
    for grp in ["Spatial reasoning", "Spatiotemporal reasoning", "Social reasoning"]:
        g_g = gemini["by_group"].get(grp, {"correct": 0, "total": 0})
        g_p = gpt["by_group"].get(grp, {"correct": 0, "total": 0})
        acc_g = g_g["correct"] / g_g["total"] if g_g["total"] else 0
        acc_p = g_p["correct"] / g_p["total"] if g_p["total"] else 0
        print(f"  {grp}:")
        print(f"    Gemini-2.5-flash-lite: {acc_g:.4f}  ({g_g['correct']}/{g_g['total']})")
        print(f"    GPT-5-nano:             {acc_p:.4f}  ({g_p['correct']}/{g_p['total']})")
    print("=" * 60)

if __name__ == "__main__":
    main()
