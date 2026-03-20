#!/usr/bin/env python
"""
把 scand_bag_extract 的提取结果包装成 LeRobot 风格的 episode：parquet + meta/episodes.jsonl。

目录结构:
  out_root/
    meta/
      episodes.jsonl
    data/
      chunk-000/
        episode_000000.parquet

parquet 列: timestamp_ns, frame_index, image_path, action.linear_x, action.linear_y,
            action.angular_z [, state.x, state.y, state.theta 若有 state.csv ]

依赖: pip install pandas pyarrow

用法:
  python scripts/extracted_to_lerobot.py \\
    --extracted-dir D:\\MyLab\\SCAND\\ahg2library_extracted \\
    --out-root D:\\MyLab\\SCAND\\scand_lerobot \\
    --episode-idx 0 \\
    --task "Go from AHG to Library."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_lerobot_episode(
    extracted_dir: str | Path,
    out_root: str | Path,
    episode_idx: int = 0,
    task: str = "Go from AHG to Library.",
    copy_frames: bool = False,
) -> None:
    extracted = Path(extracted_dir)
    out_root = Path(out_root)
    if not extracted.exists():
        raise FileNotFoundError(f"Extracted dir not found: {extracted}")

    actions_path = extracted / "actions.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"actions.csv not found in {extracted}")

    import pandas as pd

    actions = pd.read_csv(actions_path)
    actions["image_path"] = actions["frame_index"].apply(
        lambda i: f"frame_{int(i):06d}.jpg"
    )

    def frame_exists(p: str) -> bool:
        return (extracted / p).exists()

    actions = actions[actions["image_path"].apply(frame_exists)].reset_index(drop=True)
    if actions.empty:
        raise ValueError(f"No frames found in {extracted} (all missing?)")

    # 合并 state（若有 state.csv）
    state_path = extracted / "state.csv"
    if state_path.exists():
        state_df = pd.read_csv(state_path)
        use_cols = [c for c in ["frame_index", "timestamp_ns", "x", "y", "theta"] if c in state_df.columns]
        state_df = state_df[use_cols]
        if "frame_index" in state_df.columns:
            key = "frame_index"
            merge_cols = [c for c in ["x", "y", "theta"] if c in state_df.columns]
        else:
            key = "timestamp_ns"
            merge_cols = [c for c in ["x", "y", "theta"] if c in state_df.columns]
        if merge_cols:
            actions = actions.merge(
                state_df[[key] + merge_cols],
                on=key,
                how="left",
            )
        for col in ["x", "y", "theta"]:
            if col not in actions.columns:
                actions[col] = 0.0
    else:
        actions["x"] = actions["y"] = actions["theta"] = 0.0

    if copy_frames:
        import shutil
        chunk_name = f"chunk-{episode_idx // 1000:03d}"
        frames_dst = out_root / "frames" / chunk_name
        frames_dst.mkdir(parents=True, exist_ok=True)
        rel_prefix = f"frames/{chunk_name}"
        new_paths = []
        for _, row in actions.iterrows():
            src = extracted / row["image_path"]
            if src.exists():
                dst = frames_dst / row["image_path"]
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                    shutil.copy2(src, dst)
                new_paths.append(f"{rel_prefix}/{row['image_path']}")
            else:
                new_paths.append(row["image_path"])
        actions = actions.copy()
        actions["image_path"] = new_paths  # 相对 out_root 的路径，loader 用 out_root / image_path
    # 否则 image_path 保持为 frame_XXXXXX.jpg，loader 需用 extracted_dir 拼路径

    df = pd.DataFrame({
        "timestamp_ns": actions["timestamp_ns"].astype("int64"),
        "frame_index": actions["frame_index"].astype("int64"),
        "image_path": actions["image_path"].astype("string"),
        "action.linear_x": actions["linear_x"].astype("float32"),
        "action.linear_y": actions["linear_y"].astype("float32"),
        "action.angular_z": actions["angular_z"].astype("float32"),
        "state.x": actions["x"].astype("float32"),
        "state.y": actions["y"].astype("float32"),
        "state.theta": actions["theta"].astype("float32"),
    })

    chunk_dir = out_root / "data" / f"chunk-{episode_idx // 1000:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"episode_{episode_idx:06d}.parquet"
    df.to_parquet(parquet_path, index=False)
    print("Wrote:", parquet_path, "rows:", len(df))

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = meta_dir / "episodes.jsonl"
    ep = {
        "episode_index": episode_idx,
        "length": int(len(df)),
        "tasks": [task],
        "dialogs": [],
        "source": str(extracted.resolve()),
    }
    with open(episodes_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print("Updated:", episodes_path, "episode_index:", episode_idx, "length:", ep["length"])


def main():
    parser = argparse.ArgumentParser(
        description="Convert scand_bag_extract output to LeRobot-style episode (parquet + episodes.jsonl)."
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default=r"D:\MyLab\SCAND\ahg2library_extracted",
        help="Directory with frame_*.jpg and actions.csv (and optional state.csv).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"D:\MyLab\SCAND\scand_lerobot",
        help="Output root: meta/episodes.jsonl and data/chunk-XXX/episode_*.parquet.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=0,
        help="Episode index for this trajectory (default 0).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Go from AHG to Library.",
        help="Single instruction for this episode (written to tasks).",
    )
    parser.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy frame jpg into out_root/frames/chunk-XXX/ for self-contained dataset.",
    )
    args = parser.parse_args()

    try:
        build_lerobot_episode(
            extracted_dir=args.extracted_dir,
            out_root=args.out_root,
            episode_idx=args.episode_idx,
            task=args.task,
            copy_frames=args.copy_frames,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
