#!/usr/bin/env python
"""
从 SCAND 的 ROS bag 里列出 topic，或提取图像帧 + 控制量到本地目录。

支持:
- 图像: sensor_msgs/Image（原始）或 sensor_msgs/CompressedImage（压缩图）
- 控制: geometry_msgs/Twist(cmd_vel)、nav_msgs/Odometry(用 twist)、sensor_msgs/Joy(摇杆)

依赖（无需安装 ROS）:
    pip install rosbags numpy pillow

示例:
    # 列出所有 topic
    python scripts/scand_bag_extract.py --bag D:\\MyLab\\SCAND\\ahg2library.bag --list-topics

    # 你的 bag 里是压缩图 + 无 cmd_vel，用 odom 当控制量:
    python scripts/scand_bag_extract.py --bag D:\\MyLab\\SCAND\\ahg2library.bag \\
        --output-dir D:\\MyLab\\SCAND\\ahg2library_extracted \\
        --image-topic /camera/rgb/image_raw/compressed \\
        --cmd-topic /jackal_velocity_controller/odom
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from bisect import bisect_right
from pathlib import Path


def _yaw_from_loc_msg(msg) -> float | None:
    """从 localization 消息里取出 yaw/theta（rad）。支持 theta/yaw/heading/rot 或 pose.theta（Pose2Df）、四元数。"""
    theta = getattr(msg, "theta", None)
    if theta is not None:
        return float(theta)
    # amrl_msgs/msg/Localization2DMsg: theta 在 msg.pose 里 (Pose2Df)
    if hasattr(msg, "pose"):
        p = getattr(msg, "pose", None)
        if p is not None:
            theta = getattr(p, "theta", None)
            if theta is not None:
                return float(theta)
    for name in ("yaw", "heading", "rot", "rotation"):
        v = getattr(msg, name, None)
        if v is not None:
            return float(v)
    # 四元数 orientation (x,y,z,w) -> yaw
    o = getattr(msg, "orientation", None)
    if o is not None:
        x = getattr(o, "x", 0.0) or 0.0
        y = getattr(o, "y", 0.0) or 0.0
        z = getattr(o, "z", 0.0) or 0.0
        w = getattr(o, "w", 1.0) or 1.0
        # yaw from quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    p = getattr(msg, "pose", None)
    if p is not None:
        o = getattr(p, "orientation", None)
        if o is not None:
            x = getattr(o, "x", 0.0) or 0.0
            y = getattr(o, "y", 0.0) or 0.0
            z = getattr(o, "z", 0.0) or 0.0
            w = getattr(o, "w", 1.0) or 1.0
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="List topics or extract images + cmd_vel from a SCAND ROS bag."
    )
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Path to .bag file (ROS1) or bag directory (ROS2).",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="Only list topics and message types, then exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for extracted frames and actions.csv (required if not --list-topics).",
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default=None,
        help="Image topic to extract (e.g. /camera/rgb/image_rect_color).",
    )
    parser.add_argument(
        "--cmd-topic",
        type=str,
        default=None,
        help="Control topic: /cmd_vel (Twist), or /jackal_velocity_controller/odom (Odometry), or /bluetooth_teleop/joy (Joy). No default.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Save every N-th image (default: 1 = all).",
    )
    parser.add_argument(
        "--loc-topic",
        type=str,
        default=None,
        help="Optional: localization topic (e.g. /localization) for state.csv with x,y,theta.",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"Error: path does not exist: {bag_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print(
            "Error: rosbags not installed. Run: pip install rosbags",
            file=sys.stderr,
        )
        sys.exit(1)

    # 统一成列表（AnyReader 接受 path 列表）
    if bag_path.is_file():
        paths = [bag_path]
    else:
        paths = [bag_path]

    with AnyReader(paths) as reader:
        conns = list(reader.connections)
        if not conns:
            print("No connections in bag.", file=sys.stderr)
            sys.exit(1)

        if args.list_topics:
            print("Topics (name, type):")
            for c in sorted(conns, key=lambda x: x.topic):
                print(f"  {c.topic}  [{c.msgtype}]")
            return

        if not args.output_dir or not args.image_topic:
            print(
                "For extraction, pass --output-dir and --image-topic. Use --list-topics to see topic names.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.cmd_topic:
            print(
                "For extraction, pass --cmd-topic (e.g. /jackal_velocity_controller/odom or /cmd_vel).",
                file=sys.stderr,
            )
            sys.exit(1)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        image_conns = [c for c in conns if c.topic == args.image_topic]
        cmd_conns = [c for c in conns if c.topic == args.cmd_topic]

        if not image_conns:
            print(
                f"No connection for image topic: {args.image_topic}. Available:",
                [c.topic for c in conns],
                file=sys.stderr,
            )
            sys.exit(1)
        if not cmd_conns:
            print(
                f"No connection for cmd topic: {args.cmd_topic}. Available:",
                [c.topic for c in conns],
                file=sys.stderr,
            )
            sys.exit(1)

        # 收集控制量：(timestamp_ns, linear_x, linear_y, angular_z)
        cmd_list = []
        for conn, ts, raw in reader.messages(connections=cmd_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            mt = conn.msgtype
            if "Odometry" in mt:
                twist = getattr(msg, "twist", None) and getattr(msg.twist, "twist", None)
                if twist is not None:
                    lx = getattr(twist.linear, "x", 0.0) or 0.0
                    ly = getattr(twist.linear, "y", 0.0) or 0.0
                    az = getattr(twist.angular, "z", 0.0) or 0.0
                    cmd_list.append((ts, lx, ly, az))
            elif "Twist" in mt or "TwistStamped" in mt:
                if "Stamped" in mt:
                    twist = getattr(msg, "twist", msg)
                else:
                    twist = msg
                lx = getattr(getattr(twist, "linear", None), "x", 0.0) or 0.0
                ly = getattr(getattr(twist, "linear", None), "y", 0.0) or 0.0
                az = getattr(getattr(twist, "angular", None), "z", 0.0) or 0.0
                cmd_list.append((ts, lx, ly, az))
            elif "Joy" in mt:
                axes = getattr(msg, "axes", [])
                # 常见布局: axes[1] 前后, axes[0] 或 axes[3] 左右
                fwd = float(axes[1]) if len(axes) > 1 else 0.0
                turn = float(axes[0]) if len(axes) > 0 else (float(axes[3]) if len(axes) > 3 else 0.0)
                cmd_list.append((ts, fwd, 0.0, turn))
        cmd_list.sort(key=lambda x: x[0])

        # 可选：收集 /localization → (timestamp_ns, x, y, theta)，theta 为 yaw(rad)
        loc_list = []
        if args.loc_topic:
            loc_conns = [c for c in conns if c.topic == args.loc_topic]
            if loc_conns:
                for conn, ts, raw in reader.messages(connections=loc_conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if not loc_list:
                        print("LOC MSGTYPE:", conn.msgtype)
                        print("LOC MSG:", msg)
                    x = getattr(msg, "x", None)
                    if x is None and hasattr(msg, "pose"):
                        p = msg.pose
                        x = getattr(getattr(p, "position", p), "x", None)
                    y = getattr(msg, "y", None)
                    if y is None and hasattr(msg, "pose"):
                        p = msg.pose
                        y = getattr(getattr(p, "position", p), "y", None)
                    theta = _yaw_from_loc_msg(msg)
                    if x is not None and y is not None:
                        theta = theta if theta is not None else 0.0
                        loc_list.append((ts, float(x), float(y), float(theta)))
                loc_list.sort(key=lambda x: x[0])
                if loc_list:
                    print(f"Collected {len(loc_list)} localization messages from {args.loc_topic}")

        # 提取图像并记录时间戳
        image_conn = image_conns[0]
        frame_timestamps = []
        try:
            import numpy as np
            from PIL import Image as PILImage
        except ImportError:
            print("For image export install: pip install numpy pillow", file=sys.stderr)
            sys.exit(1)

        is_compressed = "CompressedImage" in image_conn.msgtype
        idx = 0
        for conn, ts, raw in reader.messages(connections=image_conns):
            if idx % args.downsample != 0:
                idx += 1
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            data = getattr(msg, "data", None)
            if data is None or len(data) == 0:
                idx += 1
                continue
            im = None
            if is_compressed:
                # sensor_msgs/CompressedImage: data 为 jpeg/png 等编码字节
                fmt = (getattr(msg, "format", None) or "jpeg").lower()
                try:
                    buf = io.BytesIO(bytes(data))
                    im = PILImage.open(buf).convert("RGB")
                except Exception:
                    idx += 1
                    continue
            else:
                h = getattr(msg, "height", 0)
                w = getattr(msg, "width", 0)
                step = getattr(msg, "step", w * 3)
                if h and w and step >= w * 3:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    if arr.size >= h * step:
                        arr = arr[: h * step].reshape((h, step))
                        arr = arr[:, : w * 3].reshape((h, w, 3))
                        if arr.shape[2] == 3:
                            arr = arr[:, :, ::-1]  # BGR -> RGB
                        im = PILImage.fromarray(arr)
            if im is not None:
                out_path = out_dir / f"frame_{idx:06d}.jpg"
                im.save(out_path, quality=95)
                frame_timestamps.append((ts, idx))
            idx += 1

        # 用二分对齐：每帧取最后一个 t<=ts 的 cmd/loc，O(N log M)
        ts_cmd = [t for t, *_ in cmd_list]
        ts_loc = [t for t, *_ in loc_list] if loc_list else []

        # 把每帧时间戳与最近的 cmd 对齐，写 actions.csv
        actions_path = out_dir / "actions.csv"
        with open(actions_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame_index", "timestamp_ns", "linear_x", "linear_y", "angular_z"]
            )
            for ts, frame_idx in frame_timestamps:
                linear_x = linear_y = angular_z = 0.0
                if ts_cmd:
                    j = bisect_right(ts_cmd, ts) - 1
                    if j >= 0:
                        _, linear_x, linear_y, angular_z = cmd_list[j]
                writer.writerow(
                    [frame_idx, ts, linear_x, linear_y, angular_z]
                )

        # 可选：写 state.csv（每帧二分对齐到最近一条 localization）
        if loc_list and ts_loc:
            state_path = out_dir / "state.csv"
            with open(state_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "timestamp_ns", "x", "y", "theta"])
                for ts, frame_idx in frame_timestamps:
                    x = y = theta = 0.0
                    j = bisect_right(ts_loc, ts) - 1
                    if j >= 0:
                        _, x, y, theta = loc_list[j]
                    writer.writerow([frame_idx, ts, x, y, theta])
            print(f"Wrote {state_path} ({len(loc_list)} loc samples)")

        print(
            f"Extracted {len(frame_timestamps)} frames to {out_dir}, actions to {actions_path}"
        )


if __name__ == "__main__":
    main()
