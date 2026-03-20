# InternVLA-N1-DualVLN SFT 与数据构建指南

本文说明如何对 `checkpoints/InternVLA-N1-DualVLN` 做 SFT（监督微调），以及如何构建/接入训练数据。依据 [InternNav 官方训练文档](https://internrobotics.github.io/user_guide/internnav/tutorials/training.html) 与本地 InternNav 仓库（`D:/MyLab/SAN/InternNav`）代码整理。

---

## 一、如何 SFT InternVLA-N1-DualVLN

### 1.1 两种训练方式

InternNav 支持两种与 DualVLN 相关的训练流程：

| 方式 | 脚本 | 说明 |
|------|------|------|
| **只训 System2** | `train_system2.sh` | 从 Qwen2.5-VL 或已有 System2 权重训「像素目标」预测，不训 System1 |
| **联合训 Dual 系统** | `train_dual_system.sh` | 从 **已训好的 System2** 权重出发，冻结 System2，只训 System1（NextDiT/DualVLN 或 NavDP） |

要对 **InternVLA-N1-DualVLN** 做 SFT，有两种常见用法：

- **在 DualVLN 上继续训（推荐）**：用 `train_dual_system.sh`，把 `system2_ckpt` 指向你已有的 **System2** 权重（若没有，可先用官方 [InternVLA-N1-System2](https://huggingface.co/InternRobotics/InternVLA-N1-System2)），`model_name_or_path` 也可直接改为 **DualVLN 的 checkpoint**（见下）。
- **从零只训 System2 再联合**：先跑 `train_system2.sh` 得到 System2，再跑 `train_dual_system.sh` 用该 System2 做 `system2_ckpt`。

下面以「用现有 DualVLN 做 SFT」为例。

### 1.2 用现有 DualVLN checkpoint 做 SFT（联合训练脚本）

在 InternNav 仓库下（例如 `D:/MyLab/SAN/InternNav`）：

1. **复制并修改联合训练脚本**  
   脚本位置以你本地 InternNav 仓库为准（官方 tutorial 为 `scripts/train/base_train/qwenvl_train/`，部分分支为 `scripts/train/qwenvl_train/`）：

```bash
cd D:/MyLab/SAN/InternNav
# 若存在 base_train 子目录则用下一行，否则用 scripts/train/qwenvl_train/
cp scripts/train/base_train/qwenvl_train/train_dual_system.sh scripts/train/base_train/qwenvl_train/train_dual_system_sft.sh
```

2. **编辑 `train_dual_system_sft.sh`**，关键修改：

- **加载 DualVLN 权重**（你本地的 System2 或完整 DualVLN 均可）：

```bash
# 用你本地的 DualVLN 作为起点（或 System2）
system2_ckpt=D:/MyLab/SocialNavSUB/checkpoints/InternVLA-N1-DualVLN
```

- **数据**：`vln_datasets` 改为你准备好的数据集名（见第二节）。

```bash
# 例如只用 r2r 一个设定，或你自定义的 scand_125cm_0_30
vln_datasets=r2r_125cm_0_30%30
```

- **输出目录**：

```bash
run_name=InternVLA-N1-DualVLN-SFT
output_dir=checkpoints/${run_name}
```

- **System1 类型**（DualVLN 对应 nextdit_async）：

```bash
system1=nextdit_async
```

3. **单机多卡启动（示例，非 sbatch）**

若不用 SLURM，可在 InternNav 根目录用 `torchrun` 直接跑：

```bash
cd D:/MyLab/SAN/InternNav
torchrun --nproc_per_node=8 internnav/trainer/internvla_n1_trainer.py \
  --model_name_or_path "D:/MyLab/SocialNavSUB/checkpoints/InternVLA-N1-DualVLN" \
  --vln_dataset_use "r2r_125cm_0_30" \
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \
  --system1 nextdit_async \
  --pixel_goal_only True \
  --output_dir checkpoints/InternVLA-N1-DualVLN-SFT \
  --num_train_epochs 3.0 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-4 \
  --bf16 \
  --num_history 8 --resize_h 384 --resize_w 384 \
  --predict_step_num 32 \
  --save_steps 5000 --logging_steps 1 \
  --gradient_checkpointing True
```

按需补全 `train_dual_system.sh` 里的其他参数（如 deepspeed、max_pixels 等）。

### 1.3 官方文档参考

- [Training (Quick Start)](https://internrobotics.github.io/user_guide/internnav/quick_start/training.html)  
- [Training (Tutorial)](https://internrobotics.github.io/user_guide/internnav/tutorials/training.html)  

联合训练时通常 **冻结 System2**（`tune_mm_vision/mlp/llm=False`），只训 System1；若你希望一起微调 System2，可把上述三个改为 `True` 并适当调小学习率。

---

## 二、数据要怎么构建？LeRobot 与 InternVLA-N1 字段

下面分两层：**LeRobotDataset 通用结构**（按官方 Dataset 页）和 **InternVLA-N1 训练所需的额外字段**（pixel goal / waypoint 等），避免把「任务特定字段」当成 LeRobot 的硬性规范。

### 2.1 LeRobotDataset 通用结构（按官方 Dataset 页）

InternNav 采用 **LeRobotDataset** 统一组织 videos / instructions / actions / metadata；官方 Dataset 页描述的是「通用目录与元数据结构」，而非某一种任务的必填列。

- **通用目录** 大致为：  
  `meta/`（episode 与任务元数据）、`data/`（按 chunk 的 parquet）、`videos/`（可选，按 chunk 的 mp4 或帧）。  
- **meta/episodes.jsonl**：每行一个 episode，例如  
  `{"episode_index": 0, "tasks": ["Turn left and go to the kitchen."], "length": 45, "dialogs": []}`。  
  `episode_index` 与 parquet 文件名对应，`tasks` 为指令，`length` 为步数。  
- **官方 Dataset 页** 还提到 **tasks.jsonl** 作为高层任务描述文件；若 loader 或工具链依赖 tasks.jsonl，需按官方说明提供。本仓库当前流程用 **episodes.jsonl 的 `tasks` 字段** 即可；若要完全贴近官方 metadata 组织，可另行增加 tasks.jsonl 及其字段说明。

**（可选）按 scene 切目录**：例如 `traj_data/r2r/<scene_id>/` 下再放 `meta/`、`data/`、`videos/`，是常见组织方式，但 **并非 LeRobot 的强制规范**；具体以你用的 loader 和 InternNav 代码里的 `data_dict` 约定为准。

### 2.2 InternVLA-N1（pixel-goal / dual system）训练所需额外字段

若要跑 **InternVLA-N1 System2 或 Joint 训练**（pixel goal、waypoint 等），在 LeRobot 通用结构之上，parquet 还需提供 **InternVLA-N1 训练管线** 所要求的列（取决于 `internvla_n1_lerobot_dataset.py` 的读取逻辑）：

- `action`：每步动作（与 `length` 一致）。  
- `pose.{setting}`：`setting = {height}cm_{pitch_2}deg`，例如 `125cm_30deg`。  
- `goal.{setting}`：该步的 pixel goal。  
- `relative_goal_frame_id.{setting}`：与 goal 对应的相对帧 id。

例如 `r2r_125cm_0_30` 会用到 `125cm_30deg` 的 pose/goal/relative_goal_frame_id。  
官方 Joint training 文档中 `pixel_goal_only=True` 表示该阶段不要求 turn/stop 等动作，**训练目标是 pixel goal / waypoints**，而不是简单 BC（视觉→连续速度）。因此「能直接接 InternVLA-N1」的数据需要在 parquet 里具备上述字段；仅有 image + 连续 action 的 BC 版 LeRobot 需再对齐或补全这些监督后才可接入。

**数据集名与路径**（InternVLA-N1 代码约定）：`data_dict` 里用 `{dataset}_{height}cm_{pitch1}_{pitch2}`（如 `r2r_125cm_0_30`），`data_path` 指向对应根目录（如 `traj_data/r2r`）；训练时工作目录一般为 InternNav 根目录，也可写绝对路径。

### 2.3 官方数据来源

- **InternData-N1**：  
  [Hugging Face - InternRobotics/InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1)  
  按官方说明下载并放到 `traj_data/r2r`、`traj_data/rxr`、`traj_data/scalevln` 等即可。

- **VLN-CE / R2R / RxR 等**：  
  若你有 VLN-CE 等原始数据，可用 InternNav 自带的 **vlnce2lerobot** 转成 LeRobot（见 `scripts/dataset_converters/vlnce2lerobot.py`），再按 2.1 结构组织并写入 `data_dict`。

---

## 三、SCAND 能不能用？

**SCAND**（Socially Compliant Navigation Dataset）是 **社交合规导航** 的机器人轨迹数据（ROSBAG + 多传感器、操纵杆控制等），不是「视觉–语言导航」的指令轨迹数据：

- 没有与每一步绑定的 **自然语言指令**（如 “Go to the kitchen”）；  
- 没有 InternNav 需要的 **pixel goal / relative_goal_frame_id** 等字段；  
- 格式是 ROSBAG，不是 LeRobot 的 `meta/episodes.jsonl` + parquet。

因此：

- **直接当 VLN 训练数据用**：不行，格式和语义都不匹配。  
- **要用 SCAND 做 SFT**：需要你自己写 **转换脚本**，把 SCAND 转成上述 LeRobot 结构：
  1. 从 ROSBAG 抽轨迹、图像、控制量。  
  2. 为每条轨迹 **构造或标注** 一句指令（例如用模板或事后标注）。  
  3. 在仿真或离线里算好 **pixel goal** 和 **relative_goal_frame_id**，写入 parquet。  
  4. 按 2.1 的 LeRobot 结构组织 `meta/episodes.jsonl` 和 `data/.../episode_*.parquet`（可选按 scene 分子目录）。  
  5. 在 `internvla_n1_lerobot_dataset.py` 的 `data_dict` 里增加一项，例如 `scand_125cm_0_30`，`data_path` 指向你导出的 `traj_data/scand`（或绝对路径），并在 `vln_datasets` 里写上 `scand_125cm_0_30`。

这样 SCAND 才能以「自建 VLN 风格数据」的形式参与 InternVLA-N1-DualVLN 的 SFT。

---

## 四、自定义数据集接入步骤小结

1. **按 LeRobot 结构组织数据**  
   - `traj_data/<你的数据集名>/` 下（可选按 scene_id 再分子目录）放 `meta/episodes.jsonl`、`data/chunk-XXX/episode_XXXXXX.parquet`，可选 `videos/`。  
   - 若 **对接 InternVLA-N1** 训练，parquet 还需含：`action`、`pose.{height}cm_{pitch}deg`、`goal.{setting}`、`relative_goal_frame_id.{setting}`（见 2.2）；若只做 BC/模仿学习，可仅含 obs + 连续 action。  

2. **在 `data_dict` 里注册**  
   编辑 `D:/MyLab/SAN/InternNav/internnav/dataset/internvla_n1_lerobot_dataset.py`，在 `data_dict` 中增加一项，例如：

```python
MY_SCAND_125CM_0_30 = {
    "data_path": "traj_data/scand",   # 或 "D:/data/scand_lerobot"
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}
data_dict["scand_125cm_0_30"] = MY_SCAND_125CM_0_30
```

3. **训练时指定**  
   `--vln_dataset_use "scand_125cm_0_30"` 或与现有数据集组合：`r2r_125cm_0_30,scand_125cm_0_30`。

4. **采样比例**  
   若只想用 30%：`scand_125cm_0_30%30`（代码里会解析 `%30` 为 30% 采样）。

---

## 五、参考文件位置（InternNav 仓库）

- 训练入口：`internnav/trainer/internvla_n1_trainer.py`  
- 数据与数据集名：`internnav/dataset/internvla_n1_lerobot_dataset.py`（`data_dict`、`get_annotations_from_lerobot_data`）  
- **训练脚本路径以仓库版本为准**：官方 Training tutorial 示例为 `scripts/train/base_train/qwenvl_train/train_system2.sh` 与 `train_dual_system.sh`；有些分支或本地克隆可能在 `scripts/train/qwenvl_train/`（无 `base_train`）。请以本地实际路径为准，复制粘贴前先确认目录是否存在。  
- VLN→LeRobot 转换：`scripts/dataset_converters/vlnce2lerobot.py`  

按上述步骤即可在本地对 `checkpoints/InternVLA-N1-DualVLN` 做 SFT，并自行构建或接入 LeRobot 格式数据；SCAND 需先转为该格式并补全指令与 pixel goal 后再用。

---

## 五.2 InternData-N1 里 VLN_CE / VLN_PE / VLN_N1 分别训练什么？和两系统、联合训练的关系

- **InternData-N1** 是官方整理的「统一数据集」，里面包含三个子集，都既有 **任务定义** 也有 **轨迹数据**（LeRobot 格式的 traj_data）：
  - **VLN-CE**（Continuous Environments）：约 170K+ episodes，对应 R2R、RxR 等。**raw_data** 是原始 json（instruction + 路径/起点终点），**traj_data** 是转成 LeRobot 的轨迹（含 pose、goal、action 等）。
  - **VLN-PE**（Physically Realistic）：8K+ episodes，物理更真实的仿真。同样有 raw_data + traj_data，在代码里常和 VLN-CE 一起按「r2r / rxr」等名字用。
  - **VLN-N1**：660K+ instructions、210K+ videos，合成数据，专门为 InternVLA-N1 准备。

- **训练阶段和数据的对应关系**（按官方脚本与 `data_dict`）：
  - **只训 System2**（`train_system2.sh`）：用的是 **r2r、rxr** 的多种 height/pitch（如 `r2r_125cm_0_30`），数据来自 **VLN-CE / VLN-PE** 的 **traj_data**（LeRobot）。训练目标是「语言+视觉 → 像素目标 / 高层决策」。
  - **联合训 Dual 系统**（`train_dual_system.sh`）：从已训好的 System2 出发，**冻结 System2**，只训 System1（NextDiT/NavDP）。用的仍是 **r2r、rxr、scalevln** 等 traj_data（其中 scalevln 对应 **VLN-N1** 或同类合成数据）。即 **VLN-CE、VLN-PE、VLN-N1 都会参与**：CE/PE 提供 r2r/rxr 轨迹，N1 提供 scalevln 或单独 vln_n1 路径。

- **R2R 和 InternData 的关系**：R2R（Room-to-Room）是 **任务/基准名**，原始就是 instruction + path 的 json。InternData-N1 把 R2R/RxR 等收进 **VLN-CE**，既保留 **raw_data**（任务 json）也提供 **traj_data**（LeRobot 轨迹）。所以：
  - **训练**用 **traj_data**（轨迹 + 图像 + action + pose + goal）；
  - **评估**用 **raw_data**（任务列表：哪个 split、哪条 instruction、哪个场景）。

---

## 五.3 评估时用不用 InternData？Training 和 Evaluate 的数据区别

- 运行 `python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py` 时，配置会链到 **`scripts/eval/configs/vln_r2r.yaml`**，其中明确写了：
  - `data_path: data/vln_ce/raw_data/r2r/{split}/{split}.json.gz`
  - `scenes_dir: data/scene_data/mp3d_ce`
  也就是说 **评估会用到 InternData 风格路径**：`data/vln_ce/raw_data/` 提供 R2R 的任务定义（val_unseen 等），`data/scene_data/mp3d_ce` 提供场景网格。若你没把 InternData 下到 `data/` 或做符号链接，评估会报错找不到文件。

- **小结**：
  - **Training**：用 InternData 的 **traj_data**（LeRobot 轨迹），数据集名如 r2r_125cm_0_30、scalevln 等。
  - **Evaluate**：用 InternData 的 **raw_data**（R2R/RxR 等 json） + **scene_data**（mp3d_ce 等）。所以 **InternData 既是训练数据也是评估任务数据**；R2R 是基准名，具体数据放在 InternData 的 vln_ce/raw_data 和 vln_ce/traj_data 里。

---

## 五.4 SCAND 混进训练做「极小部分 BC」：更适合 SFT System2 还是 Joint？

- 你当前从 rosbag 抽出来的 SCAND，经 `extracted_to_lerobot.py` 得到的是 **BC 版 LeRobot**：只有 `image_path`、`action.linear_x/angular_z`、`state.x/y/theta`，**没有** InternVLA-N1 需要的 `pose.{setting}`、`goal.{setting}`、`relative_goal_frame_id.{setting}`。
- InternNav 的 `get_annotations_from_lerobot_data`（见 `internvla_n1_lerobot_dataset.py`）会读 parquet 里的这些列；若缺失，只会打 warning 并填默认值，**没有有效的 pixel goal 监督**，相当于这些样本对 System2/System1 的 goal 预测没有帮助，甚至可能干扰。

- **结论与建议**：
  - **在补全 pixel goal（及 pose、relative_goal_frame_id）之前**：当前 SCAND **两个阶段都不适合直接混入**。要混入 InternVLA-N1 的 SFT，需要先按第二节 2.2 补全上述列（例如用 state 在 2D 地图上算 goal 像素，或离线生成再写回 parquet）。
  - **补全之后**：更推荐混进 **Joint 阶段**（`train_dual_system.sh`），用较小比例（如 `scand_125cm_0_30%5`）。原因：Joint 是「在已有 System2 上训 System1」、学轨迹/waypoint；真实场景的少量数据对 System1 的 domain 适应更有用；System2 已用大量 VLN 训好，少量 SCAND 放在 Joint 里即可。
  - 若暂时**不补 goal**：只能当「纯 BC」用在别的 pipeline（例如单独 NavDP 或简单视觉→速度的 BC），不能直接参与 InternVLA-N1 的 System2/Joint SFT。

---

## 六、手头只有一个 SCAND sample（.bag + .mp4）怎么上手

你有一个 sample 时，例如：

- **ahg2library.bag**：真实数据（相机、odom、cmd_vel 等都在这里）
- **ahg2library.mp4**：只是预览用的录像，**不能代替 .bag** 做训练；要抽帧、对齐控制量，必须用 .bag。

元数据（你表格里的那一行）可以拿来 **构造一条语言指令**，例如：

- Start=AHG, End=Library, Tags=Against Traffic  
- → 指令：`Navigate from AHG to Library, against traffic.`  
- 或简写：`From AHG to Library.`

### 6.1 推荐流程（四步）

1. **看 bag 里有什么**  
   用本仓库提供的脚本列出所有 topic（见下），确认相机 topic 和速度/定位 topic。

2. **从 bag 里抽出「帧 + 控制」**  
   脚本把 image topic 导出为帧图，把 odom（或 cmd_vel/joy）按时间对齐到每一帧，得到 `actions.csv`。可选加 `--loc-topic /localization` 得到 `state.csv`（x, y, theta）。

3. **包装成 LeRobot 目录（parquet + meta）**  
   用 **`scripts/extracted_to_lerobot.py`** 把「提取目录」转成 LeRobot 风格：`out_root/meta/episodes.jsonl` + `out_root/data/chunk-XXX/episode_XXXXXX.parquet`，每行一帧，列含 `image_path`、`action.linear_x/angular_z`、可选 `state.x/y/theta`。这样先做成「视觉 + 连续控制」的 BC 数据集，后续再加 instruction / pixel_goal 接 InternNav。

4. **再往 InternNav 的 VLN 格式靠拢（可选）**  
   用元数据写 instruction、离散化 action、补 pose/goal 占位，在 `data_dict` 里注册为 `scand_125cm_0_30` 等（见第二节）。

### 6.2 脚本用法（列出 topic + 提取帧与 cmd_vel）

在项目根目录下已提供脚本 **`scripts/scand_bag_extract.py`**（见下一节），依赖纯 Python 的 **rosbags**（无需安装 ROS）：

```bash
pip install rosbags
```

- **只列出 topic**（先确认再提取）：

```bash
python scripts/scand_bag_extract.py --bag D:\MyLab\SCAND\ahg2library.bag --list-topics
```

- **提取帧 + 控制量**：你的 bag 里是 **压缩图**（`/camera/rgb/image_raw/compressed`）且**没有 `/cmd_vel`**，可用 **odom** 当控制量（实际速度）：

```bash
python scripts/scand_bag_extract.py --bag D:\MyLab\SCAND\ahg2library.bag --output-dir D:\MyLab\SCAND\ahg2library_extracted --image-topic /camera/rgb/image_raw/compressed --cmd-topic /jackal_velocity_controller/odom
```

若想用摇杆意图代替实际速度，可用 `--cmd-topic /bluetooth_teleop/joy`。

- **可选：同时抽 2D 定位**（便于后续切轨迹、算里程、做 goal 监督）：加 `--loc-topic /localization`，会多输出 **state.csv**（`frame_index, timestamp_ns, x, y, theta`）。若 bag 里是 `amrl_msgs/msg/Localization2DMsg`，脚本会尝试读 `x,y,theta` 或 `pose.position.x/y`、`pose.orientation.z`；若字段对不上，可先 `--list-topics` 后自己看一条 msg 再改脚本。

输出目录里会有帧图、`actions.csv`，以及可选的 `state.csv`。下一步用 **6.4** 的 `extracted_to_lerobot.py` 打成 LeRobot 的 parquet + episodes.jsonl。

### 6.3 提取结果长什么样、有什么用？

脚本跑完后你会得到：

- **一万多张图**：`frame_000000.jpg` … `frame_010518.jpg`（约 30 fps × 350 秒 ≈ 10519 帧，和你这条轨迹的时长一致）。
- **actions.csv**：每一行对应一帧，列为  
  `frame_index, timestamp_ns, linear_x, linear_y, angular_z`。

**各列含义：**

| 列 | 含义 | 典型值 |
|----|------|--------|
| **linear_x** | 机体前进速度 (m/s) | 0 ≈ 静止，0.5～1.5 ≈ 正常前进 |
| **linear_y** | 机体横向速度 (m/s)，差速底盘多为 0 | 0 |
| **angular_z** | 绕竖直轴角速度 (rad/s)，左转为正 | 小正/负值 ≈ 微调方向，0 ≈ 直行 |

这些量的用途可以概括为：

1. **做离散动作（给 LeRobot / InternVLA 用）**  
   InternNav 的 LeRobot 里每步是一个**离散 action**（如 0=停、1=前、2=左转、3=右转）。你可以用 `linear_x`、`angular_z` 做简单规则离散化，例如：
   - 速度很小 → action = 0（停）；
   - 主要前进、角速度接近 0 → 1（前）；
   - 角速度明显为正/负 → 2/3（左/右）。  
   这样每一帧就对应一个整数 `action`，之后写进 parquet 的 `action` 列，才能和官方数据格式对齐。

2. **做轨迹级 instruction**  
   整条轨迹共用一句指令（如 "From AHG to Library"），写进 `meta/episodes.jsonl` 的 `tasks`。模型学的是：给这句话 + 当前帧图像（或历史帧），预测下一步离散 action（或 pixel goal，若你后面补了 goal）。

3. **降采样、筛帧**  
   一万多帧对单条轨迹来说很多，可以按 `linear_x`/`angular_z` 筛「有在动」的帧，或每隔 N 帧取一帧（`--downsample N`），再去做离散化和转 LeRobot，减少数据量、加快训练。

4. **暂不用的列**  
   `timestamp_ns` 可用于和 bag 里其他 topic 对齐；若以后用 odom 位姿算 pixel_goal，也可按时间戳对齐 pose。

总结：**这 1 万多帧 + 每帧的 linear_x / angular_z，就是「图像序列 + 连续控制」；用 6.5 打成 LeRobot 的 parquet + meta，再离散化成 action、配上 instruction、补 pose/goal 占位，就能往 InternVLA-N1 的 SFT pipeline 里接。**

### 6.4 从提取结果到 LeRobot 目录（extracted_to_lerobot）

**建议务必加 `--copy-frames`**：这样会把帧复制到 `out_root/frames/chunk-XXX/`，parquet 里 `image_path` 存为相对 out_root 的路径（如 `frames/chunk-000/frame_000001.jpg`），loader 用 `out_root / image_path` 即可，**换机器/换路径数据集仍可用**，是这套流程能「搬机器就能用」的关键。

在已有 **提取目录**（`frame_*.jpg` + `actions.csv`，可选 `state.csv`）的前提下，先安装 parquet 依赖再运行：

```bash
pip install pyarrow
python scripts/extracted_to_lerobot.py --extracted-dir D:\MyLab\SCAND\ahg2library_extracted --out-root D:\MyLab\SCAND\scand_lerobot --episode-idx 0 --task "Go from AHG to Library." --copy-frames
```

会得到：

- **out_root/data/chunk-000/episode_000000.parquet**：每行一帧，列包括 `timestamp_ns`、`frame_index`、`image_path`、`action.linear_x`、`action.linear_y`、`action.angular_z`、`state.x`、`state.y`、`state.theta`（无 state.csv 时 state 填 0）。
- **out_root/meta/episodes.jsonl**：追加一行，含 `episode_index`、`length`、`tasks`、`dialogs`、`source`。

这样得到的是 **BC 版 LeRobot**（视觉→连续速度，适合模仿学习/BC）；**还不是**能直接接 InternVLA-N1 Joint training 的数据——官方 Joint 训练目标是 pixel goal / waypoints（`pixel_goal_only=True` 等），若要接 InternVLA-N1，需要再把监督信号对齐为 pose、goal、relative_goal_frame_id 等（见第二节 2.2）。

### 6.5 可选：/localization、state.csv 与 mp4

- **/localization**：你的 bag 里有 `/localization [amrl_msgs/msg/Localization2DMsg]`，适合做 2D 位姿。提取时加 `--loc-topic /localization` 会生成 **state.csv**（每帧二分对齐到最近一条 loc 的 x, y, theta）。**theta 已按“真 yaw”处理**：脚本会尝试字段 `theta`/`yaw`/`heading`/`rot`，若是四元数则用 `atan2(2*(w*z+x*y), 1-2*(y*y+z*z))` 算 yaw，不再用 `orientation.z`。
- **首次带 loc 提取时会打印一条 LOC 消息**（`LOC MSGTYPE`、`LOC MSG`），便于确认字段名；若 theta 仍全 0，可根据打印改 `_yaw_from_loc_msg`。
- **自检**：用 `state.x`、`state.y` 画散点图看轨迹是否连续、无跳变/回零，可快速发现对齐或单位问题。
- **mp4**：若希望 LeRobot 里带视频，可 (1) 把 SCAND 自带的 mp4 复制到 `out_root/videos/chunk-000/episode_000000.mp4`，并确认帧序与 parquet 的 frame_index 一致；或 (2) 用 ffmpeg 从导出帧重新编码：  
  `ffmpeg -framerate 30 -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p episode_000000.mp4`（在提取目录下执行，输出放到 `videos/chunk-000/`）。

### 6.6 小结

- **.mp4**：仅作预览，训练用数据以 **.bag** 为准。  
- **一个 sample**：用脚本列出 topic → 提取帧 + odom（+ 可选 loc）→ 用 **extracted_to_lerobot.py** 打成 parquet + episodes.jsonl → 再按需离散 action、写 instruction、补 pose/goal，并注册到 InternNav 的 `data_dict`，即可参与 InternVLA-N1-DualVLN 的 SFT。

### 6.7 SCAND→LeRobot 当前状态

**1) theta：必须验证的步骤 + sanity check**

- **原因**：`amrl_msgs/msg/Localization2DMsg` 的 theta 在 **`msg.pose`** 里（`Pose2Df`），不是顶层 `msg.theta`。打印首条 LOC 得到：`pose=...Pose2Df(x=..., y=..., theta=-1.469...)`。
- **修改**：在 `_yaw_from_loc_msg()` 里增加对 **`msg.pose.theta`** 的读取（在尝试顶层 theta、四元数之前）。若为四元数，仍用 `yaw = atan2(2*(w*z+x*y), 1-2*(y*y+z*z))`，不再用 `orientation.z`。
- **验证**：首次带 `--loc-topic` 提取后，**务必检查** `state.csv` 里 theta 是否非 0、帧间是否连续；并用 `state.x`、`state.y` 画轨迹散点图做 sanity check（无跳变/回零）。不同 bag 或 msg 类型可能字段不同，以打印的 LOC MSG 为准。

**2) 对齐已改为二分（性能）**

- 对 cmd 和 loc 都用 **`bisect_right(ts_list, ts) - 1`** 取「最后一个 t ≤ ts」的样本，写 actions.csv / state.csv 时不再每帧从头扫列表。
- 复杂度由 O(N×M) 变为 O(N log M)，多条 bag 时更稳。

**3) 数据集自包含（--copy-frames）**

- **`--copy-frames`** 时：帧复制到 **`out_root/frames/chunk-XXX/`**，parquet 里 **`image_path`** 存为相对 out_root 的路径（如 `frames/chunk-000/frame_000001.jpg`），loader 用 `out_root / image_path` 即可，换机器/路径不失效。
- 未加 `--copy-frames` 时仍为「仅 parquet + meta」，image_path 为文件名，需配合 extracted_dir 使用。

**当前状态**

- 已有：**meta/episodes.jsonl**、**data/chunk-000/episode_000000.parquet**（逐帧：obs=image_path、action=连续速度、state=x,y,theta），theta 正确，对齐为二分。
- **依赖**：`extracted_to_lerobot.py` 写 parquet 需要 **`pip install pyarrow`**（或 fastparquet），否则会报 “Missing optional dependency 'pyarrow'”。装好后即可正常跑 `--copy-frames` 生成自包含数据集。
- **自检**：用 `state.x`、`state.y` 画散点图看轨迹是否连续、无跳变，已写入 6.5；theta 已通过 state.csv 确认非 0 且连续。

综上，两条红线（**theta 正确 yaw**、**数据集自包含**）已满足；性能用二分对齐也已改好。后续可按需加离散 action、instruction、pose/goal 占位接 InternNav。
