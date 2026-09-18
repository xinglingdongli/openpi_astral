# Astral 单臂遥操机器人 × pi0.5 适配

适配 loopkok `astral_ws`（Quest3 VR 遥操 → 左臂 7 关节 + 夹爪，Wuji 手暂未入训练）
采集的数据，在本仓库训练 pi0.5。

## 数据约定（astral_data_collect → LeRobot v2.1）

| 项 | 值 |
|---|---|
| `observation.state` | 8 维：左臂 7 关节 (rad) + 左夹爪闭合比 (0..1) |
| `action` | 8 维，**next_state 语义**（`action[t] = state[t+1]`），绝对关节角；训练管线内 7 个臂关节转 delta（相对当前 state），夹爪保持绝对 |
| `observation.images.video8` | D435i 头部第三人称（转换期已 letterbox 到 224×224） |
| `observation.images.video0` | 左腕 USB（同上 224×224） |
| `observation.images.video2` | 第二第三人称（**当前不参与训练**） |
| fps | 30 |

模型侧映射（`src/openpi/policies/astral_policy.py`）：video8→`base_0_rgb`，
video0→`left_wrist_0_rgb`，`right_wrist_0_rgb` 零填充 + mask=False
（与 DROID/Libero 的单腕约定一致）。

## 一次性准备

```bash
# 1. 数据集放进 HF_LEROBOT_HOME（软链即可；pinned lerobot 0.1.0 只认 HF_LEROBOT_HOME，
#    设旧的 LEROBOT_HOME 会直接报错）
mkdir -p ~/.cache/huggingface/lerobot/astral
ln -sfn <path-to>/astral_data_lerobot ~/.cache/huggingface/lerobot/astral/astral_teleop

# 2. 下载 pi05_base 预训练权重（~10GB，首次训练时也会自动拉，但手动拉可控）
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"

# 3. 计算归一化统计（换新数据/改 repo_id 后必须重跑）
uv run scripts/compute_norm_stats.py --config-name pi05_astral_lora
# 全量微调配置另算一份（assets 目录按 config 名隔离）
uv run scripts/compute_norm_stats.py --config-name pi05_astral
```

注意：parquet 里**不能**带视频 struct 列（v2.1 规范如此，旧版 lerobot 会崩）——
astral_data_collect 转换器已修正，老数据用 pyarrow `select` 掉即可。

图像在**转换期**已按 openpi `resize_with_pad` 的几何约定 letterbox 到 224×224
（等比缩放 + 对称黑边，非拉伸），训练时 `ResizeImages(224,224)` 为恒等操作，
模型输入与全分辨率入库再在线缩放完全一致，但省掉解码大图 + 在线缩放的 CPU 开销。
要保留原分辨率入库：转换时加 `--image-size 0`（norm stats 与图像无关，无需重算）。

## 换机器人配置怎么办（加右臂 / 换灵巧手 / 加腰头）

**采集→转换链不用改代码**：`data_collect.yaml` 的 schema（arms / end_effector /
cameras / include_waist / include_head）驱动一切，重采重转即可。

**openpi 侧也不用改代码**：`LeRobotAstralDataConfig.create()` 每次运行都从数据集
`meta/info.json` 现读布局——state 维度、`AstralOutputs` 截断维、delta/绝对划分
（名字含 `_ee_` 的维保持绝对，臂/腰/头全部转 delta）全部自动跟随。
唯一需要动的是 `camera_map`（哪个相机进哪个模型槽位，在 config.py 的
`pi05_astral*` 里改一行）。

**两个硬约束**：
1. 总 state 维 ≤ 32（pi05_base 上限）：单臂+夹爪 8 ✓ / 双臂 16 ✓ / 单臂+wuji 27 ✓ /
   **双臂+双 wuji = 38 ✗**（超限会在配置创建时直接报错，不会静默错训）
2. 布局变了 → norm stats 必须重算（delta 分布变了）：
   `uv run scripts/compute_norm_stats.py --config-name pi05_astral_lora`

## 训练

4090D (24GB) 单卡推荐 LoRA：

```bash
uv run scripts/train.py pi05_astral_lora --exp-name=run1
```

全量微调（显存大概率不够，OOM 就降 `batch_size` 或回 LoRA）：

```bash
uv run scripts/train.py pi05_astral --exp-name=run1
```

两个配置（`src/openpi/training/config.py`）：`pi05_astral`（全量，batch 8）、
`pi05_astral_lora`（LoRA，batch 32）。均 `action_horizon=50`（1.67s@30fps）、
`discrete_state_input=False`（连续 state 走 suffix，不占语言 token）、
pi05 默认 quantile 归一化。

## 部署侧注意（训练/推理对称性）

- 数据是 `action=next_state`：模型学的是「下一时刻应处的关节位置」，已隐式吸收
  实机的电机执行延迟/摩擦/阻尼（实测 VR→到位中位 200-240ms，腕部 260-315ms）。
  训练时臂关节被转成相对当前 state 的 delta（对齐 pi05 预训练惯例 + 消除
  copy-state 捷径）；**推理时 `AbsoluteActions` 会自动把当前 state 加回来**，
  部署侧下发的仍是位置目标，无需改动。不要在中间再叠一层低通/限速
  改变动力学——训练分布会被破坏。
- 推理客户端按键约定（与训练 repack 后一致）：
  `observation/state` (8,)、`observation/image` (H,W,3) uint8、
  `observation/wrist_image` (H,W,3) uint8、`prompt` str。
- 语言指令：训练靠 `prompt_from_task`（采集段必须填 task！早期 3 段空 task 的
  数据靠 `default_prompt` 兜底，仅适合冒烟，不适合正式训练）。

## 冒烟测试记录（2026-09-01，3 段/1828 帧）

- [x] `LeRobotDataset` 经 openpi 管线加载（AV1 视频解码正常）
- [x] `|action[t] - state[t+1]| max = 0.0`（加载后语义保持）
- [x] 变换链输出：state (32,)、actions (10,32)、image 224×224、右腕 mask=False
- [x] 空 prompt → `default_prompt` 兜底分词正确
- [x] norm stats 已生成（`assets/pi05_astral{,_lora}/astral/astral_teleop/`）
