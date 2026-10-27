# LingBot-VLA 2.0

`physicalai.policies.LingBotVLA2` loads the released 6B pretrain or RoboTwin checkpoint and fine-tunes it on LeRobot datasets. It uses Qwen3-VL-4B-Instruct, a 36-layer MoE action expert, and 10-step flow matching. Installation needs the `lingbotvla2` extra and a matching CUDA build of PyTorch 2.11 or later.

```bash
cd library
uv sync --frozen --extra tests --extra cu128 --extra lingbotvla2
uv run --no-sync physicalai fit --config configs/physicalai/lingbotvla2/so101.yaml
```

The SO-101 example uses a public LeRobot dataset with front and wrist views, quantile statistics, a batch size of one, and LoRA on the VLM plus a trainable action expert. A batch-one bf16-mixed optimizer step with gradient checkpointing peaked at 56.2 GB of allocated GPU memory on an H100. Leave room for allocator overhead and other processes. Full fine-tuning needs a sharded multi-GPU setup; `configs/physicalai/lingbotvla2/robotwin.yaml` is a starting point, but its dataset ID must be replaced with a LeRobot conversion of RoboTwin episodes.

MoE backend `triton` uses PyTorch grouped GEMM and native autograd; the name is retained for saved configurations. The `dense` backend remains the FP32 reference.

## Upstream fine-tuning

`configs/physicalai/lingbotvla2/real_robot.yaml` runs full fine-tuning on four CUDA GPUs with
FSDP2, FP32 action-expert settings, gradient checkpointing, Muon, MSE flow matching, and depth/video
alignment. Replace its dataset root and robot/camera mapping before running it. Normalization
uses only training episodes; arm targets are relative to current state and grippers stay absolute.
The library's default constructor uses action-only fine-tuning. Enable `training_recipe="upstream"`
and `distillation_heads=True` to train with the full objective.

Install the `lingbotvla2-teachers` extra alongside `cu128` for teacher-enabled training:

```bash
uv sync --frozen --extra tests --extra cu128 --extra lingbotvla2-teachers
uv run --no-sync physicalai fit --config configs/physicalai/lingbotvla2/real_robot.yaml
```

Teachers load lazily from pinned model revisions. For each training batch, MoGe estimates depth
from the first camera's current and future RGB frames; MoRGBD and DINO-Video generate the alignment
targets online. The frozen teachers consume additional GPU memory in each training process.
The dataset is never modified. Recorded-depth inputs, precomputed teacher targets, and Studio
integration are not included.

Teacher implementations are isolated from upstream's Transformers patches. Original licenses,
including the separate DINOv3 agreement, are included under `_vendor/`. Inference and Torch exports
do not load teachers. `teacher_weights_dir` can supply local assets for training without downloads.

A three-step, four-H100 student smoke test with synthetic teacher targets peaked at about 25.7 GB
allocated per GPU. This excludes the online teachers and does not establish task quality or exact
upstream training parity. Checkpoint save/resume and offline inference also pass smoke checks.

## Joint and camera layout

Training infers the joint groups from the joint names recorded with the dataset, so any robot whose driver follows the plugin naming convention works without robot-specific settings. Joints named `gripper` (or `*_gripper`) fill `effector.position`; all other joints fill `arm.position`. Bimanual robots prefix joints with `left_` and `right_`, and the left arm comes first. A `.pos` suffix is ignored, and each group is packed in recorded order, as upstream does. Training stops with an error when names are missing, duplicated, mix prefixed and unprefixed joints, or are not positions. For those datasets, pass explicit `state_map` and `action_map` instead. The RoboTwin maps are:

```yaml
state_map:
  arm.position: [[0, 6], [7, 13]]
  effector.position: [[6, 7], [13, 14]]
action_map:
  arm.position: [[0, 6], [7, 13]]
  effector.position: [[6, 7], [13, 14]]
```

Map dataset cameras with `image_key_reorder_map={"camera_name": 0, ...}` unless they are already named after the slots. Slots 0, 1 and 2 are `camera_top`, `camera_wrist_left` and `camera_wrist_right`; unused slots are masked. Every dataset camera must have a slot. The checkpoint's remaining joint-group order follows upstream's `real_robot.yaml`; it has not been verified zero-shot for hand, waist, head or base control.

## Checkpoints and Torch export

`LingBotVLA2.load_from_checkpoint(path)` rebuilds from the saved architecture and joint maps. Load only checkpoints you trust: Lightning deserializes Torch checkpoint files. Restore does not fetch pretrained shards. Normal inference still needs the pinned Qwen3-VL tokenizer in the local Hugging Face cache, unless the checkpoint lives in a Torch export directory with a `tokenizer/` folder. Torch exports copy the tokenizer into that folder for offline Runtime loading:

```python
from physicalai.export import ExportBackend

policy = LingBotVLA2.load_from_checkpoint("model.ckpt")
policy.eval().export("export/lingbot", backend=ExportBackend.TORCH)
```

Runtime consumes the resulting directory with `InferenceModel("export/lingbot")`. The model predicts a full 50-step canonical chunk; the policy gathers, denormalizes and trims it to `n_action_steps` for execution. Torch is the only advertised export backend. Teacher models and future frames are training-only resources. RTC and ONNX/OpenVINO paths are not included. The opt-in parity test matches an upstream dense fp32 MoE reference within `atol=2e-4, rtol=1e-4`. Upstream's faster Triton MoE path uses approximate matrix multiplication; on an H100 its output differed from the dense path by up to 0.03 for the same input. That comparison is not a RoboTwin success-rate benchmark.

The upstream implementation and Qwen3-VL-4B-Instruct are Apache-2.0. The LingBot checkpoint repositories do not declare a license in their Hugging Face metadata; check their model cards before redistributing weights.
