# LingBot-VLA 2.0 implementation plan

## Goal

Fine-tune the released LingBot-VLA 2.0 model on our own LeRobot recordings through Studio's
CLI, Python API, and training jobs. Phase 2 reproduces upstream's real-robot post-training
recipe, including teacher losses, on four H100 80 GB GPUs with FSDP and gradient checkpointing.

## Current status

Phase 1 provides released-weight inference, action-only fine-tuning, checkpoint restore,
Torch export, and Studio integration. It does not reproduce the full upstream training recipe.

- Both released checkpoints match a fixed-input dense FP32 reference at `atol=2e-4, rtol=1e-4`.
  The committed test uses zero images and checks only the final normalized action chunk.
- The shipped Triton MoE differed from the dense reference by up to 0.03 on the reported H100
  input. Keep kernel approximation checks separate from strict model-math parity.
- The Phase 2 student loads all released alignment heads and ran three full-parameter steps
  on four H100s with FSDP2, Muon, fused MoE, and checkpointing, peaking at about 25.7 GB per GPU.
  It now also passes with actual teacher features cached from a synthetic RGB-D fixture.
  The four-GPU CLI recipe, plain checkpoint save/resume, and student-only offline inference pass
  smoke checks. These are pipeline checks, not a quality benchmark.

Pinned teachers now load without upstream's Transformers patches, with their licenses retained.
Native LeRobot depth stays separate from RGB, and Studio jobs can compute depth or cache teacher
features before training. Dataset depth skips MoGe, but still needs MoRGBD/video feature targets.
Independent upstream training/reference comparisons, throughput measurement, and owned-data/robot
rollout quality remain outstanding.

## Reference

Pin upstream to `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`, the revision used by the existing
parity test:

- [Custom-data guide](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/README.md)
- [Real-robot recipe](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/real_robot/real_robot.yaml)
- [RoboTwin recipe](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/robotwin/robotwin.yaml)
- [Training guide](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/Training_Config.md)

Start custom-data training from `robbyant/lingbot-vla-v2-6b`; retain
`robbyant/lingbot-vla-v2-6b-robotwin` for regression comparisons. Use the pinned source/configs
for architecture details rather than duplicating them here. Run upstream reference capture
in a separate environment, without installing its patched Transformers into Studio.

## Agreed training recipe

| Setting                | Decision                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Student                | Full fine-tuning: VLM, action expert, query projections, and alignment heads                                                                      |
| Hardware               | Four H100 80 GB GPUs; FSDP sharding of parameters, gradients, and optimizer state                                                                 |
| Precision              | Upstream FP32 action-expert settings and FP32 master parameters/optimizer state; retain verified upstream BF16 VLM/teacher paths and kernel casts |
| Gradient checkpointing | Supported and enabled for vision and joint decoder blocks                                                                                         |
| Initial batch          | One sample per GPU, accumulation 1; support accumulation before increasing micro-batch size                                                       |
| Real-world targets     | Arm actions relative to current state; absolute grippers and end-effector poses                                                                   |
| Real-world objective   | MSE flow matching, mean/std normalization, depth/video alignment, and router losses                                                               |
| Optimizer              | Upstream Muon/AdamW parameter split, constant `5e-5`, matching group/scaling rules                                                                |
| Teachers               | Frozen MoGe-2, LingBot-Depth/MoRGBD, and DINO-Video; RGB-derived supervision, no measured depth required                                          |

Keep RoboTwin's L1 flow loss, quantile normalization, absolute actions, and cosine learning rate
separate. Preserve the existing action-only and LoRA paths, but do not use them as evidence of
full-recipe parity. BF16 action-expert experiments and further LoRA tuning come after this baseline.

## Phase 2 work order

Use the checklist below for acceptance, not just implementation progress. Complete upstream
comparisons in 2A/2B before claiming numerical fidelity, and the quality gate in 2F before
claiming equivalent fine-tuning performance.

### 2A. Capture reproducible upstream references

- [ ] Commit a capture script and manifest with source/model/teacher/tokenizer revisions or
      hashes, configs, dependency versions, precision settings, seeds, and inputs.
- [ ] Capture varied RGB inputs, prefix embeddings/position ids, velocity, action chunks,
      individual training losses, representative gradients, and optimizer updates.
- [ ] Map computation-affecting upstream settings to Studio; reject unsupported values instead
      of silently ignoring them. Choose our evaluation dataset, episode split, budget, acceptable
      quality gap, and throughput target before long runs.

### 2B. Match preprocessing and training numerics

Files: `lingbotvla2/{preprocessor,postprocessor,model,config}.py` and `components/`.

- [ ] Match resize, uint8 rounding/rescaling, normalization, patch order, missing-camera masks,
      chat tokens, and MRoPE on varied non-square images and equivalent uint8/float inputs.
- [ ] Reproduce upstream's powered-uniform time sampler, final scaling, joint masks, flow-loss
      reduction, router top-k/balance/z-loss, and precision boundaries, including load-time casts.
      Keep noise/time injectable for deterministic comparisons.
- [ ] Match mean/std and `bounds_99_woclip`, including its additive `1e-6` denominator and
      zero-width bounds. Test inverse transforms without changing other policies' normalization.
- [ ] Make upstream's inclusion of padded action steps an explicit saved objective choice.
      Preserve Phase 1's padding-masked behavior for older checkpoints and the shared padding tests.
- [ ] Pass deterministic loss/gradient comparisons and retain strict dense FP32 inference parity.
      Record separate justified tolerances for optimized kernels; do not loosen the reference test.

### 2C. Adapt our LeRobot data

Files: `data/lerobot/`, training timestamp wiring, and `lingbotvla2/{preprocessor,postprocessor,layout}.py`.

- [ ] Reuse `LeRobotDataModule` for local/HF datasets. Validate task text, state/action units,
      explicit robot/camera maps, FPS/timestamps, and stats. Keep the pretrained 55-dimensional
      layout and SO-101/bimanual mappings; reject overlapping maps or unsupported camera cases.
- [ ] Request camera frames at `[0, (chunk_size - 1) / fps]` for training-only teacher targets.
      Keep state current-only and actions at offsets `0..chunk_size-1`; match effective video FPS,
      episode-end repetition, and masks without crossing episode boundaries.
- [ ] Split current policy images from future teacher images explicitly. The current preprocessor
      selects the last frame of a time stack: leaving that unchanged would leak future RGB.
      Test that changing future frames cannot change sampled actions with fixed noise.
- [ ] Subtract current raw arm state from each action before normalization; keep grippers absolute.
      Compute action stats in that representation, then denormalize before adding state back at
      inference. Test round-trips with nonzero state and preserve the absolute-action route.
- [ ] Split by episode and compute stats only from training episodes. Test collation, device
      transfer, masks, and timing on a synthetic multi-episode fixture through CLI and Python.
      Do not commit or upload private recordings.

### 2D. Restore teacher heads and losses

Files: `lingbotvla2/{model,config,pretrained}.py`, teacher/head modules, and policy setup.

- [ ] Restore the released alignment heads, including the 76 skipped `*_align_head.*` tensors,
      with strict loading and unchanged query embeddings/projections, prefix layout, and attention.
- [ ] Resolve pinned MoGe-2, LingBot-Depth/MoRGBD, and DINO-Video assets/configs. Audit licenses
      and dependencies before porting code, preserve attribution, and lazy-load teacher packages.
- [ ] Match teacher preprocessing, feature layers/layout, detach behavior, masks, and loss weights:
      current/future depth, current video patches, and future video alignment. Preserve the video
      sequence `[warmup current, current, future]`, current index 1, and MSE/cosine terms.
- [ ] Keep teachers frozen, in eval mode, and under no-grad target extraction. Log each weighted
      and unweighted loss; fail clearly on missing assets/frames rather than disabling losses.
- [ ] Match targets, head outputs, losses, and student gradients against upstream. Check teachers
      have no gradients. Use fake teachers in regular tests and real-teacher CUDA comparisons opt-in.

### 2E. Validate full fine-tuning on four H100s

Files: `lingbotvla2/policy.py`, `components/`, training strategy, and training configs.

- [ ] Match Muon updates, parameter groups, expert/vision learning rates, clipping, schedule, and
      optimizer-state round-trips. Verify FSDP compatibility with fused 3D expert parameters;
      do not apply matrix updates to flattened shards. Lightning FSDP is not automatically
      upstream FSDP2. If distributed Muon is needed, test and record its different recipe settings;
      do not silently substitute AdamW.
- [ ] Add policy-local fused Triton MoE, Flex attention, and applicable vision attention paths.
      Preserve state-dict names and stock Transformers compatibility. Keep dense/SDPA as a
      reference/fallback, report the selected backend, and test another Qwen policy in-process.
- [ ] Validate optimized forward/backward and gradient checkpointing together with FSDP. Avoid
      double checkpoint wrappers and checkpointing frozen teachers; keep inference KV caching
      separate. Compare checkpointing on/off losses, gradients, and RNG behavior on a tiny model.
- [ ] Run repeated teacher-enabled steps on all four GPUs with checkpointing enabled. Measure
      per-rank allocated/reserved memory, including teachers, optimizer temporaries, and gathers,
      plus steady-state throughput, compilation warmup, and checkpointing recomputation cost.
- [ ] Save/resume student, optimizer, scheduler, and RNG state. Compare the next update against
      an uninterrupted run within the chosen backend tolerance.

### 2F. Finish Studio, export, and quality validation

Files: policy/checkpoint/export code, `configs/physicalai/lingbotvla2/`, backend training/job
wiring, required UI inputs, dependency extras/lockfiles, and policy docs.

- [ ] Ship a runnable four-GPU custom-data recipe with FSDP, the agreed precision boundaries,
      checkpointing, dataset/maps, teacher paths, and compute budget. Test equivalent CLI/Python
      construction and Studio job/resume forwarding on the selected training target.
- [ ] Save the objective, action representation, stats/maps, architecture, teacher references,
      and checkpointing setting. Restore older headless checkpoints without changing their
      objective; resolve recorded teacher assets only for training resume.
- [ ] Keep Torch Runtime exports student-only, with tokenizer and target-data stats/maps.
      Test current-only offline inference without teacher weights/packages or hub architecture
      lookup. Preserve the full model chunk and trim only at the output boundary.
- [ ] Compare the untouched released model, upstream fine-tuning, Phase 2, and action-only
      training on the same owned data and budget. Record per-loss/gradient checks, held-out
      denoised action metrics, and rollout success with seed variation/confidence intervals.
      Use RoboTwin only as an additional regression check.
- [ ] Publish configs, metrics, measured hardware requirements, and limitations. A smoke test or
      decreasing loss is not a quality result; leave closed-loop quality unchecked if the robot
      or simulator is unavailable. Never expose credentials or private data in artifacts/logs.

## Implementation constraints

- Keep the existing config/model/policy split, saved-config restore, strict weight loading, and
  feature-owned normalization. Preserve module names and check LoRA targets when adding heads.
- Reuse shared policy components and data hooks. Do not globally patch Transformers, change
  other policies' numerics, or add teacher dependencies to discovery/inference.
- Keep regular tests tiny and offline; mark GPU/download comparisons opt-in. Each work package
  needs focused tests and scoped prek checks. Check asset licenses before redistributing weights.

From `library/`, prepare the CUDA environment explicitly, then run without resyncing:

```bash
uv sync --frozen --extra tests --extra cu128 --extra lingbotvla2-teachers
uv run --no-sync pytest tests/unit/policies/lingbotvla2
```

Run `uvx prek run --files <changed files>` from the repository root. Update recipes/docs with
any extra dependencies introduced by teachers or kernels.

## Out of scope

RTC, ONNX/OpenVINO export, quaternion-relative pose actions, nonzero loss-free MoE bias updates,
and sequence parallelism beyond the pinned recipes. Unsupported non-default settings must fail
clearly. None blocks the agreed real-robot fine-tuning route.
