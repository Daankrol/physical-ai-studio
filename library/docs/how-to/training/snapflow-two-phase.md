# How to run two-phase SnapFlow distillation

SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) compresses
flow-matching action generation into a **single forward pass (1-NFE)** via
progressive self-distillation, matching or exceeding multi-step teacher accuracy
with a 3–10× denoising speedup.  Training is split into two phases:

| Phase | What trains | Steps |
| ----- | ----------- | ----- |
| 1 | Full model — standard flow-matching | Varies (e.g. 50 k) |
| 2 | Action expert + target-time embedding only (~10% params) | ~30 k |

Both `SmolVLA` and `Pi05` support SnapFlow.

---

## Option 1 — Two explicit CLI runs (recommended for clean optimizer state)

```bash
# Phase 1: standard flow-matching training
physicalai fit --config configs/physicalai/smolvla.yaml

# Phase 2: SnapFlow distillation, VLM frozen, warm-started from phase 1
physicalai fit \
    --config configs/physicalai/smolvla_snapflow.yaml \
    --ckpt_path ./lightning_logs/version_0/checkpoints/last.ckpt \
    --trainer.max_steps 30000
```

Substitute `smolvla` with `pi05` for the Pi05 policy.

Phase-2 config templates are provided at:

- `configs/physicalai/smolvla_snapflow.yaml`
- `configs/physicalai/pi05_snapflow.yaml`

They enable `snapflow_enabled: true`, `train_expert_only: true`, and set the
paper-default hyperparameters (`alpha: 0.5`, `lambda: 0.1`,
`num_inference_steps: 1`).

---

## Option 2 — Single run with `SnapFlowPhaseCallback`

Add the callback to your training config so phase 2 starts automatically at
`start_step` within a single `physicalai fit` call:

```yaml
# smolvla.yaml (excerpt)
trainer:
  max_steps: 80000        # phase 1 (50 k) + phase 2 (30 k)
  callbacks:
    - class_path: physicalai.train.SnapFlowPhaseCallback
      init_args:
        start_step: 50000   # phase-2 boundary
        alpha: 0.5
        lambda_: 0.1
        num_inference_steps: 1
```

At `global_step == start_step` the callback:

1. Calls `policy.enable_snapflow(alpha, lambda_, num_inference_steps)` which
   sets the SnapFlow flags, freezes the VLM backbone, and calls the existing
   `set_requires_grad()` primitive.
2. Reconfigures the optimizer so only the unfrozen parameters (action expert +
   target-time embedding) are updated, giving a clean optimizer state for phase 2.

---

## Hyperparameter guidance

| Parameter | Paper default | Notes |
| --------- | ------------- | ----- |
| `alpha` | `0.5` | FM-loss weight; keep ≥ 0.5 to preserve multi-step ability |
| `lambda_` | `0.1` | Shortcut-loss scale |
| `num_inference_steps` | `1` | 1 = full SnapFlow speedup; increase for intermediate modes |
| Phase-2 steps | ~30 k | Sufficient because target-time embedding is zero-initialized |

---

## Verifying the transition

After phase 2 starts, check the training logs for:

- `train/loss_snapflow_fm` and `train/loss_snapflow_shortcut` — both should
  be active.
- VLM parameters with `requires_grad == False` (all backbone params frozen).
- 1-step benchmark success ≈ or > the multi-step teacher.
