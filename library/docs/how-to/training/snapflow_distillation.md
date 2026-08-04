# Train a policy with SnapFlow (1-step action generation)

## What SnapFlow is

Flow-matching VLAs such as Pi0.5 and SmolVLA generate an action chunk by
integrating a learned velocity field from noise back to an action over a
`K`-step Euler loop (typically `K = 10`). Every one of those steps is a full
forward pass through the action expert, and on Pi0.5 that loop accounts for
roughly 80% of end-to-end inference latency. Simply setting `K = 1` does not
work: the velocity field is calibrated for small local steps, not one global
jump.

SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) compresses that
loop into a **single forward pass (1-NFE)** through self-distillation. Reported
results: on Pi0.5 / LIBERO it reaches 98.75% success against 97.75% for the
10-step teacher, with a 9.6x denoising speedup (274 ms -> 83 ms end to end); on
SmolVLA it cuts offline action MSE by 8.3% with a 3.56x end-to-end speedup.

Both [`Pi05`](../../explanation/policy/README.md) and `SmolVLA` support it.

## Why it needs two phases

SnapFlow is a _fine-tuning_ stage, not a from-scratch training mode. It builds
its distillation target from the model's own velocity predictions, so it needs a
model whose velocity field is already good:

| Phase | What trains                                            | Objective                       | Typical budget                 |
| ----- | ------------------------------------------------------ | ------------------------------- | ------------------------------ |
| 1     | Full model                                             | Standard flow matching          | 5-10 epochs (warm-started VLA) |
| 2     | Action expert + target-time embedding (~10% of params) | Mixed FM + shortcut consistency | 3-5 epochs                     |

Three properties make phase 2 cheap and safe:

- **The VLM backbone is frozen.** Only a thin head moves, so perception and
  language representations cannot drift while the action head is reshaped.
- **The target-time embedding is zero-initialised.** At the start of phase 2 the
  model is numerically identical to the phase-1 teacher — the transition cannot
  regress the model on contact.
- **The `alpha` mix keeps half the batch on standard flow matching.** That
  preserves the multi-step behaviour the shortcut target depends on, preventing
  catastrophic forgetting.

Distilling an undertrained phase-1 model just distills noise. Do not skip
phase 1.

---

## Option 1 — One run, phase switch via callback (recommended)

`SnapFlowPhaseCallback` flips the policy into SnapFlow mode at a configured
phase boundary and rebuilds the optimizer over the now-trainable parameters. No
checkpoint handoff, no second command.

A complete worked config ships at
[`configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml`](../../../configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml):

```bash
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml
```

The parts that matter:

```yaml
model:
  class_path: physicalai.policies.Pi05
  init_args:
    pretrained_name_or_path: lerobot/pi05_base
    # Phase 1 is plain flow matching — the callback turns SnapFlow on later.
    train_expert_only: false
    scheduler_decay_steps: null # cosine horizon = real step budget
    scheduler_warmup_steps: 100

trainer:
  max_epochs: 8 # phase 1 (5) + phase 2 (3)
  precision: bf16-mixed
  callbacks:
    - class_path: physicalai.train.SnapFlowPhaseCallback
      init_args:
        start_epoch: 5 # phase-2 boundary
        alpha: 0.5
        lambda_: 0.1
        num_inference_steps: 1
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        every_n_epochs: 2
        save_top_k: -1 # keep every staged checkpoint
        save_last: link
        filename: "epoch{epoch:03d}"
        auto_insert_metric_name: false
```

Note the YAML key is `lambda_`, with a trailing underscore (`lambda` is a Python
keyword).

At the boundary the callback:

1. Calls `policy.enable_snapflow(alpha, lambda_, num_inference_steps)`, which
   activates the mixed objective, sets `train_expert_only`, freezes the VLM via
   the policy's existing `set_requires_grad()` primitive, and refreshes the
   checkpoint hparams so checkpoints saved afterwards reload as SnapFlow
   policies.
2. Calls `trainer.strategy.setup_optimizers(trainer)` so the optimizer covers
   only the unfrozen parameters and starts with clean state.

### Step-based boundary

If your run is budgeted in steps rather than epochs, use `start_step` instead.
Exactly one of the two must be set:

```yaml
- class_path: physicalai.train.SnapFlowPhaseCallback
  init_args:
    start_step: 30000
```

### Useful overrides

```bash
# Smoke-test the wiring without training anything
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --trainer.fast_dev_run 1

# Smaller GPU
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --data.train_batch_size 8 --trainer.accumulate_grad_batches 2

# Longer total budget
physicalai fit --config configs/physicalai/pi05_finetune_and_snapflow_distillation.yaml \
    --trainer.max_epochs 20
```

### Caveat: the phase-2 LR horizon

Rebuilding the optimizer also rebuilds the LR scheduler, so phase 2 gets a fresh
warmup. The cosine decay horizon, however, is derived from
`Trainer.estimated_stepping_batches`, which reports the _total_ run budget
rather than the phase-2 remainder. Phase 2 therefore decays more slowly than a
standalone phase-2 run would. If you need an exact phase-2 decay horizon, use
Option 2.

---

## Option 2 — Two explicit runs

Use this when you want phase 2 to have its own LR horizon, or when you are
distilling from a checkpoint you already have.

```bash
# Phase 1 — standard flow-matching training
physicalai fit --config configs/physicalai/pi05.yaml

# Phase 2 — SnapFlow distillation, VLM frozen, resumed from phase 1
physicalai fit \
    --config configs/physicalai/pi05_snapflow_distillation.yaml \
    --fit.ckpt_path ./experiments/lightning_logs/version_0/checkpoints/last.ckpt \
    --trainer.max_steps 60000
```

Substitute `pi05` with `smolvla` for the SmolVLA policy. Phase-2 templates:

- `configs/physicalai/pi05_snapflow_distillation.yaml`
- `configs/physicalai/smolvla_snapflow_distillation.yaml`

Both set `snapflow_enabled: true`, `train_expert_only: true`, and the paper
defaults (`snapflow_alpha: 0.5`, `snapflow_lambda: 0.1`,
`snapflow_num_inference_steps: 1`).

Two things to be aware of:

- The flag is **`--fit.ckpt_path`**, not `--ckpt_path`. Method-level arguments
  are namespaced under the subcommand (`--validate.ckpt_path`,
  `--test.ckpt_path`, `--predict.ckpt_path`).
- `--fit.ckpt_path` is a full Lightning **resume**: it restores the global step,
  optimizer state, and LR schedule from phase 1. Set `--trainer.max_steps` to
  the _combined_ phase-1 + phase-2 budget, not the phase-2 budget alone.

---

## Option 3 — Python API

```python
from physicalai.policies import Pi05
from physicalai.train import Trainer

policy = Pi05.load_from_checkpoint(
    "experiments/phase1/last.ckpt",
    map_location="cpu",
    snapflow_enabled=True,
    snapflow_alpha=0.5,
    snapflow_lambda=0.1,
    snapflow_num_inference_steps=1,
    train_expert_only=True,
    # compile_model is excluded from saved hparams, so re-pass it explicitly.
    compile_model=True,
)

Trainer(max_epochs=3, precision="bf16-mixed").fit(policy, datamodule=datamodule)
```

Or flip an already-constructed policy after `setup()` has run:

```python
policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)
```

---

## Hyperparameter guidance

| Parameter                | Paper default            | Notes                                                                  |
| ------------------------ | ------------------------ | ---------------------------------------------------------------------- |
| `alpha`                  | `0.5`                    | FM-loss weight. Keep at or above `0.5` to preserve multi-step ability. |
| `lambda_`                | `0.1`                    | Shortcut-loss scale, balances the two gradient magnitudes.             |
| `num_inference_steps`    | `1`                      | `1` gives the full SnapFlow speedup; raise it for intermediate modes.  |
| Phase-1 budget           | 5-10 epochs              | Fewer for a warm-started VLA than for from-scratch training.           |
| Phase-2 budget           | ~3-5 epochs (~30k steps) | Short because the target-time embedding is zero-initialised.           |
| `scheduler_warmup_steps` | ~5% of total steps       | No fractional option exists; compute it from your dataset (below).     |

Converting an epoch budget into steps, for warmup sizing:

```text
steps_per_epoch = floor(train_frames / (batch_size * devices)) / accumulate_grad_batches
total_steps     = max_epochs * steps_per_epoch
warmup          = 0.05 * total_steps
```

Set `scheduler_decay_steps: null` so the cosine horizon follows
`Trainer.estimated_stepping_batches` and the LR lands on `scheduler_decay_lr`
exactly at the end of the run.

Hold out a validation split (`data.init_args.val_split`) on small datasets.
Without it there is no way to distinguish convergence from memorisation.

---

## Verifying the transition

After the phase boundary, check that:

- The log line `SnapFlowPhaseCallback: activating SnapFlow distillation at
step N / epoch M` appeared, followed by the phase-2 trainable-parameter
  count. That count should be roughly 10% of total parameters.
- `train/loss` shifts level at the boundary — expected, the objective changed.
  It should settle rather than diverge.
- `val/loss` (full-denoising action MSE) does not regress. Because the
  target-time embedding is zero-initialised, phase 2 starts at teacher parity;
  a large immediate jump means something is misconfigured.
- Every VLM parameter reports `requires_grad == False`:

  ```python
  frozen = [n for n, p in policy.named_parameters() if not p.requires_grad]
  ```

- A 1-step benchmark matches or beats the multi-step teacher:

  ```bash
  physicalai benchmark --config configs/benchmark/libero.yaml \
      --ckpt_path experiments/.../last.ckpt
  ```

## Exporting the distilled policy

The SnapFlow checkpoint exports like any other policy. The exported artifact
carries the 1-NFE sampling path, which is where the latency win materialises for
Runtime:

```bash
physicalai export --ckpt_path experiments/.../last.ckpt --backend openvino
```

See [Export and deploy](../export/export_inference.md).

## Known gaps

- There is no `--seed_everything` equivalent in the CLI. For a fully reproducible
  run, call `lightning.seed_everything(seed)` from a Python entry point.
- `scheduler_warmup_steps` takes an absolute step count only; fractions of the
  total budget must be computed by hand using the formula above.
