# Trainer

Lightning Trainer wrapper with physicalai-specific defaults and callbacks.

## Overview

The `Trainer` class extends Lightning's Trainer with:

- **Automatic callback injection** - `PolicyDatasetInteraction` for policy-data coordination
- **Better defaults** - Experiments saved to `experiments/` directory
- **Experiment naming** - Optional `experiment_name` parameter for organization

## Usage

```python
from physicalai.train import Trainer
from physicalai.policies import ACT
from physicalai.data import LeRobotDataModule

# Basic training
policy = ACT()
datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# With experiment name
trainer = Trainer(max_epochs=100, experiment_name="pusht_act_v1")
# Saves to: experiments/pusht_act_v1/version_0/

# Custom logger
from lightning.pytorch.loggers import WandbLogger
trainer = Trainer(max_epochs=100, logger=WandbLogger(project="robot"))
```

## Interface

```python
class Trainer(lightning.Trainer):
    def __init__(
        self,
        *,
        # physicalai-specific
        experiment_name: str | None = None,

        # Hardware
        accelerator: str = "auto",
        devices: int | list[int] = "auto",
        precision: str | None = None,

        # Training control
        max_epochs: int | None = None,
        gradient_clip_val: float | None = None,

        # Logging
        logger: Logger | bool | None = None,
        default_root_dir: str = "experiments",  # Changed from Lightning default

        # Validation
        num_sanity_val_steps: int = 0,  # Changed from Lightning default (2)
        check_val_every_n_epoch: int = 1,

        # All other Lightning Trainer parameters...
    ): ...
```

## Key Differences from Lightning

| Parameter              | Lightning Default | physicalai Default | Reason                                           |
| ---------------------- | ----------------- | ------------------ | ------------------------------------------------ |
| `default_root_dir`     | Current directory | `"experiments"`    | Cleaner project structure                        |
| `num_sanity_val_steps` | 2                 | 0                  | Embodied AI typically doesn't need sanity checks |

## Weight precision

With `bf16-mixed` or `16-mixed`, policies keep trainable fp16/bf16 parameters in
fp32 so small optimizer updates do not round away. Frozen parameters and buffers
keep their original dtypes; Lightning's autocast controls computation precision.
The extra weight memory depends on how many parameters are trainable, not the
size of the frozen backbone.

The shared `Policy.configure_model()` hook promotes weights after `setup()` and
before checkpoint restore, DDP wrapping, and optimizer creation. Policies that
build their model in `configure_model()` must call `super().configure_model()`
after building it. True-precision modes and precision managed by FSDP or
DeepSpeed are unchanged.

Trainable weights remain fp32 after fitting. For standalone inference or export,
reload the saved checkpoint to reconstruct the policy's configured inference
dtypes. Inference-only loads do not promote weights. Keep the original fp32
checkpoint for resuming training; casting it to bf16 discards small updates.

## Callbacks

### PolicyDatasetInteraction

Automatically injected callback that coordinates policy initialization with datamodule:

```python
class PolicyDatasetInteraction(Callback):
    def on_fit_start(self, trainer, pl_module):
        # Ensures policy can access dataset features for lazy initialization
```

This enables policies to be created without specifying input/output shapes, deferring initialization until the dataset is available.

## Output Structure

```text
experiments/
├── lightning_logs/           # Default (no experiment_name)
│   └── version_0/
│       ├── checkpoints/
│       ├── hparams.yaml
│       └── events.out.tfevents.*
└── pusht_act_v1/            # With experiment_name="pusht_act_v1"
    └── version_0/
        └── ...
```

## Examples

### GPU Training with Mixed Precision

```python
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=200,
)
```

### Multi-GPU Training

```python
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    max_epochs=100,
)
```

### Development/Debug Mode

```python
# Quick run with 1 batch
trainer = Trainer(fast_dev_run=True)

# Overfit on small subset
trainer = Trainer(overfit_batches=10)
```

## See Also

- [CLI Guide](../cli/README.md) - Training via command line
- [Policy Design](../policy/README.md) - Policy implementation
- [Data Module](../data/README.md) - Dataset management
