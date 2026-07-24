# Two-Phase SnapFlow Distillation: Analysis & Plugin Design

Status: design proposal / research note
Scope: flow-matching policies (`SmolVLA`, `Pi05`)
Related code: `library/src/physicalai/policies/{smolvla,pi05}`

---

## 1. Executive summary

**The idea** — train a flow-matching VLA normally, then *enable SnapFlow* and
fine-tune for a short period with most of the model frozen — is not just
reasonable, it is **exactly the recipe published in the SnapFlow paper**
([arXiv:2604.05656](https://arxiv.org/abs/2604.05656), "SnapFlow: One-Step Action
Generation for Flow-Matching VLAs via Progressive Self-Distillation"). SnapFlow is
defined as a *self-distillation fine-tuning stage* that warm-starts from a
pretrained flow-matching model, **freezes the VLM backbone**, and trains only the
action expert plus a small target-time embedding (~10% of parameters) for ~30k
steps. So the proposed approach is the validated, state-of-the-art method rather
than a speculative experiment.

**Key realization about our codebase:** the SnapFlow *mechanism* is already
implemented end-to-end for both `SmolVLA` and `Pi05` (mixed FM/consistency loss,
two-step Euler shortcut target, zero-initialized target-time embedding, 1-NFE
inference path). What is **not** yet expressed cleanly is the *two-phase
orchestration* — the transition "train normally → freeze VLM → distill for a
limited time". Today that transition can only be done as two separate CLI runs,
and there is no first-class "plugin" abstraction; the SnapFlow logic is baked
directly into each model's `forward` / `sample_actions`.

**Recommendation:** adopt the pragmatic **Option A** (a Lightning
`SnapFlowPhaseCallback` + a documented two-phase config recipe, reusing the
existing freezing primitives). Keep the deeper **Option B** refactor (a pluggable
distillation-objective interface) as an optional follow-up if SnapFlow variants
start to multiply. Both options are documented in §5.

---

## 2. Literature review

### 2.1 The SnapFlow method

Flow-matching VLAs (π0, π0.5, SmolVLA) generate an action chunk by integrating a
learned velocity field from noise `x₁ ~ N(0, I)` back to the action `x₀` using a
`K`-step Euler loop (typically `K = 10`). That iterative denoising dominates
inference latency — ~80% of end-to-end time on π0.5. Naively dropping to 1 step is
unreliable because the velocity field is calibrated for small steps, not a single
global jump.

SnapFlow compresses denoising into a **single forward pass (1-NFE)** via
self-distillation, with four ingredients:

1. **Corrected consistency objective.** Standard consistency training substitutes
   the *conditional* velocity `v_t = ε − x₀` for the *marginal* velocity `u_t`.
   The paper proves (Theorems 1–2) that for a *fast flow* model with target time
   `s ≠ t` this substitution injects a variance term `L_var` that suppresses
   trajectory curvature and causes systematic drift. The fix is to build the
   consistency target from the model's own marginal-velocity predictions.

2. **Two-step Euler shortcut target.** Computing the corrected target exactly is
   expensive, so it is approximated by evaluating the model at `t = 1` and
   `t = 0.5` and averaging the two velocities (trapezoidal estimate of
   `∫₀¹ u dτ`). The 1-step velocity `F_θ(x₁, s=0, t=1)` is trained (MSE) to match
   this stop-gradient target.

3. **Progressive FM / consistency mixing.** Loss is
   `L = α·L_FM + (1−α)·λ·L_shortcut`. The FM branch keeps the marginal-velocity
   estimator `u_θ` calibrated (which is what makes the shortcut target
   trustworthy); the consistency branch teaches the single-step jump. Paper
   settings: `α = 0.5`, `λ = 0.1`.

4. **Target-time embedding `φ_s`.** A **zero-initialized** two-layer MLP encodes
   the target time `s` and is added to the existing time embedding, letting one
   network switch between local velocity estimation (`s = t`, FM) and global
   one-step generation (`s = 0`). Zero-init means the pretrained teacher is
   exactly preserved at initialization — the *only* new parameters.

**Training recipe (paper §3.6 + Appendix J):** warm-start from the pretrained FM
checkpoint; **freeze the VLM backbone**; train only the action expert + `φ_s`
(~10% of params) with gradient checkpointing; ~30k steps, ~12h on one A800;
identical hyperparameters across π0.5 (3B) and SmolVLA (500M).

**Results.** On π0.5 / LIBERO (400 episodes), SnapFlow 1-step reaches **98.75%**
success vs. **97.75%** for the 10-step teacher — matching/slightly exceeding it —
with **9.6× denoising speedup** (E2E 274 ms → 83 ms). On SmolVLA it cuts offline
MSE by **8.3%** with **3.56× E2E** acceleration. Tail errors shrink most (π0.5 P95
MSE −29.4%), which matters for closed-loop robustness.

### 2.2 Why "warm-start + freeze + short fine-tune" is the right shape

- **Warm-start is required, not optional.** The shortcut target is bootstrapped
  from the model's own marginal velocity. If the velocity field is untrained the
  target is garbage. A well-trained FM teacher is the precondition for the
  virtuous cycle "better `u_θ` → better target → better 1-step predictor".
- **Freezing the VLM is what makes it cheap and stable.** Only ~10% of params
  move; the perception/language representation cannot drift while the action head
  is being re-shaped. This is the direct justification for the user's "maybe even
  with the rest of the model frozen" instinct — the paper does exactly this.
- **The α-mix prevents catastrophic forgetting** of multi-step ability: keeping
  half the batch on standard FM preserves the teacher behaviour the shortcut
  target depends on.
- **Short duration is sufficient** because the target-time embedding is zero-init
  (start = exact teacher) and only a thin head is adapted.

### 2.3 Related fast-sampling work (context)

Consistency Models and continuous-time consistency (Song et al.; Lu & Song),
MeanFlow (average-velocity modelling), Shortcut Models (two-step decomposition,
the shortcut used here), α-Flow (FM→consistency curriculum, source of the α-mix),
and robotics-specific fast samplers (Consistency Policy, FlowPolicy, ManiFlow,
FreqPolicy). SnapFlow's differentiators: theoretically-grounded *corrected*
consistency objective, minimal intervention (one zero-init MLP, no EMA/teacher
network), and validation on billion-parameter VLAs. It is **orthogonal** to
layer-distillation/token-pruning, so speedups compose.

---

## 3. Is this a good approach?

**Verdict: yes.** It reproduces a published SOTA method and aligns with its
design. Points to keep in mind:

**Strengths**
- Matches or beats the multi-step teacher while giving a 3–10× denoising speedup.
- Cheap: ~10% of params, short schedule, single GPU.
- Low blast radius: zero-init target-time head means the starting point is
  numerically identical to the teacher.

**Risks / caveats to document for users**
- **Optimizer state on resume.** Freezing parameters changes the optimizer's
  parameter groups. The clean way is to start phase 2 with a *fresh* optimizer
  (a separate `physicalai fit` run does this naturally; an in-run callback must
  reconfigure optimizers explicitly).
- **DDP unused parameters.** With the VLM frozen, DDP needs
  `find_unused_parameters=True`. This is already handled in
  `trainer.py:176-182` (auto-set for multi-GPU when `strategy == "auto"`).
- **`set_requires_grad()` is currently only re-applied when loading `weights_file`**
  (`smolvla/policy.py:304-325`). On a pure Lightning `ckpt_path` resume the freeze
  flags come from the checkpoint's saved hparams, so phase 2 must instantiate the
  policy with the phase-2 config (freeze on) rather than rely on the checkpoint's
  hparams. See §5 for how the callback closes this gap.
- **Hyperparameters.** Start from paper defaults: `α = 0.5`, `λ = 0.1`,
  `num_inference_steps = 1`, ~30k steps, VLM frozen (`train_expert_only: true`).

---

## 4. Current implementation audit

The math is done; the orchestration is not. References are `file:line`.

| Capability | Status | Location |
| --- | --- | --- |
| SnapFlow config flags (`snapflow_enabled/alpha/lambda/num_inference_steps`) | Done | `smolvla/config.py:142-165`, `pi05/config.py` |
| Zero-init target-time embedding `φ_s` | Done | `smolvla/model.py:812-821` |
| Mixed FM / consistency `forward` loss | Done | `smolvla/model.py:1198-1291`; `pi05/model.py:1099-1182` |
| Two-step Euler shortcut target | Done | `smolvla/model.py:1245-1289` |
| 1-NFE inference (`s = 0`, single step) | Done | `smolvla/model.py:1347-1365` |
| VLM/expert freezing primitives | Done | `smolvla/model.py:1564-1592` (`set_requires_grad`), `pi05/model.py:385-402` (`_set_requires_grad`) |
| Config → model wiring of flags | Done | `smolvla/policy.py:298-301`; `pi05/policy.py:258-277` |
| DDP frozen-param handling | Done | `trainer.py:176-182` |
| **Two-phase orchestration (toggle snapflow + freeze at a phase boundary)** | **Missing** | — |
| **Re-applying freeze on Lightning `ckpt_path` resume (not `weights_file`)** | **Gap** | `smolvla/policy.py:304-325` |
| **A reusable "snapflow plugin" abstraction (decoupled from model code)** | **Missing** | logic is inlined in each model |

**Takeaway:** an end user can *already* run two-phase SnapFlow today as two CLI
runs (phase 1 normal, phase 2 resumes with `snapflow_enabled: true` +
`train_expert_only: true`). What's missing is (a) an ergonomic, single-run or
single-recipe way to do it, and (b) a clean abstraction so the distillation logic
is not hard-wired into `VLAFlowMatching` / `Pi05Model`.

---

## 5. Design proposals for a "SnapFlow plugin"

Two designs are presented. **Option A is recommended now**; Option B is the
longer-term ideal if variants proliferate.

### Option A — Pragmatic: phase callback + config recipe (recommended)

Keep all existing model code. Add:

1. **A `SnapFlowPhaseCallback` (Lightning `Callback`).** It flips the policy into
   SnapFlow mode at a configured phase boundary and re-applies freezing, reusing
   the existing primitives:

   ```python
   # library/src/physicalai/train/callbacks.py (sketch)
   class SnapFlowPhaseCallback(Callback):
       """Enable SnapFlow distillation after `start_step`, freezing the VLM.

       Reuses the policy's existing snapflow flags and freeze primitives; only
       orchestrates *when* they turn on and re-applies requires_grad + optimizer.
       """
       def __init__(self, start_step: int, alpha: float = 0.5,
                    lambda_: float = 0.1, num_inference_steps: int = 1):
           self.start_step = start_step
           self.alpha, self.lambda_, self.nsteps = alpha, lambda_, num_inference_steps
           self._activated = False

       def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
           if self._activated or trainer.global_step < self.start_step:
               return
           self._activate(pl_module)
           # Fresh optimizer over the now-trainable params (frozen VLM excluded):
           trainer.strategy.setup_optimizers(trainer)
           self._activated = True

       def _activate(self, policy):
           m = policy.model._model                       # VLAFlowMatching / Pi05Model
           m._snapflow_enabled = True
           m._snapflow_alpha = self.alpha
           m._snapflow_lambda = self.lambda_
           m._snapflow_num_inference_steps = self.nsteps
           policy.config.snapflow_enabled = True          # so checkpoints persist it
           policy.config.train_expert_only = True
           m.vlm_with_expert.freeze_vision_encoder = True
           m.vlm_with_expert.train_expert_only = True
           m.vlm_with_expert.set_requires_grad()          # existing primitive
           policy.model.train()                            # re-applies eval() on frozen mods
   ```

   Notes / required small changes:
   - Expose thin setters (or make the callback's private-attribute pokes into a
     small public method like `policy.enable_snapflow(...)`) so the callback does
     not reach through `_`-prefixed internals. Recommended: add
     `enable_snapflow(alpha, lambda_, num_inference_steps)` to each policy that
     forwards to its model — this is the one small, clean model-side addition.
   - Optimizer reconfiguration must exclude frozen params. Reusing
     `configure_optimizers` (which already builds param groups from the config)
     and calling the strategy's optimizer setup is the least surprising path.
   - This gives a **single-run** two-phase experience:
     `phase-1 steps → callback fires → phase-2 distillation`, all in one `fit`.

2. **A documented two-phase *config* recipe (no code, works today).** For users
   who prefer two explicit runs and clean optimizer state:

   ```bash
   # Phase 1 — normal flow-matching training
   physicalai fit --config configs/physicalai/smolvla.yaml

   # Phase 2 — SnapFlow distillation, VLM frozen, warm-started from phase 1
   physicalai fit \
       --config configs/physicalai/smolvla_snapflow.yaml \   # snapflow_enabled: true, train_expert_only: true, alpha: 0.5, lambda: 0.1
       --ckpt_path ./checkpoints/phase1/last.ckpt \
       --trainer.max_steps 30000
   ```

   Provide `configs/physicalai/{smolvla,pi05}_snapflow.yaml` phase-2 templates
   (identical to the base config except the SnapFlow + freeze block).

**Pros:** minimal, low-risk, reuses everything, usable immediately, works for both
`SmolVLA` and `Pi05` with the same callback. **Cons:** the distillation math stays
inlined in the models; the callback still knows a little about model internals
(mitigated by the `enable_snapflow` setter).

### Option B — Full refactor: a pluggable distillation objective

Extract the FM-vs-SnapFlow decision out of the model into an injected strategy so
the model no longer hard-codes SnapFlow.

```python
# library/src/physicalai/policies/common/distillation.py (sketch)
class DistillationObjective(Protocol):
    def training_loss(self, model, batch_embeds, actions, noise, time) -> Tensor: ...
    def sample(self, model, prefix, noise) -> Tensor: ...          # inference loop
    @property
    def num_inference_steps(self) -> int: ...

class FlowMatchingObjective:      # today's default (K-step Euler)
    ...

class SnapFlowObjective:          # owns the α-mix, two-step shortcut, 1-NFE loop
    def __init__(self, alpha=0.5, lambda_=0.1, num_inference_steps=1): ...
```

The model would delegate:

```python
# VLAFlowMatching / Pi05Model
def forward(self, ...):    return self.objective.training_loss(self, ...)
def sample_actions(self, ...): return self.objective.sample(self, ...)
```

Wiring: the policy builds the objective from config
(`objective = SnapFlowObjective(...) if config.snapflow_enabled else FlowMatchingObjective(...)`),
and the phase callback simply **swaps `model.objective`** — no reaching into
private flags. Both `SmolVLA` and `Pi05` implement the same small set of hooks
(`_predict_velocity`, `embed_prefix/suffix`, `_sample_noise`) that the objective
calls, so one objective implementation serves both.

**Pros:** true plugin — new distillation schemes (e.g. MeanFlow, α-Flow variants)
drop in without touching model code; the phase toggle becomes a clean object swap;
SnapFlow logic lives in one shared place instead of duplicated across two models.
**Cons:** larger refactor touching both models' `forward`/`sample_actions`; must
carefully preserve current numerics, export paths (`sample_actions` is traced for
ONNX/OpenVINO export — see the export docs), and checkpoint/hparam compatibility.
Higher regression risk; needs parity tests against the current implementation.

### Recommendation

Ship **Option A** now: it delivers the two-phase workflow the user wants with a
small, reversible change and reuses the already-correct math. Treat **Option B**'s
`DistillationObjective` interface as the target architecture to migrate toward
*only if* additional distillation objectives are added — at which point the
duplication between `VLAFlowMatching` and `Pi05Model` becomes the motivating pain
point. Option A's `enable_snapflow()` setter is deliberately a stepping stone
toward Option B (it centralizes the toggle), so choosing A does not throw away
work if B is later adopted.

---

## 6. Concrete next steps (Option A)

1. Add `enable_snapflow(alpha, lambda_, num_inference_steps)` to `SmolVLA` and
   `Pi05` policies, forwarding to their models (sets the `_snapflow_*` fields +
   `config.snapflow_enabled`). This is the only model-side change.
2. Add `SnapFlowPhaseCallback` to `library/src/physicalai/train/callbacks.py`,
   with `start_step`, `alpha`, `lambda_`, `num_inference_steps`; on activation it
   calls `enable_snapflow`, flips freeze flags, calls the existing
   `set_requires_grad()` / `_set_requires_grad()`, re-runs `model.train()`, and
   reconfigures the optimizer over the now-trainable params.
3. Add phase-2 config templates `configs/physicalai/{smolvla,pi05}_snapflow.yaml`
   (base config + `snapflow_enabled: true`, `train_expert_only: true`,
   `snapflow_alpha: 0.5`, `snapflow_lambda: 0.1`, `snapflow_num_inference_steps: 1`).
4. Document both the single-run (callback) and two-run (ckpt resume) workflows in
   `docs/how-to/`.
5. Validate: (a) phase-2 loss shows FM + shortcut mixing active; (b) frozen VLM
   params report `requires_grad == False`; (c) 1-NFE benchmark success ≈ or >
   the multi-step teacher on a small LIBERO/PushT run.

---

## References

- Luan et al., *SnapFlow: One-Step Action Generation for Flow-Matching VLAs via
  Progressive Self-Distillation*, arXiv:2604.05656, 2026.
- Frans et al., *Shortcut Models*, 2025 (two-step Euler target).
- Zhang et al., *α-Flow*, 2025 (FM→consistency mixing curriculum).
- Geng et al., *MeanFlow*, 2025 (average-velocity modelling).
- Song et al., *Consistency Models*, 2023.
