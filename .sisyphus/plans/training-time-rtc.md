# Training-Time RTC for Pi05

## Paper: Training-Time Action Conditioning for Efficient Real-Time Chunking
**Source**: [arXiv:2512.05964](https://arxiv.org/abs/2512.05964) — Black, Ren, Equi, Levine (Physical Intelligence)

## Summary

Replace expensive inference-time inpainting (pseudoinverse guidance with VJPs per denoising step) with a training-time simulation of inference delay. The model learns to generate action **postfixes** conditioned on ground-truth action **prefixes**, eliminating all inference-time overhead for real-time chunking.

### Core Technique (3 changes)

1. **Per-token flow matching timesteps**: Instead of a single scalar `τ` for the whole chunk, allow each action token to have its own timestep. Prefix tokens get `τ=1.0` (ground truth), postfix tokens get the sampled `τ`.
2. **Feed ground-truth prefix**: During training, the first `d` actions (where `d` is a randomly sampled delay) are replaced with clean ground-truth actions in `x_t`. No noise is applied to the prefix.
3. **Mask loss to postfix only**: Only compute the flow matching loss on the postfix tokens (indices `d:H`), not the prefix.

At inference, the same mechanism conditions generation on a real action prefix from the previous chunk, with no VJP/backprop overhead.

---

## Current Pi05 Architecture (Relevant Parts)

| Component | File | Key Detail |
|---|---|---|
| Flow matching loss | `model.py::_flow_matching_loss` | Single scalar `time` per batch item, broadcast to all tokens |
| Time embedding | `model.py::embed_suffix` | `_create_sinusoidal_pos_embedding(timestep, ...)` takes `(batch,)` scalar |
| AdaRMS conditioning | `pi_gemma.py` | Time conditioning via `adarms_cond` — single vector per batch item |
| Denoising loop | `model.py::sample_actions` | Euler integration, uniform `time_tensor` across all tokens |
| Config | `config.py::Pi05Config` | `chunk_size=50`, `n_action_steps=50` |
| Inference runner | `action_chunking.py` | Simple queue, no RTC yet |

### Key Observation
The current architecture uses a **single time scalar per batch item** that conditions the entire action chunk uniformly. The paper requires **per-token timesteps** — prefix tokens at τ=1.0, postfix tokens at the sampled τ. This is the main architectural change.

---

## Implementation Plan

### Phase 1: Config Changes
**File**: `library/src/physicalai/policies/pi05/config.py`

Add to `Pi05Config`:
```python
# Training-Time RTC
enable_training_time_rtc: bool = False          # Feature flag
rtc_max_delay: int = 10                         # Max simulated delay (in action timesteps)
rtc_delay_sampling: Literal["uniform", "exponential"] = "uniform"  # Delay distribution
```

Validation: `rtc_max_delay < chunk_size`

**QA**:
```bash
uv run python -c "from physicalai.policies.pi05.config import Pi05Config; c = Pi05Config(enable_training_time_rtc=True, rtc_max_delay=10); print(c)"
```
Expected: Config instantiates without error. Verify `Pi05Config(enable_training_time_rtc=True, rtc_max_delay=60)` raises `ValueError` (exceeds `chunk_size=50`).

### Phase 2: Per-Token Time Embedding
**File**: `library/src/physicalai/policies/pi05/model.py`

**Problem**: `embed_suffix` currently takes `timestep: Tensor` of shape `(batch,)` and creates a single sinusoidal embedding that becomes `adarms_cond`. This single conditioning vector is broadcast to all action tokens via AdaRMS normalization (scale/shift/gate applied uniformly).

**Solution**: Extend `embed_suffix` to accept per-token timesteps `(batch, chunk_size)`:

1. Modify `_create_sinusoidal_pos_embedding` to handle `(batch, chunk_size)` input → output `(batch, chunk_size, dim)`.
2. Modify `embed_suffix` to produce per-token `adarms_cond` of shape `(batch, chunk_size, dim)` instead of `(batch, dim)`.
3. Modify the AdaRMS norm in `pi_gemma.py` to accept per-token conditioning (the scale/shift/gate differ per action token position).

**Backward compatibility**: When `enable_training_time_rtc=False`, pass the same timestep for all tokens (current behavior). The code path should be identical.

**QA**:
```bash
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_embed_suffix" -v
```
Expected: With scalar time `(B,)`, output shapes are unchanged from current behavior. With per-token time `(B, H)`, `adarms_cond` shape is `(B, H, D)` and `suffix_embs` shape is `(B, H, D)`.

### Phase 3: Training Loss Modification
**File**: `library/src/physicalai/policies/pi05/model.py::_flow_matching_loss`

When `enable_training_time_rtc=True`:

```python
# 1. Sample delay per batch item
delay = torch.randint(0, max_delay + 1, (batch_size,), device=device)  # (B,)

# 2. Build per-token time: shape (B, chunk_size)
#    prefix_mask[b, i] = True if i < delay[b]
prefix_mask = torch.arange(chunk_size, device=device)[None, :] < delay[:, None]
time_per_token = torch.where(prefix_mask, torch.ones_like(time[:, None]), time[:, None])

# 3. Build x_t with clean prefix
time_expanded = time_per_token[:, :, None]  # (B, H, 1)
x_t = time_expanded * noise + (1 - time_expanded) * actions
# When time=1.0 for prefix: x_t = 1.0 * noise + 0.0 * actions = noise... WAIT

# IMPORTANT: The paper's convention differs from this codebase!
# Paper (JAX):  x_t = τ * actions + (1-τ) * noise,  target = noise - actions
# Codebase:     x_t = t * noise + (1-t) * actions,   target = noise - actions
# So t=0 → clean actions, t=1 → pure noise (codebase convention)
# For prefix: we want CLEAN actions, so set t=0 (not 1!)
# BUT the paper sets τ=1 for prefix because their convention is τ=1 → clean.

# Reconciled: In THIS codebase, prefix tokens need time=0.0 (clean actions)
time_per_token = torch.where(prefix_mask, torch.zeros_like(time[:, None]), time[:, None])
# x_t for prefix = 0 * noise + 1 * actions = actions ✓

# 4. Mask loss to postfix only
postfix_mask = ~prefix_mask  # (B, H)
losses = F.mse_loss(u_t, v_t, reduction="none")  # (B, H, D)
losses = losses * postfix_mask[:, :, None].float()
loss = losses.sum() / (postfix_mask.sum() * action_dim + 1e-8)
```

**QA**:
```bash
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_rtc_loss" -v
```
Expected:
- With `delay=0`, loss matches standard flow matching loss (no masking applied)
- With `delay=5`, `x_t[:, :5, :]` equals ground-truth `actions[:, :5, :]` (clean prefix)
- Loss gradient is zero for prefix token outputs (masked out)
- With `enable_training_time_rtc=False`, loss is bitwise identical to current implementation

### Phase 4: Inference Modification (Library/PyTorch Path) (Library/PyTorch Path)
**Files**: `library/src/physicalai/policies/pi05/model.py`

Thread `action_prefix` through the full inference chain:

1. **`sample_actions`**: Add `action_prefix: Tensor | None = None` and `delay: int = 0` params.
   When provided, at each denoising step: replace the first `delay` tokens in `x_t` with `action_prefix`, and pass per-token timesteps to `denoise_step` (prefix=0.0, postfix=current integration time).

2. **`denoise_step`**: Accept per-token timesteps `(B, H)` instead of scalar `(B,)`.

3. **`predict_action_chunk`**: Add `action_prefix` and `delay` to signature, pass through to `sample_actions`.

4. **`forward` (eval branch)**: Accept optional `action_prefix`/`delay` from `batch` dict, pass to `predict_action_chunk`.

5. **`sample_input`**: When `enable_training_time_rtc=True`, include `action_prefix` (zeros of shape `(1, chunk_size, max_action_dim)`) and `delay` (scalar tensor) in the sample input dict so export traces the prefix-conditioned path.

**QA**:
```bash
# Verify inference with and without prefix produces valid action tensors
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_rtc_inference" -v
```
Expected: action output shape `(B, chunk_size, action_dim)`, prefix tokens in output match input prefix.

### Phase 5: Inference Runtime Integration (Separate PR)
**File**: `physicalai/src/physicalai/inference/runners/action_chunking.py`

> **Note**: This phase requires changes to export manifests, runtime adapters, and async scheduling.
> It is a separate PR from the core TT-RTC model changes (Phases 1-4, 6).

Changes needed (out of scope for this PR):
- Modify `ActionChunking` runner to trigger next-chunk inference **before** the queue is exhausted (async prefetch at `queue_remaining <= delay`)
- Pass remaining queued actions as `action_prefix` input to the adapter
- Update export manifest to include `action_prefix` and `delay` as model inputs
- Update runtime preprocessor to construct these inputs from the action queue state

### Phase 6: Wire Config → Model → Policy
**File**: `library/src/physicalai/policies/pi05/policy.py`

- Pass `enable_training_time_rtc`, `rtc_max_delay`, `rtc_delay_sampling` from config to `Pi05Model.__init__`
- Store as instance attributes on the model
- No changes to the training loop itself (`training_step` already calls `compute_loss`)

**QA**:
```bash
# Verify config flows through to model correctly
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_rtc_config" -v
```
Expected: `Pi05Model` instance has `enable_training_time_rtc=True` and correct `rtc_max_delay`.

---

## File Change Summary

| File | Changes | Complexity |
|---|---|---|
| `config.py` | Add 3 config fields + validation | Low |
| `model.py::_flow_matching_loss` | Per-token time, prefix masking, postfix-only loss | Medium |
| `model.py::embed_suffix` | Accept per-token timesteps `(B, H)` | Medium |
| `model.py::_create_sinusoidal_pos_embedding` | Handle 2D time input `(B, H)` | Low |
| `model.py::sample_actions` | Accept action_prefix, per-token time in denoising | Medium |
| `model.py::denoise_step` | Accept per-token timesteps | Medium |
| `model.py::predict_action_chunk` | Thread action_prefix through | Low |
| `model.py::forward` (eval) | Accept prefix from batch dict | Low |
| `model.py::sample_input` | Expose prefix inputs for export tracing | Low |
| `pi_gemma.py` | AdaRMS to handle per-token conditioning | Medium-High |
| `policy.py` | Pass new config params to model | Low |
| `action_chunking.py` | RTC prefix passing (**separate PR**, Phase 5) | High |
| `test_pi05.py` | Add tests for TT-RTC training + inference | Medium |

## Critical Design Decisions

### 1. Time Convention Mapping
The paper uses `τ=1 → clean, τ=0 → noise`. This codebase uses `t=0 → clean, t=1 → noise`. All prefix tokens get `t=0.0` in this codebase (equivalent to paper's `τ=1.0`).

### 2. Per-Token AdaRMS
The biggest architectural question: AdaRMS currently takes a single `(B, D)` conditioning vector. For per-token timesteps, we need `(B, H, D)`. Two approaches:
- **Option A**: Expand AdaRMS to handle per-token conditioning (change the norm to apply position-dependent scale/shift/gate). More invasive but cleaner.
- **Option B**: Run embed_suffix in a loop over unique timestep values (typically only 2: prefix=0.0 and postfix=sampled_t). Hacky but minimal change to AdaRMS.

**Recommendation**: Option A — it's the approach the paper uses (Figure 2: "The flow matching timestep differs between tokens"). This is also how DiT architectures handle per-token conditioning in general.

### 3. Delay Sampling Distribution
The paper uses:
- Uniform `[0, max_delay)` for real-world experiments
- Exponentially decreasing weights for simulation (higher delays need less supervision)
- Include `d=0` (no prefix, standard training) for backward compatibility

### 4. Backward Compatibility
When `enable_training_time_rtc=False`, the model should behave **identically** to the current implementation. No regression risk. All changes are behind the feature flag.

## Validation Plan

Each phase has inline QA (see above). Final integration validation:

```bash
# 1. Full test suite — all new + existing tests pass
uv run pytest library/tests/unit/policies/test_pi05.py -v

# 2. Backward compatibility — RTC disabled produces identical behavior
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_backward_compat" -v

# 3. Integration — training completes with RTC enabled
uv run pytest library/tests/unit/policies/test_pi05.py -k "test_rtc_fast_dev_run" -v

# 4. LSP diagnostics clean on all changed files
# Run lsp_diagnostics on: model.py, config.py, pi_gemma.py, policy.py
```

Expected: All pass, zero regressions, no new type errors.

## Implementation Order

1. Config changes (trivial, unblocks everything)
2. Per-token sinusoidal embedding (foundation for everything else)
3. Per-token AdaRMS conditioning (hardest part, needs careful design)
4. Training loss modification (core technique)
5. Inference modification (denoising with prefix)
6. Policy wiring (connect config to model)
7. Tests
8. Inference runtime RTC integration (Phase 5, can be separate PR)
