# LingBot-VLA 2.0 implementation plan

This plan adds LingBot-VLA 2.0 as a first-party policy (`physicalai.policies.LingBotVLA2`)
under `library/src/physicalai/policies/lingbotvla2/`. It follows the policy design proposed in
[PR #1040](https://github.com/open-edge-platform/physical-ai-studio/pull/1040). That PR is still
a draft and ships docs only, with no shared base classes yet, so this policy implements the
contract locally and keeps the names the design uses. When the shared base lands, the local
copies get deleted. The integration notes below reflect `origin/main` at `f0d9239d`.

Sources used:

- Upstream code: <https://github.com/Robbyant/lingbot-vla-v2> (Apache-2.0), mainly
  `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`, `modeling_lingbot_vla.py`,
  `qwen2_action_expert.py`, `qwen3vl_in_vla.py`, `configuration_lingbot_vla.py`,
  `lingbotvla/data/vla_data/{utils,transform}.py`, `deploy/lingbot_vla_v2_policy.py`.
- Weights: `robbyant/lingbot-vla-v2-6b` (pretrain) and `robbyant/lingbot-vla-v2-6b-robotwin`
  (RoboTwin post-train, `checkpoints/global_step_50000/hf_ckpt` plus `lingbotvla_cli.yaml`).
- Paper: arXiv 2607.06403.
- In-repo references: `policies/molmoact2/` (closest to the new design: config-driven construction,
  `from_config`, checkpoint config restore, export mixin, PEFT), `policies/xr0/` (Qwen3-VL-4B
  backbone on transformers 5.x, including export helpers), and `policies/pi05/` (pi0-style joint
  VLM and action-expert attention, RTC, SnapFlow).

## 1. Scope

In scope for the first mergeable version:

1. Load the released pretrain and RoboTwin checkpoints and run inference with numerical parity
   against upstream.
2. Fine-tune (post-train) on LeRobot datasets through `physicalai fit`, using the flow-matching
   loss plus MoE router losses. Full fine-tune on multi-GPU and LoRA on one 80 GB GPU.
3. Lightning checkpoint save and restore through the saved config, with no hub access.
4. Torch export (`ExportBackend.TORCH`) so Runtime can load the policy with `InferenceModel`.
5. Registration in the library factory, CLI configs, and the Studio backend and UI.

Deferred until someone needs them:

| Deferred item                                                                          | Why it can wait                                                                                                                                                                                           | Add it when                                                                       |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Dual-query distillation losses (MoGe-2 plus LingBot-Depth teacher, DINO-Video teacher) | Only a training-time auxiliary signal. The query tokens stay in the prefix, so inference and fine-tuning still work. The teachers are about 2.7 GB of extra checkpoints and a vendored DINOv3/MoGe stack. | Fine-tuning without distillation measurably underperforms upstream post-training. |
| Muon and distributed Muon                                                              | AdamW is the upstream default and works. `torch.optim.Muon` exists in torch 2.11, so this is cheap later.                                                                                                 | Convergence gap versus upstream is measured.                                      |
| OpenVINO and ONNX export                                                               | Needs graph-safe Qwen3-VL vision, MRoPE, and MoE rewrites plus a Runtime preprocessor component. XR0 shows it is doable but large.                                                                        | Torch export is merged and a CPU or iGPU deployment target is requested.          |
| Flex attention, flash-attn ViT, fused Triton MoE kernels, sequence parallel            | Performance only. SDPA and a dense PyTorch MoE are enough for fine-tuning at the batch sizes Studio uses.                                                                                                 | Profiling shows a clear bottleneck.                                               |
| Loss-free MoE bias updates (`bias_update_speed > 0`)                                   | The RoboTwin post-train config uses `0.0`, which freezes the bias.                                                                                                                                        | A recipe needs it. It is a single `on_before_optimizer_step` hook.                |
| Delta actions (`subtract_state`) and quaternion-relative end-effector actions          | The upstream RoboTwin config uses absolute actions.                                                                                                                                                       | A dataset needs relative actions.                                                 |

## 2. The model in brief

This section lists only the facts the implementation depends on. All of them were checked against
the upstream code and the published `model.safetensors.index.json`.

Backbone: Qwen3-VL-4B-Instruct.

- Text model: 36 layers, hidden 2560, 32 query heads, 8 KV heads, head dim 128, interleaved
  MRoPE (`mrope_section [24, 20, 20]`, theta 5e6), vocab 151936.
- Vision model: 24 blocks, hidden 1024, patch 16, spatial merge 2, temporal patch 2, deepstack
  layers `[5, 11, 17]`.
- `lm_head` is deleted. Token embeddings are kept.

Action expert: a Qwen2-style decoder, also 36 layers, so it lines up with the VLM layer by layer.

- Hidden 768, 32 query heads, 8 KV heads, head dim 128 (the q/k/v widths match the VLM so the two
  streams can share one attention call), and q/k/v biases.
- All 36 MLPs are token-level MoE blocks: 32 routed experts with intermediate size 512, top-4
  with sigmoid routing, `routed_scaling_factor 4.0`, normalized top-k weights, a shared expert with
  intermediate size 704 and no shared-expert gate, and an `e_score_correction_bias` buffer.
  Expert weights are stored fused as `[E, I, H]` parameters (`experts.gate_proj`, `up_proj`,
  `down_proj`).
- Every expert-layer RMSNorm is an AdaRMSNorm (`weight`, `gamma` Linear, `beta` Linear)
  conditioned on the flow-matching time embedding (`adanorm_time: true`). The final norm is a plain
  RMSNorm (`final_norm_adanorm: false`).
- No token embedding and no `lm_head`.

Joint forward, pi0 style: at every layer both streams compute q/k/v. The q and k tensors are
concatenated along the sequence, rotated with the VLM's MRoPE (including the expert tokens), and
passed through one attention call with a 2D mask. The output is split back per stream for `o_proj`,
the residual, and the MLP. Deepstack visual features are added to the VLM hidden states at layers
0 to 2 at image-token positions.

Prefix layout for the released config:

```text
[cam0: <vs> 64 img <ve>][cam1: 66][cam2: 66][language: 72][current_query: 8][future_query: 8]
```

- Images are resized to 256x256. The Qwen image processor then gives grid (1, 16, 16), which
  merges to 8x8 = 64 tokens. `<vs>` and `<ve>` are the vision start and end token embeddings.
- Missing cameras stay in the sequence but are masked out (`img_masks`) and filled with -1.
- Language is the task wrapped with the Qwen3 chat template (user turn, no generation prompt),
  padded or truncated to 72 tokens.
- The query segments come from `prefix_query_segments(...)`. With the released flags
  (`share_future_depth_query`, `use_shared_future_task_proj`, `use_current_shared_task_proj`) there
  are exactly two 8-token segments:

  - `current = current_shared_task_proj(cat(mean_pool(depth_align_embs), mean_pool(current_video_align_embs)))`
  - `future = future_shared_task_proj(cat(mean_pool(future_depth_align_embs), mean_pool(future_video_align_embs)))`

  Here `mean_pool` reshapes 256 learned tokens to (8, 32, 2560) and averages them to (8, 2560).

- `vlm_causal: true` makes the whole prefix causal.
- Position ids come from Qwen3-VL `get_rope_index` over fake input ids (image token ids for image
  slots, EOS for query slots).

Suffix: `[state_token][50 action tokens]`.

- `state_proj(55 -> 768)`.
- Actions go through `action_in_proj`, are concatenated with the sin-cos time embedding
  (min period 4e-3, max period 4.0), then pass through `action_time_mlp_in`, SiLU, and
  `action_time_mlp_out`.
- Attention blocks: the state is its own block, the actions form one bidirectional block, and the
  suffix sees the whole prefix. Suffix position ids continue from the max prefix position.
- Output goes through `action_out_proj(768 -> 55)` over the last 50 tokens.

Flow matching:

- `x_t = t*noise + (1-t)*actions` and `u_t = noise - actions`.
- `t ~ Beta(1.5, 1.0) * 0.999 + 0.001`.
- Loss is `L1_fm` (L1 on the velocity) masked by the per-dimension joint mask, plus MoE losses:
  sequence-wise balance loss `1e-3 * mean_layers(...)` and router z-loss
  `1e-4 * mean(logsumexp(logits)^2)`.
- Inference runs 10 Euler steps from t=1 to t=0 with dt=-0.1. The prefix KV cache is filled once.

Unified 55-dimensional state and action space: the groups are laid out in the order of the
training config's `joints` list, and each group is zero-padded to its max width.

| Group               | Width | Note                   |
| ------------------- | ----- | ---------------------- |
| `arm.position`      | 14    | Two 7-DoF arms         |
| `end.position`      | 14    | Two end-effector poses |
| `effector.position` | 2     | Grippers               |
| `waist.position`    | 4     |                        |
| `head.position`     | 2     |                        |
| `base.position`     | 3     |                        |
| `hand.position`     | 12    |                        |
| (reserved)          | 4     |                        |

The first three groups (30 dims) are the same in every published config. The order of the rest is
open question Q1.

Normalization: per group, per dimension. RoboTwin uses `bounds_99_woclip`, which is
`2*(x-q01)/(q99-q01+1e-6)-1` with no clipping. This matches `NormalizationType.QUANTILES` in
`policies/utils/normalization.py`, apart from the epsilon handling. Real-robot configs use
`meanstd`.

Checkpoint layout: 1708 tensors under the `model.` prefix, stored as fp32 (about 26 GB). The prefix
namespaces are `model.qwenvl_with_expert.qwenvl.model.{visual,language_model}`,
`model.qwenvl_with_expert.qwen_expert.model.{layers,norm}`, `model.{state_proj, action_in_proj,
action_out_proj, action_time_mlp_in, action_time_mlp_out}`, the query tensors and projections, and
the training-only `*_align_head.*` resampler heads. `config.json` holds only
`{"vlm_family": "qwen3_vl"}`, so the architecture is not described by the artifact. It lives in
`lingbotvla_cli.yaml` (RoboTwin repo only) or in the upstream training YAMLs.

## 3. Mapping the new policy design onto LingBot

| Owner (PR #1040)                        | LingBot-VLA 2.0 responsibility                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Config (`LingBotVLA2Config`)            | Ordered input and output features, camera slot order, the canonical joint layout and the dataset-to-canonical slot map, every architecture value above, `chunk_size`, `n_action_steps`, `num_denoise_steps`, prompt settings, tokenizer reference, and PEFT fields. No optimizer values, no stats, no local artifact paths other than the tokenizer reference. |
| Model (`LingBotVLA2Model`)              | Flat constructor. Qwen3-VL plus the MoE expert, joint attention, flow-matching loss, and the 10-step sampler. Returns `(B, 50, 55)`. Owns the delta indices and the default PEFT targets. Never sees `Feature`.                                                                                                                                                |
| Preprocessor                            | Observation to model tensors: camera slots, 256x256 resize, Qwen patchify plus `grid_thw`, chat-template tokenization, state and action normalization, and scattering into the 55-dim canonical layout with the joint masks.                                                                                                                                   |
| Postprocessor                           | Gather the canonical dims back to dataset order, denormalize, and trim to `n_action_steps`.                                                                                                                                                                                                                                                                    |
| Pretrained resolver (`pretrained.py`)   | Hub id or path to `(config, weights_index_path)`. Reads `lingbotvla_cli.yaml` when it exists. Never builds a model.                                                                                                                                                                                                                                            |
| Policy (`LingBotVLA2`)                  | Constructor inputs, `setup("fit")` feature adoption and validation, guarded `configure_model()`, the forward, predict, and train flow, optimizer groups, and checkpoint config save and restore.                                                                                                                                                               |
| Dataset                                 | Observed features and normalization values.                                                                                                                                                                                                                                                                                                                    |
| Capability mixins                       | `PeftPolicyMixin`/`PeftConfigMixin`/`PeftModelMixin`, `RTCPolicyMixin`/`RTCModelMixin` (phase 5).                                                                                                                                                                                                                                                              |
| Export mixin (`LingBotVLA2ExportMixin`) | Schemas, sample input, and per-backend parameters, all derived from the config.                                                                                                                                                                                                                                                                                |

The following rules come straight from the design and are easy to get wrong for this model:

- The model returns the full 50-step chunk. Trimming happens only in the postprocessor or the
  Runtime `action_chunk_trimmer`. RTC needs the full chunk.
- `configure_model()` is the only place that builds the model and processors. It starts with
  `if self.model is not None: return`, does no network access, and does not depend on trainer
  state.
- Checkpoint restore rebuilds from `checkpoint["model_config"]` and must never download the 26 GB
  pretrained artifact.
- Normalization values stay in feature and processor state. They never decide architecture. This
  is easy here because the canonical 55-dim space makes the architecture independent of the
  dataset action width.

## 4. File layout

```text
library/src/physicalai/policies/lingbotvla2/
|-- __init__.py          # exports LingBotVLA2, LingBotVLA2Config, LingBotVLA2Model
|-- README.md            # user-facing loading/training/export notes (mirror molmoact2/README.md)
|-- config.py            # LingBotVLA2Config(PeftConfigMixin, Config) + layout helpers
|-- model.py             # LingBotVLA2Model: flow matching, sampler, losses, from_config
|-- components/
|   |-- joint.py         # Qwen3-VL + expert joint layer loop, attention, MRoPE, deepstack
|   |-- expert.py        # Qwen2 expert layer, AdaRMSNorm, MoE block (dense PyTorch)
|   `-- prefix.py        # image/lang/query embedding + position ids + 2D masks
|-- preprocessor.py      # LingBotVLA2Preprocessor
|-- postprocessor.py     # LingBotVLA2Postprocessor + make_lingbotvla2_processors(config)
|-- pretrained.py        # resolve_pretrained(name_or_path) -> (config, weights_index)
|-- export.py            # LingBotVLA2ExportMixin
`-- policy.py            # LingBotVLA2(PeftPolicyMixin, LingBotVLA2ExportMixin, Policy)
```

Headers: keep the upstream Robbyant copyright line above the Intel one on files that port upstream
code, as `molmoact2/*.py` does for AllenAI.

Reuse before writing anything new:

- Stock `transformers.Qwen3VLForConditionalGeneration` (5.15.x) for the VLM modules and weights.
  Do not monkeypatch transformers the way upstream `apply_lingbot_qwen3_vl_patch()` does. That
  patch is global and would break XR0 in the same process. Drive the stock layer submodules
  (`input_layernorm`, `self_attn.q_proj/k_proj/v_proj/q_norm/k_norm/o_proj`,
  `post_attention_layernorm`, `mlp`) from our own loop, the same way `pi05/model.py` drives
  PaliGemma and Gemma layers.
- `XR0Qwen3VL` shows how to get MRoPE position ids and vision geometry on transformers 5.x.
  Borrow its approach, and its `export_openvino.py` helpers later.
- `molmoact2/model.py::load_weights` streams sharded safetensors into the model one shard at a
  time. Lift it into `policies/utils/weights.py` with a `key_map` callback so both policies share
  it. This is the second user, and without the callback it would be copied.
- `policies/utils/normalization.py::FeatureNormalizeTransform` (`QUANTILES`, `MEAN_STD`) for
  state and action normalization.
- `policies/mixins/peft` for LoRA, `policies/mixins/rtc` for RTC.

## 5. Component design

### 5.1 `config.py`

```python
@dataclass(frozen=True, kw_only=True)
class LingBotVLA2Config(PeftConfigMixin, Config):
    # feature contract (ordered)
    input_features: list[Feature]
    output_features: list[Feature]
    camera_slots: tuple[str, ...] = ("camera_top", "camera_wrist_left", "camera_wrist_right")
    camera_map: dict[str, str] = field(default_factory=dict)       # dataset image name -> slot
    joint_layout: tuple[tuple[str, int], ...] = DEFAULT_JOINT_LAYOUT  # canonical groups, sum <= 55
    state_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)   # group -> dataset slices
    action_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # horizons
    chunk_size: int = 50
    n_action_steps: int = 50
    num_denoise_steps: int = 10
    # canonical widths
    max_state_dim: int = 55
    max_action_dim: int = 55
    # prompt / image
    tokenizer_name_or_path: str = "robbyant/lingbot-vla-v2-6b"
    tokenizer_revision: str | None = "<pinned sha>"
    tokenizer_max_length: int = 72
    image_size: int = 256
    # VLM (Qwen3-VL-4B), expert, MoE, queries, flow matching: flat fields with released defaults
    ...
    loss_type: Literal["fm", "L1_fm"] = "L1_fm"
    sequence_wise_loss_coeff: float = 1e-3
    router_z_loss_coeff: float = 1e-4
    action_fp32: bool = False
```

Notes:

- Store all architecture values as flat fields, the way `MolmoAct2Config` does, including the
  Qwen3-VL text and vision config values. `model.py` builds `Qwen3VLConfig(...)` from them, so
  `configure_model()` never calls `AutoConfig.from_pretrained` (no network, deterministic restore).
  Upstream calls `AutoConfig.from_pretrained(tokenizer_path)` on every construction. That goes
  away.
- The query-token flags collapse to what the checkpoint needs: `num_task_tokens=8`,
  `num_query_backbone_tokens=256`, and `query_layout: Literal["shared_current_future"]`. Only the
  released layout is supported. Any other value raises. Add others when a checkpoint uses them.
- `distillation_heads: bool = False`. When False the model does not build the `*_align_head`
  modules and the weight loader skips those keys. The heads have no effect on inference or on the
  flow loss.
- `__post_init__` validates:
  - `n_action_steps <= chunk_size`;
  - the `joint_layout` total is at most `max_*_dim`;
  - every `state_map` and `action_map` slice fits both the dataset feature width and the group
    width, and no two slices overlap;
  - `camera_map` values are in `camera_slots`;
  - the expert head count and head dim match the VLM (required for joint attention);
  - `len(token_moe_layers) <= num_layers`.
- Maps come from `robot_layout: Literal["so101", "bimanual_6dof"] | None` presets or from explicit
  `state_map`/`action_map`. Explicit maps win. With neither, `__post_init__` raises. There is no
  contiguous default, because packing a flat vector from dim 0 puts grippers into arm slots and
  breaks the pretrained layout (see 5.5). Studio's `TrainingJobSpec` does not currently carry a
  robot layout; phase 4 must pass a selected preset (or explicit maps) into the policy constructor
  rather than guess it from the 6D or 14D feature shape. The resolved maps are written into the
  config so checkpoints and exports stay self-describing.
- `_PEFT_EXCLUSIVE_FLAGS = {"train_expert_only": False, "freeze_vision_encoder": False}`.

### 5.2 `components/expert.py`

- `AdaRMSNorm(hidden, cond_dim, eps)`: FiLM on top of an fp32 RMSNorm with zero-initialized
  `gamma` and `beta`. The parameter names must match the checkpoint (`weight`, `gamma.*`,
  `beta.*`).
- `TokenMoE`: `gate` Linear (routing logits computed in fp32 with autocast disabled, as upstream
  does, because bf16 flips top-k on ties), `e_score_correction_bias` persistent buffer, fused
  `experts.{gate_proj,up_proj,down_proj}` 3D parameters, and `shared_expert` MLP. Returns
  `(hidden, router_logits)`.
  - Compute: gather the top-k expert weights per token with `index_select` on the fused
    parameters and run a batched matmul, or compute all 32 experts densely and weight them. With
    51 tokens per sample either is cheap. Pick the dense path first because it is export-friendly
    and has no data-dependent shapes. Mark it with a `ponytail:` comment naming the ceiling
    (compute grows with E; switch to grouped GEMM if profiling shows it).
  - Keep `tokens_per_expert` only if the bias hook is added (deferred).
- `ExpertLayer`: Qwen2 attention projections with bias, `AdaRMSNorm` input and post-attention
  norms, and `TokenMoE` as the MLP. It exposes `qkv(hidden, cond)` and
  `finish(hidden, attn_out, cond)`.

### 5.3 `components/joint.py`

- `JointBackbone(vlm: Qwen3VLForConditionalGeneration, expert_layers: nn.ModuleList, expert_norm)`.
- `forward(prefix_embs | None, suffix_embs | None, mask_2d, position_ids, kv_cache, fill_cache, ada_cond, visual_pos_masks, deepstack)`
  implements the per-layer loop:
  1. q/k/v for each present stream. The VLM path applies `q_norm`/`k_norm` after the projection.
  2. Concatenate along the sequence and apply the VLM `rotary_emb` with MRoPE ids to the combined
     q/k.
  3. Update or concatenate the KV cache.
  4. One SDPA call with GQA (`enable_gqa=True`, or `repeat_kv`) and a boolean mask.
  5. Split per stream, then `o_proj`, residual, norm, MLP, residual.
  6. Deepstack add on the VLM stream for `layer_idx < 3`.
  7. Collect router logits.
- Upstream casts q/k/v to fp32 before attention. Keep that under a `attention_fp32: bool = True`
  config flag so parity tests pass. Revisit it for speed later.
- Gradient checkpointing: wrap each layer iteration with `torch.utils.checkpoint` when enabled
  (the pattern in `pi05/model.py::_apply_checkpoint`).

### 5.4 `components/prefix.py` and `model.py`

- `embed_prefix(pixel_values, image_grid_thw, img_masks, lang_tokens, lang_masks)` produces
  embeddings, pad mask, attention mask, MRoPE position ids, visual position mask, and deepstack
  features, with the layout from section 2.
  - Vision: call the stock `vlm.model.visual(pixel_values, grid_thw=...)` and use its
    `pooler_output` and `deepstack_features`. Borrow the call convention from XR0.
  - Position ids: `vlm.model.get_rope_index` over fake ids. Upstream builds a fake id tensor; do the
    same.
  - Since the image geometry is fixed (256x256, 3 slots), cache the vision geometry once per model,
    as upstream `precompute_grid_thw` does.
- `LingBotVLA2Model(PeftModelMixin, RTCModelMixin, Model)` has a flat `__init__` of plain values
  and a `from_config(config)` classmethod that picks the matching fields. If #1040's
  `TemplateModel.from_config` lands, delete ours.
  - `compute_loss(batch)` returns `(loss, {"loss", "flow_loss", "moe_seq_loss", "moe_z_loss"})`.
    The flow loss is the L1 (or MSE) velocity loss masked by `joint_mask` of shape
    `(B, chunk, 55)` and by `action_is_pad` along time. Upstream ignores `action_is_pad`; the repo
    masks padded horizon steps elsewhere (`test_action_pad_masking_regression.py`). Mask them and
    record the deviation.
  - Sequence-wise balance loss in plain PyTorch, per sample over 51 tokens, following the DeepSeek
    V3 formula. The Triton kernel is only a speed-up. Validate against the kernel output on one CUDA
    batch in a test.
  - `compute_val_loss(batch)` gives the action MSE after full denoising plus the flow loss, like
    MolmoAct2 and Pi05.
  - `predict_action_chunk(batch, noise=None)`: prefix pass that fills the cache, 10 Euler steps,
    returns `(B, 50, 55)`. `noise` is injectable for parity tests and RTC.
  - `action_delta_indices = list(range(chunk_size))`; observation and reward indices are `None`.
  - `get_default_peft_targets()` returns a regex over the VLM text and vision linears and the expert
    attention linears. Leave the MoE fused parameters and the projections fully trainable through
    `modules_to_save`-style unfreezing, the way MolmoAct2 keeps its action expert trainable.
    Experts are not `nn.Linear`, so PEFT cannot adapt them anyway.
  - `load_weights(index_path)` uses the lifted shared streaming loader with
    `key_map = strip "model." prefix`, and drops `*_align_head.*` when
    `distillation_heads=False`. It is strict on everything else and reports
    missing and unexpected keys.
  - `freeze_vision_encoder()` and `train_expert_only()` exist only if needed for memory; see 7.3.

Module names mirror upstream (`qwenvl_with_expert.qwenvl`, `qwenvl_with_expert.qwen_expert.model.layers`,
`state_proj`, ...) so the key map stays a prefix strip. This costs nothing and turns weight loading
into one line.

### 5.5 Processors

`LingBotVLA2Preprocessor(nn.Module)`, built from config plus features:

1. Images. For each slot in `camera_slots`, take the mapped dataset image or mark it missing.
   Convert to uint8-range float, resize to `image_size` with bilinear antialiasing (upstream
   `torchvision.v2.Resize`, which does not preserve the aspect ratio), normalize with mean and std
   0.5, duplicate along time (`temporal_patch_size=2`), and patchify into Qwen2-VL flat patches
   `(B*3*256, 3*2*16*16)` in torch. The geometry is constant, so there is no HF image processor in
   the training loop. `image_grid_thw` is the constant `[[1, 16, 16]] * 3`. Missing slots get -1
   pixel values and `img_masks=False`.
   - Watch out: upstream decodes images to uint8 and lets the processor rescale by /255. Studio
     datasets give float in [0, 1]. Be careful not to rescale twice (upstream has a comment about
     exactly this bug).
2. Language. Wrap the task with the Qwen3 chat template (`apply_chat_template([{"role": "user",
"content": task}], tokenize=False, add_generation_prompt=False)`), tokenize with right
   padding to 72, and return `lang_tokens` and `lang_masks`. Load the tokenizer lazily from
   `tokenizer_name_or_path@revision`; the LingBot hub repo already ships the Qwen tokenizer files.
3. State and action. Normalize the flat dataset vectors with `FeatureNormalizeTransform` using the
   feature stats (`QUANTILES` by default, `MEAN_STD` selectable through
   `normalization_mode`), then scatter into 55 dims using `state_map` and `action_map`. Build
   `state_joint_mask`, `action_joint_mask`, and `joint_mask`. Padding stays 0 after normalization,
   as upstream does. Pass `action_is_pad` through.

`LingBotVLA2Postprocessor(nn.Module)` gathers the canonical dims back to flat dataset order through
`action_map`, denormalizes, and trims to `n_action_steps`.

`make_lingbotvla2_processors(config) -> (pre, post)`. Normalization comes from
`feature.normalization_data`, so replacing features with `set_features` rebuilds the processors
and never touches the model.

RoboTwin mapping example, used by the parity test and documented in the README:

```yaml
state_map:
  arm.position: [[0, 6], [7, 13]]
  effector.position: [[6, 7], [13, 14]]
action_map: # same slices
camera_map:
  {
    cam_high: camera_top,
    cam_left_wrist: camera_wrist_left,
    cam_right_wrist: camera_wrist_right,
  }
```

The pretrained layout follows the paper (section 3.1.3) and the upstream `pad_and_concat`: each
group is filled from its first slot and zero-padded. Dual-arm robots pack the left arm into
`arm.position[0:6]` and the right arm into `[6:12]`; single-arm robots use the first 6 or 7 slots.
Grippers go to `effector.position` (left/single first). End-effector pose stays padded and masked
when the dataset has no pose. The two presets:

| Preset          | Robots                                             | Dataset vector                                   | Canonical placement                                                      |
| --------------- | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------ |
| `so101`         | SO-100/101                                         | `[5 joints, gripper]`                            | `arm.position[0:5]`, `effector.position[0:1]`                            |
| `bimanual_6dof` | Bimanual Trossen WidowX AI, ALOHA, AgileX/RoboTwin | `[L 6 joints, L gripper, R 6 joints, R gripper]` | `arm.position[0:12]`, `effector.position[0:2]` (the RoboTwin maps above) |

Camera presets map the bimanual `cam_high`/`cam_left_wrist`/`cam_right_wrist` names onto the three
slots and map SO101 front and wrist cameras to `camera_top` and `camera_wrist_left`, leaving the
third slot masked. A fourth camera such as `cam_low` cannot be silently dropped: the current
Studio mapping step rejects datasets with more cameras than slots. Use a three-camera dataset or
add an explicit dataset camera-selection step before training.

### 5.6 `pretrained.py`

`resolve_pretrained(name_or_path, *, revision) -> tuple[LingBotVLA2Config | None, Path]`:

- `snapshot_download` with `allow_patterns` for the index, the shards, the tokenizer files, and
  `lingbotvla_cli.yaml`. Skip `depth/` and `dino_video/`, which are about 2.7 GB of teachers.
- Find the weights: a root `model.safetensors.index.json`, otherwise the single
  `checkpoints/*/hf_ckpt/` directory (the RoboTwin layout).
- If `lingbotvla_cli.yaml` exists, map its `model` and `train` keys that we support (chunk size,
  `max_*_dim`, `tokenizer_max_length`, MoE values, `loss_type`) onto config overrides. Unknown or
  unsupported values that change architecture (a different query layout, `use_moe: false`) raise
  a clear error. Performance-only keys are ignored.
- Return a partial config (architecture fields only). The policy merges in features, maps, and
  horizons. The function never builds a model, as the design requires.

Pin default revisions for `robbyant/lingbot-vla-v2-6b` and `robbyant/lingbot-vla-v2-6b-robotwin`.

### 5.7 `policy.py`

```python
class LingBotVLA2(PeftPolicyMixin, LingBotVLA2ExportMixin, Policy):
    def __init__(self, input_features=None, output_features=None,
                 pretrained_name_or_path: str | Path | None = "robbyant/lingbot-vla-v2-6b", *,
                 chunk_size=50, n_action_steps=50, num_denoise_steps=10,
                 robot_layout=None, image_key_reorder_map=None, num_cameras=3,
                 state_map=None, action_map=None,
                 normalization_mode="QUANTILES",
                 gradient_checkpointing=False, compile_model=False,
                 lora_enabled=False, lora_rank=32, lora_alpha=None, lora_target_modules=None,
                 train_expert_only=False, freeze_vision_encoder=False,
                 optimizer_lr=1e-4, optimizer_vit_lr=1e-6, optimizer_betas=(0.9, 0.95),
                 optimizer_weight_decay=0.0, optimizer_grad_clip_norm=1.0,
                 scheduler_warmup_steps=0, scheduler_decay_lr=5e-5): ...
```

- Construction order (from the template): store inputs, `super().__init__(n_action_steps)`,
  `self.model = None`, `self._config = None`, processors set to `None`,
  `save_hyperparameters(ignore=["input_features", "output_features"])`. Eager
  `configure_model()` only when both features are given. A pretrained path alone is not enough,
  because the dataset layout is unknown until the features arrive. This differs from MolmoAct2's
  `norm_tag` route, since the LingBot artifacts carry no per-embodiment feature metadata.
- `configure_model()`:
  1. Return if `self.model` is already set.
  2. Resolve the config: an explicit `_config`, or the pretrained partial config merged with
     features, maps, and horizons, or defaults merged with features.
  3. `LingBotVLA2Model.from_config(config)`.
  4. Build the processors.
  5. `model.load_weights(...)` only on the pretrained route.
  6. `_apply_model_modifications()`: gradient checkpointing, freeze flags, `_inject_lora()` when
     `config.use_lora`, compile. This is the fixed order from the design: construct, load base
     weights, inject LoRA, then Lightning loads checkpoint tensors.
- `from_config(config, **training_options)` constructs with `pretrained_name_or_path=None`, sets
  `_config`, then calls `configure_model()`.
- `setup("fit")` reads `train_dataset.observation_features` and `action_features`. With no config
  yet it adopts them and calls `configure_model()`. With an existing config it validates. If only
  names, stats, or order differ and the canonical maps stay valid, it calls `set_features` and
  rebuilds the processors only. It raises when widths make the maps invalid. It never rebuilds the
  model.
- `set_features` and `rename_features` follow the template.
- `forward`, `predict_action_chunk`, `training_step`, and `compute_val_loss` are the template's
  short, explicit flow. `predict_action_chunk` returns `postprocessor(model.predict_action_chunk(...))`.
- `configure_optimizers`: AdamW with two groups, the vision tower at `optimizer_vit_lr` (upstream
  uses 1e-6) and everything else at `optimizer_lr`. Use a cosine schedule to `scheduler_decay_lr`
  over `trainer.estimated_stepping_batches` (upstream RoboTwin: 1e-4 to 5e-5 cosine; real robot:
  constant 5e-5). Apply `lora_lr_scale` when LoRA is on. Clip gradients with Lightning's
  `gradient_clip_val` (1.0) in the YAML. Do not write a custom clipper.
- Checkpoints: `on_save_checkpoint` writes `checkpoint["model_config"] = config.to_dict()` (the
  key the template uses). `on_load_checkpoint` restores the config and calls `configure_model()`,
  or compares against the existing config. `load_from_checkpoint` forces
  `pretrained_name_or_path=None`. Preserve cooperative `super()` calls because
  `RTCPolicyMixin` also implements these hooks.

`policy.py` should stay short enough to read in one pass. Export lives in `export.py`, and
architecture resolution lives in `pretrained.py` and `config.py`.

### 5.8 `export.py` (Torch first)

- `inputs_schema`: the state feature, one image feature per mapped camera (`images.<name>` naming
  as in the template), and `task` as a string. All derived from the config.
- `outputs_schema`: `action` with shape `(chunk_size, action_dim)`.
- `extra_export_args["torch"] = TorchExportParameters(preprocessors_specs=[to_float_tensor],
postprocessors_specs=[action_chunk_trimmer if chunk_size != n_action_steps])`. The Torch
  adapter calls `load_from_checkpoint`, which runs our own preprocessor and postprocessor, so this
  needs nothing on the Runtime side.
- `get_supported_export_backends() -> [ExportBackend.TORCH]`. Advertise only what is tested.
- The torch export checkpoint is about 12 GB in bf16. Check the Runtime Torch adapter load time
  and memory, and document them.

## 6. Construction routes

| Route                                                                         | Config source                                  | Weights                         | Notes                                               |
| ----------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------- | --------------------------------------------------- |
| `LingBotVLA2(input_features, output_features)` with the default pretrained id | Resolver partial config plus features and maps | Hub shards                      | Normal fine-tune entry point                        |
| `LingBotVLA2(...)` in YAML without features                                   | Deferred to `setup("fit")`                     | Hub shards at `configure_model` | CLI route                                           |
| `LingBotVLA2(pretrained_name_or_path=None, input_features, output_features)`  | Defaults plus features                         | Random                          | Unit tests with a tiny config                       |
| `LingBotVLA2.from_config(cfg)`                                                | Caller                                         | None                            | Studio and tests                                    |
| `LingBotVLA2.load_from_checkpoint(path)`                                      | `checkpoint["model_config"]`                   | Lightning state dict            | Never touches the hub except for the tokenizer (Q3) |
| Runtime `InferenceModel(export_dir)` (Torch)                                  | Via `load_from_checkpoint`                     | Exported state dict             |                                                     |

## 7. Training

### 7.1 Data

Use `LeRobotDataModule` with `data_format: physicalai`. The policy needs `action_delta_indices`
over 50 steps, with no observation history. Upstream `use_future_image` exists only for the
deferred distillation teachers, so skip it.

### 7.2 Losses and masking

Covered in 5.4. Log `train/loss`, `train/flow_loss`, `train/moe_seq_loss`, and
`train/moe_z_loss`. Upstream's per-layer MoE health metrics (maxvio, dead experts) go into one
optional debug callback later, not into the policy.

### 7.3 Memory budget (H100 80 GB, bf16-mixed)

The checkpoint is 25.5 GB of fp32 tensors, about 6.3B parameters: roughly 4.4B in the VLM and
1.8B in the action expert (the MoE experts are most of that). The peaks below are estimates.

| Mode                                    | Trainable                | Estimated peak    | Status                                    |
| --------------------------------------- | ------------------------ | ----------------- | ----------------------------------------- |
| Full fine-tune, AdamW fp32 states       | 6.3B                     | >100 GB           | Needs FSDP or DeepSpeed on 2 or more GPUs |
| LoRA on the VLM plus full action expert | about 1.8B plus adapters | about 45 to 60 GB | Default single-GPU recipe                 |
| `train_expert_only`                     | about 1.8B               | about 35 to 45 GB | Cheapest; the VLM stays frozen            |

Measure these numbers in phase 3 and put the measured batch-1 peak into the UI `minVRAM`, as the
existing entries do. Enable gradient checkpointing by default in the YAML.

### 7.4 Configs

Add `library/configs/physicalai/lingbotvla2/so101.yaml` (LoRA, single GPU) and
`library/configs/physicalai/lingbotvla2/robotwin.yaml` (full fine-tune, multi-GPU FSDP,
explicit maps). Mirror `configs/physicalai/molmoact2/so101.yaml`: comment every non-obvious
field, `ModelCheckpoint` on `val/loss`, `bf16-mixed`.

## 8. Registration and integration

Library:

- Discovery reads `policies/lingbotvla2/policy.py` without importing optional dependencies. Keep
  exactly one direct `Policy` subclass there. No manual factory dispatch or runtime registration
  is needed: `get_policy("lingbotvla2")` and `list_physicalai_policies()` discover it.
- `policies/lingbotvla2/__init__.py` exports the three classes; add them to the `TYPE_CHECKING`
  imports in `policies/__init__.py` so lazy root exports also type-check.
- `library/pyproject.toml`: a `lingbotvla2` extra (`transformers>=5.15.1,<5.16.0`,
  `safetensors`, `peft`, `Pillow`) added to `policies`. No flash-attn and no triton pins.
- `tests/unit/cli/test_cli.py`: parser tests for both YAMLs.
- `library/docs/explanation/policy/`: a short page, or extend whichever policy index #1040 lands.
  Put the policy README in the package.

Backend (`application/backend`):

- `src/training/policy_catalog.py`: add the ID, display name, description, and measured batch-1
  `min_vram` to `_STUDIO_POLICIES`. The `/api/policies/available` and `/backends` endpoints now
  discover installed policies on the selected local or remote training target. The policy extra
  must be installed on that target; an absent or incompatible extra makes it unavailable.
- `src/api/policies.py`: add only
  `_HUGGINGFACE_REQUIREMENTS["lingbotvla2"] = (("robbyant/lingbot-vla-v2-6b", False),)`. The
  tokenizer comes from the same repo, so there is no second Qwen requirement, unlike XR0.
- `src/services/model_import_service.py`: add it to `_SUPPORTED_POLICIES`. There is no longer a
  policy allowlist in `src/trainer/schemas.py`.
- `src/training/job.py`: add it to `PRETRAINED_BASE_CHECKPOINTS` and `PEFT_POLICIES`. Accept the
  existing `image_key_reorder_map` (dataset camera name -> slot index) and `num_cameras` kwargs,
  then translate them to the config's `camera_map`. Add `robot_layout` to `TrainingJobSpec` and
  forward it to this policy; do not infer gripper slots from feature width. Resume keeps the saved
  maps from the checkpoint.
- `application/backend/pyproject.toml`: include the `lingbotvla2` library extra in the backend
  dependency. Update both library and backend lockfiles and check the CUDA trainer image installs
  it. Remote trainers only advertise it when their installed image has the extra.
- Extend `tests/api/test_policy_huggingface_access.py`,
  `tests/services/test_remote_policy_catalog.py`, `tests/services/test_model_import_service.py`,
  and `tests/training/test_job.py` for discovery, target availability, layout forwarding, and
  pretrained loading. Change `application/cloudformation/remote-trainer.yaml` only if the AWS
  presets need a new instance type.

UI (`application/ui`):

- `features/models/train-model-dialog/policies.ts` now only formats VRAM. Do not add a static
  model entry. The wizard renders the selected target's `/api/policies/available` response.
- `features/models/train-model-dialog/policy-camera-slots.ts`: add three LingBot slots; the
  feature-mapping wizard sends `image_key_reorder_map` and `num_cameras`. Confirm the constructor
  accepts both, since `build_policy` otherwise warns and ignores camera mapping.
- Expose `robot_layout` for SO101 or bimanual 6-DoF training. Pick a default from the project's
  robot only if its type is available and unambiguous; otherwise require an explicit choice.
- `features/models/shared/peft.ts`: add it to `PEFT_POLICIES` after the LoRA training path works.
- Update focused wizard tests for target availability, layout selection, and camera slots.
- Following the global instructions: run `npm run type-check` in `application/ui` first, then
  `prek` with `SKIP=ui-type-check`.

## 9. Testing and validation

Unit tests go under `library/tests/unit/policies/lingbotvla2/`. They use a tiny config (2 layers,
hidden 64, 4 experts, 1 camera, grid 2x2) and run on CPU in seconds.

- `test_config.py`: validation errors, preset map resolution, `to_dict`/`from_dict` round trip,
  and resolver YAML mapping (use a fixture YAML, no network).
- `test_processors.py`:
  - canonical scatter and gather round-trip with the RoboTwin maps;
  - masks;
  - `QUANTILES` normalization matches upstream `bounds_99_woclip` within tolerance;
  - the torch patchify equals `Qwen2VLImageProcessorFast` output for a 256x256 image;
  - the chat-template tokens equal upstream `prepare_language`;
  - the float-versus-uint8 rescale regression.
- `test_model.py`:
  - output shapes, `(B, 50, 55)` from `predict_action_chunk`;
  - the loss is finite and backprop reaches the expert, projections, and VLM;
  - `joint_mask` and `action_is_pad` zero the right terms;
  - the MoE dense path equals a naive per-token loop;
  - AdaRMSNorm with zero `gamma`/`beta` equals RMSNorm;
  - the KV-cached sampler equals an uncached full forward at a fixed noise;
  - `load_weights` with a synthetic sharded checkpoint (prefix strip, align-head skip, strict
    error on unknown keys).
- `test_policy.py`:
  - explicit config construction;
  - `configure_model()` twice keeps the same object;
  - `setup("fit")` adopts features and then rejects incompatible widths;
  - `set_features` rebuilds processors but not the model;
  - checkpoint save and restore with `pretrained_name_or_path` mocked to raise, proving the hub is
    never hit;
  - the LoRA injection order (adapters present before `load_state_dict`);
  - postprocessor trimming.
- `test_export.py`: Torch export writes a manifest, and the Runtime adapter restores it and
  produces identical actions for fixed noise.
- Discovery test: `tests/unit/policies/test_policy_discovery.py` checks that the policy is listed
  without importing heavy dependencies; `test_get_policy.py` covers construction.

Integration tests (GPU, marked like the existing `test_molmoact2_e2e.py`):

- `test_lingbotvla2_parity.py`. The reference outputs come from the upstream repo run once in its
  own environment (torch 2.8, transformers 4.57.3) through `deploy/lingbot_vla_v2_policy.py` with a
  fixed seed and noise, fp32, `attention_implementation=eager`. Commit the captured tensors (or
  store them as a small HF dataset) instead of installing upstream in CI. Compare:

  1. prefix embeddings and position ids;
  2. the velocity at t=1;
  3. the final 50x55 normalized chunk.

  Targets are 1e-4 relative in fp32. For bf16, record the observed error.

- `test_lingbotvla2_e2e.py`: `physicalai fit --config configs/physicalai/lingbotvla2/so101.yaml
--trainer.fast_dev_run=true` plus predict plus Torch export on a small dataset.
- Benchmark, not in CI: evaluate the RoboTwin checkpoint on a handful of RoboTwin tasks if the gym
  is available. Otherwise run the upstream `scripts/open_loop_eval.py` method (open-loop MSE on
  held-out episodes) through our policy and compare against upstream numbers.

Before pushing, from the repo root: `uvx prek run --from-ref origin/main --to-ref HEAD`. For the
library, sync with `uv sync --frozen --extra tests --extra cu128 --extra lingbotvla2` and then use
`uv run --no-sync pytest ...`. Never use a bare `uv run`, because of the XPU fallback index.

## 10. Phases

Each phase ends green (tests plus prek) and can be reviewed on its own.

| Phase                        | Deliverable                                                                                                                         | Done when                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 0. Reference capture         | Script (kept out of the repo, or under `tools/`) that runs upstream inference on fixed inputs and saves tensors. Answers Q1 and Q2. | Reference tensors for the pretrain and RoboTwin checkpoints exist.                                               |
| 1. Model core                | `config.py`, `components/*`, `model.py`, weight loading, shared loader lifted from MolmoAct2                                        | Tiny-config unit tests pass. Real checkpoint loads strict (minus align heads). Parity items 1 to 3 pass in fp32. |
| 2. Processors and policy     | `preprocessor.py`, `postprocessor.py`, `pretrained.py`, `policy.py`, registration, `__init__`                                       | Policy unit tests pass. The RoboTwin checkpoint reproduces the reference chunk from raw observations.            |
| 3. Training                  | Losses, optimizer groups, LoRA and freeze modes, YAML configs, memory measurements                                                  | `fast_dev_run` passes. A short SO101 run lowers val loss. VRAM numbers are recorded.                             |
| 4. Export and Studio         | `export.py` (Torch), backend and UI wiring, remote-trainer entry, docs and README                                                   | Export round-trip test passes. Backend and UI tests pass. Training can be started from the UI.                   |
| 5. Follow-ups (separate PRs) | RTC mixin, loss-free bias hook, Muon, distillation heads and teachers, OpenVINO export                                              | Each is driven by a concrete request (see section 1).                                                            |

Suggested PR split: phases 1 and 2 in one library PR (model plus policy, no Studio), then phase 3,
then phase 4. RTC comes after, because `RTCModelMixin` needs the sampler to take the previous
chunk. That is a small, isolated change once the sampler exists.

## 11. Risks and open questions

Q1. Canonical group order beyond the first 30 dims. The README lists hand before waist, but
`real_robot.yaml` orders the groups arm, end, effector, waist, head, base, hand. This matters only
for robots that use those groups zero-shot. Confirm it in phase 0 from the upstream pretraining
config, or ask upstream. Until then, default `joint_layout` to the `real_robot.yaml` order and
document it.

Q2. Architecture of the pretrain checkpoint. It has no CLI YAML. Its safetensors index has exactly
the same 1708 keys as the RoboTwin checkpoint, so use the RoboTwin `lingbotvla_cli.yaml` values as
defaults. The index does not include shapes, so phase 0 confirms them from the shard headers.

Q3. Tokenizer at restore time. `configure_model()` must not hit the network, but the tokenizer
lives on the hub. Follow MolmoAct2 and keep a pinned `tokenizer_name_or_path` that resolves from
the HF cache, and document it. Alternative: save the tokenizer files next to the Torch export.
Decide in phase 4.

Q4. transformers 5.15 versus upstream 4.57.3. Vision-tower internals and `get_rope_index`
signatures changed (XR0 hit this). Parity tests are the guard. Budget time for it in phase 1.

Q5. Precision. Upstream validated RoboTwin with fp32 inference and warns that bf16 changes
success rates. Default inference dtype to fp32 for the expert path (`action_fp32` and attention
in fp32), and measure the cost of running the VLM prefix in bf16.

Q6. Suffix-to-query attention. `block_future_depth_to_action` and `block_suffix_to_future_video`
are both False in the released configs, so the action suffix attends to the query tokens. Support
only False and raise otherwise.

Q7. License and attribution. Upstream is Apache-2.0 and the weights are Apache-2.0. Keep the
Robbyant copyright headers on ported files. Check the Qwen3-VL-4B license terms carried by the
weights and record them in the README.

Q8. PR #1040 may still change (`configure_model` versus `_initialize_from_config`, config
ownership of mixin flags, and whether `from_config` lives on a template base). Implement against
the latest docs version (`configure_model`) and keep the local helpers thin so switching to a
shared base later means deleting code, not rewriting it.
