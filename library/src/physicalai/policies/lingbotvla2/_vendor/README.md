# Teacher sources and licenses

These inference modules come from Robbyant's `lingbot-vla-v2` commit
`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`, under `lingbotvla/models/vla/vision_models/`.
Studio includes only the required MoGe v2, LingBot-Depth, and DINO-Video model trees, not the
upstream training entry point or its Transformers patches. Teacher-training heads/builders,
FlashAttention integrations, and streaming KV-cache paths are omitted. Frozen video extraction
uses PyTorch FlexAttention, with SDPA retained as a numerical reference.

MoGe retains its DINOv2 backbone because the released weights use offset-based position
interpolation. Stock Transformers DINOv2 uses size-based interpolation and produces different
features at resized inputs. MoRGBD and DINO-Video also retain their RGB-D and temporal behavior.

Original notices remain in each source file. MoGe's MIT license is in `MOGE_LICENSE`;
LingBot-Depth's Apache-2.0 license is in `LINGBOT_DEPTH_LICENSE`. The DINOv2 components retain
their Apache-2.0 notices. DINOv3-derived video code uses the separate `DINOV3_LICENSE.md`, copied
from Meta's repository at `6876159a11b4df116f30f667f8c9888617df0751`. It is not relicensed as Apache.

Studio changes package imports to a private namespace, disables unpinned model/backbone
downloads, loads tensor checkpoints with `weights_only=True`, and rejects incompatible video
weights/configs. Model assets resolve through the pinned `teacher_models.py` loader.
The released MoRGBD normal decoder is unused by feature extraction and is the only geometry
weight namespace explicitly skipped by that loader.
