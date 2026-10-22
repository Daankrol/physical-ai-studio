# LingBot-VLA 2.0

LingBot-VLA 2.0 combines a Qwen3-VL-4B vision-language backbone with a 36-layer MoE action expert. Its 50-step action chunk is sampled by flow matching. Training needs an explicit robot joint layout so that gripper values do not land in arm-joint slots.

For the SO-101 example, install the `lingbotvla2` and `cu128` library extras, then run:

```bash
cd library
uv sync --frozen --extra tests --extra cu128 --extra lingbotvla2
uv run --no-sync physicalai fit --config configs/physicalai/lingbotvla2/so101.yaml
```

The sample recipe uses LoRA, a batch size of one, and quantile normalization. `configs/physicalai/lingbotvla2/robotwin.yaml` shows the 14-dimensional bimanual joint map; replace its dataset ID with a LeRobot dataset before running it. The package README at `library/src/physicalai/policies/lingbotvla2/README.md` covers camera mapping, checkpoint restore and Torch export.
