# Human-pose teleop (PoC)

Status: work in progress, not officially supported. A browser webcam drives a dual-arm follower
robot directly, without a leader arm, using MediaPipe body-pose landmarks retargeted to joint
angles on the backend. See `application/backend/src/runtime/pose_retarget.py` for the retargeting
math and its documented limitations (no gripper control, no forearm/wrist roll).

Only two robot profiles are currently supported, both dual-arm with matching per-arm DOF:

- Bimanual OpenArm (`BimanualOpenArm_Follower`), unofficial, from an unmerged plugin branch (see below).
- Bimanual SO-101 (`BimanualSO101_Follower`), already in the shipped plugin manifest.

## Prerequisites

- Node `>=24.2.0`, npm `>=11.14.0`
- [`uv`](https://docs.astral.sh/uv/)
- A webcam and a Chromium-based browser (MediaPipe's WebGL delegate needs GPU acceleration for
  reasonable frame rates)
- Outbound internet access: MediaPipe's wasm runtime and pose model are fetched from a CDN at
  runtime, and the OpenArm plugin is installed straight from GitHub

If this machine has an NVIDIA GPU, pass `--extra cuda` everywhere below instead of `--extra cpu`
or `--extra xpu`.

## 1. Backend

```bash
cd application/backend
uv sync --extra cuda --extra tests
```

### Install the OpenArm plugin (optional, unofficial)

Skip this if you only plan to test with Bimanual SO-101. OpenArm support lives on an unmerged
branch of the third-party plugin monorepo:
<https://github.com/MarkRedeman/physicalai-plugins/tree/mark/feat-openarm-motorbridge>.

It is deliberately not a `pyproject.toml` dependency. That monorepo pins its own
`physicalai-studio-plugin` source, which conflicts with this repo's local editable copy during
`uv sync` resolution. Install it directly into the venv instead, the same way Studio's own
in-app plugin installer does:

```bash
uv pip install --python .venv/bin/python \
  "git+https://github.com/MarkRedeman/physicalai-plugins.git@mark/feat-openarm-motorbridge#subdirectory=packages/physicalai-openarm-plugin"
```

This drags in its own copy of `physicalai-studio-plugin` from PyPI, which shadows the local
editable one and, on this repo's CUDA machines, can also flip `torch` back to a mismatched build.
Restore both right after:

```bash
uv sync --extra cuda --extra tests --inexact
```

`--inexact` re-syncs everything this project declares (restoring the local editable
`physicalai-studio-plugin` and the right `torch` build) without uninstalling the OpenArm plugin,
since that isn't a declared dependency.

Verify both are correct:

```bash
uv run --no-sync python -c "
import physicalai_openarm_plugin, physicalai_studio_plugin, torch
print(physicalai_openarm_plugin.__file__)
print(physicalai_studio_plugin.__file__)  # should point at ../plugin, not a git checkout
print(torch.__version__, torch.cuda.is_available())
"
```

A plain `uv sync` (without `--inexact`) prunes anything not declared in `pyproject.toml`,
including the OpenArm plugin. Re-run the `uv pip install` and `uv sync --inexact` pair whenever
that happens.

The OpenArm manifest entry that drives the in-app installer (Settings > Plugins) lives at
`application/backend/src/plugins/manifest.json`. Installing a plugin through the UI requires a
backend restart to pick up the new entry point.

### Run the backend

```bash
./run.sh
```

Serves on `:7860`. First run also clones SO101/WidowX URDF assets from GitHub.

## 2. UI

```bash
cd application/ui
npm install
npm run start
```

Proxies `/api` to `:7860`. Prints its local dev URL, typically `:3000`.

## 3. Try it

1. Register a robot of one of the supported types (`BimanualOpenArm_Follower` or
   `BimanualSO101_Follower`) in a project.
2. Add it to an environment. A leader is optional, pose teleop does not need one.
3. Open that environment's robot view. A "Pose control (PoC)" switch appears next to
   "Teleoperate" once the backend reports the follower supports it.
4. Toggle it on and grant webcam permission when prompted.

## Running the tests without hardware

```bash
cd application/backend
uv run --no-sync pytest tests/runtime/test_pose_retarget.py tests/runtime/test_action_source.py -q
```

## Known limitations

- Only the two DOF profiles above are recognized. Any other robot rejects `pose` mode with
  `pose_not_supported`.
- No gripper control from body pose. MediaPipe's Pose Landmarker has no finger detail, so a Hand
  Landmarker pass is the natural follow-up.
- Joint limits default to a wide plus-or-minus-180-degree range rather than each robot's real
  hardware limits. The catalog drivers still clamp to their own limits before actuating, so this
  is an accuracy gap, not a safety one.
- Live driving only. Pose teleop does not record LeRobot episodes yet.
