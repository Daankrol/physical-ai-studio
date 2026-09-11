# Human-pose teleop (PoC)

Status: work in progress, not officially supported. This is a "pose" teleoperator: instead of a
leader arm, a follower robot's environment camera is read by a MediaPipe pose estimator running
inside the runtime session process, and the detected 3D body landmarks are retargeted to joint
angles. See `application/backend/src/runtime/pose_retarget.py` for the retargeting math and its
documented limitations (no gripper control, no forearm/wrist roll), and
`application/backend/src/runtime/pose/` for the estimator and its background worker.

The pose camera is the same camera already listed as an environment observation source. There is
no second camera subscriber: the session already reads every environment camera each control
tick, and pose estimation runs on that same frame.

Only two robot profiles are currently supported, both dual-arm with matching per-arm DOF:

- Bimanual OpenArm (`BimanualOpenArm_Follower`), unofficial, from an unmerged plugin branch (see below).
- Bimanual SO-101 (`BimanualSO101_Follower`), already in the shipped plugin manifest.

## Prerequisites

- Node `>=24.2.0`, npm `>=11.14.0`
- [`uv`](https://docs.astral.sh/uv/)
- Outbound internet access, for installing MediaPipe's Python package, downloading its pose model
  on first use, and (optionally) installing the OpenArm plugin from GitHub

If this machine has an NVIDIA GPU, pass `--extra cuda` everywhere below instead of `--extra cpu`
or `--extra xpu`.

## 1. Backend

```bash
cd application/backend
uv sync --extra cuda --extra tests
```

### Install the pose-estimation dependency

MediaPipe is deliberately not a `pyproject.toml` dependency: its PyPI metadata pulls in
`opencv-contrib-python`, which collides with this project's `opencv-python` /
`opencv-python-headless` (all three ship files at the same `cv2/` import path, so whichever
installs last silently breaks the others). Install it with `--no-deps` and its real runtime
dependencies by hand instead:

```bash
./scripts/install_pose_deps.sh
```

The script also verifies mediapipe can actually construct a `PoseLandmarker`, not just import
cleanly.

**Linux:** mediapipe's compiled task-vision library needs a real OpenGL ES library at runtime that
the pip package does not provide; without it, construction fails with:

```
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

On Debian/Ubuntu the script installs `libgles2` (via `sudo apt-get`) automatically if missing. On
other Linux distros, install whatever OS package provides an OpenGL ES 2 ICD/dispatch library.

**macOS:** the latest mediapipe (1.0.x) has a known upstream regression
([mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)): every Tasks Vision
graph with a detector/landmarker calculator, including `PoseLandmarker`, aborts the process during
graph construction, unconditionally, even with a CPU-only delegate:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
    @ mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

Not something this project's code can work around. The script installs `mediapipe==0.10.21` on
macOS instead, which is confirmed working (real pose detection tested against the same estimator
code this project uses). That release has no wheel past Python 3.12, so it also checks the venv's
Python version and tells you to recreate it on 3.12 (`uv venv --python 3.12`) if needed. That
older mediapipe also needs a working `cv2` import (unlike 1.0.x), which this project's own
`opencv-python` dependency already satisfies, so there is nothing extra to install for that.

`mediapipe==0.10.21` also requires `protobuf<5,>=4.25.3`. Since it's installed with `--no-deps`,
this project's own much newer `protobuf` stays in place otherwise, and construction fails with:

```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

(`GetPrototype` was removed in protobuf 5+.) The script downgrades `protobuf` on macOS to fix
this. Verified safe by running the full backend test suite against the downgraded version: this
project's other protobuf users (`onnx`, `onnxruntime`, `physicalai`, `transformers`) only declare
a `>=4.25.x` lower bound, none require `>=5`. Unlike mediapipe/torch/opencv, `protobuf` is a
transitive dependency pinned in `uv.lock`, so _any_ `uv sync` (with or without `--inexact`)
restores the newer version. Re-run this script after every `uv sync` on macOS.

This failure is deliberately **not silent** in the running backend either: if the pose worker
fails to start (missing native library, model download failure, ...), the session emits a
`pose_init_failed` error, shown as a dismissable warning banner on the robot panel rather than
failing invisibly. If you toggle "Pose control" and see nothing at all, check the backend logs
for `Failed to start pose estimation` first.

Without a pose teleoperator configured, none of this is imported, so skipping this step is fine
for the OpenArm/SO-101 leader-arm teleop paths. Pose estimation only loads on first use inside a
session that has a pose teleoperator.

### Install the OpenArm plugin (optional, unofficial)

Skip this if you only plan to test with Bimanual SO-101. OpenArm support lives on an unmerged
branch of the third-party plugin monorepo:
<https://github.com/MarkRedeman/physicalai-plugins/tree/mark/feat-openarm-motorbridge>.

It is deliberately not a `pyproject.toml` dependency either, for the same kind of reason as
MediaPipe: that monorepo pins its own `physicalai-studio-plugin` source, which conflicts with this
repo's local editable copy during `uv sync` resolution. Install it directly into the venv instead,
the same way Studio's own in-app plugin installer does:

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
`physicalai-studio-plugin` and the right `torch` build) without uninstalling the OpenArm plugin or
MediaPipe, since neither is a declared dependency.

Verify the venv is correct:

```bash
uv run --no-sync python -c "
import physicalai_studio_plugin, torch
print(physicalai_studio_plugin.__file__)  # should point at ../plugin, not a git checkout
print(torch.__version__, torch.cuda.is_available())
"
```

A plain `uv sync` (without `--inexact`) prunes anything not declared in `pyproject.toml`,
including the OpenArm plugin and MediaPipe. Re-run the steps above whenever that happens.

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

## 3. Configure a pose teleoperator

Pose is a teleoperator kind, alongside "leader robot" and "none", persisted on the environment
(`schemas.environment.TeleoperatorPose`, migration
`20260911_000000_c2e5b8f1a3d6_add_pose_teleoperator_camera`):

1. Register a robot in a project. Any type works for step 4's skeleton overlay (the built-in
   `SO101_Follower` needs no plugin). Only `BimanualOpenArm_Follower` and `BimanualSO101_Follower`
   are wired up to actually be _driven_ by pose (step 5); the "Bimanual SO-101" plugin
   (`physicalai-bimanual-so101-plugin`) needs installing from Settings > Plugins first, which in
   turn needs the `plugins` feature flag enabled (off by default, see below).
2. Create or edit an environment. Add a camera to it first. Pose reuses an environment camera,
   so the picker only offers cameras already in this environment.
3. Add the robot as a follower, choose "Human pose (camera)" as its teleoperator, and pick the
   camera.
4. Open the environment. A "Pose control" switch appears next to "Teleoperate" on the robot panel.
   The camera panel for the chosen camera overlays the estimated skeleton whenever the session is
   running and a person is detected, whether or not pose is actually driving the arm. This is the
   quickest way to check the model is working at all, without needing a supported dual-arm robot.
5. Toggle "Pose control" on to drive the follower from the estimated pose. On an unsupported robot
   this correctly refuses with `pose_not_supported`, shown as a warning rather than failing the
   whole panel.

### The Plugins page is hidden by default

The Plugins nav item and route are gated behind a build-time feature flag
(`PUBLIC_ENABLE_PLUGINS`), off by default, unrelated to this PoC. Without it you cannot see
Settings > Plugins at all, so you cannot install the Bimanual SO-101 (or OpenArm) plugin. Enable
it per-browser from devtools, no rebuild needed:

```js
window.setFeatureFlag("plugins", true);
```

then reload. Or set `PUBLIC_ENABLE_PLUGINS=true` in the environment before `npm run start` /
`npm run build` for a permanent enable.

## Running the tests without hardware

```bash
cd application/backend
uv run --no-sync pytest tests/runtime/test_pose_retarget.py tests/runtime/test_action_source.py \
  tests/repositories/mappers/test_project_environment_mapper.py \
  tests/repositories/test_project_environment_repo.py -q
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
- The headless YAML runtime export (`GET .../runtime-config`) still requires a leader-robot
  teleoperator; it does not yet know how to export a pose teleoperator.
- MediaPipe's pose model file is downloaded from a Google Cloud Storage URL on first use and
  cached under the backend's `cache_dir` (see `settings.py`); no offline/air-gapped support yet.
