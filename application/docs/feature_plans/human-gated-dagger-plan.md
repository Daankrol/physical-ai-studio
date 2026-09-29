# Human-gated DAgger: implementation plan

## Goal

An operator runs a trained policy on the robot, takes over with the leader arm when the policy goes wrong, demonstrates the recovery, and hands control back. Each correction is saved as an ordinary episode in the model's dataset, and the operator retrains from the existing model. This is human-gated DAgger (HG-DAgger), the variant LeRobot ships as `--strategy.type=dagger`. Background and the difference from original DAgger are in the [research note](../explanation/dagger-human-in-the-loop.md).

Almost everything already exists. One `RuntimeSession` holds the policy and the leader at the same time, recording works in any mode, and the models page can already retrain from a base model. Four things are missing:

1. **Safe takeover.** Switching to `teleop` sends the leader's pose to the follower on the next tick. If the leader is far away, the follower jumps. The inference page's Teleoperate switch already has this hazard.
2. **Clean labels.** `RecordingCallback` records every non-`hold` tick, including policy ticks. It also reads `follower_source` after the action was sent, and `_finish_arm` flips that from another thread, so a tick can get the wrong label.
3. **A screen.** Inference and recording are separate pages. None of them loads a model and a dataset together.
4. **Episode identity.** The dataset list shows episode number and duration, but it cannot tell a demonstration from a saved human correction after a refresh or export.

## Why train on corrections, not the entire rollout

The training set already contains expert demonstrations of normal task execution. The new information in a human-gated rollout is the recovery: an observation in a state the policy reached, paired with what the human actually did next. Recording and training on the policy's own autonomous actions would treat its mistakes as expert targets, teaching them back to the model. LeRobot's corrections-only mode makes each human-controlled window a separate episode; its continuous mode tags both kinds of frames, but plain imitation training does not automatically filter out the autonomous ones.

An action-prediction model trains on observation/action samples (sometimes with short history and future action chunks), not a requirement that every training episode start at the task's initial pose. Short correction episodes can start mid-task; at their boundaries, history and action chunks must be padded or masked, not joined to unrelated frames. Mix the new correction episodes with the original full demonstrations, rather than training only on short recoveries. Evaluate whole-task rollouts separately to see whether the combined dataset helped. Keep the task text unchanged: it is a model input, not a place to store the correction label.

## Design

```text
policy ──Take over──▶ handover ramp ──▶ teleop (recorded) ──Save/Discard──▶ hold ──Resume──▶ policy
                      (not recorded)
```

- **Handover ramp.** Entering `teleop` blends from the last action sent to the live leader pose over `RUNTIME_TELEOP_HANDOVER_S` (default 1.0 s). The blend is relative to time, so it doesn't depend on joint units, which differ between robot plugins. Set it to `0` for the old instant behavior. It applies to every entry into `teleop`, so the Teleoperate switch and the recording page get the fix too. At the start of a recording both arms are usually at rest in the same pose, so the ramp is invisible there.
- **Per-tick provenance.** `StudioActionSource.update()` records which source produced the action it returned: `hold`, `handover`, `teleop`, or `policy`. The runtime calls `update()`, sends the action, then calls `on_tick` on the same thread (`physicalai.runtime.core.RobotRuntime.run`), so `RecordingCallback` can read that value without a race.
- **Record only human actions.** A frame is written only when that tick's source is `teleop` and the leader read on that tick succeeded. Hold, handover, stale-leader fallback, and policy ticks are never written. This changes behavior on purpose: Studio no longer records policy actions as demonstrations. No UI does that today.
- **Mark saved correction episodes.** Persist a per-episode `human_correction` label alongside the dataset, while normal demonstrations remain `demonstration`. The dataset list and episode viewer show a "Human correction" badge; the label is for review, not a training feature or a rewrite of the task.
- **No gaps inside an episode.** Arming the policy is rejected while an episode is open. A correction is always a single continuous stretch of human control.
- **No new command kinds.** The UI composes `set_follower_source`, `start_recording`, `save_episode`, `discard_episode`, and `start_task`. Extend `start_recording` with an optional `origin="human_correction"` (default `"demonstration"`); `save_episode` and `discard_episode` remain acknowledged.

## PR 1: backend (safe handover and clean labels)

This PR is useful without the UI because it fixes the takeover jump and the label race for existing pages.

### `backend/src/settings.py`

- Add `runtime_teleop_handover_s: float = Field(default=1.0, ge=0.0, alias="RUNTIME_TELEOP_HANDOVER_S")` next to `runtime_idle_timeout_s`.

### `backend/src/runtime/action_source.py`

- Constructor: add `handover_s: float = 1.0`. Compute `self._handover_ticks = max(1, round(handover_s * fps))` when `handover_s > 0`, else `0`.
- Keep `self._last_action` (the array returned last tick) and `self._last_action_source: ActionOrigin`, where `ActionOrigin = Literal["hold", "handover", "teleop", "policy"]`. Expose it as a read-only `last_action_source` property.
- `_read_leader` returns `(action, fresh: bool)`. `fresh` is `False` on the fallback paths.
- On any transition into `teleop` (in `_handle_set_follower_source`), store `self._handover_start = copy of self._last_action` (or the hold target if nothing has been sent yet) and `self._handover_step = 0`.
- In `update()`, the `teleop` branch does this: while `_handover_step < _handover_ticks`, return `start + (leader - start) * (_handover_step + 1) / _handover_ticks`, increment the step, and set the origin to `handover`. After that, return the leader action with origin `teleop` if `fresh`, otherwise origin `hold`. Set the origin to `policy` or `hold` in the other branches.
- `_arm_policy`: if `self._recording` is set and `is_recording` is true, emit `ErrorEvent(error_code="recording_in_progress", message="Save or discard the correction before resuming the policy.")` and return `False`.

### `backend/src/runtime/session.py`

- Pass `handover_s=get_settings().runtime_teleop_handover_s` when building `StudioActionSource`.
- Build `RecordingCallback` with `action_origin=lambda: action_source.last_action_source` instead of `follower_source`.

### `backend/src/runtime/callbacks/recording.py`

- `RecordingCallback.__init__` takes `action_origin: Callable[[], ActionOrigin]`. `on_tick` returns early unless `action_origin() == "teleop"`.
- `RecordingState` counts frames per episode: reset in `start()`, increment in `add_frame()` only when the frame was written. `stop_episode()` raises `RuntimeError("No frames were recorded. Discard the episode.")` **before** clearing `is_recording` when the count is zero, so the episode stays open and can still be discarded. Without this, an operator who presses Save during the handover would get an empty save, and it is unclear what the LeRobot writer does with one.
- Carry `origin` with the open episode. On `origin="human_correction"`, require a loaded model and leader as well as a loaded dataset, and do not mark an episode until a non-empty save succeeds.

### `backend/src/runtime/contract.py`, `backend/src/internal_datasets/`, `backend/src/schemas/dataset.py`

- Add the optional `origin` field to `StartRecordingCommand`. Keep the recording and frame feature schema unchanged: appending an `intervention` column to an existing LeRobot dataset would require migration.
- Store correction episode indices in a small `studio/episode_origins.json` sidecar inside the dataset root, written into the recording mutation's cache after a successful save and copied back on finalization. A missing entry means `demonstration`, so existing and imported datasets work without a conversion. Use an atomic replace when writing the sidecar so interruption cannot truncate it. After the LeRobot save, retain a pending `(episode_index, origin)` in the recording mutation until the sidecar write succeeds. If that write fails, return an error, let a repeated `save_episode` retry **the marker only** (not the already-saved episode), and refuse copy-back/finalization until it succeeds; never delete the cache or report an unlabeled correction as saved. Test this injected-failure path and recovery.
- Return `origin` (default `demonstration`) in both `EpisodeInfo` and `Episode`, using `InternalLeRobotDataset.get_episode_infos()` and `_build_episode_from_metadata()`. `DeleteEpisodesMutation` uses LeRobot's episode-deletion tool, which renumbers survivors and does not preserve Studio sidecars: remap surviving labels from old to new indices before overwriting the dataset. The sidecar travels with dataset copy, snapshot, download, and Studio import; absence on a foreign LeRobot import is expected.

### Tests (`backend/tests/runtime/`)

- `test_action_source.py`
  - Update `test_teleop_forwards_leader_positions_on_next_tick` to construct with `handover_s=0`.
  - New: the handover ramps from the hold target to the leader over N ticks, the origin is `handover` then `teleop`, and the first tick never jumps to the leader pose.
  - New: policy → teleop ramps from the last *policy* action, not from the measured position.
  - New: a failed leader read reports origin `hold`, even in `teleop`.
- `test_action_source_policy.py`: `start_task` is rejected with `recording_in_progress` while a recording is open and allowed after save or discard.
- `test_recording_callback.py`: switch the helper to `action_origin`. Frames are written only for `teleop`. `handover`, `policy`, and `hold` are skipped. Saving an empty episode raises and leaves the episode open, and discard still works.
- Dataset/episode tests: a saved correction appears as `human_correction` in both API shapes; a normal episode and an imported dataset without a sidecar appear as `demonstration`. Discard does not mark an episode; after deleting and renumbering episodes, the correct survivors keep their labels. Confirm the label survives cache copy-back and dataset export/import, while the action feature schema stays unchanged.

### Done when

```bash
# from application/backend/
uv sync --frozen --extra cuda --extra tests
uv run --no-sync pytest tests/runtime tests/api/test_runtime_ws.py
```

The existing recording and inference tests still pass unchanged, except for the teleop test that now passes `handover_s=0`.

## PR 2: UI (collect corrections)

### Entry: `ui/src/features/models/models-table/start-inference-dialog.tsx`

- Add a `Checkbox` labeled "Collect corrections into the training dataset". When checked, add `collect=1` to the search params.

### `ui/src/features/models/inference/use-inference-params.ts`

- Return `collect: searchParams.get('collect') === '1'`.

### `ui/src/features/models/inference/inference-page.tsx`

- The model's dataset is already fetched. Pass `dataset={collect ? dataset : undefined}` to `RuntimeSessionProvider`, and pass `collect` and `defaultTask` to `InferenceViewer`.

### `ui/src/features/robots/runtime-session-provider.tsx`

- `onOpen`: send `setFollowerSource('teleop')` only when a dataset is given **without** a model. The recording page keeps its behavior, and the collection page starts in `hold`.
- `startEpisode` keeps its current call shape but sends `origin: 'human_correction'` in `start_recording` only when the provider is in correction-collection mode; other recording requests use the default `demonstration`.

### `ui/src/features/models/inference/inference-viewer.tsx`

With `collect` set:

- Ready means `readyForInference && readyForRecording`. If `!state.has_leader`, show "Collecting corrections needs a leader arm" and disable the controls.
- Replace the Teleoperate switch with:
  - **Take over** (when not recording): `await setFollowerSource.mutateAsync('teleop')`, then `startEpisode.mutate(task)`.
  - **Save correction** and **Discard** (while recording): `await saveEpisode.mutateAsync()` or `await discardEpisode.mutateAsync()`, then `setFollowerSource.mutate('hold')`. On a failed save, show the error and keep Discard available.
  - Existing **Play** resumes the policy through `startTask`. Disable it while `state.is_recording`; the backend also rejects it.
  - A status line showing `episodes_recorded` corrections this session.
- Hotkeys: reuse the `recording.start_episode`, `recording.accept_episode`, and `recording.discard_episode` bindings for Take over, Save, and Discard. The operator needs to take over without reaching for the mouse.
- Without `collect`, the page stays as it is.

### Dataset episode display: `ui/src/features/datasets/episodes/episode-tag.tsx` and `ui/src/routes/datasets/`

- After API types are regenerated, show a compact "Human correction" badge next to the episode number/duration in the existing shared `EpisodeTag` so it appears in the episode list and detail header. Plain demonstrations keep their current display. The label is episode-level: there are no mixed-control frames to shade within a correction episode.

### Tests

- Episode list and viewer tests: the badge survives a refresh from `EpisodeInfo` and `Episode`, and never appears for demonstrations or missing legacy metadata.
- `inference-viewer.test.tsx`: in collect mode, Take over sends teleop then start_recording in order, Save sends save then hold, Play is disabled while recording, and a leaderless session disables the controls. Without collect mode the page is unchanged: the switch is present and nothing records.
- A `start-inference-dialog` test: the checkbox adds `collect=1`.
- A provider test, if one exists: model plus dataset does not auto-teleop; only correction collection sets `origin: 'human_correction'` on `start_recording`.

### Done when

```bash
# from application/ui/
npm run type-check
npm run test -- inference start-inference runtime-session
```

Then run `prek` from the repo root with `SKIP=ui-type-check` (see the note in the global AGENTS.md about that hook re-syncing the backend venv).

## PR 3: docs and hardware check

- `docs/explanation/runtime-session-architecture.md`: in **Modes**, describe the handover ramp and per-tick origin, and replace "Nothing implements it yet". In **One tick**, recording now reads the tick's origin. Add the handover to **Traps**: never feed the measured position into the ramp start.
- `docs/08-deploying-model-policies.md`: a short "Collect corrections" section covering the loop, the hotkeys, the "Human correction" badge, why only human actions are training targets, and retraining through the models page's **Retrain**, which is local only.
- `docs/explanation/dagger-human-in-the-loop.md`: update the current-state section.

On a real SO101 with a leader:

1. With the leader far from the follower, Take over: the follower moves smoothly over about 1 s with no jump.
2. Run policy, take over, correct, save, then resume the policy without resetting the scene. Repeat twice in one rollout.
3. Save during the handover: you get an error and the episode stays open. Discard works.
4. Unplug the leader mid-correction: the follower holds and the saved frames stop at the failure.
5. Close the tab mid-correction: the open episode is discarded and saved corrections appear on the dataset page.
6. The dataset page shows the new episodes with "Human correction" badges in the list and episode viewer after a refresh. Replay one: it starts from a failure state, not the rest pose. Export and reimport through Studio, then delete an earlier episode: surviving correction badges still mark the right episodes.
7. Retrain from the model on that dataset, run it, and compare against the old model on the same starting setups. Count successes and takeovers by hand.

Before step 7, confirm the dataset page shows the new episodes. The session copies the recording cache back when the last client leaves. If a training snapshot starts before that copy finishes, it can miss episodes. The recording page has the same timing. If step 7 reproduces the problem, fix it in its own PR by making the snapshot wait for an active recording mutation.

## Out of scope

- **Original DAgger:** labeling states the policy visits without taking control. That needs a separate expert-query UI, and it's unclear whether a person can give reliable labels without driving.
- **Recording autonomous frames** with an `intervention` flag, like LeRobot's `record_autonomous=true`. Add it only if you need the policy's frames for analysis. It needs a dataset feature migration, and training would have to filter those frames out.
- **Moving the leader to the follower** on actuated leaders (LeRobot's smooth handover). The follower-side ramp works with any leader. Add the leader-side move if operators find it uncomfortable to match the follower by hand.
- Automatic takeover detection, intervention-rate dashboards, weighted losses, and remote retraining from a base model.
