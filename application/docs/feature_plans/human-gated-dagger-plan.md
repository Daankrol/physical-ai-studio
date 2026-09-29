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
- **Mark correction frames in the dataset.** Every recorded frame gets an `intervention` bool column, the same name, dtype and shape LeRobot's own DAgger strategy writes (`{"dtype": "bool", "shape": (1,), "names": None}`). Correction frames are `True`, normal demonstrations `False`. An episode with any `True` frame shows a "Human correction" badge in the dataset list and episode viewer. Because the label lives in the LeRobot data itself, it survives export, import, snapshots and episode deletion with no Studio-specific bookkeeping, and LeRobot HIL datasets imported into Studio get the badge for free. Policies never see it: Studio's converter puts unknown keys into `Observation.extra`, and LeRobot builds policy inputs only from `observation.*` and `action.*`. Ronald's WIP branch (`RHeckerIntel/rhecker/human-in-loop`) used a per-frame `source` int column the same way, but never migrated existing datasets (see below).
- **No gaps inside an episode.** Arming the policy is rejected while an episode is open. A correction is always a single continuous stretch of human control.
- **No new command kinds.** The UI composes `set_follower_source`, `start_recording`, `save_episode`, `discard_episode`, and `start_task`. Extend `start_recording` with an optional `intervention: bool = False`; `save_episode` and `discard_episode` remain acknowledged.

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
- `RecordingState.start(task, intervention)` stores the flag for the open episode; `add_frame` passes it through to the mutation. `start_recording` with `intervention=True` also requires a loaded model and a leader, not just a loaded dataset.

### `backend/src/runtime/contract.py`, `backend/src/runtime/dataset_features.py`, `backend/src/internal_datasets/`, `backend/src/schemas/dataset.py`

- Add `intervention: bool = False` to `StartRecordingCommand`.
- `build_lerobot_dataset_features` adds the `intervention` feature, so every newly created dataset has it.
- **Migrate existing datasets on load.** LeRobot validates frame keys strictly (`validate_frame` rejects extra features), so a dataset created before this change cannot accept `intervention` frames. This is what breaks Ronald's branch: `start_recording_mutation` only uses the features dict when *creating* a dataset, existing ones are copied and resumed with their old schema, every frame then fails validation, and `RecordingCallback.on_tick` swallows the exception, so frames are silently dropped. Fix it in `InternalLeRobotDataset.start_recording_mutation`: when the source dataset lacks `intervention`, build the cache with LeRobot's `lerobot.datasets.dataset_tools.add_features(dataset, {"intervention": (np.zeros((total_frames, 1), dtype=bool), feature_info)}, output_dir=cache_dir)` instead of `shutil.copytree`, then resume the cache for writing. The recording flow already copies the whole dataset into the cache every session, so this adds no extra copy: it rewrites the parquet data and copies the videos without re-encoding. The migrated cache replaces the original on finalize like any other recording, so each dataset is migrated once, the first time someone records into it.
- `_process_frame` adds `"intervention": np.array([flag], dtype=bool)` to each frame, and `DatasetClient.add_frame`/`RecordingMutation.add_frame` take the flag.
- Add `intervention: bool` to both `EpisodeInfo` and `Episode`: `True` when any frame in the episode is `True`, `False` when the column is missing (older or imported datasets that nobody has recorded into). Read it from the `intervention` column (`hf_dataset.select_columns(["episode_index", "intervention"])`, grouped once per request), not from per-episode stats, because migrated episodes may not have stats for the new feature.
- Episode deletion needs no change: LeRobot's `delete_episodes` copies every column, so the flags stay with their frames when episodes are renumbered.

### Tests (`backend/tests/runtime/`)

- `test_action_source.py`
  - Update `test_teleop_forwards_leader_positions_on_next_tick` to construct with `handover_s=0`.
  - New: the handover ramps from the hold target to the leader over N ticks, the origin is `handover` then `teleop`, and the first tick never jumps to the leader pose.
  - New: policy → teleop ramps from the last *policy* action, not from the measured position.
  - New: a failed leader read reports origin `hold`, even in `teleop`.
- `test_action_source_policy.py`: `start_task` is rejected with `recording_in_progress` while a recording is open and allowed after save or discard.
- `test_recording_callback.py`: switch the helper to `action_origin`. Frames are written only for `teleop`. `handover`, `policy`, and `hold` are skipped. Saving an empty episode raises and leaves the episode open, and discard still works.
- Dataset/episode tests (`backend/tests/internal_datasets/`):
  - Recording with `intervention=True` into a **pre-existing dataset without the column** migrates it, saves the frames, and after finalize the dataset has the column, all old frames `False`, all new frames `True`. This is the regression test for the silent frame drop in Ronald's branch.
  - A new dataset gets the column at creation; a normal recording writes `False`.
  - `EpisodeInfo` and `Episode` report `intervention` correctly, and `False` for a dataset without the column.
  - After deleting an earlier episode, the surviving correction episodes still report `intervention=True`.
  - The migrated dataset still trains: build a `LeRobotDataModule` on it and run one `fast_dev_run` batch with ACT, confirming the aggregated stats load and the column ends up in `Observation.extra` only.

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
- `startEpisode` keeps its current call shape but sends `intervention: true` in `start_recording` only when the provider is in correction-collection mode; the recording page keeps sending the default `false`.

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

- After API types are regenerated, show a compact "Human correction" badge next to the episode number/duration in the existing shared `EpisodeTag` when `episode.intervention` is true, so it appears in the episode list and detail header. Plain demonstrations keep their current display. Correction episodes are all-human, so there is nothing to shade inside one; if mixed episodes are added later, Ronald's `EpisodeChart` segment shading on his branch is ready to reuse with the per-frame column.

### Tests

- Episode list and viewer tests: the badge survives a refresh from `EpisodeInfo` and `Episode`, and never appears for demonstrations or missing legacy metadata.
- `inference-viewer.test.tsx`: in collect mode, Take over sends teleop then start_recording in order, Save sends save then hold, Play is disabled while recording, and a leaderless session disables the controls. Without collect mode the page is unchanged: the switch is present and nothing records.
- A `start-inference-dialog` test: the checkbox adds `collect=1`.
- A provider test, if one exists: model plus dataset does not auto-teleop; only correction collection sets `intervention: true` on `start_recording`.

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
- **Recording autonomous frames**, like LeRobot's `record_autonomous=true` and Ronald's branch. The `intervention` column already supports it (policy frames would be `False` inside a mixed episode), but training would then have to drop non-intervention frames from correction episodes, and neither Studio's nor LeRobot's trainer does that today. Add it only if you need the policy's frames for analysis.
- **Moving the leader to the follower** on actuated leaders (LeRobot's smooth handover). The follower-side ramp works with any leader. Add the leader-side move if operators find it uncomfortable to match the follower by hand.
- Automatic takeover detection, intervention-rate dashboards, weighted losses, and remote retraining from a base model.
