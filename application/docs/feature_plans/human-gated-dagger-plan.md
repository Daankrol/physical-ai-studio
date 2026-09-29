# Human-gated DAgger: implementation plan

## Goal

An operator runs a trained policy on the robot, takes over with the leader arm when the policy goes wrong, demonstrates the recovery, and hands control back. Each correction is saved as an ordinary episode in the model's dataset, and the operator retrains from the existing model. This is human-gated DAgger (HG-DAgger), the variant LeRobot ships as `--strategy.type=dagger`. Background and the difference from original DAgger are in the [research note](../explanation/dagger-human-in-the-loop.md).

One `RuntimeSession` holds the policy and the leader at the same time, recording works in any mode, and the models page can already retrain from a base model. Safe leader alignment depends on changes to the [physicalai revision pinned by Studio](../../backend/uv.lock). Four pieces are missing:

1. **Safe takeover.** Switching to `teleop` sends the leader's pose to the follower on the next tick. If the leader is far away, the follower jumps. The inference page's Teleoperate switch already has this hazard. For an actuated SO101 leader, move the leader to the *held follower* before enabling teleop, not the follower to a distant leader pose.
2. **Clean labels.** `RecordingCallback` records every non-`hold` tick, including policy ticks. It also reads `follower_source` after the action was sent, and `_finish_arm` flips that from another thread, so a tick can get the wrong label.
3. **A screen.** Inference and recording are separate pages. None of them loads a model and a dataset together.
4. **Episode identity.** The dataset list shows episode number and duration, but it cannot tell a demonstration from a saved human correction after a refresh or export.

## Why train on corrections, not the entire rollout

The training set already contains expert demonstrations of normal task execution. The new information in a human-gated rollout is the recovery: an observation in a state the policy reached, paired with what the human actually did next. Recording and training on the policy's own autonomous actions would treat its mistakes as expert targets, teaching them back to the model. LeRobot's corrections-only mode makes each human-controlled window a separate episode; its continuous mode tags both kinds of frames, but plain imitation training does not automatically filter out the autonomous ones.

An action-prediction model trains on observation/action samples (sometimes with short history and future action chunks), not a requirement that every training episode start at the task's initial pose. Short correction episodes can start mid-task; at their boundaries, history and action chunks must be padded or masked, not joined to unrelated frames. Mix the new correction episodes with the original full demonstrations, rather than training only on short recoveries. Evaluate whole-task rollouts separately to see whether the combined dataset helped. Keep the task text unchanged: it is a model input, not a place to store the correction label.

## Design

```text
policy ──Take over──▶ follower HOLD ──▶ align leader ──▶ leader torque OFF ──▶ teleop (recorded)
                             (not recorded)       (not recorded)                │
                     Resume policy ◀── hold ◀── Save/Discard correction ◀────────┘
```

- **Handover.** Every request to enter `teleop`, including the existing inference switch and recording page, first latches the follower in `hold`. On a supported actuated leader, turn its torque on, move it at a bounded, tunable speed to the held follower position, verify fresh measured positions are within a tunable tolerance, then turn leader torque off and enter `teleop`. Keep the follower still throughout alignment. For a passive leader, require explicit manual alignment and confirmation within tolerance; never silently ramp the follower to a distant leader. Apply a small, configurable follower step limit on the first teleop ticks as a safety net against residual misalignment, not as the means of moving it across the workspace. The takeover request is not complete until the mode actually becomes `teleop`.
- **Per-tick provenance.** `StudioActionSource.update()` records which source produced the action it returned: `hold`, `handover`, `teleop`, or `policy`. The runtime calls `update()`, sends the action, then calls `on_tick` on the same thread (`physicalai.runtime.core.RobotRuntime.run`), so `RecordingCallback` can read that value without a race.
- **Record only human actions.** A frame is written only when that tick's source is `teleop` and the leader read on that tick succeeded. Hold, alignment, residual handover, stale-leader fallback, and policy ticks are never written. This changes behavior on purpose: Studio no longer records policy actions as demonstrations. No UI does that today.
- **Mark correction frames in the dataset.** Every recorded frame gets an `intervention` bool column, the same name, dtype and shape LeRobot's own DAgger strategy writes (`{"dtype": "bool", "shape": (1,), "names": None}`). Correction frames are `True`, normal demonstrations `False`. An episode with any `True` frame shows a "Human correction" badge in the dataset list and episode viewer. Because the label lives in the LeRobot data itself, it survives export, import, snapshots and episode deletion with no Studio-specific bookkeeping, and LeRobot HIL datasets imported into Studio get the badge for free. Policies never see it: Studio's converter puts unknown keys into `Observation.extra`, and LeRobot builds policy inputs only from `observation.*` and `action.*`. Ronald's WIP branch (`RHeckerIntel/rhecker/human-in-loop`) used a per-frame `source` int column the same way, but never migrated existing datasets (see below).
- **No gaps inside an episode.** Arming the policy is rejected while an episode is open. A correction is always a single continuous stretch of human control.
- **No new command kinds.** The UI composes `set_follower_source`, `start_recording`, `save_episode`, `discard_episode`, and `start_task`. For passive leaders, a confirmation flag on `set_follower_source` explicitly authorizes a pose-checked manual handover; it is not a bypass when the poses disagree. Extend `start_recording` with an optional `intervention: bool = False`; `save_episode` and `discard_episode` remain acknowledged.

## PR 0: physicalai prerequisite (separate repository)

Studio currently pins `physicalai` at `0ad4548` in [backend/uv.lock](../../backend/uv.lock). At that revision, `SO101.set_torque(enabled=...)` exists on the driver, but `SharedRobot` exposes no torque call, and `SO101.send_action` rejects all leader-role commands. Ronald's WIP `leader.send_action(robot_state)` also passes an observation rather than the required joint-position array. Torque passthrough alone will not make that call work.

- In `openvinotoolkit/physicalai`, add checked `SharedRobot.set_torque(enabled=...)` transport support to the robot owner (request/reply, not a fire-and-forget action); report unsupported drivers and failures to the caller. The current SO101 `_set_torque` only logs servo write errors, so propagate those errors (and verify torque state where supported) before acknowledging success. Do not make torque mandatory for unrelated robot plugins.
- Permit a connected SO101 leader to accept bounded position commands **only during an explicit torque-enabled alignment**, or provide an equivalent leader-feedback method. Preserve read-only, torque-off behavior during ordinary teleoperation. Keep the leader and follower calibration/unit conventions consistent, and make it impossible to accidentally disable follower torque through this handover path.
- Add owner/transport and SO101 tests for the enable, move, disable sequence; reject motion before enable and after disable. On failed movement, timeout, client disconnect, or owner exit, attempt torque-off in `finally` and surface failures. The caller must be able to verify fresh leader observations after a commanded move; a successful `send_action` publication alone is not evidence of arrival.
- Pin the reviewed upstream revision in both [backend](../../backend/uv.lock) and [library](../../../library/uv.lock) lockfiles before Studio PR 1. Do not treat locally patched site-packages as the dependency.

**Gate:** A `SharedRobot`-backed SO101 leader moves to a bounded target, reaches it within tolerance, and returns to torque-off state on success and on recoverable cancellation/failure. A hardware disconnect cannot guarantee torque-off: surface that fault and require operator intervention rather than entering teleop. Until the upstream capability lands, do not enable automatic SO101 takeover in Studio.

## PR 1: Studio backend (safe handover and clean labels)

Depends on PR 0. This PR fixes the takeover jump and the label race for existing pages, not just the correction UI.

### `backend/src/settings.py`

- Add configurable leader alignment speed, position tolerance, timeout and maximum follower step during takeover with conservative SO101 defaults. Set limits in calibrated joint units and validate them against the rig. A physical arm needs calibration knobs, not an unchangeable duration.

### `backend/src/runtime/action_source.py`

- Keep `self._last_action_source: ActionOrigin`, where `ActionOrigin = Literal["hold", "handover", "teleop", "policy"]`; expose it as a read-only `last_action_source` property. `_read_leader` returns `(action, fresh: bool)`, with `fresh=False` on fallback paths.
- Route **all** transitions into `teleop` through one guarded handover in this module. On request, latch `hold` from the last commanded action, keep emitting hold actions every tick, and run leader alignment off the control thread. Never call a blocking torque or motion request from `update()`. Publish an `aligning` state; do not publish `follower_source="teleop"` until torque is off and fresh leader **and follower** observations agree within tolerance. Limit the first follower teleop actions to a configured maximum step from the last commanded target; mark any clamped ticks as `handover`, not expert actions.
- Cancel stale handovers by generation when another mode is requested, the leader fails, the subscriber leaves, or the session tears down. Torque-off belongs in `finally` even on cancellation; if it cannot be confirmed, report a fault and leave the follower in hold. Do not resume policy motion until the leader alignment worker has stopped and torque-off is confirmed. A later callback may switch to teleop only if its generation is still current and alignment succeeded.
- Passive leaders need an explicit manual-alignment path: while holding the follower, wait for fresh matching poses, then accept a confirmed `set_follower_source` request (add a confirmation field to that existing command). Reject unsupported leaders that cannot report a fresh comparable pose. Never interpret confirmation as permission to move a misaligned follower; use state events for UI sequencing and failures.
- Set per-tick origin to `teleop` only for a fresh human-controlled leader action after alignment; stale leader actions and hold are `hold`. `_arm_policy` rejects policy resume while a recording is open with `recording_in_progress`.

### `backend/src/runtime/session.py`

- Pass the configured alignment limits when building `StudioActionSource`; session teardown cancels in-flight alignment and ensures leader torque is off before disconnecting the leader. Keep the follower holding throughout.
- Build `RecordingCallback` with `action_origin=lambda: action_source.last_action_source` instead of `follower_source`.

### `backend/src/runtime/callbacks/recording.py`

- `RecordingCallback.__init__` takes `action_origin: Callable[[], ActionOrigin]`. `on_tick` returns early unless `action_origin() == "teleop"`.
- `RecordingState` counts frames per episode: reset in `start()`, increment in `add_frame()` only when the frame was written. `stop_episode()` raises `RuntimeError("No frames were recorded. Discard the episode.")` **before** clearing `is_recording` when the count is zero, so the episode stays open and can still be discarded. Without this, an operator who presses Save before teleop is ready could get an empty save.
- `RecordingState.start(task, intervention)` stores the flag for the open episode; `add_frame` passes it through to the mutation. `start_recording` with `intervention=True` also requires a loaded model, dataset and leader, with alignment complete and `follower_source="teleop"` confirmed; reject early or failed takeover requests on the backend.

### `backend/src/runtime/contract.py`, `backend/src/runtime/dataset_features.py`, `backend/src/internal_datasets/`, `backend/src/schemas/dataset.py`

- Add `intervention: bool = False` to `StartRecordingCommand`.
- `build_lerobot_dataset_features` adds the `intervention` feature, so every newly created dataset has it.
- **Migrate existing datasets on load.** LeRobot validates frame keys strictly (`validate_frame` rejects extra features), so a dataset created before this change cannot accept `intervention` frames. This is what breaks Ronald's branch: `start_recording_mutation` only uses the features dict when *creating* a dataset, existing ones are copied and resumed with their old schema, every frame then fails validation, and `RecordingCallback.on_tick` swallows the exception, so frames are silently dropped. Fix it in `InternalLeRobotDataset.start_recording_mutation`: when the source dataset lacks `intervention`, build the cache with LeRobot's `lerobot.datasets.dataset_tools.add_features(dataset, {"intervention": (np.zeros((total_frames, 1), dtype=bool), feature_info)}, output_dir=cache_dir)` instead of `shutil.copytree`, then resume the cache for writing. The recording flow already copies the whole dataset into the cache every session, so this adds no extra copy: it rewrites the parquet data and copies the videos without re-encoding. The migrated cache replaces the original on finalize like any other recording, so each dataset is migrated once, the first time someone records into it.
- `_process_frame` adds `"intervention": np.array([flag], dtype=bool)` to each frame, and `DatasetClient.add_frame`/`RecordingMutation.add_frame` take the flag.
- Add `intervention: bool` to both `EpisodeInfo` and `Episode`: `True` when any frame in the episode is `True`, `False` when the column is missing (older or imported datasets that nobody has recorded into). Read it from the `intervention` column (`hf_dataset.select_columns(["episode_index", "intervention"])`, grouped once per request), not from per-episode stats, because migrated episodes may not have stats for the new feature.
- Episode deletion needs no change: LeRobot's `delete_episodes` copies every column, so the flags stay with their frames when episodes are renumbered.

### Tests (`backend/tests/runtime/`)

- `test_action_source.py`: replace the immediate-teleop test with an alignment test: policy → hold, leader moves toward the held follower, follower commands stay latched, torque goes off, then teleop starts from fresh matching observations. The same gate applies to recording-only entry. Manual confirmation with mismatched or stale passive-leader observations is rejected; a small residual mismatch is step-limited and tagged `handover`. Failed leader, timeout, stale read, cancellation and subscriber loss stay in hold and never start recording. A failed leader read reports origin `hold` even after successful entry.
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

Update the existing immediate-teleop tests to assert the confirmed handover instead. Keep recording-only sessions, policy inference and failure-to-hold behavior working.

## PR 2: UI (collect corrections)

### Entry: `ui/src/features/models/models-table/start-inference-dialog.tsx`

- Add a `Checkbox` labeled "Collect corrections into the training dataset". When checked, add `collect=1` to the search params.

### `ui/src/features/models/inference/use-inference-params.ts`

- Return `collect: searchParams.get('collect') === '1'`.

### `ui/src/features/models/inference/inference-page.tsx`

- The model's dataset is already fetched. Pass `dataset={collect ? dataset : undefined}` to `RuntimeSessionProvider`, and pass `collect` and `defaultTask` to `InferenceViewer`.

### `ui/src/features/robots/runtime-session-provider.tsx`

- `onOpen`: request teleop only when a dataset is given **without** a model; the recording page must wait for confirmed alignment before offering Start episode, and must provide manual alignment confirmation for passive leaders. The collection page starts in `hold`. The same guarded transition applies to the ordinary inference Teleoperate switch.
- `startEpisode` keeps its current call shape but sends `intervention: true` in `start_recording` only when the provider is in correction-collection mode; the recording page keeps sending the default `false`.

### `ui/src/features/models/inference/inference-viewer.tsx`

With `collect` set:

- Ready means `readyForInference && readyForRecording`. If `!state.has_leader`, show "Collecting corrections needs a leader arm" and disable the controls.
- Replace the Teleoperate switch with:
  - **Take over** (when not recording): request teleop and show "Aligning leader" while the follower holds. Wait for the confirmed `follower_source='teleop'` state before calling `startEpisode.mutate(task)`; on failure, show the error and leave recording off. For a passive leader, show manual alignment instructions and send the explicit confirmation only after the poses match; the backend checks them again instead of trusting the UI.
  - **Save correction** and **Discard** (while recording): request `hold` first and wait for the confirmed state so the follower stops before video encoding. Then `await saveEpisode.mutateAsync()` or `await discardEpisode.mutateAsync()`. On a failed save, remain in hold, show the error and keep Discard available.
  - Existing **Play** resumes the policy through `startTask`. Disable it while recording, aligning, or cleaning up leader torque; the backend also rejects unsafe resume requests.
  - A status line showing `episodes_recorded` corrections this session.
- Hotkeys: reuse the `recording.start_episode`, `recording.accept_episode`, and `recording.discard_episode` bindings for Take over, Save, and Discard. The operator needs to take over without reaching for the mouse.
- Without `collect`, the page stays as it is.

### Dataset episode display: `ui/src/features/datasets/episodes/episode-tag.tsx` and `ui/src/routes/datasets/`

- After API types are regenerated, show a compact "Human correction" badge next to the episode number/duration in the existing shared `EpisodeTag` when `episode.intervention` is true, so it appears in the episode list and detail header. Plain demonstrations keep their current display. Correction episodes are all-human, so there is nothing to shade inside one; if mixed episodes are added later, Ronald's `EpisodeChart` segment shading on his branch is ready to reuse with the per-frame column.

### Tests

- Episode list and viewer tests: the badge survives a refresh from `EpisodeInfo` and `Episode`, and never appears for demonstrations or missing legacy metadata.
- `inference-viewer.test.tsx`: in collect mode, Take over waits for confirmed alignment before start_recording, failed alignment never starts recording, Save waits for hold before sending save, Play is disabled while recording or aligning, and a leaderless session disables the controls. Without collect mode the existing switch remains, but it also waits for the guarded handover.
- A `start-inference-dialog` test: the checkbox adds `collect=1`.
- A provider test, if one exists: model plus dataset does not auto-teleop; recording-only sessions wait for guarded handover; only correction collection sets `intervention: true` on `start_recording`.

### Done when

```bash
# from application/ui/
npm run type-check
npm run test -- inference start-inference runtime-session
```

Then run `prek` from the repo root with `SKIP=ui-type-check` (see the note in the global AGENTS.md about that hook re-syncing the backend venv).

## PR 3: docs and hardware check

- `docs/explanation/runtime-session-architecture.md`: in **Modes**, describe off-thread leader alignment and per-tick origin, and replace "Nothing implements it yet". In **One tick**, recording now reads the tick's origin. In **Traps**, explain why follower movement toward an unaligned leader is not an acceptable default.
- `docs/08-deploying-model-policies.md`: a short "Collect corrections" section covering the loop, the hotkeys, the "Human correction" badge, why only human actions are training targets, and retraining through the models page's **Retrain**, which is local only.
- `docs/explanation/dagger-human-in-the-loop.md`: update the current-state section.

On a real SO101 with a leader:

1. With the SO101 leader far from the follower, Take over: the follower holds position, the leader moves to it within configured limits, leader torque switches off, and only then teleop starts. Confirm the operator can move the leader freely.
2. Run policy, take over, correct, save, then resume the policy without resetting the scene. Repeat twice in one rollout.
3. Interrupt alignment, unplug the leader during alignment, or simulate torque-off failure: the follower stays in hold, no correction starts, and the fault is visible. Save before alignment completes must not create an empty episode.
4. Unplug the leader mid-correction: the follower holds and the saved frames stop at the failure.
5. Close the tab mid-correction: the open episode is discarded and saved corrections appear on the dataset page.
6. The dataset page shows the new episodes with "Human correction" badges in the list and episode viewer after a refresh. Replay one: it starts from a failure state, not the rest pose. Export and reimport through Studio, then delete an earlier episode: surviving correction badges still mark the right episodes.
7. Retrain from the model on that dataset, run it, and compare against the old model on the same starting setups. Count successes and takeovers by hand.

Before step 7, confirm the dataset page shows the new episodes. The session copies the recording cache back when the last client leaves. If a training snapshot starts before that copy finishes, it can miss episodes. The recording page has the same timing. If step 7 reproduces the problem, fix it in its own PR by making the snapshot wait for an active recording mutation.

## Out of scope

- **Original DAgger:** labeling states the policy visits without taking control. That needs a separate expert-query UI, and it's unclear whether a person can give reliable labels without driving.
- **Recording autonomous frames**, like LeRobot's `record_autonomous=true` and Ronald's branch. The `intervention` column already supports it (policy frames would be `False` inside a mixed episode), but training would then have to drop non-intervention frames from correction episodes, and neither Studio's nor LeRobot's trainer does that today. Add it only if you need the policy's frames for analysis.
- Automatic takeover detection, intervention-rate dashboards, weighted losses, and remote retraining from a base model.
