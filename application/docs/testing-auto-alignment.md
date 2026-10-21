# Test Studio DAgger handover

Studio holds the follower while a supported leader moves toward it, then waits
for the operator to release alignment. It paces intermediate position targets
in software. Alignment must not write servo speed registers.

The pinned dependencies do not support automatic SO101 alignment. Testing that
path requires both the Runtime torque-control update (physicalai PR #307) and
an SO101 plugin build that permits position-only leader commands during
explicit torque-enabled alignment. Studio uses `physicalai_so101_plugin.SO101`,
not the old built-in Runtime driver. Updating only that old driver will not
change the robot Studio runs.

Use each arm's own calibration and serial port. Stop other controllers before
testing. A running robot owner keeps its old Python code after a Git pull:
disconnect existing Studio sessions and let their owners exit before restarting.

## 1. Check the dependencies

For an NVIDIA machine, install the pinned backend environment:

```bash
cd application/backend
uv sync --frozen --extra cuda --extra tests
uv run --no-sync python -c 'import inspect; from physicalai.robot import SharedRobot; from physicalai_so101_plugin import SO101; print(inspect.getfile(SharedRobot)); print(inspect.getfile(SO101)); print("set_torque:", hasattr(SharedRobot, "set_torque"))'
```

On an Intel XPU target machine, use its existing `xpu` extra instead of `cuda`.
Use `uv run --no-sync` for subsequent backend commands so the selected hardware
extras are not changed.

With the pinned dependencies, `set_torque` is `False` and Studio offers operator
release without powered movement. That path can test correction collection,
but it cannot pass the automatic-alignment checks below.

Before testing powered alignment, install reviewed Runtime and SO101 plugin
builds with the capabilities described above. Record their revisions and keep
them unchanged during the test. Confirm the printed source paths identify those
builds and `set_torque` is `True`. Do not patch site-packages or commit
machine-local dependency overrides. Update Studio's lockfiles only after both
dependency changes are reviewed.

## Reset speed limits left by earlier alignment tests

Stop Studio and all robot owner processes before opening the serial port.
Support the leader and use its port, not the follower's. For standard servo
IDs 1-6, run from `application/backend/`:

```bash
uv run --no-sync python scripts/reset_so101_speeds.py --port /dev/serial/by-id/<leader>
uv run --no-sync python scripts/reset_so101_speeds.py --port /dev/serial/by-id/<leader> --reset
```

The first command only reads. The second sets `GOAL_SPEED` (address 46) to
`0`, meaning no speed limit, and verifies each servo's readback. The script
refuses to write if any servo has torque enabled or cannot be read. It does
not enable torque, send position targets, or change calibration. Stop on any
error; a failed write can leave only part of the arm reset.

Install the position-only SO101 plugin fix before trying alignment again, or
the old driver will overwrite the reset values. Restart owners after installation.
The reset script's checks use mocked hardware; verify its output on the rig.

## 2. Start Studio

From `application/backend/`:

```bash
uv run --no-sync physicalai-studio serve
```

In a second terminal, from `application/ui/`:

```bash
npm run start
```

Run `npm ci` first only if UI dependencies need installing. Open
`http://localhost:3000` on the robot machine. Configure an SO101 follower and an
SO101 leader with valid calibrations and comparable normalized coordinates.
Matching joint names alone does not prove the coordinates are comparable.
Back up the dataset before collecting corrections.

## 3. Check takeover before running the policy

Clear the full movement path, support the leader safely for torque-off, and
keep the stop control reachable. Start from a modest, safe pose difference for
this first check; the software does not require manual pose matching.

1. Open **Run model**, enable **Collect corrections**, and choose the environment.
   Wait for the model, dataset, and environment to initialize. Do not press Play.
2. Press **Take over**. The follower should hold its last commanded pose while
   Studio shows **Aligning leader; follower is holding.**
3. Studio sends small position steps over approximately three seconds. The
   leader should move toward the held follower without changing servo speed
   settings; the follower must not follow the leader's starting pose.
4. A yellow banner reports whether the leader reached tolerance. Leader torque
   stays on and the follower stays held while Studio waits for you, without a
   release timer.
5. Take hold of the leader, then press **Release alignment**. Studio turns leader
   torque off and checks fresh feedback before enabling teleoperation and
   correction recording. This works even when automatic alignment missed
   tolerance; there is no extra pose-match check or confirmation. The follower
   starts following the leader with limited initial steps, so support the leader
   and check the remaining pose difference before releasing.
6. Move the free leader slightly and check that the follower follows it. Keep
   moving through several control ticks: catch-up should finish even while the
   leader moves, rather than leaving the follower permanently slow.
7. **Discard** this initial correction. The follower should return to hold.

The backend settings are:

| Environment variable                  | Default | Meaning                                                                         |
| ------------------------------------- | ------- | ------------------------------------------------------------------------------- |
| `LEADER_ALIGNMENT_GOAL_TIME_S`        | `3`     | Requested software-paced move duration                                          |
| `LEADER_ALIGNMENT_TIMEOUT_S`          | `15`    | Feedback deadline for reporting a missed tolerance; does not release torque     |
| `LEADER_ALIGNMENT_POSITION_TOLERANCE` | `0.5`   | Maximum permitted joint-position error in calibrated units                      |
| `LEADER_HANDOVER_MAX_FOLLOWER_STEP`   | `0.5`   | Extra catch-up movement per joint per control tick, on top of new leader motion |

The catch-up guard softens the initial pose mismatch; it is not an absolute
teleop speed limit. With a stationary leader 2 units ahead, the default follower
commands advance by 0.5 units per tick until they match. If the leader moves
another 2 units between guarded ticks, the follower can move up to 2.5 units:
2 units of leader motion plus 0.5 units of catch-up. The first step has no
leader-motion allowance. Once caught up, the follower tracks directly. Guarded
catch-up ticks remain excluded from correction data.

Keep the requested duration shorter than the feedback timeout. Replace any old
`LEADER_ALIGNMENT_SPEED` override with `LEADER_ALIGNMENT_GOAL_TIME_S`, then
restart the backend. Bus latency, existing speed limits, acceleration, and load
can extend arrival; elapsed time alone never authorizes takeover.

## 4. Test the DAgger collection loop

After the takeover-only check passes:

1. Press **Play** to run the policy with a task prompt.
2. Press **Take over** at a safe test point. The follower must hold while the
   leader aligns. When the banner reports the result, support the leader and
   press **Release alignment**. Recording opens after acknowledged torque-off
   and valid fresh feedback, even if alignment missed tolerance. Step-limited
   catch-up actions are not recorded as human demonstration frames.
3. Demonstrate a short recovery with the leader.
4. **Save correction**. Studio holds the follower before saving the episode.
5. Press **Play** to resume policy control and repeat once.
6. Check that saved episodes show the **Human correction** badge. Policy, hold,
   and alignment frames must not become human demonstration frames.

See the [human correction workflow](./08-deploying-model-policies.md#collect-human-corrections)
for the dataset and recording controls. The environment-preview **Teleoperate**
switch and ordinary **Record dataset** are separate passive-leader flows. They
follow the leader without pose matching, manual confirmation, or leader torque
commands. The initial follower-step limit still applies. Powered alignment is
off by default in the backend; only the model-run controls explicitly request
it, and the backend requires a loaded model. Opening a model run from ordinary
teleop first returns the follower to hold, so its takeover starts from the
model-specific handover path.

## If handover fails

A missed tolerance must leave the follower in hold and show the release prompt.
Leader torque stays on until you support the leader and press **Release alignment**.
The backend logs the joint with the largest leader error. Pressing release
hands control to the operator without requiring matching poses or another
confirmation. Fresh valid feedback and acknowledged torque-off are still
required. The initial follower-step limit bounds catch-up movement, but does
not prevent the follower from moving toward the leader's pose. Cancellation,
disconnect, and hardware faults still attempt torque-off without waiting for
the button.

The reported hardware runs stopped with `elbow_flex` roughly 7.6 to 8.4 normalized
units from its target; that is outside Studio's tolerance and remains unresolved.
Do not increase the tolerance just to bypass it. A connection or torque-off
failure must not start an episode; a missed tolerance still allows explicit
operator release.

If connection logs contain `Register write failed`, capture those lines before
continuing; the intended servo configuration has not been confirmed. Do not tune
PID gains blindly. If **Automatic alignment is unavailable** appears, the leader
is using the operator-release fallback, not automatic alignment. Take hold of
the leader and press **Release alignment** to take over without a pose-match
check. Do not count that as a successful automatic-alignment test.

If torque-off fails, inspect and secure the leader before continuing. Software
acknowledgments cannot guarantee physical torque state after a hardware fault.
These checks do not establish hardware safety certification.

## Restore normal dependencies

If you used local dependency overrides, stop Studio and disconnect the robot
session first. If these dependency files were clean before testing, restore
only those files from Studio's root:

```bash
git restore -- application/backend/pyproject.toml application/backend/uv.lock library/pyproject.toml library/uv.lock
cd application/backend
uv sync --frozen --extra cuda --extra tests
```

Use the hardware extra appropriate for the robot machine. If those files had
local edits before testing, restore your saved versions instead of discarding
them with `git restore`.
