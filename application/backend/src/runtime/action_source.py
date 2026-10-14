from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from physicalai.runtime import WorkerDiedError

from runtime.contract import (
    ActionOrigin,
    ErrorEvent,
    HandoverState,
    LoadModelCommand,
    SetFollowerSourceCommand,
    StartRecordingCommand,
    StartTaskCommand,
    StateData,
    StateEvent,
    StopTaskCommand,
)
from runtime.policy_loader import PolicyLoader, model_identity, requires_task

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime import PolicySource

    from runtime.callbacks.recording import RecordingState
    from runtime.contract import CommandMailbox, EventSink, FollowerSource
    from runtime.policy_loader import ModelIdentity, ObservationSnapshot


class StudioActionSource:
    """Select actions and guard every transition into leader teleoperation."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        follower: Robot,
        leader: Robot | None,
        mailbox: CommandMailbox,
        event_sink: EventSink,
        fps: float,
        camera_keys: Sequence[str] = (),
        models_dir: Path | None = None,
        position_space_comparable: bool = False,
        leader_alignment_speed: float = 1.0,
        alignment_tolerance: float = 0.5,
        alignment_timeout_s: float = 15.0,
        max_follower_step: float = 0.5,
    ) -> None:
        self._follower = follower
        self._leader = leader
        self._mailbox = mailbox
        self._event_sink = event_sink
        self._fps = max(1.0, float(fps))
        self._max_leader_failures = max(1, int(3 * fps))
        self._alignment_speed = leader_alignment_speed
        self._alignment_tolerance = alignment_tolerance
        self._alignment_timeout_s = alignment_timeout_s
        self._max_follower_step = max_follower_step
        self._position_space_comparable = position_space_comparable
        self._automatic_alignment_supported = (
            leader is not None
            and callable(getattr(leader, "set_torque", None))
            and callable(getattr(leader, "send_action", None))
        )
        self._follower_source: FollowerSource = "hold"
        self._handover_state: HandoverState = "idle"
        self._hold_target: np.ndarray | None = None
        self._last_commanded_action: np.ndarray | None = None
        self._last_leader_timestamp: float | None = None
        self._leader_failures = 0
        self._leader_reads_enabled = leader is not None
        self._failure_logged = False
        self._last_action_source: ActionOrigin = "hold"
        self._teleop_guard = False
        self._bus: object | None = None
        self._session_id: str | None = None
        self._policy: PolicySource | None = None
        self._policy_lock = threading.Lock()
        self._attached_generation = 0
        self._arm_generation = 0
        self._arming = False
        self._model_loaded = False
        self._loaded_identity: ModelIdentity | None = None
        self._task: str | None = None
        self._policy_error_emitted = False
        self._last_robot_state: RobotObservation | None = None
        self._last_camera_frames: Mapping[str, Frame] = {}
        self._last_follower_timestamp: float | None = None
        self._follower_revision = 0
        self._follower_observation: tuple[np.ndarray, float, int] | None = None
        self._handover_lock = threading.Condition(threading.RLock())
        self._handover_generation = 0
        self._handover_cancel: threading.Event | None = None
        self._handover_thread: threading.Thread | None = None
        self._torque_cleanup_active = False
        self._shutting_down = False
        self._loader = PolicyLoader(
            event_sink=event_sink,
            on_ready=self._set_policy,
            camera_keys=camera_keys,
            models_dir=models_dir,
        )
        self._recording: RecordingState | None = None

    @property
    def follower_source(self) -> FollowerSource:
        with self._handover_lock:
            return self._follower_source

    @property
    def last_action_source(self) -> ActionOrigin:
        """The origin of the action most recently returned by ``update``."""
        with self._handover_lock:
            return self._last_action_source

    @property
    def torque_cleanup_active(self) -> bool:
        with self._handover_lock:
            return self._torque_cleanup_active

    @property
    def leader_torque_safe(self) -> bool:
        """True once no handover worker can still leave leader torque enabled."""
        with self._handover_lock:
            return not self._torque_cleanup_active and not (
                self._handover_thread is not None and self._handover_thread.is_alive()
            )

    def bind_recording(self, recording: RecordingState) -> None:
        """Attach session-owned recording flags so state events include them."""
        self._recording = recording

    def state_data(self) -> StateData:
        """Return the session state currently published to the browser."""
        recording = self._recording
        with self._handover_lock:
            return StateData(
                connected=True,
                follower_source=self._follower_source,
                has_leader=self._leader_reads_enabled,
                model_loaded=self._model_loaded,
                task=self._task,
                dataset_loaded=None if recording is None else recording.dataset_loaded,
                is_recording=None if recording is None else recording.is_recording,
                episodes_recorded=None if recording is None else recording.episodes_recorded,
                handover_state=self._handover_state,
            )

    def connect(self, *, bus: object, session_id: str) -> None:
        """Connect the leader once and remember runtime context for policy loads."""
        self._bus = bus
        self._session_id = session_id
        if self._leader is not None and not self._leader.is_connected():
            self._leader.connect()
        if self._leader is not None and self._leader.joint_names != self._follower.joint_names:
            raise ValueError("Leader and follower joint names must match for teleoperation")

    def update(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],
        step: int,
    ) -> np.ndarray:
        self._last_robot_state = robot_state
        self._last_camera_frames = camera_frames
        follower_fresh = self._observe_follower(robot_state)
        with self._handover_lock:
            if self._hold_target is None:
                self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
            handover_busy = self._handover_state == "aligning" or self._torque_cleanup_active

        leader_action: np.ndarray | None = None
        leader_fresh = False
        if not handover_busy and self._leader_reads_enabled:
            leader_action, leader_fresh = self._read_leader(robot_state)
        self._drain_commands(robot_state, leader_action, leader_fresh, follower_fresh)
        policy_action = self._policy_action(robot_state, camera_frames, step)

        with self._handover_lock:
            if self._handover_state != "idle" or self._torque_cleanup_active:
                action = (
                    self._hold_target.copy()
                    if self._hold_target is not None
                    else np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
                )
                origin: ActionOrigin = "handover"
            elif self._follower_source == "teleop":
                if not leader_fresh or leader_action is None:
                    latched = (
                        self._last_commanded_action if self._last_commanded_action is not None else self._hold_target
                    )
                    action = (
                        latched.copy()
                        if latched is not None
                        else np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
                    )
                    origin = "hold"
                else:
                    action, clamped = self._limit_initial_teleop_step(leader_action)
                    origin = "handover" if clamped else "teleop"
            elif self._follower_source == "policy" and policy_action is not None:
                action = np.array(policy_action, dtype=np.float32, copy=True)
                origin = "policy"
            else:
                action = (
                    self._hold_target.copy()
                    if self._hold_target is not None
                    else np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
                )
                origin = "hold"
            self._last_commanded_action = np.array(action, dtype=np.float32, copy=True)
            self._last_action_source = origin
            return action

    def disconnect(self) -> None:
        """Cancel handover work without releasing devices; the session owns teardown."""
        self._cancel_handover()
        self._bus = None
        self._session_id = None
        self._last_camera_frames = {}

    def shutdown_handover(self, *, timeout_s: float = 5.0) -> bool:
        """Cancel and drain alignment before the session disconnects the leader."""
        with self._handover_lock:
            self._shutting_down = True
            thread = self._handover_thread
            if self._handover_cancel is not None:
                self._handover_generation += 1
                self._handover_cancel.set()
                self._follower_source = "hold"
                self._latch_hold(self._last_robot_state)
        if thread is not None:
            thread.join(timeout=timeout_s)
        return self.leader_torque_safe

    def shutdown_policy(self) -> None:
        """Stop in-flight loads and the policy execution worker. Session teardown only."""
        self._loader.shutdown()
        with self._policy_lock:
            self._cancel_arming_locked()
            policy = self._policy
            self._policy = None
            self._model_loaded = False
            self._loaded_identity = None
        if policy is not None:
            policy.disconnect()

    def _observe_follower(self, robot_state: RobotObservation) -> bool:
        timestamp = float(robot_state.timestamp)
        positions = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        with self._handover_lock:
            fresh = self._last_follower_timestamp is None or timestamp > self._last_follower_timestamp
            if fresh:
                self._last_follower_timestamp = timestamp
                self._follower_revision += 1
                self._follower_observation = positions, timestamp, self._follower_revision
                self._handover_lock.notify_all()
            return fresh

    def _limit_initial_teleop_step(self, leader_action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._teleop_guard:
            return np.array(leader_action, dtype=np.float32, copy=True), False
        previous = self._last_commanded_action
        if previous is None:
            previous = self._hold_target
        if previous is None or previous.shape != leader_action.shape:
            self._teleop_guard = False
            return np.array(leader_action, dtype=np.float32, copy=True), False
        delta = np.asarray(leader_action, dtype=np.float32) - previous
        if np.max(np.abs(delta), initial=0.0) <= self._max_follower_step:
            self._teleop_guard = False
            return np.array(leader_action, dtype=np.float32, copy=True), False
        self._teleop_guard = True
        return previous + np.clip(delta, -self._max_follower_step, self._max_follower_step), True

    def _drain_commands(
        self,
        robot_state: RobotObservation,
        leader_action: np.ndarray | None,
        leader_fresh: bool,
        follower_fresh: bool,
    ) -> None:
        for command in self._mailbox.drain():
            if isinstance(command, LoadModelCommand):
                self._handle_load_model(command)
            elif isinstance(command, StartTaskCommand):
                self._handle_start_task(command)
            elif isinstance(command, StopTaskCommand):
                self._handle_stop_task(robot_state)
            elif isinstance(command, SetFollowerSourceCommand):
                self._handle_set_follower_source(command, robot_state, leader_action, leader_fresh, follower_fresh)
            elif isinstance(command, StartRecordingCommand):
                self._handle_start_recording(command)

    def _handle_load_model(self, command: LoadModelCommand) -> None:
        if not command.force and self._loaded_identity == model_identity(command):
            logger.info("Model {} is already loaded; skipping the rebuild", command.model_id)
            return
        self._cancel_handover()
        self._loaded_identity = None
        self._cancel_arming()
        self._model_loaded = False
        self._emit_state()
        if self._bus is None or self._session_id is None:
            self._event_sink.emit(
                ErrorEvent(
                    message="Cannot load a model before the runtime session is connected.",
                    error_code="model_load_failed",
                )
            )
            return
        self._loader.request(
            command,
            self._latest_observation,
            bus=self._bus,
            session_id=self._session_id,
        )

    def _handle_start_task(self, command: StartTaskCommand) -> None:
        if not self._arm_policy(task=command.task):
            return
        self._emit_state()

    def _handle_stop_task(self, robot_state: RobotObservation) -> None:
        self._cancel_arming()
        self._enter_hold(robot_state)
        self._emit_state()

    def _handle_start_recording(self, command: StartRecordingCommand) -> None:
        recording = self._recording
        if recording is not None and recording.is_recording:
            self._emit_error(
                "An episode is already open; save or discard it before starting another.", "recording_in_progress"
            )
            return
        if command.intervention:
            with self._handover_lock:
                ready = (
                    self._model_loaded
                    and self._leader is not None
                    and self._leader_reads_enabled
                    and self._follower_source == "teleop"
                    and self._handover_state == "idle"
                    and not self._torque_cleanup_active
                )
            if not ready:
                self._event_sink.emit(
                    ErrorEvent(
                        message=(
                            "Corrections require a loaded model, dataset, connected leader, "
                            "and confirmed teleoperation."
                        ),
                        error_code="correction_not_ready",
                    )
                )
                return
        if recording is None or not recording.start(command.task, intervention=command.intervention):
            self._event_sink.emit(
                ErrorEvent(
                    message="Load a dataset before starting a recording.",
                    error_code="dataset_not_loaded",
                )
            )
            return
        self._emit_state()

    def _handle_set_follower_source(
        self,
        command: SetFollowerSourceCommand,
        robot_state: RobotObservation,
        leader_action: np.ndarray | None,
        leader_fresh: bool,
        follower_fresh: bool,
    ) -> None:
        requested = command.follower_source
        if requested == "policy":
            if not self._arm_policy(task=self._task):
                return
            self._emit_state()
            return
        if requested == "hold":
            self._cancel_arming()
            self._enter_hold(robot_state)
            self._emit_state()
            return
        self._request_teleop(command, robot_state, leader_action, leader_fresh, follower_fresh)

    def _request_teleop(  # noqa: PLR0911, PLR0915
        self,
        command: SetFollowerSourceCommand,
        robot_state: RobotObservation,
        leader_action: np.ndarray | None,
        leader_fresh: bool,
        follower_fresh: bool,
    ) -> None:
        with self._handover_lock:
            if self._follower_source == "teleop" and self._handover_state == "idle":
                return
            if self._torque_cleanup_active or self._handover_state == "aligning":
                self._emit_error(
                    "A leader alignment or torque cleanup is already active; wait for it to settle.",
                    "leader_torque_cleanup_active",
                )
                return
            self._cancel_arming()
            if self._follower_source != "hold" or self._hold_target is None:
                self._latch_hold(robot_state)
            self._follower_source = "hold"
            self._teleop_guard = True
            target = self._hold_target
            if target is None:
                self._emit_error("The follower has no latched target; takeover is unavailable.", "hold_target_missing")
                self._emit_state()
                return
            if self._leader is None:
                self._emit_error(
                    "Teleoperation requires a leader robot; the follower remains in hold.", "leader_required"
                )
                self._emit_state()
                return
            if not self._leader_reads_enabled:
                self._emit_error(
                    "The leader robot is not responding. Reconnect before resuming teleoperation.",
                    "leader_connection_lost",
                )
                self._emit_state()
                return
            if not self._position_space_comparable:
                self._handover_state = "idle"
                self._emit_error(
                    "Leader takeover is disabled because leader/follower joint-position units "
                    "or calibration are not confirmed compatible.",
                    "leader_position_space_unsupported",
                )
                self._emit_state()
                return
            if self._automatic_alignment_supported:
                generation = self._next_handover_generation_locked()
                cancel = threading.Event()
                self._handover_cancel = cancel
                self._handover_state = "aligning"
                self._torque_cleanup_active = True
                thread = threading.Thread(
                    target=self._align_leader_worker,
                    name=f"leader-handover-{generation}",
                    args=(generation, cancel, target.copy()),
                    daemon=True,
                )
                self._handover_thread = thread
                thread.start()
                self._emit_state()
                return
            if command.confirm_manual_alignment:
                if not leader_fresh or not follower_fresh or leader_action is None:
                    self._handover_state = "manual_required"
                    self._emit_error(
                        "Manual takeover needs fresh leader and follower observations; keep holding and confirm again.",
                        "manual_alignment_stale",
                    )
                    self._emit_state()
                    return
                follower = np.array(robot_state.joint_positions, dtype=np.float32, copy=False)
                if not self._poses_match(leader_action, follower):
                    self._handover_state = "manual_required"
                    self._emit_error(
                        "The leader and held follower poses do not match within tolerance. "
                        "Align them manually before confirming.",
                        "manual_alignment_mismatch",
                    )
                    self._emit_state()
                    return
                if not self._poses_match(follower, target):
                    self._handover_state = "manual_required"
                    self._emit_error(
                        "The follower has not settled at its held target. Keep holding and confirm "
                        "again when the poses match.",
                        "manual_alignment_mismatch",
                    )
                    self._emit_state()
                    return
                self._follower_source = "teleop"
                self._handover_state = "idle"
                self._emit_state()
                return
            self._handover_state = "manual_required"
            self._emit_state()
            self._emit_error(
                "This leader cannot be moved safely by Studio. Align it by hand with the held "
                "follower, then confirm the pose match.",
                "manual_alignment_required",
            )

    def _arm_policy(self, *, task: str | None) -> bool:  # noqa: PLR0911
        """Reset the policy, then warm up off this thread. Return whether mode already switched."""
        recording = self._recording
        if recording is not None and recording.is_recording:
            self._emit_error(
                "Stop or discard the open episode before resuming policy control.", "recording_in_progress"
            )
            return False
        with self._handover_lock:
            if self._torque_cleanup_active or self._handover_state != "idle":
                self._cancel_handover_locked()
                self._emit_error(
                    "Policy control is unavailable until leader alignment and torque cleanup have settled.",
                    "leader_torque_cleanup_active",
                )
                self._emit_state()
                return False
        with self._policy_lock:
            policy = self._policy
            if policy is None:
                self._emit_error("No policy is loaded.", "policy_not_loaded")
                return False
            try:
                if task is not None:
                    policy.set_task(task)
                policy.reset()
            except WorkerDiedError:
                raise
            except RuntimeError as exc:
                self._emit_error(str(exc) or "Failed to arm the policy.", "policy_reset_failed")
                return False
            self._task = task
            self._policy_error_emitted = False
            self._arm_generation += 1
            generation = self._arm_generation
            self._arming = True
        self._warn_if_prompt_missing(policy, task)
        snapshot = self._latest_observation()
        if snapshot is None:
            with self._policy_lock:
                if generation != self._arm_generation or self._policy is not policy:
                    return False
                self._arming = False
                self._follower_source = "policy"
            return True
        thread = threading.Thread(
            target=self._finish_arm,
            name=f"policy-arm-{generation}",
            args=(policy, generation, snapshot),
            daemon=True,
        )
        thread.start()
        return False

    @staticmethod
    def _warn_if_prompt_missing(policy: PolicySource, task: str | None) -> None:
        """Log when a language-conditioned policy starts with an empty prompt.

        An empty prompt is allowed (the model then runs on its prompt template
        alone), but it is rarely intended and otherwise invisible.
        """
        if task is not None and task.strip():
            return
        model = getattr(policy, "_model", None)
        if model is not None and requires_task(model):
            logger.warning("Starting a policy that expects a task prompt with an empty prompt")

    def _finish_arm(self, policy: PolicySource, generation: int, snapshot: ObservationSnapshot) -> None:
        """Warm the current observation, then switch into policy mode if still current."""
        if snapshot is None:
            return
        robot_state, camera_frames = snapshot
        try:
            policy.warmup(policy.to_model_input(robot_state, camera_frames))
        except Exception as exc:
            with self._policy_lock:
                if generation != self._arm_generation:
                    return
                self._arming = False
            logger.exception("Policy warmup failed; staying in hold")
            self._emit_error(str(exc) or "Failed to warm up the policy.", "policy_warmup_failed")
            return
        with self._policy_lock:
            if generation != self._arm_generation or self._policy is not policy:
                return
            self._arming = False
            self._follower_source = "policy"
        self._emit_state()

    def _cancel_arming(self) -> None:
        with self._policy_lock:
            self._cancel_arming_locked()

    def _cancel_arming_locked(self) -> None:
        self._arm_generation += 1
        self._arming = False

    def _cancel_handover(self) -> None:
        with self._handover_lock:
            self._cancel_handover_locked()

    def _cancel_handover_locked(self) -> None:
        thread = self._handover_thread
        if thread is not None and thread.is_alive():
            self._handover_generation += 1
            if self._handover_cancel is not None:
                self._handover_cancel.set()
            self._follower_source = "hold"
        else:
            self._handover_generation += 1
            self._handover_state = "idle"
            self._torque_cleanup_active = False
            self._handover_cancel = None
            self._handover_thread = None
        self._handover_lock.notify_all()

    def _next_handover_generation_locked(self) -> int:
        self._handover_generation += 1
        return self._handover_generation

    def _enter_hold(self, robot_state: RobotObservation) -> None:
        with self._handover_lock:
            self._cancel_handover_locked()
            self._latch_hold(robot_state)
            self._follower_source = "hold"
            self._teleop_guard = False

    def _latch_hold(self, robot_state: RobotObservation | None) -> None:
        target = self._last_commanded_action
        if target is None and robot_state is not None:
            target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        if target is not None:
            self._hold_target = np.array(target, dtype=np.float32, copy=True)

    def _align_leader_worker(  # noqa: C901, PLR0912, PLR0915
        self, generation: int, cancel: threading.Event, target: np.ndarray
    ) -> None:
        failure: Exception | None = None
        aligned = False
        with self._handover_lock:
            leader_timestamp = self._last_leader_timestamp
            follower_revision_at_start = self._follower_revision
            follower_timestamp_at_start = self._last_follower_timestamp
        follower_revision = follower_revision_at_start
        follower_timestamp = follower_timestamp_at_start
        follower_at_alignment: tuple[np.ndarray, float, int] | None = None
        try:
            if leader_timestamp is None or follower_timestamp is None:
                raise RuntimeError("Fresh leader and follower observations are required before alignment")
            follower_timestamp = float(follower_timestamp)
            set_torque = getattr(self._leader, "set_torque")
            send_action = getattr(self._leader, "send_action")
            set_torque(enabled=True)
            deadline = time.monotonic() + self._alignment_timeout_s
            step_limit = self._alignment_speed / self._fps
            while time.monotonic() < deadline:
                self._raise_if_cancelled(generation, cancel)
                observation = self._next_fresh_leader(leader_timestamp, deadline, generation, cancel)
                leader_timestamp = float(observation.timestamp)
                leader_positions = np.asarray(observation.joint_positions, dtype=np.float32)
                if leader_positions.shape != target.shape or not np.all(np.isfinite(leader_positions)):
                    raise ValueError("Leader returned an invalid pose during alignment")
                follower = self._latest_follower_after(follower_revision_at_start, follower_timestamp)
                if follower is not None:
                    follower_at_alignment = follower
                    follower_revision = follower[2]
                    follower_timestamp = follower[1]
                leader_delta = target - leader_positions
                leader_at_target = np.max(np.abs(leader_delta), initial=0.0) <= self._alignment_tolerance
                follower_at_target = (
                    follower_at_alignment is not None
                    and np.max(np.abs(target - follower_at_alignment[0]), initial=0.0) <= self._alignment_tolerance
                )
                poses_match = (
                    follower_at_alignment is not None
                    and np.max(np.abs(leader_positions - follower_at_alignment[0]), initial=0.0)
                    <= self._alignment_tolerance
                )
                if leader_at_target and follower_at_target and poses_match:
                    aligned = True
                    break
                if not leader_at_target:
                    bounded_target = leader_positions + np.clip(leader_delta, -step_limit, step_limit)
                    send_action(np.asarray(bounded_target, dtype=np.float32))
                    time.sleep(min(1.0 / self._fps, max(0.0, deadline - time.monotonic())))
            if not aligned:
                raise TimeoutError("Leader did not reach the held follower pose before the alignment timeout")
        except Exception as exc:
            failure = exc
        finally:
            torque_off_confirmed = self._turn_torque_off()

        if not torque_off_confirmed:
            failure = RuntimeError("Leader torque-off could not be confirmed; leader cleanup remains active")
        elif aligned and failure is None:
            if leader_timestamp is None or follower_timestamp is None:
                failure = RuntimeError("Fresh leader and follower timestamps were lost during alignment")
            else:
                try:
                    deadline = time.monotonic() + self._alignment_timeout_s
                    post_off = self._next_fresh_leader(leader_timestamp, deadline, generation, cancel)
                    follower = self._wait_for_follower_after(
                        follower_revision,
                        follower_timestamp,
                        deadline,
                        generation,
                        cancel,
                    )
                    if follower is None:
                        raise TimeoutError("No fresh follower observation followed leader torque-off")
                    leader_positions = np.asarray(post_off.joint_positions, dtype=np.float32)
                    follower_positions = follower[0]
                    if not self._poses_match(leader_positions, follower_positions):
                        raise RuntimeError("Fresh leader and follower poses did not match after torque-off")
                    if not self._poses_match(follower_positions, target):
                        raise RuntimeError("Follower moved away from the held target during leader alignment")
                    if not self._generation_is_current(generation, cancel):
                        raise RuntimeError("Leader alignment was cancelled before teleoperation could start")
                    with self._handover_lock:
                        if generation == self._handover_generation and not cancel.is_set() and not self._shutting_down:
                            self._follower_source = "teleop"
                            self._handover_state = "idle"
                            self._torque_cleanup_active = False
                            self._last_leader_timestamp = float(post_off.timestamp)
                            self._handover_thread = None
                            self._handover_cancel = None
                            self._handover_lock.notify_all()
                            self._emit_state()
                            return
                    failure = RuntimeError("Leader alignment was superseded before teleoperation could start")
                except Exception as exc:
                    failure = exc

        with self._handover_lock:
            self._torque_cleanup_active = not torque_off_confirmed
            if torque_off_confirmed:
                self._handover_state = "idle"
                self._handover_thread = None
                self._handover_cancel = None
            else:
                self._handover_state = "aligning"
            self._follower_source = "hold"
            self._handover_lock.notify_all()
        if failure is not None and not cancel.is_set():
            error_code = "leader_torque_off_failed" if not torque_off_confirmed else "leader_handover_failed"
            self._emit_error(str(failure) or "Leader alignment failed; the follower remains in hold.", error_code)
        self._emit_state()
        if not torque_off_confirmed and not self._shutting_down:
            self._retry_torque_off()

    def _turn_torque_off(self, *, log_failure: bool = True) -> bool:
        if self._leader is None:
            return True
        try:
            getattr(self._leader, "set_torque")(enabled=False)
        except Exception:
            if log_failure:
                logger.exception("Could not confirm leader torque-off")
            return False
        return True

    def _retry_torque_off(self) -> None:
        while not self._shutting_down:
            time.sleep(0.25)
            if self._turn_torque_off(log_failure=False):
                with self._handover_lock:
                    self._torque_cleanup_active = False
                    self._handover_state = "idle"
                    self._handover_thread = None
                    self._handover_cancel = None
                    self._follower_source = "hold"
                    self._handover_lock.notify_all()
                self._emit_state()
                return
        # Teardown makes one final attempt; never disconnect the leader unless this succeeds.
        if self._turn_torque_off(log_failure=False):
            with self._handover_lock:
                self._torque_cleanup_active = False
                self._handover_state = "idle"
                self._handover_thread = None
                self._handover_cancel = None
                self._handover_lock.notify_all()
            self._emit_state()

    def _next_fresh_leader(
        self,
        previous_timestamp: float,
        deadline: float,
        generation: int,
        cancel: threading.Event,
    ) -> RobotObservation:
        if self._leader is None:
            raise ConnectionError("Leader robot is unavailable")
        while time.monotonic() < deadline:
            self._raise_if_cancelled(generation, cancel)
            observation = self._leader.get_observation()
            timestamp = float(observation.timestamp)
            if timestamp > previous_timestamp:
                return observation
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        raise TimeoutError("Leader observations stopped advancing during alignment")

    def _latest_follower_after(self, revision: int, timestamp: float) -> tuple[np.ndarray, float, int] | None:
        with self._handover_lock:
            value = self._follower_observation
            if value is None or value[2] <= revision or value[1] <= timestamp:
                return None
            return value[0].copy(), value[1], value[2]

    def _wait_for_follower_after(
        self,
        revision: int,
        timestamp: float,
        deadline: float,
        generation: int,
        cancel: threading.Event,
    ) -> tuple[np.ndarray, float, int] | None:
        with self._handover_lock:
            while time.monotonic() < deadline:
                self._raise_if_cancelled(generation, cancel)
                value = self._follower_observation
                if value is not None and value[2] > revision and value[1] > timestamp:
                    return value[0].copy(), value[1], value[2]
                self._handover_lock.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        return None

    def _raise_if_cancelled(self, generation: int, cancel: threading.Event) -> None:
        if cancel.is_set() or not self._generation_is_current(generation, cancel):
            raise RuntimeError("Leader alignment was cancelled")

    def _generation_is_current(self, generation: int, cancel: threading.Event) -> bool:
        with self._handover_lock:
            return generation == self._handover_generation and not cancel.is_set() and not self._shutting_down

    def _poses_match(self, first: np.ndarray, second: np.ndarray) -> bool:
        return first.shape == second.shape and bool(
            np.all(np.isfinite(first))
            and np.all(np.isfinite(second))
            and np.max(np.abs(first - second), initial=0.0) <= self._alignment_tolerance
        )

    def _read_leader(self, robot_state: RobotObservation) -> tuple[np.ndarray | None, bool]:
        if self._leader is None:
            return None, False
        try:
            observation = self._leader.get_observation()
            if self._last_leader_timestamp is not None and observation.timestamp <= self._last_leader_timestamp:
                raise ConnectionError("Leader observation did not advance")
            action = np.array(observation.joint_positions, dtype=np.float32, copy=True)
            if action.shape != robot_state.joint_positions.shape or not np.all(np.isfinite(action)):
                raise ValueError(
                    f"Leader action shape {action.shape} does not match follower shape "
                    f"{robot_state.joint_positions.shape}, or contains non-finite values"
                )
        except Exception as exc:
            self._leader_failures += 1
            if not self._failure_logged:
                logger.warning("Leader observation failed; using hold: {}", exc)
                self._failure_logged = True
            if self._leader_failures >= self._max_leader_failures:
                self._disable_failed_leader(robot_state)
            return None, False

        self._leader_failures = 0
        self._failure_logged = False
        self._last_leader_timestamp = float(observation.timestamp)
        return action, True

    def _disable_failed_leader(self, robot_state: RobotObservation) -> None:
        self._leader_reads_enabled = False
        self._enter_hold(robot_state)
        logger.error("Leader failed for {} consecutive ticks; switching to hold", self._leader_failures)
        self._emit_state()
        self._emit_error(
            "The leader robot stopped responding. The follower switched to hold.",
            "leader_connection_lost",
        )

    def _policy_action(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],
        step: int,
    ) -> np.ndarray | None:
        with self._policy_lock:
            if self._policy is None or self._arming or self._follower_source != "policy":
                return None
            policy = self._policy
            try:
                return policy.update(robot_state, camera_frames, step)
            except WorkerDiedError:
                raise
            except Exception as exc:
                self._drop_policy_to_hold(robot_state, exc)
                return None

    def _drop_policy_to_hold(self, robot_state: RobotObservation, exc: Exception) -> None:
        self._enter_hold(robot_state)
        self._model_loaded = False
        self._loaded_identity = None
        self._emit_state()
        if not self._policy_error_emitted:
            logger.exception("Policy update failed; switching to hold")
            self._policy_error_emitted = True
            self._emit_error(str(exc) or "Policy inference failed.", "policy_inference_failed")

    def _set_policy(self, source: PolicySource, generation: int, identity: ModelIdentity | None = None) -> None:
        with self._policy_lock:
            if generation < self._attached_generation:
                stale = True
                previous = None
            else:
                stale = False
                self._cancel_arming_locked()
                previous = self._policy
                self._policy = source
                self._attached_generation = generation
                self._model_loaded = True
                self._loaded_identity = identity
                self._policy_error_emitted = False
        if stale:
            source.disconnect()
            return
        self._emit_state()
        if previous is not None and previous is not source:
            previous.disconnect()

    def _latest_observation(self) -> ObservationSnapshot:
        if self._last_robot_state is None:
            return None
        return self._last_robot_state, self._last_camera_frames

    def _emit_error(self, message: str, error_code: str) -> None:
        self._event_sink.emit(ErrorEvent(message=message, error_code=error_code))

    def _emit_state(self) -> None:
        self._event_sink.emit(StateEvent(data=self.state_data()))
