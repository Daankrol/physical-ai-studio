from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from runtime.action_source import StudioActionSource
from runtime.contract import (
    ErrorEvent,
    InMemoryCommandMailbox,
    QueueEventSink,
    SetFollowerSourceCommand,
    StartTaskCommand,
)

from .fakes import FakeObservation, FakeRobot


def _observation(values: list[float], timestamp: float) -> FakeObservation:
    return FakeObservation(np.array(values, dtype=np.float32), timestamp)


class _PoseLeader(FakeRobot):
    def __init__(
        self,
        positions: list[float],
        *,
        stuck: bool = False,
        observation_error: str | None = None,
        fail_after_observations: int | None = None,
    ):
        super().__init__([_observation(positions, 1.0)])
        self.position = np.asarray(positions, dtype=np.float32)
        self.timestamp = 1.0
        self.stuck = stuck
        self.observation_error = observation_error
        self.fail_after_observations = fail_after_observations
        self.observation_count = 0
        self.motion: list[np.ndarray] = []
        self.goal_times: list[float] = []

    def get_observation(self) -> FakeObservation:
        self.observation_count += 1
        if self.observation_error is not None or (
            self.fail_after_observations is not None and self.observation_count > self.fail_after_observations
        ):
            raise ConnectionError(self.observation_error or "leader lost")
        self.timestamp += 1.0
        return _observation(self.position.tolist(), self.timestamp)

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        target = np.asarray(action, dtype=np.float32).copy()
        self.motion.append(target)
        self.goal_times.append(goal_time)
        if not self.stuck:
            self.position = target


class _ActivePoseLeader(_PoseLeader):
    def __init__(self, *args, torque_off_failures: int = 0, block_motion: threading.Event | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.torque_enabled = False
        self.torque_commands: list[bool] = []
        self.torque_off_failures = torque_off_failures
        self.block_motion = block_motion
        self.order: list[str] = []

    def set_torque(self, *, enabled: bool) -> None:
        self.order.append("torque_on" if enabled else "torque_off")
        self.torque_commands.append(enabled)
        if not enabled and self.torque_off_failures:
            self.torque_off_failures -= 1
            raise ConnectionError("torque-off acknowledgement lost")
        self.torque_enabled = enabled

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        assert self.torque_enabled
        if self.block_motion is not None:
            self.block_motion.wait(timeout=2)
        self.order.append("move")
        super().send_action(action, goal_time=goal_time)


def _source(
    *,
    follower: FakeRobot,
    leader: FakeRobot | None,
    fps: float = 30,
    position_space_comparable: bool = True,
    model_loaded: bool = False,
    **kwargs,
):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    kwargs.setdefault("leader_alignment_goal_time_s", 0.02)
    for robot in (follower, leader):
        if robot is not None and robot.handover_capabilities is None:
            robot.handover_capabilities = SimpleNamespace(
                position_space="test.normalized.v1" if position_space_comparable else None,
                position_tolerance=(0.5,) * len(robot.joint_names),
                max_position_step=(0.5,) * len(robot.joint_names),
                supports_alignment=isinstance(robot, _ActivePoseLeader),
            )
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
        **kwargs,
    )
    source.connect(bus=object(), session_id="test")
    source._model_loaded = model_loaded
    # Manual ticks are the follower's hardware feedback in these tests.
    follower.get_observation = MagicMock(side_effect=lambda: source._last_robot_state)  # type: ignore[method-assign]
    return source, mailbox, events


def _drain(events: QueueEventSink) -> list:
    items = []
    while True:
        try:
            items.append(events.get_nowait())
        except Exception:
            return items


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("condition was not met in time")


def _tick_until(source: StudioActionSource, predicate) -> None:
    def tick() -> bool:
        source.update(_observation([0, 0], time.monotonic()), {}, 0)
        return predicate()

    _wait_until(tick)


def _fresh_follower(timestamp: int) -> FakeObservation:
    return _observation([0.0, 0.0], float(timestamp))


def _has_error(events: QueueEventSink, code: str) -> bool:
    return any(isinstance(event, ErrorEvent) and event.error_code == code for event in _drain(events))


def test_hold_latches_last_commanded_action() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    source, mailbox, _ = _source(follower=follower, leader=None)

    first = source.update(_observation([1, 2], 1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    second = source.update(_observation([3, 4], 2), {}, 1)

    np.testing.assert_array_equal(first, [1, 2])
    np.testing.assert_array_equal(second, [1, 2])
    assert source.last_action_source == "hold"


@pytest.mark.parametrize("model_loaded", [False, True])
def test_ordinary_teleop_follows_mismatched_leader_without_alignment(model_loaded: bool) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, -2])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        max_follower_step=0.1,
        position_space_comparable=False,
        model_loaded=model_loaded,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=False))

    action = source.update(_fresh_follower(1), {}, 0)

    np.testing.assert_allclose(action, [0.1, -0.1])
    assert source.follower_source == "teleop"
    assert source.state_data().handover_state == "idle"
    assert source.last_action_source == "handover"
    assert leader.torque_commands == []
    assert leader.motion == []
    assert not any(isinstance(event, ErrorEvent) for event in _drain(events))


def test_plain_teleop_request_does_not_align_even_with_a_loaded_model() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(_fresh_follower(1), {}, 0)

    np.testing.assert_array_equal(action, [0.5, 0.5])
    assert source.follower_source == "teleop"
    assert source.state_data().handover_state == "idle"
    assert leader.torque_commands == []
    assert leader.motion == []


def test_automatic_alignment_requires_a_loaded_model() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, events = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))

    action = source.update(_fresh_follower(1), {}, 0)

    np.testing.assert_array_equal(action, [0, 0])
    assert source.follower_source == "hold"
    assert leader.torque_commands == []
    assert leader.motion == []
    assert source.state_data().handover_state == "idle"
    assert _has_error(events, "policy_not_loaded")


def test_successful_alignment_holds_torque_until_explicit_release_then_enters_teleop() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=50,
        model_loaded=True,
        leader_alignment_goal_time_s=0.05,
        alignment_tolerance=0.01,
        alignment_timeout_s=0.2,
        max_follower_step=0.1,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))

    _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
    time.sleep(0.15)
    np.testing.assert_array_equal(source.update(_observation([0, 0], time.monotonic()), {}, 0), [0, 0])
    assert source.follower_source == "hold"
    assert source.last_action_source == "handover"
    assert leader.torque_commands == [True]
    assert leader.torque_enabled

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    _tick_until(source, lambda: source.follower_source == "teleop")

    assert source.follower_source == "teleop"
    assert leader.torque_commands == [True, False]
    assert leader.order[0] == "torque_on"
    assert leader.order[-1] == "torque_off"
    assert source.state_data().handover_state == "idle"
    assert leader.goal_times == [0.02] * len(leader.motion)
    assert len(leader.motion) == 3
    np.testing.assert_array_equal(leader.motion[-1], [0, 0])
    assert leader.order.index("torque_off") > leader.order.index("move")


@pytest.mark.parametrize("read_delay", [0.0, 0.02])
def test_alignment_streams_bounded_positions_without_native_move_timing(
    read_delay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, -2])
    observe = leader.get_observation

    def delayed_observation() -> FakeObservation:
        time.sleep(read_delay)
        return observe()

    monkeypatch.setattr(leader, "get_observation", delayed_observation)
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=100,
        model_loaded=True,
        leader_alignment_goal_time_s=0.1,
        alignment_tolerance=0.01,
        alignment_timeout_s=1,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    try:
        _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")

        assert len(leader.motion) >= 10
        assert leader.goal_times == pytest.approx([0.01] * len(leader.motion))
        positions = np.vstack(([2, -2], leader.motion))
        assert np.max(np.abs(np.diff(positions, axis=0))) <= 0.201
        np.testing.assert_allclose(leader.motion[-1], [0, 0], atol=1e-6)
        np.testing.assert_array_equal(source.update(_observation([0, 0], time.monotonic()), {}, 0), [0, 0])
        assert leader.torque_enabled
        assert source.follower_source == "hold"
    finally:
        assert source.shutdown_handover()


def test_timeout_during_ramp_does_not_send_the_final_target() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=100,
        model_loaded=True,
        leader_alignment_goal_time_s=1,
        alignment_tolerance=0.01,
        alignment_timeout_s=0.05,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    try:
        _tick_until(source, lambda: source.state_data().handover_state == "alignment_failed")
        assert leader.motion
        assert all(np.all(target > 1) for target in leader.motion)
        motion_count = len(leader.motion)
        time.sleep(0.05)
        assert len(leader.motion) == motion_count
        assert leader.torque_enabled
        assert source.follower_source == "hold"
    finally:
        assert source.shutdown_handover()


def test_release_before_alignment_result_is_rejected() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))

    source.update(_fresh_follower(1), {}, 0)

    assert source.follower_source == "hold"
    assert leader.torque_commands == []
    assert _has_error(events, "alignment_not_ready")


def test_pose_drift_before_release_still_allows_operator_takeover() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
    leader.position = np.array([2, 2], dtype=np.float32)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))

    _tick_until(source, lambda: source.follower_source == "teleop")

    assert source.state_data().handover_state == "idle"
    assert leader.torque_commands == [True, False]
    previous = source._last_commanded_action.copy()
    action = source.update(_observation([0, 0], time.monotonic()), {}, 0)
    assert np.max(np.abs(action - previous)) <= 0.5
    assert source.last_action_source == "handover"


@pytest.mark.parametrize("invalid_pose", [[float("nan"), 0], [0]])
def test_release_rejects_invalid_leader_feedback(invalid_pose: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
    set_torque = leader.set_torque

    def release_with_invalid_pose(*, enabled: bool) -> None:
        set_torque(enabled=enabled)
        if not enabled:
            leader.position = np.asarray(invalid_pose, dtype=np.float32)

    monkeypatch.setattr(leader, "set_torque", release_with_invalid_pose)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    _tick_until(source, lambda: source.state_data().handover_state == "idle")

    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]
    assert _has_error(events, "leader_handover_failed")


def test_release_still_requires_fresh_follower_feedback() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True, alignment_timeout_s=0.1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    source.update(_observation([0, 0], time.monotonic()), {}, 0)

    _wait_until(lambda: source.state_data().handover_state == "idle")

    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]
    assert _has_error(events, "leader_handover_failed")


@pytest.mark.parametrize("stuck", [False, True])
def test_shutdown_while_waiting_for_release_turns_torque_off(stuck: bool) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2], stuck=stuck)
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True, alignment_timeout_s=0.1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    result = "alignment_failed" if stuck else "awaiting_release"
    _tick_until(source, lambda: source.state_data().handover_state == result)

    assert source.shutdown_handover()
    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]


@pytest.mark.parametrize("confirmation", ["release_alignment", "confirm_manual_alignment"])
def test_unsupported_alignment_allows_takeover_without_a_pose_match(confirmation: str) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([2, -2])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.05,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    assert source.state_data().handover_state == "manual_required"

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", **{confirmation: True}))
    action = source.update(_observation([1, 1], 2), {}, 1)

    assert source.follower_source == "teleop"
    assert source.state_data().handover_state == "idle"
    np.testing.assert_array_equal(action, [0.5, -0.5])
    assert source.last_action_source == "handover"
    assert not any(isinstance(event, ErrorEvent) for event in _drain(events))


@pytest.mark.parametrize("stale_leader", [False, True])
def test_fallback_release_still_requires_fresh_robot_feedback(stale_leader: bool) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([1, 0])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.05,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    if stale_leader:
        leader.observation_error = "leader lost"
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    source.update(_fresh_follower(2 if stale_leader else 1), {}, 1)

    assert source.follower_source == "hold"
    assert source.state_data().handover_state == "manual_required"
    assert _has_error(events, "leader_handover_stale")

    leader.observation_error = None
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    source.update(_fresh_follower(3), {}, 2)
    assert source.follower_source == "teleop"


def test_fallback_release_rejects_nonfinite_follower_feedback() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([2, 2])
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))

    action = source.update(_observation([float("nan"), 0], 2), {}, 1)

    assert source.follower_source == "hold"
    np.testing.assert_array_equal(action, [0, 0])
    assert _has_error(events, "leader_handover_failed")


def test_unknown_position_space_fails_closed_even_with_torque_capability() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([0, 0])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        position_space_comparable=False,
        model_loaded=True,
    )
    source._follower_source = "policy"
    source._last_commanded_action = np.array([1.0, 2.0], dtype=np.float32)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    action = source.update(_fresh_follower(1), {}, 0)

    np.testing.assert_array_equal(action, [1, 2])
    assert source.follower_source == "hold"
    assert source.state_data().handover_state == "idle"
    assert leader.torque_commands == []
    assert _has_error(events, "leader_position_space_unsupported")


def test_leader_read_failure_returns_hold_origin_after_successful_manual_entry() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([0, 0], observation_error="lost")
    source, _, _ = _source(follower=follower, leader=leader, fps=1)
    # Manually establish a confirmed mode to isolate the tick provenance behavior.
    source._follower_source = "teleop"
    source._hold_target = np.array([0, 0], dtype=np.float32)
    action = source.update(_fresh_follower(1), {}, 0)

    np.testing.assert_array_equal(action, [0, 0])
    assert source.follower_source == "teleop"
    assert source.last_action_source == "hold"


def test_alignment_cancellation_keeps_hold_and_cleans_torque() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    release_motion = threading.Event()
    leader = _ActivePoseLeader([3, 3], block_motion=release_motion)
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=50,
        alignment_timeout_s=2,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    _wait_until(lambda: bool(leader.motion) or leader.torque_enabled)

    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    source.update(_fresh_follower(2), {}, 1)
    release_motion.set()
    _wait_until(lambda: source.state_data().handover_state == "idle")

    assert source.follower_source == "hold"
    assert leader.torque_commands[-1] is False


def test_leader_loss_during_alignment_fails_to_hold_and_turns_torque_off() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([3, 3], fail_after_observations=1)
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)

    _wait_until(lambda: source.state_data().handover_state == "idle")
    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]
    assert _has_error(events, "leader_handover_failed")


def test_alignment_timeout_waits_for_release_then_allows_takeover_outside_tolerance() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([4, 4], stuck=True)
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=100,
        alignment_timeout_s=0.1,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    _tick_until(source, lambda: source.state_data().handover_state == "alignment_failed")
    time.sleep(0.15)
    assert source.follower_source == "hold"
    assert leader.torque_commands == [True]
    assert leader.torque_enabled
    assert len(leader.motion) == 2
    assert leader.goal_times == [0.01, 0.01]

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    _tick_until(source, lambda: source.follower_source == "teleop")
    assert source.state_data().handover_state == "idle"
    assert leader.torque_commands == [True, False]
    np.testing.assert_array_equal(leader.position, [4, 4])

    previous = source._last_commanded_action.copy()
    action = source.update(_observation([0, 0], time.monotonic()), {}, 0)
    assert np.max(np.abs(action - previous)) <= 0.5
    assert source.last_action_source == "handover"


def test_torque_off_failure_blocks_policy_until_cleanup_settles() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([0, 0], torque_off_failures=1)
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        fps=100,
        alignment_timeout_s=1,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    source.update(_observation([0, 0], time.monotonic()), {}, 0)
    _wait_until(lambda: _has_error(events, "leader_torque_off_failed"))
    mailbox.apply(StartTaskCommand(task="resume"))
    source.update(_observation([0, 0], time.monotonic()), {}, 0)
    assert source.follower_source == "hold"
    assert _has_error(events, "leader_torque_cleanup_active")
    _wait_until(lambda: not source.torque_cleanup_active)
    assert source.follower_source == "hold"
    assert leader.torque_commands[-1] is False


@pytest.mark.parametrize("model_takeover", [False, True])
def test_takeover_guard_catches_up_while_the_leader_keeps_moving(model_takeover: bool) -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([2, -2])
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=model_takeover)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=model_takeover))
    source.update(_fresh_follower(1), {}, 0)
    if model_takeover:
        assert source.state_data().handover_state == "manual_required"
        mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
        source.update(_fresh_follower(2), {}, 1)

    for step, expected in enumerate(([3, -3], [5.5, -5.5], [8, -8], [10, -10]), start=1):
        leader.position = np.array([2 + 2 * step, -2 - 2 * step], dtype=np.float32)
        action = source.update(_fresh_follower(step + 2), {}, step + 1)
        np.testing.assert_array_equal(action, expected)

    assert source.last_action_source == "teleop"
    assert leader.motion == []
    leader.position = np.array([50, -50], dtype=np.float32)
    np.testing.assert_array_equal(source.update(_fresh_follower(7), {}, 6), [50, -50])


def test_takeover_guard_limits_a_stationary_gap_until_caught_up() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([2, -2])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    for step, expected in enumerate(([0.5, -0.5], [1, -1], [1.5, -1.5], [2, -2])):
        action = source.update(_fresh_follower(step + 1), {}, step)
        np.testing.assert_array_equal(action, expected)
        assert source.last_action_source == ("teleop" if step == 3 else "handover")


def test_takeover_guard_does_not_overshoot_a_reversing_leader() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([4, -4])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    np.testing.assert_array_equal(source.update(_fresh_follower(1), {}, 0), [0.5, -0.5])

    leader.position = np.array([3, -3], dtype=np.float32)
    np.testing.assert_array_equal(source.update(_fresh_follower(2), {}, 1), [2, -2])
    leader.position = np.array([1, -1], dtype=np.float32)
    np.testing.assert_array_equal(source.update(_fresh_follower(3), {}, 2), [1, -1])
    assert source.last_action_source == "teleop"


def test_takeover_guard_restarts_without_previous_leader_motion() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([2, -2])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)
    leader.position = np.array([4, -4], dtype=np.float32)
    np.testing.assert_array_equal(source.update(_fresh_follower(2), {}, 1), [3, -3])
    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    source.update(_fresh_follower(3), {}, 2)

    leader.position = np.array([20, -20], dtype=np.float32)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    np.testing.assert_array_equal(source.update(_fresh_follower(4), {}, 3), [3.5, -3.5])
    assert source.last_action_source == "handover"


def test_fallback_takeover_outside_tolerance_is_step_limited_and_unrecordable() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([0.4, 0])
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.01,
        max_follower_step=0.1,
        model_loaded=True,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_fresh_follower(1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    action = source.update(_fresh_follower(2), {}, 1)

    np.testing.assert_allclose(action, [0.1, 0])
    assert source.follower_source == "teleop"
    assert source.last_action_source == "handover"
    assert source.last_action_source != "teleop"
