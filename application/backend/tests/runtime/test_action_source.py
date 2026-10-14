from __future__ import annotations

import threading
import time

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
    **kwargs,
):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
        position_space_comparable=position_space_comparable,
        **kwargs,
    )
    source.connect(bus=object(), session_id="test")
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


def test_successful_alignment_holds_follower_and_turns_torque_off_before_teleop() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([2, 2])
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        fps=50,
        leader_alignment_speed=50,
        alignment_tolerance=0.01,
        alignment_timeout_s=2,
        max_follower_step=0.1,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    actions = []
    for tick in range(1, 300):
        action = source.update(_fresh_follower(tick), {}, tick)
        actions.append(action)
        if source.follower_source == "teleop":
            break
        assert source.last_action_source == "handover"
        np.testing.assert_array_equal(action, [0, 0])
        time.sleep(0.002)

    assert source.follower_source == "teleop"
    assert leader.torque_commands == [True, False]
    assert leader.order[0] == "torque_on"
    assert leader.order[-1] == "torque_off"
    assert source.state_data().handover_state == "idle"
    assert np.max(np.abs(np.diff(np.stack(leader.motion), axis=0))) <= 1.0
    assert all(np.array_equal(action, [0, 0]) for action in actions[:-1])
    assert leader.order.index("torque_off") > leader.order.index("move")


def test_manual_confirmation_requires_fresh_matching_leader_and_follower_poses() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([0, 0])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.05,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)
    assert source.state_data().handover_state == "manual_required"
    assert _has_error(events, "manual_alignment_required")

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", confirm_manual_alignment=True))
    source.update(_fresh_follower(2), {}, 1)

    assert source.follower_source == "teleop"
    assert source.last_action_source == "teleop"


def test_manual_confirmation_rejects_mismatch_and_stale_follower_observation() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([1, 0])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.05,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", confirm_manual_alignment=True))
    source.update(_fresh_follower(2), {}, 1)

    assert source.follower_source == "hold"
    assert source.state_data().handover_state == "manual_required"
    assert _has_error(events, "manual_alignment_mismatch")

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", confirm_manual_alignment=True))
    source.update(_fresh_follower(2), {}, 2)
    assert source.follower_source == "hold"
    assert _has_error(events, "manual_alignment_stale")


def test_unknown_position_space_fails_closed_even_with_torque_capability() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([0, 0])
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        position_space_comparable=False,
    )
    source._follower_source = "policy"
    source._last_commanded_action = np.array([1.0, 2.0], dtype=np.float32)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
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
        leader_alignment_speed=50,
        alignment_timeout_s=2,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
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
    source, mailbox, events = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)

    _wait_until(lambda: source.state_data().handover_state == "idle")
    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]
    assert _has_error(events, "leader_handover_failed")


def test_alignment_timeout_stays_in_hold_and_torque_off_is_attempted() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([4, 4], stuck=True)
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        fps=100,
        alignment_timeout_s=0.1,
        leader_alignment_speed=1,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    for tick in range(1, 30):
        source.update(_fresh_follower(tick), {}, tick)
        time.sleep(0.01)
        if source.state_data().handover_state == "idle":
            break

    assert source.follower_source == "hold"
    assert leader.torque_commands == [True, False]
    assert _has_error(events, "leader_handover_failed")


def test_torque_off_failure_blocks_policy_until_cleanup_settles() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _ActivePoseLeader([0, 0], torque_off_failures=1)
    source, mailbox, events = _source(
        follower=follower,
        leader=leader,
        fps=100,
        alignment_timeout_s=1,
        leader_alignment_speed=50,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)
    _wait_until(lambda: source.torque_cleanup_active)
    mailbox.apply(StartTaskCommand(task="resume"))
    source.update(_fresh_follower(2), {}, 1)
    assert source.follower_source == "hold"
    assert _has_error(events, "leader_torque_cleanup_active")
    _wait_until(lambda: not source.torque_cleanup_active)
    assert source.follower_source == "hold"
    assert leader.torque_commands[-1] is False


def test_manual_pose_within_tolerance_but_outside_step_limit_is_unrecordable_handover() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = _PoseLeader([0.4, 0])
    source, mailbox, _ = _source(
        follower=follower,
        leader=leader,
        alignment_tolerance=0.5,
        max_follower_step=0.1,
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_fresh_follower(1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", confirm_manual_alignment=True))
    action = source.update(_fresh_follower(2), {}, 1)

    np.testing.assert_allclose(action, [0.1, 0])
    assert source.follower_source == "teleop"
    assert source.last_action_source == "handover"
    assert source.last_action_source != "teleop"
