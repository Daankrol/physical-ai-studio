import numpy as np
import pytest

from runtime.action_source import StudioActionSource
from runtime.contract import (
    InMemoryCommandMailbox,
    PoseLandmark,
    QueueEventSink,
    SetFollowerSourceCommand,
    SetPoseLandmarksCommand,
)
from runtime.pose_retarget import PoseRetargeter

from .fakes import FakeObservation, FakeRobot


def _observation(values: list[float], timestamp: float, *, efforts: list[float] | None = None) -> FakeObservation:
    sensor_data = None if efforts is None else {"efforts": np.array(efforts, dtype=np.float32)}
    return FakeObservation(np.array(values, dtype=np.float32), timestamp, sensor_data)


def _source(*, follower: FakeRobot, leader: FakeRobot | None, fps: float = 30, pose_retargeter=None):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
        pose_retargeter=pose_retargeter,
    )
    source.connect(bus=object(), session_id="test")
    return source, mailbox, events


def test_hold_latches_the_target_on_entry() -> None:
    follower = FakeRobot([_observation([1, 2], 1), _observation([3, 4], 2)])
    source, _, _ = _source(follower=follower, leader=None)

    first = source.update(follower.get_observation(), {}, 0)
    second = source.update(follower.get_observation(), {}, 1)

    np.testing.assert_array_equal(first, [1, 2])
    np.testing.assert_array_equal(second, [1, 2])


def test_teleop_forwards_leader_positions_on_next_tick() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = FakeRobot([_observation([4, 5], 1)])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [4, 5])


def test_connect_rejects_mismatched_joint_names() -> None:
    follower = FakeRobot([_observation([0], 1)], joint_names=["follower_joint"])
    leader = FakeRobot([_observation([0], 1)], joint_names=["leader_joint"])

    with pytest.raises(ValueError, match="joint names must match"):
        _source(follower=follower, leader=leader)


def test_leader_read_error_uses_hold_and_session_survives() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [1, 2])
    assert source.follower_source == "teleop"


def test_leader_failure_is_bounded_and_emits_one_error() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, events = _source(follower=follower, leader=leader, fps=1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    state = follower.get_observation()

    for step in range(6):
        source.update(state, {}, step)

    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except Exception:
            break
    assert source.follower_source == "hold"
    assert sum(event.event == "error" for event in emitted) == 1

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(state, {}, 7)
    assert source.follower_source == "hold"


def _so101_bimanual_joint_names() -> list[str]:
    suffixes = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return [f"{side}_{suffix}" for side in ("left", "right") for suffix in suffixes]


def _bimanual_retargeter() -> PoseRetargeter:
    return PoseRetargeter(_so101_bimanual_joint_names(), joint_limits_deg={})


def _pose_landmarks() -> list[PoseLandmark]:
    landmarks = [PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]
    landmarks[23] = PoseLandmark(x=-0.1, y=0.9, z=0.0, visibility=1.0)
    landmarks[24] = PoseLandmark(x=0.1, y=0.9, z=0.0, visibility=1.0)
    landmarks[11] = PoseLandmark(x=-0.2, y=0.5, z=0.0, visibility=1.0)
    landmarks[12] = PoseLandmark(x=0.2, y=0.5, z=0.0, visibility=1.0)
    landmarks[13] = PoseLandmark(x=-0.2, y=0.8, z=0.0, visibility=1.0)
    landmarks[14] = PoseLandmark(x=0.2, y=0.8, z=0.0, visibility=1.0)
    landmarks[15] = PoseLandmark(x=-0.2, y=1.1, z=0.3, visibility=1.0)
    landmarks[16] = PoseLandmark(x=0.2, y=1.1, z=0.3, visibility=1.0)
    return landmarks


def test_set_follower_source_pose_rejected_without_retargeter() -> None:
    follower = FakeRobot([_observation([0.0, 0.0], 1)], joint_names=["joint_1", "joint_2"])
    source, mailbox, events = _source(follower=follower, leader=None)
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))

    source.update(follower.get_observation(), {}, 0)

    assert source.follower_source == "hold"
    emitted = [events.get_nowait() for _ in range(1)]
    assert emitted[0].event == "error"
    assert emitted[0].error_code == "pose_not_supported"


def test_pose_mode_drives_the_follower_from_landmarks() -> None:
    follower = FakeRobot(
        [_observation([0.0] * 12, 1), _observation([0.0] * 12, 2)],
        joint_names=_so101_bimanual_joint_names(),
    )
    source, mailbox, _ = _source(follower=follower, leader=None, pose_retargeter=_bimanual_retargeter())
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))
    mailbox.apply(SetPoseLandmarksCommand(landmarks=_pose_landmarks()))

    action = source.update(follower.get_observation(), {}, 0)

    assert source.follower_source == "pose"
    assert not np.allclose(action, 0.0)


def test_pose_mode_falls_back_to_hold_when_landmarks_go_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    follower = FakeRobot(
        [_observation([0.0] * 12, 1), _observation([0.0] * 12, 2)],
        joint_names=_so101_bimanual_joint_names(),
    )
    source, mailbox, events = _source(follower=follower, leader=None, pose_retargeter=_bimanual_retargeter())
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))
    mailbox.apply(SetPoseLandmarksCommand(landmarks=_pose_landmarks()))
    source.update(follower.get_observation(), {}, 0)
    assert source.follower_source == "pose"

    import runtime.action_source as action_source_module

    monkeypatch.setattr(action_source_module.time, "monotonic", lambda: 1e12)

    source.update(follower.get_observation(), {}, 1)

    assert source.follower_source == "hold"
    codes = []
    while True:
        try:
            event = events.get_nowait()
        except Exception:
            break
        if event.event == "error":
            codes.append(event.error_code)
    assert "pose_connection_lost" in codes
