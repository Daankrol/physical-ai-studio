import numpy as np
import pytest
from physicalai.capture import Frame

from runtime.action_source import StudioActionSource
from runtime.contract import InMemoryCommandMailbox, QueueEventSink, SetFollowerSourceCommand
from runtime.pose_retarget import PoseRetargeter

from .fakes import FakeObservation, FakePoseWorker, FakeRobot


def _observation(values: list[float], timestamp: float, *, efforts: list[float] | None = None) -> FakeObservation:
    sensor_data = None if efforts is None else {"efforts": np.array(efforts, dtype=np.float32)}
    return FakeObservation(np.array(values, dtype=np.float32), timestamp, sensor_data)


def _source(
    *,
    follower: FakeRobot,
    leader: FakeRobot | None,
    fps: float = 30,
    pose_retargeter=None,
    pose_worker=None,
    pose_camera_key=None,
    pose_camera_id=None,
):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
        pose_retargeter=pose_retargeter,
        pose_worker=pose_worker,
        pose_camera_key=pose_camera_key,
        pose_camera_id=pose_camera_id,
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


def _pose_landmarks() -> list[tuple[float, float, float, float]]:
    landmarks = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
    landmarks[23] = (-0.1, 0.9, 0.0, 1.0)
    landmarks[24] = (0.1, 0.9, 0.0, 1.0)
    landmarks[11] = (-0.2, 0.5, 0.0, 1.0)
    landmarks[12] = (0.2, 0.5, 0.0, 1.0)
    landmarks[13] = (-0.2, 0.8, 0.0, 1.0)
    landmarks[14] = (0.2, 0.8, 0.0, 1.0)
    landmarks[15] = (-0.2, 1.1, 0.3, 1.0)
    landmarks[16] = (0.2, 1.1, 0.3, 1.0)
    return landmarks


def _camera_frame(sequence: int = 1) -> Frame:
    return Frame(data=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=0.0, sequence=sequence)


def test_set_follower_source_pose_rejected_without_retargeter() -> None:
    follower = FakeRobot([_observation([0.0, 0.0], 1)], joint_names=["joint_1", "joint_2"])
    source, mailbox, events = _source(follower=follower, leader=None)
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))

    source.update(follower.get_observation(), {}, 0)

    assert source.follower_source == "hold"
    emitted = [events.get_nowait() for _ in range(1)]
    assert emitted[0].event == "error"
    assert emitted[0].error_code == "pose_not_supported"


def test_set_follower_source_pose_rejected_without_worker() -> None:
    follower = FakeRobot([_observation([0.0] * 12, 1)], joint_names=_so101_bimanual_joint_names())
    source, mailbox, events = _source(follower=follower, leader=None, pose_retargeter=_bimanual_retargeter())
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))

    source.update(follower.get_observation(), {}, 0)

    assert source.follower_source == "hold"
    emitted = [events.get_nowait() for _ in range(1)]
    assert emitted[0].error_code == "pose_not_supported"


def test_pose_mode_drives_the_follower_from_landmarks() -> None:
    follower = FakeRobot(
        [_observation([0.0] * 12, 1), _observation([0.0] * 12, 2)],
        joint_names=_so101_bimanual_joint_names(),
    )
    worker = FakePoseWorker()
    worker.world_landmarks = _pose_landmarks()
    worker.age_s = 0.01
    source, mailbox, _ = _source(
        follower=follower,
        leader=None,
        pose_retargeter=_bimanual_retargeter(),
        pose_worker=worker,
        pose_camera_key="cam",
        pose_camera_id="camera-uuid",
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))

    action = source.update(follower.get_observation(), {"cam": _camera_frame()}, 0)

    assert source.follower_source == "pose"
    assert not np.allclose(action, 0.0)
    assert worker.submitted_frames  # the camera frame reached the worker


def test_pose_mode_publishes_overlay_regardless_of_follower_source() -> None:
    follower = FakeRobot([_observation([0.0] * 12, 1)], joint_names=_so101_bimanual_joint_names())
    worker = FakePoseWorker()
    worker.image_landmarks = _pose_landmarks()
    worker.age_s = 0.01
    source, _, events = _source(
        follower=follower,
        leader=None,
        pose_retargeter=_bimanual_retargeter(),
        pose_worker=worker,
        pose_camera_key="cam",
        pose_camera_id="camera-uuid",
    )

    source.update(follower.get_observation(), {"cam": _camera_frame()}, 0)

    emitted = [events.get_nowait() for _ in range(1)]
    assert emitted[0].event == "pose"
    assert emitted[0].camera_id == "camera-uuid"
    assert len(emitted[0].landmarks) == 33


def test_pose_mode_falls_back_to_hold_when_landmarks_go_stale() -> None:
    follower = FakeRobot(
        [_observation([0.0] * 12, 1), _observation([0.0] * 12, 2)],
        joint_names=_so101_bimanual_joint_names(),
    )
    worker = FakePoseWorker()
    worker.world_landmarks = _pose_landmarks()
    worker.age_s = 0.01
    source, mailbox, events = _source(
        follower=follower,
        leader=None,
        pose_retargeter=_bimanual_retargeter(),
        pose_worker=worker,
        pose_camera_key="cam",
        pose_camera_id="camera-uuid",
    )
    mailbox.apply(SetFollowerSourceCommand(follower_source="pose"))
    source.update(follower.get_observation(), {"cam": _camera_frame()}, 0)
    assert source.follower_source == "pose"

    worker.age_s = 999.0
    source.update(follower.get_observation(), {"cam": _camera_frame(sequence=2)}, 1)

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
