from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
from physicalai.robot import SharedRobot

from runtime.action_source import StudioActionSource
from runtime.callbacks.recording import RecordingState
from runtime.contract import InMemoryCommandMailbox, QueueEventSink, SetFollowerSourceCommand, StartRecordingCommand
from runtime.session import RuntimeSession
from tests.runtime.fakes import FakeRobot
from tests.runtime.test_action_source import (
    _ActivePoseLeader,
    _has_error,
    _observation,
    _PoseLeader,
    _source,
    _tick_until,
    _wait_until,
)
from tests.runtime.test_session import _document_with_leader


def _capabilities(**changes) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "position_space": "example-arm.radians-and-mm.v1",
            "position_tolerance": (0.01, 1.0),
            "max_position_step": (0.02, 2.0),
            "supports_alignment": False,
            **changes,
        }
    )


def test_plugin_limits_apply_per_joint_without_robot_type_checks() -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _PoseLeader([0.2, 20])
    follower.handover_capabilities = _capabilities()
    leader.handover_capabilities = _capabilities(position_tolerance=(0.02, 2.0))
    source, mailbox, _ = _source(follower=follower, leader=leader)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    action = source.update(_observation([0, 0], 1), {}, 0)

    np.testing.assert_allclose(action, [0.02, 2.0])
    assert source.last_action_source == "handover"
    assert source._poses_match(np.array([0.005, 0.5]), np.zeros(2))
    assert not source._poses_match(np.array([0.02, 0.5]), np.zeros(2))


def test_settings_cannot_relax_plugin_limits() -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _PoseLeader([0, 0])
    follower.handover_capabilities = _capabilities()
    leader.handover_capabilities = _capabilities()
    source, _, _ = _source(follower=follower, leader=leader, alignment_tolerance=0.5, max_follower_step=1.0)
    np.testing.assert_allclose(source._alignment_tolerance, [0.01, 0.5])
    np.testing.assert_allclose(source._max_follower_step, [0.02, 1.0])


@pytest.mark.parametrize("leader_space", ["example-arm.degrees.v1", None, ""])
def test_matching_joint_names_do_not_authorize_incompatible_coordinates(leader_space: str | None) -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _ActivePoseLeader([0, 0])
    follower.handover_capabilities = _capabilities()
    leader.handover_capabilities = _capabilities(position_space=leader_space, supports_alignment=True)
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_observation([0, 0], 1), {}, 0)

    assert source.follower_source == "hold"
    assert not source._position_space_comparable
    assert leader.torque_commands == []


def test_declined_powered_alignment_uses_manual_release_despite_torque_method() -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _ActivePoseLeader([0, 0])
    follower.handover_capabilities = _capabilities()
    leader.handover_capabilities = _capabilities(supports_alignment=False)
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_observation([0, 0], 1), {}, 0)
    assert source.state_data().handover_state == "manual_required"
    assert leader.torque_commands == []
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    source.update(_observation([0, 0], 2), {}, 1)
    assert source.follower_source == "teleop"
    assert leader.torque_commands == []


@pytest.mark.parametrize("joint_count", [1, 6, 12, 14])
def test_passive_plugin_takeover_accepts_arbitrary_joint_counts(joint_count: int) -> None:
    positions = [0.0] * joint_count
    follower = FakeRobot(positions=[positions])
    leader = _PoseLeader(positions)
    source, mailbox, _ = _source(follower=follower, leader=leader, model_loaded=True)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_observation(positions, 1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    action = source.update(_observation(positions, 2), {}, 1)
    assert source.follower_source == "teleop"
    assert source.last_action_source == "teleop"
    assert action.shape == (joint_count,)


@pytest.mark.parametrize("limits", [(0.01,), (0, 1), (float("nan"), 1), (float("inf"), 1)])
def test_invalid_direct_plugin_limits_fail_before_handover(limits: tuple[float, ...]) -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _PoseLeader([0, 0])
    follower.handover_capabilities = _capabilities(max_position_step=limits)
    leader.handover_capabilities = _capabilities()
    with pytest.raises(ValueError, match="handover limits"):
        _source(follower=follower, leader=leader)


@pytest.mark.parametrize("leader_space", [None, "incompatible.degrees.v1", "example-arm.radians-and-mm.v1"])
def test_plain_teleop_cannot_authorize_correction_recording(leader_space: str | None) -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _PoseLeader([0, 0])
    follower.handover_capabilities = _capabilities()
    leader.handover_capabilities = _capabilities(position_space=leader_space)
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    recording = RecordingState()
    recording.attach_mutation(MagicMock())
    source.bind_recording(recording)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    mailbox.apply(StartRecordingCommand(task="recover", intervention=True))
    source.update(_observation([0, 0], 1), {}, 0)

    assert source.follower_source == "teleop"
    assert not recording.is_recording
    assert _has_error(events, "correction_not_ready")


def test_explicit_release_authorizes_corrections_only_until_control_changes() -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _PoseLeader([0, 0])
    source, mailbox, events = _source(follower=follower, leader=leader, model_loaded=True)
    recording = RecordingState()
    recording.attach_mutation(MagicMock())
    source.bind_recording(recording)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
    source.update(_observation([0, 0], 1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
    mailbox.apply(StartRecordingCommand(task="recover", intervention=True))
    source.update(_observation([0, 0], 2), {}, 1)
    assert recording.is_recording

    recording.mark_discarded()
    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    source.update(_observation([0, 0], 3), {}, 2)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    mailbox.apply(StartRecordingCommand(task="recover", intervention=True))
    source.update(_observation([0, 0], 4), {}, 3)
    assert not recording.is_recording
    assert _has_error(events, "correction_not_ready")


@pytest.mark.parametrize("delayed_cached_samples", [0, 1, 2])
def test_pre_torque_off_follower_feedback_cannot_release_handover(
    monkeypatch: pytest.MonkeyPatch, delayed_cached_samples: int
) -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _ActivePoseLeader([0, 0])
    source, mailbox, events = _source(
        follower=follower, leader=leader, model_loaded=True, fps=100, alignment_timeout_s=0.1
    )
    disabling = threading.Event()
    finish_disable = threading.Event()
    set_torque = leader.set_torque

    def blocked_torque_off(*, enabled: bool) -> None:
        if not enabled:
            disabling.set()
            assert finish_disable.wait(1)
        set_torque(enabled=enabled)

    monkeypatch.setattr(leader, "set_torque", blocked_torque_off)
    try:
        mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
        _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
        worker = source._handover_thread
        assert worker is not None
        mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
        source.update(_observation([0, 0], time.monotonic()), {}, 0)
        assert disabling.wait(1)
        cached_timestamp = time.monotonic()
        queued = _observation([0, 0], cached_timestamp + 0.000001)
        follower_read = MagicMock(return_value=queued)
        if delayed_cached_samples == 2:
            # Runtime already holds A while SharedRobot has queued B; neither is post-ack.
            monkeypatch.setattr(follower, "get_observation", follower_read)
        if delayed_cached_samples:
            finish_disable.set()
            _wait_until(lambda: source._follower_feedback_barrier is not None)
            source.update(_observation([0, 0], cached_timestamp), {}, 1)
            if delayed_cached_samples == 2:
                source.update(queued, {}, 2)
                follower_read.assert_called_once()
        else:
            source.update(_observation([0, 0], cached_timestamp), {}, 1)
            finish_disable.set()
        worker.join(1)

        assert not worker.is_alive()
        assert source.follower_source == "hold"
        assert _has_error(events, "leader_handover_failed")
    finally:
        finish_disable.set()
        source.shutdown_handover()


def test_cached_pre_torque_off_leader_feedback_cannot_release_handover(monkeypatch: pytest.MonkeyPatch) -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _ActivePoseLeader([0, 0])
    source, mailbox, events = _source(
        follower=follower, leader=leader, model_loaded=True, fps=100, alignment_timeout_s=0.1
    )
    set_torque = leader.set_torque
    get_observation = leader.get_observation

    def release_with_cached_feedback(*, enabled: bool) -> None:
        if not enabled:
            cached = get_observation()
            monkeypatch.setattr(leader, "get_observation", lambda: cached)
        set_torque(enabled=enabled)

    monkeypatch.setattr(leader, "set_torque", release_with_cached_feedback)
    try:
        mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
        _tick_until(source, lambda: source.state_data().handover_state == "awaiting_release")
        mailbox.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
        _tick_until(source, lambda: source.state_data().handover_state == "idle")

        assert source.follower_source == "hold"
        assert _has_error(events, "leader_handover_failed")
    finally:
        source.shutdown_handover()


@pytest.mark.integration
@pytest.mark.parametrize("powered", [False, True])
@pytest.mark.skipif(not hasattr(SharedRobot, "handover_capabilities"), reason="Runtime lacks capability transport")
async def test_session_reads_non_so101_capabilities_from_real_owners(powered: bool) -> None:
    capabilities = {
        "position_space": "example-arm.radians.v1",
        "position_tolerance": [0.01],
        "max_position_step": [0.02],
        "supports_alignment": False,
    }
    document = _document_with_leader(
        handover_capabilities=capabilities,
        leader_kwargs={"handover_capabilities": {**capabilities, "supports_alignment": powered}},
    )
    if powered:
        document["init_args"]["action_source"]["init_args"]["leader"]["class_path"] = (
            "tests.runtime.fakes.ActiveFakeRobot"
        )
    robot_args = [document["init_args"], document["init_args"]["action_source"]["init_args"]]
    for args, key in zip(robot_args, ("robot", "leader"), strict=True):
        args[key] = {
            "class_path": "physicalai.robot.SharedRobot",
            "init_args": {"name": f"handover-{uuid4().hex[:8]}", "robot": args[key]},
        }
    session = RuntimeSession(document, event_sink=QueueEventSink())
    stop = threading.Event()
    await session.setup()
    assert session._action_source is not None
    session._action_source._alignment_goal_time_s = 0.05
    session._action_source._alignment_timeout_s = 1.0
    thread = threading.Thread(target=session.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert await asyncio.to_thread(session.ready.wait, 10)
        assert session._action_source is not None
        assert session._action_source._position_space_comparable
        assert session._action_source._automatic_alignment_supported is powered
        np.testing.assert_allclose(session._action_source._max_follower_step, [0.02])
        if powered:
            source = session._action_source
            source._model_loaded = True
            session.apply(SetFollowerSourceCommand(follower_source="teleop", auto_align_leader=True))
            await asyncio.to_thread(_wait_until, lambda: source.state_data().handover_state == "awaiting_release")
            session.apply(SetFollowerSourceCommand(follower_source="teleop", release_alignment=True))
            await asyncio.to_thread(_wait_until, lambda: source.follower_source == "teleop")
            assert source._correction_takeover_confirmed
    finally:
        stop.set()
        await asyncio.to_thread(thread.join, 5)
        owners = [getattr(robot, "_owner", None) for robot in (session._follower, session._leader)]
        await session.teardown()
        for owner in owners:
            if owner is not None:
                owner.stop()
    assert not thread.is_alive()


def test_older_runtime_without_capability_metadata_fails_closed() -> None:
    follower = FakeRobot(positions=[[0, 0]])
    leader = _ActivePoseLeader([0, 0])
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=InMemoryCommandMailbox(),
        event_sink=QueueEventSink(),
        fps=30,
    )
    source.connect(bus=object(), session_id="old-runtime")
    assert not source._position_space_comparable
    assert not source._automatic_alignment_supported
