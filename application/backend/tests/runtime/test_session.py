import asyncio
import queue
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from physicalai.config import Config
from physicalai.robot.so101 import SO101

from runtime.contract import DisconnectCommand, QueueEventSink
from runtime.session import RuntimeSession, _position_spaces_comparable
from tests.runtime.fakes import FakeRobot, max_concurrent_connects, reset_connect_tracking


def _fake_robot_args(**kwargs: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"positions": [[1.0]], "joint_names": ["joint"]}
    args.update(kwargs)
    return args


def _document(*, connect_error: str | None = None, **follower_kwargs: Any) -> dict:
    robot_args = _fake_robot_args(**follower_kwargs)
    if connect_error is not None:
        robot_args["connect_error"] = connect_error
    return Config(
        "physicalai.runtime.RobotRuntime",
        {
            "robot": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": robot_args,
            },
            "cameras": {},
            "fps": 30.0,
        },
    ).to_dict()


def _camera_config(**kwargs: Any) -> dict[str, Any]:
    return {
        "class_path": "tests.runtime.fakes.FakeCamera",
        "init_args": kwargs,
    }


def _document_with_cameras(*names: str, **follower_kwargs: Any) -> dict:
    document = _document(**follower_kwargs)
    document["init_args"]["cameras"] = {name: _camera_config(name=name) for name in names}
    return document


def _document_with_leader(*, leader_kwargs: dict[str, Any] | None = None, **follower_kwargs: Any) -> dict:
    document = _document(**follower_kwargs)
    document["init_args"]["action_source"] = {
        "init_args": {
            "leader": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": _fake_robot_args(**(leader_kwargs or {})),
            },
        },
    }
    return document


def _so101_recipe(role: str, *, unit: str = "normalized", calibrated: bool = True, offset: int = 0) -> dict:
    names = SO101.JOINT_ORDER
    calibration = {
        name: {
            "id": index + 1,
            "drive_mode": 0,
            "homing_offset": offset,
            "range_min": 0,
            "range_max": 4095,
        }
        for index, name in enumerate(names)
    }
    return {
        "class_path": "physicalai.robot.SharedRobot",
        "init_args": {
            "robot": {
                "class_path": "physicalai.robot.SO101",
                "init_args": {
                    "role": role,
                    "unit": unit,
                    "calibration": calibration if calibrated else None,
                },
            }
        },
    }


def test_so101_position_space_requires_valid_normalized_calibrations() -> None:
    joints = SO101.JOINT_ORDER
    follower = FakeRobot(positions=[[0] * len(joints)], joint_names=joints)
    leader = FakeRobot(positions=[[0] * len(joints)], joint_names=joints)

    assert _position_spaces_comparable(
        _so101_recipe("follower", offset=3), _so101_recipe("leader", offset=17), follower, leader
    )
    assert not _position_spaces_comparable(
        _so101_recipe("follower", calibrated=False), _so101_recipe("leader"), follower, leader
    )
    assert not _position_spaces_comparable(
        _so101_recipe("follower", unit="ticks"), _so101_recipe("leader"), follower, leader
    )
    assert not _position_spaces_comparable(
        _so101_recipe("follower"), _so101_recipe("leader", unit="ticks"), follower, leader
    )
    assert not _position_spaces_comparable(
        _so101_recipe("follower"),
        _so101_recipe("leader"),
        follower,
        FakeRobot(positions=[[0] * len(joints)], joint_names=list(reversed(joints))),
    )


def test_unknown_plugin_position_space_requires_opt_in_from_both_robots() -> None:
    recipe = {"class_path": "plugin.SharedRobot", "init_args": {}}
    follower = FakeRobot(positions=[[0]], joint_names=["joint"])
    leader = FakeRobot(positions=[[0]], joint_names=["joint"])

    assert not _position_spaces_comparable(recipe, recipe, follower, leader)
    follower.handover_position_space_comparable = True
    leader.handover_position_space_comparable = True
    assert _position_spaces_comparable(recipe, recipe, follower, leader)


async def test_teardown_keeps_leader_connected_when_torque_off_is_uncertain() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document_with_leader(), event_sink=events)
    await session.setup()
    assert session._leader is not None
    assert session._action_source is not None
    session._action_source.shutdown_handover = MagicMock(return_value=False)
    session._action_source.shutdown_policy = MagicMock()
    session._leader.disconnect = MagicMock()

    await session.teardown()

    session._leader.disconnect.assert_not_called()
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            break
    assert any(getattr(event, "error_code", None) == "leader_torque_off_failed" for event in emitted)


async def test_session_runs_until_stop_signal() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    stop = threading.Event()
    await session.setup()
    thread = threading.Thread(target=session.run, args=(stop,), daemon=True)
    thread.start()
    await asyncio.to_thread(session.ready.wait, 2)
    stop.set()
    await asyncio.to_thread(thread.join, 2)
    await session.teardown()

    assert session._runtime is not None
    assert session._runtime.last_run_reason == "stop_requested"
    assert session._follower is not None
    assert not session._follower.is_connected()
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            break
    assert emitted[0].event == "state"
    assert emitted[-1].event == "lifecycle"


async def test_session_propagates_connect_error() -> None:
    session = RuntimeSession(_document(connect_error="connect failed"), event_sink=QueueEventSink())
    await session.setup()
    with pytest.raises(ConnectionError, match="connect failed"):
        session.run(threading.Event())
    await session.teardown()

    assert session._follower is not None
    assert not session._follower.is_connected()


async def test_preconnect_runs_follower_and_leader_connects_in_parallel() -> None:
    reset_connect_tracking()
    session = RuntimeSession(
        _document_with_leader(
            connect_delay=0.05,
            name="follower",
            leader_kwargs={"connect_delay": 0.05, "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    session._preconnect_devices()

    assert session._follower is not None
    assert session._leader is not None
    assert session._follower.is_connected()
    assert session._leader.is_connected()
    assert max_concurrent_connects() == 2


async def test_preconnect_disconnects_robots_when_one_connect_fails() -> None:
    session = RuntimeSession(
        _document_with_leader(
            name="follower",
            leader_kwargs={"connect_error": "leader connect failed", "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    with pytest.raises(ConnectionError, match="leader connect failed"):
        session._preconnect_devices()

    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()


async def test_disconnect_before_runtime_construction_stops_first_run() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    session.apply(DisconnectCommand())
    await session.setup()

    session.run(threading.Event())

    assert session._runtime is None
    assert session._follower is not None
    assert not session._follower.is_connected()


async def test_worker_stop_before_run_does_not_connect_hardware() -> None:
    session = RuntimeSession(_document_with_leader(), event_sink=QueueEventSink())
    await session.setup()
    stop = threading.Event()
    stop.set()

    session.run(stop)

    assert session._runtime is None
    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()
