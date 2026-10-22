import asyncio
import queue
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from physicalai.config import Config
from physicalai.robot import SharedRobot

from runtime.contract import DisconnectCommand, QueueEventSink
from runtime.session import RuntimeSession
from tests.runtime.fakes import max_concurrent_connects, reset_connect_tracking


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


def _capabilities(position_space: str = "plugin.radians.v1") -> dict[str, Any]:
    return {
        "position_space": position_space,
        "position_tolerance": [0.01],
        "max_position_step": [0.02],
        "supports_alignment": False,
    }


@pytest.mark.parametrize("leader_space", ["plugin.radians.v1", "plugin.degrees.v1", None])
async def test_session_uses_plugin_coordinate_declarations(leader_space: str | None) -> None:
    document = _document_with_leader(
        handover_capabilities=_capabilities(),
        leader_kwargs={"handover_capabilities": None if leader_space is None else _capabilities(leader_space)},
    )
    session = RuntimeSession(document, event_sink=QueueEventSink())
    try:
        await session.setup()
        session._preconnect_devices()
        assert session._action_source is not None
        session._action_source.connect(bus=object(), session_id="test-session")

        assert session._action_source._position_space_comparable is (leader_space == "plugin.radians.v1")
        assert not session._action_source._automatic_alignment_supported
    finally:
        await session.teardown()


@pytest.mark.parametrize("with_leader", [False, True])
async def test_session_defers_shared_robot_metadata_reads_until_connected(mocker, with_leader: bool) -> None:
    document = _document()

    def recipe(name: str) -> dict[str, Any]:
        return {
            "class_path": "physicalai.robot.SharedRobot",
            "init_args": {"name": name, "robot": _document()["init_args"]["robot"]},
        }

    document["init_args"]["robot"] = recipe("test-follower")
    if with_leader:
        document["init_args"]["action_source"] = {"init_args": {"leader": recipe("test-leader")}}

    def connect(robot: SharedRobot) -> None:
        robot._metadata = {"joint_names": ["joint"], "handover_capabilities": _capabilities()}
        robot._connected = True

    connect_mock = mocker.patch.object(SharedRobot, "connect", autospec=True, side_effect=connect)
    session = RuntimeSession(document, event_sink=QueueEventSink())
    try:
        await session.setup()

        connect_mock.assert_not_called()
        assert session._action_source is not None
        assert not session._action_source._position_space_comparable

        session._preconnect_devices()
        session._action_source.connect(bus=object(), session_id="test-session")

        assert connect_mock.call_count == (2 if with_leader else 1)
        supports_metadata = hasattr(SharedRobot, "handover_capabilities")
        assert session._action_source._position_space_comparable is (with_leader and supports_metadata)
    finally:
        await session.teardown()


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
