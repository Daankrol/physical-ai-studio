from uuid import uuid4

from repositories.mappers.project_environment_mapper import ProjectEnvironmentMapper
from schemas.environment import (
    CameraEnvironmentConfiguration,
    Environment,
    RobotEnvironmentConfiguration,
    TeleoperatorNone,
    TeleoperatorPose,
    TeleoperatorRobot,
)


def _environment(tele_operator) -> Environment:
    return Environment(
        id=uuid4(),
        name="Test Environment",
        robots=[RobotEnvironmentConfiguration(robot_id=uuid4(), tele_operator=tele_operator)],
        cameras=[CameraEnvironmentConfiguration(camera_id=uuid4())],
    )


def test_build_robot_links_encodes_pose_teleoperator() -> None:
    camera_id = uuid4()
    environment = _environment(TeleoperatorPose(camera_id=camera_id))

    links = ProjectEnvironmentMapper.build_robot_links(environment)

    assert len(links) == 1
    assert links[0].tele_operator_type == "pose"
    assert links[0].tele_operator_robot_id is None
    assert links[0].tele_operator_camera_id == str(camera_id)


def test_build_robot_links_encodes_robot_teleoperator() -> None:
    teleop_robot_id = uuid4()
    environment = _environment(TeleoperatorRobot(robot_id=teleop_robot_id))

    links = ProjectEnvironmentMapper.build_robot_links(environment)

    assert links[0].tele_operator_type == "robot"
    assert links[0].tele_operator_robot_id == str(teleop_robot_id)
    assert links[0].tele_operator_camera_id is None


def test_build_robot_links_encodes_none_teleoperator() -> None:
    environment = _environment(TeleoperatorNone())

    links = ProjectEnvironmentMapper.build_robot_links(environment)

    assert links[0].tele_operator_type == "none"
    assert links[0].tele_operator_robot_id is None
    assert links[0].tele_operator_camera_id is None


def test_from_schema_round_trips_pose_teleoperator() -> None:
    camera_id = uuid4()
    environment = _environment(TeleoperatorPose(camera_id=camera_id))
    db_model = ProjectEnvironmentMapper.to_schema(environment)
    db_model.robot_links = ProjectEnvironmentMapper.build_robot_links(environment)
    db_model.camera_links = ProjectEnvironmentMapper.build_camera_links(environment)

    round_tripped = ProjectEnvironmentMapper.from_schema(db_model)

    assert round_tripped.robots[0].tele_operator.type == "pose"
    assert round_tripped.robots[0].tele_operator.camera_id == camera_id
