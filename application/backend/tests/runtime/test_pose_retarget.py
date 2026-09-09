import numpy as np
import pytest

from runtime.pose_retarget import PoseLandmark, PoseRetargeter, build_arm_sides, detect_dof_profile

_OPENARM_JOINTS = [f"{side}_{suffix}" for side in ("left", "right") for suffix in ("joint_1", "joint_2")]


def _openarm_bimanual_joint_names() -> list[str]:
    suffixes = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
    return [f"{side}_{suffix}" for side in ("left", "right") for suffix in suffixes]


def _so101_bimanual_joint_names() -> list[str]:
    suffixes = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return [f"{side}_{suffix}" for side in ("left", "right") for suffix in suffixes]


def test_detect_dof_profile_matches_openarm() -> None:
    assert detect_dof_profile(_openarm_bimanual_joint_names()) == "openarm7"


def test_detect_dof_profile_matches_so101() -> None:
    assert detect_dof_profile(_so101_bimanual_joint_names()) == "so101_6dof"


def test_detect_dof_profile_rejects_single_arm() -> None:
    assert detect_dof_profile(["joint_1", "joint_2"]) is None


def test_detect_dof_profile_rejects_mismatched_sides() -> None:
    assert detect_dof_profile(["left_joint_1", "right_joint_2"]) is None


def test_build_arm_sides_resolves_indices() -> None:
    names = _openarm_bimanual_joint_names()
    sides = build_arm_sides(names, "openarm7")
    assert sides is not None
    left, right = sides
    assert names[left.joint_index[next(iter(left.joint_index))]].startswith("left_")
    assert names[right.joint_index[next(iter(right.joint_index))]].startswith("right_")


def test_pose_retargeter_rejects_unsupported_robots() -> None:
    with pytest.raises(ValueError, match="dual-arm pose-teleop"):
        PoseRetargeter(["joint_1", "joint_2"], joint_limits_deg={})


def _landmark(x: float, y: float, z: float, visibility: float = 1.0) -> PoseLandmark:
    return (x, y, z, visibility)


def _neutral_pose() -> list[PoseLandmark]:
    """33 MediaPipe landmarks, arms hanging straight down, fully visible."""
    landmarks = [_landmark(0.0, 0.0, 0.0) for _ in range(33)]
    # Hips
    landmarks[23] = _landmark(-0.1, 0.9, 0.0)
    landmarks[24] = _landmark(0.1, 0.9, 0.0)
    # Shoulders
    landmarks[11] = _landmark(-0.2, 0.5, 0.0)
    landmarks[12] = _landmark(0.2, 0.5, 0.0)
    # Elbows straight below shoulders
    landmarks[13] = _landmark(-0.2, 0.8, 0.0)
    landmarks[14] = _landmark(0.2, 0.8, 0.0)
    # Wrists straight below elbows (fully extended arm)
    landmarks[15] = _landmark(-0.2, 1.1, 0.0)
    landmarks[16] = _landmark(0.2, 1.1, 0.0)
    return landmarks


def test_pose_retargeter_holds_previous_when_landmarks_not_visible() -> None:
    names = _openarm_bimanual_joint_names()
    retargeter = PoseRetargeter(names, joint_limits_deg={})
    landmarks = [_landmark(0, 0, 0, visibility=0.0) for _ in range(33)]
    previous = np.zeros(len(names), dtype=np.float32)
    previous[0] = 42.0

    action = retargeter.retarget(landmarks, previous)

    np.testing.assert_array_equal(action, previous)


def test_pose_retargeter_extends_elbow_flex_near_zero_when_arm_straight() -> None:
    names = _openarm_bimanual_joint_names()
    retargeter = PoseRetargeter(names, joint_limits_deg={})
    previous = np.zeros(len(names), dtype=np.float32)

    action = retargeter.retarget(_neutral_pose(), previous)

    elbow_index = names.index("left_joint_4")
    assert abs(action[elbow_index]) < 5.0
