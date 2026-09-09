"""Map MediaPipe body-pose landmarks onto a dual-arm follower's joint targets.

This is the PoC's direct joint-angle retargeter: no IK, no dynamics. Each arm's
shoulder/elbow/wrist landmarks are turned into a handful of joint angles by
simple vector geometry, then clamped to the follower's own joint limits. It
only supports robots that are physically two identical arms (e.g. Bimanual
OpenArm, Bimanual SO101) — the mapping is defined once per "DOF profile" and
mirrored for the left/right side.

MediaPipe's Pose Landmarker has no finger detail, so the gripper cannot be
driven from body pose alone (that needs a Hand Landmarker, left as follow-up
work — see the human-pose-teleop skill/docs). It is held at a fixed neutral
value here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

# MediaPipe Pose Landmarker indices (world landmarks, metric 3D, hips-centered).
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_ELBOW, _RIGHT_ELBOW = 13, 14
_LEFT_WRIST, _RIGHT_WRIST = 15, 16
_LEFT_HIP, _RIGHT_HIP = 23, 24


class JointRole(StrEnum):
    """The semantic role a follower joint plays, independent of its name."""

    SHOULDER_PAN = "shoulder_pan"
    SHOULDER_LIFT = "shoulder_lift"
    ELBOW_FLEX = "elbow_flex"
    WRIST_PITCH = "wrist_pitch"
    WRIST_ROLL = "wrist_roll"
    GRIPPER = "gripper"


# Joint suffixes (after stripping a "left_"/"right_" side prefix) that this PoC
# knows how to retarget, keyed by the DOF profile they belong to. A robot only
# qualifies for pose teleop when both arms expose exactly one of these full
# joint sets; ``_DRIVEN_ROLES`` below is the (possibly smaller) subset this
# retargeter actually drives from body pose.
_DOF_PROFILES: dict[str, set[str]] = {
    "openarm7": {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"},
    "so101_6dof": {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"},
}

_DRIVEN_ROLES: dict[str, dict[str, JointRole]] = {
    "openarm7": {
        "joint_1": JointRole.SHOULDER_PAN,
        "joint_2": JointRole.SHOULDER_LIFT,
        "joint_4": JointRole.ELBOW_FLEX,
        "joint_6": JointRole.WRIST_PITCH,
        "joint_7": JointRole.WRIST_ROLL,
        "gripper": JointRole.GRIPPER,
        # joint_3 and joint_5 (arm/wrist roll) are held at rest; body pose alone
        # cannot observe forearm twist reliably.
    },
    "so101_6dof": {
        "shoulder_pan": JointRole.SHOULDER_PAN,
        "shoulder_lift": JointRole.SHOULDER_LIFT,
        "elbow_flex": JointRole.ELBOW_FLEX,
        "wrist_flex": JointRole.WRIST_PITCH,
        "wrist_roll": JointRole.WRIST_ROLL,
        "gripper": JointRole.GRIPPER,
    },
}

_GRIPPER_NEUTRAL_DEG = 0.0
_REST_DEG = 0.0


@dataclass(frozen=True)
class ArmSide:
    """Which follower joint indices belong to one arm, and their role."""

    prefix: str  # "left_" or "right_"
    joint_index: dict[JointRole, int]


# One landmark as (x, y, z, visibility) — deliberately a plain tuple, not a
# dataclass shared with ``runtime.contract.PoseLandmark``: this module has no
# dependency on the wire contract, and a nominal-vs-structural type mismatch
# between two identically-shaped dataclasses is exactly the kind of thing a
# strict type checker (rightly) rejects.
PoseLandmark = tuple[float, float, float, float]
PoseLandmarks = Sequence[PoseLandmark]

_MIN_VISIBILITY = 0.5


def detect_dof_profile(joint_names: Sequence[str]) -> str | None:
    """Return the DOF profile name both arms share, or ``None`` if unsupported."""
    sides: dict[str, set[str]] = {"left_": set(), "right_": set()}
    for name in joint_names:
        for prefix, suffixes in sides.items():
            if name.startswith(prefix):
                suffixes.add(name[len(prefix) :])
    if not sides["left_"] or sides["left_"] != sides["right_"]:
        return None
    for profile_name, suffixes in _DOF_PROFILES.items():
        if suffixes == sides["left_"]:
            return profile_name
    return None


def build_arm_sides(joint_names: Sequence[str], profile_name: str) -> tuple[ArmSide, ArmSide] | None:
    """Resolve joint indices for both arms under one DOF profile, or ``None``."""
    roles = _DRIVEN_ROLES.get(profile_name)
    if roles is None:
        return None
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    sides: list[ArmSide] = []
    for prefix in ("left_", "right_"):
        joint_index: dict[JointRole, int] = {}
        for suffix, role in roles.items():
            index = index_by_name.get(f"{prefix}{suffix}")
            if index is None:
                return None
            joint_index[role] = index
        sides.append(ArmSide(prefix=prefix, joint_index=joint_index))
    return sides[0], sides[1]


class PoseRetargeter:
    """Turn one MediaPipe pose snapshot into a follower joint-angle vector."""

    def __init__(self, joint_names: Sequence[str], joint_limits_deg: Mapping[str, tuple[float, float]]) -> None:
        profile_name = detect_dof_profile(joint_names)
        if profile_name is None:
            raise ValueError(
                "Robot joints do not match a supported dual-arm pose-teleop profile "
                f"(known profiles: {sorted(_DOF_PROFILES)})"
            )
        sides = build_arm_sides(joint_names, profile_name)
        if sides is None:  # pragma: no cover - detect_dof_profile already checked this
            raise ValueError("Failed to resolve joint indices for the detected DOF profile")
        self._joint_names = list(joint_names)
        self._left, self._right = sides
        self._limits = joint_limits_deg
        self._profile_name = profile_name

    @property
    def profile_name(self) -> str:
        return self._profile_name

    def num_joints(self) -> int:
        return len(self._joint_names)

    def retarget(self, landmarks: PoseLandmarks, previous: np.ndarray) -> np.ndarray:
        """Return a full joint-angle vector (degrees) built from one pose snapshot.

        ``previous`` seeds joints this retargeter does not drive (rest DOF,
        gripper) so a partial mapping still returns a safe, stable vector.
        """
        action = np.array(previous, dtype=np.float32, copy=True)
        for side, shoulder_i, elbow_i, wrist_i, hip_i in (
            (self._left, _LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST, _LEFT_HIP),
            (self._right, _RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST, _RIGHT_HIP),
        ):
            self._retarget_side(action, side, landmarks, shoulder_i, elbow_i, wrist_i, hip_i)
        return action

    def _retarget_side(
        self,
        action: np.ndarray,
        side: ArmSide,
        landmarks: PoseLandmarks,
        shoulder_i: int,
        elbow_i: int,
        wrist_i: int,
        hip_i: int,
    ) -> None:
        points = [landmarks[i] for i in (shoulder_i, elbow_i, wrist_i, hip_i)]
        if any(point[3] < _MIN_VISIBILITY for point in points):
            return  # keep the previous (safe) value for this arm this tick
        shoulder, elbow, wrist, hip = (np.array(p[:3], dtype=np.float64) for p in points)

        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        torso_up = shoulder - hip

        def clamp(role: JointRole, value_deg: float) -> None:
            index = side.joint_index.get(role)
            if index is None:
                return
            name = self._joint_names[index]
            lower, upper = self._limits.get(name, (-180.0, 180.0))
            action[index] = float(np.clip(value_deg, lower, upper))

        # Shoulder pan: azimuth of the upper arm around the torso's vertical axis.
        pan_sign = -1.0 if side.prefix == "left_" else 1.0
        pan_deg = pan_sign * math.degrees(math.atan2(upper_arm[0], -upper_arm[1] + 1e-9))
        clamp(JointRole.SHOULDER_PAN, pan_deg)

        # Shoulder lift: elevation of the upper arm relative to the torso's own axis.
        lift_deg = 90.0 - math.degrees(_angle_between(upper_arm, torso_up))
        clamp(JointRole.SHOULDER_LIFT, lift_deg)

        # Elbow flex: angle between upper arm and forearm, 0 deg = fully extended.
        elbow_deg = math.degrees(_angle_between(upper_arm, forearm))
        clamp(JointRole.ELBOW_FLEX, elbow_deg)

        # No reliable forearm-twist or hand signal from body pose alone.
        clamp(JointRole.WRIST_PITCH, _REST_DEG)
        clamp(JointRole.WRIST_ROLL, _REST_DEG)
        clamp(JointRole.GRIPPER, _GRIPPER_NEUTRAL_DEG)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return math.acos(cosine)
