import numpy as np
import pytest
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity

from cenic_bench.tasks.trossen_cube_pick_place import (
    CUBE_REST_HEIGHT,
    CUBE_SPAWN_X_RANGE,
    CUBE_SPAWN_Y_RANGE,
    DROP_ZONE_XY,
    NUM_ARM_JOINTS,
    TrossenCubePickPlace,
)


def zero_policy(obs):
    return np.zeros(7)


@pytest.fixture
def task():
    return TrossenCubePickPlace(zero_policy, visualize=False)


def set_cube_state(task, xyz, velocity=(0.0, 0.0, 0.0)):
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    cube_body = task.plant.GetBodyByName("cube_link")
    task.plant.SetFreeBodyPose(plant_context, cube_body, RigidTransform(xyz))
    task.plant.SetFreeBodySpatialVelocity(
        plant_context,
        cube_body,
        SpatialVelocity(np.zeros(3), np.array(velocity)),
    )


def test_properties(task):
    assert task.timeout > 0
    assert isinstance(task.task_description, str)
    assert task.task_description
    assert isinstance(task.action_description, str)
    assert task.action_description
    assert isinstance(task.action_example, np.ndarray)
    assert task.action_example.size == NUM_ARM_JOINTS + 1
    assert isinstance(task.observation_description, str)
    assert task.observation_description
    assert isinstance(task.observation_example, np.ndarray)
    assert task.observation_example.size == 2 * NUM_ARM_JOINTS + 2 + 3


def test_reset(task):
    plant_context = task.plant.GetMyContextFromRoot(task.context)
    cube_body = task.plant.GetBodyByName("cube_link")

    task.reset(task.context, seed=0)

    # Cube is placed at rest, within the spawn range, resting on the table.
    xyz0 = cube_body.EvalPoseInWorld(plant_context).translation().copy()
    assert CUBE_SPAWN_X_RANGE[0] <= xyz0[0] <= CUBE_SPAWN_X_RANGE[1]
    assert CUBE_SPAWN_Y_RANGE[0] <= xyz0[1] <= CUBE_SPAWN_Y_RANGE[1]
    assert xyz0[2] == CUBE_REST_HEIGHT
    velocity0 = cube_body.EvalSpatialVelocityInWorld(plant_context)
    np.testing.assert_array_equal(velocity0.translational(), [0, 0, 0])
    np.testing.assert_array_equal(velocity0.rotational(), [0, 0, 0])

    # Both arms are at their home (all-zero) configuration.
    for prefix in ("follower_left", "follower_right"):
        for i in range(NUM_ARM_JOINTS):
            joint = task.plant.GetJointByName(f"{prefix}_joint_{i}")
            assert joint.get_angle(plant_context) == 0.0

    # Same seed gives the same cube position, different seed a different one.
    task.reset(task.context, seed=0)
    xyz0_again = cube_body.EvalPoseInWorld(plant_context).translation()
    np.testing.assert_array_equal(xyz0_again, xyz0)
    task.reset(task.context, seed=1)
    xyz1 = cube_body.EvalPoseInWorld(plant_context).translation()
    assert not np.allclose(xyz1[:2], xyz0[:2])


@pytest.mark.parametrize(
    "xy_offset, height_offset, speed, expected",
    [
        ((0.0, 0.0), 0.0, 0.0, True),  # exactly at the drop zone, at rest
        ((0.02, 0.0), 0.0, 0.0, True),  # within position tolerance
        ((0.05, 0.0), 0.0, 0.0, False),  # outside position tolerance
        ((0.0, 0.0), 0.05, 0.0, False),  # held above the table, not placed
        ((0.0, 0.0), 0.0, 0.2, False),  # at the drop zone but still moving
        ((0.2, 0.2), 0.0, 0.0, False),  # nowhere near the drop zone
    ],
)
def test_success(task, xy_offset, height_offset, speed, expected):
    xyz = [
        DROP_ZONE_XY[0] + xy_offset[0],
        DROP_ZONE_XY[1] + xy_offset[1],
        CUBE_REST_HEIGHT + height_offset,
    ]
    set_cube_state(task, xyz, velocity=(speed, 0.0, 0.0))
    assert task.success(task.context) == expected


def test_observation():
    observations = []

    def recording_policy(obs):
        observations.append(obs)
        return np.zeros(7)

    task = TrossenCubePickPlace(recording_policy, visualize=False)
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)

    # Distinct values per joint, so a position/velocity ordering mistake
    # would show up as a mismatch instead of accidentally canceling out.
    arm_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    arm_v = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for i, (q, v) in enumerate(zip(arm_q, arm_v, strict=True)):
        joint = task.plant.GetJointByName(f"follower_left_joint_{i}")
        joint.set_angle(plant_context, q)
        joint.set_angular_rate(plant_context, v)
    gripper_q, gripper_v = 0.02, 0.07
    gripper_joint = task.plant.GetJointByName(
        "follower_left_left_carriage_joint"
    )
    gripper_joint.set_translation(plant_context, gripper_q)
    gripper_joint.set_translation_rate(plant_context, gripper_v)
    cube_xyz = [0.05, 0.1, CUBE_REST_HEIGHT]
    set_cube_state(task, cube_xyz)

    # Trigger the controller's own periodic update directly: the full
    # diagram has two different periodic rates (this controller's, and the
    # state interpolator's), so EvalUniquePeriodicDiscreteUpdate on the
    # diagram itself would be ambiguous.
    controller_context = task.controller.GetMyMutableContextFromRoot(
        task.context
    )
    task.controller.EvalUniquePeriodicDiscreteUpdate(controller_context)
    np.testing.assert_allclose(
        observations[-1],
        arm_q + arm_v + [gripper_q, gripper_v] + cube_xyz,
    )


def test_control_rate():
    num_calls = 0

    def counting_policy(obs):
        nonlocal num_calls
        num_calls += 1
        return np.zeros(7)

    # Zero action never moves the cube, so the episode runs for the full
    # timeout regardless of seed.
    task = TrossenCubePickPlace(counting_policy, visualize=False)
    assert not task.run_episode(seed=0)
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    assert not task.run_episode(seed=0)
