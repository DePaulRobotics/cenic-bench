import numpy as np
import pytest
from pydrake.math import RollPitchYaw

from cenic_bench.tasks import G1Walking


def zero_policy(obs):
    return np.zeros(29)


@pytest.fixture(scope="module")
def task():
    return G1Walking(zero_policy, visualize=False)


def test_properties(task):
    assert task.timeout > task.duration
    assert isinstance(task.task_description, str)
    assert task.task_description
    assert isinstance(task.action_description, str)
    assert task.action_description
    assert isinstance(task.action_example, np.ndarray)
    assert task.action_example.size == 29
    assert isinstance(task.observation_description, str)
    assert task.observation_description
    assert isinstance(task.observation_example, np.ndarray)
    assert task.observation_example.size == 70
    assert len(task.joint_names) == 29
    assert np.all(task.action_scale > 0)


def test_state_layout(task):
    """The policy wrapper slices the state assuming this layout."""
    plant = task.plant
    pelvis = plant.GetBodyByName("pelvis")
    assert pelvis.floating_positions_start() == 0
    assert pelvis.floating_velocities_start_in_v() == 0
    for i, name in enumerate(task.joint_names):
        joint = plant.GetJointByName(name)
        assert joint.position_start() == 7 + i
        assert joint.velocity_start() == 6 + i


def test_reset(task):
    plant_context = task.plant.GetMyContextFromRoot(task.context)

    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    np.testing.assert_array_equal(
        q0[:7], [1, 0, 0, 0, 0, 0, task.pelvis_height]
    )
    assert np.all(np.abs(q0[7:] - task.home) <= 0.05)
    np.testing.assert_array_equal(task.plant.GetVelocities(plant_context), 0)

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    assert not np.array_equal(task.plant.GetPositions(plant_context), q0)


@pytest.mark.parametrize(
    "t, x, expected",
    [
        (10.0, 5.0, False),  # right speed, but too early
        (20.0, 10.0, True),  # exactly the target speed
        (20.0, 8.2, True),  # 0.41 m/s, just inside the tolerance
        (20.0, 11.8, True),  # 0.59 m/s, just inside the tolerance
        (20.0, 7.0, False),  # 0.35 m/s, too slow
        (20.0, 13.0, False),  # 0.65 m/s, too fast
        (20.0, -10.0, False),  # walking backward
    ],
)
def test_success(task, t, x, expected):
    task.reset(task.context)
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    q = task.plant.GetPositions(plant_context)
    q[4] = x
    task.plant.SetPositions(plant_context, q)
    task.context.SetTime(t)
    assert task.success(task.context) == expected


def test_observation_and_action():
    observations = []

    def recording_policy(obs):
        observations.append(obs)
        return np.ones(29)

    task = G1Walking(recording_policy, visualize=False)
    task.reset(task.context)
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)

    # Pitch the pelvis 90 degrees nose-down, so its x axis points along world
    # -z and its z axis along world +x.
    q = task.plant.GetPositions(plant_context)
    q[:4] = RollPitchYaw(0, np.pi / 2, 0).ToQuaternion().wxyz()
    q[7:] = task.home + 0.1
    task.plant.SetPositions(plant_context, q)
    v = np.zeros(35)
    v[:3] = [0.0, 0.0, 0.3]  # angular velocity in world
    v[3:6] = [1.0, 0.0, 0.0]  # linear velocity in world
    v[6:] = 0.2
    task.plant.SetVelocities(plant_context, v)

    # A periodic discrete update queries the policy.
    values = task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        np.concatenate(
            [
                [0.0, 0.0, 1.0],  # linear velocity in the pelvis frame
                [-0.3, 0.0, 0.0],  # angular velocity in the pelvis frame
                [1.0, 0.0, 0.0],  # gravity in the pelvis frame
                [task.target_velocity, 0.0, 0.0],
                np.full(29, 0.1),
                np.full(29, 0.2),
            ]
        ),
        atol=1e-12,
    )

    # The action becomes scaled joint position targets, with zero velocity.
    np.testing.assert_allclose(
        values.get_vector(0).get_value(),
        np.concatenate([task.home + task.action_scale, np.zeros(29)]),
    )


def test_zero_policy_fails():
    num_calls = 0

    def counting_policy(obs):
        nonlocal num_calls
        num_calls += 1
        return np.zeros(29)

    # Holding the HOME pose doesn't walk, so the episode runs to the timeout.
    task = G1Walking(counting_policy, visualize=False)
    assert not task.run_episode(seed=0)
    assert num_calls == task.timeout * task.control_rate
