import numpy as np
import pytest
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialVelocity

from cenic_bench.tasks import PushT


def zero_policy(obs):
    """Apply no force."""
    return np.zeros(2)


@pytest.fixture
def task():
    """A headless task with the zero policy."""
    return PushT(zero_policy, visualize=False)


def t_height(task):
    """The height of the T's origin in the task context."""
    plant_context = task.plant.GetMyContextFromRoot(task.context)
    body = task.plant.GetBodyByName("t", task.t_block)
    return task.plant.GetFreeBodyPose(plant_context, body).translation()[2]


def set_state(task, t, pusher):
    """Set the T's [x, y, yaw, vx, vy, yaw_rate] and the pusher's state."""
    x, y, yaw, vx, vy, yaw_rate = t
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    body = task.plant.GetBodyByName("t", task.t_block)
    X_WT = RigidTransform(
        RotationMatrix.MakeZRotation(yaw), [x, y, t_height(task)]
    )
    task.plant.SetFreeBodyPose(plant_context, body, X_WT)
    task.plant.SetFreeBodySpatialVelocity(
        plant_context,
        body,
        SpatialVelocity([0.0, 0.0, yaw_rate], [vx, vy, 0.0]),
    )
    task.plant.SetPositionsAndVelocities(plant_context, task.pusher, pusher)


def test_properties(task):
    """Descriptions are non-empty and examples are correctly shaped."""
    assert task.timeout > 0
    assert isinstance(task.task_description, str)
    assert task.task_description
    assert isinstance(task.action_description, str)
    assert task.action_description
    assert isinstance(task.action_example, np.ndarray)
    assert task.action_example.size == 2
    assert isinstance(task.observation_description, str)
    assert task.observation_description
    assert isinstance(task.observation_example, np.ndarray)
    assert task.observation_example.size == 11


def test_reset(task):
    """Reset stays within its ranges and is deterministic per seed."""
    plant_context = task.plant.GetMyContextFromRoot(task.context)
    body = task.plant.GetBodyByName("t", task.t_block)
    default_height = t_height(task)

    for seed in range(100):
        task.reset(task.context, seed=seed)
        x, y, z = task.plant.GetFreeBodyPose(plant_context, body).translation()
        assert 0.1 <= np.hypot(x, y) <= 0.2
        assert z == default_height
        assert not task.plant.GetVelocities(plant_context).any()
        np.testing.assert_array_equal(
            task.plant.GetPositions(plant_context, task.pusher),
            task.pusher_start,
        )

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    assert not np.array_equal(task.plant.GetPositions(plant_context), q0)


def test_no_seed_starts_at_success(task):
    """No initial state counts as success."""
    for seed in range(100):
        task.reset(task.context, seed=seed)
        assert not task.success(task.context)


@pytest.mark.parametrize(
    "t, pusher_velocity, expected",
    [
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], True),  # at goal, at rest
        (  # every quantity just within its tolerance
            [
                0.5 * PushT.position_tolerance,
                0.0,
                0.5 * PushT.yaw_tolerance,
                0.5 * PushT.velocity_tolerance,
                0.0,
                0.5 * PushT.yaw_rate_tolerance,
            ],
            [0.5 * PushT.pusher_velocity_tolerance, 0.0],
            True,
        ),
        (  # T too far from the origin
            [0.0, 2 * PushT.position_tolerance, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0],
            False,
        ),
        (  # T yawed too far
            [0.0, 0.0, -2 * PushT.yaw_tolerance, 0.0, 0.0, 0.0],
            [0.0, 0.0],
            False,
        ),
        (  # T sliding too fast
            [0.0, 0.0, 0.0, 0.0, 2 * PushT.velocity_tolerance, 0.0],
            [0.0, 0.0],
            False,
        ),
        (  # T turning too fast
            [0.0, 0.0, 0.0, 0.0, 0.0, 2 * PushT.yaw_rate_tolerance],
            [0.0, 0.0],
            False,
        ),
        (  # pusher moving too fast
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2 * PushT.pusher_velocity_tolerance],
            False,
        ),
    ],
)
def test_success(task, t, pusher_velocity, expected):
    """Success requires the T at the target, T and pusher nearly still."""
    set_state(task, t, [*task.pusher_start, *pusher_velocity])
    assert task.success(task.context) == expected


def test_observation():
    """The policy sees the pusher state, then the T's planar state."""
    observations = []

    def recording_policy(obs):
        """Record each observation and apply no force."""
        observations.append(obs)
        return np.zeros(2)

    task = PushT(recording_policy, visualize=False)
    x, y, yaw, vx, vy, yaw_rate = 0.1, -0.05, 2.5, 0.2, -0.1, 0.4
    pusher = [0.05, -0.3, 0.1, -0.2]
    set_state(task, [x, y, yaw, vx, vy, yaw_rate], pusher)

    # A periodic discrete update queries the policy.
    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        [*pusher, x, y, np.cos(yaw), np.sin(yaw), vx, vy, yaw_rate],
        atol=1e-12,
    )


def test_control_rate():
    """A failing episode queries the policy at the control rate."""
    num_calls = 0

    def counting_policy(obs):
        """Count policy queries and apply no force."""
        nonlocal num_calls
        num_calls += 1
        return np.zeros(2)

    # With no force seed 0 fails, so the episode runs for the full timeout.
    task = PushT(counting_policy, visualize=False)
    assert not task.run_episode(seed=0)
    # One query per control period from t = 0, excluding the timeout.
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    """Applying no force fails on seed 0, and the T stays on the floor."""
    default_height = t_height(task)
    assert not task.run_episode(seed=0)
    assert t_height(task) == pytest.approx(default_height, abs=1e-3)
