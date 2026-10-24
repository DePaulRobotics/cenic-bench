import numpy as np
import pytest
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialVelocity

from cenic_bench.tasks import DiffDriveParking


def zero_policy(obs):
    """Apply no torque."""
    return np.array([0.0, 0.0])


@pytest.fixture
def task():
    """A headless task with the zero policy."""
    return DiffDriveParking(zero_policy, visualize=False)


def set_state(task, x, y, heading, vx=0.0, vy=0.0, heading_rate=0.0):
    """Set the chassis pose and velocity, keeping its default height."""
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    chassis = task.plant.GetBodyByName("chassis")
    z = task.plant.GetFreeBodyPose(plant_context, chassis).translation()[2]
    X_WC = RigidTransform(RotationMatrix.MakeZRotation(heading), [x, y, z])
    task.plant.SetFreeBodyPose(plant_context, chassis, X_WC)
    V_WC = SpatialVelocity(w=[0.0, 0.0, heading_rate], v=[vx, vy, 0.0])
    task.plant.SetFreeBodySpatialVelocity(plant_context, chassis, V_WC)


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
    assert task.observation_example.size == 7


def test_reset(task):
    """Reset stays within its ranges, at rest, deterministic per seed."""
    plant_context = task.plant.GetMyContextFromRoot(task.context)
    chassis = task.plant.GetBodyByName("chassis")
    X_WC = task.plant.GetFreeBodyPose(plant_context, chassis)
    default_z = X_WC.translation()[2]

    for seed in range(100):
        task.reset(task.context, seed=seed)
        X_WC = task.plant.GetFreeBodyPose(plant_context, chassis)
        x, y, z = X_WC.translation()
        assert abs(x) <= 1.0
        assert abs(y) <= 1.0
        assert z == default_z
        # Wheel angles are the last two positions.
        np.testing.assert_array_equal(
            task.plant.GetPositions(plant_context)[7:], [0.0, 0.0]
        )
        assert not np.any(task.plant.GetVelocities(plant_context))

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    q1 = task.plant.GetPositions(plant_context)
    assert q1[4] != q0[4]
    assert q1[5] != q0[5]
    assert q1[3] != q0[3]


def test_no_seed_starts_at_success(task):
    """No initial state counts as success."""
    for seed in range(100):
        task.reset(task.context, seed=seed)
        assert not task.success(task.context)


@pytest.mark.parametrize(
    "x, y, heading, vx, vy, heading_rate, expected",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True),  # parked at the center
        (  # slightly off center
            0.5 * DiffDriveParking.position_tolerance,
            0.5 * DiffDriveParking.position_tolerance,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
        ),
        (  # heading slightly right of +x
            0.0,
            0.0,
            -0.5 * DiffDriveParking.heading_tolerance,
            0.0,
            0.0,
            0.0,
            True,
        ),
        (  # moving and turning slowly
            0.0,
            0.0,
            0.0,
            0.5 * DiffDriveParking.speed_tolerance,
            0.0,
            0.5 * DiffDriveParking.heading_rate_tolerance,
            True,
        ),
        (  # too far from center
            2 * DiffDriveParking.position_tolerance,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        ),
        (  # heading too far from +x
            0.0,
            0.0,
            2 * DiffDriveParking.heading_tolerance,
            0.0,
            0.0,
            0.0,
            False,
        ),
        (0.0, 0.0, np.pi, 0.0, 0.0, 0.0, False),  # facing -x
        (  # moving too fast sideways
            0.0,
            0.0,
            0.0,
            0.0,
            2 * DiffDriveParking.speed_tolerance,
            0.0,
            False,
        ),
        (  # turning too fast
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2 * DiffDriveParking.heading_rate_tolerance,
            False,
        ),
    ],
)
def test_success(task, x, y, heading, vx, vy, heading_rate, expected):
    """Success requires the robot centered, facing +x, and still."""
    set_state(task, x, y, heading, vx, vy, heading_rate)
    assert task.success(task.context) == expected


def test_observation():
    """The policy sees [x, y, cos h, sin h, vx, vy, heading rate]."""
    observations = []

    def recording_policy(obs):
        """Record each observation and apply no torque."""
        observations.append(obs)
        return np.array([0.0, 0.0])

    task = DiffDriveParking(recording_policy, visualize=False)
    x, y, heading, vx, vy, heading_rate = 0.2, -0.3, 2.5, -0.4, 0.6, -0.7
    set_state(task, x, y, heading, vx, vy, heading_rate)

    # A periodic discrete update queries the policy.
    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        [x, y, np.cos(heading), np.sin(heading), vx, vy, heading_rate],
        atol=1e-12,
    )


def test_control_rate():
    """A failing episode queries the policy at the control rate."""
    num_calls = 0

    def counting_policy(obs):
        """Count policy queries and apply no torque."""
        nonlocal num_calls
        num_calls += 1
        return np.array([0.0, 0.0])

    # With no torque seed 0 fails, so the episode runs for the full timeout.
    task = DiffDriveParking(counting_policy, visualize=False)
    assert not task.run_episode(seed=0)
    # One query per control period from t = 0, excluding the timeout.
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    """Applying no torque fails to park on seed 0."""
    assert not task.run_episode(seed=0)
