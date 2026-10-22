import numpy as np
import pytest

from cenic_bench.tasks import AcrobotSwingup

A = AcrobotSwingup.angle_tolerance
V = AcrobotSwingup.velocity_tolerance


def zero_policy(obs):
    """Apply no torque."""
    return np.array([0.0])


@pytest.fixture
def task():
    """A headless task driven by the zero policy."""
    return AcrobotSwingup(zero_policy, visualize=False)


def set_state(task, q1, q2, q1_dot, q2_dot):
    """Set the joint state of the task's plant through the named joints."""
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    shoulder = task.plant.GetJointByName("shoulder")
    elbow = task.plant.GetJointByName("elbow")
    shoulder.set_angle(plant_context, q1)
    elbow.set_angle(plant_context, q2)
    shoulder.set_angular_rate(plant_context, q1_dot)
    elbow.set_angular_rate(plant_context, q2_dot)


def test_properties(task):
    """Descriptions are non-empty and examples have the right sizes."""
    assert task.timeout > 0
    assert isinstance(task.task_description, str)
    assert task.task_description
    assert isinstance(task.action_description, str)
    assert task.action_description
    assert isinstance(task.action_example, np.ndarray)
    assert task.action_example.size == 1
    assert isinstance(task.observation_description, str)
    assert task.observation_description
    assert isinstance(task.observation_example, np.ndarray)
    assert task.observation_example.size == 6


def test_action_description_reports_limit(task):
    """The action description reports the elbow torque limit."""
    limit = task.plant.GetJointActuatorByName("elbow").effort_limit()
    assert f"{limit} Nm" in task.action_description


def test_reset(task):
    """Reset starts at rest in range and is deterministic per seed."""
    plant_context = task.plant.GetMyContextFromRoot(task.context)

    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    v0 = task.plant.GetVelocities(plant_context).copy()
    assert -np.pi <= q0[0] <= np.pi
    assert -np.pi <= q0[1] <= np.pi
    np.testing.assert_array_equal(v0, [0.0, 0.0])

    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    q1 = task.plant.GetPositions(plant_context)
    assert q1[0] != q0[0]
    assert q1[1] != q0[1]


@pytest.mark.parametrize(
    "q1, q2, q1_dot, q2_dot, expected",
    [
        (np.pi, 0.0, 0.0, 0.0, True),  # upright at rest
        (-np.pi, 0.0, 0.0, 0.0, True),  # upright, wrapped shoulder
        (3 * np.pi + 0.5 * A, 2 * np.pi - 0.5 * A, 0.5 * V, -0.5 * V, True),
        (np.pi + 2 * A, 0.0, 0.0, 0.0, False),  # shoulder too far
        (np.pi, 2 * A, 0.0, 0.0, False),  # elbow too bent
        (np.pi, 0.0, 2 * V, 0.0, False),  # shoulder moving too fast
        (np.pi, 0.0, 0.0, -2 * V, False),  # elbow moving too fast
        (0.0, 0.0, 0.0, 0.0, False),  # hanging down
        (np.pi, np.pi, 0.0, 0.0, False),  # folded
    ],
)
def test_success(task, q1, q2, q1_dot, q2_dot, expected):
    """Success requires both links near upright and nearly still."""
    set_state(task, q1, q2, q1_dot, q2_dot)
    assert task.success(task.context) == expected


def test_observation():
    """The policy sees cos/sin of both angles, then both velocities."""
    observations = []

    def recording_policy(obs):
        """Record each observation and apply no torque."""
        observations.append(obs)
        return np.array([0.0])

    task = AcrobotSwingup(recording_policy, visualize=False)
    q1, q2, q1_dot, q2_dot = 0.2, 0.3, -0.4, -0.7
    set_state(task, q1, q2, q1_dot, q2_dot)

    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        [np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), q1_dot, q2_dot],
    )


def test_control_rate():
    """The policy is queried at the control rate for the whole episode."""
    num_calls = 0

    def counting_policy(obs):
        """Count queries and apply no torque."""
        nonlocal num_calls
        num_calls += 1
        return np.array([0.0])

    task = AcrobotSwingup(counting_policy, visualize=False)
    assert not task.run_episode(seed=1)
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    """With no torque, seed 1 does not reach upright."""
    assert not task.run_episode(seed=1)
