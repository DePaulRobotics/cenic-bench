import numpy as np
import pytest

from cenic_bench.tasks import CartPoleSwingup


def zero_policy(obs):
    return np.array([0.0])


@pytest.fixture
def task():
    return CartPoleSwingup(zero_policy, visualize=False)


def set_state(task, x, theta, x_dot, theta_dot):
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    task.plant.SetPositions(plant_context, [x, theta])
    task.plant.SetVelocities(plant_context, [x_dot, theta_dot])


def test_properties(task):
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
    assert task.observation_example.size == 5


def test_only_cart_is_actuated(task):
    assert task.plant.num_actuators() == 1
    actuator = task.plant.get_joint_actuator(
        task.plant.GetJointActuatorIndices()[0]
    )
    assert actuator.joint().name() == "x"
    assert actuator.effort_limit() == 10.0


def test_reset(task):
    plant_context = task.plant.GetMyContextFromRoot(task.context)

    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    v0 = task.plant.GetVelocities(plant_context).copy()
    assert -0.5 <= q0[0] <= 0.5
    assert -np.pi <= q0[1] <= np.pi
    np.testing.assert_array_equal(v0, [0.0, 0.0])

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    q1 = task.plant.GetPositions(plant_context)
    assert q1[0] != q0[0]
    assert q1[1] != q0[1]


@pytest.mark.parametrize(
    "x, theta, x_dot, theta_dot, expected",
    [
        (0.0, np.pi, 0.0, 0.0, True),  # upright at rest
        (0.0, -np.pi, 0.0, 0.0, True),  # upright, wrapped angle
        (0.0, 3 * np.pi + 0.05, 0.1, 0.1, True),  # near upright, wrapped
        (5.0, np.pi, 0.0, 0.0, True),  # cart position doesn't matter
        (0.0, np.pi, 0.0, 1.0, False),  # pole moving too fast
        (0.0, np.pi, 1.0, 0.0, False),  # cart moving too fast
        (0.0, np.pi - 0.2, 0.0, 0.0, False),  # too far from upright
        (0.0, 0.0, 0.0, 0.0, False),  # hanging down
    ],
)
def test_success(task, x, theta, x_dot, theta_dot, expected):
    set_state(task, x, theta, x_dot, theta_dot)
    assert task.success(task.context) == expected


def test_observation():
    observations = []

    def recording_policy(obs):
        observations.append(obs)
        return np.array([0.0])

    task = CartPoleSwingup(recording_policy, visualize=False)
    x, theta, x_dot, theta_dot = 0.2, 0.3, -0.4, -0.7
    set_state(task, x, theta, x_dot, theta_dot)

    # A periodic discrete update queries the policy.
    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        [x, np.cos(theta), np.sin(theta), x_dot, theta_dot],
    )


def test_control_rate():
    num_calls = 0

    def counting_policy(obs):
        nonlocal num_calls
        num_calls += 1
        return np.array([0.0])

    # Seed 1 fails, so the episode runs for the full timeout.
    task = CartPoleSwingup(counting_policy, visualize=False)
    assert not task.run_episode(seed=1)
    # Queries at t = 0, 0.02, ..., up to but not including the timeout.
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    # Seed 1 starts far from upright, so zero force never succeeds.
    assert not task.run_episode(seed=1)
