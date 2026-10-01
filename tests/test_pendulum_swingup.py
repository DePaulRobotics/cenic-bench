import numpy as np
import pytest

from cenic_bench.tasks import PendulumSwingup


def zero_policy(obs):
    return np.array([0.0])


@pytest.fixture
def task():
    return PendulumSwingup(zero_policy, visualize=False)


def set_state(task, theta, theta_dot):
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    task.plant.SetPositions(plant_context, [theta])
    task.plant.SetVelocities(plant_context, [theta_dot])


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
    assert task.observation_example.size == 3


def test_reset(task):
    plant_context = task.plant.GetMyContextFromRoot(task.context)

    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    v0 = task.plant.GetVelocities(plant_context).copy()
    assert -np.pi <= q0[0] <= np.pi
    np.testing.assert_array_equal(v0, [0.0])

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    task.reset(task.context, seed=1)
    assert task.plant.GetPositions(plant_context)[0] != q0[0]


@pytest.mark.parametrize(
    "theta, theta_dot, expected",
    [
        (np.pi, 0.0, True),  # upright at rest
        (-np.pi, 0.0, True),  # upright, wrapped angle
        (3 * np.pi + 0.05, 0.1, True),  # near upright, wrapped angle
        (np.pi, 1.0, False),  # upright but moving too fast
        (np.pi - 0.2, 0.0, False),  # too far from upright
        (0.0, 0.0, False),  # hanging down
    ],
)
def test_success(task, theta, theta_dot, expected):
    set_state(task, theta, theta_dot)
    assert task.success(task.context) == expected


def test_observation():
    observations = []

    def recording_policy(obs):
        observations.append(obs)
        return np.array([0.0])

    task = PendulumSwingup(recording_policy, visualize=False)
    theta, theta_dot = 0.3, -0.7
    set_state(task, theta, theta_dot)

    # A periodic discrete update queries the policy.
    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1], [np.cos(theta), np.sin(theta), theta_dot]
    )


def test_control_rate():
    num_calls = 0

    def counting_policy(obs):
        nonlocal num_calls
        num_calls += 1
        return np.array([0.0])

    # Seed 1 fails, so the episode runs for the full timeout.
    task = PendulumSwingup(counting_policy, visualize=False)
    assert not task.run_episode(seed=1)
    # Queries at t = 0, 0.05, ..., up to but not including the timeout.
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    # Seed 1 starts far from upright, so zero torque never succeeds.
    assert not task.run_episode(seed=1)
