import numpy as np
import pytest

from cenic_bench.tasks import CartPoleWalls


def zero_policy(obs):
    """Apply no force."""
    return np.array([0.0])


@pytest.fixture
def task():
    """A headless task with the zero policy."""
    return CartPoleWalls(zero_policy, visualize=False)


def set_state(task, x, theta, x_dot, theta_dot):
    """Set the cart-pole's positions and velocities in the task context."""
    plant_context = task.plant.GetMyMutableContextFromRoot(task.context)
    task.plant.SetPositions(plant_context, [x, theta])
    task.plant.SetVelocities(plant_context, [x_dot, theta_dot])


def test_properties(task):
    """Descriptions are non-empty and examples are correctly shaped."""
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
    """The cart is the only actuated joint, with its limit from the SDF."""
    assert task.plant.num_actuators() == 1
    actuator = task.plant.get_joint_actuator(
        task.plant.GetJointActuatorIndices()[0]
    )
    assert actuator.joint().name() == "x"
    assert actuator.effort_limit() == 10.0
    # The action description reports the limit from the SDF.
    assert "10.0 N" in task.action_description


def test_reset(task):
    """Reset stays within its ranges and is deterministic per seed."""
    plant_context = task.plant.GetMyContextFromRoot(task.context)

    for seed in range(100):
        task.reset(task.context, seed=seed)
        x, theta = task.plant.GetPositions(plant_context)
        x_dot, theta_dot = task.plant.GetVelocities(plant_context)
        assert theta == np.pi
        assert abs(x) <= 0.1
        assert abs(x_dot) <= 1.0
        assert 0.5 <= abs(theta_dot) <= 1.0

    # Same seed gives the same state, different seed gives a different one.
    task.reset(task.context, seed=0)
    q0 = task.plant.GetPositions(plant_context).copy()
    v0 = task.plant.GetVelocities(plant_context).copy()
    task.reset(task.context, seed=0)
    np.testing.assert_array_equal(task.plant.GetPositions(plant_context), q0)
    np.testing.assert_array_equal(task.plant.GetVelocities(plant_context), v0)
    task.reset(task.context, seed=1)
    assert task.plant.GetPositions(plant_context)[0] != q0[0]
    v1 = task.plant.GetVelocities(plant_context)
    assert v1[0] != v0[0]
    assert v1[1] != v0[1]


def test_no_seed_starts_at_success(task):
    """No initial state counts as success."""
    for seed in range(100):
        task.reset(task.context, seed=seed)
        assert not task.success(task.context)


@pytest.mark.parametrize(
    "x, theta, x_dot, theta_dot, expected",
    [
        (0.0, np.pi, 0.0, 0.0, True),  # upright at rest
        (0.0, -np.pi, 0.0, 0.0, True),  # upright, wrapped angle
        (  # near upright, wrapped, slowly moving
            0.0,
            3 * np.pi + 0.5 * CartPoleWalls.angle_tolerance,
            0.5 * CartPoleWalls.cart_velocity_tolerance,
            0.5 * CartPoleWalls.velocity_tolerance,
            True,
        ),
        (  # cart slightly off center
            0.5 * CartPoleWalls.cart_position_tolerance,
            np.pi,
            0.0,
            0.0,
            True,
        ),
        (  # cart slightly off center, left
            -0.5 * CartPoleWalls.cart_position_tolerance,
            np.pi,
            0.0,
            0.0,
            True,
        ),
        (  # cart too far from center
            2 * CartPoleWalls.cart_position_tolerance,
            np.pi,
            0.0,
            0.0,
            False,
        ),
        (5.0, np.pi, 0.0, 0.0, False),  # cart far from center
        (  # pole moving too fast
            0.0,
            np.pi,
            0.0,
            2 * CartPoleWalls.velocity_tolerance,
            False,
        ),
        (  # cart moving too fast
            0.0,
            np.pi,
            2 * CartPoleWalls.cart_velocity_tolerance,
            0.0,
            False,
        ),
        (  # too far from upright
            0.0,
            np.pi - 2 * CartPoleWalls.angle_tolerance,
            0.0,
            0.0,
            False,
        ),
        (0.0, 0.0, 0.0, 0.0, False),  # hanging down
    ],
)
def test_success(task, x, theta, x_dot, theta_dot, expected):
    """Success requires the pole upright and the cart centered, both still."""
    set_state(task, x, theta, x_dot, theta_dot)
    assert task.success(task.context) == expected


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_wall_pushes_bob_back(task, side):
    """A bob pressed into a wall is pushed back toward the center."""
    set_state(task, side * task.wall_distance, np.pi, 0.0, 0.0)
    plant_context = task.plant.GetMyContextFromRoot(task.context)
    cart_pole = task.plant.GetModelInstanceByName("cart_pole")
    tau = task.plant.get_generalized_contact_forces_output_port(cart_pole).Eval(
        plant_context
    )
    assert side * tau[0] < 0


def test_observation():
    """The policy sees [x, cos theta, sin theta, x_dot, theta_dot]."""
    observations = []

    def recording_policy(obs):
        """Record each observation and apply no force."""
        observations.append(obs)
        return np.array([0.0])

    task = CartPoleWalls(recording_policy, visualize=False)
    x, theta, x_dot, theta_dot = 0.2, 0.3, -0.4, -0.7
    set_state(task, x, theta, x_dot, theta_dot)

    # A periodic discrete update queries the policy.
    task.diagram.EvalUniquePeriodicDiscreteUpdate(task.context)
    np.testing.assert_allclose(
        observations[-1],
        [x, np.cos(theta), np.sin(theta), x_dot, theta_dot],
    )


def test_control_rate():
    """A failing episode queries the policy at the control rate."""
    num_calls = 0

    def counting_policy(obs):
        """Count policy queries and apply no force."""
        nonlocal num_calls
        num_calls += 1
        return np.array([0.0])

    # With no force seed 0 fails, so the episode runs for the full timeout.
    task = CartPoleWalls(counting_policy, visualize=False)
    assert not task.run_episode(seed=0)
    # One query per control period from t = 0, excluding the timeout.
    assert num_calls == task.timeout * task.control_rate


def test_zero_policy_fails(task):
    """Applying no force fails to balance the pole on seed 0."""
    assert not task.run_episode(seed=0)
