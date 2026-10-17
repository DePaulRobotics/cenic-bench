import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization

from cenic_bench.parsing import make_parser
from cenic_bench.tasks.base import Task


class CartPoleWallsPolicy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram.

    The policy is queried at a fixed rate, and the resulting force is held
    constant (zero-order hold) between queries.
    """

    def __init__(self, policy, control_rate: float):
        super().__init__()
        self.policy = policy
        self.state_port = self.DeclareVectorInputPort("state", 4)
        force = self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("force", force)

    def Update(self, context, discrete_state):
        x, theta, x_dot, theta_dot = self.state_port.Eval(context)
        obs = np.array([x, np.cos(theta), np.sin(theta), x_dot, theta_dot])
        u = self.policy(obs)
        discrete_state.set_value(u)


class CartPoleWalls(Task):
    """Balance a pole upright between two soft walls and center the cart.

    The cart slides along an endless horizontal rail. Only the cart is
    actuated; the pole swings freely. Two soft, frictionless walls stand on
    either side of the upright pole. Only the pole's tip touches them, and
    they push it back like springs. The pole starts upright and moving.

    Observations:
        - The cart position along the rail
        - The cosine of the pole's angle
        - The sine of the pole's angle
        - The cart velocity
        - The angular velocity of the pole

    Actions:
        - The horizontal force applied to the cart, held constant between
          policy queries. Its magnitude is limited by the effort limit of the
          cart's joint, defined in cart_pole.sdf.

    Success conditions:
        - The pole is close to upright
        - The pole is nearly still
        - The cart is close to the center of the rail
        - The cart is nearly still
    """

    angle_tolerance = 0.05  # rad, distance from upright to count as success
    velocity_tolerance = 0.2  # rad/s, max pole speed at upright
    cart_velocity_tolerance = 0.2  # m/s, max cart speed at upright
    cart_position_tolerance = 0.05  # m, max distance of cart from center
    control_rate = 50.0  # Hz, how often the policy is queried

    def __init__(self, policy, visualize: bool = True):
        """Initialize the cart-pole with walls task.

        Args:
            policy: A callable that takes an observation and returns an action.
            visualize: Whether to visualize the simulation. If true, connects
                       to meshcat and runs in ~realtime. Otherwise a headless
                       simulation runs as fast as possible.
        """
        self.meshcat = StartMeshcat() if visualize else None
        super().__init__(policy, realtime=visualize, accuracy=1e-3)

    def create_scene(self, policy) -> Diagram:
        builder = DiagramBuilder()

        # Note that CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = make_parser(self.plant)
        parser.AddModelsFromUrl(
            "package://cenic_bench/models/cart_pole/cart_pole.sdf"
        )
        parser.AddModelsFromUrl(
            "package://cenic_bench/models/cart_pole/soft_walls.sdf"
        )
        self.plant.set_contact_model(ContactModel.kPoint)
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(
            CartPoleWallsPolicy(policy, self.control_rate)
        )
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Start upright with the cart near center, cart and pole moving."""
        rng = np.random.default_rng(seed)
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        x = rng.uniform(-0.1, 0.1)
        x_dot = rng.uniform(-1.0, 1.0)
        theta_dot = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
        self.plant.SetPositions(plant_context, [x, np.pi])
        self.plant.SetVelocities(plant_context, [x_dot, theta_dot])

    def success(self, context) -> bool:
        """Pole near upright and centered cart, both nearly at rest."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        x, theta = self.plant.GetPositions(plant_context)
        x_dot, theta_dot = self.plant.GetVelocities(plant_context)
        # Wrapped angular distance from upright (theta = pi).
        error = np.abs(np.mod(theta, 2 * np.pi) - np.pi)
        return (
            error < self.angle_tolerance
            and abs(theta_dot) < self.velocity_tolerance
            and abs(x_dot) < self.cart_velocity_tolerance
            and abs(x) < self.cart_position_tolerance
        )

    @property
    def timeout(self):
        return 5.0

    @property
    def task_description(self):
        return (
            "Keep a pole balanced upright by pushing the cart it is hinged "
            "to, and bring the cart back to the center. Two soft walls on "
            "either side push back on the pole's tip. Success is reaching "
            f"within {self.angle_tolerance} rad of upright with pole angular "
            f"speed below {self.velocity_tolerance} rad/s, with the cart "
            f"within {self.cart_position_tolerance} m of the center and cart "
            f"speed below {self.cart_velocity_tolerance} m/s."
        )

    @property
    def action_description(self):
        max_force = self.plant.GetJointActuatorByName("x").effort_limit()
        return (
            "The horizontal force (N) to apply to the cart, limited to "
            f"+/- {max_force} N and held constant between policy queries at "
            f"{self.control_rate} Hz."
        )

    @property
    def action_example(self):
        return np.array([0.0])

    @property
    def observation_description(self):
        return (
            "Cart position (zero at the center), cosine and sine of the "
            "pole's angle (zero when hanging down, pi when upright), cart "
            "velocity, and pole angular velocity."
        )

    @property
    def observation_example(self):
        return np.array([0.0, -1.0, 0.0, 0.0, 0.0])
