import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization

from cenic_bench.parsing import make_parser
from cenic_bench.tasks.base import Task


class AcrobotPolicy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram.

    The policy is queried at a fixed rate, and the resulting torque is held
    constant (zero-order hold) between queries.
    """

    def __init__(self, policy, control_rate: float):
        super().__init__()
        self.policy = policy
        self.state_port = self.DeclareVectorInputPort("state", 4)
        torque = self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("torque", torque)

    def Update(self, context, discrete_state):
        q1, q2, q1_dot, q2_dot = self.state_port.Eval(context)
        obs = np.array(
            [np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), q1_dot, q2_dot]
        )
        u = self.policy(obs)
        discrete_state.set_value(u)


class AcrobotSwingup(Task):
    """Swing both links of an acrobot upright and hold them still.

    The acrobot is a two-link pendulum hanging from a passive shoulder. Only
    the elbow, between the two links, is actuated.

    Observations:
        - The cosine and sine of the shoulder angle
        - The cosine and sine of the elbow angle
        - The angular velocity of the shoulder
        - The angular velocity of the elbow

    Actions:
        - The torque applied at the elbow, held constant between policy
          queries. Its magnitude is limited by the effort limit of the elbow
          joint, defined in acrobot.sdf.

    Success conditions:
        - The upper link is close to upright
        - The elbow is close to straight
        - Both links are nearly still
    """

    angle_tolerance = 0.1  # rad, distance from upright to count as success
    velocity_tolerance = 0.5  # rad/s, max joint speed at upright
    control_rate = 50.0  # Hz, how often the policy is queried

    def __init__(self, policy, visualize: bool = True):
        """Initialize the acrobot swing-up task.

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
        make_parser(self.plant).AddModelsFromUrl(
            "package://cenic_bench/models/acrobot/acrobot.sdf"
        )
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(AcrobotPolicy(policy, self.control_rate))
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Start at rest with random shoulder and elbow angles."""
        rng = np.random.default_rng(seed)
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        q1 = rng.uniform(-np.pi, np.pi)
        q2 = rng.uniform(-np.pi, np.pi)
        self.plant.SetPositions(plant_context, [q1, q2])
        self.plant.SetVelocities(plant_context, [0.0, 0.0])

    def success(self, context) -> bool:
        """Both links near upright and nearly at rest."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        q1, q2 = self.plant.GetPositions(plant_context)
        q1_dot, q2_dot = self.plant.GetVelocities(plant_context)
        # Wrapped angular distances from upright (q1 = pi, q2 = 0).
        shoulder_error = np.abs(np.mod(q1, 2 * np.pi) - np.pi)
        elbow_error = np.abs(np.mod(q2 + np.pi, 2 * np.pi) - np.pi)
        return (
            shoulder_error < self.angle_tolerance
            and elbow_error < self.angle_tolerance
            and abs(q1_dot) < self.velocity_tolerance
            and abs(q2_dot) < self.velocity_tolerance
        )

    @property
    def timeout(self):
        return 173.0

    @property
    def task_description(self):
        return (
            "Swing both links of an acrobot upright by applying torque at the "
            "elbow. Success is reaching within "
            f"{self.angle_tolerance} rad of upright at the shoulder and of "
            "straight at the elbow, with both joint speeds below "
            f"{self.velocity_tolerance} rad/s."
        )

    @property
    def action_description(self):
        max_torque = self.plant.GetJointActuatorByName("elbow").effort_limit()
        return (
            "The torque (Nm) to apply at the elbow, limited to "
            f"+/- {max_torque} Nm and held constant between policy queries at "
            f"{self.control_rate} Hz."
        )

    @property
    def action_example(self):
        return np.array([0.0])

    @property
    def observation_description(self):
        return (
            "Cosine and sine of the shoulder angle (zero when hanging down), "
            "cosine and sine of the elbow angle (zero when straight), "
            "shoulder angular velocity, and elbow angular velocity."
        )

    @property
    def observation_example(self):
        return np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
