import numpy as np
from pydrake.geometry import (
    AddRigidHydroelasticProperties,
    Box,
    ProximityProperties,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization

from cenic_bench.parsing import make_parser
from cenic_bench.tasks.base import Task


class PushTPolicy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram.

    The policy is queried at a fixed rate, and the resulting forces are held
    constant (zero-order hold) between queries.
    """

    def __init__(self, policy, control_rate: float):
        """Declare the T and pusher state inputs and the force output."""
        super().__init__()
        self.policy = policy
        self.t_state_port = self.DeclareVectorInputPort("t_state", 13)
        self.pusher_state_port = self.DeclareVectorInputPort("pusher_state", 4)
        force = self.DeclareDiscreteState(2)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("force", force)

    def Update(self, context, discrete_state):
        """Query the policy with the current observation and store forces."""
        qw, qx, qy, qz, x, y, _, _, _, wz, vx, vy, _ = self.t_state_port.Eval(
            context
        )
        pusher = self.pusher_state_port.Eval(context)
        yaw = np.arctan2(
            2 * (qw * qz + qx * qy),
            qw**2 + qx**2 - qy**2 - qz**2,
        )
        obs = np.array([*pusher, x, y, np.cos(yaw), np.sin(yaw), vx, vy, wz])
        u = self.policy(obs)
        discrete_state.set_value(u)


class PushT(Task):
    """Push a T-shaped block into a target pose on a floor.

    A ball pusher moves in the horizontal plane, driven by forces along the
    world x and y axes. The T block slides freely on the floor and only moves
    when the pusher touches it. The T starts at rest near the origin with a
    random heading, and the pusher starts at rest behind it.

    Observations:
        - The pusher position in x and y
        - The pusher velocity in x and y
        - The T position in x and y
        - The cosine of the T's yaw angle
        - The sine of the T's yaw angle
        - The T velocity in x and y
        - The T yaw rate

    Actions:
        - The forces applied to the pusher along the world x and y axes, held
          constant between policy queries. Their magnitudes are limited by
          the effort limits of the pusher's joints, defined in pusher.sdf.

    Success conditions:
        - The T is close to the origin
        - The T's yaw is close to zero
        - The T is nearly still, both sliding and turning
        - The pusher is nearly still
    """

    position_tolerance = 0.05  # m, max distance of the T from the origin
    yaw_tolerance = 0.1  # rad, max T yaw from zero
    velocity_tolerance = 0.05  # m/s, max T speed in the plane
    yaw_rate_tolerance = 0.5  # rad/s, max T yaw rate
    pusher_velocity_tolerance = 0.05  # m/s, max pusher speed
    control_rate = 50.0  # Hz, how often the policy is queried
    floor_size = (2.0, 2.0, 0.1)  # m, floor box's x, y, z size
    pusher_start = (0.0, -0.45)  # m, pusher's initial x, y

    def __init__(self, policy, visualize: bool = True):
        """Initialize the Push-T task.

        Args:
            policy: A callable that takes an observation and returns an action.
            visualize: Whether to visualize the simulation. If true, connects
                       to meshcat and runs in ~realtime. Otherwise a headless
                       simulation runs as fast as possible.
        """
        self.meshcat = StartMeshcat() if visualize else None
        super().__init__(policy, realtime=visualize, accuracy=1e-3)

    def create_scene(self, policy) -> Diagram:
        """Build the T, pusher and floor; the policy drives the pusher."""
        builder = DiagramBuilder()

        # Note that CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = make_parser(self.plant)
        (self.t_block,) = parser.AddModelsFromUrl(
            "package://cenic_bench/models/t_block/t_block.sdf"
        )
        (self.pusher,) = parser.AddModelsFromUrl(
            "package://cenic_bench/models/pusher/pusher.sdf"
        )
        X_WF = RigidTransform([0.0, 0.0, -self.floor_size[2] / 2])
        box = Box(*self.floor_size)
        props = ProximityProperties()
        AddRigidHydroelasticProperties(properties=props)
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_WF, box, "floor", props
        )
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_WF, box, "floor", [0.6, 0.6, 0.6, 1.0]
        )
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(PushTPolicy(policy, self.control_rate))
        builder.Connect(
            self.plant.get_state_output_port(self.t_block),
            controller.GetInputPort("t_state"),
        )
        builder.Connect(
            self.plant.get_state_output_port(self.pusher),
            controller.GetInputPort("pusher_state"),
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Place the T at rest near the origin, the pusher at rest behind."""
        rng = np.random.default_rng(seed)
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        body = self.plant.GetBodyByName("t", self.t_block)
        z = self.plant.GetFreeBodyPose(plant_context, body).translation()[2]
        distance = rng.uniform(0.1, 0.2)
        direction = rng.uniform(-np.pi, np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        X_WT = RigidTransform(
            RotationMatrix.MakeZRotation(yaw),
            [distance * np.cos(direction), distance * np.sin(direction), z],
        )
        self.plant.SetFreeBodyPose(plant_context, body, X_WT)
        self.plant.SetPositions(plant_context, self.pusher, self.pusher_start)

    def success(self, context) -> bool:
        """T near the origin with zero yaw, T and pusher nearly at rest."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        qw, qx, qy, qz, x, y, _ = self.plant.GetPositions(
            plant_context, self.t_block
        )
        _, _, wz, vx, vy, _ = self.plant.GetVelocities(
            plant_context, self.t_block
        )
        pusher_v = self.plant.GetVelocities(plant_context, self.pusher)
        yaw = np.arctan2(
            2 * (qw * qz + qx * qy),
            qw**2 + qx**2 - qy**2 - qz**2,
        )
        return (
            np.hypot(x, y) < self.position_tolerance
            and abs(yaw) < self.yaw_tolerance
            and np.hypot(vx, vy) < self.velocity_tolerance
            and abs(wz) < self.yaw_rate_tolerance
            and np.hypot(*pusher_v) < self.pusher_velocity_tolerance
        )

    @property
    def timeout(self):
        """Episode length in seconds."""
        return 30.0

    @property
    def task_description(self):
        """What the policy must achieve."""
        return (
            "Use a ball pusher to slide a T-shaped block on a floor so it "
            "rests at the origin with zero yaw. Success is the T within "
            f"{self.position_tolerance} m of the origin and within "
            f"{self.yaw_tolerance} rad of zero yaw, sliding slower than "
            f"{self.velocity_tolerance} m/s and turning slower than "
            f"{self.yaw_rate_tolerance} rad/s, with the pusher slower than "
            f"{self.pusher_velocity_tolerance} m/s."
        )

    @property
    def action_description(self):
        """The pusher forces and their limits."""
        max_fx = self.plant.GetJointActuatorByName(
            "x", self.pusher
        ).effort_limit()
        max_fy = self.plant.GetJointActuatorByName(
            "y", self.pusher
        ).effort_limit()
        return (
            "The forces (N) [fx, fy] to apply to the pusher along the world "
            f"x and y axes, limited to +/- {max_fx} N and +/- {max_fy} N and "
            f"held constant between policy queries at {self.control_rate} Hz."
        )

    @property
    def action_example(self):
        """No force on the pusher."""
        return np.zeros(2)

    @property
    def observation_description(self):
        """The 11-number observation layout."""
        return (
            "In the world frame: pusher x, y, x velocity, y velocity; T x, "
            "y (zero at the target), cosine and sine of the T's yaw (zero at "
            "the target), T x velocity, y velocity, and yaw rate."
        )

    @property
    def observation_example(self):
        """Pusher at its start, T at the target, both at rest."""
        return np.array(
            [0.0, -0.45, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
