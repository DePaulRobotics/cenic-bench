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


class DiffDriveParkingPolicy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram.

    The policy is queried at a fixed rate, and the resulting wheel torques are
    held constant (zero-order hold) between queries.
    """

    def __init__(self, policy, control_rate: float):
        """Declare the state input, torque state, and periodic update."""
        super().__init__()
        self.policy = policy
        self.state_port = self.DeclareVectorInputPort("state", 17)
        torques = self.DeclareDiscreteState(2)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("torques", torques)

    def Update(self, context, discrete_state):
        """Query the policy with the current observation."""
        state = self.state_port.Eval(context)
        qw, qx, qy, qz, x, y = state[:6]
        wz, vx, vy = state[11:14]
        heading = np.arctan2(
            2 * (qw * qz + qx * qy), qw**2 + qx**2 - qy**2 - qz**2
        )
        obs = np.array([x, y, np.cos(heading), np.sin(heading), vx, vy, wz])
        u = self.policy(obs)
        discrete_state.set_value(u)


class DiffDriveParking(Task):
    """Drive a differential drive robot to the center of the floor and park.

    The robot rolls on a flat floor on two torque-controlled wheels and a
    frictionless caster. It starts at rest near the center of the floor, facing
    a random direction. It must reach the center, face +x, and stop there.

    Observations:
        - The chassis position along x
        - The chassis position along y
        - The cosine of the chassis heading
        - The sine of the chassis heading
        - The chassis velocity along x
        - The chassis velocity along y
        - The chassis heading rate

    Actions:
        - The left and right wheel torques, held constant between policy
          queries. Their magnitude is limited by the effort limits of the
          wheel joints, defined in diff_drive.sdf.

    Success conditions:
        - The robot is close to the center of the floor
        - The robot faces close to +x
        - The robot is nearly still
        - The robot is nearly not turning
    """

    position_tolerance = 0.05  # m, max distance of chassis from center
    heading_tolerance = 0.1  # rad, max heading error from +x
    speed_tolerance = 0.05  # m/s, max chassis speed when parked
    heading_rate_tolerance = 0.5  # rad/s, max heading rate when parked
    control_rate = 50.0  # Hz, how often the policy is queried

    def __init__(self, policy, visualize: bool = True):
        """Initialize the differential drive parking task.

        Args:
            policy: A callable that takes an observation and returns an action.
            visualize: Whether to visualize the simulation. If true, connects
                       to meshcat and runs in ~realtime. Otherwise a headless
                       simulation runs as fast as possible.
        """
        self.meshcat = StartMeshcat() if visualize else None
        super().__init__(policy, realtime=visualize, accuracy=1e-3)

    def create_scene(self, policy) -> Diagram:
        """Build the robot on a floor, driven by the wrapped policy."""
        builder = DiagramBuilder()

        # Note that CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = make_parser(self.plant)
        parser.AddModelsFromUrl(
            "package://cenic_bench/models/diff_drive/diff_drive.sdf"
        )
        X_WF = RigidTransform([0, 0, -0.05])
        floor = Box(30, 30, 0.1)
        props = ProximityProperties()
        AddRigidHydroelasticProperties(props)
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_WF, floor, "floor", props
        )
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_WF, floor, "floor", [0.8, 0.8, 0.8, 1.0]
        )
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(
            DiffDriveParkingPolicy(policy, self.control_rate)
        )
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Start at rest at a random position and heading on the floor."""
        rng = np.random.default_rng(seed)
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        chassis = self.plant.GetBodyByName("chassis")
        z = self.plant.GetFreeBodyPose(plant_context, chassis).translation()[2]
        x, y = rng.uniform(-1.0, 1.0, size=2)
        heading = rng.uniform(-np.pi, np.pi)
        X_WC = RigidTransform(RotationMatrix.MakeZRotation(heading), [x, y, z])
        self.plant.SetFreeBodyPose(plant_context, chassis, X_WC)

    def success(self, context) -> bool:
        """Chassis near the center, facing +x, and nearly at rest."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        qw, qx, qy, qz, x, y = self.plant.GetPositions(plant_context)[:6]
        wz, vx, vy = self.plant.GetVelocities(plant_context)[2:5]
        # Yaw in [-pi, pi], zero when facing +x.
        heading = np.arctan2(
            2 * (qw * qz + qx * qy), qw**2 + qx**2 - qy**2 - qz**2
        )
        return (
            np.hypot(x, y) < self.position_tolerance
            and abs(heading) < self.heading_tolerance
            and np.hypot(vx, vy) < self.speed_tolerance
            and abs(wz) < self.heading_rate_tolerance
        )

    @property
    def timeout(self):
        """Episode length in seconds."""
        return 10.0

    @property
    def task_description(self):
        """Describe the goal and success thresholds."""
        return (
            "Drive a two-wheeled differential drive robot to the center of "
            "a flat floor, face +x, and stop there. Success is reaching "
            f"within {self.position_tolerance} m of the center and within "
            f"{self.heading_tolerance} rad of facing +x, with speed below "
            f"{self.speed_tolerance} m/s and heading rate below "
            f"{self.heading_rate_tolerance} rad/s."
        )

    @property
    def action_description(self):
        """Describe the wheel torques and their limit."""
        max_torque = self.plant.GetJointActuatorByName("left").effort_limit()
        return (
            "The left and right wheel torques (N*m), each limited to "
            f"+/- {max_torque} N*m and held constant between policy queries "
            f"at {self.control_rate} Hz. Equal positive torques drive the "
            "robot forward."
        )

    @property
    def action_example(self):
        """Zero torque on both wheels."""
        return np.array([0.0, 0.0])

    @property
    def observation_description(self):
        """Describe the seven observation entries."""
        return (
            "Chassis x and y position (zero at the center of the floor), "
            "cosine and sine of the chassis heading (zero when facing +x, "
            "positive turning left), chassis x and y velocity in the world "
            "frame, and chassis heading rate."
        )

    @property
    def observation_example(self):
        """At rest at the center, facing +x."""
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
