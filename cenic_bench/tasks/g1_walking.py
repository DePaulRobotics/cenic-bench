import numpy as np
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    HalfSpace,
    ProximityProperties,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization

from cenic_bench.parsing import make_parser
from cenic_bench.tasks.base import Task


class G1Policy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram.

    The policy is queried at a fixed rate. Its action is turned into joint
    position targets for the G1's PD actuators, which are held constant
    (zero-order hold) between queries.
    """

    def __init__(self, policy, home, scale, command, control_rate: float):
        super().__init__()
        self.policy = policy
        self.home = home
        self.scale = scale
        self.command = command
        self.state_port = self.DeclareVectorInputPort("state", 36 + 35)
        desired_state = self.DeclareDiscreteState(
            np.concatenate([home, np.zeros(29)])
        )
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("desired_state", desired_state)

    def Update(self, context, discrete_state):
        x = self.state_port.Eval(context)
        quaternion, q_joints = x[0:4], x[7:36]
        w_WB, v_WB, v_joints = x[36:39], x[39:42], x[42:71]
        R_BW = RotationMatrix(Quaternion(quaternion)).inverse().matrix()
        obs = np.concatenate(
            [
                R_BW @ v_WB,
                R_BW @ w_WB,
                R_BW @ [0.0, 0.0, -1.0],
                self.command,
                q_joints - self.home,
                v_joints,
            ]
        )
        action = self.policy(obs)
        q_desired = self.home + self.scale * action
        discrete_state.set_value(np.concatenate([q_desired, np.zeros(29)]))


class G1Walking(Task):
    """Walk the Unitree G1 humanoid forward at a target speed.

    The robot starts at rest in a slightly crouched standing pose (mjlab's
    HOME keyframe), facing +x on flat ground, with small random joint offsets.

    Observations:
        - The pelvis linear velocity, in the pelvis frame
        - The pelvis angular velocity, in the pelvis frame
        - The gravity direction, in the pelvis frame
        - The velocity command (forward speed, lateral speed, yaw rate)
        - The joint positions, relative to the HOME pose
        - The joint velocities

    Actions:
        - Joint position targets for the 29 PD-controlled joints, as scaled
          offsets from the HOME pose, held constant between policy queries at
          50 Hz.

    Success conditions:
        - After 20 seconds, the pelvis's average forward speed is within
          0.1 m/s of 0.5 m/s
    """

    target_velocity = 0.5  # m/s, forward speed to track along world +x
    velocity_tolerance = 0.1  # m/s, max error of the average forward speed
    duration = 20.0  # s, how long the robot must walk
    control_rate = 50.0  # Hz, how often the policy is queried
    pelvis_height = 0.783675  # m, pelvis height in the HOME pose

    # Joint angles (rad) of mjlab's HOME pose. Other joints are zero.
    home_joints = {
        "left_hip_pitch_joint": -0.1,
        "right_hip_pitch_joint": -0.1,
        "left_knee_joint": 0.3,
        "right_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "right_ankle_pitch_joint": -0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 1.28,
        "right_elbow_joint": 1.28,
    }

    def __init__(self, policy, visualize: bool = True):
        """Initialize the G1 walking task.

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
        (g1,) = make_parser(self.plant).AddModelsFromUrl(
            "package://cenic_bench/models/unitree_g1/g1.sdf"
        )
        # The G1's collision geometry is compliant, so the ground is rigid.
        props = ProximityProperties()
        AddRigidHydroelasticProperties(props)
        AddContactMaterial(friction=CoulombFriction(1.0, 1.0), properties=props)
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground",
            props,
        )
        self.plant.Finalize()

        actuators = [
            self.plant.get_joint_actuator(i)
            for i in self.plant.GetJointActuatorIndices()
        ]
        self.joint_names = [a.joint().name() for a in actuators]
        self.home = np.array(
            [self.home_joints.get(name, 0.0) for name in self.joint_names]
        )
        # mjlab's action scale: a unit action asks for a quarter of the effort
        # limit from the PD controller.
        self.action_scale = np.array(
            [
                0.25 * a.effort_limit() / a.get_controller_gains().p
                for a in actuators
            ]
        )

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        command = np.array([self.target_velocity, 0.0, 0.0])
        controller = builder.AddSystem(
            G1Policy(
                policy, self.home, self.action_scale, command, self.control_rate
            )
        )
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(),
            self.plant.get_desired_state_input_port(g1),
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Start at rest in the HOME pose with small joint offsets."""
        rng = np.random.default_rng(seed)
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        q = np.concatenate(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, self.pelvis_height],
                self.home + rng.uniform(-0.05, 0.05, size=29),
            ]
        )
        self.plant.SetPositions(plant_context, q)
        self.plant.SetVelocities(plant_context, np.zeros(35))

    def success(self, context) -> bool:
        """Average forward speed near the target after the full duration."""
        t = context.get_time()
        if t < self.duration:
            return False
        plant_context = self.plant.GetMyContextFromRoot(context)
        # The pelvis starts at x = 0.
        x = self.plant.GetPositions(plant_context)[4]
        return abs(x / t - self.target_velocity) <= self.velocity_tolerance

    @property
    def timeout(self):
        # A little past the duration, so the monitor sees t >= duration.
        return self.duration + 0.5

    @property
    def task_description(self):
        return (
            "Make a Unitree G1 humanoid walk forward (+x) on flat ground. "
            f"Success is an average forward speed within "
            f"{self.velocity_tolerance} m/s of {self.target_velocity} m/s, "
            f"measured after {self.duration} s."
        )

    @property
    def action_description(self):
        return (
            "29 joint position targets for the PD-controlled joints, as "
            "offsets from the HOME pose scaled per joint (target = home + "
            "scale * action, with a scale near 0.25 * effort limit / PD "
            "stiffness), held constant between policy queries at "
            f"{self.control_rate} Hz. Joints are ordered as in joint_names: "
            f"{', '.join(self.joint_names)}."
        )

    @property
    def action_example(self):
        return np.zeros(29)

    @property
    def observation_description(self):
        return (
            "70 values: pelvis linear velocity (3) and angular velocity (3) "
            "in the pelvis frame, unit gravity direction in the pelvis frame "
            "(3), velocity command (forward m/s, lateral m/s, yaw rate rad/s) "
            "(3), joint positions relative to the HOME pose (29), and joint "
            "velocities (29), with joints ordered as in the action."
        )

    @property
    def observation_example(self):
        return np.concatenate(
            [
                np.zeros(6),
                [0.0, 0.0, -1.0],
                [self.target_velocity, 0.0, 0.0],
                np.zeros(58),
            ]
        )
