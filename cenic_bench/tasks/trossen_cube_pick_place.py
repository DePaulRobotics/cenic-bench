import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.systems.primitives import (
    ConstantVectorSource,
    Multiplexer,
    StateInterpolatorWithDiscreteDerivative,
)
from pydrake.visualization import AddDefaultVisualization

from cenic_bench.parsing import make_parser
from cenic_bench.tasks.base import Task

NUM_ARM_JOINTS = 6
NUM_ARM_ACTUATORS = (
    NUM_ARM_JOINTS + 2
)  # 6 joints + 2 independent gripper fingers
# All-zero joint angles put the gripper clear of the table and frame,
# so it is used as both the home pose for the controlled arm (left)
# and the fixed pose for the parked one (right).
HOME_ARM_POSITION = np.zeros(NUM_ARM_JOINTS)
HOME_GRIPPER_POSITION = 0  # close

# Table surface (z=0.02) plus the cube's half-height; verified by dropping
# the cube onto the table and checking where it settles.
CUBE_REST_HEIGHT = 0.03
# Verified reachable by the left arm's gripper via inverse kinematics, both
# at hover height (0.12m) and at grasp height (0.05m).
CUBE_SPAWN_X_RANGE = (-0.15, 0.15)
CUBE_SPAWN_Y_RANGE = (0.0, 0.3)
DROP_ZONE_XY = np.array([0.0, -0.10])


def _desired_positions(
    plant, model_instance, prefix, arm_position, gripper_position
):
    """Build the 8-actuator desired-position vector for one arm.

    `gripper_position` is sent to both carriage joints, replacing the URDF's
    mimic constraint (removed because CENIC's continuous-time plants don't
    support constraints). Indexed by JointActuator.input_start(), which is
    NOT the same ordering as the plant's positions.
    """
    u = np.zeros(plant.num_actuators(model_instance))
    for i, q in enumerate(arm_position):
        actuator = plant.GetJointActuatorByName(f"{prefix}_motor_{i}")
        u[actuator.input_start()] = q
    for name in (
        f"{prefix}_left_carriage_motor",
        f"{prefix}_right_carriage_motor",
    ):
        actuator = plant.GetJointActuatorByName(name)
        u[actuator.input_start()] = gripper_position
    return u


def _set_arm_configuration(
    plant, plant_context, prefix, arm_position, gripper_position
):
    """Set one arm's joints directly on a context, at rest.

    Operates per-joint via Joint.set_angle()/set_translation(), which sidesteps
    any question of position-vector ordering entirely.
    """
    for i, q in enumerate(arm_position):
        joint = plant.GetJointByName(f"{prefix}_joint_{i}")
        joint.set_angle(plant_context, q)
        joint.set_angular_rate(plant_context, 0.0)
    for name in (
        f"{prefix}_left_carriage_joint",
        f"{prefix}_right_carriage_joint",
    ):
        joint = plant.GetJointByName(name)
        joint.set_translation(plant_context, gripper_position)
        joint.set_translation_rate(plant_context, 0.0)


class TrossenLeftArmPolicy(LeafSystem):
    """Wraps a policy controlling the left arm; queried at a fixed rate.

    Outputs only the left arm's 8 actuator setpoints (6 joints + 2 gripper
    fingers); the right arm is driven separately, by a fixed source wired up
    in create_scene(). The policy's 7-dim action (6 joint positions + 1
    gripper position) is expanded into those 8 setpoints and held constant
    (zero-order hold) between queries.
    """

    def __init__(self, plant, model_instance, policy, control_rate):
        super().__init__()
        self._plant = plant
        self._model = model_instance
        self._policy = policy
        self._left_joints = [
            plant.GetJointByName(f"follower_left_joint_{i}")
            for i in range(NUM_ARM_JOINTS)
        ]
        self._gripper_joint = plant.GetJointByName(
            "follower_left_left_carriage_joint"
        )

        self.state_port = self.DeclareVectorInputPort(
            "plant_state", plant.num_positions() + plant.num_velocities()
        )
        desired = self.DeclareDiscreteState(NUM_ARM_ACTUATORS)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1.0 / control_rate, offset_sec=0.0, update=self.Update
        )
        self.DeclareStateOutputPort("left_desired_positions", desired)

    def _observation(self, state):
        nq = self._plant.num_positions()
        arm_q = np.array([state[j.position_start()] for j in self._left_joints])
        arm_v = np.array(
            [state[nq + j.velocity_start()] for j in self._left_joints]
        )
        gripper_q = state[self._gripper_joint.position_start()]
        gripper_v = state[nq + self._gripper_joint.velocity_start()]
        cube_body = self._plant.GetBodyByName("cube_link")
        cube_xyz = state[
            cube_body.floating_positions_start()
            + 4 : cube_body.floating_positions_start() + 7
        ]
        return np.concatenate([arm_q, arm_v, [gripper_q, gripper_v], cube_xyz])

    def Update(self, context, discrete_state):
        state = self.state_port.Eval(context)
        obs = self._observation(state)
        action = np.asarray(self._policy(obs))
        u_full = _desired_positions(
            self._plant,
            self._model,
            "follower_left",
            action[:NUM_ARM_JOINTS],
            action[NUM_ARM_JOINTS],
        )
        # follower_left's actuators are indices 0:NUM_ARM_ACTUATORS,
        # so this slice is exactly its 8 setpoints, in order.
        discrete_state.set_value(u_full[:NUM_ARM_ACTUATORS])


class TrossenCubePickPlace(Task):
    """Pick up a cube with the Trossen Stationary AI's left arm and place it
    at a fixed drop zone on the table.

    Only the left arm is controlled; the right arm stays parked.

    Observations:
        - The left arm's 6 joint positions and 6 joint velocities
        - The left gripper's position and velocity
        - The cube's position (x, y, z)

    Actions:
        - Desired positions for the left arm's 6 joints and its gripper
          (applied to both gripper fingers), updated at 10 Hz

    Success conditions:
        - cube within tolerance of a fixed drop zone, at rest
    """

    control_rate = 10.0  # Hz, how often the policy is queried
    position_tolerance = 0.03  # m, xy distance from the drop zone for success
    height_tolerance = 0.01  # m, z distance from resting on the table
    velocity_tolerance = 0.05  # m/s, max cube speed to count as at rest

    def __init__(self, policy, visualize: bool = True):
        self.meshcat = StartMeshcat() if visualize else None
        super().__init__(policy, realtime=visualize, accuracy=1e-3)

    def create_scene(self, policy) -> Diagram:
        builder = DiagramBuilder()

        # CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = make_parser(self.plant)
        parser.AddModelsFromUrl(
            "package://cenic_bench/models/trossen_stationary_ai/stationary_ai.urdf"
        )
        parser.AddModelsFromUrl("package://cenic_bench/models/cube/cube.urdf")
        self.plant.Finalize()
        self.model = self.plant.GetModelInstanceByName("stationary_ai")

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(
            TrossenLeftArmPolicy(
                self.plant, self.model, policy, self.control_rate
            )
        )
        builder.Connect(
            self.plant.get_state_output_port(), controller.state_port
        )

        # The right arm isn't part of the action; hold it at the same fixed
        # pose reset() gives it, so it never moves. Sliced the same way as
        # the left arm above: follower_right's actuators are indices
        # NUM_ARM_ACTUATORS:2*NUM_ARM_ACTUATORS.
        right_arm_u = _desired_positions(
            self.plant,
            self.model,
            "follower_right",
            HOME_ARM_POSITION,
            HOME_GRIPPER_POSITION,
        )[NUM_ARM_ACTUATORS:]
        right_arm_source = builder.AddSystem(ConstantVectorSource(right_arm_u))

        mux = builder.AddSystem(
            Multiplexer([NUM_ARM_ACTUATORS, NUM_ARM_ACTUATORS])
        )
        builder.Connect(controller.get_output_port(), mux.get_input_port(0))
        builder.Connect(
            right_arm_source.get_output_port(), mux.get_input_port(1)
        )

        nu = self.plant.num_actuators(self.model)  # both arms, 16 total
        interpolator = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(nu, 1.0 / 200.0, True)
        )
        builder.Connect(mux.get_output_port(), interpolator.get_input_port())
        builder.Connect(
            interpolator.get_output_port(),
            self.plant.get_desired_state_input_port(self.model),
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Reset both arms to their home pose, and place the cube at rest at
        a random reachable position on the table."""
        self.diagram.SetDefaultContext(context)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        for prefix in ("follower_left", "follower_right"):
            _set_arm_configuration(
                self.plant,
                plant_context,
                prefix,
                HOME_ARM_POSITION,
                HOME_GRIPPER_POSITION,
            )

        rng = np.random.default_rng(seed)
        x = rng.uniform(*CUBE_SPAWN_X_RANGE)
        y = rng.uniform(*CUBE_SPAWN_Y_RANGE)
        cube_body = self.plant.GetBodyByName("cube_link")
        self.plant.SetFreeBodyPose(
            plant_context, cube_body, RigidTransform([x, y, CUBE_REST_HEIGHT])
        )
        self.plant.SetFreeBodySpatialVelocity(
            plant_context,
            cube_body,
            SpatialVelocity(np.zeros(3), np.zeros(3)),
        )

    def success(self, context) -> bool:
        """Cube at rest, at table height, within the drop zone."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        cube_body = self.plant.GetBodyByName("cube_link")
        xyz = cube_body.EvalPoseInWorld(plant_context).translation()
        speed = np.linalg.norm(
            cube_body.EvalSpatialVelocityInWorld(plant_context).translational()
        )
        xy_error = np.linalg.norm(xyz[:2] - DROP_ZONE_XY)
        height_error = abs(xyz[2] - CUBE_REST_HEIGHT)
        return (
            xy_error < self.position_tolerance
            and height_error < self.height_tolerance
            and speed < self.velocity_tolerance
        )

    @property
    def timeout(self):
        return 15.0

    @property
    def task_description(self):
        return "Not yet implemented."

    @property
    def action_description(self):
        return "Not yet implemented."

    @property
    def action_example(self):
        return np.zeros(NUM_ARM_JOINTS + 1)

    @property
    def observation_description(self):
        return "Not yet implemented."

    @property
    def observation_example(self):
        return np.zeros(2 * NUM_ARM_JOINTS + 2 + 3)
