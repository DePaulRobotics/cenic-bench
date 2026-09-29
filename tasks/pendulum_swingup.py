import numpy as np
from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Meshcat,
    Parser,
    StartMeshcat,
)

from tasks.base import Task


class PendulumPolicy(LeafSystem):
    """Wraps a policy callable for compatibility with a Drake diagram."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.state_port = self.DeclareVectorInputPort("state", 2)
        self.DeclareVectorOutputPort("torque", 1, self.CalcTorque)

    def CalcTorque(self, context, output):
        theta, theta_dot = self.state_port.Eval(context)
        obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
        u = self.policy(obs)
        output.SetFromVector(u)


class PendulumSwingup(Task):
    """Swing an inverted pendulum to an upright position.

    Observations:
        - The cosine of the pendulum's angle
        - The sine of the pendulum's angle
        - The angular velocity of the pendulum

    Actions:
        - The torque applied to the pendulum

    Success conditions:
        - The pendulum is within 0.1 radians of upright
        - The angular velocity of the pendulum is below 0.5 radians/s
    """

    angle_tolerance = 0.1  # rad, distance from upright to count as success
    velocity_tolerance = 0.5  # rad/s, max speed at upright to count as success

    def __init__(self, policy, visualize: bool = True):
        """Initialize the pendulum swing-up task.

        Args:
            policy: A callable that takes an observation and returns an action.
            visualize: Whether to visualize the simulation. If true, connects
                       to meshcat and runs in ~realtime. Otherwise a headless
                       simulation runs as fast as possible.
        """
        self.meshcat = StartMeshcat() if visualize else None
        super().__init__(policy, realtime=visualize)

    def create_scene(self, policy) -> Diagram:
        builder = DiagramBuilder()

        # Note that CENIC requires a continuous-time plant.
        self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(self.plant).AddModelsFromUrl(
            "package://drake/examples/pendulum/Pendulum.urdf"
        )
        self.plant.Finalize()

        if self.meshcat is not None:
            AddDefaultVisualization(builder, self.meshcat)

        controller = builder.AddSystem(PendulumPolicy(policy))
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        """Start at rest at random angles between -pi and pi."""
        rng = np.random.default_rng(seed)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        self.plant.SetPositions(plant_context, [rng.uniform(-np.pi, np.pi)])
        self.plant.SetVelocities(plant_context, [0.0])

    def success(self, context) -> bool:
        plant_context = self.plant.GetMyContextFromRoot(context)
        theta = self.plant.GetPositions(plant_context)[0]
        theta_dot = self.plant.GetVelocities(plant_context)[0]
        # Wrapped angular distance from upright (theta = pi).
        error = np.abs(np.mod(theta, 2 * np.pi) - np.pi)
        return error < self.angle_tolerance and abs(theta_dot) < self.velocity_tolerance

    @property
    def timeout(self):
        return 10.0

    @property
    def task_description(self):
        return (
            "Swing a torque-controlled pendulum upright. Success is reaching within "
            f"{self.angle_tolerance} rad of upright with angular speed below "
            f"{self.velocity_tolerance} rad/s."
        )

if __name__=="__main__":
    dummy_policy = lambda obs: np.array([0.0])
    task = PendulumSwingup(dummy_policy)
    success = task.run_episode()
    print("Success:", success)