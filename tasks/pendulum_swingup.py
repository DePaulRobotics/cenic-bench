import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Meshcat,
    MeshcatVisualizer,
    Parser,
    StartMeshcat,
)

from tasks.base import Task


class PendulumPolicy(LeafSystem):
    """Wraps a policy callable, mapping pendulum state to a saturated torque."""

    def __init__(self, policy, torque_limit: float):
        super().__init__()
        self.policy = policy
        self.torque_limit = torque_limit
        self.state_port = self.DeclareVectorInputPort("state", 2)
        self.DeclareVectorOutputPort("torque", 1, self.CalcTorque)

    def CalcTorque(self, context, output: BasicVector):
        theta, theta_dot = self.state_port.Eval(context)
        obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
        u = float(np.asarray(self.policy(obs)).reshape(-1)[0])
        output.SetAtIndex(0, np.clip(u, -self.torque_limit, self.torque_limit))


class PendulumSwingup(Task):
    """Swing a torque-limited pendulum to an upright position.

    Observations:
        - The cosine of the pendulum's angle
        - The sine of the pendulum's angle
        - The angular velocity of the pendulum

    Actions:
        - The torque applied to the pendulum
    """

    urdf = "package://drake/examples/pendulum/Pendulum.urdf"
    torque_limit = 3.0  # N*m
    angle_tolerance = 0.1  # rad, distance from upright to count as success
    velocity_tolerance = 0.5  # rad/s, max speed at upright to count as success

    def __init__(self, policy, accuracy: float = 1e-3, meshcat: Meshcat | None = None):
        # Set before the base constructor, which calls create_scene.
        self.meshcat = meshcat if meshcat is not None else StartMeshcat()
        super().__init__(policy, accuracy, realtime=True)

    def create_scene(self, policy) -> Diagram:
        builder = DiagramBuilder()
        # CENIC requires a continuous-time plant.
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(self.plant).AddModelsFromUrl(self.urdf)
        self.plant.Finalize()

        MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)

        controller = builder.AddSystem(PendulumPolicy(policy, self.torque_limit))
        builder.Connect(
            self.plant.get_state_output_port(), controller.get_input_port()
        )
        builder.Connect(
            controller.get_output_port(), self.plant.get_actuation_input_port()
        )
        return builder.Build()

    def reset(self, context, seed: int = 0):
        # Start near the bottom with a small random perturbation.
        rng = np.random.default_rng(seed)
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        self.plant.SetPositions(plant_context, [rng.uniform(-0.5, 0.5)])
        self.plant.SetVelocities(plant_context, [rng.uniform(-0.5, 0.5)])

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
            "Swing a torque-limited pendulum from hanging down to upright. "
            f"The torque is limited to {self.torque_limit} N*m, less than the "
            "gravitational torque at horizontal, so the pendulum must be "
            "pumped up. Success is reaching within "
            f"{self.angle_tolerance} rad of upright with angular speed below "
            f"{self.velocity_tolerance} rad/s."
        )

if __name__=="__main__":
    dummy_policy = lambda x: 0.0
    task = PendulumSwingup(dummy_policy)
    success = task.run_episode()
    print("Success:", success)