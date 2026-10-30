# LEAP Hand

16-DoF LEAP Hand (left and right) from Carnegie Mellon University
([leaphand.com](https://leaphand.com),
[LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim)).

The URDFs and meshes are taken from
[dex-urdf](https://github.com/dexsuite/dex-urdf)
(`robots/hands/leap_hand`), which modified the original meshes and URDFs.
Distributed under the MIT license; see [LICENSE.md](LICENSE.md).

## Contact

Every collision geometry is compliant hydroelastic. The fingertip collision
meshes are treated as their convex hulls.

The fingertip collision meshes are 512-face convex hulls coarsened from the
dex-urdf meshes by greedy hull refinement: starting from the six axis-extreme
vertices of the original hull, repeatedly add the original hull vertex
farthest outside the current hull until it has 512 faces. They use only
original vertices, so they never extend outside the original surface.

## Actuation

Each of the 16 revolute joints has an effort-limited (0.95 N·m) actuator.
