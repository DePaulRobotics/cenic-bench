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

## Actuation

Each of the 16 revolute joints has an effort-limited (0.95 N·m) actuator.
