# LEAP Hand

16-DoF LEAP Hand (left and right) from Carnegie Mellon University
([leaphand.com](https://leaphand.com),
[LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim)).

The URDFs and meshes are taken from
[dex-urdf](https://github.com/dexsuite/dex-urdf)
(`robots/hands/leap_hand`), which modified the original meshes and URDFs.
Distributed under the MIT license; see [LICENSE.md](LICENSE.md).

## Contact

Every collision geometry is compliant hydroelastic (modulus 1e6 Pa,
friction 1.0). The fingertip collision meshes are treated as their convex
hulls. Objects need rigid or compliant hydroelastic properties of their own
to get hydroelastic contact with the hand; otherwise Drake falls back to
point contact.

## Actuation

Each of the 16 revolute joints has a URDF `<transmission>`, so Drake adds one
effort-limited (0.95 N·m) actuator per joint.
