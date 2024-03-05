# GravitySim
These Notebooks contain simulations of gravitational between bodies. Starting with a 1D simulation of two bodies, a 2D simulation was also implemented. Hopefully, I will get to a 3D Simulation and perhaps also tackle the 3-body problem.

To see the results look into the Notebooks for static plots showing the systems behaviours or play the .gif files.

The equations of motion were derived by building a system of OEDs using Lagrangian mechanics and solving them with SciPy's solve_ivp solver. For implementation and symbolic computation, I used Sympy.

Thanks to Logan Dihel (https://www.youtube.com/@logandihel), who kindly made his code on lagrangian mechanics for an elastic pendulum public, which I partly used and adapted for my simulations.
