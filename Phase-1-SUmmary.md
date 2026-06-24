# Project Implementation Summary

This document provides a comprehensive breakdown of the core phases involved in bridging computational fluid kinematics with predictive data science.

---

## 🔬 Phase 1: High-Fidelity Synthetic Data Generation

Because real-world aerodynamic kinematics involves non-linear differential equations that lack a simple algebraic solution, we built a custom computational physics engine in Python to synthesize our training data. Instead of utilizing idealized vacuum assumptions, this simulator models a highly realistic, chaotic flight medium.

### The Physics Core (Numerical Integration)
The dataset generator models aerodynamic drag as a macro-scale fluid obstruction framework following the quadratic drag equation:

$$F_d = \frac{1}{2} \rho v^2 C_d A$$

Because this force scales exponentially with velocity ($v^2$), acceleration changes continuously. To resolve this, the engine employs **Euler's Method for Numerical Integration** using a microscopic time step ($\Delta t = 0.001\text{ s}$). 

Inside a high-speed `while y >= 0:` loop, the engine executes the following matrix calculations every millisecond:
1. **Vector Resolution:** The diagonal velocity vector ($v$) is calculated using Pythagoras, and the instantaneous flight angle ($\theta$) is used to decompose the total drag force ($F_d$) into independent directional vectors ($F_{dx}$ and $F_{dy}$).
2. **Force Application:** Newton’s Second Law ($a = \frac{F}{m}$) is applied to both axes. The horizontal axis is subjected strictly to decelerating drag, while the vertical axis combines both gravitational pull ($g$) and directional drag vectors.
3. **State Updating:** The velocity components and coordinate displacements are iteratively updated by multiplying the accelerations by the time step ($\Delta t$), tracking the precise trajectory frame-by-frame.
