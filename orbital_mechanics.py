import numpy as np
from scipy.integrate import solve_ivp
# Earth Constants
MU_Earth = 398600.4418  # Gravitational parameter (m^3/s^2)
R_Earth = 6371.0 # Mean radius (m)

#to simulate orbit, we need to calcuulate how satellite's position and velocity change over time due to gravity.

def two_body_ode(t,state):
    """
    Differential equations for the two body problem.
    Calculates the derivatives of position and velocity.
    """
    r_vec = state[0:3]  # Position vector
    v_vec = state[3:6]  # Velocity vector

    r_norm = np.linalg.norm(r_vec)
    a_vec = -MU_Earth * r_vec / r_norm**3  # Acceleration due to gravity
    return np.concatenate((v_vec, a_vec))

def propogate_orbit(r0, v0, t_span, dt=10.0):
    """
    Propogates the orbit from intial conditions usingthe RK45 integrator.

    Params:
    r0: Initial position vector [x, y, z] in km
    v0: Initial velocity vector [vx, vy, vz] in km/s
    t_span: Time span for integration
    dt: Time step for integration
    """
    state_init = np.concatenate((r0, v0))  # Initial state vector
    t_eval = np.arange(0,t_span,dt)
    solution = solve_ivp(
        fun=two_body_ode,
        t_span=(0, t_span),
        y0=state_init,
        method='RK45',
        t_eval=t_eval,
        rtol = 1e-8,
        atol= 1e-8
    )
    return solution.t, solution.y
