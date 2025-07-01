"""
Damped Harmonic Oscillator for ULI demo purposes.

Author: Pierre Ronceray, 2025
"""

import jax.numpy as jnp
from jax import random,jit

# The parameters of the model:
dim = 6

# Constant diffusion:
D = 1.0 * jnp.eye(dim)

# Force :
diag_term = 1.
off_diag_term = 0.4
friction_term = 1.0
theta_force = jnp.array([diag_term,off_diag_term,friction_term])

@jit
def force_DH(x,v,theta):
    A = theta[0] * jnp.eye(dim) + theta[1] * (jnp.eye(dim,k=1) - jnp.eye(dim,k=-1))
    B = theta[2] * jnp.eye(dim)
    return - A @ x - B @ v
    
initial_position = jnp.zeros(dim)
initial_velocity = jnp.zeros(dim)

# Uninitialized system to be used elsewhere
from SFI.SFI_Langevin import UnderdampedLangevinProcess
DH_Model = UnderdampedLangevinProcess(force_DH,D) 

