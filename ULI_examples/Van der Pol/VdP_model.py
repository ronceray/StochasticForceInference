"""
Van der Pol oscillator with multiplicative noise for ULI demo purposes.

Author: Pierre Ronceray, 2025
"""

import jax.numpy as jnp
from jax import random,jit

# The parameters of the model:
dim = 1

# Constant diffusion:
theta_diffusion = jnp.array([1.,0.,0.])
@jit
def diffusion_VdP(X,V,theta):
    return  theta[0]* jnp.eye(dim) + theta[1] * jnp.outer(X,X) + theta[2] * jnp.outer(V,V)

# Force :
theta_force = jnp.array([2.])
@jit
def force_VdP(X,V,theta):
    return  theta[0] * (1-X**2)*V - X

initial_position = jnp.zeros(dim)
initial_velocity = jnp.zeros(dim) + 0.1

# Uninitialized system to be used elsewhere
import SFI
VdP_Model = SFI.SFI_Langevin.UnderdampedLangevinProcess(force_VdP,diffusion_VdP) 

