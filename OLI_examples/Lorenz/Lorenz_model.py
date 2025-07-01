"""
Stochastic Lorenz Model for SFI

This script defines a stochastic variant of the Lorenz system for use in 
Stochastic Force Inference (SFI). The dynamics are modeled by an overdamped 
Langevin equation with:
    - A force field derived from the Lorenz equations (drift term).
    - A state-dependent diffusion tensor (multiplicative noise).

The model allows benchmarking SFI methods and can be initialized via the 
`OverdampedLangevinProcess` class. It supports high-efficiency JAX-based 
simulations.

Author: Pierre Ronceray, 2025
"""

import jax.numpy as jnp
from jax import random,jit
import SFI


# The parameters of the model:


# Diffusion parameters: a linear diffusion gradient (multiplicative noise)
diffusion_coeff = 10.                     # Global prefactor for diffusion tensor
Dgradient_direction = jnp.array([1.,2,1]) # Direction of variations of D
alpha = 0.                              # Magnitude of variations (keep small or D will not be positive definite anymore...)
Diffusion_Lorenz = jit(lambda x,theta : theta[0] * (jnp.identity(3) + theta[1] * (jnp.outer(x, theta[2:]) + jnp.outer(theta[2:], x))))
theta_diffusion = jnp.concatenate((jnp.array([diffusion_coeff,alpha]),Dgradient_direction))


# Force field parameters (stochastic Lorenz process)
r,b,s = 20.,2.,10.     # The classic Lorenz force field parameters -
                      # here taken to subcritical values so the model
                      # is noise-driven.

theta_force = jnp.array([r,b,s])
Force_Lorenz = jit(lambda x,theta : jnp.array([ theta[2]*(x[2]-x[0]),
                                     x[0]*x[2]-theta[1]*x[1],
                                     theta[0]*x[0] - x[2] - x[1]*x[0]] ))

initial_position = jnp.array([0.,0.,0.]) 

# Uninitialized system to be used elsewhere
LorenzModel = SFI.SFI_Langevin.OverdampedLangevinProcess(Force_Lorenz,Diffusion_Lorenz) 

