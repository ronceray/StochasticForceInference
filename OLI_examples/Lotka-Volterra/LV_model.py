"""
"""

import jax.numpy as jnp
from jax import random,jit

# The parameters of the model:
dim = 6

# Parameters from chatgpt.

# 1) Intrinsic growth rates:
r = jnp.array([0.50, 0.65, 0.55, 0.65, 0.58, 0.62])

# 2) Interaction matrix A (nonzero entries within a factor 3):
A = jnp.array([
    [-0.90, -0.60,  0.00,  0.00,  0.00,  0.00],
    [ 0.00, -0.70, -0.60,  0.00,  0.00,  0.00],
    [+0.60,  0.00, -0.90,  0.00,  0.00, -0.40],
    [ 0.00,  0.00,  0.00, -0.90, -0.50,  0.00],
    [ 0.00,  0.00,  0.00,  0.00, -0.40, -0.60],
    [ 0.00,  0.00, -0.40, +0.40,  0.00, -1.00],
])


# Constant diffusion:
D = 0.002 * jnp.eye(dim)

theta_force = jnp.array([],dtype=float)

#key= random.key(4)
#key, subkey1, subkey2 = random.split(key,3)
#A = mag * random.bernoulli(subkey1, p = 0.15, shape=(dim,dim)).astype(float) + jnp.eye(dim)
#B = 1. + random.normal(subkey2,shape=(dim,))**2
@jit
def force_LV(x,theta):
    return A @ jnp.exp(x) + r
    
initial_position = jnp.zeros(dim) - 6.

# Uninitialized system to be used elsewhere
from SFI.SFI_Langevin import OverdampedLangevinProcess
LV_Model = OverdampedLangevinProcess(force_LV,D) 

