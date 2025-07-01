import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
from jax import random
from DH_model import *


ID = 'small'
Nsteps = 10000        # Number of recorded time points
dt = 0.01            # Time step between recorded points
prerun = 100        # Number of equilibration time steps to run & dump before recording start
oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling

# ID = 'large'
# Nsteps = 1000000        # Number of recorded time points
# dt = 0.001            # Time step between recorded points
# prerun = 100        # Number of equilibration time steps to run & dump before recording start
# oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling


# Initialize random key
key= random.key(0)
key, subkey = random.split(key)

# Single copy simulation:
DH_Model.initialize(initial_position,initial_velocity,theta_force)

# Perform the simulation (SDE integration)
DH_Model.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
DH_Model.compute_information()
print("Information: ",DH_Model.I)

# Save the trajectory as CSV
DH_Model.save_trajectory_data("DH_data_"+ID+".csv")

