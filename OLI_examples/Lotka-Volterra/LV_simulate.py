import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
from jax import random
from LV_model import *


ID = 'small'
Nsteps = 40000        # Number of recorded time points
dt = 0.01            # Time step between recorded points
prerun = 0        # Number of equilibration time steps to run & dump before recording start
oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling


# Initialize random key
key= random.key(0)
key, subkey = random.split(key)

LV_Model.initialize(initial_position,theta_force)

# Perform the simulation (SDE integration)
LV_Model.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
LV_Model.compute_information()
print("Information and entropy: ",LV_Model.I,LV_Model.S)

# Save the trajectory as CSV
LV_Model.save_trajectory_data("LV_data_"+ID+".csv")

