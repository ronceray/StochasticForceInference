import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, better for Langevin simulations
from jax import random
from VdP_model import *


# ID = 'large'
# Nsteps = 1000000        # Number of recorded time points
# dt = 0.001            # Time step between recorded points
# prerun = 100        # Number of equilibration time steps to run & dump before recording start
# oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling


ID = 'tiny'
Nsteps = 1000        # Number of recorded time points
dt = 0.02            # Time step between recorded points
prerun = 100        # Number of equilibration time steps to run & dump before recording start
oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling

# ID = 'small'
# Nsteps = 100000        # Number of recorded time points
# dt = 0.01            # Time step between recorded points
# prerun = 100        # Number of equilibration time steps to run & dump before recording start
# oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling



# Initialize random key
key= random.key(0)
key, subkey = random.split(key)

# Single copy simulation:
VdP_Model.initialize(initial_position,initial_velocity,theta_force,theta_diffusion)

# Perform the simulation (SDE integration)
VdP_Model.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
VdP_Model.compute_information()
print("Information: ",VdP_Model.I)

# Save the trajectory as CSV
VdP_Model.save_trajectory_data("VdP_data_"+ID+".csv")

