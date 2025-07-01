from Lorenz_model import *


ID = 'tiny'
Nsteps = 1000        # Number of recorded time points
dt = 0.01            # Time step between recorded points
prerun = 100        # Number of equilibration time steps to run & dump before recording start
oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling

# ID = 'fast'
# Nsteps = 10000        # Number of recorded time points
# dt = 0.01            # Time step between recorded points
# prerun = 100        # Number of equilibration time steps to run & dump before recording start
# oversampling = 20    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling


# ID = 'slow'
# Nsteps = 200000        # Number of recorded time points
# dt = 0.0005            # Time step between recorded points
# prerun = 1000          # Number of equilibration time steps to run & dump before recording start
# oversampling = 4       # Number of integration steps between each recorded point; simulation ddt = dt / oversampling


# Initialize random key
from jax import random
key = random.PRNGKey(0)
key, subkey = random.split(key)

# Single copy simulation:
LorenzModel.initialize(initial_position,theta_force,theta_diffusion)

# Multi copies simulation:
#Ncopies = 10
#LorenzModel.initialize(jnp.array([initial_position for i in range(Ncopies)]))

# Perform the simulation (SDE integration)
LorenzModel.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
LorenzModel.compute_information()
print("Information and entropy: ",LorenzModel.I,LorenzModel.S)

# Save the trajectory as CSV
LorenzModel.save_trajectory_data("Lorenz_data_"+ID+".csv")

