from jax import random
from ABP_model import ABPmodel,animate_ABPs,initial_position


# # Very small simulation - ~40kB:
# ID = 'tiny'
# Nsteps = 100         # Number of recorded time points
# dt = 0.05            # Time step between recorded points
# prerun = 100          # Number of equilibration time steps to run & dump before recording start
# oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
# W = 3                # Width of the particles grid (initial configuration); Nparticles = W**2


# # Small simulation - < 1MB:
# ID = 'small'
# Nsteps = 1000         # Number of recorded time points
# dt = 0.01            # Time step between recorded points
# prerun = 10          # Number of equilibration time steps to run & dump before recording start
# oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
# W = 4                # Width of the particles grid (initial configuration); Nparticles = W**2

# # Medium simulation - ~ 10MB:
ID = 'medium'
Nsteps = 10000         # Number of recorded time points
dt = 0.01            # Time step between recorded points
prerun = 10000          # Number of equilibration time steps to run & dump before recording start
oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
W = 6                # Width of the particles grid (initial configuration); Nparticles = W**2

# # Large simulation - ~ 10MB:
# ID = 'large'
# Nsteps = 1000      # Number of recorded time points
# dt = 0.01            # Time step between recorded points
# prerun = 100       # Number of equilibration time steps to run & dump before recording start
# oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
# W = 40               # Width of the particles grid (initial configuration); Nparticles = W**2


# Initialize random key
key= random.key(0)
key, subkey = random.split(key)

# Initialize the system's instance with the initial positions
ABPmodel.initialize(initial_position(W))

# Perform the simulation (SDE integration)
ABPmodel.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
ABPmodel.compute_information()
print("Information and entropy: ",ABPmodel.I,ABPmodel.S)

# Save the trajectory as CSV
ABPmodel.save_trajectory_data("ABP_data_"+ID+".csv")

# Optional : save the movie
animate_ABPs(ABPmodel,filename="ABP_animation_"+ID+".mp4",frame_skip=10,tail_length=50)
