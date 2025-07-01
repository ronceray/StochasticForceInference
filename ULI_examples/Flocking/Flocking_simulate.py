from jax import random
from Flocking_model import FlockingModel,animate_flock,initial_position_and_velocity


# Very small simulation - ~20kB:
# ID = 'tiny'
# Nsteps = 100        # Number of recorded time points
# dt = 0.1            # Time step between recorded points
# prerun = 1000          # Number of equilibration time steps to run & dump before recording start
# oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
# W = 2               # Width of the particles grid (initial configuration); Nparticles = W**3

# Small simulation
ID = 'small'
Nsteps = 25000         # Number of recorded time points
dt = 0.005            # Time step between recorded points
prerun = 5000          # Number of equilibration time steps to run & dump before recording start
oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
W = 3                # Width of the particles grid (initial configuration); Nparticles = W**3

# # Large simulation
# ID = 'large'
# Nsteps = 100000         # Number of recorded time points
# dt = 0.001           # Time step between recorded points
# prerun = 50000          # Number of equilibration time steps to run & dump before recording start
# oversampling = 10    # Number of integration steps between each recorded point; simulation ddt = dt / oversampling
# W = 5                # Width of the particles grid (initial configuration); Nparticles = W**3

# Initialize random key
key= random.key(1)
key, subkey = random.split(key)

# Initialize the system's instance with the initial positions
init_X,init_V = initial_position_and_velocity(W)
FlockingModel.initialize(init_X,init_V)

# Perform the simulation (SDE integration)
FlockingModel.simulate(dt,Nsteps,prerun=prerun,oversampling=oversampling,key=subkey)

# The information and entropy production are useful to assess feasibility of inference
FlockingModel.compute_information()
print("Information: ",FlockingModel.I)

# Save the trajectory as CSV
FlockingModel.save_trajectory_data("Flocking_data_"+ID+".csv")

# Optional : save the movie
animate_flock(FlockingModel,filename="Flocking_animation_"+ID+".mp4",frame_skip=60,tail_length=2500)
