import jax.numpy as jnp
from jax import random,jit
import pickle

    
# The parameters of the model:

omega = 0.001   # Harmonic confinement
v0 = 1.       # Preferred velocity
gamma = 1.0    # Velocity relaxation rate

D = 0.4        # Velocity diffusion

# Central force (Lennard-Jones-like):
epsilon_central = 1. # strength
R0_central = 4.      # range
def central_kernel(rij,epsilon,R0):
    return epsilon * (1-(rij/R0)**3)/((rij/R0)**6+1) 

# Alignment (somewhat Vicsek-like):
epsilon_align = 1.  # strength
R0_align = 3.       # range
def alignment_kernel(rij,epsilon,R0):
    return epsilon * jnp.exp(-rij/R0)

Dmatrix = D * jnp.eye(3)

# Aggregate parameters:
theta_single = jnp.array([gamma,v0,omega])
theta_pair = jnp.array([epsilon_central,R0_central,epsilon_align,R0_align])

@jit
def one_point_force(X,V,theta_single) :
    drive = theta_single[0] * ( theta_single[1]**2 - jnp.sum(jnp.square(V))) * V
    return - theta_single[2] * X + drive

@jit
def pair_force(Xi,Xj,Vi,Vj,theta_pair):
    rij = jnp.linalg.norm(Xi-Xj)    
    Fr = central_kernel(rij,theta_pair[0],theta_pair[1]) / rij
    Fa = alignment_kernel(rij,theta_pair[2],theta_pair[3])
    return (Xi-Xj) * Fr - (Vi-Vj) * Fa

    

R0 = R0_central * 0.75

def initial_position_and_velocity(W):
    # Initial position: 3D cubic crystal with 2*R0 spacing
    x = jnp.linspace(-W * R0, W * R0, W)
    y = jnp.linspace(-W * R0, W * R0, W)
    z = jnp.linspace(-W * R0, W * R0, W)

    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    positions = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Constant velocity in the x-direction for all particles
    velocities = jnp.zeros_like(positions)
    velocities = velocities.at[:, 0].set(v0/2)
    velocities = velocities.at[:, 1].set(v0/2)

    return positions, velocities

from SFI.SFI_Langevin import ParticlesUnderdampedLangevinProcess

# Create an uninitialized instance of the simulated system
FlockingModel = ParticlesUnderdampedLangevinProcess(one_point_force, pair_force, theta_single, theta_pair, Dmatrix)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

def animate_flock(M, filename='ABP_animation.mp4', fps=30, max_duration=100, tail_length=60, dpi=100, frame_skip=0):
    """
    Create an MP4 animation of the particles' movement.

    Args:
    filename (str): Name of the output MP4 file.
    fps (int): Frames per second for the output video.
    max_duration (float): Maximum duration of the animation in seconds.
    tail_length (int): Number of previous positions to show as a tail.
    dpi (int): Dots per inch for the output video.
    frame_skip (int): Number of frames to skip between each plotted frame.

    Returns:
    str: The filename of the saved video.
    """
    data = np.array(M.X[::frame_skip+1])
    data_V = np.array(M.V[::frame_skip+1])

    fig, ax = plt.subplots(figsize=(6,6))
    margin = 0.1
    max_abs_coord = np.abs(data[:, :, :2]).max()
    bound = max_abs_coord * (1 + margin)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal')
    plt.tight_layout()

    num_particles = data.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_particles))

    line_data = [np.zeros((tail_length, 2)) for _ in range(num_particles)]
    lines = LineCollection(line_data, colors=colors, alpha=0.5)
    ax.add_collection(lines)

    arrow_length = 3.
    quiver = ax.quiver(data[0, :, 0], data[0, :, 1],
                       arrow_length * data_V[0,:,0], arrow_length * data_V[0,:,1],
                       color=colors, scale=1, scale_units='xy', angles='xy', width=0.01, minlength=0)

    def update(frame):
        # Update lines
        start = max(0, frame - tail_length)
        segments = data[start:frame+1, :, :2].transpose(1, 0, 2)
        lines.set_segments(segments)
        """
        # Update alpha for fading effect
        alphas = np.linspace(0.1, 1, tail_length) if frame >= tail_length else np.linspace(0.1, 1, frame+1)

        # Ensure alphas array matches the number of segments
        if frame >= tail_length:
            alphas = np.tile(alphas, (num_particles, 1)).flatten()
        else:
            alphas = np.tile(alphas, (num_particles, 1))[:, :frame+1].flatten()
        lines.set_alpha(alphas)
        """
        # Update quiver
        positions = data[frame, :, :2]
        quiver.set_offsets(positions)
        quiver.set_UVC(arrow_length * data_V[frame,:,0] / np.linalg.norm(data_V[frame], axis=1),
                       arrow_length * data_V[frame,:,1] / np.linalg.norm(data_V[frame], axis=1))

        return lines, quiver

    num_frames = min(int(max_duration * fps), len(data))
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=lambda: (lines, quiver), blit=True)

    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)

    return filename
