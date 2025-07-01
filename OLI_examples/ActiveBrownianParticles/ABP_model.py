import jax.numpy as jnp
from jax import random,jit
import pickle

    
# The parameters of the model:
omega = 0.01   # Harmonic confinement
Factive = 1.   # Self propulsion
D = 0.1        # Positional diffusion
Dangular = 1. # Angular diffusion

Dmatrix = jnp.array([[ D,0,0],
                     [ 0,D,0],
                     [ 0,0,Dangular]])

theta_single = jnp.array([Factive,omega])
def harmonic_active(X,theta_single):
    Factive,omega = theta_single
    return jnp.array([
        -omega * X[0] + Factive * jnp.cos(X[2]),
        -omega * X[1] + Factive * jnp.sin(X[2]),
        0.
    ])

# Soft repulsion:
R0 = 1.       # Range
epsilon = 5.  # Strength
radial_interaction =  lambda rij,epsilon,R0 : - epsilon /((rij/R0)**2+1)

# Aligning torque with exponential decay with r: 
A = 5.   # Strength
L0 = 2.  # Range
theta_pair = jnp.array([ epsilon, R0, A, L0 ])
angular_interaction =  lambda rij,A,L0 :  A * jnp.exp(-rij/L0)

def ABP_pair(Xi,Xj,theta_pair):
    epsilon, R0, A, L0 = theta_pair
    dx,dy,dtheta = Xj[0]-Xi[0],Xj[1]-Xi[1],Xj[2]-Xi[2]
    rij = (dx**2 + dy**2)**0.5
    radial = radial_interaction(rij,epsilon, R0)/(rij+1e-14)
    angular = angular_interaction(rij,A, L0) * jnp.sin(dtheta)
    return  jnp.array([radial*dx,radial*dy,angular]) 


def initial_position(W):
    # Initial position: 2D square crystal with 2*R0 spacing
    x = jnp.linspace(-W*R0, W*R0, W)
    y = jnp.linspace(-W*R0, W*R0, W)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel(), jnp.zeros_like(xx.ravel())], axis=-1)

# The custom class for these simulations
import SFI
# Create an uninitialized instance of the simulated system
ABPmodel = SFI.SFI_Langevin.ParticlesOverdampedLangevinProcess(harmonic_active, ABP_pair, theta_single, theta_pair, Dmatrix)



def animate_ABPs(L, filename='ABP_animation.mp4', fps=30, max_duration=100, tail_length=30, dpi=100, frame_skip=0):
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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    data = np.array(L.X[::frame_skip+1])

    fig, ax = plt.subplots(figsize=(6,6))
    margin = 0.1
    max_abs_coord = np.abs(data[:, :, :2]).max()
    bound = max_abs_coord * (1 + margin)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal')
    #ax.set_title("Active Brownian Particles Simulation")
    plt.tight_layout()

    num_particles = data.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_particles))

    #scatter = ax.scatter(data[0, :, 0], data[0, :, 1], c=colors)

    line_data = [np.zeros((tail_length, 2)) for _ in range(num_particles)]
    lines = LineCollection(line_data, colors=colors, alpha=0.5)
    ax.add_collection(lines)

    arrow_length = 3.
    quiver = ax.quiver(data[0, :, 0], data[0, :, 1], 
                       arrow_length * np.cos(data[0, :, 2]), 
                       arrow_length * np.sin(data[0, :, 2]), 
                       color=colors, scale=1, scale_units='xy', angles='xy', width=0.01,minlength=0)

    def update(frame):
        # Update scatter
        #scatter.set_offsets(data[frame, :, :2])

        # Update lines
        start = max(0, frame - tail_length)
        segments = data[start:frame+1, :, :2].transpose(1, 0, 2)
        lines.set_segments(segments)

        # Update quiver
        positions = data[frame, :, :2]
        angles = data[frame, :, 2]
        quiver.set_offsets(positions)
        quiver.set_UVC(arrow_length * np.cos(angles), arrow_length * np.sin(angles))

        #return scatter, lines, quiver
        return lines, quiver


    num_frames = min(int(max_duration * fps), len(data))
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=lambda: ( lines, quiver), blit=True)

    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)

    return filename
