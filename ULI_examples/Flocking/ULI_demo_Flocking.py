import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
import jax.numpy as jnp
import pickle
from jax import random,jit,vmap

"""

A demo of Stochastic Force Inference, on the example of active
Brownian particles.

Here the interactions (alignment and repulsion) are recovered in a
"model-free" way using a linear combination of generic radial kernels
(here Gaussians).

  """

# Import the package:
import SFI
key= random.PRNGKey(0)

# We load pre-simulated "ideal" data :
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.load_trajectory_csv('Flocking_data_small.csv')
data_exact = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices)


# We degrade the data to make it more similar to experiments:
downsample = 20
ROI = 10000
data_loss_fraction = 0.
noise_level = 0.
blur = 0
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.degrade_data(metadata, particle_indices, time_indices, xvals,
                                                                             downsample = downsample,
                                                                             noise = noise_level,
                                                                             motion_blur = blur,
                                                                             ROI = ROI
                                                                             )


data = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices,compute_dXplus=True)
print("Loaded data. Number of exploitable points:",data.Nparticles.sum())


################################################################
##### II. Perform SFI.                                    ######



# Choose the radial kernels with which to fit the pair interaction -
# here Npts Gaussians
Rmax = 8
Nkernels = 10
sigma=1. * Rmax / Nkernels
r0values = jnp.linspace(0,Rmax,Nkernels)

Gaussian_kernels = lambda r : jnp.exp(-(r-r0values)**2/(2*sigma))

@jit
def one_point_functions(X,V) :
    return jnp.array([ X, V, jnp.sum(jnp.square(V)) * V ])


@jit
def pair_functions(Xi,Xj,Vi,Vj):
    rij = jnp.linalg.norm(Xi-Xj)
    kernel_values = Gaussian_kernels(rij)
    return jnp.concat([ kernel_values[:,None] * (Xi-Xj)[None,:] / rij , - kernel_values[:,None] * (Vi-Vj)[None,:] ])


force_b, force_grad_b_x, force_grad_b_v = SFI.ULI_bases.ULI_pair_interaction_basis(pair_functions,one_point_functions)

S = SFI.UnderdampedLangevinInference(data)
S.compute_diffusion_constant(method = 'WeakNoise')

S.infer_force_linear(basis_linear =  force_b,basis_linear_grad_v=force_grad_b_v,
                     #G_mode='rectangle',
                     G_mode='trapeze',
                     #G_mode='shift',
                     #G_mode='doubleshift', 
                     diffusion_method = 'WeakNoise'
                     #diffusion_method = 'noisy'
                     #diffusion_method = 'MSD'
                     )


# Load the exact model for comparison with inference results:
import Flocking_model 
Flocking_model.FlockingModel.initialize(data.X[0],data.dX[0]/data.dt)
S.compare_to_exact(data_exact=data_exact,force_exact=Flocking_model.FlockingModel.F,diffusion_exact = Flocking_model.FlockingModel.D)
S.compute_force_error() 
S.print_report()
S.compare_to_exact(data_exact=data_exact,force_exact=Flocking_model.FlockingModel.F,diffusion_exact = Flocking_model.FlockingModel.D)

S.sparsify_force()
S.compute_force_error() 
S.print_report()
S.compare_to_exact(data_exact=data_exact,force_exact=Flocking_model.FlockingModel.F,diffusion_exact = Flocking_model.FlockingModel.D)



################################################################
##### III. Plot the results and compare to exact fields.   #####

# Prepare Matplotlib:
import matplotlib.pyplot as plt
fig_size = [12,8]
params = {'axes.labelsize': 12,
          'font.size':   12,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
plt.clf()
fig = plt.figure(1)
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.94, wspace=0.35, hspace=0.3)
H,W = 2,3


 # Plot the trajectory (x and y values):
plt.subplot(H,W,1)
SFI.SFI_plotting_toolkit.plot_process(data)
plt.gca().axis('off')


# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,2)

plt.plot(data.t,data.X[:,0,0],label=r"$x(t)$")
plt.plot(data.t,data.X[:,0,1],label=r"$y(t)$")
plt.plot(data.t,data.X[:,0,2],label=r"$z(t)$")
plt.ylabel(r"one particle's coordinates")
plt.legend()
plt.xlabel(r"$t$")

plt.subplot(H,W,3)

# Histogram to determine which values of the inferred kernel make sense - here we exclude the 10 closest particle pairs.
# Consider downsampling the histogram if this is too slow.

freq = max(1,data.X.shape[0]//20)
r_hist = jnp.sort(vmap(lambda X: jnp.array([ jnp.linalg.norm(Xj[:2]-Xi[:2]) for i,Xi in enumerate(X) for Xj in X[:i] ]))(data.X[::freq]).flatten())
rmin_hist = r_hist[10]
rmax_hist = r_hist[-10]


# Select the fitting coefficients on the radial kernels and reconstruct the kernel
coeffs = S.force_coefficients_full
Fradial = lambda r : Gaussian_kernels(r).dot(coeffs[-2*Nkernels:-Nkernels])

rmax = 20
rvals = jnp.linspace(0,rmax,500)
plt.plot(rvals,vmap(Fradial)(rvals),label='inferred')
plt.plot(rvals,Flocking_model.central_kernel(rvals,Flocking_model.epsilon_central,Flocking_model.R0_central),label='exact',lw=2)

plt.ylim(-3,3.)
plt.xlim(0,10)
plt.grid()

plt.xlabel(r"inter-particle distance $r$",fontsize=12)
plt.ylabel(r"radial force $F(r)$",fontsize=12)

plt.legend(loc=0,frameon=False,fontsize=12)

plt.axvspan(0,rmin_hist,color='0.8')
plt.axvspan(rmax_hist,10*rmax,color='0.8')

plt.subplot(H,W,5)

# Select the fitting coefficients on the angular kernels
Fangular = lambda r : Gaussian_kernels(r).dot(coeffs[-Nkernels:])

rmax = 20
rvals = jnp.linspace(0,rmax,500)
plt.plot(rvals,vmap(Fangular)(rvals),label='inferred')
plt.plot(rvals,Flocking_model.alignment_kernel(rvals,Flocking_model.epsilon_align,Flocking_model.R0_align),label='exact',lw=2)

plt.ylim(-1,3.)

plt.yticks([0,3])
plt.xlim(0,10)
plt.grid()

plt.xlabel(r"inter-particle distance $r$",fontsize=12)
plt.ylabel(r"velocity alignment kernel $T(r)$",fontsize=12)

plt.legend(loc=0,frameon=False,fontsize=12)
plt.axvspan(0,rmin_hist,color='0.8')
plt.axvspan(rmax_hist,10*rmax,color='0.8')

plt.subplot(H,W,4)
SFI.SFI_plotting_toolkit.comparison_scatter(S.exact_force_values,S.ansatz_force_values,alpha=0.1,y=0.8,error=S.force_predicted_MSE**0.5)
plt.xlabel(r"exact $F_\mu(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $F_\mu(\mathbf{x})$",labelpad=0)


# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.
plt.subplot(H,W,6)
key, subkey = random.split(key)
Y = S.simulate_bootstrapped_trajectory(subkey,oversampling=10)
data_bootstrap = SFI.StochasticTrajectoryData(*Y.export_trajectory())
SFI.SFI_plotting_toolkit.plot_process(data_bootstrap,cmap='magma')
plt.gca().axis('off')

plt.show()


