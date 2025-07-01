import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
import jax.numpy as jnp
import pickle
from jax import random,jit,vmap

# Import the package:
import SFI
key= random.key(0)

# We load pre-simulated "ideal" data :
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.load_trajectory_csv('LV_data_small.csv') 
data_exact = SFI.StochasticTrajectoryData(xvals,time_indices,metadata["dt"],particle_indices=particle_indices)


# We degrade the data to make it more similar to experiments:
downsample = 200
noise_level = 0.005
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.degrade_data(metadata, particle_indices, time_indices, xvals,
                                                                             downsample = downsample,
                                                                             noise = noise_level,
                                                                             )

data = SFI.StochasticTrajectoryData(xvals,time_indices,metadata["dt"],particle_indices=particle_indices)

# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
print("Loaded data. Number of exploitable points:",data.Nparticles.sum())


################################################################
##### II. Perform SFI.                                    ######

S = SFI.OverdampedLangevinInference(data,verbosity=1,max_memory_gb=1.)
S.compute_diffusion_constant(#method='Vestergaard' # Best with large measurement error 
                              #method='MSD' # Best with ideal data 
                              method = 'WeakNoise',    # Best with large dt
                             )


poly1,descr = SFI.OLI_bases.polynomial_basis(data.d,1)
polyexp = lambda x : poly1(jnp.exp(x))
(force_b, force_grad_b),names = SFI.OLI_bases.scalar_basis(polyexp,data.d,"vector")


S.infer_force_linear(basis_linear =  force_b,basis_linear_gradient=force_grad_b,
                     #mode='Strato',G_mode='shift',diffusion_method = 'Vestergaard', # Best with large measurement error 
                     #mode='Ito',G_mode='trapeze', # Best with ideal data or large delta t
                     ) 


S.compute_force_error() 
#S.compute_diffusion_error()
S.print_report()

from LV_model import *
LV_Model.initialize(data_exact.X[0],jnp.array(metadata['params_F']))
S.compare_to_exact(data_exact=data_exact,force_exact=LV_Model.F,diffusion_exact = LV_Model.D) 

S.sparsify_force()
print("Results after sparsification:")
S.print_report()
S.compare_to_exact(data_exact=data_exact,force_exact=LV_Model.F,diffusion_exact = LV_Model.D) 


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



# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,1)

plt.plot(data.t,jnp.exp(data.X[:,0,:]))
plt.ylabel(r"$x$")
plt.xlabel(r"$t$")

plt.subplot(H,W,3)
SFI.SFI_plotting_toolkit.comparison_scatter(S.exact_force_values,S.ansatz_force_values,alpha=0.1,y=0.8,error=S.force_predicted_MSE**0.5)
plt.xlabel(r"exact $F_\mu(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $F_\mu(\mathbf{x})$",labelpad=0)

plt.subplot(H,W,4)
plt.imshow(jnp.vstack((r,A.T)),cmap='RdBu',vmin=-2,vmax=2.)

plt.subplot(H,W,5)
coeffs = jnp.zeros(S.force_sparsifier.p)
coeffs = coeffs.at[S.force_support].set(S.force_coefficients).reshape((dim+1,dim),order='F')
plt.imshow(coeffs,cmap='RdBu',vmin=-2,vmax=2.)

# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.
plt.subplot(H,W,2)
key, subkey = random.split(key)
Y = S.simulate_bootstrapped_trajectory(subkey,oversampling=10*downsample)
data_bootstrap = SFI.StochasticTrajectoryData(*Y.export_trajectory())
plt.plot(jnp.arange(Y.X.shape[0])*Y.dt,jnp.exp(Y.X[:,0,:]))
plt.gca().axis('off')


plt.suptitle("Stochastic Force Inference demo")
plt.show()



