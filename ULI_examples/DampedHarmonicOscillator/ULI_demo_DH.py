import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
import jax.numpy as jnp
import pickle
from jax import random,jit,vmap

# Import the package:
import SFI
key= random.PRNGKey(0)

# We load pre-simulated "ideal" data and use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.load_trajectory_csv('DH_data_small.csv') 
data_exact = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices,compute_dXplus=True)


# We degrade the data to make it more similar to experiments:
downsample = 1
noise_level = 0.0001
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.degrade_data(metadata, particle_indices, time_indices, xvals,
                                                                             downsample = downsample,
                                                                             noise = noise_level,
                                                                             )
data = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices,compute_dXplus=True)

print("Loaded data. Number of exploitable points:",data.Nparticles.sum())


################################################################
##### II. Perform SFI.                                    ######
S = SFI.UnderdampedLangevinInference(data)
S.compute_diffusion_constant(method = 'WeakNoise')


(force_b, force_grad_b_x, force_grad_b_v), names = SFI.ULI_bases.basis_selector({ 'type' : 'polynomial', 'order' : 1, 'mode' : 'both'},data.d,output="vector")

S.infer_force_linear(basis_linear =  force_b,basis_linear_grad_v=force_grad_b_v,
                     #M_mode = 'anticipated',
                     #M_mode = 'early',
                     M_mode = 'symmetric',
                     #G_mode='rectangle',
                     G_mode='shift', 
                     #diffusion_method = 'WeakNoise'
                     diffusion_method = 'noisy',
                     #diffusion_method = 'MSD'
                     basis_names = names
                     )

S.compute_force_error() 
S.print_report()

from DH_model import *
DH_Model.initialize(data_exact.X[0],data_exact.dX[0]/data_exact.dt,jnp.array(metadata['params_F']))
S.compare_to_exact(data_exact=data_exact,force_exact=DH_Model.F,diffusion_exact = DH_Model.D)

# Apply sparsification:
S.sparsify_force()
S.compute_force_error() 
S.print_report()

S.compare_to_exact(data_exact=data_exact,force_exact=DH_Model.F,diffusion_exact = DH_Model.D)



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
plt.title(f"Trajectory in XY plane")
plt.gca().axis('off')

# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,2)
plt.plot(data.t,data.X[:,0,0],label=r"$x(t)$")
plt.plot(data.t,data.X[:,0,1],label=r"$y(t)$")
plt.plot(data.t,data.X[:,0,2],label=r"$\theta(t)$")
plt.ylabel(r"one particle's coordinates")
plt.legend()
plt.xlabel(r"$t$")


plt.subplot(H,W,3)
SFI.SFI_plotting_toolkit.comparison_scatter(S.exact_force_values,S.ansatz_force_values,alpha=0.1,y=0.8,error=S.force_predicted_MSE**0.5)
plt.xlabel(r"exact $F_\mu(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $F_\mu(\mathbf{x})$",labelpad=0)

plt.subplot(H,W,4)
SFI.SFI_plotting_toolkit.comparison_scatter(S.exact_diffusion_values,S.ansatz_diffusion_values,alpha=0.1,y=0.8)
plt.xlabel(r"exact $D_{\mu\nu}(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $D_{\mu\nu}(\mathbf{x})$",labelpad=0)


plt.suptitle("Underdamped Langevin Inference demo")
plt.show()



