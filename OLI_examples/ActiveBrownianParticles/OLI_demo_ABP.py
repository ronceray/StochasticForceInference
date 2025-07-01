import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Ensure JAX only uses the CPU, if your GPU doesn't have enough memory
import jax.numpy as jnp
import pickle
from jax import random,jit,vmap
import SFI

"""

A demo of Stochastic Force Inference, on the example of active
Brownian particles.

Here the interactions (alignment and repulsion) are recovered in a
"model-free" way using a linear combination of generic radial kernels
(here Gaussians).

"""

# Import the package:
key= random.key(0)

# We load pre-simulated "ideal" data :
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.load_trajectory_csv('ABP_data_medium.csv')
data_exact = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices)


# We degrade the data to make it more similar to experiments:
downsample = 1
ROI = 2000
data_loss_fraction = 0.001
noise_level = jnp.array([ 0., 0., 0.0 ])
blur = 0
metadata, particle_indices, time_indices, xvals = SFI.SFI_utils.degrade_data(metadata, particle_indices, time_indices, xvals,
                                                                             downsample = downsample,
                                                                             noise = noise_level,
                                                                             motion_blur = blur,
                                                                             data_loss_fraction = data_loss_fraction,
                                                                             ROI = ROI
                                                                             )


data = SFI.StochasticTrajectoryData(xvals,time_indices,metadata['dt'],particle_indices=particle_indices)
print("Loaded data. Number of exploitable points:",data.Nparticles.sum())


################################################################
##### II. Perform SFI.                                    ######

S = SFI.OverdampedLangevinInference(data)

S.compute_diffusion_constant(method='Vestergaard'
                             #method='MSD'
                             )


# Choose the radial kernels with which to fit the pair interaction -
# here Npts Gaussians
Rmax = 10
Npts = 12
sigma=1.
def Gaussian_kernel(sigma,r0):
    # Factory to limit the scope of k
    return lambda r : jnp.exp(-(r-r0)**2/(2*sigma))
kernels = [ Gaussian_kernel(sigma,r0) for r0 in jnp.linspace(0,Rmax,Npts) ]


force_b, force_grad_b = SFI.OLI_bases.basis_selector({ 'type' : 'self_propelled_particles', 'polynomial_order' : 1, 'kernels_radial' : kernels, 'kernels_angular' : kernels },data.d,output="vector")
force_b, force_grad_b = jit(force_b), jit(force_grad_b)

S.infer_force_linear(basis_linear =  force_b,basis_linear_gradient=force_grad_b) 


S.compute_force_error() 
#S.compute_diffusion_error()
#S.compute_entropy()

S.print_report()

# Load the exact model for comparison with inference results:
import ABP_model 
ABP_model.ABPmodel.initialize(data.X[0])
S.compare_to_exact(data_exact=data_exact,force_exact=ABP_model.ABPmodel.F,diffusion_exact = ABP_model.ABPmodel.D)

S.sparsify_force()
S.compute_force_error() 
S.print_report()
S.compare_to_exact(data_exact=data_exact,force_exact=ABP_model.ABPmodel.F,diffusion_exact = ABP_model.ABPmodel.D)


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
SFI.SFI_plotting_toolkit.plot_particles(data,0,1,t=-1,colored=False,active=True,u=-0.35)


# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,2)
plt.plot(data.t,data.X[:,0,0],label=r"$x(t)$")
plt.plot(data.t,data.X[:,0,1],label=r"$y(t)$")
plt.plot(data.t,data.X[:,0,2],label=r"$\theta(t)$")
plt.ylabel(r"one particle's coordinates")
plt.legend()
plt.xlabel(r"$t$")

plt.subplot(H,W,3)

# Histogram to determine which values of the inferred kernel make sense - here we exclude the 10 closest particle pairs.
# Consider downsampling the histogram if this is too slow.

freq = max(1,data.X.shape[0]//20)
r_hist = sorted(vmap(lambda X: jnp.array([ jnp.linalg.norm(Xj[:2]-Xi[:2]) for i,Xi in enumerate(X) for Xj in X[:i] ]))(data.X[:10:freq]).flatten())
rmin = r_hist[10]


Nfuncs = len(kernels)
# Select the fitting coefficients on the radial kernels, and take the
# isotropic part:
F = lambda r : sum( kernels[i](r) * S.force_coefficients_full[-2*Nfuncs + i] for i in range(Nfuncs) )

rmax = 20
rvals = jnp.linspace(0,rmax,500)
plt.plot(rvals,F(rvals),label='inferred')
plt.plot(rvals,ABP_model.radial_interaction(rvals,ABP_model.epsilon,ABP_model.R0),label='exact',lw=2)

plt.ylim(-3,3.)

#plt.yticks([-3,1])
plt.xlim(0,10)
plt.grid()

plt.xlabel(r"inter-particle distance $r$",fontsize=12)
plt.ylabel(r"radial force $F(r)$",fontsize=12)

plt.legend(loc=0,frameon=False,fontsize=12)

plt.axvspan(0,rmin,color='0.8')

plt.subplot(H,W,5)

# Select the fitting coefficients on the angular kernels
Fangular = lambda r : sum( kernels[i](r)/r * S.force_coefficients_full[-Nfuncs + i] for i in range(Nfuncs) )

rmax = 20
rvals = jnp.linspace(0,rmax,500)
plt.plot(rvals,Fangular(rvals),label='inferred')
plt.plot(rvals,ABP_model.angular_interaction(rvals,ABP_model.A,ABP_model.L0),label='exact',lw=2)

plt.ylim(-1,3.)

plt.yticks([0,3])
plt.xlim(0,10)
plt.grid()

plt.xlabel(r"inter-particle distance $r$",fontsize=12)
plt.ylabel(r"angular torque kernel $T(r)$",fontsize=12)

plt.legend(loc=0,frameon=False,fontsize=12)
plt.axvspan(0,rmin,color='0.8')



plt.subplot(H,W,4)
SFI.SFI_plotting_toolkit.comparison_scatter(S.exact_force_values,S.ansatz_force_values,alpha=0.1,y=0.8,error=S.force_predicted_MSE**0.5)
plt.xlabel(r"exact $F_\mu(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $F_\mu(\mathbf{x})$",labelpad=0)


# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.
plt.subplot(H,W,6)
Y = S.simulate_bootstrapped_trajectory(oversampling=10,key=key)
data_bootstrap = SFI.StochasticTrajectoryData(*Y.export_trajectory())
SFI.SFI_plotting_toolkit.plot_process(data_bootstrap,cmap="magma")
plt.gca().axis('off')
SFI.SFI_plotting_toolkit.plot_particles(data_bootstrap,0,1,t=-1,colored=False,active=True,u=-0.35)

plt.suptitle("Stochastic Force Inference demo")
plt.show()



