import numpy as np

"""

A demo of Stochastic Force Inference, on the example of the stochastic
Lorenz process.

"""

# Import the package:
from StochasticForceInference import *

 
################################################################
##### I. Prepare the data (here using a simulated model). ######


# Diffusion parameters: a linear diffusion gradient (multiplicative noise)
dim=3
diffusion_coeff = 1.
D = diffusion_coeff *  np.identity(dim) 

y = np.array([1.,2,1])
D = lambda X : np.array([ np.identity(dim) + (np.einsum('m,n->mn',x,y)+np.einsum('m,n->mn',y,x))*0.05 for x in X ])


# Force field parameters (stochastic Lorenz process)
r,b,s = 6.,1.,3.

force = lambda X : np.array([[ s*(x[2]-x[0]),
                               x[0]*x[2]-b*x[1],
                               r*x[0] - x[2] - x[1]*x[0]] for x in X ])

# Note: the "for" loop runs over particles/copies of the simulation;
#   it is not used here.

# Simulation parameters
initial_position = np.array([[0.1 for i in range(dim)]]) 
dt = 0.01
oversampling = 4
prerun = 100
Npts = 10000
tau = dt * Npts
tlist = np.linspace(0.,tau,Npts)

# Run the simulation using our OverdampedLangevinProcess class
np.random.seed(1)
X = OverdampedLangevinProcess(force,D,tlist,initial_position=initial_position,oversampling=oversampling,prerun=prerun )

# Possibly blur a bit the data to mimic noise from the measurement
# device:
noise_amplitude = 0.0
noise = noise_amplitude * np.random.normal(size=X.data.shape)

# The input of the inference method is the "xlist" array, which has
# shape Nsteps x 1 x dim (the middle index is used for multiple
# particles with identical properties; we do not use it in this demo).
xlist = X.data + noise
tlist = X.t
# You can replace "xlist" and "tlist" by your own data!

# Motion blur simulator:
#xlist = (0.5*(X.data[1:] + X.data[:-1]) + noise[:-1])
#tlist = (X.t[:-1])


freq = 1
# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
data = StochasticTrajectoryData(xlist[::freq],tlist[::freq])   

center = data.X_ito.mean(axis=(0,1)) 
width  =  2.1 * abs(data.X_ito-center).max(axis=(0,1)) 



################################################################
##### II. Perform SFI.                                    ######


S = StochasticForceInference(data)  

S.compute_drift(basis = { 'type' : 'polynomial', 'order' : 2} ,
                #basis = { 'type' : 'Fourier', 'order' : 3, 'center' : center, 'width' : width, } ,
                #diffusion_mode = 'WeakNoise',  # Best for space-dependent noise with large dt
                diffusion_mode = 'MSD',        # Best for space-dependent noise with short trajectories
                #diffusion_mode = 'constant',   
                #diffusion_mode = 'Vestergaard', # Best for space-dependent noise with large measurement error 
                #mode='Ito'
) 


S.compute_diffusion(
    #method='Vestergaard',
    method='MSD',
    #method='WeakNoise',
    basis = { 'type' : 'polynomial', 'order' : 1}
) 

S.compute_force()
S.compute_drift_error() 
S.compute_diffusion_error()
S.compute_entropy()

S.print_report()

data_exact = StochasticTrajectoryData(X.data,X.t)
S.compare_to_exact(data_exact=data_exact,force_exact=force,D_exact = D)



################################################################
##### III. Plot the results and compare to exact fields.   #####

# Prepare Matplotlib:
import matplotlib.pyplot as plt
fig_size = [8,6]
params = {'axes.labelsize': 12,
          'font.size':   12,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
plt.clf()
fig = plt.figure(1)
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.94, wspace=0.35, hspace=0.3)
H,W = 2,3

# Plot the trajectory (x and y values):
plt.subplot(H,W,1)
data.plot_process(width=0.2)
plt.gca().axis('off')

# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,2)
plt.plot(data.t,data.X_ito[:,0,:])
plt.ylabel(r"$X_\mu(t)$")
plt.xlabel(r"$t$")


# Plot a slice of the force field - blue is inferred, black is the
# exact one used to generate the data.
plt.subplot(H,W,3)
data.plot_field(field=S.F_ansatz,color='b',alpha=0.4,zorder=0,width = 0.2,autoscale=True) 
data.plot_field(field=X.F,color='k',alpha=1,zorder=-1,width=0.08,autoscale=True) 



plt.subplot(H,W,4)
SFI_plotting_toolkit.comparison_scatter(S.exact_F_Ito,S.ansatz_F_Ito,alpha=0.1,y=0.8,error=S.drift_projections_self_consistent_error**0.5)
plt.xlabel(r"exact $F_\mu(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $F_\mu(\mathbf{x})$",labelpad=0)

plt.subplot(H,W,5)
SFI_plotting_toolkit.comparison_scatter(S.exact_D,S.ansatz_D,alpha=0.3,y=0.8,error=S.diffusion_projections_self_consistent_error**0.5)
plt.xlabel(r"exact $D_{\mu\nu}(\mathbf{x})$",labelpad=-1)
plt.ylabel(r"inferred $D_{\mu\nu}(\mathbf{x})$",labelpad=0)



# Use the inferred force and diffusion fields to simulate a new
# trajectory with the same times list, and plot it.
plt.subplot(H,W,6)
Y = S.simulate_bootstrapped_trajectory(oversampling=10)
data_bootstrap = StochasticTrajectoryData(Y.data,Y.t)
data_bootstrap.plot_process(width=0.15,cmap='magma')
plt.gca().axis('off')

plt.suptitle("Stochastic Force Inference demo")
plt.show()
