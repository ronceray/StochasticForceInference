import numpy as np

"""

A demo of Stochastic Force Inference, on the example of active
Brownian particles.

"""

# Import the package:
from StochasticForceInference import *

 
################################################################
##### I. Prepare the data (here using a simulated model). ######


np.random.seed(1)

omega = 0.2
Factive = 1.
D = 1.
Dangular = 0.1
W = 5

N = 1000
dt = 0.1
tau = dt * (N-1)
tlist = np.linspace(0.,tau,N)

oversampling = 10
prerun = 2**7

epsilon = 5.
R0 = 1.

def pair(R0,epsilon):
    def pair_interaction(rij):
        return epsilon * 1/((rij/R0)**2+1)
    return pair_interaction


from SFI_langevin import ParticlesOverdampedLangevinProcess
class ActiveBrownianParticles(ParticlesOverdampedLangevinProcess):
    def __init__(self,omega,pair_interaction,Factive,D,Dangular,R0,W,tlist,prerun=0,oversampling=1):
        # Initial position: 2D square crystal with 2*R0 spacing
        initial_position = np.array([ [x,y, 2*np.pi*np.random.random()] for x in np.linspace(-W*R0,W*R0,W) for y in np.linspace(-W*R0,W*R0,W) ])
        
        # Harmonic trapping:
        harmonic_active = lambda X : np.array([ - omega * X[0] + Factive * np.cos(X[2]), - omega * X[1] + Factive * np.sin(X[2]), 0. ])

        def pair(Xi,Xj):
            dx,dy = Xi[0]-Xj[0],Xi[1]-Xj[1]
            rij = (dx**2 + dy**2)**0.5
            return pair_interaction(rij) * np.array([dx,dy,0]) /rij

        Dmatrix = np.array([[ D,0,0],
                            [ 0,D,0],
                            [ 0,0,Dangular]])
        
        ParticlesOverdampedLangevinProcess.__init__( self, harmonic_active, pair, Dmatrix, tlist, initial_position, oversampling = oversampling, prerun = prerun)



pair_interaction = pair(R0,epsilon)
X = ActiveBrownianParticles(omega,pair_interaction,Factive,D,Dangular,R0,W,tlist,prerun=prerun,oversampling=oversampling)

# Possibly blur a bit the data to mimic noise from the measurement
# device:
noise_amplitude = 0.0
noise = noise_amplitude * np.random.normal(size=X.data.shape)

# The input of the inference method is the "xlist" array, which has
# shape Nsteps x 1 x dim (the middle index is used for multiple
# particles with identical properties; we do not use it in this demo).
xlist = X.data + noise

freq = 1
# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
data = StochasticTrajectoryData(xlist[::freq],tlist[::freq])   

center = data.X_ito.mean(axis=(0,1)) 
width  =  2.1 * abs(data.X_ito-center).max(axis=(0,1)) 


freq = 1
# We use a wrapper class, StochasticTrajectoryData, to format the data
# in a way that the inference methods can use.
data = StochasticTrajectoryData(xlist[::freq],tlist[::freq])   

center = data.X_ito.mean(axis=(0,1)) 
width  =  2.1 * abs(data.X_ito-center).max(axis=(0,1)) 



################################################################
##### II. Perform SFI.                                    ######

S = StochasticForceInference(data)  


# Choose the radial kernels with which to fit the pair interaction -
# here Npts Gaussians
r0 = 2.
Rmax = 10
Npts = 5
sigma=2.
def Gaussian_kernel(sigma,r0):
    # Factory to limit the scope of k
    return lambda r : np.exp(-(r-r0)**2/(2*sigma))
kernels = [ Gaussian_kernel(sigma,r0) for r0 in np.linspace(0,Rmax,Npts) ]

S.compute_drift(
    basis={ 'type' : 'self_propelled_particles', 'order' : 1, 'kernels' : kernels },\
    diffusion_mode = 'MSD',        # Best for space-dependent noise with short trajectories
    #diffusion_mode = 'constant',   
    #diffusion_mode = 'Vestergaard', # Best for space-dependent noise with large measurement error 
    #diffusion_mode = 'WeakNoise',  # Best for space-dependent noise with large dt
    #mode='Ito'  # Use only for noise-free data.
) 

S.compute_diffusion(
    #method='Vestergaard',
    method='MSD',
    #method='WeakNoise',
    basis = { 'type' : 'polynomial', 'order' : 0}
) 


S.compute_force()
S.compute_drift_error() 
S.compute_diffusion_error()
S.compute_entropy()

S.print_report()

data_exact = StochasticTrajectoryData(X.data,X.t)
S.compare_to_exact(data_exact=data_exact,force_exact=X.F,D_exact = X.D)



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
data.plot_particles(0,1,X = X.data[-1],t=-1,colored=False,active=True,u=-0.35)

Fscale = 2
# The exact forces at time t:
data.plot_particles_field(X.F,X=X.data[-1],alpha=1,zorder=0,color='k',width=0.15,scale = Fscale)
# The inferred forces:
data.plot_particles_field(S.F_ansatz,X=X.data[-1],t=-1,zorder=-1,color='b',alpha=0.3,width=0.3,scale = Fscale)

# Plot the whole trajectory (all components vs t):
plt.subplot(H,W,2)

plt.plot(data.t,data.X_ito[:,0,0],label=r"$x(t)$")
plt.plot(data.t,data.X_ito[:,0,1],label=r"$y(t)$")
plt.plot(data.t,data.X_ito[:,0,2],label=r"$\theta(t)$")
plt.ylabel(r"one particle's coordinates")
plt.legend()
plt.xlabel(r"$t$")

plt.subplot(H,W,3)

Nfuncs = len(kernels)
# Select the fitting coefficients on the radial kernels, and take the
# isotropic part:
Fmunu = S.phi_coefficients[:2,5:].reshape(2,Nfuncs,2)
Fradial = np.einsum('mkm->k',Fmunu)/2.
F = lambda r : sum( -kernels[i](r) * Fradial[i] for i in range(Nfuncs) )

rmax = 20
rvals = np.linspace(0,rmax,500)
plt.plot(rvals,F(rvals),label='inferred')
plt.plot(rvals,pair_interaction(rvals),label='exact',lw=2)

plt.ylim(0.,3.)

plt.yticks([0,3])
plt.xlim(0,10)
#plt.grid()

plt.xlabel(r"inter-particle distance $r$",fontsize=12)
plt.ylabel(r"radial force $F(r)$",fontsize=12)

plt.legend(loc=0,frameon=False,fontsize=12)

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
data_bootstrap.plot_particles(0,1,X = X.data[-1],t=-1,colored=False,active=True,u=-0.35)
plt.gca().axis('off')

plt.suptitle("Stochastic Force Inference demo")
plt.show()
