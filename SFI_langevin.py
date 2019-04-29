
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm


class OverdampedLangevinProcess(object):
    """A simple class to simulate overdamped Langevin processes.

    Both the force and the diffusion matrix are given as a lambda
    function that depend only on the position. Note that the force is
    assumed to include the mobility matrix; more precisely the
    Langevin dynamics implemented here is:

    dX/dt = F(X) + div D + \sqrt{2 D} xi

    where xi is gaussian white noise, and D is taken at time t (Ito
    convention).

    If div D is provided as a lambda function, it will be used;
    otherwise this "spurious force" will be computed through finite
    differences. It can also be bypassed if D is constant using the
    argument D_is_constant.

    The resulting simulated data is a 3D array:
    - 1st index is time index
    - 2nd is an index for multiple independent copies of the process,
      or, in the case of interacting particles systems, a particle
      index. Note that we haven't implemented multi-particles
      multi-copies simulations: independent copies are considered as
      non-interacting particles. 
    - 3rd is spatial/phase space index

    The optional argument "oversampling" allows to simulate
    intermediate points but only record a fraction of them, in cases
    where the dynamics is sensitive to the integration timestep.

    The optional argument "prerun" allows to simulate an initial
    equilibration phase (number of time steps, using the first time
    interval for dt).

    """
    
    def __init__(self, F, D, tlist, initial_position, oversampling = 1, prerun = 0, divD = None, D_is_constant = True):
        self.F = F
        self.Nparticles,self.d = initial_position.shape

        if D_is_constant:
            Dc = 1.*D
            if Dc.shape == (self.d,self.d):
                Dc = np.array([ Dc for i in range(self.Nparticles)])
            elif Dc.shape != (self.Nparticles,self.d,self.d):
                raise ValueError("Diffusion matrix shape is wrong:",Dc.shape," while dim is:",self.d)
            self.__D__ = Dc
            self.D = lambda X : self.__D__
            self.__sqrt2D__ = np.array([ sqrtm(2 * self.__D__[i,:,:]) for i in range(self.Nparticles) ])
            self.sqrt2D = lambda X : self.__sqrt2D__
            self.divD = lambda X : np.zeros((self.Nparticles,self.d))
        else:
            self.D = D
            if self.d >= 2:
                def sqrt2D(X):
                    D = self.D(X)
                    return np.array([ sqrtm(2 * D[i,:,:]) for i in range(self.Nparticles) ])
            else:
                def sqrt2D(X):
                    return (2*self.D(X))**0.5
            self.sqrt2D = sqrt2D
                
            if divD is None:
                epsilon = 1e-6
                self.__dx__ = [[ np.array([[ 0 if (i,m)!= (ind,mu) else epsilon for m in range(self.d)] for i in range(self.Nparticles) ] )\
                                  for mu in range(self.d) ] for ind in range(self.Nparticles) ] 
                def divD(x):
                    return np.einsum('jmimn->in',  np.array([[ (self.D(x+self.__dx__[ind][mu]) - self.D(x-self.__dx__[ind][mu]))/(2*epsilon) \
                                                               for mu in range(self.d)] for ind in range(self.Nparticles) ]))
                self.divD = divD
                
            else:
                self.divD = divD
                
        self.t = tlist

        self.Ntimesteps = len(self.t) 
        self.dt = self.t[1:] - self.t[:-1]
        
        self.simulate(initial_position,oversampling,prerun)
        self.compute_entropy_production()

    def dx(self,state,dt):
        """ The position increment in time dt."""
        return dt * self.F(state)  +  dt * self.divD(state)  +  np.einsum('imn,in->im',self.sqrt2D(state), np.random.normal(size=(self.Nparticles,self.d)) ) * dt**0.5 
        
    def simulate(self,initial_position,oversampling,prerun):
        state = 1. * initial_position
        # pre-equilibration:
        dt = self.dt[0] / oversampling
        for j in range( prerun*oversampling ):
            state += self.dx(state,dt)

        # Start recording:
        self.data = np.zeros((self.Ntimesteps,self.Nparticles,self.d))
        for i,delta_t in enumerate(self.dt):
            self.data[i,:,:] = state
            dt = delta_t / oversampling
            for j in range( oversampling ):
                state += self.dx(state,dt)

        self.data[-1,:,:] = state

        
    def compute_entropy_production(self):
        """Entropy production ("heat dissipated along the trajectory") and
        information. These are quantities per particle / per copy.

        """
        self.S = sum([ np.einsum('im,in,imn->', self.data[t+1,:,:]-self.data[t,:,:], self.F( 0.5*(self.data[t+1,:,:]+self.data[t,:,:]) ), np.linalg.inv(self.D(0.5*(self.data[t+1,:,:]+self.data[t,:,:])))) for t in range(self.Ntimesteps-1)])
        self.I = 0.25 * sum([ np.einsum('im,in,imn->', self.data[t+1,:,:]-self.data[t,:,:], self.F(self.data[t,:,:]), np.linalg.inv(self.D(self.data[t,:,:]))) for t in range(self.Ntimesteps-1)])/self.Nparticles

                       
    def plot_process_axes(self,a,beta,tmin=0,tmax=-1,particle_indices = None,cmap='viridis',**kwargs):
        for i in ( range(self.Nparticles) if particle_indices is None else particle_indices ):
            x,y = self.data[tmin:tmax,a,i],self.data[tmin:tmax,beta,i]
            plt.quiver(x[:-1],y[:-1],x[1:]-x[:-1],y[1:]-y[:-1],list(self.t[tmin:tmax-1]), cmap = cmap,headwidth = 1.0,headlength = 0.0, headaxislength=0., scale = 1.0,units = 'xy',lw=0.,**kwargs)
        plt.axis('equal') 
        plt.xticks([]) 
        plt.yticks([])



class ParticlesOverdampedLangevinProcess(OverdampedLangevinProcess):
    """ Simulate overdamped Langevin processes with pair
    interactions between identical particles.
    """
    
    def __init__(self, force_single, force_pair, D, tlist, initial_position, oversampling = 1, prerun = 0):
        self.Fsingle = force_single
        self.Fpair = force_pair
        self.D = D
        self.t = tlist
        Nparticles,dim = initial_position.shape
        def force(X):
            return np.array([ self.Fsingle(X[i,:]) for i in range(Nparticles) ]) + \
                   np.array([ np.sum(np.array([ self.Fpair(X[i,:],X[j,:]) for j in range(Nparticles) if i != j ]),axis=0) for i in range(Nparticles) ])

        OverdampedLangevinProcess.__init__(self,force, D, tlist, initial_position, oversampling=oversampling,prerun=prerun)


