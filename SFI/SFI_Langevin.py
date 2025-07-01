import jax.numpy as jnp
from jax import random,jit,lax,vmap
from functools import partial
from datetime import datetime
import time
from SFI.SFI_utils import sqrtm_psd
import SFI
    
class OverdampedLangevinProcess(object):
    """A simple class to simulate overdamped Langevin processes.

    Both the force and the diffusion matrix are given as functions
    that depend only on the system's state X. Note that, here and
    throughout this project, we term "force" the Ito drift. The
    Langevin dynamics implemented here thus reads:

    dX/dt = F(X_t) + sqrt{2 D(X_t)} xi(t)

    where F is the force, D the diffusion matrix (taken at time t,
    i.e. in the Ito convention), and xi is a Gaussian white noise.

    The resulting simulated data is a 3D array:
    - 1st index is time index
    - 2nd is an index for multiple independent copies of the process,
      or, in the case of interacting particles systems, a particle
      index. Note that we haven't implemented multi-particles
      multi-copies simulations: independent copies are considered as
      non-interacting particles. 
    - 3rd is spatial/phase space index

    is_multiparticle indicates whether the 2nd axis represents
    interacting particles (True) or independent copies (False):

    - for independent copies, F (and D if it's a function) is expected
      to take d-dimensional vectors as input (simple simulations)

    - for interacting particles, F takes as input Nparticles x d
      arrays; D can be a constant dxd or Nparticlesxdxd array, or a
      function that takes as input a d-dimensional vector (no
      cross-diffusion: the diffusion is independent between distinct
      particles). TO DO: Cross-diffusion could be added if the need
      arises.

    The optional simulation argument "oversampling" allows to simulate
    intermediate points but only record a fraction of them, in cases
    where the dynamics is sensitive to the integration timestep.

    The optional simulation argument "prerun" allows to simulate an
    initial equilibration phase (number of time steps, using the first
    time interval for dt).

    """
    
    def __init__(self, F : callable, D, F_is_multiparticle = False, D_is_multiparticle = False):
        """
        Force (and diffusion if a function) is provided in a parametric
        manner, lambda X,params_F : F(X,params_F), which can be
        re-used for parametric inference later.

        """
        self.uninitialized_F = F
        self.uninitialized_D = D
        self.F_is_multiparticle = F_is_multiparticle
        self.D_is_multiparticle = D_is_multiparticle
        self.metadata = {}

    def initialize(self, initial_position, params_F, params_D = None):
        if len(initial_position.shape) == 1:
            # Single simulations: add an axis for compatibility
            self.initial_position = initial_position[None, :]
        else:
            # Multiparticles/multicopy simulations
            self.initial_position = initial_position
        self.Nparticles, self.d = self.initial_position.shape

        self.params_F = params_F
        self.params_D = params_D

        # Configure the force function
        if self.F_is_multiparticle:
            self.F = lambda X: self.uninitialized_F(X, self.params_F)
        else:
            self.F = vmap(lambda X: self.uninitialized_F(X, self.params_F))

        # Configure the diffusion function
        if isinstance(self.uninitialized_D, (float, int)):
            # Constant, scalar diffusion
            self.setup_constant_D(self.uninitialized_D * jnp.eye(self.d))
        elif hasattr(self.uninitialized_D, "shape"):
            # Constant, matrix diffusion 
            self.setup_constant_D(self.uninitialized_D)
        elif callable(self.uninitialized_D):
            self.setup_variable_D(lambda X: self.uninitialized_D(X, self.params_D))  # Variable case
        else:
            raise ValueError("Diffusion input not supported.")

        # Store metadata for reproducibility
        self.metadata.update({
            "algorithm": "overdamped",
            "num_particles": self.Nparticles,
            "dimension": self.d,
            "params_F": self.params_F,
            "params_D": self.params_D,
            "initial_position": initial_position.tolist(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def setup_constant_D(self, D):
        self.D = D
        self.sqrt2D = jnp.real(sqrtm_psd(2 * self.D))
        self._constant_D = True

    def setup_variable_D(self, D):
        if self.D_is_multiparticle:
            # Here D : (Nparticles x d)-> (Nparticles x d x d)
            self.D = D
            self.sqrt2D = jit(lambda x : jnp.real(vmap(sqrtm_psd)(2 * D(x))))
        else:
            # Here D : (d)-> (d x d); we use vmap to make it (Nparticles x d)-> (Nparticles x d x d)
            self.D = vmap(D)
            self.sqrt2D = jit(vmap(lambda x : jnp.real(sqrtm_psd(2 * D(x)))))
        self._constant_D = False

    @partial(jit, static_argnums=(0,))
    def _dx(self,state,ddt,key):
        """The position increment in time ddt, given by the Euler-Maruyama
        algorithm.
        """
        if self._constant_D:
            return ddt * self.F(state)  +  jnp.einsum('mn,in->im',self.sqrt2D, random.normal(key,shape=(self.Nparticles,self.d)) ) * ddt**0.5
        else:
            return ddt * self.F(state)  +  jnp.einsum('imn,in->im',self.sqrt2D(state), random.normal(key,shape=(self.Nparticles,self.d)) ) * ddt**0.5
            


    def simulate(self, dt, Ntimesteps, key, oversampling = 1, prerun = 0):
        """The main simulation loop: first equilibrate over a prerun*dt in
        prerun*oversampling simulation steps, then simulate over a
        total time Ntimesteps*dt, recording every dt, in
        oversampling*prerun simulation steps.

        """
        self.Ntimesteps = Ntimesteps 
        self.dt = dt
        
        ddt = self.dt / oversampling
        start_time = time.time() # Measure simulation start time

        @jit
        def single_step(carry, _):
            state, key = carry
            key, subkey = random.split(key)
            #new_state = state + self._dx(state, ddt, subkey)
            has_invalid = (
                jnp.any(jnp.isnan(state)) | 
                jnp.any(jnp.isinf(state)) | 
                jnp.any(jnp.iscomplex(state))
            )
            def compute():
                key1, subkey = random.split(key)
                return state + self._dx(state, ddt, subkey), key1
            # If a NaN / infinite / complex value is encountered we
            # stop computing as there's no going back and compute time
            # with nans can be very large.
            new_state, new_key = lax.cond(has_invalid, lambda: (state, key), compute)
            return (new_state, new_key), None  # We don't need to output each substep

        @jit
        def oversample_step(carry, _):
            state, key = carry
            (state, key), _ = lax.scan(single_step, (state, key), None, length=oversampling)
            return (state, key), state  # Only output the final state of each oversample step

        state = self.initial_position
        # Pre-equilibration
        (state, key), self.X = lax.scan(oversample_step, (state, key), None, length=prerun)
        # Main simulation
        (_, _), self.X = lax.scan(oversample_step, (state, key), None, length=self.Ntimesteps)

        # Record simulation parameters and time
        self.metadata["Ntimesteps"] = Ntimesteps
        self.metadata["dt"] = dt
        self.metadata["oversampling"] = oversampling
        self.metadata["prerun"] = prerun
        self.metadata["simulation_time"] = time.time() - start_time


    def compute_information(self):
        """Compute entropy production and information (about the
        force) in the trajectory. These are useful to assess the
        learnability of force fields using the trajectory. Note that
        the information here is mesured using the Ito drift (here
        called force), which slightly differs from the original Phys
        Rev X paper.

        """
        @jit
        def S_term(t):
            dX = self.X[t+1] - self.X[t]
            X_mid = 0.5 * (self.X[t+1] + self.X[t])
            force_mid = self.F(X_mid)
            if self._constant_D:
                return jnp.einsum('im,in,mn->', dX, force_mid, jnp.linalg.pinv(self.D))
            else:
                D_inv_mid = vmap(jnp.linalg.pinv)(self.D(X_mid))
                return jnp.einsum('im,in,imn->', dX, force_mid, D_inv_mid)

        @jit
        def I_term(t):
            dX = self.X[t+1] - self.X[t]
            force_t = self.F(self.X[t])
            if self._constant_D:
                return jnp.einsum('im,in,mn->', dX, force_t, jnp.linalg.pinv(self.D))
            else:
                D_inv_t = vmap(jnp.linalg.pinv)(self.D(self.X[t]))
                return jnp.einsum('im,in,imn->', dX, force_t, D_inv_t)

        S_terms = vmap(S_term)(jnp.arange(self.Ntimesteps-1))
        I_terms = vmap(I_term)(jnp.arange(self.Ntimesteps-1))

        self.S = jnp.sum(S_terms)
        self.I = 0.25 * jnp.sum(I_terms)
        
        # Add entropy and information metrics to metadata
        self.metadata.update({
            "entropy_production": self.S.astype(float),
            "information": self.I.astype(float)
        })

    def save_trajectory_data(self, filename='trajectory_data.csv'):
        """Saves the trajectory data to a CSV file in a format similar to
        tracking data from soft matter experiments.
        """
        state_vectors,time_idx,dt,particle_idx = self.export_trajectory()
        SFI.SFI_utils.save_trajectory_csv(filename,
                            particle_idx,
                            time_idx,
                            state_vectors,
                            self.metadata
                            )
        print(f"Trajectory data saved to {filename}")

    def export_trajectory(self):
        """ Ready to be read by StochasticTrajectoryData. """
        particle_idx, time_idx, state_vectors = SFI.SFI_utils.flatten_X_to_columns(self.X)
        return state_vectors,time_idx,self.dt,particle_idx


        
class ParticlesOverdampedLangevinProcess(OverdampedLangevinProcess):
    """A derivative class to handle multi-particle systems interacting
    through pair interactions (force_pair) and external fields
    (force_single). Higher-order interactions could be added in a
    similar fashion if the need arises.

    force_single: x_i (d-dimensional) -> F_i (d-dimensional) 
    force_pair: x_i, x_j -> F_i (contribution of the pair to the force on i)
    diffusion: scalar, dxd tensor, or function x_i -> D_i (dxd)

    """
    def __init__(self, force_single, force_pair,theta_single,theta_pair, D):
        self.theta_single = theta_single
        self.theta_pair = theta_pair
        self.Fsingle = force_single
        self.Fpair = force_pair
        @jit
        def force(X,theta):
            # Compute single-particle forces
            single_forces = vmap(self.Fsingle,in_axes=(0, None))(X, theta[0:len(self.theta_single)])
            # Compute pairwise forces
            pair_forces = vmap(vmap(self.Fpair, in_axes=(None, 0, None)), in_axes=(0, None, None))(X, X, theta[len(self.theta_single):len(self.theta_single)+len(self.theta_pair)])
            # Create a mask to zero out self-interactions
            zeros = jnp.zeros_like(pair_forces[0, 0])
            Nparticles = X.shape[0]
            mask = jnp.ones((Nparticles, Nparticles)) - jnp.eye(Nparticles)
            masked_pair_forces = jnp.where(mask[:, :, jnp.newaxis],  pair_forces, zeros)
            # Sum up all pairwise forces for each particle
            total_pair_forces = jnp.sum(masked_pair_forces, axis=1)
            return single_forces + total_pair_forces
        super().__init__(force, D,F_is_multiparticle=True,D_is_multiparticle=False)

    def initialize(self,initial_position):
        self.Nparticles = initial_position.shape[0]
        super().initialize(initial_position,params_F=jnp.concat([self.theta_single,self.theta_pair]),params_D=None)
    





    
class UnderdampedLangevinProcess(OverdampedLangevinProcess):
    """Simulates underdamped Langevin dynamics:

    dX/dt = V
    dV/dt = F(X, V) + sqrt(2D(X, V)) * xi(t)

    with Verlet integration, Ito definition of the force, and a
    structure parallel to OverdampedLangevinProcess.

    """
    
    def __init__(self, F, D, F_is_multiparticle=False, D_is_multiparticle=False):
        super().__init__(F, D,F_is_multiparticle=F_is_multiparticle,D_is_multiparticle=D_is_multiparticle)

    def initialize(self, initial_position, initial_velocity, params_F, params_D=None):
        if len(initial_position.shape) == 1:
            # Single simulations: add an axis for compatibility
            self.initial_position = initial_position[None, :]
            self.initial_velocity = initial_velocity[None, :]
        else:
            # Multiparticles/multicopy simulations
            self.initial_position = initial_position
            self.initial_velocity = initial_velocity
        self.Nparticles, self.d = self.initial_position.shape
        self.params_F = params_F
        self.params_D = params_D

        # Configure the force function
        if self.F_is_multiparticle:
            self.F = lambda X, V: self.uninitialized_F(X, V, self.params_F)
        else:
            self.F = vmap(lambda X, V: self.uninitialized_F(X, V, self.params_F), in_axes=(0, 0))

        # Configure the diffusion function
        if isinstance(self.uninitialized_D, (float, int)):
            self.setup_constant_D(self.uninitialized_D * jnp.eye(self.d))
        elif hasattr(self.uninitialized_D, "shape"):
            self.setup_constant_D(self.uninitialized_D)
        elif callable(self.uninitialized_D):
            self.setup_variable_D(lambda X, V: self.uninitialized_D(X, V, self.params_D))
        else:
            raise ValueError("Diffusion input not supported.")

        # Store metadata for reproducibility
        self.metadata.update({
            "algorithm": "underdamped",
            "num_particles": self.Nparticles,
            "dimension": self.d,
            "params_F": self.params_F,
            "params_D": self.params_D,
            "initial_position": initial_position.tolist(),
            "initial_velocity": initial_velocity.tolist(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def setup_constant_D(self, D):
        self.D = D
        self.sqrt2D = jnp.real(sqrtm_psd(2 * D))
        self._constant_D = True

    def setup_variable_D(self, D):
        if self.D_is_multiparticle:
            # Here D : (Nparticles x d)-> (Nparticles x d x d)
            self.D = D
            self.sqrt2D = jit(lambda x,v : jnp.real(vmap(sqrtm_psd)(2 * D(x,v))))
        else:
            # Here D : (d)-> (d x d); we use vmap to make it (Nparticles x d)-> (Nparticles x d x d)
            self.D = vmap(D,in_axes=(0,0))
            self.sqrt2D = jit(vmap(lambda x,v : jnp.real(sqrtm_psd(2 * D(x,v))),in_axes=(0,0)))
        self._constant_D = False

    @partial(jit, static_argnums=(0,))
    def _step(self, state, ddt, key):
        """Stochastic velocity-Verlet integration step with Ito convention."""
        X, V = state
        key, subkey = random.split(key)

        if self._constant_D:
            noise = jnp.einsum('mn,in->im', self.sqrt2D, random.normal(subkey, shape=(self.Nparticles, self.d))) * jnp.sqrt(ddt)
        else:
            noise = jnp.einsum('imn,in->im', self.sqrt2D(X, V), random.normal(subkey, shape=(self.Nparticles, self.d))) * jnp.sqrt(ddt)
        V_half = V + 0.5 * ddt * self.F(X, V) + noise
        X_new = X + ddt * V_half
        V_new = V_half + 0.5 * ddt * self.F(X_new, V_half)
        
        return (X_new, V_new), key

    def simulate(self, dt, Ntimesteps, key, oversampling=1, prerun=0):
        """Main simulation loop with pre-equilibration."""
        self.Ntimesteps = Ntimesteps 
        self.dt = dt
        ddt = dt / oversampling
        start_time = time.time() # Measure simulation start time
        
        @jit
        def single_step(carry, _):
            state, key = carry
            key, subkey = random.split(key)
            return self._step(state, ddt, subkey), None
        
        @jit
        def oversample_step(carry, _):
            state, key = carry
            (state, key), _ = lax.scan(single_step, (state, key), None, length=oversampling)
            return (state, key), state

        state = (self.initial_position, self.initial_velocity)
        # Pre-equilibration
        (state, key), _ = lax.scan(oversample_step, (state, key), None, length=prerun)
        # Main simulation
        (_, _), (self.X,self.V) = lax.scan(oversample_step, (state, key), None, length=Ntimesteps)

        # Record simulation parameters and time
        self.metadata["Ntimesteps"] = Ntimesteps
        self.metadata["dt"] = dt
        self.metadata["oversampling"] = oversampling
        self.metadata["prerun"] = prerun
        self.metadata["simulation_time"] = time.time() - start_time



    def compute_information(self):
        """Compute information (about the force) in the
        trajectory.

        """
        @jit
        def I_term(t):
            force_t = self.F(self.X[t],self.V[t])
            if self._constant_D:
                return jnp.einsum('im,in,mn->', force_t, force_t, jnp.linalg.pinv(self.D))
            else:
                D_inv_t = vmap(jnp.linalg.pinv)(self.D(self.X[t],self.V[t]))
                return jnp.einsum('im,in,imn->', force_t, force_t, D_inv_t)

        I_terms = vmap(I_term)(jnp.arange(self.Ntimesteps-1))

        self.I = 0.25 * jnp.sum(I_terms) * self.dt
        
        # Add information metrics to metadata
        self.metadata.update({
            "information": self.I.astype(float)
        })

class ParticlesUnderdampedLangevinProcess(UnderdampedLangevinProcess):
    """A derivative class to handle multi-particle systems interacting
    through pair interactions (force_pair) and external fields
    (force_single).

    force_single: x_i, v_i (d-dimensional) -> F_i (d-dimensional) 
    force_pair: x_i, x_j, vi, vj -> F_i (contribution of the pair to the force on i)
    diffusion: scalar, dxd tensor, or function x_i, v_i -> D_i (dxd)

    """
    def __init__(self, force_single, force_pair,theta_single,theta_pair, D):
        self.theta_single = theta_single
        self.theta_pair = theta_pair
        self.Fsingle = force_single
        self.Fpair = force_pair
        @jit
        def force(X,V,theta):
            # Compute single-particle forces
            single_forces = vmap(self.Fsingle,in_axes=(0,0,None))(X,V,theta[0:len(self.theta_single)])
            # Compute pairwise forces
            pair_forces = vmap(vmap(self.Fpair, in_axes=(None, 0, None, 0, None)), in_axes=(0, None, 0, None, None))(X, X, V, V, theta[len(self.theta_single):len(self.theta_single)+len(self.theta_pair)])
            # Create a mask to zero out self-interactions
            zeros = jnp.zeros_like(pair_forces[0, 0])
            Nparticles = X.shape[0]
            mask = jnp.ones((Nparticles, Nparticles)) - jnp.eye(Nparticles)
            masked_pair_forces = jnp.where(mask[:, :, jnp.newaxis],  pair_forces, zeros)
            # Sum up all pairwise forces for each particle
            total_pair_forces = jnp.sum(masked_pair_forces, axis=1)
            return single_forces + total_pair_forces
        super().__init__(force, D,F_is_multiparticle=True,D_is_multiparticle=False)

    def initialize(self,initial_position,initial_velocity):
        self.Nparticles = initial_position.shape[0]
        super().initialize(initial_position,initial_velocity,params_F=jnp.concat([self.theta_single,self.theta_pair]),params_D=None)
    
