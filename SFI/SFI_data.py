import jax.numpy as jnp
from jax import jit,vmap,lax
import time
from functools import partial
from SFI.SFI_utils import assemble_X_from_columns


class StochasticTrajectoryData:
    def __init__(self, state_vectors, time_indices, dt, particle_indices=None, compute_dXminus=True,compute_dXplus=False):
        """
        Initialize the data structure from (x,t,i) format.
        Args:
            state_vectors: Array of state vectors (x) with shape (N, d).
            time_indices: Array of time indices, shape (N,).
            dt: time step. (uneven times unsupported for now)
            particle_indices: Array of particle indices, shape (N,). If None, assumes there is just one "particle".

        The following position increments are calculated:
        dX       = X(t+dt)  -  X(t)     mandatory
        dX_minus = X(t)     -  X(t-dt)  optional; for shift method / Vestergaard diffusion estimator / underdamped
        dX_plus  = X(t+2dt) -  X(t+dt)  optional; only for high-noise underdamped
        

        """
        self.dt = dt
        if particle_indices is None:
            # One single particle assumed
            particle_indices = 0 * jnp.array(time_indices)

        # Use the utility to construct structured state array and mask
        # (NaN if particle is missing):
        X, mask = assemble_X_from_columns(particle_indices,
                                          time_indices,
                                          state_vectors )
        X,mask = jnp.array(X),jnp.array(mask)
        self.Nsteps, self.Nmaxparticles, self.d = X.shape
        
        start = 1 if compute_dXminus else 0
        end = self.Nsteps-2 if compute_dXplus else self.Nsteps-1
        if end < start:
            raise ValueError("Data too short to format trajectory.")
        
        self.X = X[start:end]
        self.dX = X[start+1:end+1] - X[start:end]
        self.mask = mask[start:end] & mask[start+1:end+1]
        
        # Prepare differences for 3- and 4-point estimators
        if compute_dXplus:
            self.dX_plus = X[start+2:end+2] - X[start+1:end+1]
            self.mask = self.mask & mask[start+2:end+2]
        if compute_dXminus:
            self.dX_minus = X[start:end] - X[start-1:end-1]
            self.mask = self.mask & mask[start-1:end-1]

        self.time_indices = jnp.arange(0,end-start)
        self.t = self.time_indices * dt # Only for plotting purposes
        
        self.Nparticles = self.mask.sum(axis=-1) # The number of particles at each time step
        self.tauN = self.Nparticles.sum() * self.dt # Total time-particle normalization constant

        # If there are no missing particles / data points, set the
        # mask to None to speedup the inference:
        self.mask = None if self.mask.all() else self.mask

    def get_mask_at(self, t : int):
        """Returns the mask for a given time inde t, or None if all data is present."""
        return None if self.mask is None else self.mask[t]

    
    def trajectory_average(self, func, max_memory_gb=1.0, verbosity=0,subsampling : int = 1):
        """The key utility function of this class, used to perform a
        trajectory average where `func` is applied element-wise on
        time steps.  Optimized for performance, automatically switches
        to batched processing if memory usage exceeds the given limit.

        Args:
            func (callable): Function that takes an integer index and returns an array of size (Nparticles, ...).
            max_memory_gb (float): Maximum allowed memory usage in GB.
            verbosity (int): 0 -> silent
                             1 -> indicate time and memory usage.

        Returns:
            jnp.ndarray: The averaged result, over time and particles..

        """        
        def apply_func_and_mask(indices, mask = None):
            """Apply the function and mask its output, returning the unnormalized sum."""
            if mask is None:
                return jnp.sum(vmap(jit(lambda t : jnp.sum(func(t),axis=0)))(indices[::subsampling]),axis=0)
            else:
                @jit
                def func_masked(t,m):
                    func_value = func(t)
                    expanded_mask = jnp.expand_dims(m, axis=tuple(range(1, func_value.ndim)))
                    return jnp.sum(jnp.where(expanded_mask, func_value, 0.0),axis=0)
                return jnp.sum(vmap(func_masked,in_axes=(0,0))(indices[::subsampling],mask[::subsampling]),axis=0)

        # Quickly estimate memory usage
        num_steps, num_particles = self.Nsteps//subsampling, self.Nmaxparticles

        single_output = func(self.time_indices[0])
        # Note that the " * num_particles" is conservative here (for
        # pair interactions): multi-particle basis functions tend to
        # have internal operations using Nparticles**2 which are not
        # reflected in the output.
        single_output_size_bytes = single_output.nbytes  * num_particles
        estimated_total_memory_usage_gb = (single_output_size_bytes * num_steps / subsampling) / (1024**3)
        
        if estimated_total_memory_usage_gb <= max_memory_gb:
            # Single batch average calculation (does not exceed memory limit)
            if verbosity >= 1:
                start_time = time.time()
                print(f"  Computing average of array with shape {single_output.shape} with subsampling {subsampling} using full mode. Estimated memory usage: {estimated_total_memory_usage_gb:.3f} GB.")
            total_sum = apply_func_and_mask(self.time_indices, self.mask)  # Full summation
        else:
            # Batched mode to avoid saturating the memory.
            batch_size = int(max_memory_gb * (1024**3) / single_output_size_bytes * subsampling)
            batch_size = max(1, batch_size)  # Ensure at least one step per batch
            num_full_batches = num_steps // batch_size  # Only count full batches
            remainder = num_steps % batch_size  # Remaining steps for the last batch

            if verbosity >= 1:
                start_time = time.time()
                print(f"  Computing average of array with shape {single_output.shape} with subsampling {subsampling} using batched mode. Estimated memory usage: {estimated_total_memory_usage_gb:.3f} GB.")
                print(f"  Batch size: {batch_size}, Full batches: {num_full_batches}, Last batch size: {remainder}")

            def batch_step(i, accum_sum):
                """Process a single batch and accumulate results."""
                start_idx = i * batch_size

                batch_indices = lax.dynamic_slice(self.time_indices, (start_idx,), (batch_size,))
                batch_mask = None if self.mask is None else lax.dynamic_slice(self.mask, (start_idx, 0), (batch_size, num_particles))

                return accum_sum + apply_func_and_mask(batch_indices, batch_mask)  # Unnormalized sum

            # Process full batches
            total_sum = lax.fori_loop(
                0, num_full_batches, batch_step, jnp.zeros_like(single_output)[0]  # Start accumulation at zero
            )

            # Process the last batch separately (if it exists)
            if remainder > 0:
                start_idx = num_full_batches * batch_size  # Start of last batch
                batch_indices = self.time_indices[start_idx:num_steps]
                batch_mask = None if self.mask is None else self.mask[start_idx:num_steps]
                total_sum += apply_func_and_mask(batch_indices, batch_mask)

        # Normalize by tauN at the very end
        tauN = self.Nparticles[::subsampling].sum() * self.dt # Total time-particle normalization constant
        result = self.dt * total_sum / tauN
        
        if verbosity >= 1:
            #result.block_until_ready() # For benchmarking purposes, force evaluation now
            runtime = time.time() - start_time
            print("   Run time: ", runtime)
        return result

    # These functions are occasionally used
    @partial(jit, static_argnums=(0,))
    def X_minus(self,t):
         """ X(t-dt) """
         return self.X[t] - self.dX_minus[t]
    @partial(jit, static_argnums=(0,))
    def X_plus(self,t : int):
         """ X(t+dt) """
         return self.X[t] + self.dX[t]
    @partial(jit, static_argnums=(0,))
    def X_plusplus(self,t : int):
         """ X(t+2dt) """
         return self.X[t] + self.dX[t] + self.dX_plus[t]
    
    def detach_from_jax(self):
        """Convert all JAX arrays inside this object to NumPy arrays
        to prevent memory leaks. Use this before deleting this object,
        as the Jax traces might persist otherwise. Important when
        performing a large number of inference runs in the same run
        (e.g. for benchmarking the method over many
        parameters/trajectories).

        """
        import jax
        import gc
        for attr_name in vars(self):  # Loop through all attributes
            attr_value = getattr(self, attr_name)

            if isinstance(attr_value, jnp.ndarray):  # If it's a JAX array
                setattr(self, attr_name, jax.device_get(attr_value))  # Convert to NumPy

            elif isinstance(attr_value, dict):  # If it's a dictionary, check inside
                for key, value in attr_value.items():
                    if isinstance(value, jnp.ndarray):
                        attr_value[key] = jax.device_get(value)

            elif isinstance(attr_value, list):  # If it's a list, check each item
                for i in range(len(attr_value)):
                    if isinstance(attr_value[i], jnp.ndarray):
                        attr_value[i] = jax.device_get(attr_value[i])

        # Clear any lingering references in JAX's cache
        jax.clear_caches()
        jax.device_get(jax.numpy.zeros(1))  # Forces JAX to clear buffers
        gc.collect()


