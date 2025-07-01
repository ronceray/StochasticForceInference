from jax import jit,vmap
import jax.numpy as jnp
from functools import partial
from SFI.SFI_data import StochasticTrajectoryData
from SFI.SFI_sparsity import SparseModelSelector
from SFI import SFI_utils
from SFI.SFI_base_inference import BaseLangevinInference

class UnderdampedLangevinInference(BaseLangevinInference): 
    """UnderdampedLangevinInference (ULI) main class.

    This class provides tools for inferring force (drift) and
    diffusion tensors from stochastic trajectory data based on
    second-order (underdamped) Langevin dynamics. It closely parallels
    the first-order inference class, StochasticForceInference.

    Core Equation:
    --------------
    The dynamics are described by the 1st order autonomous stochastic differential equation (SDE):
        dx/dt = v
        dv/dt = F(x,v) + sqrt(2D(x,v)) dxi(t)
    where:
    - `F(x,v)` is the Ito drift (force) term.
    - `D(x,v)` is the diffusion tensor, evaluated in the Ito convention.
    - `dxi(t)` is Gaussian white noise.
    Here x is a 2D array of shape Nparticles x dimension. All particles are assumed to have identical properties. 

    This class provides tools to approximate F(x,v) and D(x,v) from a time series x(t) formatted
    as StochasticTrajectoryData.

    The central difference with OverdampedLangevinInference is that the velocities v(t) are assumed not to be observed,
    but can only calculated by differentiating x(t) [so-called "secant velocities"]. This strongly complexifies the
    inference, as it induces non-trivial correlations between velocity and acceleration. This is documented in the
    original ULI paper, "Inferring the dynamics of underdamped stochastic systems" - DB Brückner, P Ronceray,
    CP Broedersz - Physical review letters, 2020.

    Key Features:
    --------------
    - Force Inference:
      - Linear combination of basis functions (`infer_force_linear`).
    - Diffusion Inference:
      - Constant diffusion estimation (`infer_diffusion_constant`).
      - State-dependent diffusion with basis functions (`infer_diffusion_linear`).

    Workflow:
    ---------
    1. Initialize with `StochasticTrajectoryData` containing the trajectory.
    2. Use the `infer_*` methods to infer force and diffusion fields.
    3. Optionally compute error estimates and/or compare with exact data for validation.

    Indices Convention:
    -------------------
    The code uses jnp.einsum for array manipulation, with a consistent index naming scheme for clarity:
    - `t` : Time index, = 0..Ntimesteps-1
    - `a, b, c...` : Basis function indices, = 0..Nfunctions - 1.
    - `m, n, o...` : Spatial indices, = 0..dim-1.
    - `i, j...` : Particle indices.
    We also use these indices as shortcuts for array shapes. For
    instance `basis_linear : im -> iam` reads `basis_linear has input
    a jnp.array of shape (Nparticles,dim) and outputs a jnp.array of
    shape (Nparticles,Nfunctions,dim)`.

    Dependencies:
    -------------
    - JAX for accelerated numerical computations and auto-differentiation.
    - `SFI_data.StochasticTrajectoryData` for input formatting and trajectory management.
    - `ULI_bases`, while not required, provides useful pre-defined bases (eg polynomials, pair interactions...)
    - `SFI_langevin`, to simulate bootstrapped trajectories using inferred fields

    Example:
    --------
    Fully documented examples in the "examples" folder: Damped harmonic oscillator, Van der Pol oscillator, flocking particles...

    """

        
    def infer_force_linear(self, 
                           basis_linear: callable, 
                           basis_linear_grad_v: callable = None,
                           * ,
                           M_mode : str = 'symmetric', 
                           diffusion_method: str = 'noisy', 
                           G_mode: str = "trapeze", 
                           basis_names = None):
        """Infer the force field as a linear combination of basis functions.

        This method computes the force field coefficients (`self.force_coefficients`) using
        the provided basis functions. The force field is represented as:

            force_ansatz(x,v) = sum_a basis_linear(x,v)[:,a] * force_coefficients[a]

        These coefficients are computed by solving a linear system:

            G . force_coefficients = force_moments

        and the different options account for the manner to compute G
        and force_moments.

        Args:
        
            basis_linear (callable): im,im,mask -> ima
                The fitting functions, encoded as a single callable
                taking as input the state x and v (each jnp.array, shape
                (Nparticles,dim)) and the mask (jnp.array of booleans,
                shape (Nparticles,)). See ULI_bases for simple ways to
                construct these.

            basis_linear_grad_v (callable, optional): im,im,mask -> iman
                The gradient of basis_linear with respect to v: 
                basis_linear_gradient[i,m,a,n] = d basis_linear(x,mask)[i,m,a] / dv[i,n]
                (can also be automatically provided in ULI_bases).

            diffusion_method (str):
                Mode for diffusion tensor estimation. Defaults to 'noisy'.

            G_mode (str):
                Method for computing the Gram normalization matrix G. Options are:
                    - "rectangle": Standard (symmetric) normalization as in the 2020 ULI.
                    - "trapeze": Trapezoidal correction to reduce discretization bias (default).
                    - ...

        Outputs:
            Updates the following attributes:
                - self.force_coefficients: The inferred coefficients for the basis functions.
                - self.force_ansatz: Callable function representing the inferred force field.
                - self.force_G: The normalization matrix used in the inference process.

        """
        if M_mode == 'auto':
            raise NotImplementedError("auto mode TODO")

        # Store arguments for downstream use:
        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.__force_diffusion_method__ = diffusion_method
        self.force_basis_names = basis_names
        self.force_basis_linear = basis_linear
        self.force_basis_linear_grad_v = basis_linear_grad_v

        # Call the base class template:
        self._infer_force_linear_template()


        
    def infer_diffusion_linear(self, 
                               basis_linear: callable, 
                               method: str = 'noisy', 
                               G_mode: str = "rectangle",
                               basis_names = None) -> None:
        """
        Fit the diffusion field as a linear combination of basis functions.

        This method computes the coefficients of the diffusion tensor field (`self.diffusion_coefficients`) using
        the provided basis functions. The diffusion tensor is represented as:

            diffusion_ansatz(x,v,mask) = sum_a basis_linear(x,v,mask)[:,a] * diffusion_coefficients[a]

        Args:
            basis_linear (callable): im, im, mask -> iamn
                The basis functions for diffusion inference. Encoded as a callable that takes input
                `x` and `v` (array of shape `(N_particles, dimension)`) and `mask` (array of shape `(N_particles,)`) and returns an array of shape
                `(N_particles, N_basis_functions, dimension, dimension)`.

            method (str):
                The method used for local diffusion tensor estimation. Supported options include:
                See _diffusion_estimator documentation for additional information.

            G_mode (str):
                The method used to compute the normalization matrix `G`. Options are:
                - "rectangle": Default normalization.
                - "trapeze": Trapezoidal correction. Note: not tested extensively.

        Updates:
            self.diffusion_coefficients: The inferred coefficients for the diffusion basis functions.
            self.diffusion_ansatz: Callable representing the inferred diffusion tensor field.
            self.diffusion_G: The normalization matrix used in the inference process.
        
        Note:
            This ansatz is not guaranteed to be nonnegative.
        """
        if method == 'auto':
            raise NotImplementedError("auto mode TODO")

        self.diffusion_basis_linear = basis_linear
        self.__diffusion_method__ = method
        self.__diffusion_G_mode__ = G_mode
        self.diffusion_basis_names = basis_names

        # Call the base class template:
        self._infer_diffusion_linear_template()
        
                
    def simulate_bootstrapped_trajectory(self, key, oversampling = 1, simulate = True):
        """
        Simulate an underdamped Langevin trajectory with the inferred force and diffusion fields.

        This function generates a trajectory using the ansatz force field and diffusion tensor inferred
        from the input data, matching the original time series and initial conditions.

        Args:
            key: JAX random key for generating noise in the simulation.
            oversampling (int, optional): Factor for oversampling (i.e. number of intermediate simulated points between two recorded points). Defaults to 1.
            simulate: if True, performs the simulation with the first data point as initial position;
                      if False, returns an uninitialized object which can be simulated with flexible initial position and parameters.

        Returns:
            UnderdampedLangevinProcess: Simulated Langevin process object.
        """
        from SFI.SFI_Langevin import UnderdampedLangevinProcess

        if hasattr(self, 'diffusion_ansatz'):
            # Variable diffusion tensor
            if self.verbosity >= 1: print("Simulating bootstrapped trajectory with state-dependent diffusion.")
            X = UnderdampedLangevinProcess(
                self.force_ansatz,
                self.diffusion_ansatz,
                F_is_multiparticle=True,
                D_is_multiparticle=True
            )
        else:
            # Constant diffusion tensor
            if self.verbosity >= 1: print("Simulating bootstrapped trajectory assuming constant diffusion.")
            X = UnderdampedLangevinProcess(
                self.force_ansatz,
                self.diffusion_average,
                F_is_multiparticle=True
            )

        if simulate:
            # Initialize and simulate the process
            X.initialize(self.data.X[0],self.data.dX[0]/self.data.dt,params_F=None,params_D=None)
            X.simulate(self.data.dt, self.data.Nsteps, key, oversampling=oversampling, prerun=0)

        return X


    #################################################################
    ################       BACKEND       ############################
    #################################################################
        
    def __G_matrix__(self, b_left: callable, b_right: callable, G_mode: str, einsum_string: str, subsampling : int = 1) -> jnp.ndarray:
        """Compute the normalization matrix G (Gram matrix) based on the selected mode.

        Args:
            basis_linear (callable): im,mask->ia..  Function to compute the basis values.

            G_mode (str): Mode for G computation ('rectangle' or 'trapeze').
        
            einsum_string (str): Einstein summation convention string
            for einsum operation. Note that this should still include
            particles in the output, as per data.trajectory_average
            requirement for masking. Example: 'iam,ibn->iabmn' for
            output 'abmn' (Nfunctions x Nfunctions x dim x dim).

            subsampling: permits a faster evaluation of G when
            precision is not needed (for instance for error
            estimation).

        Returns:
            jnp.ndarray: Normalization matrix G. Shape Nfunctions x Nfunctions x ...

        """
        if self.verbosity >= 1: print("Computing G matrix with einsum:",einsum_string)
        if G_mode== "rectangle":
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.dX[t]/self.data.dt,self.data.get_mask_at(t))
                br = b_right(self.data.X[t],self.data.dX[t]/self.data.dt,self.data.get_mask_at(t))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        elif G_mode== "trapeze":
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.dX[t]/self.data.dt,self.data.get_mask_at(t))
                br = 0.5 * ( b_right(self.data.X_minus(t),self.data.dX_minus[t]/self.data.dt,self.data.get_mask_at(t))
                           + b_right(self.data.X[t],self.data.dX[t]/self.data.dt,self.data.get_mask_at(t)))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        elif G_mode== "shift":
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.dX[t]/self.data.dt,self.data.get_mask_at(t))
                br = b_right(self.data.X_minus(t),self.data.dX_minus[t]/self.data.dt,self.data.get_mask_at(t))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        elif G_mode== "doubleshift":
            def Ginst(t):
                bf = b_left(self.data.X_plus(t),self.data.dX_plus[t]/self.data.dt,self.data.get_mask_at(t))
                br = b_right(self.data.X_minus(t),self.data.dX_minus[t]/self.data.dt,self.data.get_mask_at(t))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        else:
            raise KeyError("Wrong G_mode argument")
        result = self.data.trajectory_average(Ginst,verbosity=self.verbosity-1,max_memory_gb=self.max_memory_gb,subsampling=subsampling)
        return result


    ###### BACKEND: HOOKS REQUIRED BY BASE CLASS ####


    def _diffusion_estimator(self, method: str) -> jnp.ndarray:
        """These local diffusion estimators provide instantaneous,
    noisy estimates of the diffusion tensor. Noise is O(1) and depends
    on trajectory length and sampling. Select the appropriate
    estimator for the specific system being analyzed.

        """

        dX = lambda t : self.data.dX[t]  # Displacement at time step t
        dX_minus = lambda t : self.data.dX_minus[t] 
        dX_plus = lambda t : self.data.dX_plus[t] 

        if method in [ "noisy", "WeakNoise", "Lambda" ] and not hasattr(self.data, "dX_plus"):
            raise RuntimeError(f"Need to compute dX_plus at the StochasticTrajectoryData initialization to use method {method}.")

        
        def symmetrize(M):
            return 0.5*(M + jnp.einsum('inm->imn',M))
        
        estimators = {
            "MSD" : lambda t : 0.75 * jnp.einsum('im,in->imn', dX(t)-dX_minus(t), dX(t)(t)-dX_minus(t)) / self.data.dt ** 3,

            # Experimental, 3rd-order-derivative-based estimator to reduce drift-induced biases
            "WeakNoise" : lambda t : 0.5 * jnp.einsum('im,in->imn', 2 * dX(t)-dX_minus(t)-dX_plus(t), 2 * dX(t)-dX_minus(t)-dX_plus(t)) / self.data.dt ** 3,

            # The derivation of this one is ugly!
            "noisy" : lambda t : 3 * symmetrize( jnp.einsum('im,in->imn',dX(t),dX(t))             * (-1 ) + # a
                                               jnp.einsum('im,in->imn',dX_minus(t),dX_minus(t)) * ( 1 ) + # b
                                               jnp.einsum('im,in->imn',dX_plus(t),dX_plus(t))   * ( 1 ) + # c
                                               jnp.einsum('im,in->imn',dX_plus(t),dX_minus(t))  * (-3 ) + # d
                                               jnp.einsum('im,in->imn',dX(t),dX_plus(t))        * ( 1 ) + # e 
                                               jnp.einsum('im,in->imn',dX(t),dX_minus(t))       * ( 1 )   # f
                                              ) / ( 11 * self.data.dt**3 ),

            # That one too...
            "Lambda" : lambda t : symmetrize( jnp.einsum('im,in->imn',dX(t),dX(t))             * ( 10) +  # a
                                            jnp.einsum('im,in->imn',dX_minus(t),dX_minus(t)) * ( 1 ) +  # b
                                            jnp.einsum('im,in->imn',dX_plus(t),dX_plus(t))   * ( 1 ) +  # c
                                            jnp.einsum('im,in->imn',dX_plus(t),dX_minus(t))  * ( 8 ) +  # d
                                            jnp.einsum('im,in->imn',dX(t),dX_plus(t))        * (-10) +  # e 
                                            jnp.einsum('im,in->imn',dX(t),dX_minus(t))       * (-10)    # f
                                           ) / 44
        }

        if method not in estimators:
            raise KeyError(f"Invalid diffusion estimation method: {method}")

        return jit(estimators[method])

    def _force_G_matrix(self)-> jnp.ndarray:
        b_left = self.force_basis_linear
        b_right = jit(lambda X,V,mask : self.force_basis_linear(X,V,mask) @ self.A_inv )
        return self.__G_matrix__(b_left,b_right,self.__force_G_mode__,'iam,ibm->iab')

    def _force_moments(self) -> jnp.ndarray:
        # Compute the moments:
        # Define instantaneous term for velocity coefficients
        if self.__force_M_mode__ == 'symmetric':
            """ Recommended for noisy / real data. """
            l = 1.
            def at(t):
                return (self.data.dX[t]-self.data.dX_minus[t]) / self.data.dt**2
            def vt(t):
                return ((l/2)*self.data.dX[t]+(1-l/2)*self.data.dX_minus[t]) / self.data.dt
            def xt(t):
                return (self.data.X[t] + self.data.X_minus(t) + self.data.X_plus(t)) / 3.
        elif self.__force_M_mode__ == 'early':
            """ Simple version that works for ideal data. """
            l = 0.
            def at(t):
                return (self.data.dX[t]-self.data.dX_minus[t]) / self.data.dt**2
            def vt(t):
                return self.data.dX_minus[t] / self.data.dt
            def xt(t):
                return self.data.X[t]
        elif self.__force_M_mode__ == 'anticipated':
            """ Decorrelates the acceleration and velocity using a
            time lag rather than an explicit gradient-based
            correction. Usually not a great alternative, unless the
            basis is non-differentiable. """
            l = -0.5
            def at(t):
                return (self.data.dX_plus[t]-self.data.dX[t]) / self.data.dt**2
            def vt(t):
                return self.data.dX_minus[t] / self.data.dt
            def xt(t):
                return (self.data.X[t] + self.data.X_plusplus(t) + self.data.X_plus(t)) / 3.
        else:
            raise ValueError("Mode not recognized for linear_inference:",mode)

        @jit
        def ahat_b(t):
            """ Basis-acceleration instantaneous product. """
            return jnp.einsum('im,ian,mn->ia', at(t), self.force_basis_linear(xt(t),vt(t),self.data.get_mask_at(t)),
                              self.A_inv, optimize=True)

        phi_moments = self.data.trajectory_average(ahat_b,verbosity=self.verbosity-1,max_memory_gb=self.max_memory_gb)

        @jit
        def diffusion_grad_b_v(t):
            """ Diffusion-gradient instantaneous correction term. """
            grad_b_v_t = self.force_basis_linear_grad_v(xt(t),vt(t),self.data.get_mask_at(t))
            diffusion = self._diffusion_estimator(self.__force_diffusion_method__)
            return - jnp.einsum('imn,no,iaom->ia', diffusion(t), self.A_inv, grad_b_v_t, optimize=True)

        if l != -0.5:
            w_moments = ((1+2*l)/3.) * self.data.trajectory_average(diffusion_grad_b_v,
                                                                    verbosity=self.verbosity-1,max_memory_gb=self.max_memory_gb)
        else:
            w_moments = 0.

        # Combine to obtain the force moments.
        return phi_moments + w_moments


    def _diffusion_G_matrix(self)-> jnp.ndarray:
        # Compute normalization matrix G
        return self.__G_matrix__(self.diffusion_basis_linear, self.diffusion_basis_linear,
                                 self.__diffusion_G_mode__, 'iamn,ibmn->iab')
    

    def _diffusion_moments(self) -> jnp.ndarray:

        if self.verbosity >= 1: print("Computing diffusion linear coefficients.")
        # Parse the local diffusion estimator
        D_inst = self._diffusion_estimator(self.__diffusion_method__)

        if self.verbosity >= 1: print("Computing diffusion integral.")
        # Define the average of diffusion tensor components with the basis functions
        if self.__diffusion_method__ == 'noisy':
            Xt = lambda t : 0.25*(self.data.X[t] + self.data.X_minus(t) + self.data.X_plus(t) + self.data.X_plusplus(t))
            Vt = lambda t : (4 * self.data.dX[t] + 1 * (self.data.dX_plus[t]+self.data.dX_minus[t])) / (6 * self.data.dt)
        elif self.__diffusion_method__ == 'MSD':
            Xt = lambda t : (self.data.X[t] + self.data.X_minus(t) + self.data.X_plus(t))/3.
            Vt = lambda t : (self.data.dX[t] + self.data.dX_minus[t]) / (2 * self.data.dt)
        elif self.__diffusion_method__ == 'WeakNoise':
            # Experimental
            Xt = lambda t : (self.data.X[t] + self.data.X_plus(t))/2.
            Vt = lambda t : (self.data.dX[t]) / (self.data.dt)
        @jit
        def bD(t):
            return jnp.einsum('iamn,inm->ia',self.diffusion_basis_linear(Xt(t), Vt(t), self.data.get_mask_at(t)),
                              D_inst(t),optimize=True)
        # Compute trajectory-averaged coefficients
        return self.data.trajectory_average(bD,verbosity=self.verbosity-1,max_memory_gb=self.max_memory_gb)

    def _update_force_ansatz(self) -> None :
        # Define the force field as a callable function
        def force_ansatz(x: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
            """
            Compute the inferred force field at given positions.

            Args:
                x, v (jnp.ndarray): Particle positions and velocities, shape (N_particles, dimension).
                mask (jnp.ndarray, optional): Boolean mask indicating active particles. 
                    Defaults to all True.

            Returns:
                jnp.ndarray: Inferred force field, shape (N_particles, dimension).
            """
            if mask is None:
                mask = jnp.ones(x.shape[0], dtype=bool)  # Default: all particles are active
            result = self.force_coefficients.dot(self.force_basis_linear(x, v, mask)[:,self.force_support,:])  # Contribution from basis functions
            # Mask out inactive particles by setting their values to NaN
            return jnp.where(mask[:, None], result, jnp.nan)
        self.force_ansatz = jit(force_ansatz) 

        
    def _update_diffusion_ansatz(self) -> None : 
        # Define the diffusion tensor ansatz as a callable function
        def diffusion_ansatz(x: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
            """
            Compute the inferred diffusion tensor at given positions.

            Args:
                x (jnp.ndarray): Particle positions, shape `(N_particles, dimension)`.
                mask (jnp.ndarray, optional): Boolean mask indicating active particles, shape `(N_particles,)`.
                    Defaults to all True.

            Returns:
                jnp.ndarray: Diffusion tensor field, shape `(N_particles, dimension, dimension)`.
            """
            if mask is None:
                mask = jnp.ones(x.shape[0], dtype=bool)  # Default: all particles are active
            linear_term = jnp.einsum('iamn,a->imn', self.diffusion_basis_linear(x,v,mask), self.diffusion_coefficients)
            # Mask out inactive particles by setting their values to NaN
            return jnp.where(mask[:, None, None], linear_term, jnp.nan)

        # JIT-compile the diffusion ansatz for performance
        self.diffusion_ansatz = jit(diffusion_ansatz)

    def _compute_sampled_values(self,function,indices,data):
        # Tiny helper for sampled average
        return vmap(function,in_axes=(0, 0))(data.X[indices],data.dX[indices]/data.dt)

