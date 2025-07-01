from jax import jit,vmap
import jax.numpy as jnp
from functools import partial
from SFI.SFI_data import StochasticTrajectoryData
from SFI.SFI_sparsity import SparseModelSelector
from SFI import SFI_utils
from SFI.SFI_base_inference import BaseLangevinInference

class OverdampedLangevinInference(BaseLangevinInference): 
    """Stochastic Force Inference Class

    This class provides tools for inferring force (drift) and
    diffusion tensors from stochastic trajectory data based on
    Langevin dynamics. It supports both linear and nonlinear basis
    function methods.

    Core Equation:
    --------------
    The dynamics are described by the 1st order autonomous stochastic differential equation (SDE):
        dx/dt = F(x) + sqrt(2D(x)) dxi(t)
    where:
    - `F(x)` is the Ito drift (force) term.
    - `D(x)` is the diffusion tensor, evaluated in the Ito convention.
    - `dxi(t)` is Gaussian white noise.
    Here x is a 2D array of shape Nparticles x dimension. All particles are assumed to have identical properties. 

    This class provides tools to approximate F(x) and D(x) from a time series x(t) formatted as StochasticTrajectoryData.

    Note that the `Ito` and `Strato` variants of the force inference
    routines do NOT refer to the convention in which the SDE is
    expressed (which is always Ito), but to the way stochastic
    integrals are performed to compute parameters.

    Key Features:
    --------------
    - Force Inference:
      - Linear combination of basis functions (`infer_force_linear`).
      - Nonlinear parametric functions (`infer_force_nonlinear`).
    - Diffusion Inference:
      - Constant diffusion estimation (`infer_diffusion_constant`).
      - State-dependent diffusion with basis functions (`infer_diffusion_linear` and `infer_diffusion_nonlinear`).
    - Sparsification:
      - Force sparsification for linear inference `sparsify_force`, implementing PASTIS and other information criteria.
    - Error Estimation:
      - Normalized mean-squared error (MSE) prediction for both force and diffusion.
    - Comparison Tools for method benchmarking:
      - Evaluate inferred fields against known exact models (`compare_to_exact`).
    - Simulation:
      - Generate trajectories using inferred fields (`simulate_bootstrapped_trajectory`).

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
    - `SFI_sparsity`, for linear inference: solving, information measurement, and model selection.
    - `OLI_bases`, while not required, provides useful pre-defined bases (eg polynomials, pair interactions...)
    - `SFI_langevin`, to simulate bootstrapped trajectories using inferred fields

    Verbosity:
    ----------
    Verbosity level controls standard output:
         0 -> silent
         1 -> print the main steps of the inference; compact report for sparse solver
         2 -> step-by-step printing and detailed sparsification report
         3 -> even more reports, including time-averages
        
    Example:
    --------
    Fully documented examples in the "examples" folder: Lorenz model, ActiveBrownianParticles, Ornstein-Uhlenbeck...

    """
    
    def infer_force_linear(self, 
                           basis_linear: callable, 
                           basis_linear_gradient: callable = None,
                           * ,
                           M_mode: str = 'auto', 
                           G_mode: str = "trapeze", 
                           basis_names = None):
        """Infer the force field as a linear combination of basis functions (linear regression).

        This method computes the force field coefficients (`self.force_coefficients`) using
        the provided basis functions. The force field is represented as:

            force_ansatz(x) = sum_a basis_linear(x)[:,a] * force_coefficients[a]

        These coefficients are computed by solving a linear system:

            G . force_coefficients = force_moments

        and the different options account for the manner to compute G
        and force_moments. In its simplest form
            G_ab = < b_a(xt) b_b(xt) >                [G_mode = 'rectangle']
            force_moments[a] = < dX[t]/dt b_a(xt) >   [mode = 'Ito' ]
        but this is rarely the best choice of parameters.

        Args:
        
            basis_linear (callable): im,mask -> ima
                The fitting functions, encoded as a single callable
                taking as input the state x (jnp.array, shape
                (Nparticles,dim)) and the mask (jnp.array of booleans,
                shape (Nparticles,)). See SFI_bases for simple ways to
                construct these.

            basis_linear_gradient (callable, optional): im,mask -> iman
                The gradient of basis_linear with respect to x: 
                basis_linear_gradient[i,m,a,n] = d basis_linear(x,mask)[i,m,a] / dx[i,n]
                (can also be automatically provided in SFI_bases).

            mode (str, optional):
                Method to perform stochastic inference ('Ito' or 'Strato'). 
                'Strato' (default) is more robust to measurement noise, but requires gradient evaluation.
                'Ito' is simpler and faster, but less robust.
                'auto' assesses which parameters are best suitable based on the data using the measurement noise trace.

            G_mode (str):
                Method for computing the Gram normalization matrix G. Options are:
                    - "rectangle": Standard (symmetric) normalization as in the 2020 PRX.
                    - "trapeze": Trapezoidal correction as in 2024 Amiri et al. PRR (default).
                    - "shift": <b_a(xt) b_b(x_t+dt) > to avoid measurement noise correlations.

            basis_names (List[str] | None):
                Printable names of basis functions. If None, generates generic b0, b1...

        Outputs:
            Updates the following attributes:
                - self.force_sparsifier: SparseModelSelector object for further model selection.
                - self.force_coefficients: The inferred coefficients for the basis functions.
                - self.force_ansatz: Callable function representing the inferred force field.
                - self.force_G: The normalization matrix used in the inference process.

        """
        if M_mode == 'auto':
            # Automatic presets:
            if self.Lambda_trace > 0. :
                # Assume measurement noise dominates biases:
                if basis_linear_gradient is not None:
                    M_mode = 'Strato'
                    G_mode = 'shift'
                else:
                    M_mode = "Ito-shift"
                    G_mode = "shift"
            else:
                # Assume time discretization dominates biases:
                M_mode = "Ito"
                G_mode = "trapeze"
            if self.verbosity >= 1:
                print(f"Automatically selecting force inference parameters: M_mode {M_mode}, G_mode {G_mode} (Lambda trace: {self.Lambda_trace}). ")
                
        # Store arguments for downstream use:
        self.__force_M_mode__ = M_mode
        self.__force_G_mode__ = G_mode
        self.force_basis_names = basis_names
        self.force_basis_linear = basis_linear
        self.force_basis_linear_gradient = basis_linear_gradient

        # Call the base class template:
        self._infer_force_linear_template()
        
    def infer_diffusion_linear(self, 
                               basis_linear: callable,
                               * ,
                               M_mode: str = 'auto',
                               G_mode: str = "rectangle",
                               basis_names = None) -> None:
        """
        Fit the diffusion field as a linear combination of basis functions.
        
        This method computes the coefficients of the diffusion tensor field (`self.diffusion_coefficients`) using
        the provided basis functions. The diffusion tensor is represented as:

            diffusion_ansatz(x, mask) = sum_a basis_linear(x, mask)[:,a] * diffusion_coefficients[a]

        Args:
            basis_linear (callable): im, mask -> iamn
                The basis functions for diffusion inference. Encoded as a callable that takes input
                `x` (array of shape `(N_particles, dimension)`) and `mask` (array of shape `(N_particles,)`) and returns an array of shape
                `(N_particles, N_basis_functions, dimension, dimension)`.

            M_mode (str):
                The method used for local diffusion tensor estimation and moments computation.
                See _diffusion_estimator documentation for additional information.

            G_mode (str):
                The method used to compute the normalization matrix `G`. Not investigated extensively yet for diffusion inference.

            basis_names (list[str]): descriptors for the basis functions. Generically filled if None.

        Updates:
            self.diffusion_coefficients: The inferred coefficients for the diffusion basis functions.
            self.diffusion_ansatz: Callable representing the inferred diffusion tensor field.
            self.diffusion_G: The normalization matrix used in the inference process.
        
        Note:
            This ansatz is not guaranteed to be nonnegative.
        """
        if M_mode == 'auto':
            # Automatic presets:
            if self.Lambda_trace > 0. :
                # Assume measurement noise dominates biases:
                M_mode = 'Vestergaard'
                G_mode = "rectangle"
            else:
                # Assume time discretization dominates biases:
                M_mode = "WeakNoise"
                G_mode = "rectangle"
            if self.verbosity >= 1:
                print(f"Automatically selecting diffusion inference parameters: M_mode {M_mode}, G_mode {G_mode} (Lambda trace: {self.Lambda_trace}). ")

        # Store the basis functions and method for later use
        self.diffusion_basis_linear = basis_linear
        self.__diffusion_M_mode__ = M_mode
        self.__diffusion_G_mode__ = G_mode
        self.diffusion_basis_names = basis_names

        # Call the base class template:
        self._infer_diffusion_linear_template()

    def simulate_bootstrapped_trajectory(self, key, oversampling = 1, simulate = True):
        """
        Simulate an overdamped Langevin trajectory with the inferred force and diffusion fields.

        This function generates a trajectory using the ansatz force field and diffusion tensor inferred
        from the input data, matching the original time series and initial conditions.

        Args:
            key: JAX random key for generating noise in the simulation.
            oversampling (int, optional): Factor for oversampling (i.e. number of intermediate simulated points between two recorded points). Defaults to 1.
            simulate: if True, performs the simulation with the first data point as initial position;
                      if False, returns an uninitialized object which can be simulated with flexible initial position and parameters.

        Returns:
            OverdampedLangevinProcess: Simulated Langevin process object.
        """
        from SFI.SFI_Langevin import OverdampedLangevinProcess

        if hasattr(self, 'diffusion_ansatz'):
            # Variable diffusion tensor
            if self.verbosity >= 1: print("Simulating bootstrapped trajectory with state-dependent diffusion.")
            X = OverdampedLangevinProcess(
                self.force_ansatz,
                self.diffusion_ansatz,
                F_is_multiparticle=True,
                D_is_multiparticle=True
            )
        else:
            # Constant diffusion tensor
            if self.verbosity >= 1: print("Simulating bootstrapped trajectory assuming constant diffusion.")
            X = OverdampedLangevinProcess(
                self.force_ansatz,
                self.diffusion_average,
                F_is_multiparticle=True
            )
        if simulate:
            # Initialize and simulate the process
            X.initialize(self.data.X[0],params_F=None,params_D=None)
            X.simulate(self.data.dt, self.data.Nsteps, key, oversampling=oversampling, prerun=0)

        return X


    #################################################################
    ################       BACKEND       ############################
    #################################################################
        
    # Hooks required by BaseLangevinInference:
    def _force_G_matrix(self)-> jnp.ndarray:
        b_left = self.force_basis_linear
        b_right = jit(lambda X,mask : self.force_basis_linear(X,mask) @ self.A_inv )
        return self.__G_matrix__(b_left,b_right,self.__force_G_mode__,'iam,ibm->iab')

    def _force_moments(self):
        """ Backend for force moments computation (uses stored self.force_basis_linear and options). """
        
        if self.__force_M_mode__ == 'Ito' or self.__force_M_mode__ == 'Ito-shift' :
            if self.verbosity >= 2: print("Computing Ito force coefficients.")
            
            @jit
            def xdot_b(t):
                # Define shifted or non-shifted basis function - velocity product.
                dXdt = self.data.dX[t]/self.data.dt
                if self.__force_M_mode__ == 'Ito-shift':
                    bt = self.force_basis_linear(self.data.X[t]-self.data.dX_minus[t],self.data.get_mask_at(t))
                else:
                    bt = self.force_basis_linear(self.data.X[t],self.data.get_mask_at(t))
                return jnp.einsum('im,mn,ian->ia', dXdt, self.A_inv, bt, optimize=True)

            if self.verbosity >= 2: print("Computing Ito integral.")
            # Compute the average < dx/dt b(xt) >
            return self.data.trajectory_average(xdot_b,verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)

        elif self.__force_M_mode__ == 'Strato':
            if not hasattr(self,"force_basis_linear_gradient"):
                raise RuntimeError("Strato force inference requires basis_linear_gradient. Provide it or fall back to Ito-shift method.")
            if self.verbosity >= 2: print("Computing Strato force coefficients.")
            # Define instantaneous term for velocity coefficients
            @jit
            def xdot_circ_b(t):
                dXdt = self.data.dX[t] / self.data.dt 
                bx = 0.5 * (
                    self.force_basis_linear(self.data.X[t], self.data.get_mask_at(t)) +
                    self.force_basis_linear(self.data.X[t] + self.data.dX[t], self.data.get_mask_at(t))
                )
                return jnp.einsum('im,mn,ian->ia', dXdt, self.A_inv, bx, optimize=True)

            # Store this intermediate, potentially useful by-product
            # (see current estimation, entropy production
            # estimation... in the 2020 PRX):
            self.force_v_moments = self.data.trajectory_average(xdot_circ_b,verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)

            if self.verbosity >= 2: print("Computing Strato gradient term.")
            # Compute w_alpha = - < D(t) D_av^-1 grad b_alpha >
            D_inst = self._diffusion_estimator("Vestergaard")
            D_grad_b_average = self.data.trajectory_average(
                jit(lambda t: jnp.einsum(
                    'imn,no,iaom->ia',
                    D_inst(t),
                    self.A_inv,
                    self.force_basis_linear_gradient(self.data.X[t], self.data.get_mask_at(t)),
                    optimize=True
                )),
                verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)

            force_moments = self.force_v_moments - D_grad_b_average

            return force_moments
            

    def _diffusion_G_matrix(self)-> jnp.ndarray: 
        return self.__G_matrix__(self.diffusion_basis_linear,
                                 self.diffusion_basis_linear,
                                 self.__diffusion_G_mode__, 'iamn,ibmn->iab')

    def _diffusion_moments(self) -> jnp.ndarray: 
        if self.verbosity >= 2: print("Computing diffusion linear moments.")
        # Parse the local diffusion estimator
        D_local_t = self._diffusion_estimator(self.__diffusion_M_mode__)
        @jit
        def bD(t):
            return jnp.einsum('iamn,inm->ia',
                              self.diffusion_basis_linear(self.data.X[t],self.data.get_mask_at(t)),
                              D_local_t(t))
        return self.data.trajectory_average(bD,verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)

    def _update_force_ansatz(self) -> None : 
        # Define the force field as a callable function
        def force_ansatz(x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
            """
            Compute the inferred force field at given positions.

            Args:
                x (jnp.ndarray): Particle positions, shape (N_particles, dimension).
                mask (jnp.ndarray, optional): Boolean mask indicating active particles. 
                    Defaults to all True.

            Returns:
                jnp.ndarray: Inferred force field, shape (N_particles, dimension).
            """
            result = self.force_coefficients_full.dot(self.force_basis_linear(x, mask))
            # Mask out inactive particles by setting their values to NaN
            if mask is None:
                return result
            else:
                return jnp.where(mask[:, None], result, jnp.nan)
        self.force_ansatz = jit(force_ansatz)

    def _update_diffusion_ansatz(self) -> None : 
        # Define the diffusion tensor ansatz as a callable function
        @jit
        def diffusion_ansatz(x: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
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
            result = jnp.einsum('iamn,a->imn', self.diffusion_basis_linear(x, mask), self.diffusion_coefficients_full)
            # Mask out inactive particles by setting their values to NaN
            return jnp.where(mask[:, None, None], result, jnp.nan)
        self.diffusion_ansatz = diffusion_ansatz
        
    def __G_matrix__(self, b_left: callable, b_right: callable, G_mode: str, einsum_string: str, subsampling : int = 1) -> jnp.ndarray:
        """Compute the normalization matrix G = < b_left b_right > (Gram matrix) based on the selected mode.

        Args:
            basis_linear (callable): im,mask->ia..  Function to compute the basis values.

            G_mode (str): Mode for G computation ('rectangle' | 'trapeze' | 'shift').
        
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
        if self.verbosity >= 2: print("Computing G matrix with einsum:",einsum_string)
        if G_mode== "rectangle":
            # The classic normalization matrix G=<bi(xt) bj(xt)> from
            # the 2020 PRX.
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.get_mask_at(t))
                br = b_right(self.data.X[t],self.data.get_mask_at(t))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        elif G_mode=="trapeze":
            # Trapezoidal integration G=<bi(xt) (bj(xt)+bj(xt+dt))/2 >
            # proposed by A. Gerardos [PRR 2024]. Robust for larger
            # dt.
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.get_mask_at(t))
                br = 0.5 * ( b_right(self.data.X[t],self.data.get_mask_at(t))
                          +  b_right(self.data.X_plus(t),self.data.get_mask_at(t)))                
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        elif G_mode=="shift":
            # Shifted integration G=<bi(xt) bj(xt+dt) > to reduce
            # measurement noise bias AND (to a lower extent) finite
            # dt bias.
            def Ginst(t):
                bl = b_left(self.data.X[t],self.data.get_mask_at(t))
                br = b_right(self.data.X_plus(t),self.data.get_mask_at(t))
                return jnp.einsum(einsum_string,bl,br,optimize=True)
        else:
            raise KeyError("Wrong G_mode argument")
        return self.data.trajectory_average(jit(Ginst),verbosity=self.verbosity-2,
                                            max_memory_gb=self.max_memory_gb,subsampling=subsampling)


    def _diffusion_estimator(self, method: str) -> callable:
        """These local diffusion estimators provide instantaneous,
        noisy estimates of the diffusion tensor. Noise is O(1) and
        depends on trajectory length and sampling.  Selects and
        applies the appropriate estimation method based on the
        provided `method` argument.

        Args:
            method (str): Diffusion estimation method. Options include:
                - "MSD": Standard mean squared displacement estimator.
                - "Vestergaard": Noise-robust estimator using past and future steps.
                - "WeakNoise": Removes persistent velocity bias for large dt systems.
                - "Fcorrected": Accounts for deterministic drift by subtracting inferred force.
                - "Lambda": Estimates measurement noise based on anti-correlated displacement.

        Returns:
            callable: int t -> jnp.ndarray: Estimated diffusion tensor at time t.

        """

        dX = lambda t : self.data.dX[t]  # Displacement at time step t
        dX_minus = lambda t : self.data.dX_minus[t] 

        if method in [ "Vestergaard", "WeakNoise", "Lambda" ] and not hasattr(self.data, "dX_minus"):
            raise RuntimeError(f"Need to compute dX_minus at the StochasticTrajectoryData initialization to use method {method}.")

        estimators = {
            "MSD": lambda t : jnp.einsum('im,in->imn', dX(t), dX(t)) / (2 * self.data.dt),

            "Vestergaard": lambda t : (
                # Uses a three-point estimator to mitigate measurement noise
                jnp.einsum('im,in->imn', dX(t) + dX_minus(t), dX(t) + dX_minus(t)) +
                jnp.einsum('im,in->imn', dX(t), dX_minus(t)) +
                jnp.einsum('im,in->imn', dX_minus(t), dX(t))) / (4 * self.data.dt),

            "WeakNoise": lambda t : (
                # Removes systematic velocity contributions in large dt regimes
                jnp.einsum('im,in->imn', dX(t) - dX_minus(t), dX(t) - dX_minus(t)) / (4 * self.data.dt)
            ),

            "Fcorrected": lambda t : (
                # Accounts for deterministic dynamics by subtracting the inferred force contribution
                jnp.einsum('im,in->imn', dX(t) - self.data.dt * self.force_ansatz(self.data.X[t]), 
                                          dX(t) - self.data.dt * self.force_ansatz(self.data.X[t])) / (2 * self.data.dt)
            ),

            "Lambda": lambda t : (
                # Estimates measurement noise by using anti-correlated displacement terms
                -(  jnp.einsum('im,in->imn', dX(t), dX_minus(t)) +
                    jnp.einsum('im,in->imn', dX_minus(t), dX(t))) / 2 ) 
        }

        if method not in estimators:
            raise KeyError(f"Invalid diffusion estimation method: {method}")

        return jit(estimators[method])

        
    def _compute_sampled_values(self,function,indices,data):
        # Tiny helper for sampled average
        return vmap(function)(data.X[indices])



