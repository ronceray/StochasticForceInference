from abc import ABC, abstractmethod
from jax import jit,vmap
import jax.numpy as jnp
from functools import partial
from SFI.SFI_data import StochasticTrajectoryData
from SFI.SFI_sparsity import SparseModelSelector
from SFI import SFI_utils

class BaseLangevinInference(ABC): 
    """Stochastic Force Inference main class

    This class provides tools for inferring force (drift) and
    diffusion tensors from stochastic trajectory data based on
    Langevin dynamics. It contains the shared logic for Overdamped and
    Underdamped Langevin inference.

    These subclasses must implement a handful of hooks that depend on
    the physics (e.g. whether velocities are observed). The details of
    the physics assumptions and definitions, as well as extensive doc
    strings, are given in the headers of these classes.

    Key Features:
    --------------
    - Force Inference:
      - Linear combination of basis functions (`infer_force_linear`).
      - Nonlinear parametric functions (`infer_force_nonlinear` [COMING SOON]).
    - Diffusion Inference:
      - Constant diffusion estimation (`infer_diffusion_constant`).
      - State-dependent diffusion with basis functions (`infer_diffusion_linear` and `infer_diffusion_nonlinear` [COMING SOON]).
    - Sparsification:
      - Force sparsification for linear inference using the class SparseModelSelector, implementing PASTIS and other information criteria.
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
    3. Optionally sparsify the results using the `*_sparsify` routines to mitigate overfitting.
    4. Optionally compute error estimates and/or compare with exact data for validation.

    Indices Convention:
    -------------------
    The code uses jnp.einsum for array manipulation, with a consistent index naming scheme for clarity:
    - `t` : Time index, = 0..Ntimesteps-1
    - `a, b, c...` : Basis function indices, = 0..Nfunctions - 1.
    - `m, n, o...` : State / spatial indices, = 0..dim-1.
    - `i, j...` : Particle indices, used only for systems of interacting particles (size Nparticles, or 1 if there is no particles structure).
    We also use these indices as shortcuts for array shapes. For
    instance `basis_linear : im -> iam` reads `basis_linear has input
    a jnp.array of shape (Nparticles,dim) and outputs a jnp.array of
    shape (Nparticles,Nfunctions,dim)`.

    Dependencies:
    -------------
    - JAX for accelerated numerical computations and auto-differentiation.
    - `SFI_data.StochasticTrajectoryData` for input formatting and trajectory management.
    - `SFI_sparsity`, for linear inference: solving, information measurement, and model selection.
    - `SFI_Langevin`, to simulate bootstrapped trajectories using inferred fields

    Verbosity:
    ----------
    Verbosity level controls standard output:
         0 -> silent
         1 -> print the steps of the inference; compact report for sparse solver
         2 -> extensive reports for time-averages and sparsification
         3 -> debugging

    """
    
    
    def __init__(self, data: StochasticTrajectoryData, * , verbosity = 1, max_memory_gb=1.0):
        """ Initialize the object. """
        self.data = data
        self.verbosity = verbosity          # A global verbosity level.
        self.max_memory_gb = max_memory_gb  # Max memory to use when computing averages (approximate).

    def _infer_force_linear_template(self):
        """Infer the force field as a linear combination of basis
        functions (linear regression).  Internal helper – DO NOT CALL
        DIRECTLY.

        This is the *Template Method* that implements the steps common
        to both overdamped and under-damped variants. See the detailed
        docstrings of the public methods in these classes.
        
        Args:
            None, uses stored attributes from the subclass call.
        
        Outputs:
            Updates the following attributes:
                - self.force_sparsifier: SparseModelSelector object for further model selection.
                - self.force_coefficients: The inferred coefficients for the basis functions.
                - self.force_ansatz: Callable function representing the inferred force field.
                - self.force_G: The normalization matrix used in the inference process.

        """
        if hasattr(self,"force_G_full"):
            raise RuntimeError("Force has already been inferred on this object - create a new instance to re-infer.")
        
        # Call subclass-specific hooks to compute the normal matrix and moments:
        self.force_G_full = self._force_G_matrix()
        self.force_moments = self._force_moments()

        # Pretty-printing utility:
        if self.force_basis_names is None:
            self.force_basis_names,_ = SFI_utils.make_variable_names(len(self.force_moments),symbol="b")

        # Use the SparseModelSelector class to solve the linear system
        # (and potentially prepare for model selection).
        self.force_sparsifier = SparseModelSelector(M=self.data.tauN * self.force_moments, G=self.data.tauN * self.force_G_full)
        self._update_force_coefficients(self.force_sparsifier.total_C)


    def sparsify_force(self, *, criterion = 'PASTIS', p = 0.05, beam_width = 3, max_k = None):
        """Front-end for force coefficients sparsification. Performs a
        Pareto-front construction via beam search, then maximizes the
        selected information criterion along the front and updates
        force coefficients and ansatz accordingly.

        """
        if max_k is None:
            max_k = self.force_sparsifier.p
        self.force_sparsifier.build_pareto_front(max_k=max_k, beam_width=beam_width, 
                                                 aic_patience=2, report_time=True, verbosity=self.verbosity)
        k, support, score, coeffs = self.force_sparsifier.select_by_ic(criterion, p_param=p)
        self._update_force_coefficients(coeffs,support)

                       
    def compute_diffusion_constant(self, method: str = 'auto') -> None:
        """
        Compute the constant part (non-state-dependent) of the diffusion tensor.

        This method estimates the average diffusion tensor (`self.diffusion_average`) using a specified
        local estimator. The result is averaged over the entire trajectory.

        Args:
            method (str): The method used for the local diffusion matrix estimation.
            See _diffusion_estimator documentation for additional information.
        """
        if hasattr(self,"diffusion_average"):
            raise RuntimeError("Diffusion has already been inferred on this object - create a new instance to re-infer.")

        self.Lambda = self.data.trajectory_average(self._diffusion_estimator("Lambda"),
                                                   verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)
        self.Lambda_trace = self.Lambda.trace()
        if self.verbosity >= 1: print(f"Measurement noise trace: {self.Lambda_trace}.")

        if self.verbosity >= 2: print("Computing diffusion constant.")
        # Select the local diffusion matrix estimator using subclass hook:
        diffusion_local_t = self._diffusion_estimator(method)

        # Compute the averaged diffusion tensor
        self.diffusion_average = self.data.trajectory_average(diffusion_local_t,verbosity=self.verbosity-2,max_memory_gb=self.max_memory_gb)

        # A is a normalization matrix for quasi-likelihood estimates.
        self.A = 2 * self.diffusion_average

        # Check that A is positive definite (negative eigenvalues are
        # possible with noisy estimator).
        eigvals = jnp.linalg.eigvals(self.A)
        if jnp.any(eigvals <= 0):
            print("Normalization matrix A is not positive definite: non-positive eigenvalues detected. Change method for diffusion constant.")

        self.A_inv = jnp.linalg.inv(self.A)
        self.sqrtA = jnp.real(SFI_utils.sqrtm_psd(self.A))
        self.sqrtA_inv = jnp.linalg.inv(self.sqrtA)
        

    def _infer_diffusion_linear_template(self) -> None:
        """
        Fit the diffusion field as a linear combination of basis functions. 
        Internal helper – DO NOT CALL DIRECTLY.

        This is the *Template Method* that implements **all the steps
        common** to both overdamped and under-damped variants. See the
        detailed docstrings of the public methods in these classes.

        Args:
            basis_linear (callable): state, mask -> iamn

            M_mode (str):
                The method used for diffusion moments computation.

            G_mode (str):
                The method used to compute the normalization matrix `G`.

            basis_names (list[str]): descriptors for the basis functions. Generically filled if None.

        Updates:
            self.diffusion_moments: The moments of the diffusion estimator on the basis functions.
            self.diffusion_coefficients: The inferred coefficients for the diffusion basis functions.
            self.diffusion_ansatz: Callable representing the inferred diffusion tensor field.
            self.diffusion_G: The normalization matrix used in the inference process.
        
        Note:
            This ansatz is not guaranteed to be nonnegative.
        """
        if hasattr(self,"diffusion_moments"):
            raise RuntimeError("Diffusion has already been inferred on this object - create a new instance to re-infer.")

        # Compute normalization matrix G and moments using subclass-specific hooks
        self.diffusion_G_full = self._diffusion_G_matrix()
        self.diffusion_moments = self._diffusion_moments()

        # Pretty printing utility:
        if self.diffusion_basis_names is None:
            self.diffusion_basis_names,_ = SFI_utils.make_variable_names(len(self.diffusion_moments),symbol="d")

        # Use the SparseModelSelector class to solve the linear system
        # (and potentially prepare for model selection).
        self.diffusion_sparsifier = SparseModelSelector(M=self.data.tauN * self.diffusion_moments,
                                                    G=self.data.tauN * self.diffusion_G_full)
        self._update_diffusion_coefficients(self.diffusion_sparsifier.total_C)
           

    def sparsify_diffusion(self, *, criterion = 'PASTIS', p = 0.05, beam_width = 3, max_k = None):
        """Front-end for diffusion coefficients sparsification. Performs a
        Pareto-front construction via beam search, then maximizes the
        selected information criterion along the front and updates
        diffusion coefficients and ansatz accordingly.

        """
        raise NotImplementedError("Diffusion sparsification not ready yet.")
        # if max_k is None:
        #     max_k = self.diffusion_sparsifier.p
        # self.diffusion_sparsifier.build_pareto_front(max_k=max_k, beam_width=beam_width, 
        #                                          aic_patience=2, report_time=True, verbosity=self.verbosity)
        # k, support, score, coeffs = self.diffusion_sparsifier.select_by_ic(criterion, p_param=p)
        # self._update_diffusion_coefficients(coeffs,support)

    ########################## ERROR ANALYSIS ###########################

    def compute_force_error(self):
        """
        Estimate sampling error for force inference.

        This method evaluates the covariance of the inferred force coefficients, the standard error,
        and computes the predicted normalized mean squared error (MSE) of the inferred force field. This analysis
        assumes that the sampling error dominates, and measurement noise or discretization biases are
        not explicitly addressed. It is common to OLI and ULI (by construction of the normal matrix G).

        Updates:
            self.force_coefficients_covariance (jnp.ndarray): Covariance matrix of the force coefficients.
            self.force_coefficients_stderr (jnp.ndarray): Standard error for each force coefficient.
            self.force_information (float): Estimated information content of the inferred force field.
            self.force_normalized_MSE (float): Normalized mean squared error of the inferred force field.

        """
        # Estimate the covariance of the force coefficients
        self.force_coefficients_covariance = (2 / self.data.tauN) * self.force_G_pinv

        # Calculate the standard error for each force coefficient
        self.force_coefficients_stderr = jnp.einsum('aa->a', self.force_coefficients_covariance)**0.5

        # Compute mean squared error of the force field
        force_MSE = float(jnp.einsum('ab,ba', self.force_G, self.force_coefficients_covariance))

        # Compute normalized MSE
        force_capacity = 0.5 * self.force_coefficients_full @ self.force_moments
        self.force_information = float(force_capacity * self.data.tauN)
        self.force_predicted_MSE = float(force_MSE / force_capacity)

    def compute_diffusion_error(self):
        """

        """
        raise NotImplementedError("Diffusion error estimation not ready yet.")


    def print_report(self):
        """
        Print a summary report of the inference results.

        Provides insights into the inferred diffusion and force fields, along with error metrics
        such as sampling error, trajectory length, discretization bias, and measurement noise.
        """
        print("\n  --- StochasticForceInference Report --- ")

        # Average diffusion tensor
        print("Average diffusion tensor:\n", self.diffusion_average)

        # Measurement noise tensor
        print("Measurement noise tensor:\n", self.Lambda)

        # Entropy production
        if hasattr(self, 'DeltaS'):
            print("Entropy production: inferred/bootstrapped error:", self.DeltaS, self.error_DeltaS)

        # Force inference metrics
        if hasattr(self, 'force_predicted_MSE'):
            print("Force estimated information:", self.force_information)
            print("Force: estimated normalized mean squared error (sampling only):", self.force_predicted_MSE)
            # To add: bias estimates

        # Diffusion error metrics
        if hasattr(self, 'diffusion_predicted_MSE'):
            print("Diffusion estimated information:", self.diffusion_information)
            print("Diffusion: estimated normalized mean squared error (sampling only):", self.diffusion_predicted_MSE)
            # To add: bias estimates

        if hasattr(self, 'force_coefficients'):
            coeffs_stderr = self.force_coefficients_stderr if hasattr(self, 'force_coefficients_stderr') else None
            print("Force model:\n", SFI_utils.simple_function_print(self.force_basis_names,
                                                                    self.force_support,
                                                                    self.force_coefficients,
                                                                    coeffs_stderr = coeffs_stderr),"\n")


    def compare_to_exact(self, 
                         data_exact: StochasticTrajectoryData = None, 
                         force_exact: callable = None, 
                         diffusion_exact = None,  # callable | jnp.ndarray
                         maxpoints: int = 1000) -> None:
        """
        Compare inferred fields to exact data.

        This routine computes the mean squared error (MSE) between inferred
        and exact force/diffusion fields for artificial data with known models,
        and the normalized variant (normalized with mean squared inferred value).

        Args:
            data_exact (StochasticTrajectoryData, optional): Exact trajectory data. Defaults to `self.data`.
            force_exact (callable, optional): Exact force function. state -> im.
            diffusion_exact (callable | jnp.ndarray, optional): Exact diffusion tensor. If callable: state -> imn
                If constant tensor, pass directly as `jnp.ndarray` of shape `(dimension, dimension)`.
            maxpoints (int): Maximum number of points to sample for comparison.

        Updates:
            self.MSE_force (float): Mean squared error for the force field.
            self.MSE_diffusion (float): Mean squared error for the diffusion tensor.
        """
        # Use exact data or fall back to the input trajectory
        data_exact = data_exact if data_exact is not None else self.data

        # Select sample indices for comparison
        indices = data_exact.time_indices[::max(1, data_exact.Nsteps // max(1, maxpoints // data_exact.Nmaxparticles))]
        ivals = jnp.arange(0, len(indices) - 1, 1)

        if self.verbosity >= 1:
            print("Comparing to exact data...")

        # Compare force fields if applicable
        if hasattr(self, 'force_ansatz') and force_exact is not None:
            self.exact_force_values = self._compute_sampled_values(force_exact,indices,data_exact)
            self.ansatz_force_values = self._compute_sampled_values(self.force_ansatz,indices,data_exact)
            self.MSE_force = jnp.sum(vmap(lambda i: jnp.einsum('im,in,mn->', self.exact_force_values[i] - self.ansatz_force_values[i],
                                                               self.exact_force_values[i] - self.ansatz_force_values[i],
                                                               self.A_inv))(ivals)) 
            self.NMSE_force = self.MSE_force / jnp.sum(vmap(lambda i: jnp.einsum('im,in,mn->',
                                                                                 self.ansatz_force_values[i],
                                                                                 self.ansatz_force_values[i],
                                                                                 self.A_inv))(ivals))
            
            if self.verbosity >= 1:
                print("   Actual normalized mean squared error on force:", self.NMSE_force)

        # Prepare callable versions of diffusion tensors
        def make_callable(tensor) -> callable:
            if callable(tensor):
                return tensor
            elif hasattr(tensor, 'shape'):  # Treat constant tensors as callable
                def extended_tensor(x: jnp.ndarray, v = None) -> jnp.ndarray:
                    n_particles, d = x.shape
                    return jnp.tile(tensor[None, :, :], (n_particles, 1, 1))
                return extended_tensor
            else:
                raise ValueError("diffusion_exact must be either callable or a dxd array.")

        if diffusion_exact is not None:
            diffusion_exact_callable = make_callable(diffusion_exact)
            diffusion_ansatz_callable = make_callable(self.diffusion_ansatz if hasattr(self, 'diffusion_ansatz') else self.diffusion_average)
            
            self.exact_diffusion_values  =  self._compute_sampled_values(diffusion_exact_callable,indices,data_exact)
            self.ansatz_diffusion_values =  self._compute_sampled_values(diffusion_ansatz_callable,indices,data_exact)
            self.MSE_diffusion = jnp.sum(vmap(lambda i: jnp.einsum('imn,iop,no,pm->',
                                                                   self.exact_diffusion_values[i]-self.ansatz_diffusion_values[i],
                                                                   self.exact_diffusion_values[i]-self.ansatz_diffusion_values[i],
                                                                   self.A_inv,self.A_inv))(ivals))
            self.NMSE_diffusion = self.MSE_diffusion /  jnp.sum(vmap(lambda i: jnp.einsum('imn,iop,no,pm->',self.ansatz_diffusion_values[i],self.ansatz_diffusion_values[i],self.A_inv,self.A_inv))(ivals))
            if self.verbosity >= 1:
                print("   Actual mean squared error on diffusion:", self.NMSE_diffusion)
        if self.verbosity >= 1: print(" ")



    #################################################################
    ################       BACKEND       ############################
    #################################################################

    def _update_force_coefficients(self, coeffs, support = None, jit_ansatz = True):
        """Writes or updates the force coefficients, force ansatz, and
        stores the Gram matrix G. Use on the coefficients computed
        with the force_sparsifier. Optional argument "support"
        indicates which basis functions correspond to the coefficients
        (default: all of them).

        Args:
            coeffs (float array): the force coefficients corresponding to the support elements.
            support (None or int list/array): the basis function indices corresponding to the
                    coefficients; all other coefficients are assumed zero.
            jit_ansatz: whether to jit-compile the resulting force ansatz.

        """
        self.force_coefficients = coeffs
        if support is None:
            """ Full support. """
            self.force_support = jnp.arange(self.force_sparsifier.p)
        else:            
            self.force_support = jnp.array(support)
        self.force_G = self.force_G_full[jnp.ix_(self.force_support,self.force_support)]
        self.force_G_pinv = SFI_utils.stable_pinv(self.force_G)
        # Sparse coeffs on the complete basis:
        self.force_coefficients_full = jnp.zeros_like(self.force_moments)
        if len(self.force_support) > 0:
            self.force_coefficients_full = self.force_coefficients_full.at[self.force_support].set(coeffs)
        # Call the ansatz-constructing subclass-specific hook:
        self._update_force_ansatz()


    def _update_diffusion_coefficients(self, coeffs, support = None, jit_ansatz = True):
        """Writes or updates the diffusion coefficients, diffusion ansatz, and
        stores the Gram matrix G. Use on the coefficients computed
        with the diffusion_sparsifier. Optional argument "support"
        indicates which basis functions correspond to the coefficients
        (default: all of them).

        Args:
            coeffs (float array): the diffusion coefficients corresponding to the support elements.
            support (None or int list/array): the basis function indices corresponding to the
                    coefficients; all other coefficients are assumed zero.
            jit_ansatz: whether to jit-compile the resulting diffusion ansatz.

        """
        self.diffusion_coefficients = coeffs
        if support is None:
            """ Full support. """
            self.diffusion_support = jnp.arange(self.diffusion_sparsifier.p)
        else:            
            self.diffusion_support = jnp.array(support)
        self.diffusion_G = self.diffusion_G_full[jnp.ix_(self.diffusion_support,self.diffusion_support)]
        self.diffusion_G_pinv = SFI_utils.stable_pinv(self.diffusion_G)
        # Sparse coeffs on the complete basis:
        self.diffusion_coefficients_full = jnp.zeros_like(self.diffusion_moments)
        self.diffusion_coefficients_full = self.diffusion_coefficients_full.at[self.diffusion_support].set(coeffs)
        # Call the ansatz-constructing subclass-specific hook:
        self._update_diffusion_ansatz()



    def _detach_from_jax(self):
        """Convert all JAX arrays inside this object to NumPy arrays
        to prevent memory leaks. Use this before deleting this object,
        as the Jax traces might persist otherwise. Important when
        performing a large number of inference runs in the same run
        (e.g. for benchmarking the method over many
        parameters/trajectories).

        """
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


    

    ### Subclass hooks ###

    @abstractmethod
    def _force_G_matrix(self)-> jnp.ndarray: ...

    @abstractmethod
    def _force_moments(self) -> jnp.ndarray: ...

    @abstractmethod
    def _diffusion_G_matrix(self)-> jnp.ndarray: ...

    @abstractmethod
    def _diffusion_moments(self) -> jnp.ndarray: ...

    @abstractmethod
    def _diffusion_estimator(self, method) -> callable: ...
    
    @abstractmethod
    def _update_force_ansatz(self) -> None : ...

    @abstractmethod
    def _update_diffusion_ansatz(self) -> None : ...

    @abstractmethod
    def _compute_sampled_values(self,function,indices,data) -> jnp.ndarray : ...
