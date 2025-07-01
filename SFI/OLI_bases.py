
""" A library of fitting bases for Stochastic Force Inference. """



import jax.numpy as jnp
from jax import random,jit,vmap,jacfwd,jacrev
import SFI.SFI_utils

def basis_selector(basis, dimension, output="vector", symmetric = True, isotropic = False):
    """
    Selects the appropriate basis function for stochastic force inference.

    This function is specifically designed for **linear inference**, providing a structured
    way to build basis functions for force estimation (`output="vector"`) and diffusion
    estimation (`output="tensor"`). It supports various basis types, including polynomial,
    Fourier, grid binning, and interaction-based bases.

    The selected output function should conform to the structure:
        - **basis_linear (callable):** `im, mask -> iam`
          where `i` represents particles, `a` represents basis functions, and `m` is the dimension.
        - **basis_linear_gradient (callable):** `im, mask -> iamn`
          providing gradients with respect to spatial dimensions. If a gradient is not provided,
          it is computed automatically using JAX autodiff (`jacfwd`).
    
    This function is responsible for incorporating the **particle structure** into custom 
    bases. If a function is provided in the form `m -> am` (for vector) or `m -> a` (for scalar),
    this function ensures that it is properly transformed into a structure that accounts for 
    multiple particles, making it compatible with inference requirements.

    Args:
        basis (dict):
            Dictionary specifying the basis type and required parameters.
            **Non-interacting Bases:**
                - "polynomial": Requires 'order' (int). Approximates smooth functions.
                - "Fourier": Requires 'order' (int), 'center' (list), 'width' (list).
                  Suitable for periodic functions.
                - "grid_binning": Requires 'order' (int), 'center' (list), 'width' (list).
                  Useful for discretized function approximations.
                - "custom_scalar": Requires 'functions' (callable), optionally 'gradient'.
                  Allows user-defined scalar bases, ensuring particle structure incorporation.
                  If no gradient is provided, it is computed using JAX autodiff.
                - "custom_vector": Requires 'functions' (callable), optionally 'gradient'.
                  Used for vectorial basis functions with particle structure.
                  If no gradient is provided, it is computed using JAX autodiff.
                - "custom_tensor": Requires 'functions' (callable), optionally 'gradient'.
                  Used for tensorial basis functions in diffusion inference.
                  If no gradient is provided, it is computed using JAX autodiff.
            
            **Interacting Bases:**
                - "particles_pair_interaction": Requires 'pair' (callable), 'single' (callable).
                  Models interactions between particles.
                - "self_propelled_particles": Requires 'kernels_radial' (list), 'kernels_angular' (list),
                  'polynomial_order' (int). Models active particle systems.
        
        dimension (int):
            Number of spatial dimensions.
        output (str, optional):
            Specifies whether to return a vector basis (`"vector"` for force estimation) 
            or a tensor basis (`"tensor"` for diffusion estimation). Defaults to "vector".

    Returns:
        tuple:
            - funcs (callable): Selected basis function.
            - grad (callable): Gradient of the basis function, automatically computed if not provided.

    Raises:
        KeyError: If an unknown basis type is provided.

    Example:
        >>> basis = {"type": "polynomial", "order": 3}
        >>> funcs, grad = basis_selector(basis, dimension=2, output="vector")
    """
    basis_type = basis["type"]

    if basis_type == "polynomial":
        poly,descr = polynomial_basis(dimension, basis["order"])
        funcs_and_grad, descriptor = scalar_basis(
            poly,
            dimension, output,
            symmetric=symmetric, isotropic=isotropic, descriptors = descr
        )

    elif basis_type == "Fourier":
        funcs_and_grad, descriptor = scalar_basis(
            Fourier_basis(dimension, basis["order"],
                          basis["center"], basis["width"]),
            dimension, output,
            symmetric=symmetric, isotropic=isotropic
        )

    elif basis_type == "grid_binning":
        funcs_and_grad, descriptor = scalar_basis(
            binning_basis(dimension, basis["order"],
                          basis["center"], basis["width"]),
            dimension, output,
            symmetric=symmetric, isotropic=isotropic
        )

    elif basis_type == "custom_scalar":
        funcs_and_grad, descriptor = scalar_basis(
            basis["functions"], dimension, output
        )

    elif basis_type in ("custom_vector", "custom_tensor"):
        funcs_and_grad, descriptor = prepare_basis_function(basis["functions"])

    elif basis_type == "particles_pair_interaction":
        funcs_and_grad, descriptor = pair_interaction_basis(
            basis["pair"], basis["single"]
        )

    elif basis_type == "self_propelled_particles":
        funcs_and_grad, descriptor = SPP_interaction_basis(
            basis["kernels_radial"],
            basis["kernels_angular"],
            basis["polynomial_order"]
        )

    else:
        raise KeyError(f"Unknown basis type: {basis_type}")

    return funcs_and_grad, descriptor




def scalar_basis(C_func, dimension, output_type, * , symmetric=False, isotropic=False, descriptors=None):
    """Transform a scalar basis function (m -> a) into a function and its
    gradient structured to be usable directly by SFI. It does the
    following operations:
    
    wrap_scalar_basis:
    
       * Wrap over the d dimensions: each scalar function is repeated
         for each vector / tensor component. In the tensor case some
         symmetries (symmetric / isotropic) can be enforced.

    prepare_basis_function:
       * Differentiate using JaX autodiff (note that the gradient will
         not necessarily be used, depending on the choice of inference
         parameters).

       * Vectorize by mapping over particles (most often only one
         "particle" is involved as scalar bases are for
         non-interacting particles, but necessary for SFI code
         architecture).

       * Mask missing data points / NaNs. If all data points are valid
         (mask=None) this operation is skipped.
    
    Args:
        C_func (callable): Scalar basis function taking input of shape (dimension,) and returning (nfunctions,).
        output_type (str): Either "vector" (for force) or "tensor" (for diffusion).
        symmetric (bool): Whether to enforce symmetry in the tensor.
        isotropic (bool): Whether to use an isotropic form (function * identity matrix).
    
    Returns:
    
        basis_function (callable): X (jnp.array(Nparticles,dimension)), mask (None or jnp.array(Nparticles,) -> jnp.array(Nparticles,nfunctions,dimension) [vector]
                                                                                                           or jnp.array(Nparticles,nfunctions,dimension,dimension) [tensor]
    
        basis_function_gradient (callable): X (jnp.array(Nparticles,dimension)), mask (None or jnp.array(Nparticles,) -> jnp.array(Nparticles,nfunctions,dimension,dimension) [vector]
                                                                                                           or jnp.array(Nparticles,nfunctions,dimension,dimension,dimension) [tensor]
    
    """
    wrapped_basis = wrap_scalar_basis(C_func, dimension, output_type, symmetric=symmetric,isotropic=isotropic)
    if descriptors is None:
        # Infer the number of functions through a dummy evaluation on
        # random numbers (less error-prone than ones or zeros):
        Nfunctions = C_func(random.normal(random.key(0),shape=(dimension,))).shape[0]
        descriptors, _ = SFI.SFI_utils.make_variable_names(Nfunctions,symbol="b")
    wrapped_descriptors = wrap_descriptors(descriptors, dimension, output_type, symmetric=symmetric,isotropic=isotropic)
    return prepare_basis_function(wrapped_basis),wrapped_descriptors


def wrap_scalar_basis(C_func, dimension, output_type, * , symmetric=False, isotropic=False):
    """
    Constructs a function that transforms a scalar basis function (m -> a) into 
    a structured vector or tensor basis function (m -> am or m -> amn).
    
    Args: same as scalar_basis.
    
    Returns:
        Callable function (dimension,) -> (nfunctions, dimension) or  (nfunctions, dimension, dimension) 
    """
    
    if output_type == "vector":
        def structured_function(state):
            """
            Computes the structured basis function for a given input.
            The function handles the expansion of scalar basis function values to structured output.
            """
            C_values = C_func(state) 
            n, = C_values.shape
            # Initialize the output array with zeros
            output = jnp.zeros((dimension * n, dimension))
            # Fill in the non-zero values
            for d in range(dimension):
                output = output.at[ d*n:(d+1)*n, d].set(C_values)
            return output
    
    elif output_type == "tensor":
        if isotropic:
            # Trivial case, just multiply by the identity matrix - for isotropic diffusion.
            structured_function = lambda state: jnp.eye(dimension)[None, :, :] * C_func(state)[:, None, None]
        
        elif symmetric:
            def structured_function(state):
                C_values = C_func(state) 
                n, = C_values.shape
                # Initialize the output array with zeros
                output = jnp.zeros((dimension * (dimension + 1) * n // 2, dimension, dimension))
                # Fill in the non-zero values
                counter = 0
                for d1 in range(dimension):
                    for d2 in range(d1+1):
                        output = output.at[counter*n:(counter+1)*n, d1,d2].set(C_values)
                        output = output.at[counter*n:(counter+1)*n, d2,d1].set(C_values)
                        counter += 1
                return output
        else:
            def structured_function(state):
                C_values = C_func(state) 
                n, = C_values.shape
                # Initialize the output array with zeros
                output = jnp.zeros((dimension * dimension * n, dimension, dimension))
                # Fill in the non-zero values
                counter = 0
                for d1 in range(dimension):
                    for d2 in range(dimension):
                        output = output.at[counter*n:(counter+1)*n, d1,d2].set(C_values)
                        counter += 1
                return output
    else:
        raise ValueError("Invalid output_type. Must be 'vector' or 'tensor'.")

    return jit(structured_function)

def wrap_descriptors(descriptors, dimension, output_type, * , symmetric=False, isotropic=False):
    """
    Wraps the corresponding descriptors for pretty printing.
    """
    _ , subscripts = SFI.SFI_utils.make_variable_names(dimension)
    
    if output_type == "vector":
        return [ d+f"·e{i}" for i in subscripts for d in descriptors ]
    
    elif output_type == "tensor":
        if isotropic:
            return descriptors
        
        elif symmetric:
            return [ d+(f"·(e{i}{j}+e{j}{i})" if d1 != d2 else f"·e{i}{j}") for d1,i in enumerate(subscripts) for d2,j in enumerate(subscripts[:d1+1]) for d in descriptors ]
        else:
            return [ d+f"·e{i}{j}" for i in subscripts for j in subscripts for d in descriptors ]
    else:
        raise ValueError("Invalid output_type. Must be 'vector' or 'tensor'.")

def vectorize_and_mask(func):
    """
    Constructs a function that applies vectorization and masking.
    
    This is necessary for SFI, which requires:
    - Vectorization over particles (required even when there is only one "particle" for code compatibility).
    - Potential masking to exclude missing particles from calculations.
    
    Args:
        func (callable): Function mapping state -> jnp.ndarray (shape '...')
    
    Returns:
        vectorized_and_masked_function (callable): Function mapping states, mask -> jnp.ndarray (shape 'Nparticles, ...')
    """
    @jit
    def vectorized_and_masked_function(states, mask=None):
        # Vectorize over particles
        func_values = vmap(func)(states)  
        if mask is not None:
            # Mask to zero out missing particles with extra axes to
            # generically broadcast over the output dimensions of the
            # function.
            extra_axes = tuple([ None for i in range(len(func_values.shape)-1) ])
            mask = mask[:, *extra_axes]
            func_values = jnp.where(mask, func_values, 0.0)  # Zero out masked particles
        return func_values
    return vectorized_and_masked_function

def prepare_basis_function(structured_basis):
    """
    Constructs a function that applies vectorization, differentiation, and masking.
    
    This is necessary for SFI, which requires:
    - Vectorization over particles (required even when there is only one "particle" for code compatibility).
    - Gradient, here obtained by automatic differentiation using `jacfwd`.
    - Potential masking to exclude missing particles from calculations.
    
    Args:
        structured_basis (callable): Function mapping (dimension,) -> (Nfunctions,dimension) [vector] or (Nfunctions,dimension,dimension) [tensor].
    
    Returns:
        Tuple of (structured basis function, structured basis function gradient).
    """
    C = vectorize_and_mask(structured_basis)
    C_x = vectorize_and_mask(jacfwd(structured_basis))
    return C,C_x


#### BASIC SCALAR BASES #### 


def polynomial_basis(dim: int,
                     order: int,
                     *,
                     variable_names: list[str] | None = None):
    """
    Build a monomial basis and its human-readable descriptors.

    Args:
        dim (int): Number of dimensions.
        order (int): Maximum polynomial order.
    """
    if variable_names is None:
        variable_names,_ = SFI.SFI_utils.make_variable_names(dim)

    # Generate monomial indices
    from itertools import combinations_with_replacement
    coeffs_list = [list(c) for o in range(order + 1) for c in combinations_with_replacement(range(dim), o)]
    
    # Find the max length of any term
    max_len = max(map(len, coeffs_list))

    # Pad with `-1` (invalid index) to ensure uniform shape (JAX-compatible)
    padded_coeffs = jnp.full((len(coeffs_list), max_len), -1, dtype=int)
    for i, c in enumerate(coeffs_list):
        if len(c)>0:
            padded_coeffs = padded_coeffs.at[i, :len(c)].set(c)  # JAX-compatible indexing

    padded_coeffs = jnp.array(padded_coeffs,dtype=int)  # Convert to JAX array

    @jit
    def basis_function(x):
        """
        Evaluates the polynomial basis functions at a given input.

        Args:
            x (jnp.ndarray): Input array of shape (dim,).

        Returns:
            jnp.ndarray: Evaluated basis functions.
        """
        x_selected = jnp.where(padded_coeffs >= 0, x[padded_coeffs], 1.0)  # Replace invalid indices with 1
        return jnp.prod(x_selected, axis=1)  # Compute product along each term

    # Construct the descriptors:
    from collections import Counter
    desc = []
    _SUPERS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    for idx_list in coeffs_list:
        if not idx_list:
            desc.append("1")
        else:
            # Join the corresponding symbols
            pieces = []
            for idx in sorted(Counter(idx_list)):               # deterministic order
                power = idx_list.count(idx)
                name  = variable_names[idx]
                if power == 1:
                    pieces.append(name)
                else:
                    pieces.append(name + str(power).translate(_SUPERS))
            desc.append("".join(pieces))

    return basis_function, desc


def Fourier_basis(dim: int, order: int, center: jnp.ndarray, width: jnp.ndarray):
    """
    Constructs a Fourier basis function for stochastic force inference.

    Args:
        dim (int): Number of spatial dimensions.
        order (int): Maximum Fourier order per dimension.
        center (jnp.ndarray): Center of the Fourier expansion (shape: (dim,)).
        width (jnp.ndarray): Scaling factor for the Fourier terms (shape: (dim,)).

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: A function mapping an input `X` (shape: (dim,))
        to a Fourier basis vector (shape: (2 * N_terms + 1,)).
    """

    # Generate multi-index coefficients (combinations of indices up to `order` per dimension)
    from itertools import combinations_with_replacement
    coeffs = jnp.array(
        [jnp.bincount(jnp.array(c), minlength=dim) for o in range(1, order + 1) 
         for c in combinations_with_replacement(range(dim), o)], dtype=int
    )

    N_terms = coeffs.shape[0]  # Number of Fourier basis functions

    @jit
    def Fourier(X: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Fourier basis vector for a given input.

        Args:
            X (jnp.ndarray): Input point (shape: (dim,)).

        Returns:
            jnp.ndarray: Fourier basis vector (shape: (2 * N_terms + 1,)).
        """
        # Normalize and scale X
        Xc = 2 * jnp.pi * (X - center) / width

        # Compute cosine and sine terms efficiently
        exponents = jnp.einsum('nd,d->n', coeffs, Xc)  # Compute dot product for all terms
        cos_terms = jnp.cos(exponents)
        sin_terms = jnp.sin(exponents)

        # Concatenate constant term (1) with cos and sin terms
        return jnp.concatenate([jnp.ones((1,)), cos_terms, sin_terms])

    return Fourier

def binning_basis(dim: int, order: int, center: jnp.ndarray, width: jnp.ndarray):
    """
    Constructs a binning basis function for discretizing a space into `order^dim` bins.

    Args:
        dim (int): Number of spatial dimensions.
        order (int): Number of bins per dimension.
        center (jnp.ndarray): Center of the binning grid (shape: (dim,)).
        width (jnp.ndarray): Width of the binning grid (shape: (dim,)).

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: A function mapping an input `X` (shape: (dim,))
        to a one-hot vector (shape: (order**dim,)) representing the bin it belongs to.
    """
    Nfuncs = order ** dim  # Total number of bins

    @jit
    def index_finder(x: jnp.ndarray) -> int:
        """
        Determines the 1D index corresponding to the bin in which `x` belongs.

        Args:
            x (jnp.ndarray): Input point (shape: (dim,)).

        Returns:
            int: The index of the corresponding bin, or Nfuncs (out of bounds).
        """
        # Compute bin indices in each dimension, ensuring they remain within valid bounds
        relative_pos = (x - (center - 0.5 * width)) / width
        projection_indices = jnp.floor(relative_pos * order).astype(int)

        # Check validity (JAX-compatible replacement for Python `if`)
        valid_mask = jnp.logical_and(projection_indices >= 0, projection_indices < order)
        valid = jnp.all(valid_mask)

        # Compute index normally if valid, else return Nfuncs (invalid index)
        index_values = projection_indices * order ** jnp.arange(dim)
        return jnp.where(valid, jnp.sum(index_values), Nfuncs)  # Use Nfuncs as out-of-bounds marker

    @jit
    def grid_function(X: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the binning basis function, returning a one-hot vector
        indicating which bin the input `X` belongs to.

        Args:
            X (jnp.ndarray): Input point (shape: (dim,)).

        Returns:
            jnp.ndarray: Vector indicating the bin (shape: (Nfuncs,)).
        """
        val = jnp.zeros(Nfuncs)
        n = index_finder(X)
        # JAX-compatible conditional setting (avoids Python `if`)
        return val.at[jnp.where(n < Nfuncs, n, 0)].set(jnp.where(n < Nfuncs, 1.0, 0.0))

    return grid_function




########## MULTIPARTICLE BASES ############

def construct_interaction_function(fpair,fsingle=None,concatenation_axis=None):
    """Construct a function that wraps the pair and single-particle
    interaction functions over all particles, in a vectorized way, and
    consistently applies masking. Generic inputs (works for force and
    force gradient).

    """
    def interaction_function(state, mask = None):
        # Compute all pairwise interactions at once
        # This creates an array of shape (N, N, ...)
        # TODO : test alternative using triangular evaluation then symmetrization
        all_pairs = vmap(vmap(fpair, in_axes=(None, 0)), in_axes=(0, None))(state, state)
        N = all_pairs.shape[0]
        # Generic list of extra axes to broadcast the masks into the right shape:
        #extra_axes = ( None for i in range(len(all_pairs.shape)-2) )
        extra_axes = tuple([ None for i in range(len(all_pairs.shape)-2) ])
        # Create a mask to zero out self-interactions and absent particles
        # with extra axes to broadcast over the output dimensions of fpair
        pair_mask = (jnp.eye(N) == 0)[:, :, *extra_axes] & (True if mask is None else (mask[:, None, *extra_axes] & mask[None, :, *extra_axes]))
        
        # Apply the mask to the pairwise interactions
        masked_pairs = jnp.where(pair_mask,all_pairs,0.)
        
        # Sum over all interactions for each particle
        # This reduces the (N, N, Nfitfunctions, d) array to (N, Nfitfunctions, d)
        pair_sum = jnp.sum(masked_pairs, axis=1)
        
        if fsingle is not None:
            # Compute single-particle interactions
            # This creates an array of shape (N, ...)
            single = vmap(fsingle)(state)
            # Mask out absent particles
            if mask is not None:
                single = jnp.where(mask[:, *extra_axes],single,0.)
 
            if concatenation_axis is None:
                # Simply add the pair and single particle
                # contributions (e.g. for nonlinear fitting).
                return single + pair_sum
            else:
                # Concatenate single and pair interactions along the designated axis.
                return jnp.concatenate((single, pair_sum), axis=concatenation_axis)
        else:
            return pair_sum

    return interaction_function


def pair_interaction_basis(bpair, bsingle=None):
    """
    Creates basis functions for particle interactions, including both pairwise and single-particle interactions.
    
    This function is designed for systems with no cross-diffusion, i.e., no D_ij_munu terms with i != j.
    It allows for a massive simplification of the gradient evaluation.

    Args:
    bpair: Function that computes the interaction between two particles.
           It should take two d-dimensional vectors as arguments: xi (position of particle i) and xj (position of particle j).
           Returns an array of shape (Nfpairfunctions, d): pair interation on particle i.
    bsingle: Optional. Function that computes single-particle interactions.
             It should take one argument: x (position of a particle).
             Returns an array of shape (Nsinglefunctions, d).

    Returns:
    C: Function that computes the total interaction for each particle.
    grad_C: Function that computes the gradient of C with respect to particle positions.
    """

    # Pre-compile the Jacobian functions for efficiency
    grad_bpair = jit(jacfwd(bpair))
    grad_bsingle = jit(jacfwd(bsingle)) if bsingle is not None else None

    C = construct_interaction_function(bpair,fsingle=bsingle,concatenation_axis = 1)
    grad_C = construct_interaction_function(grad_bpair,fsingle=grad_bsingle,concatenation_axis = 1)
    
    return C,grad_C


def pair_interaction_basis_nonlinear(bpair, nparams_pair, bsingle=None, nparams_single=0):
    """
    Creates basis functions for particle interactions, including both pairwise and single-particle interactions,
    with parametric inference capabilities.
    
    Args:
    bpair: Function that computes the interaction between two particles.
           It should take three arguments: xi, xj (positions of particles i and j), and theta_pair (parameters).
    bsingle: Optional. Function that computes single-particle interactions.
             It should take two arguments: x (position of a particle) and theta_single (parameters).
    theta0_pair: Initial or example parameters for bpair, used to determine the shape.
    theta0_single: Initial or example parameters for bsingle, used to determine the shape.

    Returns:
    C: Function that computes the total interaction for each particle.
    grad_C_x: Function that computes the gradient of C with respect to particle positions.
    grad_C_theta: Function that computes the gradient of C with respect to parameters.
    grad_C_x_theta: Function that computes the mixed second derivative of C.
    """

    # Determine the shapes of theta_pair and theta_single
    theta_pair_shape = nparams_pair
    theta_single_shape = nparams_single

    # Pre-compile the Jacobian and gradient functions for efficiency
    grad_bpair_x = jit(jacfwd(bpair, argnums=0))
    grad_bpair_theta = jit(jacfwd(bpair, argnums=2))
    grad_bpair_x_theta = jit(jacfwd(jacrev(bpair, argnums=2), argnums=0))

    grad_bsingle_x = jit(jacfwd(bsingle, argnums=0)) if bsingle is not None else None
    grad_bsingle_theta = jit(jacfwd(bsingle, argnums=1)) if bsingle is not None else None
    grad_bsingle_x_theta = jit(jacfwd(jacrev(bsingle, argnums=1), argnums=0)) if bsingle is not None else None

    def split_theta(theta):
        """Split the global theta into theta_pair and theta_single."""
        theta_pair = theta[nparams_single:nparams_single+nparams_pair]
        theta_single = theta[:nparams_single]
        return theta_pair, theta_single

    def construct_nonlinear_interaction(pair,single=None,concatenation_axis = None):
        @jit
        def nonlinear_interaction(X, mask, theta):
            theta_pair, theta_single = split_theta(theta)
            pair_partial = lambda x1,x2 : pair(x1,x2,theta_pair)
            single_partial = None if single is None else lambda x : single(x,theta_single)
            return construct_interaction_function(pair_partial,fsingle=single_partial,concatenation_axis=concatenation_axis)(X,mask)
        return nonlinear_interaction

    C = construct_nonlinear_interaction(bpair,bsingle)
    C_x = construct_nonlinear_interaction(grad_bpair_x,grad_bsingle_x)
    C_theta = construct_nonlinear_interaction(grad_bpair_theta,grad_bsingle_theta,concatenation_axis=2)
    C_x_theta = construct_nonlinear_interaction(grad_bpair_x_theta,grad_bsingle_x_theta,concatenation_axis=2) # Check axis
    return C,C_x,C_theta,C_x_theta
    




def SPP_pair_interaction(kernels_radial,kernels_angular):
    # Interactions between polar particles (x,y,theta), with radial
    # interactions F_i = F(r_ij).(xj-xi) and sinusoidal torque
    # F_i,theta = sin(theta_j-theta_i) g(r_ij). If either of these is
    # an empty list it will be discarded.
    def pair(Xi,Xj):
        Xij = Xj[:2] - Xi[:2]
        Rij = jnp.linalg.norm(Xij)
        sTij = jnp.sin(Xj[2] - Xi[2])
        
        radial_terms = jnp.array([f(Rij) for f in kernels_radial])
        angular_terms = jnp.array([f(Rij) for f in kernels_angular])
        
        interactions = jnp.zeros((len(kernels_radial) + len(kernels_angular), 3))
        interactions = interactions.at[:len(kernels_radial), :2].set((radial_terms / Rij)[:, None] * Xij)
        interactions = interactions.at[len(kernels_radial):, 2].set(angular_terms * sTij / Rij)
        
        return jnp.nan_to_num(interactions)
    return jit(pair)

def SPP_interaction_basis(kernels_radial, kernels_angular, polynomial_order):
    """
    Create interaction basis for Self-Propelled Particles (SPP) system.

    Args:
    kernels_radial (list): List of radial interaction kernel functions.
    kernels_angular (list): List of angular interaction kernel functions.
    polynomial_order (int): Order of the polynomial basis for single-particle interactions.

    Returns:
    function: Combined interaction basis function.
    """
    # Create polynomial basis function for 2D space
    poly, _ = polynomial_basis(2, polynomial_order)
    
    @jit
    def bsingle(X):
        """
        Compute single-particle interaction basis.

        Args:
        X (jnp.array): Particle state vector, shape (3,) [x, y, theta]

        Returns:
        jnp.array: Single-particle interaction basis, shape (N_basis, 3)
                   where N_basis = 2 * n_poly_terms + 2
                   (n_poly_terms for each spatial dimension + 2 for self-propulsion)
        """
        cos_theta, sin_theta = jnp.cos(X[2]), jnp.sin(X[2])
        
        # Self-propulsion terms, shape (2, 3)
        self_propulsion = jnp.array([[cos_theta, sin_theta, 0],
                                     [sin_theta, -cos_theta, 0]])
        
        # Polynomial basis terms, shape (n_poly_terms,)
        poly_scalar = poly(X[:2])

        # Create polynomial terms for x and y dimensions, shape (2 * n_poly_terms, 3)
        poly_combined = jnp.zeros((2 * len(poly_scalar), 3))
        poly_combined = poly_combined.at[:len(poly_scalar), 0].set(poly_scalar)
        poly_combined = poly_combined.at[len(poly_scalar):, 1].set(poly_scalar)
        
        # Stack polynomial and self-propulsion terms, shape (N_basis, 3)
        return jnp.vstack([poly_combined, self_propulsion])
    
    # Create pair interaction function
    bpair = SPP_pair_interaction(kernels_radial, kernels_angular)
    
    # Combine single-particle and pair interactions
    return pair_interaction_basis(bpair, bsingle)
