\
""" A library of fitting bases for Underdamped Langevin
Inference. Closely parallels / reuses the overdamped file, with the
difference that velocities are included as arguments of basis
functions."""



import jax.numpy as jnp
from jax import random,jit,jacfwd,jacrev

def basis_selector(basis, dimension, output="vector", symmetric = True, isotropic = False):
    basis_type = basis["type"]

    if basis_type == "polynomial":
        poly,descr = underdamped_polynomial_basis(dimension, basis["order"],basis['mode'])
        funcs_and_grad, descriptor = scalar_basis(
            poly,
            dimension, output,
            symmetric=symmetric, isotropic=isotropic, descriptors = descr
        )
    elif basis_type == "custom_scalar":
        funcs_and_grad, descriptor = scalar_basis(
            basis["functions"], dimension, output
        )

    elif basis_type in ("custom_vector", "custom_tensor"):
        funcs_and_grad, descriptor = prepare_ULI_basis_function(basis["functions"])
    
    else:
        raise KeyError(f"Unknown basis type: {basis['type']}")
    
    return funcs_and_grad, descriptor

def scalar_basis(C_func, dimension, output_type, * , symmetric=False, isotropic=False, descriptors=None):
    import SFI.OLI_bases
    # We reuse the OLI_bases implementation, just converting X,V into a state tuple (X,V) then back
    wrapped_function = SFI.OLI_bases.wrap_scalar_basis(lambda state : C_func(state[0],state[1]), dimension, output_type, symmetric=False, isotropic=False)
    if descriptors is None:
        # Infer the number of functions through a dummy evaluation on
        # random numbers (less error-prone than ones or zeros):
        Nfunctions = C_func(random.normal(random.key(0),shape=(dimension,)),random.normal(random.key(1),shape=(dimension,))).shape[0]
        descriptors, _ = SFI.SFI_utils.make_variable_names(Nfunctions,symbol="b")
    wrapped_descriptors = SFI.OLI_bases.wrap_descriptors(descriptors, dimension, output_type, symmetric=symmetric,isotropic=isotropic)
    return prepare_ULI_basis_function(lambda X,V : wrapped_function((X,V))),wrapped_descriptors
    
def prepare_ULI_basis_function(structured_basis):
    # We reuse the OLI_bases implementation, just converting X,V into
    # a state tuple (X,V) then back; a difference here is that we also
    # (and mostly) want the v derivative of the basis.

    # 1. Pack the arguments & differentiate
    basis = lambda state : structured_basis(state[0],state[1])
    basis_x = lambda state : jacfwd(structured_basis,argnums=0)(state[0],state[1])
    basis_v = lambda state : jacfwd(structured_basis,argnums=1)(state[0],state[1])

    # 2. Vectorize and mask
    import SFI.OLI_bases
    C = SFI.OLI_bases.vectorize_and_mask(basis)
    C_x = SFI.OLI_bases.vectorize_and_mask(basis_x)
    C_v = SFI.OLI_bases.vectorize_and_mask(basis_v)

    # 3. Unpack arguments & jit
    C_jit = jit(lambda X,V,mask=None : C((X,V),mask) )
    C_x_jit = jit(lambda X,V,mask=None : C_x((X,V),mask) )
    C_v_jit = jit(lambda X,V,mask=None : C_v((X,V),mask) )
    
    return C_jit,C_x_jit,C_v_jit


def underdamped_polynomial_basis(dim, order, mode="both", variable_names=None):
    """
    Generates a polynomial basis function up to a given order for underdamped (x,v) data.
    Depending on `mode`, the polynomial is built on:
      - 'both': x and v, i.e. total dimension = 2*dim
      - 'x':    only x, dimension = dim
      - 'v':    only v, dimension = dim

    Args:
        dim (int): Number of dimensions for x or v individually.
        order (int): Maximum polynomial order.
        mode (str): 'both', 'x', or 'v' (default 'both').
        variable_names (optional): for pretty printing.

    Returns:
        function: A JIT-compiled function that computes polynomial basis values.
                  signature: basis((x, v)) -> jnp.ndarray of shape (n_basis_terms,)

    """
    import SFI.OLI_bases
    # Decide how many total dimensions we will build polynomials over
    total_dim = 2*dim if mode == "both" else dim

    # Deal with variable names:
    variable_names, _ = SFI.SFI_utils.make_variable_names(dim,symbol="x")
    variable_names_v, _ = SFI.SFI_utils.make_variable_names(dim,symbol="v")
    if mode == "v": variable_names = variable_names_v
    elif mode == "both":
        variable_names = variable_names + variable_names_v

    # Generate the monomials and their description:
    P, desc = SFI.OLI_bases.polynomial_basis(
        total_dim, order,
        variable_names=variable_names)
    if mode == "both":
        basis = lambda x,v : P(jnp.concatenate([x, v], axis=0))
    elif mode == "x":
        basis = lambda x,v : P(x)
    elif mode == "v":
        basis = lambda x,v : P(v)
    else:
        raise ValueError(f"Unknown mode='{mode}'. Expect 'both', 'x', or 'v'.")

    return jit(basis),desc


def ULI_pair_interaction_basis(pair, single=None):
    """
    Creates basis functions for particle interactions, including both pairwise and single-particle interactions.
    
    This function is designed for systems with no cross-diffusion, i.e., no D_ij_munu terms with i != j.
    It allows for a massive simplification of the gradient evaluation.

    Args:
    bpair: Function that computes the interaction between two particles.
           It should take 4 d-dimensional vectors as arguments: xi,xj,vi,vj (position and velocity of particle i and j)
           Returns an array of shape (Nfpairfunctions, d): pair interation on particle i.
    bsingle: Optional. Function that computes single-particle interactions.
             It should take one argument: x (position of a particle).
             Returns an array of shape (Nsinglefunctions, d).

    Returns:
    C: Function that computes the total interaction for each particle.
    grad_C: Function that computes the gradient of C with respect to particle positions.
    """

    # 1. Pack the arguments & differentiate
    bsingle = (lambda state : single(state[0],state[1])) if single is not None else None
    bsingle_x = (lambda state : jacfwd(single,argnums=0)(state[0],state[1])) if single is not None else None
    bsingle_v = (lambda state : jacfwd(single,argnums=1)(state[0],state[1])) if single is not None else None
    bpair = lambda state1,state2 : pair(state1[0],state2[0],state1[1],state2[1])
    bpair_x = lambda state1,state2 : jacfwd(pair,argnums=0)(state1[0],state2[0],state1[1],state2[1])
    bpair_v = lambda state1,state2: jacfwd(pair,argnums=2)(state1[0],state2[0],state1[1],state2[1])
    # 2. Vectorize and mask
    import SFI.OLI_bases
    C = SFI.OLI_bases.construct_interaction_function(bpair,fsingle=bsingle,concatenation_axis = 1)
    C_x = SFI.OLI_bases.construct_interaction_function(bpair_x,fsingle=bsingle_x,concatenation_axis = 1)
    C_v = SFI.OLI_bases.construct_interaction_function(bpair_v,fsingle=bsingle_v,concatenation_axis = 1)
    
    # 3. Unpack arguments & jit
    C_jit = jit(lambda X,V,mask=None : C((X,V),mask) )
    C_x_jit = jit(lambda X,V,mask=None : C_x((X,V),mask) )
    C_v_jit = jit(lambda X,V,mask=None : C_v((X,V),mask) )
    return C_jit,C_x_jit,C_v_jit


def ULI_pair_interaction_basis_nonlinear(pair, nparams_pair, single=None, nparams_single=0):
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

    # 1. Pack the arguments & differentiate
    bsingle = (lambda state,theta : single(state[0],state[1],theta)) if single is not None else None
    bsingle_v = (lambda state,theta : jacfwd(single,argnums=1)(state[0],state[1],theta)) if single is not None else None
    bsingle_t = (lambda state,theta : jacfwd(single,argnums=2)(state[0],state[1],theta)) if single is not None else None
    bsingle_vt = (lambda state,theta : jacrev(jacfwd(single,argnums=1),argnums=2)(state[0],state[1],theta)) if single is not None else None
    bpair = lambda state1,state2,theta : pair(state1[0],state2[0],state1[1],state2[1],theta)
    bpair_v = lambda state1,state2,theta: jacfwd(pair,argnums=2)(state1[0],state2[0],state1[1],state2[1],theta)
    bpair_t = lambda state1,state2,theta: jacfwd(pair,argnums=4)(state1[0],state2[0],state1[1],state2[1],theta)
    bpair_vt = lambda state1,state2,theta: jacrev(jacfwd(pair,argnums=2),argnums=4)(state1[0],state2[0],state1[1],state2[1],theta)

    def split_theta(theta):
        """Split the global theta into theta_pair and theta_single."""
        theta_pair = theta[nparams_single:nparams_single+nparams_pair]
        theta_single = theta[:nparams_single]
        return theta_pair, theta_single

    from SFI.OLI_bases import construct_interaction_function
    def construct_nonlinear_interaction(pair,single=None,concatenation_axis = None):
        def nonlinear_interaction(state, mask, theta):
            theta_pair, theta_single = split_theta(theta)
            pair_partial = lambda state1,state2 : pair(state1,state2,theta_pair)
            single_partial = None if single is None else lambda state : single(state,theta_single)
            return construct_interaction_function(pair_partial,fsingle=single_partial,concatenation_axis=concatenation_axis)(state,mask)
        return nonlinear_interaction

    # 2. Vectorize and mask
    C = construct_nonlinear_interaction(bpair,bsingle)
    C_v = construct_nonlinear_interaction(bpair_v,bsingle_v)
    C_t = construct_nonlinear_interaction(bpair_t,bsingle_t)
    C_vt = construct_nonlinear_interaction(bpair_vt,bsingle_vt)

    # 3. Unpack arguments & jit:
    C_jit = jit(lambda X,V,mask,theta : C((X,V),mask,theta) )
    Cv_jit = jit(lambda X,V,mask,theta : C_v((X,V),mask,theta) )
    Ct_jit = jit(lambda X,V,mask,theta : C_t((X,V),mask,theta) )
    Cvt_jit = jit(lambda X,V,mask,theta : C_vt((X,V),mask,theta) )
    
    return C_jit,Cv_jit,Ct_jit,Cvt_jit
