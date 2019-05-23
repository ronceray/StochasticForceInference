
""" A library of projection bases for Stochastic Force Inference. """



import numpy as np



def basis_selector(basis,data):
    if basis['type'] == 'polynomial':
        funcs = polynomial_basis(data.d,basis['order'])
    elif basis['type'] == 'Fourier':
        funcs = Fourier_basis(data.d,basis['order'],basis['center'],basis['width'])
    elif basis['type'] == 'Chebyshev':
        funcs = Chebyshev_basis(data.d,basis['order'],basis['center'],basis['width'])
    elif basis['type'] == 'FourierPlusHarmonic':
        funcs = polynomial_basis(data.d,1,False) + Fourier_basis(data.d,basis['order'],basis['center'],basis['width'])
    elif basis['type'] == 'Hermite':
        funcs = Hermite_polynomial_basis(data.d,basis['order'],basis['center'],basis['std'])
    elif basis['type'] == 'particles_pair_interaction': 
        funcs = particles_pair_interaction(data.d,basis['kernels'])
    elif basis['type'] == 'particles_pair_interaction_scalar': 
        funcs = particles_pair_interaction_scalar(basis['kernels'])
    elif basis['type'] == 'particles_and_polynomial': 
        funcs = particles_polynomial_composite_basis(data.d,basis['order'],basis['kernels'])
    elif basis['type'] == 'particles_and_polynomial_scalar': 
        funcs = particles_polynomial_scalar_composite_basis(data.d,basis['order'],basis['kernels'])
    elif basis['type'] == 'coarse_graining':  
        funcs = coarse_graining_basis(data.d,basis['width'],basis['center'],basis['order'])
    elif basis['type'] == 'FORMA':  
        funcs = FORMA_basis(data.d,basis['width'],basis['center'],basis['order'])
    elif basis['type'] == 'self_propelled_particles': 
        funcs = self_propelled_particles_basis(basis['order'],basis['kernels'])
    elif basis['type'] == 'single_self_propelled_particle': 
        funcs = single_self_propelled_particle_basis()
    elif basis['type'] == 'custom': 
        funcs = basis['functions']
    else:
        raise KeyError("Unknown basis type.") 
    return funcs



def polynomial_basis(dim,order):
    # A simple polynomial basis, X -> X_mu X_nu ... up to polynomial
    # degree 'order'.
    
    # We first generate the coefficients, ie the indices mu,nu.. of
    # the polynomials, in a non-redundant way. We start with the
    # constant polynomial (empty list of coefficients) and iteratively
    # add new indices.
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int)) 
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ] 
    return lambda X : np.array([[ np.prod(x[c]) for c in coeffs ] for x in X])


def Fourier_basis(dim,order,center,width):
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int))
        
    coeffs = [ c for degree in coeffs[1:] for c in degree ]
    if dim >= order:
        # Two paradigms for the computation of the function: sparse if
        # dim > order, dense otherwise.
        def Fourier(X):
            Xc = 2 * np.pi* (X - center) / width
            return np.array([ [ 1. ] + [ np.cos(sum(x[c])) for c in coeffs ] + [ np.sin(sum(x[c])) for c in coeffs ] for x in Xc])
    else:
        coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(dim) ] for c in coeffs ])
        def Fourier(X):
            Xc = 2 * np.pi* (X - center) / width
            return np.array([ [ 1. ] + [ np.cos( x.dot(c)) for c in coeffs_lowdim ] + [ np.sin( x.dot(c)) for c in coeffs_lowdim ] for x in Xc])
    return Fourier 


def Chebyshev_basis(dim,order,center,width):
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int))
        
    coeffs = [ c for degree in coeffs[1:] for c in degree ]
    coeffs_lowdim = np.array([ [ list(c).count(i) for i in range(dim) ] for c in coeffs ])
    def Chebyshev(X):
        Xc = (X - center) / width
        return np.array([[ np.prod([ np.cos(n[d]*np.arccos(x[d])) for d in range(dim)]) for n in coeffs_lowdim ] for x in Xc])
    return Chebyshev

def Hermite_polynomial_basis(dim,order,center,std,has_const = True):
    # A simple polynomial basis times a gaussian kernel,
    #        X -> exp( -|X - center|^2 / 2 s^2) X_mu X_nu ...
    # up to polynomial degree 'order'.  If "has const", also include X -> 1
    # in the basis.
    coeffs = [ np.array([[]],dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]:
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim ):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs,dtype=int)) 

    coeffs = [ c for degree in coeffs for c in degree ]
    sigma = 2*std**2
    def Hermite(X):
        Y = np.array([ x - center for x in X ])
        w = np.array([ np.exp(- np.linalg.norm(y)**2 / sigma )  for y in Y ])
        if has_const:
            return np.array([ [ 1.] + [ w[i]*np.prod(y[c]) for c in coeffs ] for i,y in enumerate(Y)])
        else:
            return np.array([         [ w[i]*np.prod(y[c]) for c in coeffs ] for i,y in enumerate(Y)])
    return Hermite 
    


### COARSE GRAINING ####
def coarse_graining_basis(dim,width,center,order):
    print("""Warning, using a non-differentiable basis (coarse-graining).  Do
    NOT consider the StochasticForceInference methods "F_projections"
    and its dependencies; use instead the phi_ansatz (Ito
    estimate). The entropy production and capacity will need to be
    manually re-computed. Note that SFI with multiplicative noise
    and/or measurement noise will NOT work with this choice of
    basis.""")
    Nfuncs = order**dim
    def index_finder(x):
        projection_indices = [ int(np.floor( (x[mu] - (center[mu] - 0.5*width[mu]))/(width[mu]) * order)) for mu in range(dim) ]
        return sum([ ( imu * order**mu if  0 <= imu < order else np.inf )  for mu,imu in enumerate(projection_indices) ])
    
    def grid_function(X):
        val = np.zeros((X.shape[0],Nfuncs))
        for i in range(X.shape[0]):
            n = index_finder(X[i])
            if 0 <= n < Nfuncs:
                val[i,n] = 1. 
        return val
    return grid_function 

 
### FORMA  ####
""" Linear-by-parts grid coarse-graining; method used in
High-performance reconstruction of microscopic force fields from Brownian trajectories
Laura Perez Garcia, Jaime Donlucas Perez, Giorgio Volpe, Alejandro V. Arzola and Giovanni Volpe 
Nature Communicationsvolume 9, Article number: 5166 (2018) 
"""
def FORMA_basis(dim,width,center,order):
    print("""Warning, using a non-differentiable basis (linear-by-parts
    coarse-graining).  Do NOT consider the StochasticForceInference
    methods "F_projections" and its dependencies; use instead the
    phi_ansatz (Ito estimate). The entropy production and capacity
    will need to be manually re-computed. Note that SFI with
    multiplicative noise and/or measurement noise will NOT work with
    this choice of basis.""")
    Ncells = order**dim
    Nfuncs = (dim+1) * Ncells

    def index_finder(x):
        projection_indices = [ int(np.floor( (x[mu] - (center[mu] - 0.5*width[mu]))/(width[mu]) * order)) for mu in range(dim) ]
        return sum([ ( imu * order**mu if  0 <= imu < order else np.inf )  for mu,imu in enumerate(projection_indices) ])
    
    def grid_function(X):
        val = np.zeros((X.shape[0],Nfuncs))
        for i in range(X.shape[0]):
            n = index_finder(X[i])
            if 0 <= n < Nfuncs:
                val[i,n] = 1.
                for mu in range(dim):
                    val[i,n+(mu+1)*Ncells] = X[i,mu]
        return val
    return grid_function 


def particles_pair_interaction(dim,kernels):
    # Radially symmetric vector-like pair interactions as a sum of
    # kernels.  Two-particle functions are chosen to be of the form
    # f(R_ij) * (Xj - Xi)/Rij for a given set of functions f
    # (kernels).
    def pair_function_spherical(X):
        # X is a Nparticles x dim - shaped array.
        Nparticles = X.shape[0]
        Xij = np.array([[ Xj - Xi for j,Xj in enumerate(X) ] for i,Xi in enumerate(X) ])
        Rij = np.linalg.norm(Xij,axis=2)
        f_Rij = np.nan_to_num(np.array([ f(Rij)/Rij for f in kernels ]))
        # Combine the kernel index f and the spatial index m into a
        # single function index a:
        return np.einsum('fij,ijm->ifm',f_Rij,Xij).reshape((Nparticles,dim * len(kernels)))
    return pair_function_spherical


def particles_pair_interaction_scalar(kernels):
    # Radially symmetric scalar-like pair interactions as a sum of
    # kernels.  Two-particle functions are chosen to be of the form
    # f(R_ij) for a given set of functions f (kernels).
    def pair_function_spherical(X):
        # X is a Nparticles x dim - shaped array.
        Nparticles = X.shape[0]
        Xij = np.array([[ Xj - Xi for j,Xj in enumerate(X) ] for i,Xi in enumerate(X) ])
        Rij = np.linalg.norm(Xij,axis=2)
        f_Rij = np.nan_to_num(np.array([ f(Rij) for f in kernels ]))
        # Combine the kernel index f and the spatial index m into a
        # single function index a:
        return np.einsum('fij->if',f_Rij)
    return pair_function_spherical




def particles_polynomial_composite_basis(dim,order_single,kernels):
    # A composite basis: single-particle forces as polynomials
    # (external field), and radially symmetric pair interactions as a
    # sum of kernels.
    poly = polynomial_basis(dim,order_single)
    pair = particles_pair_interaction(dim,kernels)
    return lambda X :  np.array([ v for v in poly(X).T ]+[ v for v in pair(X).T ]).T


def particles_polynomial_scalar_composite_basis(dim,order_single,kernels):
    # A composite basis: single-particle forces as polynomials
    # (external field), and radially symmetric pair interactions as a
    # sum of kernels.
    poly = polynomial_basis(dim,order_single)
    pair = particles_pair_interaction_scalar(kernels)
    return lambda X :  np.array([ v for v in poly(X).T ]+[ v for v in pair(X).T ]).T


def self_propelled_particles_basis(order_single,kernels):
    # A basis adapted to 2D self-propelled particles without alignment
    self_propulsion =  lambda X : np.array([ np.cos(X[:,2]), np.sin(X[:,2]) ]).T 
    poly = polynomial_basis(2,order_single)
    pair = particles_pair_interaction(2,kernels)
    return lambda X :  np.array([ v for v in poly(X[:,:2]).T ]+[ v for v in pair(X[:,:2]).T ]+[ v for v in self_propulsion(X).T ]).T
    



def single_self_propelled_particle_basis():
    # Palacci lab tests
    self_propulsion_functions = [ lambda X : np.array([ 1 + 0.*X[:,2] , np.cos(X[:,2]), np.sin(X[:,2]) ]) ]
    return self_propulsion_functions

