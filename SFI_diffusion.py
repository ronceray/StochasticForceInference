
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import LinAlgError

from SFI_projectors import TrajectoryProjectors

class DiffusionInference(object):
    """Implementation of the diffusion tensor field inference method
    presented in "Anna Frishman and Pierre Ronceray, Learning force
    fields from stochastic trajectories, arXiv:1809.09650, 2018.".

    It performs a linear regression of a local diffusion estimator
    with a normalized set of functions.

    Parameters:
    - data and basis arguments are as in SFI_forces.
    - diffusion_method selects the local diffusion estimator: 'MSD' or
      'Vestergaard'.
    """ 
    def __init__(self,data,basis,diffusion_method = 'MSD',verbose=True):
        self.data = data
        self.basis = basis

        # Select a projection basis
        import SFI_bases
        if hasattr(self.basis, "functions"):
            funcs = self.basis.functions
        else:
            funcs = SFI_bases.basis_selector(self.basis,self.data)

        # Prepare the functions:
        self.projectors = TrajectoryProjectors(self.data.inner_product_empirical,funcs)

        # Select the (noisy) local diffusion matrix estimator:
        if diffusion_method == 'MSD':
            D_local = [ np.einsum('im,in->imn',self.data.dX[t],self.data.dX[t])/(2*dt) for t,dt in enumerate(self.data.dt)   ]
            integration_style = 'Stratonovich' 
        elif diffusion_method == 'Vestergaard':
            # Local estimator inspired by "Vestergaard CL, Blainey PC,
            # Flyvbjerg H (2014). Optimal estimation of diffusion
            # coefficients from single-particle trajectories. Physical
            # Review E 89(2):022726.".
            #
            # It is unbiased with respect to measurement noise, at the
            # cost of a 4x slower convergence. Use this estimator if
            # measurement noise is the limiting factor on inferring
            # D. Note that the error is minimized when symmetrizing the
            # correction term and integrating in Ito, i.e. evaluating
            # the projector at the initial point of the interval.
            dXp = self.data.dX[1:]
            dXm = self.data.dX[:-1]
            dX0 = dXp + dXm
            D_local = [ (   np.einsum('im,in->imn',dX0[t],dX0[t])
                        +   np.einsum('im,in->imn',dXp[t],dXm[t])
                        +   np.einsum('im,in->imn',dXm[t],dXp[t]))
                        /(4*dt) for t,dt in enumerate(self.data.dt[1:])   ]
            # There is one point fewer than for usual integration. We
            # integrate in a symmetric way with respect to the
            # estimator, i.e. over X[1:-1].
            integration_style = 'StratonovichOnPoint'
        else:
            raise KeyError("Wrong diffusion_method argument.")
        # Reshape into vectors as inner product allows for only one
        # non-particle index:
        D_local_reshaped = [ np.array([ flatten_symmetric(Di,self.data.d) for Di in D]) for D in D_local ]
        D_projections_reshaped = self.data.inner_product_empirical(D_local_reshaped, self.projectors.c, integration_style = integration_style) 
        # Back to matrix form:
        self.D_projections = np.array([ inflate_symmetric(Di,self.data.d) for Di in D_projections_reshaped.T]).T 
        self.D_ansatz,self.D_coefficients = self.projectors.projector_combination(self.D_projections) 

        self.D_average = np.einsum('t,tmn->mn',self.data.dt[:len(D_local)],np.array([ np.einsum('imn->mn',D) for D in D_local]))/self.data.tauN

        # Defining a derivative-based ansatz for div D:
        def divD(x):         
            return np.einsum('mna,jmia->in', self.D_coefficients, self.projectors.grad_b(x) )
        self.divD_ansatz = divD
        
        self.bootstrapped_Dproj_error = np.prod(self.D_projections.shape) / sum(self.data.Nparticles)
        self.safety_checks()
        if verbose:
            self.print_report()

    def compute_accuracy(self,D_exact,divD_exact,data_exact=None,verbose=True):
        """Evaluate the precision of the method when the actual diffusion
        matrix is known.

        """
        if data_exact is None:
            # In the case of noisy input data, we want to compare to
            # the diffusion inferred on the real trajectory, not the
            # noisy one (which would use values of the diffusion field
            # that weren't explored by the trajectory, and thus cannot
            # be predicted).
            data_exact = self.data
            
        self.exact_D  = [ D_exact(X) for X in data_exact.X_ito ]
        self.ansatz_D = [ self.D_ansatz(X) for X in data_exact.X_ito ]
        self.exact_divD  = [ divD_exact(X) for X in data_exact.X_ito ]
        self.ansatz_divD = [ self.divD_ansatz(X) for X in data_exact.X_ito ]

        # Evaluate the precision as < ||(De-Di)/(De+Di)||^2 >
        self.D_precision = np.array([ np.linalg.norm( np.einsum('imn,ino->imo', np.linalg.inv(D + self.ansatz_D[t]), D - self.ansatz_D[t]))**2 for t,D in enumerate(self.exact_D) ]).mean()
        self.divD_precision = np.array([ np.linalg.norm(d-self.ansatz_divD[t])**2/(np.linalg.norm(d)**2 + np.linalg.norm(self.ansatz_divD[t])**2)  for t,d in enumerate(self.exact_divD) ]).mean()
        
        D_local_reshaped = [ np.array([ flatten_symmetric(De,self.data.d) for De in D_exact(X)]) for X in data_exact.X_ito ]
        D_projections_reshaped = self.data.inner_product_empirical( D_local_reshaped, self.projectors.c, integration_style = 'Ito' ) 
        self.exact_D_projections = np.array([ inflate_symmetric(Di,self.data.d) for Di in D_projections_reshaped.T]).T 
        self.D_projections_error = np.linalg.norm(self.exact_D_projections - self.D_projections)**2 / np.linalg.norm(self.D_projections)**2 
        
        if verbose:
            print("Error on inferred D along trajectory:",self.D_precision)
            print("Error on inferred D projection; bootstrapped error:",self.D_projections_error,self.bootstrapped_Dproj_error)
            print("Error on div D:",self.divD_precision)


    def print_report(self):
        """Tell us a bit about yourself."""
        print("Average diffusion tensor:\n",self.D_average)
        print("Bootstrapped error on projections:",self.bootstrapped_Dproj_error)
        print("Eigenvalues variations around average D: min/max/std:",self.emin,self.emax,self.estd)
 
    def safety_checks(self):
        """Basic tests to check the variations of the diffusion around the
        mean, along the trajectory: examine the variations of the
        eigenvalues of Dav^-1/2 . D(X) . Dav^-1/2. This allows the
        identification of negative eigenvalues (the method isn't
        protected against it) and is useful to establish whether the
        results are reliable or should be further reliable before, eg,
        inverting D(X).
        """
        D_average_sqrtinv = sqrtm(np.linalg.inv(self.D_average))
        ansatz_D_Ito = [ self.D_ansatz(X) for X in self.data.X_ito ]
        ansatz_D_dimensionless = [ np.einsum('mn,ino,op->imp',D_average_sqrtinv,D,D_average_sqrtinv) for D in ansatz_D_Ito ]
        eigenvalues_D = np.array([ np.linalg.eigh(Di)[0] for D in ansatz_D_dimensionless for Di in D ])
        self.emin,self.emax,self.estd = eigenvalues_D.min(),eigenvalues_D.max(),eigenvalues_D.std()
        self.Dmin,self.Dmax = self.D_average*self.emin, self.D_average*self.emax
        if self.emin < 0:
            print("Warning, the inferred diffusion matrix has negative eigenvalues along the trajectory!")
            
    def regularized_D_ansatz(self, cutoff_low, cutoff_high = np.inf ):
        """Returns a D(X) function such that all eigenvalues (of the matrix
        normalized by the average diffusion matrix) are truncated to
        fall within the [low,high] cutoff interval. """
        D_average_sqrt = sqrtm(self.D_average)
        D_average_sqrtinv = sqrtm(np.linalg.inv(self.D_average))
        def D_reg(X):
            D_in = self.D_ansatz(X)
            D_in_norm = np.einsum('mn,ino,op->imp',D_average_sqrtinv, D_in ,D_average_sqrtinv) 
            D_out_norm = []
            for Di in D_in_norm:
                evals,evecs = np.linalg.eigh(Di)
                evals_truncated = np.array([ max( min( cutoff_high, v ), cutoff_low ) for v in evals ])
                D_out_norm.append(np.einsum('mn,n,on->mo',evecs,evals_truncated,evecs))
            D_out = np.einsum('mn,ino,op->imp', D_average_sqrt, np.array(D_out_norm), D_average_sqrt )
            return D_out
        return D_reg

    
            


        
def flatten_symmetric(M,dim):
    # A helper function to go from dxd array to d(d+1)/2 vector with
    # the upper triangular values.
    return np.array([ M[i,j] for i in range(dim) for j in range(i+1)])

def inflate_symmetric(V,dim):
    # The revert operation
    M = np.zeros((dim,dim))
    k = 0
    for i in range(dim):
        for j in range(i+1):
            M[i,j] = V[k]
            M[j,i] = V[k]
            k += 1
    return M
