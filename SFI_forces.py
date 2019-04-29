#
# StochasticForceInference is a package developed by Pierre Ronceray
# and Anna Frishman aimed at reconstructing force fields and computing
# entropy production in Brownian processes by projecting forces onto a
# finite dimensional functional space. Compatible Python 3.6 / 2.7.
#
# Reference: Anna Frishman and Pierre Ronceray, Learning force fields
#     from stochastic trajectories, arXiv:1809.09650, 2018.
#


import numpy as np


class StochasticForceInference(object):
    """A class to infer the force field, currents and entropy production
    from the geometry of a single trajectory of arbitrary overdamped
    systems. Note that everywhere here, "force" is taken to mean
    "mobility matrix times force".

    The input 'data' argument has type StochasticTrajectoryData (see
    SFI_data).

    The 'basis' dictionary sets the choice of basis for fitting of the
    force field, and its parameters. See SFI_bases for examples of
    standard bases. Alternatively, custom functions can be specified
    as { 'functions' : f } where f takes Nparticles x dim arrays as
    argument and returns a Nparticles x Nfunctions array.

    The 'diffusion_data' is a dictionary. It can be several things,
    depending on the value of the "type" parameter:
    - "type" = "constant": "D" is a dim x dim array (constant
      diffusion)
    - "type" = "function": "Dfunction" is a lambda function of X
      (multiplicative noise case) that is provided externally (eg to
      benchmark force inference when the exact diffusion tensor field
      is known). It should then take Nparticles x dim arrays as input,
      and return a Nparticles x input x input array.
    - "type" = "DiffusionInference": "DI" is a DiffusionInference
      object obtained using the SFI_diffusion module of this
      package. This is more efficient than externally providing the
      D_ansatz of this object. This object can be regularized to avoid
      producing close to zero (or even negative) diffusion
      coefficient.

    This class makes extensive use of "einsum" (see scipy manual), in
    order to leave the tensorial structure apparent, with explicit
    nature of the indices. We use the following conventions: 
    - t for temporal indices;
    - m,n... for spatial indices = 1..d (corresponds to mu,nu... in text);
    - a,b... for tensorial indices of the projection basis (alpha,beta... in text);
    - i,j... for particle indices (i,j.. in text).

    The "compute_phi" argument calculates the Ito-based projection of
    Xdot. It can be useful when using a non-differentiable basis for
    which the Stratonovich-based average does not work (eg for grid
    coarse-graining) but is otherwise not necessary to the method.

    """ 
    
    def __init__(self,data,diffusion_data,basis,verbose=True, compute_phi = False):
        self.data = data
        self.basis = basis
        self.diffusion_data = diffusion_data
        self.compute_phi = compute_phi

        # Parse the diffusion input: two possibilities -
        #  - either a matrix (constant - no dependency on space or
        #  particule).
        #  - or a lambda function returning a Nparticles x dim x dim
        #  array. In this case the noise is multiplicative and we
        #  adapt formulas accordingly.
        if self.diffusion_data["type"] == "constant":
            # Constant diffusion tensor.
            if self.diffusion_data["D"].shape != (self.data.d,self.data.d):
                raise ValueError("Constant D must be a dim x dim array.")
            self.D =  self.diffusion_data["D"]
            self.Dinv = np.linalg.inv(self.D)
            # The diffusion then has no divergence.
            
        elif self.diffusion_data["type"] == "function":
            self.D = self.diffusion_data["Dfunction"]
            self.Dinv = lambda X : np.linalg.inv(self.D(X))
        
        elif self.diffusion_data["type"] == "DiffusionInference":
            self.DI = self.diffusion_data["DI"]
            self.D = self.DI.D_ansatz
            if self.diffusion_data["cutoff"] is not None:
                self.Dreg = self.DI.regularized_D_ansatz(self.diffusion_data["cutoff"] )
                self.Dinv = lambda X : np.linalg.inv(self.Dreg(X))
            else:
                self.Dinv = lambda X : np.linalg.inv(self.D(X))
        
        # Select a projection basis - either by specifying it
        # explicitly with the 'functions' keyword of the 'basis'
        # argument, or by parsing it among the pre-defined bases
        # defined in SFI_bases.
        import SFI_bases
        if hasattr(self.basis, "functions"):
            funcs = self.basis.functions
        else:
            funcs = SFI_bases.basis_selector(self.basis,self.data)

        # Prepare the functions:
        from SFI_projectors import TrajectoryProjectors
        self.projectors = TrajectoryProjectors(self.data.inner_product_empirical,funcs)

        # Compute the projection coefficients. Indices m,n... denote
        # spatial indices; indices i,j.. denote particle indices;
        # indices a,b,... are used for the tensorial structure of the
        # projections.

        # The velocity projection coefficients (onto the projectors,
        # ie the self.projectors.c functions) are given by
        # Stratonovich integration of x_dot(t) c_n(x(t)).
        self.v_projections = np.einsum('ma,ab->mb',self.data.inner_product_empirical( self.data.Xdot, self.projectors.b, integration_style = 'Stratonovich' ), self.projectors.H )
        
        # The ansatz reconstructs the velocity field as
        #     v_ansatz_mu(x) = sum_a v_projections_mu_a c_a(x)
        # For convenience we also store the v_coefficients of v onto
        # the initial functions, b.
        self.v_ansatz,self.v_coefficients = self.projectors.projector_combination(self.v_projections)

        if self.compute_phi:
            # The Ito average of xdot. Note that it biased by measurement noise.
            # phi_mu(x) = < xdot_mu | x >_ito = F_mu(x) + divD_mu(x)
            self.phi_projections = np.einsum('ma,ab->mb',self.data.inner_product_empirical( self.data.Xdot, self.projectors.b, integration_style = 'Ito' ), self.projectors.H )
            self.phi_ansatz,self.phi_coefficients = self.projectors.projector_combination(self.phi_projections) 

        if self.diffusion_data["type"] == "constant":
            # The projection coefficients of g = grad log P:
            self.g_projections = - np.einsum('ma,ab->mb',self.data.trajectory_integral(lambda t : np.einsum('imia->ma',self.projectors.grad_b(self.data.X_strat[t]))), self.projectors.H )
            self.w_projections = np.einsum('mn,na->ma',self.D,self.g_projections) 

        elif self.diffusion_data["type"] == "function":
            # Compute the projection of w = D grad log P using an
            # integration by parts. This could be optimized.
            epsilon = 1e-6
            def D_c(x):
                return np.einsum('imn,ia->imna', self.D(x), self.projectors.c(x))
            def grad_D_c(x):
                Nparticles,dim = x.shape
                dx = [[ np.array([[ 0 if (i,m)!= (ind,mu) else epsilon for m in range(dim)] for i in range(Nparticles) ])   for mu in range(dim)] for ind in range(Nparticles) ] 
                return np.einsum('inimna->ma',np.array([[ (D_c(x+dx[ind][mu]) - D_c(x-dx[ind][mu]))/(2*epsilon) for mu in range(dim)] for ind in range(Nparticles) ] ))
            self.w_projections = - self.data.trajectory_integral(lambda t : grad_D_c(self.data.X_strat[t]))

        elif self.diffusion_data["type"] == "DiffusionInference":
            # Compute the projection of w = D grad log P using an
            # integration by parts, with shortcuts to compute only
            # gradients of the projection functions once.
            def grad_bD_bF(X):
                # The gradient of the product of the b functions of
                # the SFI instance (self) and the DiffusionInference,
                # traced over particles.
                return np.einsum('imia,ib->mab',self.DI.projectors.grad_b(X),self.projectors.b(X))+\
                       np.einsum('ia,imib->mab',self.DI.projectors.b(X),self.projectors.grad_b(X))

            int_grad_bD_bF = self.data.trajectory_integral(lambda t : grad_bD_bF(self.data.X_strat[t]))
            self.w_projections = - np.einsum('mab,mna,bc->nc', int_grad_bD_bF, self.DI.D_coefficients,self.projectors.H, optimize=True)
              
        self.w_ansatz,self.w_coefficients = self.projectors.projector_combination(self.w_projections)
        
        # Reconstruct the force F_mu = v_mu + w_mu
        self.F_projections = self.w_projections + self.v_projections
        self.F_ansatz,self.F_coefficients = self.projectors.projector_combination(self.F_projections)

        # The systematic bias on heat, entropy production and
        # information:
        self.Nb = np.prod(self.v_projections.shape) 
        self.Sdot_bias = 2 * self.Nb /self.data.tauN
        self.C_bias = 0.5 * self.Nb /self.data.tauN

        # Compute the entropy production and information along the
        # trajectory. Note they are biased estimators.

        if self.diffusion_data["type"] == "constant":
            # When D is constant there's a simple formula for the
            # information and entropy production: D^-1_munu F_mualpha F_numalpha.
            self.Heat        =        np.einsum('ma,na,mn->',self.v_projections ,self.F_projections, self.Dinv) * self.data.tauN
            self.DeltaS      =        np.einsum('ma,na,mn->',self.v_projections ,self.v_projections, self.Dinv) * self.data.tauN
            self.Information = 0.25 * np.einsum('ma,na,mn->',self.F_projections ,self.F_projections, self.Dinv) * self.data.tauN
        else:

            Dinv_Ito       = [ self.Dinv(X) for X in self.data.X_ito ] 
            ansatz_F_Ito   = [ self.F_ansatz(X) for X in self.data.X_ito ]
            ansatz_v_Ito   = [ self.v_ansatz(X) for X in self.data.X_ito ]

            self.Information = 0.25 * self.data.trajectory_integral(lambda t : np.einsum('imn,im,in->',Dinv_Ito[t],ansatz_F_Ito[t],ansatz_F_Ito[t]) ) * self.data.tauN
            self.DeltaS =  self.data.trajectory_integral(lambda t : np.einsum('imn,im,in->',Dinv_Ito[t],ansatz_v_Ito[t],ansatz_v_Ito[t]) ) * self.data.tauN
            self.Heat =  self.data.trajectory_integral(lambda t : np.einsum('imn,im,in->',self.Dinv(self.data.X_strat[t]),self.F_ansatz(self.data.X_strat[t]),self.data.Xdot[t])) * self.data.tauN
            

        # Per-particle rates:
        self.Sdot     = self.DeltaS / self.data.tauN
        self.Capacity = self.Information / self.data.tauN
        
        # The self-consistent uncertainty on the entropy production
        # and capacity estimators. IMPORTANT NOTE: this is only for
        # constant diffusions, we do not have a self-consistent
        # estimator of the error in the multiplicative noise case.
        self.Sdot_error = (2 * self.Sdot/self.data.tauN )**0.5 + self.Sdot_bias
        self.C_error = (2. * self.Capacity/self.data.tauN )**0.5 + self.C_bias
        
        self.error_DeltaS = self.data.tauN * self.Sdot_error
        self.error_Information = self.data.tauN * self.C_error

        self.projections_self_consistent_error = 0.5 * self.Nb / self.Information

        if verbose:
            self.print_report()

            
    def compute_accuracy(self,F_exact,D_exact,data_exact=None,verbose=True):
        """Evaluate the accuracy of the method when the actual force field is
        known.
        """
        if data_exact is None:
            # In the case of noisy input data, we want to compare to
            # the force inferred on the real trajectory, not the noisy
            # one (which would use values of the force field that
            # weren't explored by the trajectory, and thus cannot be
            # predicted).
            data_exact = self.data
            
        self.exact_F_Ito = [ F_exact(X) for X in data_exact.X_ito ]
        self.exact_F_Strat = [ F_exact(X) for X in data_exact.X_strat ]
        self.ansatz_F_Ito   = [ self.F_ansatz(X) for X in data_exact.X_ito ]
        if self.compute_phi:
            self.ansatz_phi = [ self.phi_ansatz(X) for X in data_exact.X_ito ]

        self.exact_D = D_exact
        self.exact_Dinv = lambda X : np.linalg.inv(self.exact_D(X))

        #self.total_Information = 0.25 * np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_ito[t]),self.exact_F_Ito[t],data_exact.Xdot[t]) for t in range(len(data_exact.Xdot))]))
        self.total_Information = 0.25 * np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_ito[t]),self.exact_F_Ito[t],self.exact_F_Ito[t]) for t in range(len(data_exact.Xdot))]))
        self.total_Heat = np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_strat[t]),self.exact_F_Strat[t],data_exact.Xdot[t]) for t in range(len(data_exact.Xdot))]))
        
        self.ansatz_F_Ito   = [ self.F_ansatz(X) for X in data_exact.X_ito ]
        self.exact_F_projections = data_exact.inner_product_empirical( self.exact_F_Ito, self.projectors.c, integration_style = 'Ito' ) 
        self.exact_F_ansatz,self.exact_F_coefficients = self.projectors.projector_combination(self.exact_F_projections)
            
        self.force_projections_error = 0.25 * np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_ito[t]),self.exact_F_Ito[t]-self.ansatz_F_Ito[t],self.exact_F_Ito[t]-self.ansatz_F_Ito[t]) for t in range(len(data_exact.Xdot))])) / self.Information

        self.exact_Information = 0.25 * np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_ito[t]),self.exact_F_ansatz(data_exact.X_ito[t]),self.exact_F_ansatz(data_exact.X_ito[t])) for t in range(len(data_exact.Xdot))]))
        self.exact_Heat = np.einsum('t,t->', data_exact.dt, np.array([ np.einsum('imn,im,in->',self.exact_Dinv(data_exact.X_strat[t]),self.exact_F_ansatz(data_exact.X_strat[t]),data_exact.Xdot[t]) for t in range(len(data_exact.Xdot))]))

        if verbose:
            # Total heat / information are with the full force field;
            # exact are with the exact projection of the force field
            # onto the trajectory projector.
            print("Heat: total/exact/inferred/inferred DeltaS/bootstrapped error",self.total_Heat,self.exact_Heat,self.Heat,self.DeltaS,self.error_DeltaS)
            print("Information: total/exact/inferred/bootstrapped error",self.total_Information,self.exact_Information,self.Information,self.error_Information)
            print("Information / entropy bias:",self.C_bias*data_exact.tauN,self.Sdot_bias*data_exact.tauN)
            print("Information in bits: total/exact/inferred/error",self.total_Information/np.log(2),self.exact_Information/np.log(2),self.Information/np.log(2),self.error_Information/np.log(2))
            print("Error on projections/self-consistent error:",self.force_projections_error,self.projections_self_consistent_error)
                
    def print_report(self):
        """ Tell us a bit about yourself.
        """
        print("Heat: inferred/DeltaS/bootstrapped error",self.Heat,self.DeltaS,self.error_DeltaS)
        print("Information: inferred/bootstrapped error",self.Information,self.error_Information)
        print("Information / entropy bias:",self.C_bias*self.data.tauN,self.Sdot_bias*self.data.tauN)
        print("Information in bits: inferred/error",self.Information/np.log(2),self.error_Information/np.log(2))
        print("Self-consistent error of F projections:",self.projections_self_consistent_error)

        
    def simulate_bootstrapped_trajectory(self,oversampling=1,use_drift_ansatz=False,divD=None):
        """Simulate an overdamped Langevin trajectory with the inferred
        ansatz force field and similar time series and initial
        conditions as the input data.
        """
        from SFI_langevin import OverdampedLangevinProcess
        if use_drift_ansatz:
            # Here we use the Ito average of xdot (the drift) as force
            # field; it is our best approximation of F+div D if a
            # variable D is co-inferred and there is no measurement
            # noise.
            return OverdampedLangevinProcess(self.phi_ansatz, self.D, self.data.t, initial_position = 1. * self.data.X_ito[0],oversampling=oversampling, D_is_constant = (self.diffusion_data["type"] == "constant"),divD = lambda X : 0.*X)
        else:
            # In this case we simply use our force ansatz as the
            # force, and let the divD deal with the diffusion-induced
            # drift.
            return OverdampedLangevinProcess(self.F_ansatz, self.D, self.data.t, initial_position = 1. * self.data.X_ito[0],oversampling=oversampling, D_is_constant = (self.diffusion_data["type"] == "constant"), divD = divD)

    
    def cycle_analysis(self):

        """Analyze the order 1 outcome of the expansion and find the cycles
        that produce the most entropy.

        """
        if self.basis['type'] != 'polynomial' or self.basis['order'] != 1:
            raise KeyError("Cycle analysis only valid for order 1 polynomial expansion.")
        
        # Non-dimensionalize the velocity coefficients in a symmetric
        # way by going to covariance-identity coordinates. For a
        # hierarchical polynomial basis B[1][1] is C^{-1/2}, with C
        # the covariance matrix.
        Omega = np.einsum('ai,ib->ab',self.projectors.H[1:,1:],self.v_projections[:,1:])
        Omega_antisym = (Omega - Omega.T)/2.
        # Find cycles:
        evals,evecs = np.linalg.eigh( Omega_antisym.dot(Omega_antisym.T) )
        self.cycles = []
        Hinv = np.linalg.inv(self.projectors.H[1:,1:])
        for i in range(self.data.d//2):
            # Get back to real-space coordinates:
            u = np.einsum('ia,a',Hinv,evecs[:,self.data.d - 1 - 2*i ])
            v = np.einsum('ia,a',Hinv,evecs[:,self.data.d - 2 - 2*i ])
            # Re-orthonormalize the vectors
            u /= np.linalg.norm(u)
            v -= (u.dot(v))*u
            v /= np.linalg.norm(v)
            s = 2. * evals[self.data.d - 1 - 2*i] 
            self.cycles.append((s,u,v)) 
        self.cycles.sort( key = lambda s : -s[0] )
