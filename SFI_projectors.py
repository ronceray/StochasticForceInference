import numpy as np
from scipy.linalg import sqrtm, qr, LinAlgError

def flip(H):
    (w,h) = H.shape
    return np.array([[ H[w-1-i,h-1-j] for i in range(h)] for j in range(w)])
    

class TrajectoryProjectors(object):
    """This class deals with the orthonormalization of the input functions
    b with respect to the inner product.

    The b function is a single lambda function with array
    input/output, with structure

    (Nparticles,dim) -> (Nparticles,Nfunctions)

    where Nfunctions is the number of functions in the basis.

    is_interacting indicates whether the functions couple the
    different particles, or if they are treated as independent
    (leading to a speedup of the code).

    """
    def __init__(self,inner_product,functions,is_interacting):
        # The inner product should not contract spatial indices - only
        # perform time integration.
        self.inner = inner_product
        self.b = functions
        self.is_interacting = is_interacting
        self.B = self.inner(self.b,self.b) 
        self.Nfunctions = self.B.shape[0]
        # Compute the normalization matrix H = B^{-1/2}:
        try:
            self.H = flip(qr(sqrtm(flip(np.linalg.inv(self.B))),mode='r')[0])
            #self.H = sqrtm(np.linalg.inv(self.B))
        except LinAlgError:
            print("Warning, singular normalization matrix (redundant basis or dataset too small), using pseudo-inverse.")
            self.H = flip(qr(sqrtm(flip(np.linalg.pinv(self.B))),mode='r')[0])
            #self.H = np.real(sqrtm(np.linalg.pinv(self.B)))        
        #self.c = lambda X : np.einsum( 'ia,ab->ib', self.b(X), self.H )
        #self.grad_c = lambda X, epsilon=1e-6 : np.einsum( 'jmia,ab->jmib', self.grad_b(X,epsilon), self.H )

    def projector_combination(self,c_coefficients):
        """Returns a lambda function sum_alpha c_coefficients[..,alpha] *
        c_alpha(x). Avoids re-computing the dot product with the
        normalization matrix H."""
        b_coefficients = np.einsum('...a,ba->...b', c_coefficients, self.H)   
        function = lambda x : np.einsum('ia,...a->i...',self.b(x), b_coefficients) 
        return function, 1.*b_coefficients

    def truncated_projector_combination(self,truncated_c_coefficients):
        """Returns a lambda function sum_alpha c_coefficients[..,alpha] *
        c_alpha(x). Avoids re-computing the dot product with the
        normalization matrix H."""
        truncation_order = truncated_c_coefficients.shape[1] 
        b_coefficients = np.einsum('...a,ba->...b', truncated_c_coefficients, self.H[:truncation_order,:truncation_order])   
        function = lambda x : np.einsum('ia,...a->i...',self.b(x)[:,:truncation_order], b_coefficients) 
        return function, 1.*b_coefficients


    def grad_b(self,x,epsilon=1e-6):
        """The numerical gradient of the input function (to do: add an option
        for analytical formula for gradient). Output has shape:
        Nparticles x dim x Nparticles x Nfunctions.

        """
        Nparticles,d = x.shape
        dx = [[ np.array([[ 0 if (i,m)!= (ind,mu) else epsilon for m in range(d)] for i in range(Nparticles) ] )\
                for mu in range(d) ] for ind in range(Nparticles) ]  
        return np.array([[ (self.b(x+dx[ind][mu]) - self.b(x-dx[ind][mu]))/(2*epsilon) for mu in range(d)] for ind in range(Nparticles) ])

    def grad_b_noninteracting(self,x,epsilon=1e-6):
        """The numerical gradient of the input function when all cross terms
        between particles are assumed independent (ie 

        d b_ia (x) / d x_jm = 0 if i != j
        
        Output has shape:   dim x Nparticles x Nfunctions.

        """
        Nparticles,d = x.shape
        dx = [ np.array([[ 0 if m != mu else epsilon for m in range(d)] for i in range(Nparticles) ] )\
                for mu in range(d) ] 
        return np.array([ (self.b(x+dx[mu]) - self.b(x-dx[mu]))/(2*epsilon) for mu in range(d)] )
