import numpy as np
from scipy.linalg import sqrtm, LinAlgError



class TrajectoryProjectors(object):
    """This class deals with the orthonormalization of the input functions
    b with respect to the inner product.

    The b function is a single lambda function with array
    input/output, with structure

    (Nparticles,dim) -> (Nparticles,Nfunctions)

    where Nfunctions is the number of functions in the basis.
    """
    def __init__(self,inner_product,functions):
        # The inner product should not contract spatial indices - only
        # perform time integration.
        self.inner = inner_product
        self.b = functions
        self.B = self.inner(self.b,self.b)
        # Compute the normalization matrix H = B^{-1/2}:
        try:
            self.H = sqrtm(np.linalg.inv(self.B))
        except LinAlgError:
            print("Warning, singular normalization matrix (redundant basis or dataset too small), using pseudo-inverse.")
            self.H = np.real(sqrtm(np.linalg.pinv(self.B)))
        self.c = lambda X : np.einsum( 'ia,ab->ib', self.b(X), self.H )
        self.grad_c = lambda X, epsilon=1e-6 : np.einsum( 'jmia,ab->jmib', self.grad_b(X,epsilon), self.H )

    def projector_combination(self,c_coefficients):
        """Returns a lambda function sum_alpha c_coefficients[..,alpha] *
        c_alpha(x). Avoids re-computing the dot product with the
        normalization matrix H."""
        b_coefficients = np.einsum('...a,ab->...b', c_coefficients, self.H)   
        function = lambda x : np.einsum('ia,...a->i...',self.b(x), b_coefficients) 
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
