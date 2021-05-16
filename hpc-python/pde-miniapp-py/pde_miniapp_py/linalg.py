"""
Collection of linear algebra operations and CG solver
"""
from mpi4py import MPI
# from numpy import lingalfg as LA
import numpy as np
from . import data
from . import operators

def hpc_dot(x, y):
    """Computes the inner product of x and y"""
    # ... implement ...
    # the standard for-loop implementations 
    z_local = np.zeros(1)
    z=np.zeros(1)
    #for i in np.arange(0, x.inner.shape[0]):
    #    for j in np.arange(0,x.inner.shape[1]):
    #       z_local[0] += x.inner[i,j]*y.inner[i,j]
    #x.domain.comm.Allreduce(z_local, z, op=MPI.SUM)
    # the numpy function 
    z_local[0]= np.dot(x.inner.flat,y.inner.flat)
    x.domain.comm.Allreduce(z_local, z, op=MPI.SUM)
    return z[0]

def hpc_norm2(x):
    """Computes the 2-norm of x"""
    # ... implement ...
    sum_local = np.zeros(1)
    sum = np.zeros(1)
    #for i in np.arange(0, x.inner.shape[0]):
    #    for j in np.arange(0, x.inner.shape[1]):
    #        sum_local[0] += x.inner[i,j]*x.inner[i,j]
    #x.domain.comm.Allreduce(sum_local, sum, op=MPI.SUM)
    
    # the numpy function 
    sum_local[0] = np.dot(x.inner.flat(),x.inner.flat())
    x.domain.comm.Allreduce(sum_local, sum, op=MPI.SUM)
    #return LA.norm(x)
    return np.sqrt(sum)[0]

class hpc_cg:
    """Conjugate gradient solver class: solve the linear system A x = b"""
    def __init__(self, domain):
        self._Ap = data.Field(domain)
        self._r  = data.Field(domain)
        self._p  = data.Field(domain)

        self._xold  = data.Field(domain)
        self._v  = data.Field(domain)
        self._Fxold  = data.Field(domain)
        self._Fx  = data.Field(domain)
        self._v  = data.Field(domain)

    def solve(self, A, b, x, tol, maxiter):
        """Solve the linear system A x = b"""
        # initialize
        A(x, self._Ap)
        self._r.inner[...] = b.inner[...] - self._Ap.inner[...]
        self._p.inner[...] = self._r.inner[...]
        delta_kp = hpc_dot(self._r, self._r)
 
        # iterate
        converged = False
        for k in range(0, maxiter):
            delta_k = delta_kp
            if delta_k < tol**2:
                converged = True
                break
            A(self._p, self._Ap)
            alpha = delta_k/hpc_dot(self._p, self._Ap)
            x.inner[...] += alpha*self._p.inner[...]
            self._r.inner[...] -= alpha*self._Ap.inner[...]
            delta_kp = hpc_dot(self._r, self._r)
            self._p.inner[...] = ( self._r.inner[...]
                                  + delta_kp/delta_k*self._p.inner[...] )

        return converged, k + 1

