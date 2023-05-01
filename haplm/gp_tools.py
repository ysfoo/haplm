import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.gp.util import (
    cholesky,
    solve_lower,
    stabilize,
)

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi

# Spatiotemporal covariance function
# ls_spat is length scale in degrees
# ls_temp is time scale
# sg = SphereGneiting(1, 1)
# print(sg.full(np.array([[0, 0, 0], [2, 0, 2]])).eval())
# returns 1 / 7
class SphereGneiting(pm.gp.cov.Covariance):
    def __init__(self, ls_spat, ls_temp,
                 func_spat=lambda x: 1.0/(1.0+x),
                 func_temp=lambda t: 1.0+t, active_dims=None):
        super().__init__(3, active_dims)
        self.ls_spat = pt.as_tensor_variable(ls_spat)
        self.ls_temp = pt.as_tensor_variable(ls_temp)
        self.func_spat = func_spat
        self.func_temp = func_temp

    def diag(self, X):
        return pt.alloc(1.0, X.shape[0])

    def gc_dist(self, X, Xs):
        lam1, phi1 = X[:,0]*DEG_TO_RAD, X[:,1]*DEG_TO_RAD
        lam2, phi2 = Xs[:,0]*DEG_TO_RAD, Xs[:,1]*DEG_TO_RAD
        dist = pt.arccos(pt.clip(pt.outer(pt.sin(phi1), pt.sin(phi2)) +
                                 pt.outer(pt.cos(phi1), pt.cos(phi2)) *
                                 pt.cos(pt.reshape(lam1, (-1, 1)) - pt.reshape(lam2, (1, -1))),
                                 -1, 1))
        return dist * RAD_TO_DEG

    def full(self, X, Xs=None):
        Xs = X if Xs is None else Xs
        tdiff = (pt.reshape(X[:,2], (-1, 1)) - pt.reshape(Xs[:,2], (1, -1))) / self.ls_temp
        tcmp_inv = 1.0 / self.func_temp(tdiff*tdiff)
        #pt.printing.Print("cov")(self.func_spat(self.gc_dist(X, Xs) * tcmp_inv / self.ls_spat) * tcmp_inv)
        return self.func_spat(self.gc_dist(X, Xs) * tcmp_inv / self.ls_spat) * tcmp_inv
    
# extend PyMC's GP implementation to include marginal predictive distribution
class GP(pm.gp.gp.Latent):    
    def _build_marg_cond(self, Xnew, X, f, cov_total, mean_total):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, f - mean_total(X))
        mu = self.mean_func(Xnew) + pt.dot(pt.transpose(A), v)

        Kss = self.cov_func(Xnew, diag=True)
        var = Kss - pt.sum(pt.square(A), 0)
        return mu, var

    def marg_cond(self, name, Xnew, given=None, **kwargs): 
        givens = self._get_given_vals(given)
        mu, var = self._build_marg_cond(Xnew, *givens)
        return pm.Normal(name, mu=mu, sigma=pt.sqrt(var), **kwargs) 