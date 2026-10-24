"""Reproduction of the published local-SIS prescription (Wu et al. 2025).

Equation (31)/(40) with the secant of equations (28)/(30). All primary
comparisons use exact image geometry. Optional source-geometry simplifications
are explicitly phase-converted to the SAME exact minimum-delay convention.
"""
import numpy as np
from .approximations import PreparedApproximation
from .lens import potential,radial_derivatives


class LocalSISBaseline:
    def __init__(self,y,kappa_s):
        self.prepared = PreparedApproximation(y,kappa_s)
        p = self.prepared
        d1,d2,*_ = radial_derivatives(y,kappa_s)
        self.source_amplitude = float(np.sqrt(1+d2+d1/y))
        self.source_delay_error = float(-potential(y,kappa_s)-p.image.minimum_delay)

    def center(self,w):
        u = np.asarray(w,dtype=float)
        if np.any(u<=0):
            raise ValueError('center requires positive w')
        y,k = self.prepared.y,self.prepared.kappa_s
        # f(q)/(w*y^3)=psi(sqrt(2)/q)/(sqrt(2)*y^2), psi(0)=0.
        return potential(np.sqrt(2)/(u*y),k)/(np.sqrt(2)*y*y)

    def evaluate(self,w):
        p = self.prepared
        u = np.asarray(w,dtype=float)
        if np.any(u==0):
            raise ValueError('published asymptotic baseline is undefined at zero')
        v = p.evaluate(abs(u),'image1')+self.center(abs(u))*np.exp(1j*abs(u)*p.center_delay)
        v = np.where(u<0,np.conj(v),v)
        return v.item() if v.ndim==0 else v

    def evaluate_source_geometry(self,w):
        """Illustrative III.3 source approximations, NOT the primary Eq.31 test.

        Uses mu~1+2*kappa(y), t_min~-psi(y), omits image bGO as discussed
        there. Conversion to exact time zero puts the delay error in GO;
        the center carrier is then the same as in the exact-geometry baseline.
        """
        p = self.prepared
        u = np.asarray(w,dtype=float)
        if np.any(u==0):
            raise ValueError('nonzero frequency required')
        v = self.source_amplitude*np.exp(1j*abs(u)*self.source_delay_error)
        v = v+self.center(abs(u))*np.exp(1j*abs(u)*p.center_delay)
        v = np.where(u<0,np.conj(v),v)
        return v.item() if v.ndim==0 else v
