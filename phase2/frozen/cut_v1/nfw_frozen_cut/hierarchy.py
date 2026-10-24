"""Candidate matched image/finite-momentum center expansions.

The hierarchy is a research candidate, not a uniform-domain guarantee.
Exact spatial kernels must not be called complete Fresnel--Born results.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from nfw_wave.image_expansion import ImageExpansion
from .spatial import nfw_power_hankel, nfw_linear_hankel


@lru_cache(maxsize=2048)
def _unit_power(q, power):
    result = nfw_power_hankel(q, power)
    if result.warnings:
        raise ArithmeticError("spatial quadrature did not converge: "+"; ".join(result.warnings))
    return result.value


@lru_cache(maxsize=2048)
def _linear(q, laps):
    return nfw_linear_hankel(q, laps)


class FiniteCenterNFW:
    def __init__(self,y,kappa_s):
        self.y, self.kappa_s = float(y), float(kappa_s)
        self.image_expansion = ImageExpansion(y,kappa_s)
        self.image = self.image_expansion.image
        self.center_delay = y*y/2-self.image.minimum_delay

    def evaluate(self,w,order=2):
        a = np.asarray(w,dtype=float)
        u = np.abs(a)
        q = u*self.y
        terms = []
        for frequency, momentum in zip(u.reshape(-1), q.reshape(-1)):
            k = self.kappa_s
            P = k*_linear(float(momentum),0)
            center = -frequency**2*P
            if order >= 2:
                DP = k*_linear(float(momentum),1)
                Q = k*k*_unit_power(float(momentum),2)
                center += .5j*frequency**3*(DP+Q)
            terms.append(center)
        center = np.asarray(terms).reshape(a.shape)
        value = self.image_expansion.evaluate(u)+center*np.exp(1j*u*self.center_delay)
        value = np.where(a < 0, value.conjugate(), value)
        return value.item() if value.ndim == 0 else value
