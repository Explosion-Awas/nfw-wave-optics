"""Numerical scalar diffraction on analytic split-Hankel contours.

This algorithm is independent of GLoW's equal-delay contour integration.
The adaptive quadrature error is not a rigorous bound on total error.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import hankel1e, hankel2e
from .lens import potential, single_image


@dataclass(frozen=True)
class ReferenceResult:
    value: complex
    error_estimate: float
    seconds: float
    evaluations: int
    warnings: tuple[str,...]


def amplification_contour(w: float, y: float, kappa_s: float, *,
                          tol=1e-10, shape=1.0, tail_scale=1.0) -> ReferenceResult:
    """Compute F with exp(+iwT) and the exact minimum delay removed.

    J0=(H1+H2)/2. H1 is integrated along z=(1+i*a)t;
    H2 along z=t+i*a*t*(t-x_min)/(t+x_min). Both contours remain
    in Re(z)>0; scaled Hankel functions avoid exponentially large factors.
    Changing shape/tail_scale/tol supplies distinct convergence checks.
    """
    start = perf_counter()
    if w <= 0 or tol <= 0 or shape <= 0:
        raise ValueError('w, tol, and shape must be positive')
    image = single_image(y,kappa_s)
    xm,tm = image.x,image.minimum_delay
    width = 1/np.sqrt(w*min(image.lambda_r,1.0))
    end = xm+tail_scale*18*width*max(1,1/shape)
    central = min(1/(w*y),width)
    points = [0.0,end,xm]
    points += [n*central for n in [0.1,0.5,1,3,8,20,40,80]]
    points += [xm+n*width for n in [-10,-6,-3,-1,1,3,6,10]]
    points = sorted(set(float(p) for p in points if 0 <= p <= end))
    n_eval = 0
    def integrand(t,sign):
        nonlocal n_eval
        n_eval += 1
        if sign == 1:
            z = (1+1j*shape)*t
            dz = 1+1j*shape
            h = hankel1e(0,w*y*z)
        else:
            z = t+1j*shape*t*(t-xm)/(t+xm)
            dz = 1+1j*shape*(1-2*xm*xm/(t+xm)**2)
            h = hankel2e(0,w*y*z)
        phase = (z*z+y*y)/2-potential(z,kappa_s)+sign*y*z-tm
        return z*dz*h*np.exp(1j*w*phase)
    accum = 0j
    error = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', IntegrationWarning)
        for sign in [1,-1]:
            for left,right in zip(points[:-1],points[1:]):
                target = tol/max(w,1)/len(points)
                re,er = quad(lambda t: integrand(t,sign).real,left,right,
                             epsabs=target,epsrel=tol/8,limit=200)
                im,ei = quad(lambda t: integrand(t,sign).imag,left,right,
                             epsabs=target,epsrel=tol/8,limit=200)
                accum += re+1j*im
                error += er+ei
    return ReferenceResult(-1j*w/2*accum,w/2*error,perf_counter()-start,
                           n_eval,tuple(str(c.message) for c in caught))


def amplification_real_tail(w: float,y: float,kappa_s: float,*,tol=1e-10) -> ReferenceResult:
    """Cross-check: real-axis J0 quadrature, with only the far tail rotated.

    Different contour and no Hankel decomposition. Slower when w*y*y is large.
    """
    from scipy.special import jve
    start = perf_counter()
    image = single_image(y,kappa_s)
    R = image.x+max(2.0,2/np.sqrt(w))
    end = 18/np.sqrt(w)
    dz = np.exp(1j*np.pi/4)
    neval = 0
    def f(t,tail):
        nonlocal neval
        neval += 1
        z = R+dz*t if tail else complex(t)
        qz = w*y*z
        phase = (z*z+y*y)/2-potential(z,kappa_s)-image.minimum_delay
        return z*jve(0,qz)*np.exp(1j*w*phase+abs(qz.imag))*(dz if tail else 1)
    total = 0j
    err = 0.0
    seg = max(1,int(np.ceil(w*R*(R+y)/40)))
    points = np.linspace(0,R,seg+1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', IntegrationWarning)
        for tail,pts in [(False,points),(True,[0,end])]:
            for left,right in zip(pts[:-1],pts[1:]):
                re,er = quad(lambda t:f(t,tail).real,left,right,epsabs=tol/(w*(seg+1)),epsrel=tol/8,limit=500)
                im,ei = quad(lambda t:f(t,tail).imag,left,right,epsabs=tol/(w*(seg+1)),epsrel=tol/8,limit=500)
                total += re+1j*im
                err += er+ei
    return ReferenceResult(-1j*w*total,w*err,perf_counter()-start,neval,
                           tuple(str(c.message) for c in caught))
