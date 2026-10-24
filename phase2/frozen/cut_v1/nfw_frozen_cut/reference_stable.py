"""Same split-Hankel integral, stable phase and F-normalized tolerances.

This is a numerical hygiene improvement for the second-stage reference,
not a claim of a new wave-optics algorithm. First-stage code is unchanged.
"""
from __future__ import annotations
from time import perf_counter
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import hankel1e, hankel2e
from nfw_wave.lens import potential, single_image
from nfw_wave.reference import ReferenceResult


def amplification_stable(w,y,kappa_s,*,tol=1e-12,shape=1.,tail_scale=1.):
    start = perf_counter()
    image = single_image(y,kappa_s)
    xm,tm = image.x,image.minimum_delay
    width = 1/np.sqrt(w*min(image.lambda_r,1.))
    end = xm+tail_scale*18*width*max(1.,1/shape)
    central = min(1/(w*y),width)
    points = [0.,end,xm]
    points += [n*central for n in [.1,.5,1.,3.,8.,20.,40.,80.]]
    points += [xm+n*width for n in [-10.,-6.,-3.,-1.,1.,3.,6.,10.]]
    points = sorted(set(float(p) for p in points if 0<=p<=end))
    neval = 0
    def integrand(t,sign):
        nonlocal neval
        neval += 1
        if sign == 1:
            z = (1+1j*shape)*t
            dz = 1+1j*shape
            h = hankel1e(0,w*y*z)
        else:
            z = t+1j*shape*t*(t-xm)/(t+xm)
            dz = 1+1j*shape*(1-2*xm*xm/(t+xm)**2)
            h = hankel2e(0,w*y*z)
        # Avoid (z*z+y*y)/2-y*z cancellation near the minimum image.
        phase = (z+sign*y)**2/2-potential(z,kappa_s)-tm
        return z*dz*h*np.exp(1j*w*phase)
    total = 0j
    error = 0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",IntegrationWarning)
        for sign in (1,-1):
            for left,right in zip(points[:-1],points[1:]):
                # Accumulation is multiplied by w/2; normalize absolute
                # tolerances to the requested final F, also for w<<1.
                target = tol/(4*w*(len(points)-1))
                re,er = quad(lambda t:integrand(t,sign).real,left,right,
                             epsabs=target,epsrel=tol/8,limit=300)
                im,ei = quad(lambda t:integrand(t,sign).imag,left,right,
                             epsabs=target,epsrel=tol/8,limit=300)
                total += re+1j*im
                error += er+ei
    return ReferenceResult(-.5j*w*total,.5*w*error,perf_counter()-start,neval,
                           tuple(str(c.message) for c in caught))
