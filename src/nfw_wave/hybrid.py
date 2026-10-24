"""Conservative empirical gate with explicit numerical fallback.

No returned error estimate is advertised as a rigorous mathematical bound.
The analytic envelope is calibrated only in the stated finite parameter box.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .approximations import PreparedApproximation
from .lens import radial_derivatives
from .reference import amplification_contour


@dataclass(frozen=True)
class HybridResult:
    value: complex
    method: str
    error_estimate: float
    empirical: bool
    seconds: float
    diagnostics: dict


class HybridLens:
    def __init__(self,y,kappa_s,*,safety_factor=2.0):
        if safety_factor<2.0:
            raise ValueError('safety_factor must be >= the calibrated conservative value 2')
        self.prepared = PreparedApproximation(y,kappa_s)
        self.safety_factor = float(safety_factor)
        p = self.prepared
        r,a,b = p.image.x,p.image.lambda_r,p.image.lambda_t
        _,_,d3,d4 = radial_derivatives(r,kappa_s)
        self.delta_abs = abs(d4)/(8*a*a)+5*d3*d3/(24*a**3)+abs(d3)/(4*r*a*a)+abs(a-b)/(8*r*r*a*b)

    def diagnostics(self,w):
        p = self.prepared
        y,k,r = p.y,p.kappa_s,p.image.x
        a,b = p.image.lambda_r,p.image.lambda_t
        u = abs(float(w))
        if u==0:
            raise ValueError('positive nonzero frequency is required')
        img = p.image.amplitude*(self.delta_abs/(u*u*r*r*min(a,b,1)**2)+p.delta**2/u**2)
        ctr = abs(p.center(u,'ft0'))*8/(u*y*y)*(1+2*k*(abs(np.log(u*y))+np.euler_gamma+1.5))
        floor = 3e-14*(1+u*abs(p.center_delay))*p.image.amplitude
        return dict(s=u*y*y/2,q=u*y,eta_image=1/(u*r*r*min(a,b)),
                    calibrated_box=bool(.03<=k<=.5 and .25<=y<=10 and .1<=u<=300),
                    image_proxy=float(img),center_proxy=float(ctr),
                    empirical_estimate=float(self.safety_factor*(img+ctr)+floor),
                    safety_factor=self.safety_factor)

    def evaluate(self,w,*,atol=1e-6):
        if atol<=0:
            raise ValueError('atol must be positive')
        start = perf_counter()
        p = self.prepared
        u = abs(float(w))
        d = self.diagnostics(w)
        if d['calibrated_box'] and d['s']>=5 and d['empirical_estimate']<=atol:
            return HybridResult(p.evaluate(w,'ft0'),'asymptotic_ft0',d['empirical_estimate'],
                                True,perf_counter()-start,d)
        target = min(1e-8,atol/20)
        for attempt in range(3):
            a = amplification_contour(u,p.y,p.kappa_s,tol=target,shape=.85)
            b = amplification_contour(u,p.y,p.kappa_s,tol=target/10,shape=1.2,tail_scale=1.2)
            error = max(abs(a.value-b.value),a.error_estimate,b.error_estimate,
                        3e-14*(1+u*abs(p.center_delay))*abs(b.value))
            if error<=atol/4 and not a.warnings and not b.warnings:
                value = b.value if w>0 else np.conj(b.value)
                d.update(numerical_convergence_delta=abs(a.value-b.value),
                         numerical_warning_count=len(a.warnings+b.warnings))
                return HybridResult(value,'numerical_contour',float(error),False,
                                    perf_counter()-start,d)
            target /= 10
        raise ArithmeticError(f'numerical reference did not meet requested tolerance {atol}; '
                              f'last estimate={error}; warnings={a.warnings+b.warnings}')
