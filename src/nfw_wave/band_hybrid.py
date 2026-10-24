"""Band-level empirical residual control without dividing by pointwise residuals."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class BandResult:
    w: np.ndarray
    values: np.ndarray
    methods: tuple[str,...]
    error_estimates: np.ndarray
    estimated_residual_relative_l2: float
    seconds: float
    empirical: bool
    vectorized_fast_path: bool = False


def evaluate_band(lens,w,*,atol=1e-6,residual_rtol=.01):
    """Refine selected frequencies until both requested diagnostics pass.

    Uses d(log w). If ||error|| <= E and approximate residual has norm R,
    then the true residual norm is at least R-E. Here E is an empirical /
    convergence estimate, not a proven bound, so the returned control is
    empirical too. Refining the frequency mesh is a separate responsibility.
    """
    start = perf_counter()
    w = np.asarray(w,dtype=float)
    if w.ndim!=1 or len(w)<2 or np.any(w<=0) or np.any(np.diff(w)<=0):
        raise ValueError('need >=2 strictly increasing positive frequencies')
    if atol<=0 or residual_rtol<=0:
        raise ValueError('positive absolute and residual tolerances are required')
    logw = np.log(w)
    width = logw[-1]-logw[0]
    norm = lambda z: float(np.sqrt(trapezoid(abs(np.asarray(z))**2,logw)/width))
    p = lens.prepared
    y,k,r = p.y,p.kappa_s,p.image.x
    a,b = p.image.lambda_r,p.image.lambda_t
    in_box = .03<=k<=.5 and .25<=y<=10 and np.all((w>=.1)&(w<=300))
    if in_box and np.all(w*y*y/2>=5):
        image_proxy = p.image.amplitude*(lens.delta_abs/(w*w*r*r*min(a,b,1)**2)+p.delta**2/w**2)
        center_proxy = abs(p.center(w,'ft0'))*8/(w*y*y)*(1+2*k*(abs(np.log(w*y))+np.euler_gamma+1.5))
        estimates = lens.safety_factor*(image_proxy+center_proxy)+3e-14*(1+w*abs(p.center_delay))*p.image.amplitude
        values = p.evaluate(w,'ft0')
        E,R = norm(estimates),norm(values-p.image.amplitude)
        if np.max(estimates)<=atol and R>1e-13 and E<=residual_rtol/(1+residual_rtol)*R:
            return BandResult(w,values,('asymptotic_ft0',)*len(w),estimates,E/(R-E),
                              perf_counter()-start,True,True)
    answers = [lens.evaluate(float(u),atol=atol) for u in w]
    for _ in range(3*len(w)+1):
        values = np.array([a.value for a in answers])
        errors = np.array([a.error_estimate for a in answers])
        E = norm(errors)
        R = norm(values-lens.prepared.image.amplitude)
        if R<=1e-13:
            raise ArithmeticError('diffraction residual is unresolved in this frequency band')
        if E<=residual_rtol/(1+residual_rtol)*R:
            return BandResult(w,values,tuple(a.method for a in answers),errors,E/(R-E),
                              perf_counter()-start,any(a.empirical for a in answers))
        j = int(np.argmax(errors))
        tighter = min(atol,residual_rtol/(1+residual_rtol)*R/4)
        answers[j] = lens.evaluate(float(w[j]),atol=tighter)
    raise ArithmeticError('band residual diagnostic did not converge; use an independently converged numerical method')
