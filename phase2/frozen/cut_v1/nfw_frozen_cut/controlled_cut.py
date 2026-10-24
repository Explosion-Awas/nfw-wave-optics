"""Empirical quadrature diagnostics; explicitly not rigorous error bounds."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .batch_cut import PreparedCutNFW
from .reference_stable import amplification_stable


@dataclass(frozen=True)
class ControlledCutResult:
    value: object
    residual: object
    error_estimate: object
    route: str
    converged: bool
    seconds: float
    estimator_type: str = "empirical_node_indicator"
    warnings: tuple = ()
    band_relative_indicator: object = None


class ControlledCutNFW:
    def __init__(self,y,kappa_s):
        self.y,self.kappa_s = float(y),float(kappa_s)
        self.coarse = PreparedCutNFW(y,kappa_s,image_nodes=48,center_nodes=48)
        self.fine = PreparedCutNFW(y,kappa_s,image_nodes=96,center_nodes=96)
        self.higher = None

    def evaluate(self,w,*,atol=1e-8,band_rtol=None):
        start = perf_counter()
        if not np.isfinite(atol) or atol<=0:
            raise ValueError("absolute tolerance must be finite and positive")
        frequencies = np.asarray(w,dtype=float)
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies==0):
            raise ValueError("require finite nonzero frequencies")
        if band_rtol is not None:
            if (not np.isfinite(band_rtol) or band_rtol<=0 or frequencies.ndim!=1
                or frequencies.size<2 or np.any(frequencies<=0) or np.any(np.diff(frequencies)<=0)):
                raise ValueError("band gate requires positive increasing frequencies and positive rtol")
        q = np.abs(frequencies)*self.y
        in_box = (8<=self.y<=256 and .03<=self.kappa_s<=.7
                  and np.all((q>=.3)&(q<=12)))
        if not in_box:
            return self._reference(frequencies,atol,start,band_rtol)
        coarse,fine = self.coarse.evaluate(w),self.fine.evaluate(w)
        floor = 16*np.finfo(float).eps*(1+np.abs(np.asarray(w)*self.fine.p0))
        estimate = 2*np.abs(fine.residual-coarse.residual)+floor
        indicator = self._band_indicator(frequencies,fine.residual,estimate,band_rtol)
        converged = bool(np.all(estimate<=atol) and (band_rtol is None or indicator<=band_rtol))
        route = "cut96" if converged else "cut96_unresolved"
        if not converged:
            if self.higher is None:
                self.higher = PreparedCutNFW(self.y,self.kappa_s,image_nodes=192,center_nodes=192)
            previous,fine = fine,self.higher.evaluate(w)
            estimate = 2*np.abs(fine.residual-previous.residual)+floor
            indicator = self._band_indicator(frequencies,fine.residual,estimate,band_rtol)
            converged = bool(np.all(estimate<=atol) and (band_rtol is None or indicator<=band_rtol))
            route = "cut192" if converged else "cut192_unresolved"
        if not converged:
            return self._reference(frequencies,atol,start,band_rtol)
        return ControlledCutResult(fine.value,fine.residual,estimate,
                                   route,
                                   converged,perf_counter()-start,band_relative_indicator=indicator)

    @staticmethod
    def _band_indicator(frequencies,residuals,estimates,band_rtol):
        if band_rtol is None:
            return None
        # The triangle-inequality form is not a proof because E is empirical.
        logw = np.log(frequencies)
        e = np.sqrt(np.trapezoid(np.asarray(estimates)**2,logw))
        r = np.sqrt(np.trapezoid(np.abs(residuals)**2,logw))
        return float(e/(r-e)) if r>e else float("inf")

    def _reference(self,frequencies,atol,start,band_rtol):
        # QUAD diagnostics and a roundoff floor, not a rigorous certificate.
        values,errors,messages = [],[],[]
        target = max(5e-14,min(2e-12,atol/16))
        for w in frequencies.reshape(-1):
            ref = amplification_stable(float(abs(w)),self.y,self.kappa_s,tol=target)
            values.append(ref.value if w>0 else ref.value.conjugate())
            floor = 16*np.finfo(float).eps*(1+abs(w*self.fine.p0))
            errors.append(max(4*ref.error_estimate,floor))
            messages.extend(ref.warnings)
        values = np.array(values).reshape(frequencies.shape)
        errors = np.array(errors).reshape(frequencies.shape)
        residuals = values-self.fine.image.amplitude
        indicator = self._band_indicator(frequencies,residuals,errors,band_rtol)
        converged = bool(not messages and np.all(errors<=atol)
                         and (band_rtol is None or indicator<=band_rtol))
        if frequencies.ndim == 0:
            values,residuals,errors = values.item(),residuals.item(),errors.item()
        return ControlledCutResult(values,residuals,errors,"reference",converged,
                                   perf_counter()-start,"adaptive_quad_diagnostic",tuple(messages),indicator)
