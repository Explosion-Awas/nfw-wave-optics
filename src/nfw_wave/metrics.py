"""Complex and frequency-band residual metrics, without pointwise residual ratios."""
import numpy as np
from scipy.integrate import trapezoid


def band_metrics(w,approx,reference,go,*,residual_floor=1e-12):
    w = np.asarray(w,dtype=float)
    a,r = np.asarray(approx,dtype=complex),np.asarray(reference,dtype=complex)
    if w.ndim!=1 or len(w)<2 or np.any(np.diff(w)<=0):
        raise ValueError('frequency band must have >=2 strictly increasing frequencies')
    x = np.log(w)
    length = x[-1]-x[0]
    e = a-r
    residual = r-go
    error_power = float(trapezoid(abs(e)**2,x)/length)
    residual_power = float(trapezoid(abs(residual)**2,x)/length)
    return dict(max_complex_abs=float(np.max(abs(e))),
                rms_complex_abs=float(np.sqrt(error_power)),
                max_phase_rad=float(np.max(abs(np.angle(a*np.conj(r))))),
                band_residual_rms=float(np.sqrt(residual_power)),
                band_residual_relative_l2=float(np.sqrt(error_power/residual_power))
                    if residual_power>residual_floor**2 else None,
                band_measure='d(log w)')
