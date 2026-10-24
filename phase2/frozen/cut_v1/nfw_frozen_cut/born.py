"""Phase-explicit complete order-1/2/3 Fresnel--Born baselines for standard NFW.

With psi=k*p and P_w the normalized exp(+i*w*|x-y|**2/2) propagator,
B=-i*w*(P_w[p](y)-p(y)). The quadratic free Fresnel phase is retained.
At order n, C_n=(-iw)**n/n! * P_w[(p-p(y))**n].
This is numerical quadrature, not a fast Fourier/central-cusp surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import factorial
from time import perf_counter
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import hankel1e, hankel2e, jve


@dataclass(frozen=True)
class BornResult:
    """Quadrature diagnostics, NOT a bound on the physical Born truncation.

    ``coefficient`` is B (None when the k=0 shortcut skips its evaluation).
    ``phase_shift`` is the time shift used to transport the source polynomial.
    ``error_estimate`` includes k scaling but excludes tail truncation,
    binary64 phase/geometry errors, and higher-strength terms.
    Numerical warnings and model caveats are deliberately separate.
    """
    value: complex
    error_estimate: float
    seconds: float
    evaluations: int
    warnings: tuple[str, ...]
    phase: str
    coefficient: complex | None = None
    coefficient_error_estimate: float = 0.0
    model_warnings: tuple[str, ...] = ()
    phase_shift: float = 0.0
    model_error_estimate: None = None
    method: str = "split_hankel"
    completion: str = "first_order"
    order: int = 1
    coefficients: tuple[complex, ...] = ()
    coefficient_error_estimates: tuple[float, ...] = ()
    terms: tuple[complex, ...] = ()


def _evaluate_profile_born(w, y, strength, profile, *, tol=1e-10,
                           shape=1.0, tail_scale=1.0, method="split_hankel", order=1):
    """Source-gauge coefficient for an analytic radial test profile.

    ``split_hankel`` uses the parent's FREE contours (xm=y, tm=0).
    ``real_tail`` uses unsplit J0 on the real axis and a rotated far tail.
    Test profiles must be analytic along the chosen contours and not outgrow
    their Gaussian damping. No global arbitrary-precision state is changed.
    """
    start = perf_counter()
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= order <= 3:
        raise ValueError("order must be an integer from 1 through 3")
    py = profile(y)
    width = 1/np.sqrt(w)
    evaluations = 0

    if method == "real_tail":
        radius = y+max(2.0, 2*width)
        tail_end = tail_scale*18*width
        rotation = np.exp(1j*np.pi/4)
        segments = max(1, int(np.ceil(w*radius*(radius+y)/40)))
        points = list(np.linspace(0, radius, segments+1))
        points += [n/(w*y) for n in [0.1, 0.5, 1, 3, 8, 20]]
        points = sorted(set(float(p) for p in points if 0 <= p <= radius))
        jobs = [(False, points), (True, [0.0, tail_end])]
        factor = -w*w

        def integrand(t, tail):
            nonlocal evaluations
            evaluations += 1
            z = radius+rotation*t if tail else complex(t)
            dz = rotation if tail else 1.0
            qz = w*y*z
            phase = (z*z+y*y)/2
            return (z*dz*jve(0, qz)*np.exp(1j*w*phase+abs(qz.imag))
                    *(profile(z)-py)**power)
    else:
        end = y+tail_scale*18*width*max(1, 1/shape)
        central = min(1/(w*y), width)
        points = [0.0, end, y]
        points += [n*central for n in [0.1, 0.5, 1, 3, 8, 20, 40, 80]]
        points += [y+n*width for n in [-10, -6, -3, -1, 1, 3, 6, 10]]
        points = sorted(set(float(p) for p in points if 0 <= p <= end))
        jobs = [(1, points), (-1, points)]
        factor = -w*w/2

        def integrand(t, sign):
            nonlocal evaluations
            evaluations += 1
            if sign == 1:
                z = (1+1j*shape)*t
                dz = 1+1j*shape
                h = hankel1e(0, w*y*z)
            else:
                z = t+1j*shape*t*(t-y)/(t+y)
                dz = 1+1j*shape*(1-2*y*y/(t+y)**2)
                h = hankel2e(0, w*y*z)
            phase = (z*z+y*y)/2+sign*y*z
            return z*dz*h*np.exp(1j*w*phase)*(profile(z)-py)**power

    coefficients, coefficient_errors = [], []
    intervals = sum(len(pts)-1 for _, pts in jobs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for power in range(1, order+1):
            accum, error = 0.0j, 0.0
            power_factor = factor*(-1j*w)**(power-1)/factorial(power)
            for tag, pts in jobs:
                for left, right in zip(pts[:-1], pts[1:]):
                    target = tol/max(abs(power_factor), 1)/intervals
                    re, er = quad(lambda t: integrand(t, tag).real, left, right,
                                  epsabs=target, epsrel=tol/8, limit=250)
                    im, ei = quad(lambda t: integrand(t, tag).imag, left, right,
                                  epsabs=target, epsrel=tol/8, limit=250)
                    accum += re+1j*im
                    error += er+ei
            coefficients.append(complex(power_factor*accum))
            coefficient_errors.append(float(abs(power_factor)*error))
    terms = tuple(strength**n*c for n, c in enumerate(coefficients, 1))
    total_error = sum(abs(strength)**n*e for n, e in enumerate(coefficient_errors, 1))
    return BornResult(1+sum(terms), total_error,
                      perf_counter()-start, evaluations,
                      tuple(dict.fromkeys(str(c.message) for c in caught)),
                      "source", coefficients[0], coefficient_errors[0],
                      ("Numerical error excludes tail truncation and Born model error.",),
                      method=method, order=order, coefficients=tuple(coefficients),
                      coefficient_error_estimates=tuple(coefficient_errors), terms=terms)


def evaluate_full_born(w: float, y: float, kappa_s: float, *, phase="minimum",
                       tol=1e-10, shape=1.0, tail_scale=1.0,
                       method="split_hankel", order=1) -> BornResult:
    """Evaluate a complete potential-power Born truncation through order=1,2,3.

    ``source``: 1+sum(k**n*C_n), with potential k*(p(x)-p(y)).
    ``minimum`` (default): exactly transport that source-gauge truncation by
        exp[-i*w*(t_min+k*p(y))], using the parent's exact single-image minimum.
        This is a phase-resummed truncation, NOT a strict strength Taylor
        polynomial. It is the finite-strength phase-fair comparison baseline.
    ``minimum_linear``: legacy untransported first-order Taylor diagnostic;
        order=1 only. It is NOT a finite-strength minimum-gauge alignment.

    Never promote either a favorable or unfavorable untransported diagnostic
    into the main comparison. Exact transport is prescribed, never fitted.
    Finite Born order need not be uniform at high w; the stronger comparison
    to an order-2/3 candidate requires full Born2/3, not just Born1.

    w=0 returns the NFW continuous limit 1; w<0 uses conjugation. y>0 and
    k>=0 are required. Only exact minimum transport requires a single image.
    tol targets each C_n quadrature, not a guaranteed total F error.
    tail_scale>=1 lengthens the finite contour; no rigorous tail bound is
    claimed. shape changes split-Hankel contours (unused by real_tail).
    """
    start = perf_counter()
    if not np.all(np.isfinite([w, y, kappa_s, tol, shape, tail_scale])):
        raise ValueError("all numeric parameters must be finite real scalars")
    if y <= 0 or kappa_s < 0 or tol <= 0 or shape <= 0 or tail_scale < 1:
        raise ValueError("require y>0, kappa_s>=0, tol>0, shape>0, tail_scale>=1")
    if phase not in {"source", "minimum", "minimum_linear"}:
        raise ValueError("phase must be source, minimum, or minimum_linear")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= order <= 3:
        raise ValueError("order must be an integer from 1 through 3")
    if phase == "minimum_linear" and order != 1:
        raise ValueError("minimum_linear is a legacy first-order Taylor diagnostic only")
    if method not in {"split_hankel", "real_tail"}:
        raise ValueError("method must be split_hankel or real_tail")
    if kappa_s == 0:
        return BornResult(1.0+0.0j, 0.0, perf_counter()-start, 0, (), phase, method=method,
                          order=order, completion=f"phase_resummed_born{order}" if phase == "minimum" else f"source_born{order}")
    if w < 0:
        positive = evaluate_full_born(-w, y, kappa_s, phase=phase, tol=tol,
                                      shape=shape, tail_scale=tail_scale, method=method, order=order)
        return replace(positive, value=positive.value.conjugate(),
                       coefficient=positive.coefficient.conjugate(),
                       coefficients=tuple(c.conjugate() for c in positive.coefficients),
                       terms=tuple(c.conjugate() for c in positive.terms),
                       seconds=perf_counter()-start)
    from nfw_wave.lens import potential, single_image
    if w == 0:
        result = BornResult(1.0+0.0j, 0.0, perf_counter()-start, 0, (), phase,
                            coefficient=0.0j, method=method, order=order,
                            coefficients=(0.0j,)*order, coefficient_error_estimates=(0.0,)*order,
                            terms=(0.0j,)*order)
    else:
        result = _evaluate_profile_born(w, y, kappa_s, lambda z: potential(z, 1.0),
                                        tol=tol, shape=shape, tail_scale=tail_scale, method=method, order=order)
    shift = 0.0
    if phase == "minimum":
        image = single_image(y, kappa_s)
        shift = image.minimum_delay + potential(y, kappa_s)
    return replace(result, value=result.value*np.exp(-1j*w*shift),
                   completion=f"phase_resummed_born{order}" if phase == "minimum" else f"source_born{order}",
                   phase=phase, phase_shift=float(shift), seconds=perf_counter()-start,
                   model_warnings=result.model_warnings+(
                       "Strength expansion is at fixed w,y and is not uniform at high w.",
                       "minimum is a phase-resummed source truncation with the exact time shift; "
                       "minimum_linear is an untransported legacy first-order diagnostic only.",))


def evaluate_go_anchored_born(w: float, y: float, kappa_s: float, *,
                              tol=1e-10, shape=1.0, tail_scale=1.0,
                              method="split_hankel") -> BornResult:
    """Fixed GO completion A_exact + kappa_s*(B-K_source), in minimum gauge.

    K_source=Delta p(y)/2 is the first-strength GO amplitude coefficient.
    This no-switch, no-fit higher-order completion retains the entire Fresnel
    coefficient, but is NOT the paper's matched time-domain interpolation and
    is NOT a second Born term. It need not improve at low frequency.
    """
    start = perf_counter()
    result = evaluate_full_born(w, y, kappa_s, phase="minimum_linear", tol=tol,
                                shape=shape, tail_scale=tail_scale, method=method)
    if kappa_s == 0:
        return replace(result, phase="minimum", completion="go_anchored",
                       seconds=perf_counter()-start)
    if w == 0:
        raise ValueError("GO completion does not preserve the zero-frequency normalization")
    from nfw_wave.lens import single_image, radial_derivatives
    amplitude = single_image(y, kappa_s).amplitude
    d1, d2, _, _ = radial_derivatives(y, 1.0)
    local = (d2+d1/y)/2
    value = amplitude+kappa_s*(result.coefficient-local)
    return replace(result, value=complex(value), phase="minimum", completion="go_anchored",
                   seconds=perf_counter()-start,
                   model_warnings=result.model_warnings+(
                       "Fixed GO completion is not uniformly superior and lacks the exact low-w limit.",))
