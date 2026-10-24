"""Spatial transforms for the second-stage, finite-momentum NFW study.

H[f](q) = integral_0^infinity r*f(r)*J0(q*r) dr, q>0.  These are
spatial transforms, not full Fresnel propagators or full-wave error bounds.
"""
from __future__ import annotations

import numpy as np
from nfw_wave.lens import _SMALL, potential


def nfw_fields(z, kappa_s: float = 1.0):
    """Return (psi, alpha, convergence), analytically continued to Re(z)>0."""
    a = np.asarray(z, dtype=complex)
    x = a.reshape(-1)
    alpha = np.empty_like(x)
    convergence = np.empty_like(x)
    small = np.abs(x) < 0.07
    zz = x[small]
    if zz.size:
        L = np.log(2/zz)
        aa = np.zeros_like(zz)
        cc = np.zeros_like(zz)
        for n, A, B in _SMALL:
            aa += zz**(n-1)*(n*A*L+n*B-A)
            cc += zz**(n-2)*(n*n*A*L+n*n*B-2*n*A)/2
        alpha[small], convergence[small] = kappa_s*aa, kappa_s*cc
    regular = ~small
    zz = x[regular]
    if zz.size:
        D = zz*zz-1
        near = np.abs(D) < 0.15
        f = np.empty_like(D)
        h = np.empty_like(D)
        dd = D[near]
        if dd.size:
            f[near] = np.polynomial.polynomial.polyval(dd, [(-1)**j/(2*j+1) for j in range(22)])
            h[near] = np.polynomial.polynomial.polyval(dd, [(-1)**j/(2*j+3) for j in range(22)])
        dd = D[~near]
        if dd.size:
            root = np.sqrt(dd)
            f[~near] = np.arctan(root)/root
            h[~near] = (1-f[~near])/dd
        alpha[regular] = 4*kappa_s*(np.log(zz/2)+f)/zz
        convergence[regular] = 2*kappa_s*h
    answer = (np.asarray(potential(a, kappa_s)), alpha.reshape(a.shape), convergence.reshape(a.shape))
    return tuple(v.item() if v.ndim == 0 else v for v in answer)


from dataclasses import dataclass
from time import perf_counter
from scipy.integrate import quad
from scipy.special import hankel1e


@dataclass(frozen=True)
class SpatialResult:
    value: float
    error_estimate: float
    seconds: float
    evaluations: int
    warnings: tuple[str, ...]


def hankel_from_laplacian(q, laplacian, *, slope=1.0, atol=1e-12, rtol=1e-10):
    """Compute H[f]=-H[Delta f]/q² using a rotated Hankel contour.

    This requires analyticity of Delta f in the swept sector and no
    origin flux/contact term. The quadrature error excludes contour/tail
    assumptions; it is not a rigorous error bound on the transform.
    """
    started = perf_counter()
    q = float(q)
    d = 1/slope+1j  # q*z=d*t, exponential damping exp(-t).
    def integrand(t):
        z = d*t/q
        return float(-np.real(d*d*t*laplacian(z)*hankel1e(0, d*t)
                              *np.exp((-1+1j/slope)*t))/q**4)
    feature = q/abs(d)
    edges = sorted(set([0.0, 1., 3., 8., 20., 40., 60.]
                       + [feature*b for b in (.05, .2, .5, 1., 2., 4., 8.)
                          if feature*b < 60.]))
    total = error = 0.0
    neval = 0
    messages = []
    for a, b in zip(edges[:-1], edges[1:]):
        result = quad(integrand, a, b, epsabs=atol/(len(edges)-1), epsrel=rtol/4,
                      limit=300, full_output=1)
        total += result[0]
        error += result[1]
        neval += result[2]["neval"]
        if len(result) > 3:
            messages.append(str(result[3]))
    return SpatialResult(total, error, perf_counter()-started, neval, tuple(messages))


def nfw_power_hankel(q, power=1, **quad_options):
    """H[p**power](q), p=psi(kappa_s=1), excluding q=0 contact terms."""
    def laplacian(z):
        p, a, c = nfw_fields(z)
        source = 2*power*c*p**(power-1)
        if power > 1:
            source += power*(power-1)*p**(power-2)*a*a
        return source
    return hankel_from_laplacian(q, laplacian, **quad_options)


def nfw_linear_hankel(q, laps=0):
    """D_q**laps H[p](q), laps=0,1,2, from nonoscillatory Laplace kernels."""
    q = float(q)
    def integrand(t):
        a, b = q*q, t*t
        if laps == 0:
            rational = -4/(a*(a+b))
        elif laps == 1:
            rational = -16*(4*a*a+3*a*b+b*b)/(a*a*(a+b)**3)
        elif laps == 2:
            rational = -256*(9*a**4+9*a**3*b+10*a*a*b*b+5*a*b**3+b**4)/(a**3*(a+b)**5)
        else:
            raise ValueError("supported linear Laplacians are 0,1,2")
        return np.exp(-t)*t*rational
    # Integrate a naturally normalized quantity to retain relative precision
    # over many decades of q and derivative order.
    scale = q**(4+2*laps)/(4*[1,16,576][laps])
    result = quad(lambda t: scale*integrand(t), 0, np.inf,
                  epsabs=1e-13, epsrel=2e-13, limit=300)
    return result[0]/scale
