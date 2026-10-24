"""Experimental nonperturbative NFW cut and image-contour quadrature.

Quadrature node differences are empirical diagnostics, never rigorous bounds.
The first-stage solver and other second-stage implementations are unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.special import hankel2e, k0e, kve, roots_hermite, roots_laguerre, roots_legendre, sici
from nfw_wave.lens import single_image


@dataclass(frozen=True)
class ImageQuadrature:
    value: complex
    residual: complex
    amplitude: float
    raw_free: complex
    evaluations: int


@dataclass(frozen=True)
class CutResult:
    value: complex
    residual: complex
    amplitude: float
    image_residual: complex
    center: complex
    raw_free: complex
    seconds: float
    evaluations: int


def amplification_cut(w, y, kappa_s, *, image_nodes=96, center_nodes=96,
                      subtract_free=True, subtract_leading=True, center_angle=-np.pi/4):
    """Prototype F=I_full+exp(i*w*tau_c)*C; residual computed before adding A.

    Full-line closure uses the original log/acos lower-half-plane sheet.
    Defaults are an exploratory finite-q preset (96 GH + 96 cut nodes), not
    a tolerance-driven algorithm. A node count is not an accuracy certificate.
    This version uses direct binary64 potential differences for R3 and is
    NOT validated for ultra-high-frequency/near-roundoff diffraction residuals.
    """
    start = perf_counter()
    image = gaussian_image(w, y, kappa_s, nodes=image_nodes, subtract_free=subtract_free)
    center = center_cut(w, y, kappa_s, nodes=center_nodes, subtract_leading=subtract_leading, angle=center_angle)
    im = single_image(y, kappa_s)
    carrier = np.exp(1j*w*(y*y/2-im.minimum_delay))
    residual = image.residual+carrier*center
    return CutResult(image.amplitude+residual, residual, image.amplitude,
                     image.residual, center, image.raw_free, perf_counter()-start,
                     image.evaluations+center_nodes)


@lru_cache(maxsize=16)
def _hermite(nodes):
    return roots_hermite(nodes)


def gaussian_image(w, y, kappa_s, *, nodes=64, subtract_free=False):
    """Full-line Gaussian contour on the explicitly continued lower sheet.

    raw_free is independently integrated, never normalized to one.
    """
    if not np.all(np.isfinite([w,y,kappa_s])) or w <= 0 or y <= 0 or kappa_s < 0:
        raise ValueError("require finite w,y>0 and kappa_s>=0")
    im = single_image(y, kappa_s)
    t, weights = _hermite(nodes)
    q = w*y
    scale = np.sqrt(w*im.lambda_r)
    dz = (1+1j)/scale
    delta = dz*t
    z = im.x+delta
    p0 = continued_potential(im.x, kappa_s).real
    r3 = continued_potential(z, kappa_s)-p0-(im.x-y)*delta-(1-im.lambda_r)*delta**2/2
    f = -.5j*w*dz*z*hankel2e(0, q*z)*np.exp(-1j*w*r3)
    dz0 = (1+1j)/np.sqrt(w)
    z0 = y+dz0*t
    f0 = -.5j*w*dz0*z0*hankel2e(0, q*z0)
    raw_free = complex(np.dot(weights, f0))
    if subtract_free:
        residual = complex(np.dot(weights, f-im.amplitude*f0))
        return ImageQuadrature(im.amplitude+residual, residual, im.amplitude, raw_free, 2*nodes)
    value = complex(np.dot(weights, f))
    return ImageQuadrature(value, value-im.amplitude, im.amplitude, raw_free, 2*nodes)


@lru_cache(maxsize=16)
def _laguerre(nodes):
    return roots_laguerre(nodes)


@lru_cache(maxsize=16)
def _rotated_nodes(nodes):
    # Resolve the branch scale near u=0 separately from the damped tail.
    nnear = 2*nodes//3
    x, v = roots_legendre(nnear)
    far, wf = roots_laguerre(nodes-nnear)
    near = 2*(1+x)
    return np.r_[near, 4+far], np.r_[2*v*np.exp(-near), np.exp(-4)*wf]


def center_cut(w, y, kappa_s, *, nodes=160, subtract_leading=True, angle=0.):
    """Complete nonperturbative cut by Laguerre, optionally subtracting C1.

    No accuracy guarantee follows from a fixed node count. The subtraction
    integrates exp(-i*w*d)*sinh(w*b)-w*b without losing its small remainder.
    angle=0 uses ordinary Laguerre on u=q*t. For -pi/4<=angle<0, analytic
    a(t),b(t) (no longer real/imaginary parts) are integrated on a downward
    ray with 2*nodes//3 Legendre nodes on [0,4] and shifted Laguerre thereafter.
    """
    q = w*y
    if not np.all(np.isfinite([w,y,kappa_s])) or w <= 0 or y <= 0 or kappa_s < 0:
        raise ValueError("w,y finite positive and kappa_s finite nonnegative are required")
    if kappa_s == 0:
        return 0j
    u, weights = _laguerre(nodes)
    if angle == 0:
        t = u/q
        a, b = imaginary_potential(t, kappa_s)
        kernel = u*k0e(u)
    else:
        if not -np.pi/4 <= angle <= 0:
            raise ValueError("cut rotation must be in [-pi/4,0]")
        u, weights = _rotated_nodes(nodes)
        rot = np.exp(1j*angle)
        t = rot*u/q
        h = np.log1p(t*t/(2*(1+np.sqrt(1+t*t))))
        a, b = 2*kappa_s*h*(2*np.log(t/2)-h), 2*np.pi*kappa_s*h
        kernel = rot**2*u*kve(0, rot*u)*np.exp((1-rot)*u)
    x = w*b
    phase = -1j*w*(t*t/2+a)
    if subtract_leading:
        small = np.abs(x) < .1
        smx = np.empty_like(x)
        xx = x[small]**2
        smx[small] = x[small]*xx*(1/6+xx*(1/120+xx*(1/5040+xx/362880)))
        smx[~small] = np.sinh(x[~small])-x[~small]
        residual = np.sinh(x)*np.expm1(phase)+smx
        return complex(center_leading(q, y, kappa_s)+(2*w/(np.pi*q*q))*np.dot(weights, kernel*residual))
    return complex((2*w/(np.pi*q*q))*np.dot(weights, kernel*np.exp(phase)*np.sinh(x)))


def center_leading(q, y, kappa_s):
    """Exact first cut coefficient 4*kappa_s*S(q)/y**2 (q=w*y)."""
    si, ci = sici(q)
    return 4*kappa_s*(np.sin(q)*(np.pi/2-si)-np.cos(q)*ci)/y**2


def continued_potential(z, kappa_s):
    """NFW original log/acos sheet in Re(z)>=0 or Im(z)<0.

    The lower-left quadrant is evaluated explicitly, not passed through the
    first-stage potential's right-half-plane contract. The small-z identity
    analytically continues from positive real z through the lower half-plane.
    """
    a = np.asarray(z, dtype=complex)
    if np.any((a.real < 0) & (a.imag >= 0)):
        raise ValueError("upper-left/negative-real branch is outside this prototype")
    flat = a.reshape(-1)
    out = np.zeros_like(flat)
    small = (np.abs(flat) < .07) & (flat != 0)
    zz = flat[small]
    h = np.log1p(-zz*zz/(2*(1+np.sqrt(1-zz*zz))))
    out[small] = -2*kappa_s*h*(2*np.log(2/zz)+h)
    zz = flat[np.abs(flat) >= .07]
    out[np.abs(flat) >= .07] = 2*kappa_s*(np.log(zz/2)**2+np.arccos(1/zz)**2)
    out = out.reshape(a.shape)
    return out.item() if out.ndim == 0 else out


def imaginary_potential(t, kappa_s):
    """Stable exact boundary value psi(+i*t)=a+i*b, including t=0."""
    t = np.asarray(t, dtype=float)
    if np.any(t < 0) or kappa_s < 0:
        raise ValueError("t and kappa_s must be nonnegative")
    h = np.log1p(t*t/(2*(1+np.hypot(1, t))))
    L = np.log(np.where(t > 0, t, 2)/2)
    return 2*kappa_s*h*(2*L-h), 2*np.pi*kappa_s*h
