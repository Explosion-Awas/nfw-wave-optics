"""Standard NFW potential, scaled by r_s; psi0(GLoW) = 4*kappa_s."""
from __future__ import annotations
import numpy as np


def potential(x, kappa_s: float):
    """Analytic NFW potential with psi(0)=0, including complex Re(x)>0.

    Near the logarithmic cusp use the convergent small-x expansion to
    avoid subtracting two squared logarithms. The x=1 singularity is removable.
    """
    a = np.asarray(x, dtype=np.complex128)
    z = a.reshape(-1)
    out = np.zeros(z.shape, dtype=np.complex128)
    small = (abs(z) < 0.07) & (z != 0)
    zz = z[small]
    if zz.size:
        L = np.log(2/zz)
        t = zz*zz
        out[small] = kappa_s*t*(L + t*((3*L-1)/8 + t*((20*L-9)/96 + t*((210*L-107)/1536 + t*((504*L-275)/5120 + t*((27720*L-15797)/368640))))))
    normal = (abs(z) >= 0.07)
    zz = z[normal]
    if zz.size:
        out[normal] = 2*kappa_s*(np.log(zz/2)**2 + np.arccos(1/zz)**2)
    out = out.reshape(a.shape)
    if np.isrealobj(x):
        out = out.real
    return out.item() if out.ndim == 0 else out


_SMALL = [(2,1.0,0.0), (4,3/8,-1/8), (6,20/96,-9/96),
          (8,210/1536,-107/1536), (10,504/5120,-275/5120),
          (12,27720/368640,-15797/368640)]
_F1 = np.array([1,-2/3,7/15,-12/35,83/315,-146/693,523/3003,
                -952/6435,14051/109395,-26206/230945,
                32867/323323,-62260/676039,1423159/16900975])


def radial_derivatives(x: float, kappa_s: float):
    """Return psi'(x), psi''(x), psi'''(x), psi''''(x) for x>0."""
    if x <= 0 or kappa_s < 0:
        raise ValueError('x must be positive and kappa_s nonnegative')
    if kappa_s == 0:
        return (0.0,)*4
    if x < 0.07:
        L = np.log(2/x)
        out = []
        for order in range(1,5):
            v = 0.0
            for power,A,B in _SMALL:
                n = power
                for _ in range(order):
                    A,B = n*A,n*B-A
                    n -= 1
                v += x**n*(A*L+B)
            out.append(kappa_s*v)
        return tuple(out)
    if abs(x-1) < 0.08:
        coeff = _F1.copy()
        ff = []
        for n in range(4):
            ff.append(np.polynomial.polynomial.polyval(x-1, coeff))
            coeff = np.polynomial.polynomial.polyder(coeff)
        f,f1,f2,f3 = ff
    else:
        D = x*x-1
        f = np.arccos(1/x)/np.sqrt(D) if x>1 else np.arccosh(1/x)/np.sqrt(-D)
        f1 = (1/x-x*f)/D
        f2 = (-1/x**2-f-3*x*f1)/D
        f3 = (2/x**3-4*f1-5*x*f2)/D
    g = np.log(x/2)+f
    g1,g2,g3 = 1/x+f1, -1/x**2+f2, 2/x**3+f3
    return (4*kappa_s*g/x,
            4*kappa_s*(g1/x-g/x**2),
            4*kappa_s*(g2/x-2*g1/x**2+2*g/x**3),
            4*kappa_s*(g3/x-3*g2/x**2+6*g1/x**3-6*g/x**4))


from dataclasses import dataclass
from scipy.optimize import brentq


@dataclass(frozen=True)
class Image:
    x: float
    amplitude: float
    minimum_delay: float
    lambda_r: float
    lambda_t: float
    caustic_y: float


def single_image(y: float, kappa_s: float) -> Image:
    """Accurate positive minimum, rejecting the multiple-image source region."""
    if y <= 0 or kappa_s < 0:
        raise ValueError('single-image calculation requires y>0 and kappa_s>=0')
    if kappa_s == 0:
        return Image(y,1.0,0.0,1.0,1.0,0.0)
    lo = -350.0
    if radial_derivatives(np.exp(lo),kappa_s)[1] > 1:
        hi = 1.0
        while radial_derivatives(np.exp(hi),kappa_s)[1] > 1:
            hi += 1
        rc = np.exp(brentq(lambda logx: radial_derivatives(np.exp(logx),kappa_s)[1]-1,lo,hi))
        yc = radial_derivatives(rc,kappa_s)[0]-rc
    else:
        yc = 0.0
    if y <= yc:
        raise ValueError(f'not in single-image region: y={y}, caustic_y={yc}')
    hi = max(y+1,2.0)
    f = lambda x: x-radial_derivatives(x,kappa_s)[0]-y
    while f(hi) <= 0:
        hi *= 2
    xm = brentq(f,y,hi,xtol=2e-14,rtol=9e-16)
    d1,d2,_,_ = radial_derivatives(xm,kappa_s)
    lr,lt = 1-d2, y/xm
    return Image(xm,float(1/np.sqrt(lr*lt)),
                 float((xm-y)**2/2-potential(xm,kappa_s)),float(lr),float(lt),float(yc))
