"""Asymptotic candidates; numerical validity is determined by experiments."""
from __future__ import annotations
import numpy as np
from scipy.special import sici


def nfw_fourier_kernel(q):
    """S(q)=sin(q)*(pi/2-Si(q))-cos(q)*Ci(q), stably for q>0.

    Also S(q)=integral_0^infinity exp(-t)*t/(q*q+t*t) dt.
    The asymptotic branch avoids catastrophic trigonometric cancellation.
    """
    x = np.asarray(q,dtype=float)
    if np.any(x<=0):
        raise ValueError('q must be positive')
    z = x.reshape(-1)
    out = np.empty_like(z)
    direct = z<=50
    si,ci = sici(z[direct])
    out[direct] = np.sin(z[direct])*(np.pi/2-si)-np.cos(z[direct])*ci
    for i in np.where(~direct)[0]:
        u = z[i]
        term = 1/u**2
        total = term
        for n in range(1,50):
            nxt = -term*(2*n)*(2*n+1)/u**2
            if abs(nxt) >= abs(term):
                break
            total += nxt
            term = nxt
            if abs(term)<abs(total)*2e-16:
                break
        out[i] = total
    out = out.reshape(x.shape)
    return out.item() if out.ndim==0 else out


from .lens import single_image,radial_derivatives


class PreparedApproximation:
    """Accurate GO prepared once for a frequency band.

    All center quantities here exclude the common central phase.
    'cusp0/1': known high-q center tail and its next logarithmic order.
    'ft0': exact NFW spatial-transform Born center, with phase omitted inside.
    'ft1': experimental mixed-order correction (exact radial, asymptotic psi²).
    These are candidates, not an automatic accuracy promise.
    """
    def __init__(self,y,kappa_s):
        self.y,self.kappa_s = float(y),float(kappa_s)
        self.image = single_image(y,kappa_s)
        r = self.image.x
        a,b = self.image.lambda_r,self.image.lambda_t
        _,_,d3,d4 = radial_derivatives(r,kappa_s)
        self.delta = (d4/(8*a*a)+5*d3*d3/(24*a**3)+d3/(4*r*a*a)
                      +(a-b)/(8*r*r*a*b))
        self.center_delay = y*y/2-self.image.minimum_delay

    def center(self,w,mode='ft0'):
        u = np.asarray(w,dtype=float)
        if np.any(u<=0):
            raise ValueError('center requires positive w')
        k,y = self.kappa_s,self.y
        q = u*y
        if mode in ['go','image1']:
            return np.zeros_like(u,dtype=complex)
        if mode in ['cusp0','cusp1']:
            c = 4*k/(u*u*y**4)+0j
        elif mode in ['ft0','ft1']:
            c = 4*k*nfw_fourier_kernel(q)/y**2+0j
        else:
            raise ValueError(f'unknown approximation {mode}')
        if mode == 'cusp1':
            c = c+1j*(-32*k+64*k*k*(np.log(q)+np.euler_gamma-1.5))/(u**3*y**6)
        if mode == 'ft1':
            flat = q.reshape(-1)
            dp = np.empty_like(flat)
            for j,z in enumerate(flat):
                if z<=50:
                    si,ci = sici(z)
                    S = nfw_fourier_kernel(z)
                    Sp = np.cos(z)*(np.pi/2-si)+np.sin(z)*ci-1/z
                    lap = (-S+1/z**2)/z**2-3*Sp/z**3+4*S/z**4
                else:
                    term = 1/z**4
                    lap = 16*term/z**2
                    for n in range(1,40):
                        term *= -(2*n)*(2*n+1)/z**2
                        add = (4+2*n)**2*term/z**2
                        lap += add
                        if abs(add)<abs(lap)*2e-16:
                            break
                dp[j] = -4*k*lap
            Q = 128*k*k/q**6*(np.log(q)+np.euler_gamma-1.5)
            c = c+0.5j*u**3*(dp.reshape(q.shape)+Q)
        c = np.asarray(c)
        return c.item() if c.ndim==0 else c

    def evaluate(self,w,mode='ft0'):
        a = np.asarray(w,dtype=float)
        if np.any(a==0):
            raise ValueError('asymptotics are not defined at zero frequency')
        u = abs(a)
        value = np.full(a.shape,self.image.amplitude,dtype=complex)
        if mode != 'go':
            value += 1j*self.image.amplitude*self.delta/u
        value += self.center(u,mode)*np.exp(1j*u*self.center_delay)
        value = np.where(a<0,np.conj(value),value)
        return value.item() if value.ndim==0 else value
