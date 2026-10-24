"""Second-stage prepared NFW cut evaluator; domain validation is separate."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from nfw_wave.lens import single_image
from scipy.special import hankel2e, kve
from nfw_wave.approximations import nfw_fourier_kernel
from .cut_solver import continued_potential, _hermite, _rotated_nodes


@dataclass(frozen=True)
class BatchCutResult:
    value: object
    residual: object
    amplitude: float
    seconds: float


class PreparedCutNFW:
    def __init__(self,y,kappa_s,*,image_nodes=96,center_nodes=96):
        self.y,self.kappa_s = float(y),float(kappa_s)
        if not np.all(np.isfinite([self.y,self.kappa_s])) or self.y<=0 or self.kappa_s<0:
            raise ValueError("require finite y>0 and kappa_s>=0")
        if any(not isinstance(n,(int,np.integer)) or n<2 for n in [image_nodes,center_nodes]):
            raise ValueError("quadrature node counts must be integers >=2")
        self.image_nodes,self.center_nodes = image_nodes,center_nodes
        self.image = single_image(self.y,self.kappa_s)
        self.p0 = continued_potential(self.image.x,self.kappa_s).real
        self.t,self.hw = _hermite(image_nodes)
        self.u,self.cw = _rotated_nodes(center_nodes)
        self.rot = np.exp(-1j*np.pi/4)
        self.center_kernel = (self.cw*self.rot**2*self.u*kve(0,self.rot*self.u)
                              *np.exp((1-self.rot)*self.u))
        self.delay = self.y*self.y/2-self.image.minimum_delay

    def evaluate(self,w):
        """Evaluate nonzero signed frequencies, in bounded vector blocks.

        This is the same full cut/image integral as the scalar prototype.
        Fixed nodes and a prepared object do not constitute an error bound.
        The prospectively tested finite-q domain is documented separately.
        """
        start = perf_counter()
        frequencies = np.asarray(w,dtype=float)
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies==0):
            raise ValueError("require finite nonzero frequencies")
        flat = np.abs(frequencies.reshape(-1))
        residuals = np.empty(flat.shape,dtype=complex)
        im,k,y = self.image,self.kappa_s,self.y
        for lo in range(0,len(flat),512):
            frequency = flat[lo:lo+512,None]
            q = frequency*y
            dz = (1+1j)/np.sqrt(frequency*im.lambda_r)
            delta = dz*self.t
            z = im.x+delta
            r3 = continued_potential(z,k)-self.p0-(im.x-y)*delta-(1-im.lambda_r)*delta**2/2
            image_f = -.5j*frequency*dz*z*hankel2e(0,q*z)*np.exp(-1j*frequency*r3)
            dz0 = (1+1j)/np.sqrt(frequency)
            z0 = y+dz0*self.t
            free_f = -.5j*frequency*dz0*z0*hankel2e(0,q*z0)
            image_residual = np.sum(self.hw*(image_f-im.amplitude*free_f),axis=1)

            t = self.rot*self.u/q
            h = np.log1p(t*t/(2*(1+np.sqrt(1+t*t))))
            a,b = 2*k*h*(2*np.log(t/2)-h),2*np.pi*k*h
            x = frequency*b
            phase = -1j*frequency*(t*t/2+a)
            small = np.abs(x)<.1
            smx = np.empty_like(x)
            xx = x[small]**2
            smx[small] = x[small]*xx*(1/6+xx*(1/120+xx*(1/5040+xx/362880)))
            smx[~small] = np.sinh(x[~small])-x[~small]
            cut_residual = np.sinh(x)*np.expm1(phase)+smx
            center = (4*k*nfw_fourier_kernel(q[:,0])/y**2
                      +(2*frequency[:,0]/(np.pi*q[:,0]**2))
                      *np.sum(self.center_kernel*cut_residual,axis=1))
            residuals[lo:lo+512] = image_residual+np.exp(1j*frequency[:,0]*self.delay)*center
        residuals = np.where(frequencies.reshape(-1)<0,residuals.conjugate(),residuals)
        values = (self.image.amplitude+residuals).reshape(frequencies.shape)
        residuals = residuals.reshape(frequencies.shape)
        if frequencies.ndim == 0:
            values,residuals = values.item(),residuals.item()
        return BatchCutResult(values,residuals,self.image.amplitude,perf_counter()-start)
