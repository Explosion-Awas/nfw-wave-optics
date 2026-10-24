"""Local post-GO expansion of the unique NFW minimum image.

With exp(+i*w*T) and T_min=0, F_image = A*(1+i*c1/w+c2/w**2).
This is the smooth image term, NOT the central cusp contribution or the full
wave-optics amplification. See reports/image_expansion.md for the derivation.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from .lens import Image, radial_derivatives, single_image


def _higher_radial_derivatives(r: float, kappa_s: float) -> tuple[float, float]:
    """Return psi5 and psi6; preparation only, no global precision changes."""
    ctx = mp.mp.clone()
    ctx.dps = 60
    rr, kk = ctx.mpf(r), ctx.mpf(kappa_s)

    def psi(z):
        return 2 * kk * (ctx.log(z / 2)**2 + ctx.acos(1 / z)**2)

    return tuple(float(ctx.re(ctx.diff(psi, rr, order))) for order in (5, 6))


def _coefficients(image: Image, kappa_s: float) -> tuple[float, float, float]:
    if kappa_s == 0:
        return image.amplitude, 0.0, 0.0

    r = image.x
    a, b = image.lambda_r, image.lambda_t
    _, _, d3, d4 = radial_derivatives(r, kappa_s)
    c1 = (d4 / (8 * a * a) + 5 * d3 * d3 / (24 * a**3)
          + d3 / (4 * r * a * a) + (a - b) / (8 * r * r * a * b))
    # Preparation only: arbitrary-precision differentiation independently
    # supplies psi5 and psi6, including the removable singularity at r=1.
    d5, d6 = _higher_radial_derivatives(r, kappa_s)
    c2 = (
        -3 * (a - b) * (3 * a + 5 * b) / (128 * r**4 * a**2 * b**2)
        + d3 * (a - 5 * b) / (32 * r**3 * a**3 * b)
        + d4 * (5 * b - a) / (64 * r**2 * a**3 * b)
        - d5 / (16 * r * a**3) - d6 / (48 * a**3)
        + 5 * d3**2 * (7 * b - a) / (192 * r**2 * a**4 * b)
        - 35 * d3 * d4 / (96 * r * a**4)
        - 7 * d3 * d5 / (48 * a**4) - 35 * d4**2 / (384 * a**4)
        - 35 * d3**3 / (96 * r * a**5)
        - 35 * d3**2 * d4 / (64 * a**5)
        - 385 * d3**4 / (1152 * a**6)
    )
    return image.amplitude, float(c1), float(c2)


class ImageExpansion:
    """Prepare the local image coefficients once, evaluate a frequency band.

    ``coefficients`` is the tuple ``(A, c1, c2)``. The real ``c2`` is the
    coefficient itself, not minus a named second-order delay correction.
    The inherited ``single_image`` check rejects the multiple-image region.
    No uniform approximation near the NFW cusp/critical curves is claimed.
    """

    def __init__(self, y: float, kappa_s: float):
        self.y, self.kappa_s = float(y), float(kappa_s)
        if not np.isfinite(self.y) or not np.isfinite(self.kappa_s):
            raise ValueError("image expansion requires finite y and kappa_s")
        self.image = single_image(self.y, self.kappa_s)
        self.amplitude, self.c1, self.c2 = _coefficients(self.image, self.kappa_s)

    @property
    def coefficients(self) -> tuple[float, float, float]:
        """Return ``(A, c1, c2)`` without repeating image finding/differentiation."""
        return self.amplitude, self.c1, self.c2

    def evaluate(self, w):
        """Image-only expansion; real nonzero frequencies, scalar or array.

        Signed frequencies satisfy F(-w)=conj(F(w)). There is no central
        phase in this local term, because the minimum delay is zero.
        """
        u = np.asarray(w, dtype=float)
        if np.any(~np.isfinite(u)) or np.any(u == 0):
            raise ValueError("image expansion requires finite, nonzero frequencies")
        value = self.amplitude * (1 + 1j * self.c1 / u + self.c2 / u**2)
        return value.item() if value.ndim == 0 else value


def expansion_coefficients(y: float, kappa_s: float) -> tuple[float, float, float]:
    """Return ``(A, c1, c2)`` in ``A*(1+1j*c1/w+c2/w**2)``.

    For repeated frequency evaluations prefer :class:`ImageExpansion`.
    Preparation uses 60-decimal-digit differentiation only for psi5 and psi6;
    evaluations use ordinary NumPy arithmetic, not numerical quadrature.
    """
    return ImageExpansion(y, kappa_s).coefficients
