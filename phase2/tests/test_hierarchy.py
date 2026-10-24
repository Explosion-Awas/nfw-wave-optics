"""Full-wave tests of the finite-q center hierarchy, not only coefficient checks."""
import numpy as np
import pytest


def test_second_finite_q_center_reduces_error_beyond_spatial_leading_term():
    from nfw_wave.reference import amplification_contour
    from nfw_wave.approximations import PreparedApproximation
    from nfw_wave.image_expansion import ImageExpansion
    from nfw_advanced.hierarchy import FiniteCenterNFW
    y, k, w = 80., .3, 1/80.
    old = PreparedApproximation(y,k)
    image = ImageExpansion(y,k)
    leading = image.evaluate(w)+old.center(w,"ft0")*np.exp(1j*w*old.center_delay)
    reference = amplification_contour(w,y,k,tol=2e-12,shape=.85)
    assert not reference.warnings
    new = FiniteCenterNFW(y,k).evaluate(w,order=2)
    assert abs(new-reference.value) < abs(leading-reference.value)/4
