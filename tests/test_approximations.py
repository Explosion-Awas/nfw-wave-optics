import numpy as np
import pytest
from scipy.integrate import quad


def test_nfw_fourier_kernel_matches_nonoscillatory_laplace_integral():
    from nfw_wave.approximations import nfw_fourier_kernel
    for q in [1e-3,0.1,1.0,10.0,40.0,1000.0]:
        exact,err = quad(lambda t: np.exp(-t)*t/(q*q+t*t),0,80,epsabs=1e-14,epsrel=2e-13,limit=400)
        assert np.isclose(nfw_fourier_kernel(q),exact,rtol=1e-11,atol=2e-16)


def test_prepared_asymptotics_have_free_limit_conjugacy_and_correct_high_frequency_term():
    from nfw_wave.approximations import PreparedApproximation
    from nfw_wave.reference import amplification_contour
    free = PreparedApproximation(2.0,0.0)
    for mode in ['go','image1','cusp0','cusp1','ft0','ft1']:
        assert abs(free.evaluate(10.,mode)-1) < 1e-14
    p = PreparedApproximation(1.0,0.1)
    w = 2000.
    numerical = amplification_contour(w,1.,.1,tol=3e-11).value
    corrected = p.evaluate(w,'cusp1')
    assert abs(corrected-numerical) < abs(p.image.amplitude-numerical)/100
    assert p.evaluate(-w,'ft0') == np.conj(p.evaluate(w,'ft0'))
    assert np.isclose(p.center(w,'ft0')/p.center(w,'cusp0'),1,rtol=1e-5)
    assert abs(p.center(w,'ft1')-p.center(w,'cusp1')) < 1e-12
