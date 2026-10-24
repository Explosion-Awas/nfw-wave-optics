"""Independent arbitrary-precision checks of right-half-plane spatial kernels."""
import mpmath as mp
import numpy as np
import pytest


@pytest.mark.parametrize("z", [1e-6, 0.3, 1.0, 1.03, 0.01+0.015j, 0.7+0.2j, 1+0.03j, 5+2j])
def test_complex_nfw_fields_match_independent_potential_derivatives(z):
    from nfw_advanced.spatial import nfw_fields
    ctx = mp.mp.clone()
    ctx.dps = 70
    zz = ctx.mpc(complex(z).real, complex(z).imag)
    k = ctx.mpf("0.2")
    def p(x):
        return 2*k*(ctx.log(x/2)**2 + ctx.acos(1/x)**2)
    alpha = ctx.diff(p, zz)
    expected = [p(zz), alpha, (ctx.diff(p, zz, 2)+alpha/zz)/2]
    actual = nfw_fields(z, 0.2)
    for a, b in zip(actual, expected):
        assert complex(a) == pytest.approx(complex(b), rel=3e-12, abs=2e-15)


@pytest.mark.parametrize("q", [0.3, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("slope", [0.7, 1.0, 1.3])
def test_inverse_laplacian_matches_known_bessel_transform(q, slope):
    from scipy.special import k0
    from nfw_advanced.spatial import hankel_from_laplacian
    # f(r)=1/(1+r²), H[f]=K0(q); independent closed transform.
    def laplacian(z):
        return 4*(z*z-1)/(1+z*z)**3
    result = hankel_from_laplacian(q, laplacian, slope=slope)
    assert result.value == pytest.approx(k0(q), rel=2e-9, abs=2e-12)
    assert result.error_estimate >= 0
    assert not result.warnings


@pytest.mark.parametrize("q", [0.3, 1., 3., 10., 50.])
def test_first_nfw_power_transform_matches_independent_laplace_integral(q):
    from scipy.integrate import quad
    from nfw_advanced.spatial import nfw_power_hankel
    S = quad(lambda t: np.exp(-t)*t/(q*q+t*t), 0, np.inf, epsabs=1e-14)[0]
    expected = -4*S/q**2
    result = nfw_power_hankel(q, 1)
    assert result.value == pytest.approx(expected, rel=2e-9, abs=2e-13)
    assert not result.warnings


@pytest.mark.parametrize("power", [2, 3])
def test_nonlinear_power_source_matches_independent_mp_differentiation(power):
    from nfw_advanced.spatial import nfw_power_hankel, hankel_from_laplacian
    ctx = mp.mp.clone()
    ctx.dps = 45
    def lap(z):
        zz = ctx.mpc(z)
        def f(x):
            return (2*(ctx.log(x/2)**2+ctx.acos(1/x)**2))**power
        return complex(ctx.diff(f, zz, 2)+ctx.diff(f, zz)/zz)
    # Tests the nonlinear Poisson source independently, not a second
    # oscillatory algorithm; the K0 and n=1 tests cover the contour operator.
    expected = hankel_from_laplacian(1.3, lap).value
    actual = nfw_power_hankel(1.3, power)
    assert actual.value == pytest.approx(expected, rel=3e-9, abs=3e-11)
    assert not actual.warnings


@pytest.mark.parametrize("power", [2, 3])
def test_nfw_power_high_momentum_matches_independent_mellin_cusp(power):
    from math import comb
    from nfw_advanced.spatial import nfw_power_hankel
    ctx = mp.mp.clone()
    ctx.dps = 65
    q = ctx.mpf(3000)
    def mellin(a):
        return 2**(2*a+1)*ctx.gamma(a+1)*ctx.rgamma(-a)*q**(-2*a-2)
    expected = sum(comb(power,j)*ctx.log(2)**(power-j)*(-ctx.mpf("0.5"))**j
                   *ctx.diff(mellin, power, j) for j in range(power+1))
    actual = nfw_power_hankel(float(q), power, atol=1e-34)
    assert actual.value == pytest.approx(float(expected), rel=8e-6, abs=0)
    assert not actual.warnings


@pytest.mark.parametrize("q", [0.3, 1.0, 3.0, 12.0])
@pytest.mark.parametrize("laps", [0, 1, 2])
def test_linear_transform_laplacians_match_independent_mp_differentiation(q, laps):
    from nfw_advanced.spatial import nfw_linear_hankel
    ctx = mp.mp.clone()
    ctx.dps = 60
    def p(x):
        S = ctx.sin(x)*(ctx.pi/2-ctx.si(x))-ctx.cos(x)*ctx.ci(x)
        return -4*S/x**2
    def D(f):
        return lambda x: ctx.diff(f, x, 2)+ctx.diff(f, x)/x
    f = p
    for _ in range(laps):
        f = D(f)
    expected = float(f(ctx.mpf(q)))
    actual = nfw_linear_hankel(q, laps)
    assert actual == pytest.approx(expected, rel=3e-11, abs=1e-13)
