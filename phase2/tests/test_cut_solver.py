"""Independent cut/Gaussian prototype checks, added one tracer at a time."""
import importlib

import mpmath as mp
import numpy as np
import pytest


def solver():
    try:
        return importlib.import_module("nfw_advanced.cut_solver")
    except ModuleNotFoundError:
        pytest.fail("cut_solver and its exact imaginary-potential kernel are missing")


def test_imaginary_potential_matches_original_mp_log_acos():
    ctx = mp.mp.clone()
    ctx.dps = 75
    for t in [1e-12, 1e-5, .1, 1., 3., 1000.]:
        for k in [.03, .7]:
            z, kk = ctx.j*ctx.mpf(t), ctx.mpf(k)
            expected = 2*kk*(ctx.log(z/2)**2+ctx.acos(1/z)**2)
            a, b = solver().imaginary_potential(t, k)
            assert abs(complex(a, b)-complex(expected)) < 2e-14*abs(complex(expected))
    a, b = solver().imaginary_potential(np.array([0., .1, 1.]), .3)
    assert a[0] == b[0] == 0


def test_center_leading_coefficient_matches_mp_sici():
    ctx = mp.mp.clone()
    ctx.dps = 60
    for q in [.3, 1., 3., 12.]:
        qq, k, y = ctx.mpf(q), ctx.mpf('.3'), ctx.mpf(80)
        expected = 4*k*(ctx.sin(qq)*(ctx.pi/2-ctx.si(qq))-ctx.cos(qq)*ctx.ci(qq))/y**2
        assert abs(solver().center_leading(q, 80., .3)-float(expected)) < 2e-17


def direct_center(q, y, k):
    """Adaptive K0 integral using original log/acos, not cut_solver's kernel."""
    from scipy.integrate import quad_vec
    from scipy.special import k0
    w = q/y
    def integrand(u):
        t = u/q
        z = 1j*t
        p = 2*k*(np.log(z/2)**2+np.arccos(1/z)**2)
        return (2*w/np.pi)*u*k0(u)/q**2*np.exp(-1j*w*(t*t/2+p.real))*np.sinh(w*p.imag)
    return sum(quad_vec(integrand, a, b, epsabs=2e-15, epsrel=2e-13)[0]
               for a, b in zip([0, .1, 1, 3, 8, 20, 45], [.1, 1, 3, 8, 20, 45, 100]))


def test_nonperturbative_center_matches_independent_complete_integral():
    for q, y, k in [(1., 80., .3), (3., 8., .7), (.3, 256., .03)]:
        expected = direct_center(q, y, k)
        actual = solver().center_cut(q/y, y, k, nodes=160)
        assert abs(actual-expected) < 3e-12
    assert solver().center_cut(.2, 20., 0., nodes=64) == 0j


def test_lower_half_continuation_matches_original_mp_across_imaginary_axis():
    ctx = mp.mp.clone()
    ctx.dps = 75
    points = [2+3j, 2-3j, .001-.002j, -.002-.003j, -2-3j, -1-.01j, -40-30j, -1j]
    for z in points:
        zz, k = ctx.mpc(z), ctx.mpf('.7')
        expected = 2*k*(ctx.log(zz/2)**2+ctx.acos(1/zz)**2)
        actual = solver().continued_potential(z, .7)
        assert abs(actual-complex(expected)) < 2e-14*max(1, abs(complex(expected)))


def test_raw_free_gaussian_quadrature_converges_without_normalization():
    errors = []
    for n in [16, 64, 256]:
        actual = solver().gaussian_image(.3/8, 8., 0., nodes=n, subtract_free=False)
        errors.append(abs(actual.value-1))
        assert actual.raw_free == actual.value
    assert errors[-1] < 3e-12
    assert errors[-1] < errors[0]/1000


def test_free_subtraction_preserves_exact_free_limit_and_nonzero_image():
    free = solver().gaussian_image(.3/8, 8., 0., nodes=16, subtract_free=True)
    assert free.value == 1
    assert free.residual == 0j
    assert free.raw_free != 1
    from nfw_advanced.reference_stable import amplification_stable
    from nfw_wave.lens import single_image
    for q, y, k in [(.3, 8., .7), (1., 80., .3)]:
        image = single_image(y, k)
        expected = amplification_stable(q/y, y, k, tol=1e-12).value
        expected -= np.exp(1j*q/y*(y*y/2-image.minimum_delay))*direct_center(q, y, k)
        actual = solver().gaussian_image(q/y, y, k, nodes=256, subtract_free=True)
        assert abs(actual.value-expected) < 5e-12
        assert abs(actual.residual-(expected-image.amplitude)) < 5e-12


def test_complete_field_and_weak_complex_residual_against_stable_reference():
    from nfw_advanced.reference_stable import amplification_stable
    for q, y, k in [(.3, 80., .7), (1., 20., .3), (3., 8., .7), (12., 256., .03)]:
        ref = amplification_stable(q/y, y, k, tol=1e-12, shape=.8)
        actual = solver().amplification_cut(q/y, y, k, image_nodes=96, center_nodes=160)
        assert abs(actual.value-ref.value) < 1e-11
        expected_residual = ref.value-actual.amplitude
        assert abs(actual.residual-expected_residual) < 1e-5*abs(expected_residual)+3e-14
        assert abs(actual.value-(actual.amplitude+actual.residual)) < 3e-16


def test_rotated_complete_cut_resolves_low_qy_without_more_nodes():
    q, y, k = .3, 8., .7
    expected = direct_center(q, y, k)
    actual = solver().center_cut(q/y, y, k, nodes=96, angle=-np.pi/4)
    assert abs(actual-expected) < 1e-10
    from nfw_advanced.reference_stable import amplification_stable
    reference = amplification_stable(q/y, y, k, tol=1e-12).value
    field = solver().amplification_cut(q/y, y, k, image_nodes=160, center_nodes=96, center_angle=-np.pi/4)
    assert abs(field.value-reference) < 1e-10


@pytest.mark.parametrize("w,y,k", [(0.,20.,.3),(-1.,20.,.3),(np.nan,20.,.3),(.1,0.,.3),(.1,20.,-.1)])
def test_solver_rejects_parameters_outside_positive_single_image_problem(w,y,k):
    for fn in [solver().gaussian_image, solver().center_cut, solver().amplification_cut]:
        with pytest.raises(ValueError):
            fn(w,y,k)


def test_default_field_preset_handles_the_low_qy_validation_corner():
    from nfw_advanced.reference_stable import amplification_stable
    actual=solver().amplification_cut(.3/8,8.,.7)
    reference=amplification_stable(.3/8,8.,.7,tol=1e-12).value
    assert abs(actual.value-reference)<1e-10


def test_weak_nonzero_residual_against_independent_mp_gamma_anchor():
    # experiments/cut_solver/weak_mp.json: finite Gamma+T, order80/dps50;
    # order50/dps40 field differs by 5.34e-29. Exact MP GO is subtracted.
    ctx=mp.mp.clone()
    ctx.dps=50
    expected=ctx.mpc('-1.20723501116268878247937818760280108e-8',
                     '2.34085317474416471068197666045453997e-9')
    actual=solver().amplification_cut(12/256,256.,.03)
    assert abs(ctx.mpc(actual.residual)-expected)/abs(expected)<1e-6
