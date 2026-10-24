"""Independent normalization, gauge and quadrature checks for full Born."""
from importlib import import_module, util

import numpy as np
import pytest


def test_quadratic_profile_has_gaussian_exact_first_coefficient_one():
    # Completing the two-dimensional Gaussian gives minimum-gauge
    # F_exact(k) = 1/(1-k), independently of w and y.
    for w, y in [(0.3, 0.7), (2.0, 1.4), (11.0, 3.0)]:
        result = born_module()._evaluate_profile_born(
            w, y, 0.02, lambda z: z*z/2, tol=1e-10
        )
        assert result.coefficient == pytest.approx(1.0, abs=2e-8)
        assert result.value == pytest.approx(1.02, abs=1e-9)
        assert result.coefficient_error_estimate < 1e-7
        assert result.evaluations > 0
        assert not result.warnings


def test_constant_potential_gauge_invariance():
    baseline = born_module()._evaluate_profile_born(1.7, 1.3, 0.04, lambda z: z*z/2)
    shifted = born_module()._evaluate_profile_born(1.7, 1.3, 0.04, lambda z: z*z/2+17.0)
    constant = born_module()._evaluate_profile_born(1.7, 1.3, 0.04, lambda z: 17.0)
    assert shifted.coefficient == pytest.approx(baseline.coefficient, abs=2e-10)
    assert constant.coefficient == 0j
    assert constant.value == 1+0j


def test_nfw_coefficient_matches_independent_small_strength_derivative():
    from nfw_wave.reference import amplification_real_tail
    w, y, h = 2.2, 1.3, 2e-4
    result = born_module().evaluate_full_born(w, y, h, phase="source", tol=2e-11)
    f_h = amplification_real_tail(w, y, h, tol=2e-12)
    f_half = amplification_real_tail(w, y, h/2, tol=2e-12)
    derivative = (4*f_half.value-f_h.value-3)/h
    print("NFW independent weak-strength derivative:", derivative,
          "full Born B:", result.coefficient,
          "difference:", abs(derivative-result.coefficient))
    assert not f_h.warnings and not f_half.warnings and not result.warnings
    assert result.coefficient == pytest.approx(derivative, abs=2e-6)
    assert result.value == pytest.approx(1+h*result.coefficient, abs=1e-15)


def test_exact_minimum_gauge_is_explicit_time_transport_not_a_phase_penalty():
    from nfw_wave.lens import potential, single_image
    from nfw_wave.reference import amplification_real_tail
    w, y, k = 7.0, 1.2, 0.12
    source = born_module().evaluate_full_born(w, y, k, phase="source")
    minimum = born_module().evaluate_full_born(w, y, k, phase="minimum")
    linear = born_module().evaluate_full_born(w, y, k, phase="minimum_linear")
    image = single_image(y, k)
    shift = image.minimum_delay + potential(y, k)
    phase = np.exp(-1j*w*shift)
    exact_minimum = amplification_real_tail(w, y, k, tol=2e-11)
    exact_source = exact_minimum.value/phase
    assert minimum.phase_shift == pytest.approx(shift, abs=1e-15)
    assert minimum.value == pytest.approx(source.value*phase, abs=2e-13)
    assert linear.value == pytest.approx(source.value, abs=2e-13)
    assert minimum.phase == "minimum" and linear.phase == "minimum_linear"
    assert abs(minimum.value-exact_minimum.value) == pytest.approx(
        abs(source.value-exact_source), abs=2e-13
    )
    assert minimum.model_error_estimate is None
    assert minimum.model_warnings


@pytest.mark.parametrize("phase", ["source", "minimum", "minimum_linear"])
def test_negative_frequency_conjugation_and_zero_frequency_limit(phase):
    positive = born_module().evaluate_full_born(3.1, 1.4, 0.06, phase=phase)
    negative = born_module().evaluate_full_born(-3.1, 1.4, 0.06, phase=phase)
    zero = born_module().evaluate_full_born(0.0, 1.4, 0.06, phase=phase)
    assert negative.value == pytest.approx(positive.value.conjugate(), abs=1e-14)
    assert negative.coefficient == pytest.approx(positive.coefficient.conjugate(), abs=1e-14)
    assert negative.phase_shift == positive.phase_shift
    assert negative.error_estimate == positive.error_estimate
    assert zero.value == 1+0j and zero.evaluations == 0
    assert not negative.warnings and not zero.warnings


@pytest.mark.parametrize("w,y", [(0.07, 6.0), (1.3, 1.0), (12.0, 3.0), (60.0, 5.0)])
def test_second_unsplit_bessel_contour_agrees_with_split_hankel(w, y):
    split = born_module().evaluate_full_born(w, y, 0.08, phase="source", tol=2e-11)
    changed = born_module().evaluate_full_born(
        w, y, 0.08, phase="source", tol=8e-12, shape=0.55, tail_scale=1.2
    )
    unsplit = born_module().evaluate_full_born(
        w, y, 0.08, phase="source", tol=8e-12, method="real_tail", tail_scale=1.2
    )
    delta = max(abs(split.coefficient-unsplit.coefficient),
                abs(changed.coefficient-unsplit.coefficient))
    print("independent contours:", w, y, "B=", split.coefficient, "difference=", delta)
    assert delta < 3e-8
    assert unsplit.method == "real_tail"
    assert not split.warnings and not changed.warnings
    # Unsplit high-frequency real-axis sums reach binary64 roundoff before
    # this deliberately strict tolerance. Preserve, inspect and report it;
    # independent measured agreement, not silence, is the acceptance test.
    if unsplit.warnings:
        assert w == 60.0
        assert all("roundoff" in message for message in unsplit.warnings)
    assert unsplit.coefficient_error_estimate < 3e-8


@pytest.mark.parametrize("w,y,a", [(0.4, 0.6, 0.2), (3.1, 1.5, 0.4), (17.0, 2.8, 0.1)])
def test_finite_fresnel_phase_matches_an_independent_gaussian_convolution(w, y, a):
    denominator = 1+2j*a/w
    propagated = np.exp(-a*y*y/denominator)/denominator
    expected = -1j*w*(propagated-np.exp(-a*y*y))
    result = born_module()._evaluate_profile_born(
        w, y, 0.03, lambda z: np.exp(-a*z*z), tol=2e-11, shape=0.7
    )
    assert result.coefficient == pytest.approx(expected, abs=2e-9)


@pytest.mark.parametrize("override", [
    {"w": float("nan")}, {"w": float("inf")}, {"y": 0.0}, {"y": -1.0},
    {"y": float("inf")}, {"kappa_s": -0.1}, {"kappa_s": float("nan")},
    {"tol": 0.0}, {"tol": float("nan")}, {"shape": 0.0}, {"shape": float("inf")},
    {"tail_scale": 0.5}, {"phase": "unknown"}, {"method": "unknown"},
])
def test_invalid_parameters_are_rejected_even_on_the_free_shortcut(override):
    params = dict(w=2.0, y=1.0, kappa_s=0.0)
    params.update(override)
    with pytest.raises(ValueError):
        born_module().evaluate_full_born(**params)


def test_free_shortcut_does_not_misreport_an_unevaluated_coefficient():
    result = born_module().evaluate_full_born(3.0, 2.0, 0.0, method="real_tail")
    assert result.coefficient is None
    assert result.method == "real_tail"


def test_fixed_go_completion_has_the_exact_geometric_limit_and_same_born_derivative():
    from nfw_wave.lens import single_image
    module = born_module()
    y, k = 2.0, 0.08
    amplitude = single_image(y, k).amplitude
    low = module.evaluate_go_anchored_born(40.0, y, k)
    high = module.evaluate_go_anchored_born(320.0, y, k)
    assert abs(high.value-amplitude) < abs(low.value-amplitude)/4
    h = 2e-4
    f_h = module.evaluate_go_anchored_born(2.2, 1.3, h)
    f_half = module.evaluate_go_anchored_born(2.2, 1.3, h/2)
    derivative = (4*f_half.value-f_h.value-3)/h
    full = module.evaluate_full_born(2.2, 1.3, h, phase="minimum_linear")
    assert derivative == pytest.approx(full.coefficient, abs=1e-6)
    assert high.phase == "minimum"
    assert high.completion == "go_anchored"
    assert high.model_warnings and not high.warnings
    assert module.evaluate_go_anchored_born(-320.0, y, k).value == pytest.approx(
        high.value.conjugate(), abs=1e-13
    )
    assert module.evaluate_go_anchored_born(2.0, y, 0.0).value == 1+0j
    with pytest.raises(ValueError, match="zero.frequency"):
        module.evaluate_go_anchored_born(0.0, y, k)


def test_independent_mpmath_rotated_ray_and_exact_nfw_formula():
    import mpmath
    initial_precision = mpmath.mp.dps
    ctx = mpmath.mp.clone()
    ctx.dps = 42
    w, y = ctx.mpf("1.7"), ctx.mpf("0.9")
    rotation = ctx.exp(ctx.j*ctx.pi/4)

    def p(z):
        if z == 0:
            return ctx.mpf(0)
        return 2*(ctx.log(z/2)**2+ctx.acos(1/z)**2)

    py = ctx.re(p(y))
    expected = []
    for power in [1,2,3]:
        def integrand(t):
            z = rotation*t
            return (rotation*z*ctx.besselj(0, w*y*z)
                    *ctx.exp(ctx.j*w*(z*z+y*y)/2)*(p(z)-py)**power)
        integral = ctx.quad(integrand, [0, .1, .5, 1, 2, 4, 8, 14])
        expected.append(complex((-ctx.j*w)**(power+1)/ctx.factorial(power)*integral))
    result = born_module().evaluate_full_born(float(w), float(y), .04, phase="source", order=3, tol=2e-11)
    differences = [abs(a-b) for a,b in zip(expected,result.coefficients)]
    print("mpmath independent ray C1/C2/C3=", expected, "differences=", differences)
    assert result.coefficients == pytest.approx(expected, abs=2e-10)
    assert mpmath.mp.dps == initial_precision


def test_order_two_matches_independent_exact_quadratic_gaussian_series():
    # Exact source-gauge Gaussian: (1-k)^(-1)*exp[-iw*k^2*y^2/(2*(1-k))].
    w, y, k = 2.3, 1.4, .03
    result = born_module()._evaluate_profile_born(
        w, y, k, lambda z:z*z/2, order=2, tol=2e-11
    )
    c2 = 1-1j*w*y*y/2
    assert result.order == 2
    assert result.coefficients == pytest.approx((1, c2), abs=2e-9)
    assert result.terms == pytest.approx((k, k*k*c2), abs=2e-12)
    assert result.value == pytest.approx(1+k+k*k*c2, abs=2e-12)
    assert len(result.coefficient_error_estimates) == 2
    assert not result.warnings


def test_order_two_public_phase_resummed_truncation_and_conjugation():
    from nfw_wave.lens import single_image, potential
    module = born_module()
    w, y, k = 1.7, .9, .08
    source = module.evaluate_full_born(w, y, k, phase="source", order=2)
    minimum = module.evaluate_full_born(w, y, k, phase="minimum", order=2)
    shift = single_image(y, k).minimum_delay+potential(y, k)
    assert minimum.value == pytest.approx(np.exp(-1j*w*shift)*(1+sum(source.terms)), abs=2e-13)
    assert minimum.completion == "phase_resummed_born2"
    negative = module.evaluate_full_born(-w, y, k, order=2)
    assert negative.value == pytest.approx(minimum.value.conjugate(), abs=1e-14)
    assert negative.coefficients == pytest.approx(tuple(c.conjugate() for c in minimum.coefficients), abs=1e-14)
    assert negative.terms == pytest.approx(tuple(c.conjugate() for c in minimum.terms), abs=1e-14)
    assert module.evaluate_full_born(w, y, 0, order=2).order == 2
    assert module.evaluate_full_born(0, y, k, order=2).coefficients == (0j, 0j)
    with pytest.raises(ValueError, match="first.order"):
        module.evaluate_full_born(w, y, k, order=2, phase="minimum_linear")


def test_order_three_matches_exact_gaussian_series_and_public_nfw_conjugation():
    w, y, k = 1.8, 1.1, .025
    result = born_module()._evaluate_profile_born(w, y, k, lambda z:z*z/2, order=3, tol=2e-11)
    expected = (1, 1-1j*w*y*y/2, 1-1j*w*y*y)
    assert result.coefficients == pytest.approx(expected, abs=2e-9)
    assert result.value == pytest.approx(1+sum(k**n*c for n,c in enumerate(expected,1)), abs=2e-12)
    positive = born_module().evaluate_full_born(w, y, k, order=3)
    negative = born_module().evaluate_full_born(-w, y, k, order=3)
    assert positive.order == 3 and len(positive.coefficients) == 3
    assert positive.completion == "phase_resummed_born3"
    assert negative.value == pytest.approx(positive.value.conjugate(), abs=1e-14)
    assert negative.terms == pytest.approx(tuple(c.conjugate() for c in positive.terms), abs=1e-14)


@pytest.mark.parametrize("order", [2, 3])
def test_higher_orders_constant_gauge_and_nontrivial_gaussian_moments(order):
    from math import comb, factorial
    w, y, a, k = 1.9, 1.2, .3, .04
    profile = lambda z: np.exp(-a*z*z)
    base = born_module()._evaluate_profile_born(w, y, k, profile, order=order, tol=2e-11)
    shifted = born_module()._evaluate_profile_born(w, y, k, lambda z:profile(z)+13, order=order, tol=2e-11)
    assert shifted.coefficients == pytest.approx(base.coefficients, abs=3e-9)
    py = np.exp(-a*y*y)
    expected = []
    for n in range(1,order+1):
        moment = sum(comb(n,j)*(-py)**(n-j)*np.exp(-j*a*y*y/(1+2j*j*a/w))/(1+2j*j*a/w)
                     for j in range(n+1))
        expected.append((-1j*w)**n/factorial(n)*moment)
    assert base.coefficients == pytest.approx(expected, abs=3e-9)


def test_second_nfw_strength_coefficient_with_minimum_delay_counterterm():
    from nfw_wave.lens import potential, single_image, radial_derivatives
    from nfw_wave.reference import amplification_real_tail
    w, y, h = 2.2, 1.3, .002
    minimum_values = {}
    source_values = {}
    for strength in [h, h/2, h/4]:
        f = amplification_real_tail(w, y, strength, tol=5e-13)
        shift = single_image(y,strength).minimum_delay+potential(y,strength)
        minimum_values[strength] = f.value
        source_values[strength] = f.value*np.exp(1j*w*shift)
    def second_coefficient(values, step):
        return (values[step]-2*values[step/2]+1)/(step*step/2)
    source_richardson = 2*second_coefficient(source_values,h/2)-second_coefficient(source_values,h)
    min_richardson = 2*second_coefficient(minimum_values,h/2)-second_coefficient(minimum_values,h)
    result = born_module().evaluate_full_born(w,y,.01,order=2,phase="source",tol=1e-11)
    source_c2 = result.coefficients[1]
    alpha = radial_derivatives(y,1)[0]
    expected_min_c2 = source_c2+1j*w*alpha*alpha/2
    print("second NFW source/minimum coefficients:", source_richardson, min_richardson,
          "predictions:", source_c2, expected_min_c2)
    assert source_richardson == pytest.approx(source_c2, abs=6e-6)
    assert min_richardson == pytest.approx(expected_min_c2, abs=6e-6)


@pytest.mark.parametrize("order", [0, 4, 1.5, True])
def test_invalid_order_is_rejected(order):
    with pytest.raises(ValueError, match="order"):
        born_module().evaluate_full_born(1,1,0,order=order)


def born_module():
    assert util.find_spec("nfw_advanced.born") is not None, "full Born module is missing"
    return import_module("nfw_advanced.born")


def test_free_lens_has_exact_normalization_and_no_quadrature():
    result = born_module().evaluate_full_born(2.3, 1.7, 0.0)
    assert result.value == 1.0 + 0.0j
    assert result.error_estimate == 0.0
    assert result.evaluations == 0
    assert result.phase == "minimum"
    assert result.seconds >= 0.0
    assert result.warnings == ()
