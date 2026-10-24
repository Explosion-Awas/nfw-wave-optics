"""Independent checks of the local, minimum-image post-GO expansion."""
import numpy as np
import pytest


@pytest.mark.parametrize("y", [0.03, 0.7, 1.0, 4.0, 20.0])
def test_free_lens_coefficients_are_exactly_zero(y):
    from nfw_wave.image_expansion import expansion_coefficients

    assert expansion_coefficients(y, 0.0) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("y,k", [(0.02, 0.1), (0.4, 0.2), (1.0, 0.1), (3.0, 0.8), (10.0, 0.3)])
def test_first_order_coefficient_matches_existing_delta(y, k):
    from nfw_wave.approximations import PreparedApproximation
    from nfw_wave.image_expansion import expansion_coefficients

    expected = PreparedApproximation(y, k)
    amplitude, c1, _ = expansion_coefficients(y, k)
    assert amplitude == expected.image.amplitude
    assert c1 == pytest.approx(expected.delta, rel=2e-13, abs=2e-14)


@pytest.mark.parametrize("y,k", [(1.0, 0.1), (2.0, 0.2)])
def test_second_order_coefficient_is_measured_by_independent_diffraction(y, k):
    from nfw_wave.approximations import PreparedApproximation
    from nfw_wave.image_expansion import expansion_coefficients
    from nfw_wave.reference import amplification_contour

    prepared = PreparedApproximation(y, k)
    amplitude, c1, c2 = expansion_coefficients(y, k)
    frequencies = np.array([200.0, 400.0, 800.0])
    measured = []
    for w in frequencies:
        reference = amplification_contour(w, y, k, tol=2e-12)
        assert not reference.warnings
        # Remove BOTH known cusp orders, with the exact central phase.
        center = prepared.center(w, "cusp1") * np.exp(1j * w * prepared.center_delay)
        remainder = (reference.value - center) / amplitude - 1 - 1j * c1 / w
        measured.append(w * w * remainder)
        first_order = amplitude * (1 + 1j * c1 / w) + center
        second_order = first_order + amplitude * c2 / w**2
        assert abs(reference.value - second_order) < abs(reference.value - first_order) / 20
    measured = np.asarray(measured)
    errors = abs(measured.real - c2)
    assert errors[-1] < 3e-5, (c2, measured)
    assert errors[-1] < errors[0], (c2, measured)
    # An O(1/w) envelope allows the remaining oscillatory cusp terms.
    assert np.all(errors * frequencies < 0.08), (c2, measured)
    assert np.all(abs(measured.imag) * frequencies < 0.08), measured


def test_prepared_expansion_evaluates_signed_frequency_arrays_without_central_term():
    from nfw_wave.image_expansion import ImageExpansion, expansion_coefficients

    prepared = ImageExpansion(1.0, 0.1)
    assert prepared.coefficients == expansion_coefficients(1.0, 0.1)
    assert prepared.amplitude == prepared.image.amplitude
    w = np.array([[200.0, 400.0], [-200.0, -400.0]])
    expected = prepared.amplitude * (1 + 1j * prepared.c1 / w + prepared.c2 / w**2)
    np.testing.assert_array_equal(prepared.evaluate(w), expected)
    np.testing.assert_array_equal(prepared.evaluate(-w), prepared.evaluate(w).conj())
    assert isinstance(prepared.evaluate(200.0), complex)
    assert ImageExpansion(0.3, 0.0).evaluate(0.001) == 1 + 0j


@pytest.mark.parametrize("w", [0.0, np.nan, np.inf, -np.inf, [1.0, 0.0]])
def test_prepared_expansion_rejects_invalid_frequencies(w):
    from nfw_wave.image_expansion import ImageExpansion

    with pytest.raises(ValueError, match="finite.*nonzero"):
        ImageExpansion(1.0, 0.1).evaluate(w)


@pytest.mark.parametrize("y,k", [(np.nan, 0.1), (np.inf, 0.1), (1.0, np.nan), (1.0, np.inf)])
def test_nonfinite_parameters_are_rejected_before_image_search(y, k):
    from nfw_wave.image_expansion import expansion_coefficients

    with pytest.raises(ValueError, match="finite"):
        expansion_coefficients(y, k)


def _cartesian_gaussian_coefficients(y, k):
    """Independent 2D Taylor/Gaussian calculation: no Hankel coefficients.

    Expand -psi(sqrt((r+u)**2+v**2)) directly as bivariate polynomials,
    then contract the 3rd--6th degree tensors with the exact Gaussian
    covariance. All differentiation and contractions use 85 decimal digits.
    """
    from math import prod
    import mpmath as mp
    from nfw_wave.lens import single_image

    ctx = mp.mp.clone()
    ctx.dps = 85
    r = ctx.mpf(single_image(y, k).x)
    strength = ctx.mpf(k)

    def psi(z):
        return 2 * strength * (ctx.log(z / 2)**2 + ctx.acos(1 / z)**2)

    derivatives = {n: ctx.re(ctx.diff(psi, r, n)) for n in range(1, 7)}
    a, b = 1 - derivatives[2], 1 - derivatives[1] / r

    def multiply(left, right, degree=12):
        out = {}
        for (i, j), x in left.items():
            for (p, q), z in right.items():
                key = (i + p, j + q)
                if sum(key) <= degree:
                    out[key] = out.get(key, ctx.mpf(0)) + x * z
        return out

    # Delta r = r * [(1 + 2u/r + (u^2+v^2)/r^2)^(1/2) - 1].
    inner = {(1, 0): 2 / r, (2, 0): 1 / r**2, (0, 2): 1 / r**2}
    power, radial_shift = {(0, 0): ctx.mpf(1)}, {}
    for n in range(1, 7):
        power = multiply(power, inner, degree=6)
        coefficient = r * ctx.binomial(ctx.mpf("0.5"), n)
        for key, value in power.items():
            radial_shift[key] = radial_shift.get(key, ctx.mpf(0)) + coefficient * value
    tensors = {n: {} for n in range(3, 7)}
    power = {(0, 0): ctx.mpf(1)}
    for n in range(1, 7):
        power = multiply(power, radial_shift, degree=6)
        coefficient = -derivatives[n] / ctx.factorial(n)
        for key, value in power.items():
            degree = sum(key)
            if degree in tensors:
                tensors[degree][key] = tensors[degree].get(key, ctx.mpf(0)) + coefficient * value

    def gaussian(polynomial):
        value = ctx.mpc(0)
        for (i, j), coefficient in polynomial.items():
            if i % 2 == 0 and j % 2 == 0:
                moment = (ctx.j**((i + j) // 2)
                          * prod(range(i - 1, 0, -2)) * prod(range(j - 1, 0, -2))
                          / (a**(i // 2) * b**(j // 2)))
                value += coefficient * moment
        return value

    p3, p4, p5, p6 = (tensors[n] for n in range(3, 7))
    p33 = multiply(p3, p3)
    first = ctx.j * gaussian(p4) - gaussian(p33) / 2
    second = (ctx.j * gaussian(p6) - gaussian(multiply(p3, p5))
              - gaussian(multiply(p4, p4)) / 2
              - ctx.j * gaussian(multiply(p33, p4)) / 2
              + gaussian(multiply(p33, p33)) / 24)
    assert abs(ctx.re(first)) < ctx.mpf("1e-65")
    assert abs(ctx.im(second)) < ctx.mpf("1e-65")
    return float(ctx.im(first)), float(ctx.re(second))


@pytest.mark.parametrize("y,k", [
    (0.002, 0.02), (0.02, 0.1), (0.4, 0.2), (0.7, 0.1),
    (1 - 0.4 * (1 - np.log(2)), 0.1), (1.0, 0.1),
    (1.0, 0.3), (2.0, 0.2), (3.0, 0.8), (10.0, 0.3),
])
def test_hankel_coefficients_match_independent_2d_gaussian_contractions(y, k):
    from nfw_wave.image_expansion import expansion_coefficients

    _, c1, c2 = expansion_coefficients(y, k)
    expected1, expected2 = _cartesian_gaussian_coefficients(y, k)
    assert c1 == pytest.approx(expected1, rel=2e-9, abs=1e-12)
    assert c2 == pytest.approx(expected2, rel=2e-9, abs=1e-12)
    assert isinstance(c1, float) and isinstance(c2, float)


def test_hankel2_asymptotic_signs_against_exact_special_function():
    from scipy.special import hankel2e

    z = np.array([30.0, 100.0, 300.0])
    exact = hankel2e(0, z) * np.sqrt(np.pi * z / 2) * np.exp(-0.25j * np.pi)
    series2 = 1 + 1j / (8 * z) - 9 / (128 * z**2)
    assert np.all(abs(exact - series2) * z**3 < 0.08)
    assert np.all(exact.imag > 0)


def test_reference_contours_are_stable_at_frequencies_used_for_c2_measurement():
    from nfw_wave.reference import amplification_contour, amplification_real_tail

    for w in (200.0, 800.0):
        first = amplification_contour(w, 1.0, 0.1, tol=2e-12, shape=0.7)
        second = amplification_contour(w, 1.0, 0.1, tol=3e-13, shape=1.3, tail_scale=1.2)
        assert not first.warnings and not second.warnings
        assert abs(first.value - second.value) < 2e-12
        if w == 200.0:
            independent = amplification_real_tail(w, 1.0, 0.1, tol=2e-12)
            assert not independent.warnings
            assert abs(independent.value - second.value) < 2e-12


def test_fifth_and_sixth_radial_derivatives_at_scale_radius_have_exact_values():
    import mpmath as mp
    from nfw_wave.image_expansion import _higher_radial_derivatives

    ctx = mp.mp.clone()
    ctx.dps = 85
    k = ctx.mpf("0.1")
    # Obtain independently by Taylor-expanding the analytic NFW deflection.
    exact5 = k * (ctx.mpf(6616) / 105 - 96 * ctx.log(2))
    exact6 = k * (480 * ctx.log(2) - ctx.mpf(73960) / 231)
    d5, d6 = _higher_radial_derivatives(1.0, 0.1)
    assert d5 == pytest.approx(float(exact5), rel=3e-15)
    assert d6 == pytest.approx(float(exact6), rel=3e-15)


def test_preparation_does_not_change_global_mpmath_precision():
    import mpmath as mp
    from nfw_wave.image_expansion import ImageExpansion

    original_precision = mp.mp.prec
    ImageExpansion(1.0, 0.1)
    assert mp.mp.prec == original_precision


@pytest.mark.parametrize("y,k", [(0.0, 0.1), (-1.0, 0.1), (1.0, -0.1), (0.001, 0.3)])
def test_unphysical_or_multiple_image_parameters_are_rejected(y, k):
    from nfw_wave.image_expansion import expansion_coefficients

    with pytest.raises(ValueError):
        expansion_coefficients(y, k)
