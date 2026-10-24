import pytest


def test_hybrid_uses_calibrated_asymptotic_only_when_estimate_meets_target():
    from nfw_wave.hybrid import HybridLens
    from nfw_wave.reference import amplification_contour
    lens = HybridLens(5.0,0.1,safety_factor=2.0)
    fast = lens.evaluate(100.,atol=1e-6)
    assert fast.method == 'asymptotic_ft0'
    assert fast.error_estimate <= 1e-6
    assert fast.empirical
    ref = amplification_contour(100.,5.,.1,tol=3e-11)
    assert abs(fast.value-ref.value) < 1e-6
    fallback = lens.evaluate(0.1,atol=1e-6)
    assert fallback.method == 'numerical_contour'
    assert abs(fallback.value-amplification_contour(.1,5.,.1,tol=3e-11).value) < 1e-7
    outside = HybridLens(20.,.1).evaluate(30.,atol=1e-6)
    assert outside.method == 'numerical_contour'
    with pytest.raises(ValueError):
        lens.evaluate(1.,atol=-1)
