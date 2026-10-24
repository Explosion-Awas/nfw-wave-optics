import numpy as np


def test_band_hybrid_controls_absolute_error_and_frequency_band_residual():
    from nfw_wave.hybrid import HybridLens
    from nfw_wave.band_hybrid import evaluate_band
    from nfw_wave.reference import amplification_contour
    from nfw_wave.metrics import band_metrics
    lens = HybridLens(1.,.1)
    w = np.geomspace(30.,300.,7)
    ans = evaluate_band(lens,w,atol=1e-3,residual_rtol=.01)
    refs = np.array([amplification_contour(float(u),1.,.1,tol=3e-11).value for u in w])
    metric = band_metrics(w,ans.values,refs,lens.prepared.image.amplitude)
    assert metric['max_complex_abs'] < 1e-3
    assert metric['band_residual_relative_l2'] < .01
    assert ans.estimated_residual_relative_l2 <= .01
    assert 'numerical_contour' in ans.methods


def test_vectorized_band_fast_path_matches_frozen_pointwise_envelope():
    from nfw_wave.hybrid import HybridLens
    from nfw_wave.band_hybrid import evaluate_band
    lens = HybridLens(10.,.1)
    w = np.linspace(20.,30.,31)
    ans = evaluate_band(lens,w,atol=1e-6,residual_rtol=.01)
    assert ans.vectorized_fast_path
    assert all(m=='asymptotic_ft0' for m in ans.methods)
    np.testing.assert_allclose(ans.values,lens.prepared.evaluate(w,'ft0'),rtol=0,atol=0)
    expected = [lens.diagnostics(float(u))['empirical_estimate'] for u in w]
    np.testing.assert_allclose(ans.error_estimates,expected,rtol=3e-15)
