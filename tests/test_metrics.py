import numpy as np


def test_band_metrics_do_not_divide_by_zero_crossing_of_residual():
    from nfw_wave.metrics import band_metrics
    w = np.array([1.,2.,4.])
    go = 1.1
    ref = np.array([go+0.01j,go,go-0.01j])
    approx = ref+1e-4
    m = band_metrics(w,approx,ref,go)
    assert np.isclose(m['max_complex_abs'],1e-4)
    assert np.isfinite(m['band_residual_relative_l2'])
    assert m['band_residual_relative_l2'] < 0.02
    z = band_metrics(w,np.ones(3),np.ones(3),1.)
    assert z['band_residual_relative_l2'] is None
