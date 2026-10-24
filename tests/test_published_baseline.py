import mpmath as mp
import numpy as np


def test_published_wu31_baseline_uses_potential_secant_and_exact_geometry():
    from nfw_wave.published_baseline import LocalSISBaseline
    from nfw_wave.approximations import PreparedApproximation
    y,k,w = 3.,.1,2.
    b = LocalSISBaseline(y,k)
    with mp.workdps(60):
        rr = mp.sqrt(2)/(mp.mpf(w)*y)
        psi = 2*mp.mpf(str(k))*(mp.log(rr/2)**2+mp.acos(1/rr)**2)
        exact_center = float(mp.re(psi)/(mp.sqrt(2)*y*y))
    assert np.isclose(b.center(w),exact_center,rtol=2e-12)
    p = PreparedApproximation(y,k)
    expected = p.evaluate(w,'image1')+exact_center*np.exp(1j*w*p.center_delay)
    assert abs(b.evaluate(w)-expected)<1e-13
    assert b.evaluate(-w)==np.conj(b.evaluate(w))
    q = 1e5
    ratio = b.center(q/y)/(4*k/((q/y)**2*y**4))
    assert np.isclose(ratio,np.log(np.sqrt(2)*q)/(2*np.sqrt(2)),rtol=2e-8)


def test_source_geometry_delay_is_converted_to_common_exact_time_zero():
    from nfw_wave.published_baseline import LocalSISBaseline
    b = LocalSISBaseline(1.,.1)
    w = 50.
    center_part = b.center(w)*np.exp(1j*w*b.prepared.center_delay)
    source_go = b.source_amplitude*np.exp(1j*w*b.source_delay_error)
    assert abs(b.evaluate_source_geometry(w)-(source_go+center_part))<1e-13
    assert abs(b.source_delay_error)>1e-4
