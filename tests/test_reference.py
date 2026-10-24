import numpy as np
import pytest


@pytest.mark.parametrize('w,y', [(0.03,0.3), (1,1), (30,3), (300,10)])
def test_split_hankel_free_lens_is_exactly_unity(w,y):
    from nfw_wave.reference import amplification_contour
    got = amplification_contour(w,y,0.0,tol=3e-11)
    assert abs(got.value-1) < 2e-10
    assert got.error_estimate < 1e-9


@pytest.mark.parametrize('w,y,k',[(0.1,0.3,0.1),(3.0,1.0,0.3),(20.0,3.0,0.1)])
def test_nfw_split_hankel_converges_and_matches_real_segment_contour(w,y,k):
    from nfw_wave.reference import amplification_contour, amplification_real_tail
    a = amplification_contour(w,y,k,tol=2e-9,shape=0.7)
    b = amplification_contour(w,y,k,tol=2e-11,shape=1.3,tail_scale=1.2)
    c = amplification_real_tail(w,y,k,tol=3e-11)
    assert abs(a.value-b.value) < 2e-8, (a,b)
    assert abs(b.value-c.value) < 3e-9, (b,c)
    assert not b.warnings and not c.warnings


def test_high_frequency_coarse_contour_resolves_short_hankel_tail_without_warning():
    from nfw_wave.reference import amplification_contour
    coarse = amplification_contour(300.,2.,.3,tol=3e-9,shape=.75)
    fine = amplification_contour(300.,2.,.3,tol=3e-11,shape=1.25)
    assert not coarse.warnings
    assert abs(coarse.value-fine.value) < 2e-9
