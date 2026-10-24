"""Reference checks at low w and large offset: preserve normalization accuracy."""
import pytest


def test_stable_low_frequency_reference_matches_independent_real_tail():
    from nfw_advanced.reference_stable import amplification_stable
    from nfw_wave.reference import amplification_real_tail
    w,y,k = .3/128.,128.,.1
    result = amplification_stable(w,y,k,tol=3e-13,shape=.8)
    other = amplification_real_tail(w,y,k,tol=2e-11)
    assert not result.warnings
    assert not other.warnings
    assert abs(result.value-other.value) < 3e-12
    assert result.error_estimate < 1e-12


def test_stable_reference_free_lens_large_offset():
    from nfw_advanced.reference_stable import amplification_stable
    result = amplification_stable(10/2048.,2048.,0.,tol=5e-13)
    assert not result.warnings
    assert result.value == pytest.approx(1+0j,abs=8e-13)
