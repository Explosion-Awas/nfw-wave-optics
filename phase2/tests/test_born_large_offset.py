"""A complete quadratic Gaussian detects avoidable Born phase cancellation."""

def test_full_born_free_phase_is_stable_at_large_source_offset():
    from nfw_advanced.born import _evaluate_profile_born
    result = _evaluate_profile_born(1.,10000.,.01,lambda z:z*z/2,
                                    tol=1e-10,order=1)
    # Independently completed 2D Gaussian, in source potential gauge:
    # (1-k)^(-1)*exp[-i*w*k^2*y^2/(2*(1-k))]. Its first coefficient is 1.
    assert abs(result.coefficient-1) < 1e-7
