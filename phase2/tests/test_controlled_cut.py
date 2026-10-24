"""Empirical node indicators are checked against independent full-wave values."""
import pytest


def test_controlled_cut_returns_explicit_empirical_error_indicator():
    from nfw_advanced.controlled_cut import ControlledCutNFW
    from nfw_advanced.reference_stable import amplification_stable
    w,y,k = 1/27.7,27.7,.183
    result = ControlledCutNFW(y,k).evaluate(w,atol=1e-8)
    reference = amplification_stable(w,y,k,tol=2e-12)
    assert result.converged
    assert result.route == "cut96"
    assert result.error_estimate <= 1e-8
    assert abs(result.value-reference.value) <= result.error_estimate+reference.error_estimate
    assert result.estimator_type == "empirical_node_indicator"


def test_controlled_cut_refines_when_initial_node_indicator_fails():
    from nfw_advanced.controlled_cut import ControlledCutNFW
    from nfw_advanced.reference_stable import amplification_stable
    result = ControlledCutNFW(8.,.7).evaluate(.3/8,atol=1e-10)
    reference = amplification_stable(.3/8,8.,.7,tol=2e-12)
    assert result.converged
    assert result.route == "cut192"
    assert result.error_estimate <= 1e-10
    assert abs(result.value-reference.value) <= result.error_estimate+reference.error_estimate


def test_controlled_cut_uses_numerical_reference_outside_calibration_box():
    from nfw_advanced.controlled_cut import ControlledCutNFW
    from nfw_advanced.reference_stable import amplification_stable
    result = ControlledCutNFW(2.,.1).evaluate(.5,atol=1e-10)
    reference = amplification_stable(.5,2.,.1,tol=1e-12)
    assert result.route == "reference"
    assert result.estimator_type == "adaptive_quad_diagnostic"
    assert result.converged and not result.warnings
    assert abs(result.value-reference.value) <= 1e-10


def test_band_gate_cannot_pass_on_absolute_tolerance_alone():
    import numpy as np
    from nfw_advanced.controlled_cut import ControlledCutNFW
    w = np.linspace(.47,5.3,9)/27.7
    result = ControlledCutNFW(27.7,.183).evaluate(w,atol=1e-8,band_rtol=1e-16)
    assert not result.converged
    assert result.band_relative_indicator > 1e-16


@pytest.mark.parametrize("atol",[0.,-1.,float("nan"),float("inf")])
def test_controlled_cut_rejects_invalid_absolute_tolerance(atol):
    from nfw_advanced.controlled_cut import ControlledCutNFW
    with pytest.raises(ValueError):
        ControlledCutNFW(27.7,.183).evaluate(1/27.7,atol=atol)
