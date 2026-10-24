"""Prepared, batch-frequency evaluation must preserve the scalar prototype."""
import numpy as np
import pytest


def test_prepared_batch_matches_scalar_cut_and_residual():
    from nfw_advanced.cut_solver import amplification_cut
    from nfw_advanced.batch_cut import PreparedCutNFW
    y,k = 27.7,.183
    frequencies = np.geomspace(.47,5.3,9)/y
    prepared = PreparedCutNFW(y,k)
    actual = prepared.evaluate(frequencies)
    expected = [amplification_cut(float(w),y,k) for w in frequencies]
    np.testing.assert_allclose(actual.value,[x.value for x in expected],rtol=0,atol=3e-13)
    np.testing.assert_allclose(actual.residual,[x.residual for x in expected],rtol=0,atol=3e-13)
    assert actual.amplitude == pytest.approx(expected[0].amplitude,rel=0,abs=0)


def test_prepared_batch_solves_geometry_only_once(monkeypatch):
    from nfw_advanced import batch_cut,cut_solver
    original = batch_cut.single_image
    calls = []
    def counted(y,k):
        calls.append((y,k))
        return original(y,k)
    monkeypatch.setattr(batch_cut,"single_image",counted)
    monkeypatch.setattr(cut_solver,"single_image",counted)
    prepared = batch_cut.PreparedCutNFW(27.7,.183)
    prepared.evaluate(np.geomspace(.47,5.3,9)/27.7)
    prepared.evaluate(np.array([.7,1.4])/27.7)
    assert calls == [(27.7,.183)]


def test_batch_signed_frequencies_obey_conjugation():
    from nfw_advanced.batch_cut import PreparedCutNFW
    prepared = PreparedCutNFW(27.7,.183)
    frequencies = np.array([.47,1.,5.3])/27.7
    positive = prepared.evaluate(frequencies)
    negative = prepared.evaluate(-frequencies)
    np.testing.assert_allclose(negative.value,positive.value.conjugate(),rtol=0,atol=1e-14)
    np.testing.assert_allclose(negative.residual,positive.residual.conjugate(),rtol=0,atol=1e-14)


def test_batch_preserves_shapes_across_block_boundary():
    from nfw_advanced.batch_cut import PreparedCutNFW
    from nfw_advanced.cut_solver import amplification_cut
    q = np.geomspace(.47,5.3,17*47).reshape(17,47)
    actual = PreparedCutNFW(27.7,.183).evaluate(q/27.7)
    assert actual.value.shape == q.shape
    for i in [0,511,512,-1]:
        expected = amplification_cut(float(q.ravel()[i]/27.7),27.7,.183)
        assert actual.residual.ravel()[i] == pytest.approx(expected.residual,rel=0,abs=5e-15)


def test_batch_free_lens_is_exact():
    from nfw_advanced.batch_cut import PreparedCutNFW
    r = PreparedCutNFW(27.7,0.).evaluate(np.array([-.01,.1,1.]))
    np.testing.assert_array_equal(r.value,np.ones(3))
    np.testing.assert_array_equal(r.residual,np.zeros(3))


@pytest.mark.parametrize("w",[0.,float("nan"),float("inf")])
def test_batch_rejects_invalid_frequency(w):
    from nfw_advanced.batch_cut import PreparedCutNFW
    with pytest.raises(ValueError):
        PreparedCutNFW(27.7,.183).evaluate(w)


@pytest.mark.parametrize("n",[1,2.5,float("nan")])
def test_batch_rejects_noninteger_node_counts(n):
    from nfw_advanced.batch_cut import PreparedCutNFW
    with pytest.raises(ValueError):
        PreparedCutNFW(27.7,.183,image_nodes=n)


def test_batch_scalar_returns_complex_values():
    from nfw_advanced.batch_cut import PreparedCutNFW
    r = PreparedCutNFW(27.7,.183).evaluate(.5/27.7)
    assert isinstance(r.value,complex) and isinstance(r.residual,complex)
