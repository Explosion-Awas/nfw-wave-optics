import mpmath as mp
import numpy as np
import pytest


def test_nfw_potential_matches_high_precision_at_center_transition_and_complex_points():
    from nfw_wave.lens import potential

    with mp.workdps(65):
        for z in [1e-8, 1e-3, 0.1, 0.99999, 1.0, 1.00001, 3.0, 0.01+0.006j, 0.6-0.2j, 2+1j]:
            zz = mp.mpc(z)
            expected = 2*mp.mpf('0.1')*(mp.log(zz/2)**2 + mp.acos(1/zz)**2)
            got = potential(z, 0.1)
            assert abs(got-complex(expected)) < 2e-13*max(abs(complex(expected)), 1e-25), (z, got, expected)
    assert potential(0, 0.1) == 0


def test_radial_derivatives_and_single_image_geometry_match_high_precision():
    from nfw_wave.lens import radial_derivatives, single_image

    with mp.workdps(60):
        k = mp.mpf('0.1')
        def psi(z):
            return 2*k*(mp.log(z/2)**2 + mp.acos(1/z)**2)
        for x in [1e-5, 0.02, 0.7, 0.99, 1.0, 1.01, 1.4, 5.0]:
            got = radial_derivatives(x, float(k))
            for n, value in enumerate(got, 1):
                expected = float(mp.re(mp.diff(psi, mp.mpf(str(x)), n)))
                assert np.isclose(value, expected, rtol=2e-10, atol=2e-12), (x,n,value,expected)
        image = single_image(0.7, float(k))
        exact_x = mp.findroot(lambda z: z-mp.diff(psi,z)-mp.mpf('0.7'), 0.9)
        assert abs(image.x-float(mp.re(exact_x))) < 2e-13
        assert abs(image.x-radial_derivatives(image.x,float(k))[0]-0.7) < 2e-13
        assert image.amplitude > 1
        assert image.minimum_delay < 0
        assert image.caustic_y < 0.7
        with pytest.raises(ValueError, match='single'):
            single_image(image.caustic_y/2, float(k))
