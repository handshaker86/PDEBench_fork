import numpy as np

from pdebench.models.rebuttal_evaluate import continuity_residual, evaluate


def test_periodic_identical_fields_have_zero_error():
    rng = np.random.default_rng(3)
    field = rng.normal(size=(2, 2, 8, 8))
    aggregate, rows = evaluate(field, field, 0.5, 0.5)
    assert aggregate["rmse"] == 0.0
    assert aggregate["e_div"] == 0.0
    assert len(rows) == 2


def test_constant_density_and_velocity_satisfy_continuity():
    density = np.ones((2, 6, 7))
    velocity = np.ones((2, 2, 6, 7))
    residual = continuity_residual(density, density, velocity, 0.1, 1.0, 1.0)
    np.testing.assert_allclose(residual, 0.0)
