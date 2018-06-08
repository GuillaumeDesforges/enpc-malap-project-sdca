import unittest
import numpy as np
import engine.estimators.projections as projections


class TestIdentityProjection(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(5)
        self.projection = projections.identity_projection

    def test(self):
        x_proj = self.projection(self.x)
        np.testing.assert_array_equal(x_proj, self.x)


class TestPolynomialProjection(unittest.TestCase):
    def setUp(self):
        self.degree = 2
        self.projection = projections.build_polynomial_projection(degree=self.degree)

    def test1d(self):
        x = np.arange(10)
        x_proj = self.projection(x)
        x_expe = np.stack([x**0, x**1, x**2], axis=-1)
        np.testing.assert_allclose(x_proj, x_expe)

    def test2d(self):
        x = np.arange(20).reshape((10, 2))
        x_proj = self.projection(x)
        x_expe = np.stack([
            x[:, 0]**0 * x[:, 1]**0,
            x[:, 0]**0 * x[:, 1]**1,
            x[:, 0]**0 * x[:, 1]**2,
            x[:, 0]**1 * x[:, 1]**0,
            x[:, 0]**1 * x[:, 1]**1,
            x[:, 0]**2 * x[:, 1]**0], axis=-1)
        np.testing.assert_allclose(x_proj, x_expe)
