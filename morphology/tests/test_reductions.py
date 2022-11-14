""" Tests for the pearson residuals class """

# Imports
import unittest

# 3rd party
import numpy as np

# Our own imports
from atv_trem2_morpho import reductions

# Tests


class TestCorrespondenceAnalysis(unittest.TestCase):

    def test_factors_simple_data(self):

        gen = np.random.default_rng(12345)
        data = gen.random((100, 10))

        corr_ana = reductions.CorrespondenceAnalysis(n_components=4)
        trans = corr_ana.fit_transform(data)

        self.assertEqual(trans.shape, (100, 4))


class TestCalcPearsonResiduals(unittest.TestCase):

    def test_raw_residuals(self):

        # Based on the demo at https://www.statology.org/pearson-residuals/
        # matches the R function "chisq.test(data)$residuals"
        data = np.array([
            [120, 90, 40],
            [110, 95, 45],
        ])

        res = reductions.calc_pearson_residuals(data, residuals='raw')
        exp = np.array([
            [0.466, -0.2599, -0.383],
            [-0.466, 0.2599, 0.383],
        ])

        np.testing.assert_allclose(res, exp, atol=1e-2)

    def test_standardized_residuals(self):

        # Based on the demo at https://www.statology.org/pearson-residuals/
        # matches the R function "chisq.test(data)$stdres"
        data = np.array([
            [120, 90, 40],
            [110, 95, 45],
        ])

        res = reductions.calc_pearson_residuals(data, residuals='standardized')
        exp = np.array([
            [0.897, -0.463, -0.595],
            [-0.897, 0.463, 0.595],
        ])

        np.testing.assert_allclose(res, exp, atol=1e-2)

    def test_corral_residuals(self):

        # This is the definition given in Hsu and Colhane 2021
        # It's more normalized than the standard definition
        data = np.array([
            [120, 90, 40],
            [110, 95, 45],
        ])
        res = reductions.calc_pearson_residuals(data, residuals='corral')
        exp = np.array([
            [0.020851, -0.011625, -0.01715],
            [-0.020851, 0.011625, 0.01715],
        ])

        np.testing.assert_allclose(res, exp, atol=1e-4)

    def test_corral_residuals_with_sqrt(self):

        # This is the definition given in Hsu and Colhane 2021
        # It's more normalized than the standard definition
        data = np.array([
            [120, 90, 40],
            [110, 95, 45],
        ])
        res = reductions.calc_pearson_residuals(data, transform='sqrt', residuals='corral')
        exp = np.array([
            [0.01121, -0.00433, -0.009119],
            [-0.011173, 0.004316, 0.009089],
        ])

        np.testing.assert_allclose(res, exp, atol=1e-4)

    def test_freeman_tukey_residuals(self):

        # This is the definition given in Hsu and Colhane 2021
        # It supposedly stablizes overdispersion
        data = np.array([
            [120, 90, 40],
            [110, 95, 45],
        ])
        res = reductions.calc_pearson_residuals(data, residuals='freeman-tukey')
        exp = np.array([
            [0.021625, -0.010516, -0.015609],
            [-0.019998,  0.012674,  0.018507],
        ])

        np.testing.assert_allclose(res, exp, atol=1e-4)
