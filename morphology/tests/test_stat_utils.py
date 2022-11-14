""" Tests for the stat tools """

# Imports
import unittest

# 3rd party
import numpy as np

# Our own imports
from atv_trem2_morpho import stat_utils

# Tests


class TestCalcPValues(unittest.TestCase):

    def assert_pvalues_equal(self, res, exp, places: int = 5):
        self.assertEqual(set(res), set(exp))
        for key, res_val in res.items():
            exp_val = exp[key]
            self.assertAlmostEqual(res_val, exp_val, places=5)

    def test_one_comparison(self):

        val1 = np.array([0, 0.1, -0.1])

        res = stat_utils.calc_pvalues([val1])
        exp = {}

        self.assert_pvalues_equal(res, exp)

    def test_two_comparisons_sig(self):

        val1 = np.array([0, 0.1, -0.1])
        val2 = np.array([1, 1.1, 0.9])

        res = stat_utils.calc_pvalues([val1, val2])
        exp = {(0, 1): 0.0002552}

        self.assert_pvalues_equal(res, exp)

    def test_two_comparisons_non_sig(self):

        val1 = np.array([0, 0.1, -0.1])
        val2 = np.array([1, 1.1, -1.0])

        res = stat_utils.calc_pvalues([val1, val2])
        exp = {}

        self.assert_pvalues_equal(res, exp)

    def test_three_comparisons(self):

        val1 = np.array([0, 0.1, -0.1])
        val2 = np.array([1, 1.1, 0.9])
        val3 = np.array([2, 2.1, 1.9])

        res = stat_utils.calc_pvalues([val1, val2, val3])
        exp = {
            (0, 1): 0.0002552,
            (0, 2): 4.9448451e-05,
            (1, 2): 0.0002552,
        }
        self.assert_pvalues_equal(res, exp)
