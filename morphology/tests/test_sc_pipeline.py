""" Tests for the tools in the single cell pipeline """

# Imports
import unittest

# 3rd party
import pandas as pd

# Our own imports
from atv_trem2_morpho import sc_pipeline

# Tests


class TestCalcCountDf(unittest.TestCase):

    def test_counts_simple_label(self):

        df = pd.DataFrame({
            'Label': [1, 1, 2, 2, 2, 3, 3, 4],
            'Ignore': [0, 0, 0, 0, 0, 0, 0, 0],
        })

        res = sc_pipeline.calc_count_df(df, label_column='Label')
        exp = pd.DataFrame({
            'LabelType': ['Label', 'Label', 'Label', 'Label'],
            'LabelID': [1, 2, 3, 4],
            'CountCells': [2, 3, 2, 1],
            'TotalCells': [8, 8, 8, 8],
            'PercentCells': [25.0, 37.5, 25.0, 12.5],
        })

        pd.testing.assert_frame_equal(res, exp, check_like=True)

    def test_counts_disaggregated_by_animal_id(self):

        df = pd.DataFrame({
            'Label': [1, 1, 2, 2, 2, 3, 3, 4],
            'Ignore': [0, 0, 0, 0, 0, 0, 0, 0],
            'Animal ID': [1, 2, 1, 2, 1, 2, 1, 2],
        })

        res = sc_pipeline.calc_count_df(df, label_column='Label', id_columns='Animal ID')
        exp = pd.DataFrame({
            'LabelType': ['Label', 'Label', 'Label', 'Label', 'Label', 'Label', 'Label', 'Label'],
            'Animal ID': [1, 2, 1, 2, 1, 2, 1, 2],
            'LabelID': [1, 1, 2, 2, 3, 3, 4, 4],
            'CountCells': [1, 1, 2, 1, 1, 1, 0, 1],
            'TotalCells': [4, 4, 4, 4, 4, 4, 4, 4],
            'PercentCells': [25.0, 25.0, 50.0, 25.0, 25.0, 25.0, 0.0, 25.0],
        })
        pd.testing.assert_frame_equal(res, exp, check_like=True)
