""" Tests for the surface grouping tool """

# Imports
import unittest

# 3rd party
import pandas as pd

# Our own imports
from atv_trem2_morpho import surface_grouper

# Tests


class TestSurfaceGrouper(unittest.TestCase):

    def test_calc_sum_df(self):

        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 3, 4],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'C'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'IsValue': [0, 1, 1, 1, 0, 0, 0, 1, 1],
        })

        grouper = surface_grouper.SurfaceGrouper(df, key_columns=['Animal ID', 'Animal Class'])
        grouper.calc_sum_df(
            sum_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'SumValue'},
            total_column='TotalEntries',
        )

        res_df = grouper.sum_df
        exp_df = pd.DataFrame({
            'Animal ID': [1, 2, 3, 4],
            'Animal Class': ['A', 'B', 'A', 'C'],
            'SumValue': [3, 7, 26, 9],
            'TotalValue': [1, 2, 1, 1],
            'TotalEntries': [2, 2, 4, 1],
            'PercentValue': [50.0, 100.0, 25.0, 100.0],
        })

        pd.testing.assert_frame_equal(res_df, exp_df)

    def test_calc_mean_df(self):

        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 3, 4],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'C'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'IsValue': [0, 1, 1, 1, 0, 0, 0, 1, 1],
        })

        grouper = surface_grouper.SurfaceGrouper(df, key_columns=['Animal ID', 'Animal Class'])
        grouper.calc_mean_df(
            mean_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'MeanValue'},
        )

        res_df = grouper.mean_df
        exp_df = pd.DataFrame({
            'Animal ID': [1, 2, 3, 4],
            'Animal Class': ['A', 'B', 'A', 'C'],
            'MeanValue': [1.5, 3.5, 6.5, 9.0],
            'IsValue': [0.5, 1.0, 0.25, 1.0],
        })

        pd.testing.assert_frame_equal(res_df, exp_df)

    def test_calc_median_df(self):

        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 3, 4],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'C'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'IsValue': [0, 1, 1, 1, 0, 0, 0, 1, 1],
        })

        grouper = surface_grouper.SurfaceGrouper(df, key_columns=['Animal ID', 'Animal Class'])
        grouper.calc_median_df(
            median_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'MedianValue'},
        )

        res_df = grouper.median_df
        exp_df = pd.DataFrame({
            'Animal ID': [1, 2, 3, 4],
            'Animal Class': ['A', 'B', 'A', 'C'],
            'MedianValue': [1.5, 3.5, 6.5, 9.0],
            'IsValue': [0.5, 1, 0, 1],
        })

        pd.testing.assert_frame_equal(res_df, exp_df)

    def test_calc_excel_table(self):

        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 3, 4],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'C'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'IsValue': [0, 1, 1, 1, 0, 0, 0, 1, 1],
        })
        grouper = surface_grouper.SurfaceGrouper(df, key_columns=['Animal ID', 'Animal Class'])

        grouper.calc_sum_df(
            sum_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'SumValue'},
            total_column='TotalEntries',
        )
        grouper.calc_mean_df(
            mean_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'MeanValue'},
        )

        res_df = grouper.calc_excel_table()
        exp_df = pd.DataFrame({
            'Animal ID': [1, 2, 3, 4],
            'Animal Class': ['A', 'B', 'A', 'C'],
            'SumValue': [3, 7, 26, 9],
            'PercentValue': [50.0, 100.0, 25.0, 100.0],
            'TotalValue': [1, 2, 1, 1],
            'TotalEntries': [2, 2, 4, 1],
            'MeanValue': [1.5, 3.5, 6.5, 9.0],
            'IsValue': [0.5, 1.0, 0.25, 1.0],
        })
        pd.testing.assert_frame_equal(res_df, exp_df)

    def test_calc_row_stats(self):

        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'C', 'C', 'C', 'C'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        })
        grouper = surface_grouper.SurfaceGrouper(
            df, key_columns=['Animal ID', 'Animal Class'])
        grouper.calc_sum_df(
            sum_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'SumValue'},
            total_column='TotalEntries',
        )

        res_df = grouper.calc_row_stats(
            group_column='Animal Class',
            animal_column='Animal ID',
            value_column='SumValue',
            comparisons=[(0, 1), (0, 2)],
            order=['A', 'C', 'B'])
        exp_df = pd.DataFrame({
            'Value': ['SumValue'],
            'NumClasses': [3],
            'NumAnimals_Class=0': [3],
            'Mean_Class=0': [14.0],
            'Std_Class=0': [7.874007874011811],
            'NumAnimals_Class=1': [4],
            'Mean_Class=1': [14.5],
            'Std_Class=1': [1.118033988749895],
            'NumAnimals_Class=2': [3],
            'Mean_Class=2': [12.0],
            'Std_Class=2': [4.08248290463863],
            'ANOVA_F': [0.1655601659751037],
            'ANOVA_P': [0.8506421959486199],
            'TTest_Stat_Class=0,1': [-0.10592047651328705],
            'TTest_PRaw_Class=0,1': [0.9197637128300704],
            'TTest_PAdj_Class=0,1': [0.9451331339987729],
            'TTest_Stat_Class=0,2': [0.3188964020716404],
            'TTest_PRaw_Class=0,2': [0.7657632266247949],
            'TTest_PAdj_Class=0,2': [0.9451331339987729],
            'RankSumTest_Stat_Class=0,1': [8.0],
            'RankSumTest_PRaw_Class=0,1': [0.6285714285714286],
            'RankSumTest_PAdj_Class=0,1': [0.8620408163265306],
            'RankSumTest_Stat_Class=0,2': [6.0],
            'RankSumTest_PRaw_Class=0,2': [0.7],
            'RankSumTest_PAdj_Class=0,2': [0.8620408163265306],
        })
        pd.testing.assert_frame_equal(res_df, exp_df, rtol=1e-3, atol=1e-3)

    def test_calc_excel_stats(self):
        df = pd.DataFrame({
            'Animal ID': [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
            'Animal Class': ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'B'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'IsValue': [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        })
        grouper = surface_grouper.SurfaceGrouper(
            df, key_columns=['Animal ID', 'Animal Class'])
        grouper.calc_sum_df(
            sum_columns=['Value', 'IsValue'],
            rename_columns={'Value': 'SumValue'},
            total_column='TotalEntries',
        )

        res_df = grouper.calc_excel_stats(
            group_column='Animal Class',
            animal_column='Animal ID',
            comparisons=[(0, 1)],
            order=['A', 'B'],
        )
        exp_df = pd.DataFrame({
            'Value': ['SumValue', 'PercentValue', 'TotalValue', 'TotalEntries'],
            'NumClasses': [2, 2, 2, 2],
            'NumAnimals_Class=0': [3, 3, 3, 3],
            'Mean_Class=0': [14.0, 50.0, 1.3333333333333333, 2.3333333333333335],
            'Std_Class=0': [7.874007874011811, 40.824829046386306, 1.247219128924647, 0.4714045207910317],
            'NumAnimals_Class=1': [3, 3, 3, 3],
            'Mean_Class=1': [12.0, 50.0, 0.6666666666666666, 1.6666666666666667],
            'Std_Class=1': [4.08248290463863, 40.824829046386306, 0.4714045207910317, 0.4714045207910317],
            'ANOVA_F': [0.1016949152542373, 0.0, 0.5, 1.9999999999999998],
            'ANOVA_P': [0.7657632266247953, 1.0, 0.5185185185185183, 0.23019964108049892],
            'TTest_Stat_Class=0,1': [0.3188964020716404, 0.0, 0.7071067811865475, 1.4142135623730951],
            'TTest_PRaw_Class=0,1': [0.7657632266247949, 1.0, 0.5185185185185183, 0.23019964108049873],
            'TTest_PAdj_Class=0,1': [0.7657632266247949, 1.0, 0.5185185185185183, 0.23019964108049873],
            'RankSumTest_Stat_Class=0,1': [6.0, 4.5, 5.5, 7.0],
            'RankSumTest_PRaw_Class=0,1': [0.7, 1.0, 0.8136637157667919, 0.3016995824783478],
            'RankSumTest_PAdj_Class=0,1': [0.7, 1.0, 0.8136637157667919, 0.3016995824783478],
        })
        pd.testing.assert_frame_equal(res_df, exp_df, rtol=1e-3, atol=1e-3)
