# Imports
import unittest

# 3rd party
from lxml import etree

import numpy as np

import pandas as pd

# Our own imports
from . import helpers

from atv_trem2_morpho import prism_writer

# Tests


class TestCleanGroupValues(unittest.TestCase):

    def test_cleans_stupid_whitespace(self):

        group = ' Foo  \t\nBAr '
        res = prism_writer.clean_group_values(group)
        exp = 'Foo BAr'
        self.assertEqual(res, exp)


class TestColumnEncoder(unittest.TestCase):

    def test_encodes_animal_ids(self):

        encoder = prism_writer.ColumnEncoder('animal_id')

        res = encoder.encode(None)
        self.assertEqual(res, '')

        res = encoder.encode(12)
        self.assertEqual(res, '12')

        res = encoder.encode('104')
        self.assertEqual(res, '104')

        res = encoder.encode('QZ104')
        self.assertEqual(res, '104')

    def test_encodes_sex(self):

        encoder = prism_writer.ColumnEncoder('sex')

        res = encoder.encode(None)
        self.assertEqual(res, '')

        res = encoder.encode('female')
        self.assertEqual(res, '0')

        res = encoder.encode('m')
        self.assertEqual(res, '1')

        res = encoder.encode('Male')
        self.assertEqual(res, '1')

        res = encoder.encode(0)
        self.assertEqual(res, '0')

        res = encoder.encode(1.0)
        self.assertEqual(res, '1')

        res = encoder.encode(b'M')
        self.assertEqual(res, '1')

    def test_encodes_generic_float(self):

        encoder = prism_writer.ColumnEncoder('waffles')

        res = encoder.encode(np.nan)
        self.assertEqual(res, '')

        res = encoder.encode(0.1)
        self.assertEqual(res, '0.1000000000000000')

        res = encoder.encode(-1.43567)
        self.assertEqual(res, '-1.4356700000000000')

        res = encoder.encode(1e6)
        self.assertEqual(res, '1000000.0000000000000000')

        res = encoder.encode(np.inf)
        self.assertEqual(res, '')

    def test_encodes_generic_int(self):

        encoder = prism_writer.ColumnEncoder('waffles')

        res = encoder.encode(np.nan)
        self.assertEqual(res, '')

        res = encoder.encode(12)
        self.assertEqual(res, '12')

        res = encoder.encode(-3)
        self.assertEqual(res, '-3')

    def test_encodes_generic_str(self):

        encoder = prism_writer.ColumnEncoder('waffles')

        res = encoder.encode('')
        self.assertEqual(res, '')

        res = encoder.encode('  ')
        self.assertEqual(res, '')

        res = encoder.encode('Test')
        self.assertEqual(res, 'Test')

        res = encoder.encode('  White\t\nSpace\t\n ')
        self.assertEqual(res, 'White\t\nSpace')


class TestPrismWriter(helpers.FileSystemTestCase):

    def test_write_one_way_table(self):

        df = pd.DataFrame({
            'Group': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Value1': [0.5, 0.4, 0.3, 1.1, 1.2, 1.3, 2.5, 2.6, 2.7],
            'Value2': [2.5, 2.4, 2.3, 3.1, 3.2, 3.3, 4.5, 4.6, 4.7],
        })

        outfile = self.tempdir / 'foo.psfx'
        self.assertFalse(outfile.is_file())

        with prism_writer.PrismWriter(self.tempdir / 'foo.psfx') as writer:
            writer.add_one_way_table(data=df,
                                     group_column='Group',
                                     value_column='Value1')
            writer.add_one_way_table(data=df,
                                     group_column='Group',
                                     value_column='Value2')
        self.assertTrue(outfile.is_file())

        with outfile.open('rb') as fp:
            root = etree.fromstring(fp.read())

        nodes = root.findall('./Table')
        self.assertEqual(len(nodes), 2)

        for node in nodes:
            title = node.find('./Title').text
            self.assertEqual(node.attrib['TableType'], 'OneWay')
            self.assertIn(title, {'Value1', 'Value2'})

    def test_write_two_way_table(self):

        df = pd.DataFrame({
            'Day': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Group': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Value1': [0.5, 0.4, 0.3, 1.1, 1.2, 1.3, 2.5, 2.6, 2.7],
            'Value2': [2.5, 2.4, 2.3, 3.1, 3.2, 3.3, 4.5, 4.6, 4.7],
        })

        outfile = self.tempdir / 'foo.psfx'
        self.assertFalse(outfile.is_file())

        with prism_writer.PrismWriter(self.tempdir / 'foo.psfx') as writer:
            writer.add_two_way_table(data=df,
                                     xcolumn='Day',
                                     ycolumn='Group',
                                     value_column='Value1')
            writer.add_two_way_table(data=df,
                                     xcolumn='Day',
                                     ycolumn='Group',
                                     value_column='Value2')
        self.assertTrue(outfile.is_file())

        with outfile.open('rb') as fp:
            root = etree.fromstring(fp.read())

        nodes = root.findall('./Table')
        self.assertEqual(len(nodes), 2)

        for node in nodes:
            title = node.find('./Title').text
            self.assertEqual(node.attrib['TableType'], 'TwoWay')
            self.assertIn(title, {'Value1', 'Value2'})


class TestNormalizeIdVals(unittest.TestCase):

    def test_normalizes_numbers(self):
        self.assertEqual(prism_writer.normalize_id_vals(1), 1.0)
        self.assertEqual(prism_writer.normalize_id_vals(2.5), 2.5)
        self.assertTrue(np.isnan(prism_writer.normalize_id_vals(np.nan)))

    def test_normalizes_strings(self):
        self.assertEqual(prism_writer.normalize_id_vals('1'), 1.0)
        self.assertEqual(prism_writer.normalize_id_vals('2a'), 2.0)
        self.assertEqual(prism_writer.normalize_id_vals('2-5'), 2.5)
        self.assertTrue(np.isnan(prism_writer.normalize_id_vals('waffles')))

    def test_normalizes_df_column(self):

        df = pd.DataFrame({
            'ID': [1, 'a', '3.0', np.nan, '5', '10a'],
        })
        df['Res'] = df['ID'].map(prism_writer.normalize_id_vals)

        exp = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 10.0])
        np.testing.assert_allclose(df['Res'].values, exp)


class TestPrismOneWayTable(unittest.TestCase):

    def test_converts_groups(self):

        df = pd.DataFrame({
            'X': ['A', 'B', 'C', 'D', 'B', 'A', 'B', 'A', 'C'],
            'Y': ['1', '1', '1', '1', '1', '2', '2', '2', '2'],
            'V': [10, 11, 12, 13, 14, 20, 21, 22, 23],
        })

        table = prism_writer.PrismOneWayTable(
            data=df, group_column='X',
        )
        table.group_columns()
        res_index = table._index

        exp_index = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 1],
            [1, 0],
            [2, 1],
            [2, 0],
            [1, 2],
        ])
        np.testing.assert_allclose(res_index, exp_index)

        res_data = table.group_data('V')

        # rows are A, B, C, D
        # cols are padded 1 vs 2
        exp_data = {
            'A': np.array([10, 20, 22]),
            'B': np.array([11, 14, 21]),
            'C': np.array([12, 23, np.nan]),
            'D': np.array([13, np.nan, np.nan]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

    def test_converts_other_groups(self):

        df = pd.DataFrame({
            'X': ['A', 'B', 'C', 'D', 'B', 'A', 'B', 'A', 'C', 'B'],
            'Y': ['1', '1', '1', '1', '1', '2', '2', '2', '2', '3'],
            'V': [10, 11, 12, 13, 14, 20, 21, 22, 23, 30],
        })

        table = prism_writer.PrismOneWayTable(
            data=df, group_column='X', other_group_column='Y',
        )
        table.group_columns()
        res_index = table._index

        exp_index = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 1],
            [3, 0],
            [3, 1],
            [4, 0],
            [3, 2],
            [6, 1],
        ])
        np.testing.assert_allclose(res_index, exp_index)

        res_data = table.group_data('V')

        # rows are A, B, C, D
        # cols are padded 1 vs 2 vs 3
        exp_data = {
            'A': np.array([10, np.nan, np.nan, 20, 22, np.nan, np.nan]),
            'B': np.array([11, 14, np.nan, 21, np.nan, np.nan, 30]),
            'C': np.array([12, np.nan, np.nan, 23, np.nan, np.nan, np.nan]),
            'D': np.array([13, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

    def test_converts_other_groups_subset(self):

        df = pd.DataFrame({
            'X': ['A', 'B', 'C', 'D', 'B', 'A', 'B', 'A', 'C', 'B'],
            'Y': ['1', '1', '1', '1', '1', '2', '2', '2', '2', '3'],
            'V': [10, 11, 12, 13, 14, 20, 21, 22, 23, 30],
        })

        table = prism_writer.PrismOneWayTable(
            data=df, group_column='X', other_group_column='Y',
        )
        table.group_columns()

        res_data = table.group_data('V', subset='1')
        # rows are A, B, C, D
        # cols are padded 1 vs 2 vs 3
        exp_data = {
            'A': np.array([10, np.nan]),
            'B': np.array([11, 14]),
            'C': np.array([12, np.nan]),
            'D': np.array([13, np.nan]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

        res_data = table.group_data('V', subset='2')
        # rows are A, B, C, D
        # cols are padded 1 vs 2 vs 3
        exp_data = {
            'A': np.array([20, 22]),
            'B': np.array([21, np.nan]),
            'C': np.array([23, np.nan]),
            'D': np.array([np.nan, np.nan]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)


class TestPrismTwoWayTable(unittest.TestCase):

    def test_converts_groups(self):

        df = pd.DataFrame({
            'X': ['A', 'B', 'C', 'A', 'B', 'A', 'C'],
            'Y': ['1', '1', '1', '2', '2', '2', '2'],
            'V': [10, 11, 12, 20, 21, 22, 23],
        })

        table = prism_writer.PrismTwoWayTable(
            data=df, xcolumn='X', ycolumn='Y',
        )
        table.group_columns()
        res_index = table._index

        exp_index = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [2, 1, 0],
        ])
        np.testing.assert_allclose(res_index, exp_index)

        res_data = table.group_data('V')

        # rows are A, B, C
        # cols are padded 1 vs 2
        exp_data = {
            '1': np.array([
                [10, np.nan],
                [11, np.nan],
                [12, np.nan],
            ]),
            '2': np.array([
                [20, 22],
                [21, np.nan],
                [23, np.nan],
            ]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

    def test_converts_groups_other_group(self):

        df = pd.DataFrame({
            'O': ['M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F'],
            'X': ['A', 'B', 'C', 'D', 'B', 'A', 'B', 'A', 'C', 'B'],
            'Y': ['1', '1', '1', '1', '1', '2', '2', '2', '2', '3'],
            'V': [10, 11, 12, 13, 14, 20, 21, 22, 23, 30],
        })

        table = prism_writer.PrismTwoWayTable(
            data=df, xcolumn='X', ycolumn='Y',
            other_group_column='O',
        )
        table.group_columns()
        res_index = table._index

        exp_index = np.array([
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [3, 0, 0],
            [1, 0, 0],
            [0, 1, 2],
            [1, 1, 2],
            [0, 1, 3],
            [2, 1, 0],
            [1, 2, 0],
        ])
        np.testing.assert_allclose(res_index, exp_index)

        res_data = table.group_data('V')
        # rows are A, B, C, D
        # cols are padded 1 vs 2
        exp_data = {
            '1': np.array([
                [np.nan, np.nan, 10., np.nan],
                [14., np.nan, 11., np.nan],
                [np.nan, np.nan, 12., np.nan],
                [13., np.nan, np.nan, np.nan],
            ]),
            '2': np.array([
                [np.nan, np.nan, 20., 22.],
                [np.nan, np.nan, 21., np.nan],
                [23., np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]),
            '3': np.array([
                [np.nan, np.nan, np.nan, np.nan],
                [30., np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ])
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

    def test_converts_groups_other_group_subset(self):

        df = pd.DataFrame({
            'O': ['M', 'M', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F'],
            'X': ['A', 'B', 'C', 'D', 'B', 'A', 'B', 'A', 'C', 'B'],
            'Y': ['1', '1', '1', '1', '1', '2', '2', '2', '2', '3'],
            'V': [10, 11, 12, 13, 14, 20, 21, 22, 23, 30],
        })

        table = prism_writer.PrismTwoWayTable(
            data=df, xcolumn='X', ycolumn='Y',
            other_group_column='O',
        )
        table.group_columns()

        res_data = table.group_data('V', subset='F')
        # rows are A, B, C, D
        # cols are padded 1 vs 2
        exp_data = {
            '1': np.array([
                [np.nan],
                [14.],
                [np.nan],
                [13.],
            ]),
            '2': np.array([
                [np.nan],
                [np.nan],
                [23.],
                [np.nan],
            ]),
            '3': np.array([
                [np.nan],
                [30.],
                [np.nan],
                [np.nan],
            ])
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

        res_data = table.group_data('V', subset='M')
        # rows are A, B, C, D
        # cols are padded 1 vs 2
        exp_data = {
            '1': np.array([
                [10., np.nan],
                [11., np.nan],
                [12., np.nan],
                [np.nan, np.nan],
            ]),
            '2': np.array([
                [20., 22.],
                [21., np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]),
            '3': np.array([
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ])
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)


class TestPlaqueGrouper(unittest.TestCase):

    def test_groups_one_set(self):

        df = pd.DataFrame({
            'Tissue normalized 30-125 um2 Plaque Count': [1, 2, np.nan],
            'Tissue normalized 125-250 um2 Plaque Count': [3, 4, np.nan],
            'Tissue normalized 250-500 um2 Plaque Count': [5, 6, 7],
            'Tissue normalized <500 um2 Plaque Count': [8, np.nan, np.nan],  # typo, but should still work
        })
        grouper = prism_writer.PlaqueGrouper()
        pivot_df = grouper.pivot(df)

        exp_df = pd.DataFrame({
            'PlaqueGroup': [
                '30-125 um2', '30-125 um2',
                '125-250 um2', '125-250 um2',
                '250-500 um2', '250-500 um2', '250-500 um2',
                '>500 um2',
            ],
            'Category': ['Tissue normalized Plaque Count']*8,
            'Value': [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
        })

        pd.testing.assert_frame_equal(pivot_df, exp_df)

    def test_groups_one_set_different_names(self):

        df = pd.DataFrame({
            'Tissue normalized Plaque Count (30-125 µm 2) ': [1, 2, np.nan],
            'Tissue normalized Plaque Count (125-250 µm 2)': [3, 4, np.nan],
            'Tissue normalized Plaque Count (250-500 µm 2)': [5, 6, 7],
            'Tissue normalized Plaque Count (>500 µm 2)': [8, np.nan, np.nan],  # typo, but should still work
        })
        grouper = prism_writer.PlaqueGrouper()
        pivot_df = grouper.pivot(df)

        exp_df = pd.DataFrame({
            'PlaqueGroup': [
                '30-125 um2', '30-125 um2',
                '125-250 um2', '125-250 um2',
                '250-500 um2', '250-500 um2', '250-500 um2',
                '>500 um2',
            ],
            'Category': ['Tissue normalized Plaque Count']*8,
            'Value': [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
        })

        pd.testing.assert_frame_equal(pivot_df, exp_df)

    def test_groups_two_sets_missing_categories(self):

        df = pd.DataFrame({
            'Tissue normalized 30-125 um2 Plaque Count': [1, 2, np.nan],
            'Tissue normalized 125-250 um2 Plaque Area': [30, 40, np.nan],
            'Tissue normalized 250-500 um2 Plaque Count': [5, 6, 7],
            'Tissue normalized <500 um2 Plaque Area': [80, np.nan, np.nan],  # typo, but should still work
        })
        grouper = prism_writer.PlaqueGrouper()
        pivot_df = grouper.pivot(df)

        exp_df = pd.DataFrame({
            'PlaqueGroup': [
                '30-125 um2', '30-125 um2',
                '125-250 um2', '125-250 um2',
                '250-500 um2', '250-500 um2', '250-500 um2',
                '>500 um2',
            ],
            'Category': ['Tissue normalized Plaque Count']*2 +
                        ['Tissue normalized Plaque Area']*2 +
                        ['Tissue normalized Plaque Count']*3 +
                        ['Tissue normalized Plaque Area'],
            'Value': [
                1.0, 2.0, 30.0, 40.0, 5.0, 6.0, 7.0, 80.0,
            ],
        })

        pd.testing.assert_frame_equal(pivot_df, exp_df)

    def test_keeps_extra_columns(self):

        df = pd.DataFrame({
            'Animal ID': ['0001', '0002', '0003'],
            'Sex': ['M', 'M', 'F'],
            'Tissue normalized 30-125 um2 Plaque Count': [1, 2, np.nan],
            'Tissue normalized 125-250 um2 Plaque Area': [30, 40, np.nan],
            'Tissue normalized 250-500 um2 Plaque Count': [5, 6, 7],
            'Tissue normalized <500 um2 Plaque Area': [80, np.nan, 90],  # typo, but should still work
        })
        grouper = prism_writer.PlaqueGrouper()
        pivot_df = grouper.pivot(df, extra_columns=['Sex', 'Animal ID'])

        exp_df = pd.DataFrame({
            'PlaqueGroup': [
                '30-125 um2', '30-125 um2',
                '125-250 um2', '125-250 um2',
                '250-500 um2', '250-500 um2', '250-500 um2',
                '>500 um2', '>500 um2',
            ],
            'Category': ['Tissue normalized Plaque Count']*2 +
                        ['Tissue normalized Plaque Area']*2 +
                        ['Tissue normalized Plaque Count']*3 +
                        ['Tissue normalized Plaque Area']*2,
            'Value': [
                1.0, 2.0, 30.0, 40.0, 5.0, 6.0, 7.0, 80.0, 90.0,
            ],
            'Sex': [
                'M', 'M', 'M', 'M', 'M', 'M', 'F', 'M', 'F'
            ],
            'Animal ID': [
                '0001', '0002', '0001', '0002', '0001', '0002', '0003', '0001', '0003',
            ],
        })
        pd.testing.assert_frame_equal(pivot_df, exp_df)

    def test_can_convert_to_long_tables(self):

        df = pd.DataFrame({
            'Animal ID': ['0001', '0002', '0003'],
            'Sex': ['M', 'M', 'F'],
            'Prism Group': [1, 2, 1],
            'Prism Names': ['foo', 'bar', 'foo'],
            'Tissue normalized 30-125 um2 Plaque Count': [1, 2, np.nan],
            'Tissue normalized 125-250 um2 Plaque Area': [30, 40, np.nan],
            'Tissue normalized 250-500 um2 Plaque Count': [5, 6, 7],
            'Tissue normalized <500 um2 Plaque Area': [80, np.nan, 90],  # typo, but should still work
        })
        grouper = prism_writer.PlaqueGrouper()
        pivot_df = grouper.pivot(df, extra_columns=[
            'Sex', 'Animal ID', 'Prism Group', 'Prism Names'])

        # Subset columns and turn back into tables
        count_df = pivot_df[pivot_df['Category'] == 'Tissue normalized Plaque Count']

        table = prism_writer.PrismTwoWayTable(
            data=count_df, xcolumn='PlaqueGroup', ycolumn='Prism Group',
            other_group_column='Sex', name_column='Prism Names',
            order=grouper.order,
        )
        table.group_columns()
        res_data = table.group_data('Value')
        exp_data = {
            'foo': np.array([
                [np.nan, np.nan, 1.],
                [np.nan, np.nan, np.nan],
                [7., np.nan, 5.],
                [np.nan, np.nan, np.nan],
            ]),
            'bar': np.array([
                [np.nan, np.nan, 2.],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 6.],
                [np.nan, np.nan, np.nan],
            ]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)

        area_df = pivot_df[pivot_df['Category'] == 'Tissue normalized Plaque Area']

        table = prism_writer.PrismTwoWayTable(
            data=area_df, xcolumn='PlaqueGroup', ycolumn='Prism Group',
            other_group_column='Sex', name_column='Prism Names',
            order=grouper.order,
        )
        table.group_columns()
        res_data = table.group_data('Value')
        exp_data = {
            'foo': np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 30.],
                [np.nan, np.nan, np.nan],
                [90., np.nan, 80.],
            ]),
            'bar': np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 40.],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]),
        }
        self.assertEqual(res_data.keys(), exp_data.keys())
        for key, res in res_data.items():
            exp = exp_data[key]
            np.testing.assert_allclose(res, exp)
