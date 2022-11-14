""" Surface grouper class

Collect stats with sums, means, and/or medians:

.. code-block:: python

    grouper = SurfaceGrouper(
        asc_stats_df,
        key_columns=key_columns,
    )
    grouper.calc_sum_df(
        sum_columns=sum_columns,
        total_column='TotalSpots',
        rename_columns={'Volume': 'TotalVolume'},
    )
    grouper.plot_all(
        plotdir=plotdir,
        var_name='Phenotype',
    )

See :py:class:`SurfaceGrouper` for details

"""

# Imports
import re
import shutil
import pathlib
import functools
import itertools
from typing import List, Tuple, Optional, Callable, Dict

# 3rd party
import numpy as np

import pandas as pd

from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Our own imports
from . import prism_writer
from .plot_utils import plot_boxes

# Constants
reNORM = re.compile(r'[^0-9a-z]+', re.IGNORECASE)

# Classes


class SurfaceGrouper(object):
    """ Group surfaces and calculate mean, median, or sum columns

    :param DataFrame df:
        The data frame to make multiple groups over
    :param list[str] key_columns:
        Primary keys to group the dataframe over
    :param str timepoint_column:
        If not None, which column represents the time index
    """

    def __init__(self, df: pd.DataFrame,
                 key_columns: List[str],
                 timepoint_column: Optional[str] = None):
        self.df = df
        self.key_columns = key_columns
        self.timepoint_column = timepoint_column

        self.sum_columns = None
        self.mean_columns = None
        self.median_columns = None

        self.sum_df = None
        self.mean_df = None
        self.median_df = None

        self._excel_df = None

    def calc_sum_df(self,
                    sum_columns: List[str],
                    rename_columns: Optional[Dict[str, str]] = None,
                    total_column: str = 'TotalSurfaces'):
        """ Calculate the sum data frame

        Columns that start with "Is" will be totaled and have percentages calculated

        :param list[str] sum_columns:
            The columns in the original dataframe to sum up
        :param dict[str, str] rename_columns:
            A dictionary of "original column": "new column" names to change
        :param str total_column:
            Name for the column to use to total out the dataframe
        """

        sum_columns_real = []
        for column in sum_columns:
            if column not in sum_columns_real:
                sum_columns_real.append(column)
        sum_columns = sum_columns_real

        df = self.df.copy()
        df[total_column] = 1

        if total_column not in sum_columns:
            sum_columns.append(total_column)
        # Filter the columns we're about to calculate (percentages)
        sum_columns = [c if not c.startswith('Percent') else 'Is' + c[len('Percent'):]
                       for c in sum_columns]
        sum_columns = [c for c in sum_columns if c in df.columns]

        # Merge all the counts and keys, then sum
        sum_df = df[self.key_columns + sum_columns].groupby(
            self.key_columns, as_index=False).sum()

        if rename_columns is None:
            rename_columns = {}

        # Total things up, and also calculate percentages
        feature_columns = [c for c in sum_columns
                           if c != total_column and not c.startswith('Is')]
        for column in sum_columns:
            if column == total_column or not column.startswith('Is'):
                continue
            pct_column = f'Percent{column[2:]}'
            is_column = f'Total{column[2:]}'
            sum_df[pct_column] = sum_df[column] / sum_df[total_column]*100

            # Save these for the final plots
            feature_columns.append(pct_column)
            feature_columns.append(is_column)
            rename_columns[column] = is_column

        # Apply the renames
        feature_columns = [rename_columns.get(c, c) for c in feature_columns]
        sum_df = sum_df.rename(columns=rename_columns)
        feature_columns.append(total_column)

        self.sum_columns = feature_columns
        self.sum_df = sum_df

        self._excel_df = None

    def calc_mean_df(self,
                     mean_columns: List[str],
                     rename_columns: Optional[Dict[str, str]] = None):
        """ Calculate the mean dataframe

        :param list[str] mean_columns:
            The list of columns to calculate means for
        :param dict[str, str] rename_columns:
            The columns to rename
        """
        mean_columns_real = []
        for column in mean_columns:
            if column not in mean_columns_real:
                mean_columns_real.append(column)
        mean_columns = mean_columns_real

        df = self.df.copy()

        # Merge all the counts and keys, then sum
        mean_df = df[self.key_columns + mean_columns].groupby(
            self.key_columns, as_index=False).mean()
        if rename_columns is None:
            rename_columns = {}

        # Apply the renames
        feature_columns = [rename_columns.get(c, c) for c in mean_columns]
        mean_df = mean_df.rename(columns=rename_columns)

        self.mean_columns = feature_columns
        self.mean_df = mean_df

        self._excel_df = None

    def calc_median_df(self,
                       median_columns: List[str],
                       rename_columns: Optional[Dict[str, str]] = None):
        """ Calculate the median dataframe

        :param list[str] median_columns:
            The list of columns to calculate medians for
        :param dict[str, str] rename_columns:
            The columns to rename
        """
        median_columns_real = []
        for column in median_columns:
            if column not in median_columns_real:
                median_columns_real.append(column)
        median_columns = median_columns_real

        df = self.df.copy()

        # Merge all the counts and keys, then sum
        median_df = df[self.key_columns + median_columns].groupby(
            self.key_columns, as_index=False).median()
        if rename_columns is None:
            rename_columns = {}

        # Apply the renames
        feature_columns = [rename_columns.get(c, c) for c in median_columns]
        median_df = median_df.rename(columns=rename_columns)

        self.median_columns = feature_columns
        self.median_df = median_df

        self._excel_df = None

    def write_prism_tables(self,
                           writer: prism_writer.PrismWriter,
                           var_column: str,
                           animal_column: str,
                           ylabels: Optional[Dict[str, str]] = None,
                           order: Optional[List[str]] = None,
                           subset: Optional[Callable] = None,
                           prefix: str = ''):
        """ Add tables to a prism file

        :param PrismWriter writer:
            The open PrismWriter object
        :param str var_column:
            Which column contains the prism group id information
        :param str animal_column:
            Which column contains the individual sample id information (animal ids)
        :param dict[str, str] ylabels:
            Renames for each column from key: value
        :param list[str] order:
            Order for the values in var_column to be plotted
        :param Callable subset:
            If not None, this callable is passed a DataFrame and returns a boolean mask for which rows to keep
        :param str prefix:
            Prefix to append to the table name
        """

        if ylabels is None:
            ylabels = {}

        # Plot all the combos
        pairs = [
            ('sum', self.sum_df, self.sum_columns),
            ('mean', self.mean_df, self.mean_columns),
            ('median', self.median_df, self.median_columns),
        ]
        add_first_column = animal_column is not None
        for key, df, columns in pairs:
            if df is None or columns is None:
                continue

            # Subset the data on the current key
            if subset is not None:
                mask = subset(df)
                df = df[mask].copy()

            if order is None:
                order = list(sorted(np.unique(df[var_column])))

            df['PrismGroup'] = df[var_column].map(lambda x: order.index(x))

            # Add the animal column if requested
            if add_first_column:
                writer.add_one_way_table(
                    data=df,
                    group_column='PrismGroup',
                    name_column=var_column,
                    value_column=animal_column,
                    table_title=f'{prefix}Animal ID',
                )
                add_first_column = False

            for value_column in columns:
                writer.add_one_way_table(
                    data=df,
                    group_column='PrismGroup',
                    name_column=var_column,
                    value_column=value_column,
                    table_title=f'{prefix}{ylabels.get(value_column, value_column)}',
                )

    def calc_excel_table(self, subset: Optional[Callable] = None) -> pd.DataFrame:
        """ Calculate the concatenation of all the value columns

        :param Callable subset:
            If not None, this callable is passed a DataFrame and returns a boolean mask for which rows to keep
        :returns:
            A DataFrame with all the columns concatenated into a single table
        """
        if self._excel_df is not None:
            return self._excel_df

        # Plot all the combos
        pairs = [
            ('sum', self.sum_df, self.sum_columns),
            ('mean', self.mean_df, self.mean_columns),
            ('median', self.median_df, self.median_columns),
        ]
        add_first_column = True

        # Stack everything into a single table
        all_df = {}
        for key, df, columns in pairs:
            if df is None or columns is None:
                continue

            # Subset the data on the current key
            if subset is not None:
                mask = subset(df)
                df = df[mask].copy()

            if add_first_column:
                all_df.update({
                    k: df[k] for k in self.key_columns
                })
                add_first_column = False
            all_df.update({
                k: df[k] for k in columns
            })

        # Stack everything together
        all_df = pd.DataFrame(all_df)

        # Drop invalid columns
        drop_columns = []
        for k in all_df.columns:
            if k in self.key_columns:
                continue
            if np.all(~np.isfinite(all_df[k].values)):
                drop_columns.append(k)
        if drop_columns != []:
            all_df = all_df[[c for c in all_df.columns
                             if c not in drop_columns]].copy()
        self._excel_df = all_df
        return all_df

    def calc_row_stats(self,
                       group_column: str,
                       animal_column: str,
                       value_column: str,
                       comparisons: Optional[List[Tuple[int]]] = None,
                       order: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """ Calculate statistical parameters for a single value column

        :param str group_column:
            Which column contains the group levels (e.g. 'Treatment' has levels 'Drug1', 'Drug2', etc)
        :param str animal_column:
            Which column contains the individual sample levels (e.g. 'Animal ID' has levels 1, 2, 3, 4)
        :param str value_column:
            Which column contains the variable to be evaluated
        :param list[tuple[int]] comparisons:
            If not None, which pairs of groups to compare (if None, all pairwise combinations will be tried)
        :param list[str] order:
            If not None, the order for the categories in group_column (otherwise they're in list(sorted(np.unique))) order
        :returns:
            A DataFrame with summary stats for this variable
        """
        all_df = self.calc_excel_table()
        if value_column not in all_df.columns:
            return None

        all_df = all_df[[group_column, animal_column, value_column]].copy()

        unique_levels = list(sorted(np.unique(all_df[group_column])))
        if order is None:
            order = unique_levels
        if set(unique_levels) != set(order):
            raise KeyError(f'Group column {group_column} has levels {unique_levels} but got order {order}')

        if comparisons is None:
            comparisons = list(itertools.combinations(range(len(unique_levels)), 2))

        stat_df = {
            'Value': [value_column],
            'NumClasses': [len(order)],
        }

        stat_values = {}

        for i, level in enumerate(order):
            mask = all_df[group_column] == level

            level_df = all_df[mask]

            values = level_df[value_column].values
            values = values[np.isfinite(values)]

            stat_df[f'NumAnimals_Class={i}'] = values.shape

            stat_df[f'Mean_Class={i}'] = np.mean(values)
            stat_df[f'Std_Class={i}'] = np.std(values)

            stat_values[i] = values

        f, p = f_oneway(*list(stat_values.values()))
        stat_df['ANOVA_F'] = [f]
        stat_df['ANOVA_P'] = [p]

        stat_functions = [
            ('TTest', functools.partial(ttest_ind, equal_var=True, alternative='two-sided')),
            ('RankSumTest', functools.partial(mannwhitneyu, alternative='two-sided')),
        ]
        for stat_prefix, stat_function in stat_functions:
            stat_value = []
            stat_p_uncorr = []
            for left_id, right_id in comparisons:
                left_values, right_values = stat_values[left_id], stat_values[right_id]

                stat, p = stat_function(left_values, right_values)
                stat_value.append(stat)
                stat_p_uncorr.append(p)

            if len(stat_p_uncorr) <= 1:
                stat_p_corr = stat_p_uncorr
            else:
                stat_p_corr = multipletests(stat_p_uncorr, alpha=0.05, method='hs',
                                            is_sorted=False)[1]
            assert len(stat_value) == len(stat_p_uncorr)
            assert len(stat_value) == len(stat_p_corr)
            assert len(stat_value) == len(comparisons)

            for stat, p_uncorr, p_corr, (left_id, right_id) in zip(stat_value,
                                                                   stat_p_uncorr,
                                                                   stat_p_corr,
                                                                   comparisons):
                stat_df[f'{stat_prefix}_Stat_Class={left_id},{right_id}'] = stat
                stat_df[f'{stat_prefix}_PRaw_Class={left_id},{right_id}'] = p_uncorr
                stat_df[f'{stat_prefix}_PAdj_Class={left_id},{right_id}'] = p_corr
        return pd.DataFrame(stat_df)

    def calc_excel_stats(self,
                         group_column: str,
                         animal_column: str,
                         comparisons: Optional[List[Tuple[int]]] = None,
                         order: Optional[List[str]] = None) -> pd.DataFrame:
        """ Calculate all the stats for all the defined value columns

        :param str group_column:
            Which column contains the group levels (e.g. 'Treatment' has levels 'Drug1', 'Drug2', etc)
        :param str animal_column:
            Which column contains the individual sample levels (e.g. 'Animal ID' has levels 1, 2, 3, 4)
        :param list[tuple[int]] comparisons:
            If not None, which pairs of groups to compare (if None, all pairwise combinations will be tried)
        :param list[str] order:
            If not None, the order for the categories in group_column (otherwise they're in list(sorted(np.unique))) order
        :returns:
            A DataFrame with summary stats for this variable
        """

        stats_df = []
        for value_columns in (self.sum_columns, self.mean_columns, self.median_columns):
            if value_columns is None:
                continue
            for value_column in value_columns:
                stat_df = self.calc_row_stats(
                    group_column=group_column,
                    animal_column=animal_column,
                    value_column=value_column,
                    comparisons=comparisons,
                    order=order)
                if stat_df is not None:
                    stats_df.append(stat_df)
        return pd.concat(stats_df, ignore_index=True)

    def write_excel_stats(self, outfile: pathlib.Path,
                          group_column: str,
                          animal_column: str,
                          comparisons: Optional[List[Tuple[int]]] = None,
                          order: Optional[List[str]] = None):
        """ Write the excel stats out to a file

        :param Path outfile:
            The excel file to write
        """
        outfile.parent.mkdir(parents=True, exist_ok=True)
        stat_df = self.calc_excel_stats(
            group_column=group_column,
            animal_column=animal_column,
            comparisons=comparisons,
            order=order,
        )
        stat_df = stat_df.sort_values('ANOVA_P')
        stat_df.to_excel(outfile, index=False)

    def write_excel_table(self, outfile: pathlib.Path,
                          subset: Optional[Callable] = None):
        """ Write out the excel table for this assay

        :param Path outfile:
            The excel file to write
        :param Callable subset:
            If not None, this callable is passed a DataFrame and returns a boolean mask for which rows to keep
        """
        outfile.parent.mkdir(parents=True, exist_ok=True)
        all_df = self.calc_excel_table(subset=subset)
        all_df.to_excel(outfile, index=False)

    def plot_all(self,
                 var_column: str,
                 plotdir: pathlib.Path,
                 ylabels: Optional[Dict[str, str]] = None,
                 ylims: Optional[Dict[str, Tuple[float]]] = None,
                 force_zero_ylim: bool = False,
                 order: Optional[List[str]] = None,
                 xticklabels: Optional[List[str]] = None,
                 comparisons: Optional[List[Tuple[int]]] = None,
                 prefix: str = '',
                 suffix: str = '.png',
                 subset: Optional[Callable] = None,
                 overwrite: bool = False,
                 **kwargs):
        """ Make all the plots for any of grouped dataframes we have

        :param str var_column:
            The category column to make boxplots for
        :param Path plotdir:
            Directory to write the plots to
        """

        if overwrite and plotdir.is_dir():
            shutil.rmtree(plotdir)
        plotdir.mkdir(parents=True, exist_ok=True)

        if ylabels is None:
            ylabels = {}

        if ylims is None:
            ylims = {}

        # Plot all the combos
        pairs = [
            ('sum', self.sum_df, self.sum_columns),
            ('mean', self.mean_df, self.mean_columns),
            ('median', self.median_df, self.median_columns),
        ]
        for key, df, columns in pairs:
            if df is None or columns is None:
                continue

            # Subset the data on the current key
            if subset is not None:
                mask = subset(df)
                df = df[mask].copy()

            if order is None:
                order = list(sorted(np.unique(df[var_column])))

            if xticklabels is None:
                xticklabels = order
            assert len(order) == len(xticklabels)

            for value_column in columns:
                outname = reNORM.sub('_', value_column).lower().strip('_')
                plotfile = plotdir / f'{prefix}{key}_{outname}{suffix}'

                ylabel = ylabels.get(value_column, value_column)
                ylim = ylims.get(value_column)
                if force_zero_ylim and ylim is None:
                    ylim = [0, None]

                # Make boxplots
                plot_boxes(df=df, var_name=var_column,
                           value_name=value_column,
                           order=order, plotfile=plotfile,
                           xticklabels=xticklabels,
                           ylabel=ylabel,
                           ylim=ylim,
                           pvalue_comparisons=comparisons,
                           suffix=suffix,
                           **kwargs)
