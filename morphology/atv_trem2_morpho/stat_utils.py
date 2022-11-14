""" Tools to calculate stats on various comparisons for plot purposes

Functions:

* :py:func:`calc_power`: Run power calculations to estimate sample size needed to hit a p-value threshold
* :py:func:`calc_pvalues`: Run a one-way ANOVA followed by post-hoc paired t-tests
* :py:func:`calc_pvalue_symbols`: Convert p-values to '*', '**', '*******', etc
* :py:func:`add_pvalues_to_plot`: Plot p-values above a seaborn generated bar or boxplot

"""

# Imports
import itertools
from typing import Dict, Tuple, List

# 3rd party
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, f_oneway

from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.multitest import multipletests

# Functions


def add_pvalues_to_plot(ax: plt.Axes, pvalues: Dict[Tuple[float], float],
                        style: str = 'bars',
                        fontsize: int = 64,
                        linewidth: int = 5,
                        ypct: float = 0.95):
    """ Add p- values to a plot

    :param Axes ax:
        The axes object to add pvalues to
    :param dict[tuple[float], float]:
        The dictionary of (left, right): pvalue coordinates to plot
    """
    pvalue_symbols = calc_pvalue_symbols(pvalues)
    ylim = np.max(ax.get_ylim())

    if style == 'above':
        # Left is assumed to be control
        check_left = None
        for (left, right), symbol in pvalue_symbols.items():
            if check_left is None:
                check_left = left
            elif check_left != left:
                raise ValueError(f'All left conditions must be the same: got {left} expected {check_left}')
            ax.text(right, ylim*ypct, s=symbol, color='black', size=fontsize,
                    horizontalalignment='center', verticalalignment='center')
    elif style in ('bar', 'bars'):
        # Offset each star and line by a little bit
        ct = 0.0
        for (left, right), symbol in pvalue_symbols.items():
            if symbol == '':
                continue
            mid = (left + right)/2
            yoff = ylim*(0.9+ct)
            ax.plot([left, right], [yoff, yoff], '-', color='black', linewidth=linewidth)
            ax.text(mid, yoff, s=symbol, color='black', size=fontsize,
                    horizontalalignment='center', verticalalignment='center')
            ct += 0.02
    else:
        raise KeyError(f'Unknown style: "{style}"')


def calc_pvalue_symbols(pvalues: Dict[object, float]) -> Dict[object, str]:
    """ Get symbols for the pvalues returned by calc_pvalues

    :param dict[object, float] pvalues:
        A dictionary where pvalues are the values
    :returns:
        A dictionary where the corresponding symbol is the value (or '' for not significant)
    """
    symbols = {
        '*': 0.05,
        '**': 0.01,
        '***': 0.001,
    }
    pvalue_symbols = {}
    for key, pvalue in pvalues.items():
        symbol = ''
        for k, v in sorted(symbols.items(), key=lambda x: x[1]):
            if pvalue < v:
                symbol = k
                break
        pvalue_symbols[key] = symbol
    return pvalue_symbols


def calc_power(groups: List[np.ndarray],
               comparisons: List[Tuple[int]] = None,
               power: float = 0.8,
               alpha: float = 0.05,
               ratio: float = 1.0,
               tol: float = 1e-3) -> Dict[Tuple[int], float]:
    """ Power calculations

    :param List[ndarray] groups:
        The list of groups to compare
    :param list[tuple[int]] comparisons:
        A list of (i, j) tuples for columns in groups to compare (default all comparisons)
    :returns:
        For each comparison, the minimum expected number of samples to get a significant result
    """
    if comparisons is None:
        comparisons = itertools.combinations(range(len(groups)), 2)

    # Post Hoc tests
    num_samples = {}
    for i, j in comparisons:
        val1 = groups[i]
        val2 = groups[j]

        cat1_mean = np.mean(val1)
        cat1_std = np.std(val1)

        cat2_mean = np.mean(val2)
        cat2_std = np.std(val2)

        # Params we want
        cat_std = np.sqrt(cat2_std**2 + cat1_std**2)
        cat_delta = abs(cat1_mean - cat2_mean)

        if cat_std < tol or cat_delta < tol:
            continue

        # Solve how many samples (in group 1) are needed for a significant result
        num = tt_ind_solve_power(effect_size=cat_delta/cat_std,
                                 nobs1=None,
                                 power=power,
                                 alpha=alpha,
                                 ratio=ratio,
                                 alternative='two-sided')
        num_samples[(i, j)] = num
    return num_samples


def calc_pvalues(groups: List[np.ndarray],
                 alpha_anova: float = 0.05,
                 alpha_multi: float = 0.05,
                 equal_var: bool = True,
                 comparisons: List[Tuple[int]] = None,
                 multitest_method: str = 'fdr_bh') -> Dict[Tuple[int], float]:
    """ Calculate pvalues for comparisons between groups

    Run a one-way ANOVA followed by pair-wise t-tests and then a correction
    for multiple comparisons

    :param List[ndarray] groups:
        The list of groups to compare
    :param float alpha_anova:
        The minimum ANOVA value to run comparisons at
    :param float alpha_multi:
        The family-wise error rate for multiple comparisons correction
    :param list[tuple[int]] comparisons:
        A list of (i, j) tuples for columns in groups to compare (default all comparisons)
    :returns:
        An array that contains all indices where p < alpha_multi after correction
    """
    if len(groups) < 2:
        return {}
    if any([len(g) < 3 for g in groups]):
        return {}

    # Run some stats
    _, pvalue = f_oneway(*groups)

    # Conservative, some literature says you can do post-hoc tests anyway
    if pvalue > alpha_anova:
        return {}

    if comparisons is None or comparisons == 'all':
        comparisons = itertools.combinations(range(len(groups)), 2)

    pvalues = {}
    # Post Hoc tests
    for i, j in comparisons:
        if i > j:
            i, j = j, i
        if (i, j) in pvalues:
            continue
        val1 = groups[i]
        val2 = groups[j]
        _, pvalue = ttest_ind(
            val1, val2,
            equal_var=equal_var,
            alternative='two-sided')
        pvalues[(i, j)] = pvalue

    # If we only did one comparison, we're done
    if len(pvalues) < 2:
        return {k: v for k, v in pvalues.items() if v < alpha_multi}

    # Multiple comparison corrections
    pvalue_arr = np.array(list(pvalues.values()))
    pvalue_corr = multipletests(pvalue_arr, alpha=alpha_multi, method=multitest_method,
                                is_sorted=False, returnsorted=False)[1]

    # Aggregate over final p-values
    assert len(pvalues) == len(pvalue_corr)
    pvalues_final = {}
    for key, pvalue in zip(pvalues, pvalue_corr):
        if pvalue >= alpha_multi:
            continue
        pvalues_final[key] = pvalue
    return pvalues_final
