""" Utilities for making pretty plots

Plot functions:

* :py:func:`plot_image_mip`: Make a Maximum intensity plot of a single channel volume
* :py:func:`plot_label_mip`: Make a Maximum intensity plot of a label volume
* :py:func:`plot_boxes`: Make styled boxplots, barplots, and lineplots

Plot decoration functions:

* :py:func:`add_scalebar`: Add a scalebar to a plot
* :py:func:`add_colorbar`: Add a colorbar to an image plot

Utility functions:

* :py:func:`calc_mip`: Calculate the Maximum intensity plot of an image
* :py:func:`calc_scalebar_len`: Calculate the scalebar size
* :py:func:`get_label_cmap`: Get a colormap for label data, where <= 0 is black

"""

# Imports
import pathlib
import itertools
from typing import Tuple, Optional, Callable, List, Dict, Union

# 3rd party
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib.image import AxesImage
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Our own imports
from . import stat_utils
from .plot_style import set_trem2_prism

# Plot functions


def plot_boxes(df: pd.DataFrame,
               var_name: str = 'Distance',
               value_name: str = 'Value',
               hue_var_name: Optional[str] = None,
               xlabel: str = '',
               ylabel: str = '',
               xticklabels: List[str] = None,
               order: List[str] = None,
               hue_order: Optional[List[str]] = None,
               ignore_var_levels: Optional[List[str]] = None,
               plotfile: Optional[pathlib.Path] = None,
               suffix: str = '.pdf',
               palette: str = 'Set1',
               ylim: Optional[Tuple[float]] = None,
               xlim: Optional[Tuple[float]] = None,
               boxplot_width: float = 0.6,
               linewidth: float = 2.0,
               fliers_size: float = 20,
               figsize: Union[str, float, Tuple[float]] = 'auto',
               figsize_y: float = 12.0,
               figsize_x: float = 2.0,
               xticklabel_rotation: float = 0.0,
               title: Optional[str] = None,
               violin_bw: Union[float, str] = 'scott',
               err_bar_join: bool = True,
               err_bar_ci: float = 95.0,
               showfliers: bool = True,
               plot_style: str = 'box',
               pvalue_style: str = 'above',
               pvalue_comparisons: Optional[List[Tuple[int]]] = None,
               pvalues: Optional[Dict[str, float]] = None,
               hide_legend: bool = True,
               edge_color: str = '#C2C2C2',
               fill_color: str = '#E9E9E9',
               capsize: float = 0.2,
               min_samples_per_bin: int = 1,
               axvline: Optional[List[float]] = None,
               axvline_kwargs: Optional[Dict[str, object]] = None,
               axhline: Optional[List[float]] = None,
               axhline_kwargs: Optional[Dict[str, object]] = None):
    """ Make a boxplot for the comparison

    :param DataFrame df:
        A dataframe with columns for x, y, and possibly hue
    :param str var_name:
        The name of the categorical variable
    :param str value_name:
        The name for the value columns
    :param str hue_var_name:
        If not None, the name for the hue categorical column
    :param str xlabel:
        The label for the x-axis
    :param str ylabel:
        The label for the y-axis
    :param list[str] xticklabels:
        Labels for the x ticks
    :param list[tuple[int]] pvalue_comparisons:
        If not None, the list of (left, right) comparisons to run
    :param union[str, float] violin_bw:
        Either the name of a reference rule or the scale factor to use when
        computing the kernel bandwidth. ('scott', 'silverman', float)
    :param bool err_bar_ci:
        Width of the confidence intervals for the error plots, or 'sd' to draw standard deviations
    :param bool err_bar_join:
        If True, join the error bar plots with lines
    :param list[float] axvline:
        If not None, a list of x-positions for vertical lines to add to the plot
    :param dict[str, object] axvline_kwargs:
        If not None, a dictionary of additional kwargs to pass to the axvline function
    :param list[float] axhline:
        If not None, a list of y-positions for horizontal lines to add to the plot
    :param dict[str, object] axhline_kwargs:
        If not None, a dictionary of additional kwargs to pass to the axhline function
    """
    if ignore_var_levels is None:
        ignore_var_levels = []
    unique_levels = {level for level in set(np.unique(df[var_name])) if level not in ignore_var_levels}
    if order is None:
        order = list(sorted(set(np.unique(df[var_name]))))
    if set(order) < set(unique_levels):
        raise ValueError(f'Got "{var_name}" order {order} but have levels {unique_levels}')
    total_columns = len(order)
    if hue_var_name is not None:
        if hue_order is None:
            hue_order = list(sorted(set(np.unique(df[hue_var_name]))))
        unique_levels = set(np.unique(df[hue_var_name]))
        if set(hue_order) < set(unique_levels):
            raise ValueError(f'Got "{hue_var_name}" order {hue_order} but have levels {unique_levels}')
        total_columns *= len(hue_order)

    if plot_style in ('box', 'boxes', 'violin', 'violins', 'bar', 'bars'):
        figsize_x = figsize_x*total_columns
    elif plot_style in ('line', 'lines', 'err', 'err_bar', 'err_bars'):
        if err_bar_join or hue_var_name is not None:
            figsize_x = figsize_y
        else:
            figsize_x = figsize_x*total_columns
    else:
        raise KeyError(f'Unknown plot_style "{plot_style}"')

    # Scale figures by the number of categories
    if figsize == 'auto':
        figsize = (figsize_x, figsize_y)
    elif isinstance(figsize, (int, float)):
        figsize = (figsize_x, figsize)

    if fill_color in (None, '') and hue_var_name is not None:
        color_kwargs = {'palette': palette}
    elif fill_color in (None, ''):
        color_kwargs = {'color': '#E9E9E9'}
    else:
        color_kwargs = {'color': fill_color}

    # Zero out any bins that don't meet our sample requirement
    df = df.copy()
    if min_samples_per_bin > 1:
        if hue_order is None:
            for var_cat in order:
                mask = df[var_name] == var_cat
                if np.sum(mask) < min_samples_per_bin:
                    print(f'Zeroing level "{var_cat}" in "{var_name}"')
                    df.loc[mask, value_name] = np.nan
        else:
            for var_cat, hue_cat in itertools.product(order, hue_order):
                mask = (df[var_name] == var_cat) & (df[hue_var_name] == hue_cat)
                if np.sum(mask) < min_samples_per_bin:
                    print(f'Zeroing level "{var_cat}", "{hue_cat}" in "{var_name}", "{hue_var_name}"')
                    df.loc[mask, value_name] = np.nan

    # Plot the differences
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):
        if plot_style in ('box', 'boxes'):
            sns.boxplot(data=df, x=var_name, hue=hue_var_name, y=value_name, ax=ax,
                        order=order, showfliers=False, hue_order=hue_order,
                        linewidth=linewidth, width=boxplot_width,
                        boxprops={'edgecolor': edge_color},
                        whiskerprops={'color': edge_color},
                        medianprops={'color': edge_color},
                        capprops={'color': edge_color},
                        **color_kwargs)
        elif plot_style in ('violin', 'violins'):
            sns.violinplot(data=df, x=var_name, hue=hue_var_name, y=value_name, ax=ax,
                           order=order, showfliers=False,
                           hue_order=hue_order,
                           split=False, bw=violin_bw,
                           linewidth=linewidth, width=boxplot_width,
                           **color_kwargs)
        elif plot_style in ('bar', 'bars'):
            if fill_color in (None, '') and hue_var_name is not None:
                color_kwargs = {'palette': palette}
            elif fill_color in (None, ''):
                color_kwargs = {'facecolor': '#E9E9E9'}
            else:
                color_kwargs = {'facecolor': fill_color}
            sns.barplot(data=df, x=var_name, hue=hue_var_name, y=value_name, ax=ax,
                        order=order, hue_order=hue_order, dodge=False,
                        linewidth=linewidth, edgecolor=edge_color,
                        errorbar=('ci', err_bar_ci),
                        errwidth=linewidth,
                        capsize=capsize,
                        error_kw={
                            'ecolor': edge_color,
                            'capsize': capsize,
                            'elinewidth': linewidth,
                        },
                        **color_kwargs)

            # Stupid hack to fix the boxplot width
            locs = ax.get_xticks()
            for i, patch in enumerate(ax.patches):
                # Change the bar width
                patch.set_width(boxplot_width)
                # Then recenter the bar
                patch.set_x(locs[i] - (boxplot_width * .5))
        elif plot_style in ('line', 'lines'):
            sns.lineplot(data=df, x=var_name, hue=hue_var_name, y=value_name, ax=ax,
                         hue_order=hue_order, err_style='band',
                         errorbar=('ci', err_bar_ci),
                         linewidth=linewidth*2, palette=palette)
        elif plot_style in ('err', 'err_bar', 'err_bars'):
            if hue_var_name is None:
                color_kwargs = {'color': 'black'}
            else:
                color_kwargs = {'palette': palette}
            sns.pointplot(data=df, x=var_name, hue=hue_var_name, y=value_name, ax=ax,
                          order=order, hue_order=hue_order, dodge=False, errwidth=linewidth*2, scale=0.5,
                          capsize=capsize, linewidth=linewidth*2, join=err_bar_join,
                          errorbar=('ci', err_bar_ci),
                          markers='_', linestyles='-', **color_kwargs)
        else:
            raise KeyError(f'Unknown plot_style "{plot_style}"')

        # Plot the raw observations
        if showfliers:
            if hue_var_name is not None:
                color_kwargs = {'palette': palette}
            else:
                color_kwargs = {}
            sns.stripplot(data=df, x=var_name, y=value_name, hue=hue_var_name, dodge=True,
                          order=order, hue_order=hue_order, size=fliers_size,
                          edgecolor=edge_color, linewidth=linewidth, clip_on=False,
                          **color_kwargs)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    # Annotate with significance bars
    if pvalue_comparisons is not None:
        values = []
        for cat_key in order:
            values.append(df[df[var_name] == cat_key][value_name].values)
        pvalues = stat_utils.calc_pvalues(values, comparisons=pvalue_comparisons)

    if pvalues is not None:
        if hue_var_name is not None:
            raise NotImplementedError('P-values for subplots not yet supported')
        stat_utils.add_pvalues_to_plot(ax=ax, pvalues=pvalues, style=pvalue_style, ypct=0.98)

    # Add vertical lines to the plot
    if axvline is not None:
        if isinstance(axvline, (int, float, str)):
            axvline = [axvline]
        axvline = [float(v) for v in axvline]
        if axvline_kwargs is None:
            axvline_kwargs = {}
        default_kwargs = {'ymin': 0.0, 'ymax': 1.0, 'linewidth': linewidth}
        for key, val in default_kwargs.items():
            if axvline_kwargs.get(key) is None:
                axvline_kwargs[key] = val
        for xcoord in axvline:
            ax.axvline(x=xcoord, **axvline_kwargs)

    # Add horizontal lines to the plot
    if axhline is not None:
        if isinstance(axhline, (int, float, str)):
            axhline = [axhline]
        axhline = [float(v) for v in axhline]
        if axhline_kwargs is None:
            axhline_kwargs = {}
        default_kwargs = {'xmin': 0.0, 'xmax': 1.0, 'linewidth': linewidth}
        for key, val in default_kwargs.items():
            if axhline_kwargs.get(key) is None:
                axhline_kwargs[key] = val
        for ycoord in axhline:
            ax.axhline(y=ycoord, **axhline_kwargs)

    # Fix up the x and y axes
    ax = set_trem2_prism(ax)

    if title is None:
        ax.set_title('')
    else:
        ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Move the xticklabels around
    ax.set_xlabel(xlabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if abs(xticklabel_rotation) > 1e-5:
        if xticklabel_rotation < 0.0:
            horizontalalignment = 'left'
        else:
            horizontalalignment = 'right'

        plt.setp(ax.get_xticklabels(),
                 rotation=xticklabel_rotation,
                 horizontalalignment=horizontalalignment,
                 rotation_mode="anchor")

    # Clean up the legend
    legend = ax.get_legend()
    if hide_legend and legend is not None:
        legend.remove()

    plt.tight_layout()
    if plotfile is None:
        plt.show()
    else:
        plotfile = pathlib.Path(plotfile)
        plotfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(plotfile.parent / f'{plotfile.stem}{suffix}', transparent=True)
        plt.close()


def plot_image_mip(block: np.ndarray,
                   plotfile: Optional[pathlib.Path] = None,
                   midplane: Optional[int] = None,
                   num_slices: int = 10,
                   cmap: str = 'gray',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   pct_min: float = 2.0,
                   pct_max: float = 98.0,
                   reduce_fxn: Callable = np.max):
    """ Plot a MIP of the current image data

    :param ndarray block:
        The 3D z, y, x volume to plot
    :param Path plotfile:
        If not None, the file to write the plot to
    :param int midplane:
        Which slice index to use as the middle of the MIP
    :param int num_slices:
        How many slices to use in the MIP
    :param str cmap:
        Colormap to use for the resulting slices
    :param float vmin:
        If not None, the minimum value to plot (lower values will be black)
    :param float vmax:
        If not None, the maximum value to plot (higher values will be white)
    :param float pct_min:
        The percentile minimum to plot if vmin isn't given (out of 100, so 25.0 is the 25th percentile)
    :param float pct_max:
        The percentile maximum to plot if vmax isn't given (out of 100, so 75.0 is the 75th percentile)
    :param callable reduce_fxn:
        The projection function to call to reduce the volume to 2D
    """
    mip = calc_mip(
        block, midplane=midplane, num_slices=num_slices,
        reduce_fxn=reduce_fxn)
    rows, cols = mip.shape

    # Make the bounds of the MIP make sense
    if vmin is None:
        vmin = np.percentile(mip, pct_min)
    if vmax is None:
        vmax = np.percentile(mip, pct_max)

    aspect_ratio = cols/rows

    fig, ax = plt.subplots(1, 1, figsize=(8, 8*aspect_ratio))
    ax.imshow(mip, cmap=cmap, vmin=vmin, vmax=vmax,
              aspect='equal', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if plotfile is None:
        plt.show()
    else:
        plotfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(plotfile, transparent=True)
        plt.close()


def plot_label_mip(block: np.ndarray,
                   plotfile: Optional[pathlib.Path] = None,
                   midplane: Optional[int] = None,
                   num_slices: int = 10,
                   cmap: str = 'tab20c',
                   reduce_fxn: Callable = np.max,
                   add_label_numbers: bool = False,
                   min_label_size: int = 500,
                   fontsize: float = 10):
    """ Plot a MIP of the current label data

    :param ndarray block:
        The 3D z, y, x volume to plot
    :param Path plotfile:
        If not None, the file to write the plot to
    :param int midplane:
        Which slice index to use as the middle of the MIP
    :param int num_slices:
        How many slices to use in the MIP
    :param str cmap:
        Colormap to use for the resulting label image
    :param callable reduce_fxn:
        The projection function to call to reduce the volume to 2D
    """
    mip = calc_mip(
        block, midplane=midplane, num_slices=num_slices,
        reduce_fxn=reduce_fxn)
    rows, cols = mip.shape
    aspect_ratio = cols/rows

    cmap, norm = get_label_cmap(mip, cmap=cmap)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8*aspect_ratio))
    ax.imshow(mip, cmap=cmap, norm=norm, interpolation='nearest',
              origin='lower', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if add_label_numbers:
        yy, xx = np.mgrid[:rows, :cols]
        for i, level in enumerate(np.unique(mip)):
            if level == 0:
                continue
            mask = mip == level
            if np.sum(mask) < min_label_size:
                continue

            cx = np.mean(xx[mask])
            cy = np.mean(yy[mask])

            color = cmap(i)

            # FIXME: Switch text between black and white based on palette color
            ax.text(cx, cy, str(level),
                    size=fontsize,
                    color='#ffffff', weight='bold',
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox={
                        'boxstyle': "square",
                        'edgecolor': '#ffffff',
                        'facecolor': color,
                    })

    plt.tight_layout()

    if plotfile is None:
        plt.show()
    else:
        plotfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(plotfile, transparent=True)
        plt.close()

# Decoration functions


def add_scalebar(ax: plt.Axes, img: np.ndarray,
                 x_scale: Optional[float] = None,
                 scalebar_len: Optional[float] = None,
                 bar_len: Optional[float] = None,
                 color: str = 'w') -> plt.Axes:
    """ Add a scalebar to an image

    :param Axes ax:
        The axis object
    :param ndarray img:
        The image to plot
    :param float x_scale:
        If not None, scale of the x direction in px/um
    :param float scalebar_len:
        If not None, the length of the bar in pixel space
    :param float bar_len:
        If not None, the length of the bar in real space
    :param str color:
        The color for the scalebar and scalebar font
    :returns:
        The modified Axes object
    """
    rows, cols = img.shape[:2]
    if scalebar_len is None or bar_len is None:
        bar_len, scalebar_len = calc_scalebar_len(cols, x_scale)

    ax.plot([cols*0.9 - scalebar_len, cols*0.9],
            [rows*0.95, rows*0.95], linewidth=2, color=color)
    ax.text(cols*0.9 - scalebar_len/2, rows*0.92, f'${bar_len:0.0f} \\mu m$',
            color=color, horizontalalignment='center', verticalalignment='center',
            fontdict={'family': 'Arial', 'weight': 'bold', 'size': 20})
    return ax


def add_colorbar(ax: plt.Axes, im: AxesImage,
                 side: str = 'right',
                 size: str = '5%',
                 pad: str = '5%',
                 label: str = '') -> plt.Axes:
    """ Add a colorbar to an axis

    Add a colorbar with default limits to the right of the image:

    .. code-block:: python

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(image, cmap='viridis')
        add_colorbar(ax, im)

    :param Axes ax:
        The axis to add a colorbar to
    :param AxesImage im:
        The object returned by plt.imshow()
    :param str side:
        Which side to add the axis to ('top', 'left', 'bottom', 'right')
    :param str size:
        How big an axis to add, as a fraction of the figure size
    :param str pad:
        How much padding to add between the current axis and this new axis
    :param str label:
        The label for the colorbar
    :returns:
        The axis containing the colorbar
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size=size, pad=pad)
    plt.colorbar(im, cax=cax, label=label)
    return cax

# Utilities


def calc_mip(block: np.ndarray,
             midplane: Optional[int] = None,
             num_slices: int = 10,
             reduce_fxn: Callable = np.max) -> np.ndarray:
    """ Calculate the maximum intensity projection

    :param ndarray block:
        The z, y, x block to calculate a MIP for
    :param int midplane:
        If not None, the midplane of the image
    :param int num_slices:
        Number of slices to calculate the MIP over
    :param Callable reduce_fxn:
        A numpy function to use to reduce the axis (default: np.max)
    :returns:
        The y, x MIP for the block, generated in by the reduce function
    """

    # FIXME: Make this work with an axis argument
    slices, cols, rows = block.shape
    if midplane is None:
        midplane = slices//2

    slice_st = max([0, midplane - num_slices//2])
    slice_ed = min([slices, slice_st + num_slices])

    # Volumes are stored z, y, x
    return reduce_fxn(block[slice_st:slice_ed, :, :], axis=0)


def calc_scalebar_len(cols: int, x_scale: float,
                      bar_len: Optional[float] = None,
                      bar_pct_min: float = 0.1,
                      bar_pct_max: float = 0.3) -> Tuple[float]:
    """ Calculate the length for a scalebar

    :param int cols:
        The number of image columns (width)
    :param float x_scale:
        The conversion factor in px/um
    :param float bar_len:
        If not None, the desired bar length in um
    :returns:
        A tuple of (bar_len, scale_bar_len) the first in um, the second in px
    """
    # Work out the length (in um) of the whole image
    image_len = cols/x_scale

    # Try to autogenerate a "nice" length (in um)
    if bar_len is None:
        bar_min_len = image_len * bar_pct_min
        bar_max_len = image_len * bar_pct_max
        pow10 = np.floor(np.log10(bar_min_len))

        # Use a set of "nice" values, in order of "niceness"
        try_values = np.array([1.0, 5.0, 10.0, 2.5, 7.5,
                               2.0, 4.0, 6.0, 8.0,
                               3.0, 7.0, 9.0])
        try_values = try_values*10**pow10
        try_mask = np.logical_and(try_values >= bar_min_len, try_values <= bar_max_len)

        # We should be good here, but just in case
        if not np.any(try_mask):
            bar_len = np.mean([bar_min_len, bar_max_len])
        else:
            bar_len = try_values[try_mask]
            bar_len = bar_len[0]
    # Scale the bar to the image width (in px)
    return bar_len, bar_len * x_scale


def get_label_cmap(data: np.ndarray, cmap: str = 'Set1'):
    """ Get a labeled colormap, with 0 as a true black

    :param ndarray data:
        The data to get labels for
    :param str cmap:
        The original matplotlib colormap
    :returns:
        A Tuple of cmap, norm for plotting in plt.imshow
    """

    cmap = mplcm.get_cmap(cmap)
    cmap_colors = []
    unique_levels = list(sorted(np.unique(data)))
    bounds = [unique_levels[0] - 0.5]

    for i, level in enumerate(unique_levels):
        if i == 0:
            cmap_colors.append((0, 0, 0))
        else:
            cmap_colors.append(cmap((i-1) % cmap.N))
        bounds.append(level+0.5)
    assert len(bounds) == len(cmap_colors) + 1

    cmap = colors.ListedColormap(cmap_colors)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm
