""" Different kinds of plot styles

Style functions:

* :py:func:`set_trem2_split_spines`: Split the spines from the axis like the Trem2 paper
* :py:func:`set_trem2_prism`: Match the spine style of the Trem2 prism graphs

"""

# Imports
from matplotlib.axes import Axes

# Functions


def set_trem2_split_spines(ax: Axes) -> Axes:
    """ Weird split spine style """

    ax.set_clip_on(False)

    ax.yaxis.set_clip_on(False)

    ax.xaxis.set_clip_on(False)
    ax.xaxis.set_tick_params(which='both', length=0, pad=10)
    for loc, spine in ax.spines.items():
        if loc == 'bottom':
            spine.set_position(('outward', 20))
        if loc == 'left':
            spine.set_position(('outward', 20))
    ax.xaxis.set_ticks_position('bottom')
    return ax


def set_trem2_prism(ax: Axes) -> Axes:
    """ Match the prism style """

    ax.set_clip_on(False)

    ax.yaxis.set_clip_on(False)
    ax.yaxis.set_tick_params(which='both', length=10, pad=5)

    ax.xaxis.set_clip_on(False)
    ax.xaxis.set_tick_params(which='both', length=10, pad=5)
    return ax
