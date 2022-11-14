Introduction
============

This repository contains code and data to reproduce the image analysis results
in the 2022 ATV:Trem2 paper.

Installing
----------

The code should be installed into a fresh python 3.10 virtual environment:

.. code-block:: bash

    python3.10 -m venv path/to/env
    source path/to/env/bin/activate
    python3.10 -m pip install --upgrade pip
    python3.10 -m pip install -r requirements.txt

This package has been tested on OS X 10.15 (Catalina), but may work on other
operating systems with minor modifications.

Generating the Morphology Plots
-------------------------------

The morphology plots can be generated with the script ``plot_seg_stats.py`` in
the ``scripts`` directory:

.. code-block:: bash

    cd scripts
    ./plot_seg_stats.py morpho

All plots will be generated under the ``plots/morpho`` directory:

* Ext Figure 2B - ``single_cell_plots/atv_trem2-umap-category.png``
* Ext Figure 2C - ``single_cell_plots/cluster_counts/atv_trem2-category_cluster1_percent.png``
* Ext Figure 2D - ``single_cell_plots/cluster_volcanos/atv_trem2-volcano-labelkmeansumap1.png``
* Ext Figure 2E - ``single_cell_plots/atv_trem2-category_heatmap.png``

.. note:: Plot appearance may change slightly due to the random nature of the UMAP algorithm, but the resulting clusters should be stable from run to run.

CD74 Intensity Plots
--------------------

The CD74 intensity plots can be generated with the script ``plot_seg_stats.py`` in
the ``scripts`` directory:

.. code-block:: bash

    $ ./plot_seg_stats.py cd74

Results will be found under ``plots/int_cd74/``:

* Ext Figure 2G - ``per_animal_plots/atv_trem2-mean_normintensitymean_ch_4.png``

Axl Intensity Plots
-------------------

The Axl intensity plots can be generated with the script ``plot_seg_stats.py`` in
the ``scripts`` directory:

.. code-block:: bash

    $ ./plot_seg_stats.py axl

Results will be found under ``plots/int_axl/``:

* Ext Figure 2I - ``per_animal_plots/atv_trem2-mean_normintensitymean_ch_3.png``
