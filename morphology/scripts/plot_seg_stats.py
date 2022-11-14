#!/usr/bin/env python3

""" Generate the final plots for single cell microglial analysis

Generate the morphology plots (Ext Figure 2 B-E).

.. code-block:: bash

    $ ./plot_seg_stats.py morpho

Results will be found under ``plots/morpho/``:

* Ext Figure 2B - ``single_cell_plots/atv_trem2-umap-category.png``
* Ext Figure 2C - ``single_cell_plots/cluster_counts/atv_trem2-category_cluster1_percent.png``
* Ext Figure 2D - ``single_cell_plots/cluster_volcanos/atv_trem2-volcano-labelkmeansumap1.png``
* Ext Figure 2E - ``single_cell_plots/atv_trem2-category_heatmap.png``

Generate the CD74 intensity plots (Ext Figure 2 G)

.. code-block:: bash

    $ ./plot_seg_stats.py cd74

Results will be found under ``plots/int_cd74/``:

* Ext Figure 2G - ``per_animal_plots/atv_trem2-mean_normintensitymean_ch_4.png``

Generate the AXL intensity plots (Ext Figure 2 I)

.. code-block:: bash

    $ ./plot_seg_stats.py axl

Results will be found under ``plots/int_axl/``:

* Ext Figure 2I - ``per_animal_plots/atv_trem2-mean_normintensitymean_ch_3.png``

Each of these runs the corresponding plot function:

* :py:func:`plot_morphology_single_cell` - Mophology analysis
* :py:func:`plot_cd74_single_cell` - CD74 analysis
* :py:func:`plot_axl_single_cell` - Axl analysis

Note that plot appearance may change slightly due to the random nature of the
UMAP algorithm, but the resulting clusters should be stable from run to run.

"""

# Imports
import re
import sys
import shutil
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
if str(THISDIR.parent) not in sys.path:
    sys.path.insert(0, str(THISDIR.parent))

# 3rd party
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Our own imports
from atv_trem2_morpho import qc_utils, plot_consts
from atv_trem2_morpho.sc_pipeline import run_single_cell_pipeline

# Constants

reNORM = re.compile(r'[^0-9a-z]+', re.IGNORECASE)
SUFFIX = '.png'

# Functions


def parse_lpl_treatment(filename: str) -> pd.Series:
    """ Parse the treatment from the filename """

    re_sections = [
        re.compile(r'''^
            lng_[0-9]+_mnf_[0-9]+_d(ay)?(?P<timepoint>[0-9]+)_(?P<treatment>(4d9)|(rsv))_(?P<animal_id>[0-9]+)_[0-9]+
        $''', re.IGNORECASE | re.VERBOSE),
    ]
    for re_section in re_sections:
        match = re_section.match(filename)
        if not match:
            continue
        timepoint = int(match.group('timepoint'))
        treatment = match.group('treatment')
        treatment = f'ATV:{treatment.upper()}'
        animal_id = int(match.group('animal_id'))
        return pd.Series({
            'Animal ID': animal_id,
            'Timepoint': timepoint,
            'Treatment': treatment,
            'Category': f'{treatment}/{timepoint} day',
        })
    raise ValueError(f'Cannot parse input file: "{filename}"')


def parse_axl_treatment(filename: str) -> pd.Series:
    """ Parse the animal id """
    reFILENAME = re.compile(r'''^
        lng_[0-9]+_mnf_[0-9]+_nh(?P<animal_id>[0-9]+)_[0-9]+
    $''', re.IGNORECASE | re.VERBOSE)
    match = reFILENAME.match(filename)
    if not match:
        print(filename)
        assert False
    animal_id = match.group('animal_id')
    return pd.Series({
        'Animal ID': f'NH{int(animal_id):02d}',
    })


def merge_categories(rec: pd.Series) -> str:
    """ Merge the categories into a single set

    :param Series rec:
        The record to merge
    :returns:
        The merged category
    """

    group = rec['Group']
    timepoint = rec['Time']

    return f'{group}/{timepoint} day'


def load_and_qc_data(indir: pathlib.Path,
                     iba1_channel: int = 3) -> pd.DataFrame:
    """ Load and QC the data

    :param Path indir:
        Directory to load all the metadata from
    :returns:
        The merged and QC'ed data frame
    """

    # Read the raw surface stats in
    iba1_stats_df = pd.read_excel(indir / f'stats_segment_merge_ch={iba1_channel}.xlsx')
    iba1_image_df = pd.read_excel(indir / 'stats_image_merge.xlsx')

    # Read the study info in
    study_df = pd.read_excel(indir / 'study_ids.xlsx',
                             sheet_name='Sheet1')

    # Read the QC data
    qc_df = pd.read_excel(indir / 'segment_qc.xlsx')
    qc_df = qc_df.rename(columns={'Prefix': 'File Prefix'})
    qc_df['Status'] = qc_df['Status'].str.lower().str.strip()
    qc_df['Animal ID'] = qc_df['Animal ID'].str.upper().str.strip()

    # Load and clean the study table
    study_df['Animal ID'] = study_df['Animal ID'].str.lower().str.strip()
    study_df['Time'] = study_df['Time'].map(lambda x: int(x.split(' day')[0]))
    study_columns = [
        'Animal ID', 'Time', 'Group', 'Batch',
    ]
    study_df = study_df[study_columns]

    # Now drop those bad tiles from the full stats
    iba1_stats_df = qc_utils.qc_stats_df(
        stats_df=iba1_stats_df, qc_df=qc_df, study_df=study_df,
        min_size=500, max_size=5000, size_column='Volume',
        check_animal_ids=True,
        remove_batch_effect=True, good_qc_statuses=['ok', 'good'])
    iba1_image_df = qc_utils.qc_stats_df(
        stats_df=iba1_image_df, qc_df=qc_df, study_df=study_df,
        size_column=None,
        check_animal_ids=True,
        remove_batch_effect=True, good_qc_statuses=['ok', 'good'])

    # Fuse the category definitions for easy plotting
    iba1_stats_df['Category'] = iba1_stats_df[['Group', 'Time']].agg(merge_categories, axis=1)
    iba1_image_df['Category'] = iba1_image_df[['Group', 'Time']].agg(merge_categories, axis=1)
    return iba1_stats_df, iba1_image_df

# Main functions


@plt.rc_context(plot_consts.RC_PARAMS_POSTER)
def plot_morphology_single_cell(suffix: str = SUFFIX):
    """ Plot the single cell analysis for Dan's Edu data

    :param Path indir:
        The directory with the segmentations
    :param str suffix:
        The suffix to save the plots under
    :param str study:
        Which set of study parameters to load
    """
    indir = THISDIR.parent / 'data' / 'morpho'
    if not indir.is_dir():
        raise OSError(f'Expected morphology data at "{indir}"')

    iba1_stats_df, iba1_image_df = load_and_qc_data(indir)

    # Plots for the Trem2 single cell
    plotdir = THISDIR.parent / 'plots' / 'morpho'
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    order = [
        'ATV:RSV/1 day',
        'ATV:4D9/1 day',
        'ATV:4D9/7 day',
        'ATV:4D9/14 day',
        'ATV:4D9/28 day',
    ]
    channel_names = {
        'Ch=1': 'DAPI',
        'Ch=2': 'Edu',
        'Ch=3': 'Iba1',
    }

    bad_columns = [
        'Block', 'Section', 'Region', 'File Prefix', 'Dir Prefix',
        'Filename', 'Channel', 'Surface',
        'Status', 'Time', 'Group', 'Batch',
    ]
    drop_columns = [
        'IntensityShellNumVoxels_Ch=3',
        'IntensityCoreNumVoxels_Ch=3',
        'SurfaceRatio', 'ConvexVolumeRatio', 'NumVoxels', 'NumSurfaceVoxels',
    ]
    comparisons = [(0, 1), (0, 2), (0, 3), (0, 4)]
    sum_columns = [
        'PercentGliaSpherical',
        'PercentGliaElongated',
        'PercentGliaConvex',
        'PercentGliaSurface',
        'PercentGliaLarge',
    ]
    xticklabels = [
        'ATV:ISO', 'ATV:mTREM2 D1', 'ATV:mTREM2 D7', 'ATV:mTREM2 D14', 'ATV:mTREM2 D28',
    ]
    cluster_labels = {
        '0': 'Homeostatic',
        '1': 'Responsive',
    }
    ylims = {}

    # Select interesting features to plot on the volcano
    volcano_label_features = [
        'IntensityMean_Ch=3',
        'ConvexSurfaceAreaRatio',
        'HausdorffDim_Ch=3',
        'LacunaCoeff_Ch=3',
        # 'ShollCriticalLabels_Ch=3',
        'SkeletonMeanBranchLen',
        'PercentSurface',
        'SurfaceAreaVolume',
        'ShollCriticalBranches_Ch=3',
        'SkeletonNumShortBranches',
        # 'SkeletonNumBranches',
        # 'ShollNumLabels_Ch=3',
        # 'Volume',
        'LacunaRatio_Ch=3',
        'EllipseAspectRatio',
        'PercentConvex',
    ]

    # Copy steve's pallete style
    viridis_palette = sns.color_palette('viridis', 4)
    var_palette = sns.blend_palette([
        (0.76, 0.76, 0.76)
    ] + list(reversed(viridis_palette)))

    run_single_cell_pipeline(
        stats_df=iba1_stats_df,
        image_df=iba1_image_df,
        image_norm_method='div',
        outdir=plotdir,
        var_column='Category',
        var_palette=var_palette,
        animal_column='Animal ID',
        batch_column='Batch',
        prefix='atv_trem2-',
        suffix=suffix,
        ylims=ylims,
        cluster_labels=cluster_labels,
        channel_names=channel_names,
        order=order,
        xticklabels=xticklabels,
        xticklabel_rotation=45.0,
        drop_channels=[1, 2],  # Drop DAPI, Edu
        drop_columns=drop_columns,
        drop_intensity_features=False,
        drop_norm_intensity_features=True,
        drop_volume_features=False,
        drop_centile_features=False,
        drop_shell_core_features=False,
        bad_columns=bad_columns,
        sum_columns=sum_columns,
        num_reduction_components=10,
        num_kmeans_clusters=2,
        comparisons=comparisons,
        plot_style='bars',
        plot_err_capsize=0.3,
        reduction_type='pca',
        normalization_type='quantile',
        cluster_coord_column='umap',
        projection_coord_column='pca',
        projection_type='umap',
        projection_kwargs={
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
        },
        volcano_kwargs={
            'approach': '1-vs-rest',
            'label_features': volcano_label_features,
            'label_top_k': -1,
            'label_top_k_up': -1,
            'label_top_k_down': -1,
        },
    )


@plt.rc_context(plot_consts.RC_PARAMS_POSTER)
def plot_cd74_single_cell(suffix: str = SUFFIX):
    """ Plot the results for LPL and CD74 staining """

    indir = THISDIR.parent / 'data' / 'int_cd74'
    if not indir.is_dir():
        raise OSError(f'Expected CD74 intensity data at "{indir}"')

    iba1_image_df = pd.read_excel(indir / 'stats_image_merge.xlsx')
    iba1_stats_df = pd.read_excel(indir / 'stats_segment_merge_ch=2.xlsx')

    iba1_image_df['Batch'] = iba1_image_df['Dir Prefix'].map({
        'part1': 1,
        'part2': 2,
        'part3': 3,
    })
    iba1_stats_df['Batch'] = iba1_stats_df['Dir Prefix'].map({
        'part1': 1,
        'part2': 2,
        'part3': 3,
    })
    qc_df = pd.read_excel(indir / 'segment_qc.xlsx')

    order = [
        'ATV:RSV/1 day',
        'ATV:4D9/1 day',
        'ATV:4D9/7 day',
        'ATV:4D9/14 day',
        'ATV:4D9/28 day',
    ]
    channel_names = {
        'Ch=1': 'DAPI',
        'Ch=2': 'Iba1',
        'Ch=3': 'LPL',
        'Ch=4': 'CD74',
    }
    bad_columns = [
        'Block', 'Section', 'Region', 'File Prefix', 'Dir Prefix',
        'Filename', 'Channel', 'Surface',
        'Status', 'Time', 'Group', 'Batch',
    ]
    drop_columns = [
        'NumVoxels', 'NumSurfaceVoxels', 'Timepoint',
    ]
    comparisons = [(0, 1), (0, 2), (0, 3), (0, 4)]
    xticklabels = [
        'ATV:ISO', 'ATV:mTREM2 D1', 'ATV:mTREM2 D7', 'ATV:mTREM2 D14', 'ATV:mTREM2 D28',
    ]
    cluster_labels = {
        '0': 'Homeostatic',
        '1': 'Responsive',
    }
    ylims = {}

    study_df = qc_df['File Prefix'].apply(parse_lpl_treatment)
    qc_df['Animal ID'] = study_df['Animal ID']

    study_df = study_df.drop_duplicates(keep='first', ignore_index=True)

    # Now drop those bad tiles from the full stats
    iba1_stats_df = qc_utils.qc_stats_df(
        stats_df=iba1_stats_df, qc_df=qc_df, study_df=study_df,
        min_size=500, max_size=5000, size_column='Volume',
        check_animal_ids=True,
        remove_batch_effect=False, good_qc_statuses=['ok', 'good'])
    iba1_image_df = qc_utils.qc_stats_df(
        stats_df=iba1_image_df, qc_df=qc_df, study_df=study_df,
        size_column=None,
        check_animal_ids=True,
        remove_batch_effect=False, good_qc_statuses=['ok', 'good'])

    # Plots for the CD74 single cell
    plotdir = THISDIR.parent / 'plots' / 'int_cd74'
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    # Copy steve's pallete style
    viridis_palette = sns.color_palette('viridis', 4)
    var_palette = sns.blend_palette([
        (0.76, 0.76, 0.76)
    ] + list(reversed(viridis_palette)))

    iba1_stats_df = qc_utils.calc_extended_values(
        iba1_stats_df, image_df=iba1_image_df, norm_method='div')

    iba1_stats_df['IsCD74Positive'] = iba1_stats_df['NormIntensityMean_Ch=4'] > 2.0
    sum_columns = ['IsCD74Positive']

    ylabels = {
        'TotalCD74Positive': 'Total CD74+ Cells',
        'PercentCD74Positive': '% CD74+ Cells',
    }

    run_single_cell_pipeline(
        stats_df=iba1_stats_df,
        image_df=iba1_image_df,
        image_norm_method='div',
        sum_columns=sum_columns,
        outdir=plotdir,
        var_column='Category',
        var_palette=var_palette,
        animal_column='Animal ID',
        batch_column='Batch',
        prefix='atv_trem2-',
        suffix=suffix,
        skip_feature_plots=True,
        ylims=ylims,
        ylabels=ylabels,
        cluster_labels=cluster_labels,
        channel_names=channel_names,
        order=order,
        xticklabels=xticklabels,
        xticklabel_rotation=45.0,
        drop_channels=[1],  # Drop DAPI, Edu
        drop_columns=drop_columns,
        bad_columns=bad_columns,
        num_reduction_components=10,
        num_kmeans_clusters=2,
        comparisons=comparisons,
        plot_style='bars',
        plot_err_capsize=0.3,
        reduction_type='pca',
        normalization_type='quantile',
        cluster_coord_column='umap',
        projection_coord_column='pca',
        projection_type='umap',
        projection_kwargs={
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
        },
    )


@plt.rc_context(plot_consts.RC_PARAMS_POSTER)
def plot_axl_single_cell(suffix: str = SUFFIX):
    """ Plot the results for P2RY12 and Axl staining """

    indir = THISDIR.parent / 'data' / 'int_axl'
    if not indir.is_dir():
        raise OSError(f'Expected CD74 intensity data at "{indir}"')

    iba1_image_df = pd.read_excel(indir / 'stats_image_merge.xlsx')
    iba1_stats_df = pd.read_excel(indir / 'stats_segment_merge_ch=2.xlsx')
    qc_df = pd.read_excel(indir / 'segment_qc.xlsx')

    order = [
        'ATV:RSV/1 day',
        'ATV:4D9/1 day',
    ]
    channel_names = {
        'Ch=1': 'P2RY12',
        'Ch=2': 'Iba1',
        'Ch=3': 'Axl',
    }
    bad_columns = [
        'Block', 'Section', 'Region', 'File Prefix', 'Dir Prefix',
        'Filename', 'Channel', 'Surface',
        'Status', 'Time', 'Group', 'Batch',
    ]
    drop_columns = [
        'NumVoxels', 'NumSurfaceVoxels', 'Timepoint',
    ]
    comparisons = [(0, 1)]
    xticklabels = [
        'ATV:ISO', 'ATV:mTREM2 D1'
    ]
    cluster_labels = {
        '0': 'Homeostatic',
        '1': 'Responsive',
    }
    ylims = {}

    res_df = qc_df['File Prefix'].apply(parse_axl_treatment)
    qc_df['Animal ID'] = res_df['Animal ID']

    study_df = pd.read_excel(indir / 'study_ids.xlsx')
    study_df['Animal ID'] = study_df['Animal ID'].map(lambda x: f'NH{int(x[2:]):02d}')
    study_df['Category'] = study_df['Group'].str.cat(study_df['Time'], sep='/')

    # Now drop those bad tiles from the full stats
    iba1_stats_df = qc_utils.qc_stats_df(
        stats_df=iba1_stats_df, qc_df=qc_df, study_df=study_df,
        min_size=500, max_size=5000, size_column='Volume',
        check_animal_ids=True,
        remove_batch_effect=False, good_qc_statuses=['ok', 'good'])
    iba1_image_df = qc_utils.qc_stats_df(
        stats_df=iba1_image_df, qc_df=qc_df, study_df=study_df,
        size_column=None,
        check_animal_ids=True,
        remove_batch_effect=False, good_qc_statuses=['ok', 'good'])

    # Plots for the Trem2 single cell
    plotdir = THISDIR.parent / 'plots' / 'int_axl'
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    # Copy steve's pallete style
    var_palette = sns.color_palette('viridis', 2)

    iba1_stats_df = qc_utils.calc_extended_values(
        iba1_stats_df, image_df=iba1_image_df, norm_method='div')

    iba1_stats_df['IsAxlPositive'] = iba1_stats_df['NormIntensityMean_Ch=3'] > 2.5
    sum_columns = ['IsAxlPositive']

    ylabels = {
        'TotalAxlPositive': 'Total Axl+ Cells',
        'PercentAxlPositive': '% Axl+ Cells',
    }

    run_single_cell_pipeline(
        stats_df=iba1_stats_df,
        image_df=iba1_image_df,
        image_norm_method='div',
        sum_columns=sum_columns,
        outdir=plotdir,
        var_column='Category',
        var_palette=var_palette,
        animal_column='Animal ID',
        batch_column='Batch',
        prefix='atv_trem2-',
        suffix=suffix,
        skip_feature_plots=True,
        ylims=ylims,
        ylabels=ylabels,
        cluster_labels=cluster_labels,
        channel_names=channel_names,
        order=order,
        xticklabels=xticklabels,
        xticklabel_rotation=45.0,
        drop_columns=drop_columns,
        bad_columns=bad_columns,
        num_reduction_components=10,
        num_kmeans_clusters=2,
        comparisons=comparisons,
        plot_style='bars',
        plot_err_capsize=0.3,
        reduction_type='pca',
        normalization_type='quantile',
        cluster_coord_column='umap',
        projection_coord_column='pca',
        projection_type='umap',
        projection_kwargs={
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
        },
    )

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('study', choices=(
                            'morpho', 'morphology',
                            'cd74', 'int_cd74',
                            'axl', 'int_axl',
                        ),
                        help='Which study to plot (one of "morpho", "cd74", "axl")')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    study = args.study.lower().strip().replace(' ', '_')

    if study in ('morpho', 'morphology'):
        plot_morphology_single_cell()
    elif study in ('cd74', 'int_cd74'):
        plot_cd74_single_cell()
    elif study in ('axl', 'int_axl'):
        plot_axl_single_cell()
    else:
        raise KeyError(f'Unknown study "{study}"')


if __name__ == '__main__':
    main()
