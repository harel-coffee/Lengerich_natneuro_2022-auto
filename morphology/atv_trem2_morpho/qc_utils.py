""" Tools for QC-ing and summarizing the data

Functions:

* :py:func:`qc_stats_df`: Merge the statistics, QC, and study data frames
* :py:func:`bin_by_rank`: Bin a population into equally sized bins using the data ranks
* :py:func:`bin_by_dist`: Bin a population into bins with different kinds of distance spacing
* :py:func:`calc_bin_edges`: Calculate the edges of bins for :py:func:`bin_by_dist`
* :py:func:`calc_extended_values`: Calculate additional values from the available values in the dataset

"""

# Imports
from typing import Optional, Union, Dict, List

# 3rd party
import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.neighbors import BallTree

# Functions


def format_section_id(section: Union[str, float]) -> str:
    """ Format the section id

    :param int/str section:
        The section id as a str/float/int
    :returns:
        A cleanly formatted section number as a string
    """
    if isinstance(section, (int, float)):
        if np.isnan(section):
            return ''
        return f's{int(section):02d}'
    if section.startswith('s'):
        section = section[1:]
    if section == '':
        return ''
    return f's{int(section):02d}'


def calc_extended_values(stats_df: pd.DataFrame,
                         image_df: Optional[pd.DataFrame] = None,
                         norm_method: str = 'div') -> pd.DataFrame:
    """ Calculate additonal values for the study data

    :param DataFrame stats_df:
        The stats database for the image
    :param DataFrame image_df:
        If not None, the image dataframe to calculate normalized intensity
    :param str norm_method:
        How to normalize image intensity values, one of 'sub' or 'div'
    :returns:
        A database extended with new values
    """
    # Calculate additional ratios
    if 'SurfaceArea' in stats_df and 'Volume' in stats_df:
        stats_df['SurfaceAreaVolume'] = stats_df['SurfaceArea'] / stats_df['Volume']

    # Calculate percentages
    if 'Volume' in stats_df and 'ConvexVolume' in stats_df:
        stats_df['PercentConvex'] = stats_df['Volume'] / stats_df['ConvexVolume'] * 100
    if 'NumSurfaceVoxels' in stats_df and 'NumVoxels' in stats_df:
        stats_df['PercentSurface'] = stats_df['NumSurfaceVoxels'] / stats_df['NumVoxels'] * 100
    if 'NumCoreVoxels' in stats_df and 'NumVoxels' in stats_df:
        stats_df['PercentCore'] = stats_df['NumCoreVoxels'] / stats_df['NumVoxels'] * 100
    if 'SkeletonVoxels' in stats_df and 'SkeletonNumBranchpoints' in stats_df:
        stats_df['PercentBranchVoxels'] = stats_df['SkeletonNumBranchpoints'] / stats_df['SkeletonVoxels'] * 100
    if 'SkeletonVoxels' in stats_df and 'NumVoxels' in stats_df:
        stats_df['PercentSkeleton'] = stats_df['SkeletonVoxels'] / stats_df['NumVoxels'] * 100

    # Don't try and run the per-image normalization
    if image_df is None:
        return stats_df

    # Make sure we didn't get any weird duplications in the image data
    if 'Dir Prefix' not in image_df.columns:
        image_df['Dir Prefix'] = ''
    if 'Dir Prefix' not in stats_df.columns:
        stats_df['Dir Prefix'] = ''

    image_keys = image_df['Dir Prefix'].str.cat(image_df['File Prefix'], sep='-')
    image_dups = image_keys.duplicated()

    if np.any(image_dups):
        raise ValueError(f'Got duplicated keys in image data: {image_keys[image_dups]}')

    # Merge the image volume values back in
    all_vol_columns = ['Dir Prefix', 'File Prefix']
    all_vol_columns.extend(c for c in image_df.columns
                           if c.startswith(('VolumeIntensityMean_', 'VolumeIntensityPct50_')))
    stats_df = stats_df.merge(image_df[all_vol_columns],
                              how='left', on=('Dir Prefix', 'File Prefix'),
                              validate='many_to_one').copy()

    # Create normalized intensity columns
    norm_stats_df = {}
    for column in stats_df.columns:
        if not column.startswith(('Intensity', 'IntensityCore', 'IntensityShell')):
            continue
        if '_Ch=' not in column:
            continue
        if 'NumVoxels' in column:
            continue
        if 'In_Ch=' in column:
            continue
        if f'Norm{column}' in stats_df.columns:
            continue

        channel_prefix, channel_num = column.rsplit('_Ch=', 1)
        channel_num = int(channel_num)
        if channel_prefix.endswith(('Min', 'Max', 'Mean', 'Std')) or 'Pct' in channel_prefix:
            vol_column = f'VolumeIntensityMean_Ch={channel_num}'
        else:
            raise KeyError(f'Unknown intensity value column: {column}')

        if norm_method.startswith('sub'):
            norm_stats_df[f'Norm{column}'] = stats_df[column] - stats_df[vol_column]
        elif norm_method.startswith('div'):
            norm_stats_df[f'Norm{column}'] = stats_df[column] / (stats_df[vol_column] + 1.0)
        else:
            raise KeyError(f'Unknown intensity normalization method "{norm_method}"')

    stats_df = pd.concat([stats_df, pd.DataFrame(norm_stats_df)], axis=1)

    # Remove the volumetric columns
    for column in all_vol_columns:
        if column in ('Dir Prefix', 'File Prefix'):
            continue
        del stats_df[column]
    return stats_df.copy()


def find_mutual_neighbors(left: np.ndarray, right: np.ndarray,
                          dist: Optional[np.ndarray] = None):
    """ Find mutual nearest neighbors

    :param ndarray left:
        The n x k array of neighbors returned by the left ``BallTree.query``
        on the right coordinates

        left_tree = BallTree(left_coords)
        left = left_tree.query(right_coords, k=1)

    :param ndarray right:
        The m x k array of neighbors returned by the right ``BallTree.query``
        on the left coordinates

        right_tree = BallTree(right_coords)
        right = right_tree.query(left_coords, k=1)

    :param ndarray dist:
        If not None, the n x k array of distances from left to right
    :returns:
        A p x 2 array of (right, left) pairs that are mutual neighbors
        If ``dist`` is passed, a p x 3 array of (right, left, dist) pairs
    """
    if dist is not None:
        assert dist.shape == left.shape
    neighbors = []
    for i, left_inds in enumerate(left):
        for j in left_inds:
            right_inds = right[j, :]
            mask = right_inds == i
            if np.any(mask):
                if dist is None:
                    neighbors.append((i, j))
                else:
                    neighbors.append((i, j, dist[i, mask][0]))
    if dist is None:
        return np.array(neighbors, dtype=[('right', 'int64'), ('left', 'int64')])
    else:
        return np.array(neighbors, dtype=[('right', 'int64'), ('left', 'int64'), ('dist', 'float64')])


def average_per_cell(delta: np.ndarray,
                     mnn: np.ndarray,
                     left_neighbors: np.ndarray,
                     left_dists: np.ndarray,
                     right_neighbors: np.ndarray,
                     right_dists: np.ndarray,
                     sigma: float = 0.1) -> np.ndarray:
    """ Implement a gaussian smoothed vector per cell

    :param ndarray delta:
        The array of differences between neighbors
    :param ndarray mnn:
        The set of right, left, dist triples for mutual nearest neighbors
    :param ndarray left_neighbors:
        For every element in left, these are the knn in right
    :param ndarray left_dists:
        For every element in left, these are the knn distances in right
    :param ndarray right_neighbors:
        For every element in right, these are the knn in left
    :param ndarray right_dists:
        For every element in right, these are the knn distances in left
    :param float sigma:
        The smoothing kernel applied to the gaussian function
    :returns:
        A vector of offsets to apply to every element in right
    """
    assert mnn.shape[0] == delta.shape[0]
    assert left_neighbors.shape == left_dists.shape
    assert right_neighbors.shape == right_dists.shape

    # These are the same shape as the right array
    delta_sum = np.zeros((left_dists.shape[0], delta.shape[1]), dtype=np.float64)
    delta_count = np.zeros((left_dists.shape[0], ), dtype=np.float64)

    left = mnn['left']  # Indexes into the left array, the same shape as right_neighbors
    right = mnn['right']  # Indexes into the right array, the same shape as left_neighbors

    for i in range(mnn.shape[0]):
        # right_mask = right == i
        #
        # delta_sum[i, :] += np.mean(delta[right_mask, :], axis=0)
        # delta_count[i] += 1.0
        #
        #
        #
        # print(i, right_mask, left_mask)
        # assert False
        #
        left_ind = left[i]
        right_ind = right[i]

        # For a given point in left, here are all the neighbors in right
        inds = right_neighbors[left_ind, :]
        dist = right_dists[left_ind, :]

        # Pull out the delta for this neighbor, and the distance
        key_delta = delta[i:i+1, :]

        key_ind = np.nonzero(inds == right_ind)
        assert len(key_ind) == 1
        key_ind = key_ind[0]

        # Need a real gaussian for scaling
        weight = np.exp(-0.5*(dist - dist[key_ind])**2/sigma**2)/sigma/np.sqrt(2*np.pi)
        weight = weight[:, np.newaxis]

        delta_sum[inds, :] += key_delta*weight
        delta_count[inds] += weight[:, 0]

    # Take however much weight we got, if it was less than a full sample
    delta_count[delta_count < 1] = 1.0
    return delta_sum / delta_count[:, np.newaxis]


def remove_batch_effect_mnn(labels: np.ndarray,
                            values: np.ndarray,
                            num_neighbors: int = 5,
                            sigma: float = 1.0) -> np.ndarray:
    """ Remove the batch effect from a set of values

    This implements the mutual nearest neighbors correction proposed in
    Haghverdi et al 2018

    :param ndarray labels:
        A 1D vector with labels for each batch
    :param values values:
        A 2D matrix with cells on the rows and values on the columns
    :param int num_neighbors:
        How many mutual neighbors to consider in the dataset
    :param float sigma:
        The smoothing kernel applied to the gaussian function (larger is smoother)
    """
    levels = list(sorted(np.unique(labels), key=lambda x: np.sum(labels == x), reverse=True))
    if len(levels) < 2:
        return values
    print(f'Got batch levels: {levels}')

    # Step 1 - Cosine normalization
    scales = np.linalg.norm(values, axis=1)[:, np.newaxis]
    values = values / scales

    # Step 2 - for each batch, find mutual nearest neighbors
    batches = [values[labels == label, :] for label in levels]
    left_batch = batches.pop(0)

    # Left batch is the currently integrated dataset
    # Right batch is the data to integrate
    while batches:
        left_tree = BallTree(left_batch)

        right_batch = batches.pop(0)
        right_tree = BallTree(right_batch)

        # Seach for nearest neighbors in each direction
        left_dists, left_neighbors = left_tree.query(right_batch, k=num_neighbors,
                                                     return_distance=True)
        right_dists, right_neighbors = right_tree.query(left_batch, k=num_neighbors,
                                                        return_distance=True)

        assert left_dists.shape[0] == right_batch.shape[0]
        assert left_neighbors.shape[0] == right_batch.shape[0]

        assert right_dists.shape[0] == left_batch.shape[0]
        assert right_neighbors.shape[0] == left_batch.shape[0]

        # Pairs of neighbors are our marker cells
        mnn = find_mutual_neighbors(left_neighbors, right_neighbors,
                                    dist=left_dists)

        # For each marker cell, get the delta vector in expression for each gene
        right_points, left_points = mnn['right'], mnn['left']
        delta = left_batch[left_points, :] - right_batch[right_points, :]

        # delta = average_per_cell(delta, mnn, left_neighbors, left_dists,
        #                          right_neighbors, right_dists,
        #                          sigma=sigma)
        # assert delta.shape == right_batch.shape
        # print(np.mean(delta, axis=0))
        delta = np.median(delta, axis=0)[np.newaxis, :]
        right_batch = right_batch + delta

        # Now the left dataset is the sum of the old left data and the newly corrected right
        left_batch = np.concatenate([left_batch, right_batch], axis=0)

    # Now left batch has all the data, but in the wrong order
    final_values = np.zeros_like(left_batch)
    idx = 0
    final_means = None
    for label in levels:
        mask = labels == label
        step = int(np.sum(mask))

        label_scale = scales[mask, :]
        label_values = left_batch[idx:idx+step, :]*label_scale

        if final_means is None:
            final_means = np.mean(label_values, axis=0)[np.newaxis, :]
            offset = np.zeros_like(final_means)
        else:
            offset = final_means - np.mean(label_values, axis=0)[np.newaxis, :]

        # Rescale the data back to real values
        final_values[mask, :] = label_values + offset
        idx += step

    return final_values


def remove_batch_effect_means(labels: np.ndarray,
                              values: np.ndarray,
                              mode: str = 'median') -> np.ndarray:
    """ Remove the batch effect from the data

    This method estimates a per-batch mean/std for each column and then equalizes
    them across samples

    :param ndarray labels:
        A 1D vector with labels for each batch
    :param values values:
        A 2D matrix with cells on the rows and values on the columns
    :param str mode:
        One of 'mean' or 'median' - which stat to correct with
    :returns:
        A batch corrected version of values
    """

    levels = list(sorted(np.unique(labels)))
    if len(levels) < 2:
        return values
    print(f'Got batch levels: {levels}')

    values = values.astype(np.float64)
    batch_intercept = []
    batch_scale = []
    batch_size = []

    for level in levels:
        mask = labels == level
        batch_values = values[mask, :]

        if mode == 'mean':
            batch_intercept.append(np.nanmean(batch_values, axis=0))
            batch_scale.append(np.nanstd(batch_values, axis=0))
        elif mode == 'median':
            p25, p50, p75 = np.nanpercentile(batch_values, [25, 50, 75], axis=0)
            batch_intercept.append(p50)
            batch_scale.append(p75 - p25)
        else:
            raise KeyError(f'Unknown rescaling mode: "{mode}"')

        batch_size.append(np.sum(mask))

    # Make the statistics similar to the size-weighted average over all batches
    target_intercept = np.average(batch_intercept, axis=0, weights=batch_size)
    target_scale = np.average(batch_scale, axis=0, weights=batch_size)

    final_values = np.zeros_like(values)
    for i, level in enumerate(levels):
        mask = labels == level

        intercept = batch_intercept[i]
        scale = batch_scale[i]

        # Center, scale then offset again
        batch_values = values[mask, :] - intercept[np.newaxis, :]
        batch_values *= target_scale[np.newaxis, :] / (scale[np.newaxis, :] + 1e-5)
        batch_values += target_intercept[np.newaxis, :]

        final_values[mask, :] = batch_values
    return final_values


def drop_na_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Drop columns which are all NA

    :param DataFrame df:
        The data frame to clean
    :returns:
        A new data frame with all empty/NA columns dropped
    """
    keep_columns = []
    drop_columns = []
    for column in df.columns:
        if np.all(df[column].isna()):
            drop_columns.append(column)
            continue
        if np.all(df[column].apply(str).str.upper().str.strip().isin(['NA', 'NAN', ''])):
            drop_columns.append(column)
            continue
        keep_columns.append(column)
    if drop_columns != []:
        print(f'Dropping NA columns: {drop_columns}')
    if len(keep_columns) < 1:
        raise ValueError(f'No columns to keep in df: {df.columns}')
    return df[keep_columns].copy()


def qc_stats_df(stats_df: pd.DataFrame,
                qc_df: Optional[pd.DataFrame] = None,
                study_df: Optional[pd.DataFrame] = None,
                min_size: Optional[int] = -1,
                max_size: Optional[int] = -1,
                size_column: Optional[str] = 'Volume',
                batch_column: str = 'Batch',
                timepoint_column: Optional[str] = None,
                var_column: Optional[str] = None,
                check_animal_ids: bool = False,
                remove_batch_effect: bool = False,
                good_qc_statuses: Optional[List[str]] = None) -> pd.DataFrame:
    """ QC and filter the stats data frame

    :param DataFrame stats_df:
        The dataframe with the actual stats
    :param DataFrame image_df:
        If not None, the overall image stats
    :param DataFrame qc_df:
        If not None, the dataframe with the QC column
    :param DataFrame study_df:
        The dataframe with the study metadata
    :param int min_size:
        If not None, minimum number of voxels in a surface
    :param int max_size:
        If not None, maximum number of voxels in a surface
    :param str size_column:
        Name of the column to use for size (default: 'Volume')
    :param bool check_animal_ids:
        If True, make sure that all Animal IDs survive QC
    :param bool remove_batch_effect:
        If True, remove the batch effect (requires a 'Batch' column)
    :param str batch_column:
        Name of the column to use for batch regression (default: 'Batch')
    :param list[str] good_qc_statuses:
        The list of values in the "Status" column that are considered "good"
    :returns:
        A cleaned and QC'ed data frame
    """
    if good_qc_statuses is None:
        good_qc_statuses = ['ok', 'okay', 'good']
    elif isinstance(good_qc_statuses, str):
        good_qc_statuses = [good_qc_statuses]

    # Clean up the Animal ID column
    stats_df = drop_na_columns(stats_df)
    if qc_df is not None:
        qc_df = drop_na_columns(qc_df)
    if study_df is not None:
        study_df = drop_na_columns(study_df)

    if 'Animal ID' in stats_df.columns:
        stats_df['Animal ID'] = stats_df['Animal ID'].apply(str).str.upper().str.strip()

    name_columns = ['Animal ID', 'Block', 'Section', 'Dir Prefix', 'File Prefix']
    name_columns = [n for n in name_columns if n in stats_df.columns]

    # If the block is non-unique or empty, just ignore it
    if 'Block' in name_columns and len(np.unique(stats_df['Block'])) < 2:
        name_columns.remove('Block')

    # Check the QC if we have it
    if qc_df is not None:
        if 'Prefix' in qc_df.columns:
            qc_df = qc_df.rename(columns={'Prefix': 'File Prefix'})
        if 'Status' not in qc_df.columns:
            raise ValueError(f'QC DataFrame should have a "Status" column, got: {qc_df.columns}')

        qc_df['Status'] = qc_df['Status'].str.lower().str.strip()
        if 'Animal ID' in qc_df:
            qc_df['Animal ID'] = qc_df['Animal ID'].apply(str).str.upper().str.strip()
        if 'Section' in qc_df:
            qc_df['Section'] = qc_df['Section'].map(format_section_id)

        okay_mask = qc_df['Status'].isin(good_qc_statuses)
        num_okay = np.sum(okay_mask)
        num_total = okay_mask.shape[0]
        print(f'Got {num_okay}/{num_total} good tiles ({num_okay/num_total:0.1%})')

        if 'Status' in stats_df.columns:
            del stats_df['Status']
        value_columns = ['Status']
        for extra_value_column in ['Animal ID', 'Section', 'Region', 'Batch']:
            if extra_value_column not in stats_df.columns and extra_value_column in qc_df.columns:
                value_columns.append(extra_value_column)
        stats_df = stats_df.merge(qc_df[name_columns + value_columns],
                                  on=name_columns, how='left')

        pre_animal_ids = set(np.unique(stats_df['Animal ID']))

        okay_mask = stats_df['Status'].isin(good_qc_statuses)
        stats_df = stats_df[okay_mask]

        post_animal_ids = set(np.unique(stats_df['Animal ID']))

        if post_animal_ids != pre_animal_ids:
            if check_animal_ids:
                raise ValueError(f'Lost some animal IDs during QC: {pre_animal_ids - post_animal_ids}')
            else:
                print(f'Lost some animal IDs during QC: {pre_animal_ids - post_animal_ids}')

    # Merge with the study data if we got some
    if study_df is not None:
        study_name_columns = ['Animal ID', 'Block', 'Section']
        study_df['Animal ID'] = study_df['Animal ID'].apply(str).str.upper().str.strip()
        study_name_columns = [c for c in study_name_columns if c in study_df.columns]
        if 'Section' in study_df:
            study_df['Section'] = study_df['Section'].map(format_section_id)
        stats_df = stats_df.merge(study_df, on=study_name_columns, how='inner')

    # Filter out bad segmentations
    if 'Surface' in stats_df.columns:
        num_background = stats_df.shape[0]
        stats_df = stats_df[stats_df['Surface'] != 'Background']
        num_background = num_background - stats_df.shape[0]
        print(f'Filtered {num_background} background surfaces ({num_background/stats_df.shape[0]:0.2%})')

    if size_column is not None:
        if max_size is not None and max_size > 0:
            stats_df = stats_df[stats_df[size_column] <= max_size]
        if min_size is not None and min_size > 0:
            stats_df = stats_df[stats_df[size_column] >= min_size]

    print(f'Got {stats_df.shape} records after filtering')

    if remove_batch_effect:
        if batch_column not in stats_df.columns:
            raise KeyError(f'Need "{batch_column}" column indicating the batch number: {stats_df.columns}')

        category_columns = [
            'Animal ID', 'Block', 'Section', 'Region', 'File Prefix', 'Filename',
            'Channel', 'Surface', 'Status', 'Time', 'Timepoint', 'Group', 'Category',
            'Treatment', batch_column,
        ]
        if timepoint_column is not None:
            category_columns.append(timepoint_column)
        if var_column is not None:
            category_columns.append(var_column)

        value_columns = [c for c in stats_df.columns
                         if c not in category_columns and is_numeric_dtype(stats_df[c])]
        batch_labels = stats_df[batch_column].values
        old_values = stats_df[value_columns].values

        new_values = remove_batch_effect_means(batch_labels, old_values, mode='median')
        # new_values = remove_batch_effect_mnn(batch_labels, old_values,
        #                                      num_neighbors=20, sigma=2.0)
        # new_values = remove_batch_effect_mnn(batch_labels, new_values,
        #                                      num_neighbors=10, sigma=2.0)
        # new_values = remove_batch_effect_mnn(batch_labels, new_values,
        #                                      num_neighbors=5, sigma=2.0)
        for i, value_column in enumerate(value_columns):
            stats_df[value_column] = new_values[:, i]

    return stats_df


def calc_bin_edges(bin_min: float,
                   bin_max: float,
                   bin_spacing: Union[str, float] = 'linear',
                   num_bins: int = 3) -> np.ndarray:
    """ Calculate the edges of bins

    :param float bin_min:
        Smallest bin value
    :param float bin_max:
        Largest bin value
    :param str/float bin_spacing:
        Spacing between bins (one of linear, area, volume, log) or an exponent to raise the bins to
    :param int num_bins:
        How many bins to generate
    :returns:
        An array of n+1 bins where bin_edges[0] == bin_min and bin_edges[-1] == bin_max
    """

    # Different kinds of spacing
    if isinstance(bin_spacing, (int, float)):
        bin_spacing = float(bin_spacing)
    elif bin_spacing in ('log', 'logarithmic'):
        return np.logspace(bin_min, bin_max, num_bins+1)
    elif bin_spacing in ('linear', 'radius', 'radial'):
        # Equal radii spaced bins
        power = 1
    elif bin_spacing in ('area', 'circle', 'circular'):
        # Equal area spaced bins
        power = 2
    elif bin_spacing in ('volume', 'sphere', 'spherical'):
        # Equal volume spaced bins
        power = 3
    else:
        raise ValueError(f'Unknown bin spacing {bin_spacing}')

    # Transform, linspace, invert
    d_min = np.abs(bin_min)**power
    d_max = np.abs(bin_max)**power
    bin_edges = np.linspace(d_min, d_max, num_bins+1)
    return bin_edges**(1/power)


def bin_by_dist(rec: np.ndarray,
                num_bins: int = 3,
                bin_spacing: str = 'linear',
                bin_edges: Optional[np.ndarray] = None) -> np.ndarray:
    """ Bin a set of values into equally sized dist bins

    :param ndarray rec:
        The population values to split up
    :param int num_bins:
        The number of bins to split the population into
    :param str bin_spacing:
        How to spread the bins out (a valid argument to :py:func:`calc_bin_edges`)
    :param ndarray bin_edges:
        If not None, use these bin edges regardless
    :returns:
        An array the same shape as rec where ``range(0, num_bins)`` are the
        distances from smallest to largest, and ``num_bins`` is the nan-bin
    """
    # If we got pre-supplied bins, just use those
    if bin_edges is not None:
        bin_edges = np.array(bin_edges)
        num_bins = bin_edges.shape[0]-1

    # Split the nans out from the other values
    nan_mask = np.isnan(rec)
    labels = np.full(rec.shape, dtype=int, fill_value=num_bins)

    if np.all(nan_mask) or num_bins < 1:
        return labels

    # Order the data by rank
    real_rec = rec[~nan_mask]

    # FIXME: Support ranking methods other than linear
    if bin_edges is None:
        dist_min = np.min(real_rec)
        dist_max = np.max(real_rec)
        bin_edges = calc_bin_edges(bin_min=dist_min,
                                   bin_max=dist_max,
                                   bin_spacing=bin_spacing,
                                   num_bins=num_bins)

    # Handle the case where we don't get a left and right edge
    if bin_edges.shape[0] < 2:
        labels[~nan_mask] = 0
        return labels
    assert bin_edges.shape[0] == num_bins+1

    # Split each label on the bin size
    real_labels = np.full(real_rec.shape, dtype=int, fill_value=num_bins)
    real_labels[real_rec < bin_edges[0]] = 0
    real_labels[real_rec >= bin_edges[-1]] = num_bins

    for i, (dist_low, dist_high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = np.logical_and(real_rec >= dist_low, real_rec < dist_high)
        real_labels[mask] = i
    labels[~nan_mask] = real_labels
    return labels


def bin_by_rank(rec: np.ndarray, num_bins: int = 3) -> np.ndarray:
    """ Bin a set of values into equally sized rank bins

    :param ndarray rec:
        The population values to split up
    :param int num_bins:
        The number of bins to split the population into
    :returns:
        An array the same shape as rec where ``range(0, num_bins)`` are the
        ranks from smallest to largest, and ``num_bins`` is the nan-bin
    """

    # Split the nans out from the other values
    nan_mask = np.isnan(rec)
    labels = np.full(rec.shape, dtype=int, fill_value=num_bins)

    if np.all(nan_mask):
        return labels

    # Order the data by rank
    real_rec = rec[~nan_mask]
    order = np.argsort(real_rec)

    # Split each label on the bin size
    bin_size = int(np.ceil(order.shape[0]/num_bins))
    bin_st = 0
    real_labels = np.full(real_rec.shape, dtype=int, fill_value=num_bins)
    for i in range(0, num_bins):
        if bin_st >= order.shape[0]:
            break
        bin_ed = min([bin_st + bin_size, order.shape[0]])
        inds = order[bin_st:bin_ed]
        real_labels[inds] = i
        bin_st = bin_ed
    labels[~nan_mask] = real_labels
    return labels


def load_ylabels(channel_names: Optional[Dict[str, str]] = None,
                 ylabels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """ Load the standardized names for each feature

    :param dict[str, str] channel_names:
        If not None, a mapping of 'Ch=1': 'DAPI', 'Ch=2': 'Iba1', etc
    :returns:
        A list of column name: human readable column name
    """
    if ylabels is None:
        ylabels = {}

    ylabels.update({
        'NumVoxels': 'Number of Voxels',
        'NumCoreVoxels': 'Number of Voxels in the Core',
        'NumShellVoxels': 'Number of Voxels in the Shell',
        'CoreRatio': 'Ratio of Core to Total Voxels',
        'ShellRatio': 'Ratio of Shell to Total Voxels',
        'Volume': 'Volume $(\\mu m^3)$',
        'SurfaceArea': 'Surface Area $(\\mu m^2)$',
        'SurfaceAreaVolume': 'Surface Area:Volume $(\\mu m^{-1})$',
        'ConvexVolume': 'Convex Volume $(\\mu m^3)$',
        'ConvexVolumeRatio': 'Volume:Convex Volume',
        'ConvexSurfaceArea': 'Convex Surface Area $(\\mu m^2)$',
        'ConvexSurfaceAreaRatio': 'Surface Area:Convex Hull Surface Area',
        'BBoxVolume': 'Bounding Box Volume $(\\mu m^3)$',
        'BBoxMinorAxis': 'Bounding Box Minor Axis $(\\mu m)$',
        'BBoxMiddleAxis': 'Bounding Box Mid Axis $(\\mu m)$',
        'BBoxMajorAxis': 'Bounding Box Major Axis $(\\mu m)$',
        'BBoxAspectRatio': 'Bounding Box Aspect Ratio',
        'EquivSphereRadius': 'Sphere Equiv Radius $(\\mu m)$',
        'PercentCore': '% Core Voxels',
        'PercentConvex': '% Convexity',
        'PercentSurface': '% Surface Voxels',
        'PercentBranchVoxels': '% Branch Voxels',
        'PercentSkeleton': '% Skeleton Volume',
        'Sphericity': 'Sphericity',
        'EllipseAspectRatio': 'Aspect Ratio',
        'EllipseMajorAxis': 'Major Axis $(\\mu m)$',
        'EllipseMiddleAxis': 'Mid Axis $(\\mu m)$',
        'EllipseMinorAxis': 'Minor Axis $(\\mu m)$',
        'SkeletonNumBranches': 'Number of Branches',
        'SkeletonNumBranchpoints': 'Number of Branchpoints',
        'SkeletonNumShortBranches': 'Number of Short Branches',
        'SkeletonNumLongBranches': 'Number of Long Branches',
        'SkeletonVoxels': 'Number of Skeleton Voxels',
        'PercentGliaLarge': '% Large Microglia',
        'PercentGliaConvex': '% Convex Microglia',
        'PercentGliaSpherical': '% Spherical Microglia',
        'PercentGliaElongated': '% Elongated Microglia',
        'PercentGliaSurface': '% High Surface Area:Volume Microglia',
        'PercentCells': '% Cells',
        'TotalGliaLarge': 'Number of Large Microglia',
        'TotalGliaConvex': 'Number of Convex Microglia',
        'TotalGliaSpherical': 'Number of Spherical Microglia',
        'TotalGliaElongated': 'Number of Elongated Microglia',
        'TotalGliaSurface': 'Number of High Surface Area:Volume Microglia',
        'ShollNumRadiusBins': 'Number of Sholl Shells',
        'ShollNumLabels': 'Number of Sholl Branches',
        'ShollRadiusMax': 'Maximum Sholl Radius $(\\mu m)$',
        'ShollCriticalBranches': 'Branch Voxels at Critical Radius',
        'ShollCriticalLabels': 'Number of Branches at Critical Radius',
        'ShollCriticalRadius': 'Critical Radius $(\\mu m)$',
        'ShollSchoenenRamificationIndex': 'Schoenen Ramification Index',
        'ShollBranchIndex': 'Sholl Branch Index',
        'ShollRegressionCoeff': 'Sholl Branch Slope',
        'LacunaMean': 'Mean Lacuna',
        'LacunaStd': 'Std Lacuna',
        'LacunaRatio': 'Lacuna Ratio',
        'LacunaCoeff': 'Lacuna Coefficient',
        'HausdorffDim': 'Hausdorff Dimension',
        'HausdorffPrefactor': 'Hausdorff Prefactor',
    })

    # Add in the names for the mean/median/std etc
    param_names = {
        'Mean': 'Mean',
        'Std': 'Std',
        'Min': 'Min',
        'Max': 'Max',
        'Pct05': '5th Centile',
        'Pct25': '25th Centile',
        'Pct50': 'Median',
        'Pct75': '75th Centile',
        'Pct95': '95th Centile',
    }
    for key, value in param_names.items():
        ylabels[f'Skeleton{key}BranchLen'] = f'{value} Branch Length (px)'
    for key, value in param_names.items():
        ylabels[f'{key}Radius'] = f'{value} Radius $(\\mu m)$'

    for channel_id, channel_name in channel_names.items():
        # Sholl Analysis
        ylabels[f'ShollNumRadiusBins_{channel_id}'] = 'Number of Sholl Shells'
        ylabels[f'ShollNumLabels_{channel_id}'] = 'Number of Sholl Branches'
        ylabels[f'ShollRadiusMax_{channel_id}'] = 'Maximum Sholl Radius $(\\mu m)$'
        ylabels[f'ShollCriticalBranches_{channel_id}'] = 'Branch Voxels at Critical Radius'
        ylabels[f'ShollCriticalLabels_{channel_id}'] = 'Number of Branches at Critical Radius'
        ylabels[f'ShollCriticalRadius_{channel_id}'] = 'Critical Radius $(\\mu m)$'
        ylabels[f'ShollSchoenenRamificationIndex_{channel_id}'] = 'Schoenen Ramification Index'
        ylabels[f'ShollBranchIndex_{channel_id}'] = 'Sholl Branch Index'
        ylabels[f'ShollRegressionCoeff_{channel_id}'] = 'Sholl Branch Slope'

        # Fractal Analysis
        ylabels[f'LacunaMean_{channel_id}'] = 'Mean Lacuna'
        ylabels[f'LacunaStd_{channel_id}'] = 'Std Lacuna'
        ylabels[f'LacunaRatio_{channel_id}'] = 'Lacuna Ratio'
        ylabels[f'LacunaCoeff_{channel_id}'] = 'Lacuna Coefficient'
        ylabels[f'HausdorffDim_{channel_id}'] = 'Hausdorff Dimension'
        ylabels[f'HausdorffPrefactor_{channel_id}'] = 'Hausdorff Prefactor'

        # Distance Analysis
        ylabels[f'DistMin_{channel_id}'] = f'Min Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistMean_{channel_id}'] = f'Mean Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistStd_{channel_id}'] = f'Std Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistMax_{channel_id}'] = f'Max Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistPct05_{channel_id}'] = f'5th Centile Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistPct25_{channel_id}'] = f'25th Centile Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistPct50_{channel_id}'] = f'Median Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistPct75_{channel_id}'] = f'75th Centile Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistPct95_{channel_id}'] = f'95th Centile Distance to {channel_name} $(\\mu m)$'
        ylabels[f'DistNumNear_{channel_id}'] = f'Number of Near {channel_name} Objects'
        ylabels[f'DistNumFar_{channel_id}'] = f'Number of Far {channel_name} Objects'

        # Intensity Analysis
        for key, value in param_names.items():
            ylabels[f'Intensity{key}_{channel_id}'] = f'{channel_name} {value} Intensity (AU)'
            ylabels[f'IntensityShell{key}_{channel_id}'] = f'{channel_name} {value} Shell Intensity (AU)'
            ylabels[f'IntensityCore{key}_{channel_id}'] = f'{channel_name} {value} Core Intensity (AU)'

            ylabels[f'NormIntensity{key}_{channel_id}'] = f'Normalized {channel_name} {value} Intensity (AU)'
            ylabels[f'NormIntensityShell{key}_{channel_id}'] = f'Normalized {channel_name} {value} Shell Intensity (AU)'
            ylabels[f'NormIntensityCore{key}_{channel_id}'] = f'Normalized {channel_name} {value} Core Intensity (AU)'
    return ylabels
