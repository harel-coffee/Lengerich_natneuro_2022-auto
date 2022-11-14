""" Single cell morphology quantification pipeline

Main function:

* :py:func:`run_single_cell_pipeline`: Run the pipeline with standard settings

Main Class:

* :py:class:`SingleCellMorphoPipeline`: Single cell morphology pipeline

"""
# Imports
import re
import json
import shutil
import pathlib
import traceback
import itertools
from typing import Optional, List, Tuple, Set, Dict, Union

# 3rd party
import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.stats import pearsonr, ttest_ind
from scipy.cluster.hierarchy import linkage as cluster_linkage, leaves_list

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm
from matplotlib.gridspec import GridSpec

import seaborn as sns

try:
    import umap
except ImportError:
    umap = None

import hdbscan

from sklearn.preprocessing import (
    QuantileTransformer, StandardScaler, RobustScaler, PowerTransformer, LabelEncoder,
)
from sklearn.decomposition import (
    PCA, SparsePCA, LatentDirichletAllocation, FastICA, KernelPCA,
)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Our own imports
from . import surface_grouper, qc_utils, reductions, plot_utils, prism_writer

# Constants
reNORM = re.compile('[^a-z0-9]+', re.IGNORECASE)
# EDGECOLOR = '#C2C2C2'
EDGECOLOR = '#000000'
# FILLCOLOR = '#C2C2C2'
FILLCOLOR = '#EBEBEB'
LINEWIDTH = 2.0  # Width of the lines
MARKERSIZE = 50.0  # Size of the scatter points

# Classes


class SingleCellMorphoPipeline(object):
    """ Single cell morphology pipeline

    Pipeline:

    .. code-block:: python

        proc = SingleCellMorphoPipeline(data)

        # Remove bad surfaces
        proc.filter_by('Volume', min_val=100)

        # Set up aggregation categories
        proc.set_value_columns(category_columns=['Genotype', 'Animal ID'])
        proc.set_palette_column(column='Genotype', palette='Set1')

        # Normalize the data
        proc.normalize_data(transformer='quantile')
        proc.plot_clustermap('heatmap.pdf')

        # Reduce dimensionality
        proc.project_pca(n_components=8)
        proc.plot_scatter('pca.pdf', coord_column='pca')
        proc.calc_pca_loadings()

        proc.project_umap()
        proc.plot_scatter('umap.pdf', coord_column='umap')

        # Cluster the data
        proc.cluster_kmeans(n_clusters=5, coord_column='pca')
        proc.cluster_hdbscan(min_samples=10, min_cluster_size=100, coord_column='umap')

        # Calculate the sum over genotypes
        proc.calc_cluster_counts(id_column='Animal ID', category_columns=['Genotype'])
        proc.plot_cluster_counts(x='Genotype', hue='LabelHDBSCANUmap', y='LabelHDBSCANUmapPercent',
                                 plotfile=plotdir / f'genotype_cluster_percent{suffix}',
                                 order=['WTWT', 'WTKI', 'KIKI'])

    :param DataFrame df:
        The per-cell surface data for the microglia morphology
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.raw_df = df.copy()

        self.category_columns = None
        self.value_columns = None
        self.bad_columns = None

        self.palette_column = None
        self.row_colors = {}
        self.unique_colors = {}
        self.label_unique_colors = {}

        # Filtered data
        self.block_df = None

        # Processed data
        self.transformed_block = None

        # Projected data
        self.pca_coords = None
        self.sparse_pca_coords = None
        self.kernel_pca_coords = None
        self.ica_coords = None
        self.lda_coords = None
        self.ca_coords = None
        self.tsne_coords = None
        self.umap_coords = None

        # Range settings
        self.pca_range = None
        self.sparse_pca_range = None
        self.kernel_pca_range = None
        self.ica_range = None
        self.ca_range = None
        self.umap_range = None
        self.tsne_range = None

        # Counts data
        self.counts_df = None

    def filter_by(self, column: str,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None):
        """ Filter the columns by a given value

        :param str column:
            The column name to filter by
        :param float min_val:
            If not None, keep cells with this value or higher
        :param float max_val:
            If not None, keep cells with this value or lower
        """
        mask = np.ones((self.df.shape[0], ), dtype=bool)
        if min_val is not None:
            mask = np.logical_and(self.df[column] >= min_val)
        if max_val is not None:
            mask = np.logical_and(self.df[column] <= max_val)
        if not np.any(mask):
            raise ValueError(f'No values for {min_val} <= "{column}" <= {max_val}')
        self.df = self.df[mask]

    def drop_channel(self, channel_idx: int):
        """ Drop particular channels from the analysis

        :param int channel_idx:
            The channel index to drop
        """
        print(f'Dropping channel {channel_idx}')
        drop_columns = [c for c in self.df.columns if c.endswith(f'_Ch={channel_idx}')]
        self.drop_columns(drop_columns)

    def drop_columns(self, drop_columns: List[str]):
        """ Drop specific features from the analysis

        :param list[str] drop_columns:
            The list of columns to drop
        """
        drop_columns = [c for c in self.df.columns if c in drop_columns]
        keep_columns = [c for c in self.df.columns if c not in drop_columns]
        self.df = self.df[keep_columns].copy()

        if self.transformed_block is not None:
            keep_inds = [i for i, c in enumerate(self.value_columns) if c in keep_columns]
            self.transformed_block = self.transformed_block[:, keep_inds]
        if self.value_columns is not None:
            self.value_columns = [c for c in self.value_columns if c in keep_columns]
        if self.block_df is not None:
            block_keep_columns = [c for c in self.block_df.columns if c in keep_columns]
            self.block_df = self.block_df[block_keep_columns].copy()
        total_columns = len(keep_columns) + len(drop_columns)
        print(f'Dropped {len(drop_columns)}/{total_columns} columns ({len(keep_columns)} remain)')

    def set_value_columns(self,
                          category_columns: List[str],
                          bad_columns: Optional[List[str]] = None):
        """ Set the value columns

        :param list[str] category_columns:
            The list of columns to ignore because they're categories
        :param list[str] bad_columns:
            If not None, the value columns to ignore
        """
        # Properly pack the categories
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        if bad_columns is None:
            bad_columns = []
        if isinstance(bad_columns, str):
            bad_columns = [bad_columns]

        # Augment bad columns with any non-numeric type columns
        for c in self.df.columns:
            if c in bad_columns or c in category_columns:
                continue
            if not is_numeric_dtype(self.df[c]):
                bad_columns.append(c)

        # Convert the dataframe to a normal numpy array
        value_columns = [c for c in self.df.columns
                         if c not in category_columns and c not in bad_columns]

        block_df = self.df[value_columns]

        print(f'Unscaled value shape: {block_df.shape}')

        self.category_columns = category_columns
        self.bad_columns = bad_columns
        self.value_columns = value_columns

        self.block_df = block_df

    def calc_palette_column(self, column: str,
                            palette: str = 'Set1',
                            order: Optional[List[object]] = None,
                            xticklabels: Optional[List[object]] = None) -> np.ndarray:
        """ Calculate a palette for this column

        :param str column:
            The column to use for the palette
        :param str palette:
            The name of the palette to use
        :param list[object] order:
            The order to use for elements in the column (default: lexographic order)
        """
        unique_values = list(sorted(np.unique(self.df[column])))
        if order is None:
            order = unique_values
        if set(order) < set(unique_values):
            raise ValueError(f'Order contains elements {order}, but {column} has levels {unique_values}')
        if len(order) < len(unique_values):
            raise ValueError(f'Order has a different number of elements from {column}: {len(order)} vs {len(unique_values)}')

        if xticklabels is None:
            xticklabels = order
        if len(order) != len(xticklabels):
            raise ValueError(f'Order has {len(order)} elements but ticklabels has {len(xticklabels)}')

        n_colors = len(order)

        palette = sns.color_palette(palette, n_colors=n_colors)

        # Assign a color to each of the rows, and store the label mapping
        lut = dict(zip(order, palette))
        row_colors = self.df[column].map(lut)
        self.row_colors[column] = row_colors.values
        self.unique_colors[column] = lut
        self.label_unique_colors[column] = dict(zip(xticklabels, palette))
        return self.row_colors[column]

    def set_palette_column(self, column: str,
                           palette: str = 'Set1',
                           order: Optional[List[object]] = None,
                           xticklabels: Optional[List[object]] = None):
        """ Set the column to be used for subsequent color coding

        :param str column:
            The column to use for the palette
        :param str palette:
            The name of the palette to use
        :param list[object] order:
            The order to use for elements in the column (default: lexographic order)
        :param list[object] xticklabels:
            Human readable labels corresponding to the elements in order (default: order)
        """
        self.calc_palette_column(column, palette=palette, order=order,
                                 xticklabels=xticklabels)
        self.palette_column = column

    def normalize_data(self, transformer: str = 'quantile',
                       iqr_tol: float = 1e-2,
                       outfile: Optional[pathlib.Path] = None):
        """ Normalize the raw data

        :param str transformer:
            One of 'standard', 'robust', 'quantile', 'power'
        :param float iqr_tol:
            Minimum inner-quartile range for a column to be considered "non-zero"
        :param Path outfile:
            If not None, write the scaled data to a file
        """
        transformer = transformer.lower().strip()

        # Normalize the data
        if transformer == 'raw':
            proc = None
        elif transformer == 'standard':
            proc = StandardScaler()
        elif transformer == 'robust':
            proc = RobustScaler()
        elif transformer == 'quantile':
            n_quantiles = int(min([1000, self.block_df.shape[0]]))
            proc = QuantileTransformer(n_quantiles=n_quantiles,
                                       output_distribution='normal')
        elif transformer == 'power':
            proc = PowerTransformer()
        else:
            raise KeyError(f'Unknown normalizing transformer: "{transformer}"')

        if proc is None:
            data = self.block_df.values
        else:
            data = proc.fit_transform(self.block_df.values)

        # Drop blocks where there is no dynamic range
        p_min, p_max = np.percentile(data, [25, 75], axis=0)
        keep_inds,  = np.where(np.abs(p_max - p_min) > iqr_tol)
        drop_inds = [i for i in range(data.shape[1]) if i not in keep_inds]

        drop_columns = [self.block_df.columns[i] for i in drop_inds]
        print(f'Dropping uninformative features: {drop_columns}')
        data = data[:, keep_inds]
        new_columns = [self.block_df.columns[i] for i in keep_inds]

        # Make sure we didn't accidentally create NaN or Inf
        # keep_rows = np.sum(~np.isfinite(data), axis=1) < 1
        # data = data[keep_rows, :]

        print(f'Scaled value shape: {data.shape}')

        self.transformed_block = data
        self.value_columns = new_columns

        if outfile is not None:
            print(f'Saving scaled data to {outfile}')
            out_df = {column: data[:, i] for i, column in enumerate(new_columns)}
            for category_column in self.category_columns:
                out_df[category_column] = self.df[category_column].values
            out_df = pd.DataFrame(out_df)
            out_df.to_excel(outfile, index=False, columns=(self.category_columns+self.value_columns))

    # Reductions

    def project_pca(self, n_components: int = 8):
        """ Project the data using PCA

        :param int n_components:
            The number of PCA components to generate
        """
        print(f'Projecting onto PCA with {n_components} components')
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(self.transformed_block)
        self.pca_coords = pca_coords

    def project_ica(self, n_components: int = 8,
                    random_state: int = 42):
        """ Project the data using ICA

        :param int n_components:
            The number of ICA components (sources) to generate
        """
        print(f'Projecting onto ICA with {n_components} components')
        ica = FastICA(n_components=n_components,
                      random_state=random_state)
        ica_coords = ica.fit_transform(self.transformed_block)
        self.ica_coords = ica_coords

    def project_sparse_pca(self, n_components: int = 8,
                           alpha: float = 1.0,
                           ridge_alpha: float = 0.01,
                           method: str = 'lars',
                           random_state: int = 42):
        """ Project the data using Sparse PCA

        :param int n_components:
            The number of Sparse PCA components to generate
        """
        print(f'Projecting onto Sparse PCA with {n_components} components')
        pca = SparsePCA(n_components=n_components,
                        alpha=alpha,
                        ridge_alpha=ridge_alpha,
                        method=method,
                        random_state=random_state)
        sparse_pca_coords = pca.fit_transform(self.transformed_block)
        self.sparse_pca_coords = sparse_pca_coords

    def project_kernel_pca(self, n_components: int = 8,
                           kernel: str = 'linear',
                           gamma: Optional[float] = None,
                           degree: int = 3,
                           random_state: int = 42):
        """ Project the data using Kernel PCA

        :param int n_components:
            The number of Kernel PCA components to generate
        """
        print(f'Projecting onto Kernel PCA with {n_components} components')
        pca = KernelPCA(n_components=n_components,
                        kernel=kernel,
                        gamma=gamma,
                        degree=degree,
                        random_state=random_state)
        kernel_pca_coords = pca.fit_transform(self.transformed_block)
        self.kernel_pca_coords = kernel_pca_coords

    def project_lda(self, n_components: int = 8,
                    random_state: int = 42):
        """ Project the data using LDA

        :param int n_components:
            The number of LDA components (topics) to generate
        """
        print(f'Projecting onto LDA with {n_components} components')
        lda = LatentDirichletAllocation(n_components=n_components,
                                        random_state=random_state)
        # FIXME: Handle negative values here?
        lda_coords = lda.fit_transform(self.transformed_block)
        self.lda_coords = lda_coords

        self.lda_components = lda.components_

    def project_ca(self, n_components: int = 8,
                   pearson_residuals: str = 'corral',
                   pearson_transform: str = 'none'):
        """ Project the data using Correspondence Analysis

        :param int n_components:
            The number of CA components to generate
        """
        print(f'Projecting onto CA with {n_components} components')
        ca = reductions.CorrespondenceAnalysis(
            n_components=n_components,
            pearson_residuals=pearson_residuals,
            pearson_transform=pearson_transform)
        # FIXME: Handle negative values here?
        # print(np.min(self.transformed_block), np.max(self.transformed_block))
        # assert False
        positive_block = self.transformed_block - np.min(self.transformed_block) + 1.0
        ca_coords = ca.fit_transform(positive_block)
        self.ca_coords = ca_coords

    def calc_explained_variance(self, coord_column: str = 'pca'):
        """ Calculate the explained variance

        :param str coord_column:
            The column containing projected data
        """
        # Work out the variance of the original dataset
        cov_raw = np.cov(self.transformed_block, rowvar=False)
        var_raw = np.diag(cov_raw)
        total_var_raw = np.sum(var_raw)

        # Work out the variance in the transformed dataset
        coords = self.get_coord_column(coord_column)
        cov_coords = np.cov(coords, rowvar=False)
        var_coords = np.diag(cov_coords)
        pct_var_coords = var_coords / total_var_raw

        setattr(self, f'{coord_column}_explained_variance', var_coords)
        setattr(self, f'{coord_column}_explained_variance_ratio', pct_var_coords)

    def _calc_topk_correlations(self, v1: np.ndarray,
                                top_k: int = 5,
                                min_correlation: float = 0.5,
                                ignore_indexes: Optional[Set[int]] = None) -> Tuple[List[str]]:
        """ Calculate the topk features interacting with a given feature set

        :param ndarray v1:
            The vector to calculate correlation with
        :param int top_k:
            Maximum number of components to generate
        :param float min_correlation:
            Minimum (anti-)correlation to keep
        :param set[int] ignore_indexes:
            If not None, don't try to calculate correlation with this index
        """
        if ignore_indexes is None:
            ignore_indexes = set()

        all_rs = []
        all_r_inds = []
        for i in range(self.transformed_block.shape[1]):
            if i in ignore_indexes:
                continue

            v2 = self.transformed_block[:, i]

            r = pearsonr(v1, v2)[0]
            all_rs.append(r)
            all_r_inds.append(i)

        # Look for highly correlated and anti-correlated columns
        r_order = np.argsort(all_rs)
        all_rs = np.array(all_rs)[r_order]
        all_r_inds = np.array(all_r_inds)[r_order]

        all_positive_r_inds = all_r_inds[all_rs >= min_correlation]
        all_negative_r_inds = all_r_inds[all_rs <= -min_correlation]

        bottom_rs = all_negative_r_inds[:top_k]
        top_rs = all_positive_r_inds[-top_k:]

        bottom_r_cols = [self.value_columns[i] for i in bottom_rs]
        top_r_cols = [self.value_columns[i] for i in top_rs]
        return bottom_r_cols, top_r_cols

    def calc_reduction_loadings(self, top_k: int = 5,
                                min_correlation: float = 0.5,
                                coord_column: str = 'pca'):
        """ Calculate the loadings for each coordinate of the reductions

        :param int top_k:
            How many items to print for correlated and anti-correlated measures
        :param float min_correlation:
            Minimum correlation for a value to be (anti-)correlated with the component
        :param str coord_column:
            Which coordinate column to use for the loadings
        """
        coords = self.get_coord_column(coord_column)

        for i in range(coords.shape[1]):
            v1 = coords[:, i]

            bottom_r_cols, top_r_cols = self._calc_topk_correlations(
                v1, ignore_indexes={i}, top_k=top_k,
                min_correlation=min_correlation,
            )
            print(f'{coord_column.upper()} {i+1}: Bottom {bottom_r_cols}')
            print(f'{coord_column.upper()} {i+1}: Top {top_r_cols}')

    def calc_feature_loadings(self, feature_column: str,
                              ignore_columns: Optional[List[str]] = None,
                              top_k: int = 5,
                              min_correlation: float = 0.5):
        """ Calculate the correlation of a feature column with every other column

        :param str feature_column:
            The name of a feature column to compare
        :param list[str] ignore_columns:
            If not None, a list of column names to ignore
        :param int top_k:
            How many items to print for correlated and anti-correlated measures
        :param float min_correlation:
            Minimum correlation to display
        """
        if ignore_columns is None:
            ignore_columns = []
        if isinstance(ignore_columns, str):
            ignore_columns = [ignore_columns]

        # Work out the index of the feature and any columns to ignore
        feature_idx = self.value_columns.index(feature_column)
        ignore_indexes = []
        for ignore_column in ignore_columns:
            try:
                idx = self.value_columns.index(ignore_column)
            except ValueError:
                continue
            ignore_indexes.append(idx)
        ignore_indexes = set(ignore_indexes)
        ignore_indexes.add(feature_idx)

        # Calculate correlation with each column
        v1 = self.transformed_block[:, feature_idx]
        bottom_r_cols, top_r_cols = self._calc_topk_correlations(
            v1, ignore_indexes=ignore_indexes, top_k=top_k,
            min_correlation=min_correlation,
        )
        print(f'{feature_column}: Bottom {bottom_r_cols}')
        print(f'{feature_column}: Top {top_r_cols}')

    def get_coord_column(self, coord_column: str) -> np.ndarray:
        """ Get the coordinate column by name

        :param str coord_column:
            The name of the coordinates to load
        :returns:
            The (cells x features) coordinates
        """
        coord_column = coord_column.lower().strip()
        if coord_column.endswith('_coords'):
            coord_column = coord_column.rsplit('_', 1)[0]
        if coord_column == 'raw':
            coords = self.transformed_block
        else:
            coords = getattr(self, f'{coord_column}_coords', None)
        if coords is None:
            raise KeyError(f'No coordinate column for "{coord_column}"')
        return coords

    def get_scatter_combos(self,
                           combos: Optional[List[Tuple[int]]] = None,
                           coord_column: str = 'pca') -> List[Tuple[int]]:
        """ Get the axis combos for scatter plots

        :param list[tuple[int]] combos:
            A list of (x-axis, y-axis) indexes for plots
        :param str coord_column:
            Which coordinate column to load defaults for
        :returns:
            A list of x-axis, y-axis tuples to plot
        """
        if combos is None:
            num_dims = self.get_coord_column(coord_column).shape[1]
            if num_dims >= 4:
                combos = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            elif num_dims == 3:
                combos = [(0, 1), (0, 2), (1, 2)]
            elif num_dims == 2:
                combos = [(0, 1)]
            else:
                raise ValueError(f'Cannot generate 2D plots for coords with {num_dims} features')
        return combos

    def project_tsne(self, n_components: int = 2,
                     coord_column: str = 'pca',
                     perplexity: float = 30.0,
                     early_exaggeration: float = 12.0,
                     learning_rate: Union[float, str] = 200.0,
                     metric: str = 'euclidean'):
        """ Project the data onto TSNE space

        :param int n_components:
            The number of TSNE components to generate
        :param str coord_column:
            Which column to use for the TSNE projection
        """
        print('Calculating TSNE coordinates...')
        coords = self.get_coord_column(coord_column)

        print(f'Input {coord_column.upper()} TSNE coordinates have shape: {coords.shape}')

        proc = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    metric=metric,
                    square_distances=True)
        coords = proc.fit_transform(coords)
        self.tsne_coords = coords

    def project_umap(self, n_components: int = 2,
                     n_neighbors: int = 30,
                     min_dist: float = 0.0,
                     random_state: int = 42,
                     metric: str = 'euclidean',
                     coord_column: str = 'pca'):
        """ Project the data onto UMAP space

        :param int n_components:
            The number of UMAP components to generate
        :param int n_neighbors:
            How many neighbors to consider when generating the manifold
        :param float min_dist:
            Minimum distance between projections
        :param int random_state:
            Random seed for reproducibility
        """
        print('Calculating UMAP coordinates...')
        coords = self.get_coord_column(coord_column)

        print(f'Input {coord_column.upper()} UMAP coordinates have shape: {coords.shape}')

        proc = umap.UMAP(random_state=random_state,
                         n_neighbors=n_neighbors,
                         min_dist=min_dist,
                         metric=metric,
                         n_components=n_components)
        coords = proc.fit_transform(coords)
        self.umap_coords = coords

    def cluster_kmeans(self, n_clusters: int = 5,
                       coord_column: str = 'pca',
                       palette: str = 'Set1',
                       random_state: int = 42):
        """ Cluster the coordinates with KMeans

        Calculate 5 clusters using the pca projection

        .. code-block:: python

            proc.cluster_kmeans(n_clusters=5, coord_column='pca')

        :param int n_clusters:
            The number of clusters to find
        :param str coord_column:
            The coordinate column to use ('pca', 'umap', 'tsne')
        """
        coords = self.get_coord_column(coord_column)

        labels = KMeans(n_clusters=n_clusters,
                        random_state=random_state).fit_predict(coords)

        num_clusters = len(np.unique(labels))
        print(f'Got {num_clusters} KMeans {coord_column.upper()} clusters')

        # Swap the labels so that 0 is the largest cluster, 1 second largest, etc
        cluster_sizes = []
        for i in range(num_clusters):
            cluster_sizes.append(np.sum(labels == i))
        final_labels = np.zeros_like(labels)

        for i, j in enumerate(reversed(np.argsort(cluster_sizes))):
            final_labels[labels == j] = i

        label_column = f'LabelKMeans{coord_column.capitalize()}'
        self.df[label_column] = final_labels
        self.set_palette_column(label_column, palette=palette)

    def cluster_hdbscan(self,
                        min_samples: int = 10,
                        min_cluster_size: int = 100,
                        coord_column: str = 'pca',
                        palette: str = 'Set1'):
        """ Cluster the coordinates with HDBSCAN

        Generate clusters using the PCA projection

        .. code-block:: python

            proc.cluster_hdbscan(coord_column='pca')

        :param int min_samples:
            Minimum points to sample per round
        :param int min_cluster_size:
            Minimum points per cluster
        :param str coord_column:
            The coordinate column to use ('pca', 'umap', 'tsne')
        """
        coords = self.get_coord_column(coord_column)

        labels = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        ).fit_predict(coords)

        num_clusters = len(np.unique(labels))
        print(f'Got {num_clusters} HDBSCAN {coord_column.capitalize()} clusters')

        # Swap the labels so that 0 is the largest cluster, 1 second largest, etc
        cluster_sizes = []
        for i in range(num_clusters):
            cluster_sizes.append(np.sum(labels == i))
        final_labels = np.zeros_like(labels)

        for i, j in enumerate(reversed(np.argsort(cluster_sizes))):
            final_labels[labels == j] = i

        label_column = f'LabelHDBSCAN{coord_column.capitalize()}'
        self.df[label_column] = final_labels
        self.set_palette_column(label_column, palette=palette)

    def calc_cluster_counts(self, id_column: str = 'Animal ID',
                            category_columns: Optional[List[str]] = None,
                            label_columns: Optional[List[str]] = None,
                            count_columns: Optional[List[str]] = None):
        """ Calculate the counts matrix for the clusters

        Group all the microglia by animal, then count up the number of members
        in each cluster, separating by treatment

        .. code-block:: python

            proc.calc_cluster_counts(id_column='Animal ID', category_columns=['Treatment'])

        :param str id_column:
            The ID column to use for each animal
        :param list[str] category_columns:
            Additional category columns (e.g. 'Genotype', 'Timepoint', etc)
        :param list[str] label_columns:
            If not None, which labels to aggregate
        """
        # Make sure we get at least one ID back
        if category_columns is None:
            category_columns = []
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        id_columns = [id_column] + category_columns
        if len(id_columns) < 1:
            raise ValueError('Need at least one cell ID to group by')

        # Make sure we get at least one label to count
        if label_columns is None:
            label_columns = [c for c in self.df.columns if c.startswith('Label')]
        elif isinstance(label_columns, str):
            label_columns = [label_columns]
        if len(label_columns) < 1:
            raise ValueError('No label columns found to count')

        # Count up additional indicator columns
        if count_columns is None:
            count_columns = []
        elif isinstance(count_columns, str):
            count_columns = [count_columns]

        # Loop through each label and total it out
        merge_count_df = []
        for label_column in label_columns:
            count_df = calc_count_df(
                df=self.df,
                label_column=label_column,
                id_columns=id_columns,
                count_columns=count_columns,
            )
            merge_count_df.append(count_df)

        self.counts_df = pd.concat(merge_count_df, ignore_index=True)

    def calc_fold_change(self, label_type: str,
                         label1: Set[object],
                         label2: Set[object],
                         use_normalized_data: bool = True,
                         multitest_method: str = 'hs',
                         multitest_alpha: float = 0.05) -> pd.DataFrame:
        """ Calculate the fold changes between groups in label1 vs label2

        Fold change is calculated as (label2 - label1) for each feature

        :param str label_type:
            The label column to compare over
        :param set[object] label1:
            The labels in the reference (control) group
        :param set[object] label2:
            The labels in the treatment group
        :param bool use_normalized_data:
            If True, use the rescaled data matrix (otherwise use the raw data)
        :param str multitest_method:
            Which multiple comparisons test to run
        :param float multitest_alpha:
            Minimum alpha value for the multiple tests
        :returns:
            A data frame of all features compared between the two groups
        """
        if isinstance(label1, (int, str, float)):
            label1 = {label1, }
        if isinstance(label2, (int, str, float)):
            label2 = {label2, }
        label1 = set(label1)
        label2 = set(label2)

        labels = self.df[label_type].values
        mask1 = np.isin(labels, np.array(list(set(label1))))
        mask2 = np.isin(labels, np.array(list(set(label2))))

        if use_normalized_data:
            data = self.transformed_block
        else:
            data = self.df[self.value_columns].values

        # Make sure we actually got some data
        skip = False
        if np.sum(mask1) < 3:
            print(f'Not enough elements for {label_type} in {label1}')
            skip = True
        if np.sum(mask2) < 3:
            print(f'Not enough elements for {label_type} in {label2}')
            skip = True
        if skip:
            return None

        # Calculate the 1 vs 1 comparisons for these two clusters
        df = {
            'Mean1': [],
            'Mean2': [],
            'Std1': [],
            'Std2': [],
            'PValueUncorr': [],
            'Feature': [],
        }
        for i, feature in enumerate(self.value_columns):
            values = data[:, i]
            value1 = values[mask1]
            value2 = values[mask2]
            mean1 = np.nanmean(value1)
            mean2 = np.nanmean(value2)

            std1 = np.nanstd(value1)
            std2 = np.nanstd(value2)

            ttest = ttest_ind(value1, value2, equal_var=False, nan_policy='omit')

            df['Feature'].append(feature)
            df['Mean1'].append(mean1)
            df['Mean2'].append(mean2)
            df['Std1'].append(std1)
            df['Std2'].append(std2)
            df['PValueUncorr'].append(ttest.pvalue)

        df = pd.DataFrame(df)
        df['Difference'] = df['Mean2'] - df['Mean1']
        num_samples1 = np.sum(mask1)
        num_samples2 = np.sum(mask2)
        pct1 = num_samples1 / mask1.shape[0]
        pct2 = num_samples2 / mask2.shape[0]

        # Calculate Cohen's d
        # FIXME: should we use the sample std here (n-1)?
        df['PooledStd'] = np.sqrt(df['Std1']**2*pct1 + df['Std2']**2*pct2)
        df['EffectSize'] = np.abs(df['Difference']) / df['PooledStd']
        df['PValue'] = multipletests(df['PValueUncorr'].values,
                                     method=multitest_method,
                                     alpha=multitest_alpha)[1]
        df['Label1'] = ','.join(str(level) for level in label1)
        df['Label2'] = ','.join(str(level) for level in label2)
        df['LabelType'] = label_type
        df = df.sort_values(['PValue', 'EffectSize'], ascending=[True, False])
        return df

    def find_markers(self, label_type: str,
                     label1: Set[object],
                     label2: Set[object],
                     pvalue_threshold: float = 0.05,
                     effect_size_threshold: float = 0.2,
                     use_normalized_data: bool = True,
                     multitest_method: str = 'hs') -> pd.DataFrame:
        """ Find the marker features that distinguish between label1 and label2

        Fold change is calculated as (label2 - label1) for each feature

        Uses the data calculated from :py:meth:`calc_fold_change`

        :param str label_type:
            The label column to compare over
        :param set[object] label1:
            The labels in the reference (control) group
        :param set[object] label2:
            The labels in the treatment group
        :param float pvalue_threshold:
            Maximum p-value (corrected) to accept
        :param float effect_size_threshold:
            Minimum effect size (Cohen's d) to accept
            0.01 - very small effect, 0.2 - small effect, 0.5 - medium effect,
            0.8 - large effect, 1.2 - very large effect
        :param bool use_normalized_data:
            If True, use the rescaled data matrix (otherwise use the raw data)
        :param str multitest_method:
            Which multiple comparisons test to run
        :returns:
            A data frame of all significant features between the two groups
        """
        df = self.calc_fold_change(
            label_type=label_type,
            label1=label1,
            label2=label2,
            use_normalized_data=use_normalized_data,
            multitest_method=multitest_method,
            multitest_alpha=pvalue_threshold,
        )
        if df is None:
            return pd.DataFrame({
                'Label1': [],
                'Label2': [],
                'Mean1': [],
                'Mean2': [],
                'Difference': [],
                'LabelType': [],
                'Feature': [],
                'EffectSize': [],
                'PValue': [],
            })
        df = df[df['PValue'] <= pvalue_threshold]
        df = df[df['EffectSize'] >= effect_size_threshold]

        keep_columns = [
            'Feature', 'LabelType', 'Label1', 'Label2',
            'Mean1', 'Mean2', 'Difference',
            'EffectSize', 'PValue',
        ]
        df = df[keep_columns].copy()
        return df.sort_values(['PValue', 'EffectSize'], ascending=[True, False])

    def generate_pair_labels(self, label_type: str,
                             approach: str = '1-vs-rest',
                             control_level: Optional[object] = None,
                             order: Optional[List[str]] = None,
                             ignore_var_levels: Optional[List[str]] = None):
        """ Generate combinatorial comparisons between groups

        This is used by :py:meth:`find_all_markers` and :py:meth:`plot_all_volcanos`
        to make all combinatorial comparisons within a set of labelings

        :param str label_type:
            The label to do the 1 vs rest analysis over
        :param str approach:
            One of '1-vs-all', '1-vs-rest' or '1-vs-control'
        :param object control_level:
            If not None, the control level to compare to in the '1-vs-control' assay.
            To compare to multiple control categories, pass a set of levels
        :param list[str] order:
            If not None, use this as the order to compare labels in
        :returns:
            A generator of label1, label2 assays to apply
        """
        # Normalize the approach string a bit
        approach = approach.lower().strip().replace('one-', '1-').replace('-ctrl', '-control')
        all_levels = set(np.unique(self.df[label_type]))

        # Make sure our control groups make sense
        if approach == '1-vs-control':
            if control_level is None:
                raise ValueError('Define a control level when using the 1-vs-control method')
            if isinstance(control_level, (str, int, float)):
                control_level = {control_level, }
            control_level = set(control_level)
            if not all([c in all_levels for c in control_level]):
                raise ValueError(f'Control level "{control_level}" not found for {label_type}: got {all_levels}')

        if order is None:
            order = list(sorted(all_levels))
        if ignore_var_levels is not None:
            all_levels = {level for level in all_levels if level not in ignore_var_levels}
        assert set(order) <= all_levels
        if control_level is not None:
            assert control_level in order

        # Do the combinatorial search
        for level in order:
            label2 = {level, }
            if approach == '1-vs-all':
                label1 = set(all_levels)
            elif approach == '1-vs-rest':
                label1 = {lvl for lvl in all_levels if lvl != level}
            elif approach == '1-vs-control':
                if level in control_level:
                    continue
                label1 = set(control_level)
            yield label1, label2

    def find_all_markers(self, label_type: str,
                         approach: str = '1-vs-rest',
                         control_level: Optional[object] = None,
                         pvalue_threshold: float = 0.05,
                         effect_size_threshold: float = 0.2,
                         use_normalized_data: bool = True,
                         multitest_method: str = 'hs') -> pd.DataFrame:
        """ Find markers of a cluster using the 1-vs-rest or 1-vs-all method

        For this analysis, each label is sent to :py:meth:`find_markers` as
        "label2", and all the other levels are sent as "label1". Hence, positive
        "Difference" vales in the feature are UP in that label relative to the rest

        This uses the analysis in :py:meth:`find_markers` and :py:meth:`calc_fold_change`

        :param str label_type:
            The label to do the 1 vs rest analysis over
        :param str approach:
            One of '1-vs-all', '1-vs-rest' or '1-vs-control'
        :param object control_level:
            If not None, the control level to compare to in the '1-vs-control' assay.
            To compare to multiple control categories, pass a set of levels
        :param float pvalue_threshold:
            The maximum pvalue to accept
        :param float effect_size_threshold:
            The minimum effect size to accept
        :param bool use_normalized_data:
            If True, use the normalized data
        :param str multitest_method:
            Which multiple testing correction to use
        :returns:
            A DataFrame of all significant markers per cluster
        """
        pairs = self.generate_pair_labels(
            label_type=label_type,
            approach=approach,
            control_level=control_level,
        )
        # Do the combinatorial search
        all_dfs = []
        for label1, label2 in pairs:
            df = self.find_markers(
                label_type=label_type,
                label1=label1,
                label2=label2,
                pvalue_threshold=pvalue_threshold,
                effect_size_threshold=effect_size_threshold,
                use_normalized_data=use_normalized_data,
                multitest_method=multitest_method,
            )
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)

    def write_cluster_counts(self, outfile: pathlib.Path):
        """ Write the cluster counts to an excel spreadsheet

        :param Path outfile:
            The excel file to write
        """
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile = outfile.parent / f'{outfile.stem}.xlsx'
        self.counts_df.to_excel(outfile, index=False)

    def write_surface_data(self, outfile: pathlib.Path):
        """ Write the updated surface data to an excel spreadsheet

        :param Path outfile:
            The excel file to write
        """
        outfile.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_excel(outfile, index=False)

    def write_cluster_prism(self, outfile: pathlib.Path,
                            label_type: str,
                            var_column: str,
                            animal_column: str = 'Animal ID',
                            order: Optional[List[object]] = None):
        """ Write the prism file for the clusters

        :param Path outfile:
            Path to write the output file to
        :param str label_type:
            Which cluster label to generate the plots for
        :param str var_column:
            Which variable to sort the columns into
        :param str animal_column:
            Which column to use for the animal IDs
        :param list[object] order:
            Order for the var_column entries
        """
        counts_df = self.counts_df
        if order is None:
            order = list(sorted(np.unique(counts_df[var_column])))

        with prism_writer.PrismWriter(outfile) as writer:
            proc_df = counts_df[counts_df['LabelType'] == label_type]
            proc_df['PrismGroup'] = proc_df[var_column].map(lambda x: order.index(x))
            levels = np.unique(proc_df['LabelID'])

            for i, level in enumerate(levels):
                level_df = proc_df[proc_df['LabelID'] == level]
                cluster_name = f'% Cells (Cluster {level+1})'
                level_df = level_df.rename(columns={'PercentCells': cluster_name})

                # Add in the animal ID table too
                if i == 0:
                    writer.add_one_way_table(
                        data=level_df,
                        group_column='PrismGroup',
                        name_column=var_column,
                        value_column=animal_column)

                writer.add_one_way_table(
                    data=level_df,
                    group_column='PrismGroup',
                    name_column=var_column,
                    value_column=cluster_name)

    def write_animal_data(self, outdir: pathlib.Path,
                          var_column: str,
                          animal_column: str = 'Animal ID',
                          mean_columns: Optional[List[str]] = None,
                          median_columns: Optional[List[str]] = None,
                          sum_columns: Optional[List[str]] = None,
                          prefix: str = '',
                          suffix: str = '.png',
                          plot_style: str = 'boxes',
                          capsize: float = 0.2,
                          ylabels: Optional[Dict[str, str]] = None,
                          ylims: Optional[Dict[str, Tuple[float]]] = None,
                          xlabel: str = '',
                          force_zero_ylim: bool = False,
                          order: Optional[List[str]] = None,
                          xticklabels: Optional[List[str]] = None,
                          xticklabel_rotation: float = 0.0,
                          comparisons: Optional[List[int]] = None,
                          palette: str = 'Set1',
                          overwrite: bool = False,
                          edgecolor: str = EDGECOLOR,
                          linewidth: int = LINEWIDTH,
                          fillcolor: str = FILLCOLOR):
        """ Write the animal data out to a folder

        Animal data includes stat tables, count tables, and per-animal plots

        :param Path outdir:
            The directory to write the animal data to
        :param str var_column:
            The main grouping column (treatment/genome/etc)
        :param str animal_column:
            The column containing animal ids
        :param list[str] mean_columns:
            Columns to group together by averaging
        :param list[str] median_columns:
            Columns to group together by taking the median
        :param list[str] sum_columns:
            Columns to group together by calculating a total sum
        """
        if plot_style in ('err', 'err_bar', 'err_bars'):
            err_bar_join = False
            err_bar_ci = 'sd'
        else:
            err_bar_join = True
            err_bar_ci = 95

        outdir = pathlib.Path(outdir)
        if overwrite and outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        key_columns = [var_column, animal_column]
        grouper = surface_grouper.SurfaceGrouper(self.raw_df, key_columns=key_columns)
        if mean_columns is not None:
            grouper.calc_mean_df(mean_columns=mean_columns)
        if median_columns is not None:
            grouper.calc_median_df(median_columns=median_columns)
        if sum_columns is not None:
            grouper.calc_sum_df(sum_columns=sum_columns)

        grouper.write_excel_table(outdir / f'{prefix}stats_per_animal.xlsx')
        grouper.write_excel_stats(
            outdir / f'{prefix}pvalues_per_animal.xlsx',
            group_column=var_column,
            animal_column=animal_column,
            order=order,
            comparisons=comparisons,
        )
        with prism_writer.PrismWriter(outdir / f'{prefix}stats_per_animal.pzfx') as writer:
            grouper.write_prism_tables(writer, var_column=var_column,
                                       animal_column=animal_column,
                                       ylabels=ylabels,
                                       order=order)
        grouper.plot_all(var_column=var_column,
                         plotdir=outdir,
                         prefix=prefix,
                         suffix=suffix,
                         ylabels=ylabels,
                         ylims=ylims,
                         xlabel=xlabel,
                         force_zero_ylim=force_zero_ylim,
                         order=order,
                         plot_style=plot_style,
                         capsize=capsize,
                         err_bar_join=err_bar_join,
                         err_bar_ci=err_bar_ci,
                         xticklabels=xticklabels,
                         xticklabel_rotation=xticklabel_rotation,
                         comparisons=comparisons,
                         palette=palette,
                         edge_color=edgecolor,
                         fill_color=fillcolor,
                         linewidth=linewidth)

    def merge_top_features(self, all_dfs: List[pd.DataFrame],
                           include_standard_features: bool = True,
                           top_k: Optional[int] = None,
                           up_top_k: Optional[int] = None,
                           down_top_k: Optional[int] = None) -> List[str]:
        """ Merge the top features from a number of find_all_markers assays

        Merge two sets of markers, taking the top 3 features from each

        .. code-block:: python

            treatment_df = proc.find_all_markers(
                label_type='Treatment',
                approach='1-vs-rest')
            cluster_df = proc.find_all_markers(
                label_type='LabelKMeansPCA',
                approach='1-vs-rest')
            mean_columns = proc.merge_top_features(
                [treatment_df, cluster_df],
                include_standard_features=False,
                top_k=3,
            )

        :param list[DataFrame] all_dfs:
            A list of one or more dataframes from :py:meth:`find_all_markers`
        :param bool include_standard_features:
            If True, always add a set of standard features to plot
        :param int top_k:
            If not None and >= 0, only take the top features from both up and down
        :param int up_top_k:
            If not None and >= 0, only take the top features from up (0 to skip features which are up)
        :param int down_top_k:
            If not None and >= 0, only take the top features from down (0 to skip features which are down)
        :returns:
            A list of key features that are important across dataframes
        """
        if all_dfs is None:
            all_dfs = []
        elif isinstance(all_dfs, pd.DataFrame):
            all_dfs = [all_dfs]

        if up_top_k is None and down_top_k is None:
            up_top_k = top_k
            down_top_k = top_k

        # Load the standard features
        mean_columns = set()
        if include_standard_features:
            # Standard volumetric measures
            mean_columns.update([
                'Volume', 'MeanRadius', 'SurfaceArea', 'SurfaceAreaVolume',
                'PercentConvex', 'PercentSurface', 'PercentSkeleton',
                'Sphericity', 'EllipseAspectRatio', 'PercentBranchVoxels',
                'SkeletonNumBranches', 'SkeletonMeanBranchLen', 'SkeletonNumBranchpoints',
                'SkeletonNumShortBranches', 'SkeletonNumLongBranches', 'SkeletonVoxels',
            ])
            for c in self.df.columns:
                # Standard intensity measures
                if c.startswith(('IntensityMean_Ch=', 'IntensityShellMean_Ch=', 'IntensityCoreMean_Ch=')):
                    mean_columns.add(c)
                if c.startswith(('IntensityPct50_Ch=', 'IntensityShellPct50_Ch=', 'IntensityCorePct50_Ch=')):
                    mean_columns.add(c)
                if c.startswith(('NormIntensityMean_Ch=', 'NormIntensityShellMean_Ch=', 'NormIntensityCoreMean_Ch=')):
                    mean_columns.add(c)
                if c.startswith(('NormIntensityPct50_Ch=', 'NormIntensityShellPct50_Ch=', 'NormIntensityCorePct50_Ch=')):
                    mean_columns.add(c)

                # Standard fractal measures
                if c.startswith(('HausdorffDim_Ch=', 'LacunaMean_Ch=', 'LacunaCoeff_Ch=', 'LacunaRatio_Ch=')):
                    mean_columns.add(c)
                if c.startswith(('ShollCriticalRadius_Ch=', 'ShollCriticalLabels_Ch=', 'ShollCriticalBranches_Ch=')):
                    mean_columns.add(c)

        # Look through the top feature dataframes and find the unbiased features
        for df in all_dfs:
            all_label1s = np.unique(df['Label1'])
            all_label2s = np.unique(df['Label2'])
            for label1, label2 in itertools.product(all_label1s, all_label2s):
                mask = (df['Label1'] == label1).values & (df['Label2'] == label2).values
                if not np.any(mask):
                    continue
                sub_df = df[mask]
                up_df = sub_df[sub_df['Difference'] > 0]
                down_df = sub_df[sub_df['Difference'] < 0]

                up_df = up_df.sort_values(['PValue', 'Difference'], ascending=[True, False])
                down_df = down_df.sort_values(['PValue', 'Difference'], ascending=[True, True])

                if up_top_k is not None and up_top_k >= 0:
                    up_df = up_df.iloc[:up_top_k, :]
                if down_top_k is not None and down_top_k >= 0:
                    down_df = down_df.iloc[:down_top_k, :]
                up_features = set(np.unique(up_df['Feature']))
                down_features = set(np.unique(down_df['Feature']))

                mean_columns.update(up_features)
                mean_columns.update(down_features)

        # Validate that all the columns are in the data set
        mean_columns = [c for c in mean_columns if c in self.df.columns]
        print(f'Got {len(mean_columns)} interesting feature columns')
        return mean_columns

    # Plotting methods

    def plot_legend(self, column: str,
                    plotfile: Optional[pathlib.Path] = None,
                    marker: str = 'o',
                    cluster_labels: Optional[Dict[str, str]] = None):
        """ Make the legend for the column

        :param str column:
            The column to plot the legend for
        :param Path plotfile:
            File to save the plot to
        :param str marker:
            Marker to use for the color indicator (i.e. '.', 'o', 'c', etc)
        """
        if cluster_labels is None:
            cluster_labels = {}
        unique_colors = self.label_unique_colors.get(column)

        labels = [cluster_labels.get(str(c), str(c)) for c in unique_colors.keys()]
        longest_label = max([len(label) for label in labels])

        # Fake a plot to hook the legend machinery
        fig, ax = plt.subplots(1, 1, figsize=(max([longest_label*0.5, 6]), len(unique_colors)*0.75))
        handles = [
            ax.plot([], [], marker=marker, color=c, ls="none")[0]
            for c in unique_colors.values()
        ]
        legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

        fig = legend.figure
        fig.canvas.draw()
        ax.axis('off')

        plt.tight_layout()

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, dpi="figure", transparent=True)
            plt.close()

    def plot_clustermap(self, plotfile: Optional[pathlib.Path] = None,
                        cmap: str = 'icefire',
                        vmin: float = -3,
                        vmax: float = 3,
                        palette_column: Optional[str] = None,
                        row_order: Optional[List[str]] = None,
                        hide_row_clusters: bool = False,
                        hide_col_clusters: bool = False,
                        cluster_method: str = 'average',
                        cluster_metric: str = 'euclidean'):
        """ Plot the clustermap

        :param Path plotfile:
            The file to write the clustermap to
        :param str cmap:
            Color map for the clustered heatmap
        :param float vmin:
            The minimum value for the clustermap
        :param float vmax:
            The maximum value for the clustermap
        :param str palette_column:
            If not None, plot the cluster palette from this column
        :param list[str] row_order:
            If not None, group the cells into treatments by row, but subcluster under that
        :param bool hide_row_clusters:
            If True, don't show the clustering on the rows (cells)
        :param bool hide_col_clusters:
            If True, don't show the clustering on the columns (features)
        """
        if palette_column is None:
            palette_column = self.palette_column
        row_colors = self.row_colors.get(palette_column)

        if row_order is None:
            row_cluster = True
            block = self.transformed_block
        else:
            row_cluster = False
            hide_row_clusters = True

            categories = self.df[palette_column].values
            unique_colors = self.unique_colors.get(palette_column)
            block = self.transformed_block

            if len(row_order) != len(unique_colors):
                raise ValueError(f'Got row order {row_order} but color table {unique_colors}')

            if not all([c in unique_colors for c in row_order]):
                raise ValueError(f'Got row order keys {row_order} but color table keys {unique_colors.keys()}')

            # Perform clustering under each group
            final_block = []
            final_colors = []
            for category in row_order:
                color = unique_colors[category]
                mask = categories == category
                if not np.any(mask):
                    continue

                cat_block = block[mask, :]

                clusters = cluster_linkage(cat_block, method=cluster_method, metric=cluster_metric)
                leaf_order = leaves_list(clusters)
                final_block.append(cat_block[leaf_order, :])
                final_colors.append(np.array([color for _ in range(cat_block.shape[0])]))

            block = np.concatenate(final_block, axis=0)
            row_colors = np.concatenate(final_colors, axis=0)
            assert block.shape == self.transformed_block.shape
            assert row_colors.shape[0] == block.shape[0]

        # Generate the clustermap now
        g = sns.clustermap(block,
                           method=cluster_method, metric=cluster_metric,
                           row_cluster=row_cluster, col_cluster=True, cmap=cmap,
                           row_colors=row_colors,
                           vmin=vmin, vmax=vmax)
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticks([])

        print(f'Value column order: {[self.value_columns[i] for i in g.dendrogram_col.reordered_ind]}')

        if hide_row_clusters:
            g.ax_row_dendrogram.set_visible(False)
        if hide_col_clusters:
            g.ax_col_dendrogram.set_visible(False)

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            g.savefig(plotfile, transparent=True)
            plt.close()

            # Save the columns off for later inspection
            value_column_order = pd.DataFrame({
                'Column': [self.value_columns[i] for i in g.dendrogram_col.reordered_ind],
            })
            value_column_order.to_excel(plotfile.parent / f'{plotfile.stem}_column_order.xlsx', index=True)

    def plot_all_volcanos(self, label_type: str,
                          plotdir: pathlib.Path,
                          approach: str = '1-vs-rest',
                          control_level: Optional[object] = None,
                          pvalue_threshold: float = 0.05,
                          pvalue_tol: float = 1e-15,
                          effect_size_threshold: float = 0.2,
                          use_normalized_data: bool = True,
                          order: Optional[List[str]] = None,
                          ignore_var_levels: Optional[List[str]] = None,
                          multitest_method: str = 'hs',
                          prefix: str = '',
                          suffix: str = '.png',
                          cluster_labels: Optional[Dict[str, str]] = None,
                          **kwargs):
        """ Generate all the volcano plots for a comparison

        Generate every possible comparison of one treatment vs the rest

        .. code-block:: python

            proc.plot_all_volcanos(
                'Treatment',
                plotdir='./treatment_volcanos',
                approach='1-vs-rest',
                prefix='21-100',
                suffix='.png')

        :param str label_type:
            The column with label values to analyze
        :param Path plotdir:
            Directory to save all the plots to
        :param str approach:
            One of '1-vs-all', '1-vs-rest' or '1-vs-control' (see :py:meth:`generate_pair_labels`)
        :param object control_level:
            Level(s) to use as the control value for 1-vs-control
        :param float pvalue_threshold:
            Maximum significant p-value
        :param float pvalue_tol:
            Minimum p-value to allow to prevent divide by zero errors
        :param float effect_size_threshold:
            Minimum effect size to allow
        :param bool use_normalized_data:
            If True, use the normalized data (recommended)
        :param str multitest_method:
            Method to use to control for multiple comparisons
        :param str prefix:
            Prefix for the names of the files in the plots
        :param str suffix:
            Suffix to save the plots with
        :param \\*\\*kwargs:
            Additional arguments to pass to :py:meth:`plot_volcano`
        """
        if cluster_labels is None:
            cluster_labels = {}
        label_outname = reNORM.sub('_', label_type).lower().strip('_')

        # Make sure the plot directory exists
        plotdir = pathlib.Path(plotdir)
        plotdir.mkdir(parents=True, exist_ok=True)

        # Generate the comparisons according to the selected assay
        pairs = self.generate_pair_labels(
            label_type=label_type,
            approach=approach,
            control_level=control_level,
            order=order,
            ignore_var_levels=ignore_var_levels)
        for i, (label1, label2) in enumerate(pairs):
            plotfile = plotdir / f'{prefix}volcano-{label_outname}{i}{suffix}'
            title = ', '.join(cluster_labels.get(str(level), str(level))
                              for level in label2)

            # Calculate fold change (if possible between these groups)
            fold_change_df = self.calc_fold_change(
                label_type=label_type,
                label1=label1,
                label2=label2,
                use_normalized_data=use_normalized_data,
                multitest_method=multitest_method,
                multitest_alpha=pvalue_threshold,
            )
            if fold_change_df is None:
                continue
            print(f'Plotting volcano {label1} vs {label2}')
            self.plot_volcano(fold_change_df, plotfile=plotfile,
                              pvalue_threshold=pvalue_threshold,
                              pvalue_tol=pvalue_tol,
                              effect_size_threshold=effect_size_threshold,
                              title=title,
                              **kwargs)

    def plot_volcano(self, fold_change_df: pd.DataFrame,
                     plotfile: Optional[pathlib.Path] = None,
                     figsize: Tuple[float] = (16, 16),
                     pvalue_threshold: float = 0.05,
                     pvalue_tol: float = 1e-15,
                     effect_size_threshold: float = 0.2,
                     label_features: Optional[List[str]] = None,
                     label_top_k: int = 10,
                     label_top_k_up: int = -1,
                     label_top_k_down: int = -1,
                     label_renames: Optional[Dict[str, str]] = None,
                     markersize: int = 20,
                     fontsize: int = 20,
                     title: str = ''):
        """ Make some volcano plots

        Compare Control vs Treatment groups, then plot the volcano

        .. code-block:: python

            fold_change_df = proc.calc_fold_change(
                label_type='Treatment',
                label1='Control',
                label2='Treated',
            )
            proc.plot_volcano(fold_change_df,
                              plotfile='volcano-control_vs_treated.png',
                              title='Control vs Treatment')

        :param DataFrame fold_change_df:
            The data frame from the fold change analysis (see :py:meth:`calc_fold_change`)
        :param Path plotfile:
            The path to write the fold change file to
        :param tuple[float] figsize:
            Size of the figure to generate
        :param float pvalue_threshold:
            Maximum significant p-value
        :param float pvalue_tol:
            Minimum p-value to allow to prevent divide by zero errors
        :param float effect_size_threshold:
            Minimum effect size to allow
        :param set[str] label_features:
            If not None, specifically label these features
        :param int label_top_k:
            If > 0, also plot the labels for the top k up and down features
        :param dict[str, str] label_renames:
            A dictonary of "label name": "display name" values for the plotted labels
        :param str title:
            Title for the plot
        """
        if label_renames is None:
            label_renames = {}

        # Pull out the key values to split into significant/not-significant
        effect_size = fold_change_df['EffectSize'].values
        pvalue = fold_change_df['PValue'].values
        difference = fold_change_df['Difference'].values
        log_pvalue = -np.log10(pvalue + pvalue_tol)  # Log10 + an eps to prevent divide by 0
        fold_change_df['Log10P'] = log_pvalue

        # Split everything into up, down, not-significant
        is_sig_down = (
            (effect_size > effect_size_threshold) &
            (pvalue < pvalue_threshold) &
            (difference < 0)
        )
        is_sig_up = (
            (effect_size > effect_size_threshold) &
            (pvalue < pvalue_threshold) &
            (difference > 0)
        )
        is_not_sig = ~(is_sig_up | is_sig_down)

        # Figure out what, if anything, we should label
        if label_features is None:
            label_features = set()
        if isinstance(label_features, str):
            label_features = {label_features}
        label_features = set(label_features)

        # Add in the top and bottom features if requested
        if label_top_k > 0:
            label_top_k_up = label_top_k_down = label_top_k
        if label_top_k_down > 0 and np.any(is_sig_down):
            fold_change_df_down = fold_change_df[['Difference', 'Log10P', 'Feature']]
            fold_change_df_down = fold_change_df_down.iloc[is_sig_down, :]
            fold_change_df_down = fold_change_df_down.sort_values(
                ['Log10P', 'Difference'], ascending=[False, True])
            label_features.update(fold_change_df_down['Feature'][:label_top_k_down])
        if label_top_k_up > 0 and np.any(is_sig_up):
            fold_change_df_up = fold_change_df[['Difference', 'Log10P', 'Feature']]
            fold_change_df_up = fold_change_df_up.iloc[is_sig_up, :]
            fold_change_df_up = fold_change_df_up.sort_values(
                ['Log10P', 'Difference'], ascending=[False, False])
            label_features.update(fold_change_df_up['Feature'][:label_top_k_up])

        # Force the plot to be symmetrical about zero
        min_difference = np.min(difference)
        max_difference = np.max(difference)
        rng_difference = np.max([np.abs(min_difference), np.abs(max_difference)])
        max_log_pvalue = -np.log10(pvalue_tol)

        xlim = [-1.5*rng_difference, 1.5*rng_difference]
        ylim = [-0.1, max_log_pvalue*1.1]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if np.any(is_not_sig):
            ax.plot(difference[is_not_sig], log_pvalue[is_not_sig],
                    marker='o', markersize=markersize, linewidth=0,
                    markeredgecolor='#BFBFBF',
                    markerfacecolor='#404040')
        if np.any(is_sig_down):
            # Have to sort to make the minimum values at the start
            fold_change_df_down = fold_change_df[['Difference', 'Log10P', 'Feature']]
            fold_change_df_down = fold_change_df_down.iloc[is_sig_down, :]
            fold_change_df_down = fold_change_df_down.sort_values(
                ['Log10P', 'Difference'], ascending=[False, True])

            x_down = fold_change_df_down['Difference']
            y_down = fold_change_df_down['Log10P']
            ax.plot(x_down, y_down,
                    marker='o', markersize=markersize, linewidth=0,
                    markeredgecolor='#A6CEE3',
                    markerfacecolor='#1F78B4')

            # Draw labels if we got any to plot
            left_coord = xlim[0] + 0.05
            top_coord = max_log_pvalue*0.8
            down_features = fold_change_df_down['Feature']
            label_feature_mask = down_features.isin(label_features)
            for x, y, feature in zip(x_down[label_feature_mask],
                                     y_down[label_feature_mask],
                                     down_features[label_feature_mask]):
                ax.plot([x, left_coord], [y, top_coord], '-', linewidth=3,
                        color='#A6CEE3')
                ax.text(left_coord, top_coord, label_renames.get(feature, feature),
                        size=fontsize, color='#ffffff', weight='bold',
                        horizontalalignment='left',
                        verticalalignment='center',
                        bbox={
                            'boxstyle': "square",
                            'edgecolor': '#A6CEE3',
                            'facecolor': '#1F78B4',
                        })
                top_coord -= max_log_pvalue*0.05

        if np.any(is_sig_up):
            # Have to sort to make the maximum values at the start
            fold_change_df_up = fold_change_df[['Difference', 'Log10P', 'Feature']]
            fold_change_df_up = fold_change_df_up.iloc[is_sig_up, :]
            fold_change_df_up = fold_change_df_up.sort_values(
                ['Log10P', 'Difference'], ascending=[False, False])

            x_up = fold_change_df_up['Difference']
            y_up = fold_change_df_up['Log10P']
            ax.plot(x_up, y_up,
                    marker='o', markersize=markersize, linewidth=0,
                    markeredgecolor='#FB9998',
                    markerfacecolor='#E31C1B')

            right_coord = xlim[1] - 0.05
            top_coord = max_log_pvalue*0.8
            up_features = fold_change_df_up['Feature']
            label_feature_mask = up_features.isin(label_features)
            for x, y, feature in zip(x_up[label_feature_mask],
                                     y_up[label_feature_mask],
                                     up_features[label_feature_mask]):
                ax.plot([x, right_coord], [y, top_coord], '-', linewidth=3,
                        color='#FB9998')
                ax.text(right_coord, top_coord, label_renames.get(feature, feature),
                        size=fontsize, color='#ffffff', weight='bold',
                        horizontalalignment='right',
                        verticalalignment='center',
                        bbox={
                            'boxstyle': "square",
                            'edgecolor': '#FB9998',
                            'facecolor': '#E31C1B',
                        })
                top_coord -= max_log_pvalue*0.05

        # Work out how many features are up or down regulated
        num_up = np.sum(is_sig_up)
        pct_up = num_up/is_sig_up.shape[0]
        num_down = np.sum(is_sig_down)
        pct_down = num_down/is_sig_down.shape[0]

        left_coord = xlim[0] + 0.05
        right_coord = xlim[1] - 0.05
        ax.text(right_coord, max_log_pvalue*0.05, f'{num_up} ({pct_up:0.1%}) Up',
                size=28, color='#000000', weight='bold',
                horizontalalignment='right',
                verticalalignment='center')
        ax.text(left_coord, max_log_pvalue*0.05, f'{num_down} ({pct_down:0.1%}) Down',
                size=28, color='#000000', weight='bold',
                horizontalalignment='left',
                verticalalignment='center')

        ax.set_xlabel('Normalized Difference')
        ax.set_ylabel('$-log_{10}(p)$')
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_scatter(self, plotfile: Optional[pathlib.Path] = None,
                     palette_column: Optional[str] = None,
                     figsize: Tuple[float] = (16, 16),
                     combos: Optional[List[Tuple[int]]] = None,
                     merge_column: Optional[str] = None,
                     coord_column: str = 'pca',
                     cache_plot_range: bool = False,
                     coord_name: Optional[str] = None,
                     xlim: Optional[Tuple[float]] = None,
                     ylim: Optional[Tuple[float]] = None,
                     palette_labels: Optional[Dict[str, str]] = None,
                     add_palette_labels: bool = False,
                     edgecolor: str = EDGECOLOR,
                     linewidth: float = LINEWIDTH,
                     markersize: float = MARKERSIZE,
                     fontsize: float = 36):
        """ Plot dimension reduction columns as a scatter plot

        Plot the interactions between PCs 1-4

        .. code-block:: python

            proc.plot_scatter(plotdir / '21-100-pca.png', coord_column='pca',
                              combos=[(0, 1), (0, 2), (1, 2), (0, 3)])

        Plot the cluster labels in umap space, annotating each cluster with its number:

        .. code-block:: python

            proc.plot_scatter(plotdir / '21-100-umap-cluster.png',
                              coord_column='umap',
                              palette_column='LabelKMeansPCA',
                              add_palette_labels=True)

        :param Path plotfile:
            Path to save the final plots to
        :param str palette_column:
            If not None, which column to use to color the points
        :param tuple[float] figsize:
            Size of the plot figure
        :param list[tuple[int]] combos:
            The list of PCA components to plot (e.g. [(0, 1), (0, 2), (1, 2)])
            to compare PCs 1 vs 2, 1 vs 3, 2 vs 3
        :param str merge_column:
            If not None, collapse all coordinate values using this column
        :param bool add_palette_labels:
            If True, add labels for each of the classes in the palette_column
        """
        if palette_labels is None:
            palette_labels = {}
        coords = self.get_coord_column(coord_column)

        if palette_column is None:
            palette_column = self.palette_column
        row_colors = self.row_colors.get(palette_column)
        unique_colors = self.unique_colors.get(palette_column)

        # Plot combos of the first few coords
        combos = self.get_scatter_combos(combos, coord_column)
        if coord_name is None:
            coord_name = {
                'pca': 'PC',
                'ica': 'IC',
                'umap': 'UMAP',
                'tsne': 'TSNE',
            }.get(coord_column, '')
        num_combos = len(combos)
        num_cols = int(min([4, np.ceil(np.sqrt(num_combos))]))
        num_rows = int(np.ceil(num_combos/num_cols))

        # If requested, merge over a category
        if merge_column is not None:
            categories = self.df[merge_column]
            assert categories.shape[0] == coords.shape[0]
            new_coords = []
            new_row_colors = []
            if row_colors is not None:
                row_colors = np.array([r for r in row_colors])
                assert row_colors.shape[0] == coords.shape[0]
                assert row_colors.shape[1] == 3
            for category in np.unique(categories):
                mask = categories == category
                new_coords.append(np.mean(coords[mask, :], axis=0))
                if row_colors is not None:
                    new_row_color = row_colors[mask]
                    new_row_colors.append(np.mean(new_row_color, axis=0))
            coords = np.stack(new_coords, axis=0)
            if row_colors is not None:
                row_colors = np.array(new_row_colors)

        figsize_x, figsize_y = figsize
        aspect = num_rows / num_cols

        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(figsize_x, figsize_y*aspect),
                                 squeeze=False)
        axes = axes.flatten()
        for idx, (i, j) in enumerate(combos):
            ax = axes[idx]
            ax.scatter(coords[:, i], coords[:, j], s=markersize, c=row_colors, marker='o',
                       edgecolor=edgecolor, linewidth=linewidth)
            ax.set_xlabel(f'{coord_name} {i+1}')
            ax.set_ylabel(f'{coord_name} {j+1}')

            if add_palette_labels:
                for key, val in unique_colors.items():
                    # Work out the center of mass of each cluster
                    mask = self.df[palette_column] == key
                    mean_x = np.mean(coords[mask, i])
                    mean_y = np.mean(coords[mask, j])

                    # Switch the label color from black to white based on facecolor
                    label_color_tuple = mplcolors.to_rgba(val, 1.0)
                    label_grey = np.mean(label_color_tuple[:3])
                    if label_grey > 0.8:
                        text_color = 0.0
                    else:
                        text_color = 1.0
                    text_color = mplcolors.to_rgba((text_color, text_color, text_color), 1.0)

                    # Plot the text on the scatter in a box with a facecolor matching the cluster
                    ax.text(mean_x, mean_y, palette_labels.get(str(key), str(key)),
                            size=fontsize,
                            color=text_color, weight='bold',
                            horizontalalignment='center',
                            verticalalignment='center',
                            bbox={
                                'boxstyle': "square",
                                'edgecolor': edgecolor,
                                'facecolor': val,
                            })

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

        # Hide the remaining boxes
        for idx in range(len(combos), num_rows*num_cols):
            axes[idx].axis('off')

        # Save and potentially reload a previous range
        coord_range = getattr(self, f'{coord_column}_range')
        if coord_range is None:
            coord_range = {}

        for idx, (i, j) in enumerate(combos):
            xlim = coord_range.get(i)
            ylim = coord_range.get(j)
            if xlim is not None:
                axes[idx].set_xlim(xlim)
            if ylim is not None:
                axes[idx].set_ylim(ylim)

            if cache_plot_range:
                xlim = axes[idx].get_xlim()
                ylim = axes[idx].get_ylim()
                coord_range[i] = xlim
                coord_range[j] = ylim
        setattr(self, f'{coord_column}_range', coord_range)

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_feature_scatter(self,
                             value_column: str,
                             value_type: str = 'normalized',
                             plotfile: Optional[pathlib.Path] = None,
                             figsize: Tuple[float] = (16, 16),
                             coord_column: str = 'pca',
                             palette: str = 'vlag',
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             alpha: float = 0.8,
                             combos: Optional[List[Tuple[int]]] = None,
                             cache_plot_range: bool = False,
                             coord_name: Optional[str] = None,
                             xlim: Optional[Tuple[float]] = None,
                             ylim: Optional[Tuple[float]] = None,
                             title: str = '',
                             edgecolor: str = EDGECOLOR,
                             linewidth: float = LINEWIDTH,
                             markersize: float = MARKERSIZE):
        """ Plot individual features on a coordinate plot

        :param str value_column:
            Which column to plot over the coordinates
        :param str value_type:
            One of 'raw' or 'normalized'
        """

        # Load the transformed block and scale
        if value_type in ('norm', 'normal', 'normalized'):
            if value_column not in self.value_columns:
                return
            values = self.transformed_block[:, self.value_columns.index(value_column)]
        elif value_type in ('raw'):
            values = self.df[value_column].values
        else:
            raise ValueError(f'Unknown value type: {value_type}')
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        norm = mplcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        norm_values = norm(values)
        cmap = sns.color_palette(palette, as_cmap=True)
        row_colors = cmap(norm_values, alpha=alpha)

        mappable = mplcm.ScalarMappable(norm=norm, cmap=cmap)

        # Load the coordinate column
        coords = self.get_coord_column(coord_column)

        # Plot combos of the first few coords
        combos = self.get_scatter_combos(combos, coord_column)
        if coord_name is None:
            coord_name = {
                'pca': 'PC',
                'ica': 'IC',
                'umap': 'UMAP',
                'tsne': 'TSNE',
            }.get(coord_column, '')
        num_combos = len(combos)
        num_cols = int(min([4, np.ceil(np.sqrt(num_combos))]))
        num_rows = int(np.ceil(num_combos/num_cols))

        figsize_x, figsize_y = figsize
        aspect = num_rows / num_cols

        width_ratios = [0.95/num_cols for _ in range(num_cols)]
        width_ratios.append(0.05)

        fig = plt.figure(constrained_layout=True,
                         figsize=(figsize_x*1.1, figsize_y*aspect),)
        gs = GridSpec(num_rows, num_cols+1, figure=fig,
                      width_ratios=width_ratios)

        # Load the axes via the gridspec
        axes = []
        for idx, (i, j) in enumerate(combos):
            gs_i = idx // num_rows
            gs_j = idx % num_rows
            ax = fig.add_subplot(gs[gs_i, gs_j])
            ax.scatter(coords[:, i], coords[:, j], s=markersize, c=row_colors, marker='o',
                       linewidth=linewidth, edgecolor=edgecolor)
            ax.set_xlabel(f'{coord_name} {i+1}')
            ax.set_ylabel(f'{coord_name} {j+1}')
            axes.append(ax)

        # Staple a colorbar to the right side
        cax = fig.add_subplot(gs[:, num_cols])
        fig.colorbar(mappable, cax=cax)

        # Save and potentially reload a previous range
        coord_range = getattr(self, f'{coord_column}_range')
        if coord_range is None:
            coord_range = {}

        for idx, (i, j) in enumerate(combos):
            xlim = coord_range.get(i)
            ylim = coord_range.get(j)
            if xlim is not None:
                axes[idx].set_xlim(xlim)
            if ylim is not None:
                axes[idx].set_ylim(ylim)

            if cache_plot_range:
                xlim = axes[idx].get_xlim()
                ylim = axes[idx].get_ylim()
                coord_range[i] = xlim
                coord_range[j] = ylim
        setattr(self, f'{coord_column}_range', coord_range)

        fig.suptitle(title)

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_reduction_elbow(self, plotfile: Optional[pathlib.Path] = None,
                             figsize: Tuple[float] = (12, 12),
                             coord_column: str = 'pca',
                             plot_percent_explained: bool = True,
                             axis1_color: str = 'black',
                             axis2_color: str = 'grey'):
        """ Plot the elbow plot for the PCA analysis

        .. code-block:: python

            proc.plot_reduction_elbow(plotdir / '21-100-pca_elbow.png')

        :param Path plotfile:
            Path to save the final plot to
        :param str coord_column:
            The name of the reduction coordinate to plot explained variance for
        :param bool plot_percent_explained:
            If True, plot % variance explained, not raw variance
        """

        if plot_percent_explained:
            ycoord = getattr(self, f'{coord_column}_explained_variance_ratio')*100
            ylabel = '% Variance Explained'
        else:
            ycoord = getattr(self, f'{coord_column}_explained_variance')
            ylabel = 'Variance Explained'
        ytotal = np.cumsum(ycoord)

        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = ax1.twinx()
        pca_num = np.arange(len(ycoord)) + 1.0

        ax1.plot(pca_num, ycoord, linestyle='-', marker='o', color=axis1_color,
                 linewidth=2, markersize=20)
        ax2.plot(pca_num, ytotal, linestyle='-', marker='o', color=axis2_color,
                 linewidth=2, markersize=20)

        ax1.set_ylim([0, np.max(ycoord)*1.1])
        ax2.set_ylim([0, np.max(ytotal)*1.1])
        ax1.tick_params(axis='y', labelcolor=axis1_color)
        ax2.tick_params(axis='y', labelcolor=axis2_color)

        coord_name = {
            'pca': 'Principal',
            'ica': 'Independent',
            'sparse_pca': 'Sparse Principal',
            'kernel_pca': 'Kernel Principal',
        }.get(coord_column, coord_column.replace('_', ' ').capitalize())

        ax1.set_xlabel(f'{coord_name} Component')
        ax1.set_ylabel(ylabel, color=axis1_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(True)
        ax1.spines['right'].set_visible(False)

        ax2.set_ylabel(f'Total {ylabel}', color=axis2_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(True)

        # Fiddle with the spines

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_clustering_scores(self,
                               plotfile: Optional[pathlib.Path] = None,
                               max_num_clusters: int = 10,
                               coord_column: str = 'pca',
                               random_state: int = 42,
                               figsize: Tuple[float] = (36, 12)):
        """ Make a cluster score plot for K-means clusters

        Plots the slihouette, Davies-Bouldin, and Calinski-Harabasz scores

        :param Path plotfile:
            The file to write the plot to
        :param int max_num_clusters:
            Try cluster numbers between 2 and this many
        :param str coord_column:
            Which coordinate column to cluster
        """
        print(f'Generating silhouette plots for {coord_column}')
        coords = self.get_coord_column(coord_column)

        num_clusters = []
        slihouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        for n_clusters in range(2, max_num_clusters):
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = clusterer.fit_predict(coords)

            num_clusters.append(n_clusters)
            slihouette_scores.append(silhouette_score(coords, cluster_labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(coords, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(coords, cluster_labels))

        num_clusters = np.array(num_clusters)
        slihouette_scores = np.array(slihouette_scores)
        calinski_harabasz_scores = np.array(calinski_harabasz_scores)
        davies_bouldin_scores = np.array(davies_bouldin_scores)

        slihouette_idx = np.argmax(slihouette_scores)
        calinski_harabasz_idx = np.argmax(calinski_harabasz_scores)
        davies_bouldin_idx = np.argmin(davies_bouldin_scores)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        ax1.plot(num_clusters, slihouette_scores, linestyle='-', marker='o',
                 linewidth=2, markersize=20)
        ax1.plot(num_clusters[slihouette_idx],
                 slihouette_scores[slihouette_idx], marker='o',
                 markersize=24, color='red')
        ax1.set_ylim([0, np.max(slihouette_scores)*1.1])
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title(f'Silhouette n={num_clusters[slihouette_idx]} ({slihouette_scores[slihouette_idx]:0.4f})')

        ax2.plot(num_clusters, calinski_harabasz_scores, linestyle='-', marker='o',
                 linewidth=2, markersize=20)
        ax2.plot(num_clusters[calinski_harabasz_idx],
                 calinski_harabasz_scores[calinski_harabasz_idx], marker='o',
                 markersize=24, color='red')
        ax2.set_ylim([0, np.max(calinski_harabasz_scores)*1.1])
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Calinski-Harabasz Score')
        ax2.set_title(f'Calinski-Harabasz n={num_clusters[calinski_harabasz_idx]} ({calinski_harabasz_scores[calinski_harabasz_idx]:0.4f})')

        ax3.plot(num_clusters, davies_bouldin_scores, linestyle='-', marker='o',
                 linewidth=2, markersize=20)
        ax3.plot(num_clusters[davies_bouldin_idx],
                 davies_bouldin_scores[davies_bouldin_idx], marker='o',
                 markersize=24, color='red')
        ax3.set_ylim([0, np.max(davies_bouldin_scores)*1.1])
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Davies-Bouldin Score')
        ax3.set_title(f'Davies-Bouldin n={num_clusters[davies_bouldin_idx]} ({davies_bouldin_scores[davies_bouldin_idx]:0.4f})')

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_cluster_term_accuracy(self,
                                   label_type: str,
                                   max_k: int = 10,
                                   test_size: float = 0.5,
                                   plotfile: Optional[pathlib.Path] = None,
                                   figsize: Tuple[float] = (10, 10),
                                   xticks: Optional[List[float]] = None,
                                   yticks: Optional[List[float]] = None,
                                   ylim: Optional[Tuple[float]] = None) -> pd.DataFrame:
        """ Plot the accuracy of using the best terms per cluster

        Plot the accuracy for the unbiased clustering:

        .. code-block:: python

            cluster_acc_df = proc.plot_cluster_term_accuracy(
                label_type='LabelKMeansPCA',
                plotfile=plotdir / '21-100-cluster_accuracy.png',
                max_k=10)

        Plot the accuracy for a set of treatments vs controls:

        .. code-block:: python

            treatment_acc_df = proc.plot_cluster_term_accuracy(
                label_type='Treatment',
                plotfile=plotdir / '21-100-treatment_accuracy.png',
                max_k=10)

        :param str label_type:
            Which cluster or label to use for the classes
        :param int max_k:
            Test all the possible feature predictors from k=1, 2,..., max_k
        :param Path plotfile:
            The plot file to write out
        :param List[float] xticks:
            Manually set the x ticks
        :param List[float] yticks:
            Manually set the y ticks
        :param Tuple[float] ylim:
            Manually set the y limits
        """
        labels = self.df[label_type].values
        data = self.transformed_block
        labels = LabelEncoder().fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42)

        num_features = []
        top_features = []
        scores = []
        for i in range(1, max_k):
            # Make a pipeline with a k-best selector and an Support Vector classifier
            pipeline = make_pipeline(
                SelectKBest(f_classif, k=i), SVC()
            )
            pipeline.fit(X_train, y_train)

            kbest_inds = pipeline.named_steps['selectkbest'].get_support(indices=True)
            kbest_columns = ','.join(str(self.value_columns[ind]) for ind in kbest_inds)

            # Stash the top features for later
            top_features.append(kbest_columns)
            num_features.append(i)
            scores.append(pipeline.score(X_test, y_test))

        data = pd.DataFrame({
            'NumFeatures': num_features,
            'Scores': scores,
            'TopFeatures': top_features,
        })
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(x='NumFeatures', y='Scores', data=data)
        ax.set_xlabel('Number of features')
        ax.set_ylabel('Score')
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()
        return data

    def plot_correlation(self, x: str, y: str, hue: Optional[str] = None,
                         plotfile: Optional[pathlib.Path] = None,
                         hue_order: Optional[List[str]] = None,
                         xlabel: Optional[str] = None,
                         ylabel: Optional[str] = None,
                         palette: str = 'Set1',
                         figsize: Tuple[float] = (12, 12),
                         xlim: Optional[Tuple[float]] = None,
                         ylim: Optional[Tuple[float]] = None,
                         normalized: bool = False):
        """ Plot the correlation between two features

        :param str x:
            The feature to plot on the x-axis
        :param str y:
            The feature to plot on the y-axis
        :param str hue:
            If not None, the column to use for hue data
        :param bool normalized:
            If True, use the normalized feature columns (otherwise, use the raw values)
        """

        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        # Load the normalized data if requested
        if normalized:
            xindex = self.value_columns.index(x)
            yindex = self.value_columns.index(y)
            df = pd.DataFrame({
                x: self.transformed_block[:, xindex],
                y: self.transformed_block[:, yindex],
            })
            if hue is not None:
                assert df.shape[0] == self.df.shape[0]
                df[hue] = self.df[hue].values
        else:
            # Otherwise, use the raw data
            df = self.df

        # Make the actual scatterplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.scatterplot(data=df, x=x, y=y, hue=hue, hue_order=hue_order,
                        palette=palette,
                        edgecolor='#C2C2C2', linewidth=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        plt.tight_layout()

        if plotfile is None:
            plt.show()
        else:
            plotfile = pathlib.Path(plotfile)
            plotfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plotfile, transparent=True)
            plt.close()

    def plot_feature(self, x: str, y: str, hue: Optional[str] = None,
                     merge_column: Optional[str] = None,
                     merge_function: str = 'mean',
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     comparisons: Optional[List[Tuple[int]]] = None,
                     edgecolor: str = EDGECOLOR,
                     linewidth: float = LINEWIDTH,
                     fillcolor: str = FILLCOLOR,
                     force_zero_ylim: bool = False,
                     ylim: Optional[Tuple[float]] = None,
                     **kwargs):
        """ Plot results from the per-surface data

        :param str x:
            The category column to plot
        :param str y:
            The feature column to plot
        :param str hue:
            The hue category to plot
        :param str merge_column:
            If not None, first merge the data over this column
        :param str merge_function:
            Which function to apply after pd.groupby
        :param \\*\\*kwargs:
            Other arguments to pass to :py:func:`plot_utils.plot_boxes`
        """

        df = self.df

        if merge_column is not None:
            group_columns = [merge_column, x]
            if hue is not None:
                group_columns.append(hue)

            # Group then apply the reduction
            grp = df[group_columns + [y]].groupby(group_columns)
            df = getattr(grp, merge_function)()
            df = df.reset_index()

        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y
        if force_zero_ylim and ylim is None:
            ylim = [0, None]

        plot_utils.plot_boxes(
            df=df, var_name=x, hue_var_name=hue, value_name=y,
            xlabel=xlabel,
            ylabel=ylabel,
            ylim=ylim,
            pvalue_comparisons=comparisons,
            edge_color=edgecolor,
            linewidth=linewidth,
            fill_color=fillcolor,
            **kwargs,
        )

    def plot_cluster_counts(self, label_type: str, x: str, y: str,
                            hue: Optional[str] = None,
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            filter_x_id: Optional[object] = None,
                            filter_hue_id: Optional[object] = None,
                            comparisons: Optional[List[Tuple[int]]] = None,
                            edgecolor: str = EDGECOLOR,
                            linewidth: float = LINEWIDTH,
                            fillcolor: str = FILLCOLOR,
                            force_zero_ylim: bool = False,
                            **kwargs):
        """ Plot results from the cluster counts data

        :param str label_type:
            The label to generate counts over (LabelKmeansPCA)
        :param str x:
            Column to use for the x-values
        :param str y:
            Column to use for the y-values
        :param str hue:
            If not None, column to use for the hue
        :param object filter_x_id:
            If not None, subset the x-column to only this value
        :param object filter_hue_id:
            If not None, subset the hue-column to only this value
        :param \\*\\*kwargs:
            Other arguments to pass to :py:func:`plot_utils.plot_boxes`
        """
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        if force_zero_ylim:
            ylim = (0, 100)
        else:
            ylim = None

        df = self.counts_df[self.counts_df['LabelType'] == label_type]
        all_label_types = np.unique(self.counts_df['LabelType'])
        if filter_x_id is not None:
            df = df[df[x] == filter_x_id]
            x = hue
            hue = None
        if filter_hue_id is not None:
            df = df[df[hue] == filter_hue_id]
            hue = None
        if x is None:
            raise ValueError('If filtering the x column, supply a value for "hue"')

        if df.shape[0] < 1:
            raise ValueError(f'No label type "{label_type}" in {all_label_types}')

        plot_utils.plot_boxes(
            df=df, var_name=x, hue_var_name=hue, value_name=y,
            xlabel=xlabel,
            ylabel=ylabel,
            pvalue_comparisons=comparisons,
            edge_color=edgecolor,
            linewidth=linewidth,
            fill_color=fillcolor,
            ylim=ylim,
            **kwargs
        )

    def write_column_descriptions(self, outfile: pathlib.Path, ylabels: Dict[str, str]):
        """ Write the column description table

        :param Path outfile:
            The excel file to write out
        :param str ylabels:
            The dictionary of column to description fields
        """
        df = pd.DataFrame({'Feature': self.value_columns})
        df['Feature'] = df['Feature'].astype(str)
        df['Description'] = df['Feature'].map(ylabels)

        # Fix the column labels
        df['Description'] = df['Description'].fillna('').astype(str)
        df['Description'] = df['Description'].map(lambda x: x.replace('$(\\mu m)$', '(μm)'))
        df['Description'] = df['Description'].map(lambda x: x.replace('$(\\mu m^{-1})$', '(μm^-1)'))
        df['Description'] = df['Description'].map(lambda x: x.replace('$(\\mu m^3)$', '(μm^3)'))
        df['Description'] = df['Description'].map(lambda x: x.replace('$(\\mu m^2)$', '(μm^2)'))

        df = df.sort_values('Feature')
        df.to_excel(outfile, index=False)

    def plot_all_cluster_counts(self,
                                label_type: str,
                                var_column: str,
                                animal_column: str,
                                plotdir: pathlib.Path,
                                var_palette: str = 'Set1',
                                timepoint_column: Optional[str] = None,
                                distance_column: Optional[str] = None,
                                distance_palette: Optional[str] = None,
                                batch_column: Optional[str] = None,
                                distance_order: Optional[List[str]] = None,
                                distance_xticklabels: Optional[List[str]] = None,
                                prefix: str = '',
                                suffix: str = '.png',
                                plot_style: str = 'boxes',
                                xlabel: str = '',
                                cluster_labels: Optional[Dict[str, str]] = None,
                                capsize: float = 0.2,
                                order: Optional[List[str]] = None,
                                xticklabels: Optional[List[str]] = None,
                                ignore_var_levels: Optional[List[str]] = None,
                                xticklabel_rotation: float = 0.0,
                                comparisons: Optional[List[Tuple[int]]] = None,
                                edgecolor: str = EDGECOLOR,
                                linewidth: float = LINEWIDTH,
                                figsize_x: float = 2.0,
                                force_zero_ylim: bool = False):
        """ Plot the cluster counts for all the different clusters

        :param str label_type:
            Which cluster to select from the counts data
        :param str var_column:
            Which column contains the category data
        :param str animal_column:
            Which column contains the animal ids
        :param Path plotdir:
            Directory to save the plots to
        :param str timepoint_column:
            If not None, which column contains the timepoint labels
        :param str batch_column:
            If not None, which column contains the batch labels
        :param str distance_column:
            If not None, which column contains the distance bin labels
        """
        if cluster_labels is None:
            cluster_labels = {}
        if plot_style in ('err', 'err_bar', 'err_bars'):
            err_bar_join = False
            err_bar_ci = 'sd'
        else:
            err_bar_join = True
            err_bar_ci = 95

        if batch_column is None:
            num_batches = 1
            batch_order = None
        else:
            batch_order = list(sorted(np.unique(self.counts_df[batch_column])))
            num_batches = len(batch_order)

        if timepoint_column is None:
            num_timepoint_columns = 1
            timepoint_order = None
        else:
            timepoint_order = list(sorted(np.unique(self.counts_df[timepoint_column])))
            num_timepoint_columns = len(timepoint_order)

        norm_var_column = reNORM.sub('_', var_column).lower().strip('_')

        for cluster_id in np.unique(self.counts_df['LabelID']):
            ylabel = cluster_labels.get(str(cluster_id), f'Cluster {cluster_id}')

            self.plot_cluster_counts(label_type=label_type,
                                     x=var_column, y='PercentCells',
                                     hue='LabelID', filter_hue_id=cluster_id,
                                     plotfile=plotdir / f'{prefix}{norm_var_column}_cluster{cluster_id}_percent{suffix}',
                                     order=order, xticklabels=xticklabels, ignore_var_levels=ignore_var_levels,
                                     ylabel=f'% Cells in {ylabel}',
                                     xlabel=xlabel,
                                     palette=var_palette,
                                     xticklabel_rotation=xticklabel_rotation,
                                     comparisons=comparisons,
                                     plot_style=plot_style,
                                     err_bar_ci=err_bar_ci,
                                     err_bar_join=err_bar_join,
                                     capsize=capsize,
                                     suffix=suffix,
                                     edgecolor=edgecolor,
                                     linewidth=linewidth,
                                     figsize_x=figsize_x,
                                     force_zero_ylim=force_zero_ylim)
            self.plot_cluster_counts(label_type=label_type,
                                     x=animal_column, y='PercentCells',
                                     hue='LabelID', filter_hue_id=cluster_id,
                                     plotfile=plotdir / f'{prefix}animal_cluster{cluster_id}_percent{suffix}',
                                     ylabel=f'% Cells in {ylabel}',
                                     xlabel=xlabel,
                                     xticklabel_rotation=xticklabel_rotation,
                                     plot_style='bars',
                                     capsize=capsize,
                                     suffix=suffix,
                                     edgecolor=edgecolor,
                                     linewidth=linewidth,
                                     figsize_x=figsize_x,
                                     force_zero_ylim=force_zero_ylim)
            if num_batches > 1:
                self.plot_cluster_counts(label_type=label_type,
                                         x=batch_column, y='PercentCells',
                                         hue='LabelID', filter_hue_id=cluster_id,
                                         plotfile=plotdir / f'{prefix}batch_cluster{cluster_id}_percent{suffix}',
                                         ylabel=f'% Cells in {ylabel}',
                                         xlabel=xlabel, order=batch_order,
                                         plot_style='bars',
                                         capsize=capsize,
                                         suffix=suffix,
                                         edgecolor=edgecolor,
                                         linewidth=linewidth,
                                         figsize_x=figsize_x,
                                         force_zero_ylim=force_zero_ylim)
            if distance_column is not None:
                print(distance_order)
                print(distance_xticklabels)
                self.plot_cluster_counts(label_type=label_type,
                                         x=distance_column, y='PercentCells',
                                         hue='LabelID', filter_hue_id=cluster_id,
                                         plotfile=plotdir / f'{prefix}dist_cluster{cluster_id}_percent{suffix}',
                                         ylabel=f'% Cells in {ylabel}',
                                         xlabel=xlabel, order=distance_order,
                                         xticklabels=distance_xticklabels,
                                         plot_style='bars',
                                         capsize=capsize,
                                         suffix=suffix,
                                         palette=distance_palette,
                                         edgecolor=edgecolor,
                                         linewidth=linewidth,
                                         figsize_x=figsize_x,
                                         force_zero_ylim=force_zero_ylim)

        # Show the timepoint data
        if num_timepoint_columns > 1:
            all_df = self.counts_df[self.counts_df['LabelType'] == label_type]

            if force_zero_ylim:
                ylim = (0, 100)
            else:
                ylim = None
            if xlabel in (None, ''):
                timepoint_xlabel = 'Timepoint'
            else:
                timepoint_xlabel = f'{xlabel} Timepoint'
            plot_utils.plot_boxes(
                df=all_df, var_name=timepoint_column,
                hue_var_name='LabelID',
                value_name='PercentCells',
                xlabel=timepoint_xlabel,
                order=timepoint_order,
                ylabel='% Cells in Cluster',
                plot_style='lines',
                showfliers=False,
                suffix=suffix,
                plotfile=plotdir / f'{prefix}tp-total_cluster_percent{suffix}',
                edge_color=edgecolor,
                linewidth=linewidth,
                ylim=ylim,
            )
            for i, treatment in enumerate(order):
                df = all_df[all_df[var_column] == treatment]
                plot_utils.plot_boxes(
                    df=df, var_name=timepoint_column,
                    hue_var_name='LabelID',
                    value_name='PercentCells',
                    xlabel=timepoint_xlabel,
                    order=timepoint_order,
                    ylabel=f'{treatment} % Cells in Cluster',
                    plot_style='lines',
                    showfliers=False,
                    suffix=suffix,
                    edge_color=edgecolor,
                    linewidth=linewidth,
                    ylim=ylim,
                    plotfile=plotdir / f'{prefix}tp-{norm_var_column}{i}_total_cluster_percent{suffix}',
                )
            for cluster_id in np.unique(all_df['LabelID']):
                ylabel = cluster_labels.get(str(cluster_id), f'Cluster {cluster_id}')
                df = all_df[all_df['LabelID'] == cluster_id]
                plot_utils.plot_boxes(
                    df=df, var_name=timepoint_column,
                    hue_var_name=var_column, hue_order=order,
                    order=timepoint_order,
                    palette=var_palette,
                    value_name='PercentCells',
                    xlabel=timepoint_xlabel,
                    ylabel=f'% Cells in {ylabel}',
                    plot_style='lines',
                    showfliers=False,
                    suffix=suffix,
                    edge_color=edgecolor,
                    linewidth=linewidth,
                    ylim=ylim,
                    plotfile=plotdir / f'{prefix}tp-{norm_var_column}_cluster{cluster_id}_percent{suffix}',
                )

# Functions


def to_json_types(data: object) -> object:
    """ Convert to a json object """
    if data in (True, False, None):
        return data
    if isinstance(data, dict):
        return {str(k): to_json_types(v) for k, v in data.items()}
    if isinstance(data, (tuple, list, set, np.ndarray)):
        return [to_json_types(v) for v in data]
    if isinstance(data, (str, pathlib.Path)):
        return str(data)
    data_type = getattr(data, 'dtype', type(data))
    if np.issubdtype(data_type, np.integer):
        return int(data)
    if np.issubdtype(data_type, np.floating):
        return float(data)
    raise ValueError(f'Unknown conversion for data type {type(data)}: {data}')


def calc_count_df(df: pd.DataFrame,
                  label_column: str,
                  id_columns: Optional[List[str]] = None,
                  count_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """ Take a dataframe of individual rows and convert to counts

    :param DataFrame df:
        The data frame with one record per row
    :param str label_column:
        The column with individual levels to count over
    :param list[str] id_columns:
        If not None, additional columns to disaggregate by (condition, animal id, etc)
    :param list[str] count_columns:
        If not None, additional columns to sum over
    :returns:
        A dataframe consisting of counts and percents for each level in label_column
    """
    if id_columns is None:
        id_columns = []
    elif isinstance(id_columns, str):
        id_columns = [id_columns]
    if count_columns is None:
        count_columns = []
    elif isinstance(count_columns, str):
        count_columns = [count_columns]

    # Rename indicator columns to total columns
    count_renames = {}
    for count_column in count_columns:
        if count_column.startswith('Is'):
            count_rename = f'Total{count_column[2:]}'
        else:
            count_rename = f'Total{count_column}'
        count_renames[count_column] = count_rename

    # Add a mock column if we got no ID columns to total over
    df = df.copy()
    if id_columns == []:
        df['FakeID'] = 1
        id_columns = ['FakeID']

    block_df = df[id_columns + [label_column] + count_columns].copy()
    block_df['CountCells'] = 1

    # Define an explicit multi-index to capture clusters with no members
    # This calculates all unique levels for the compound ids (but is inefficient)
    count_ids = {tuple(rec.values) for _, rec in df[id_columns].iterrows()}
    final_ids = []
    for level in np.unique(df[label_column]):
        final_ids.extend(tuple(list(rec)+[level]) for rec in count_ids)
    count_midx = pd.MultiIndex.from_tuples(final_ids, names=(id_columns + [label_column]))

    # Sum once over all id columns + label = counts/label
    count_df = block_df.groupby(id_columns + [label_column]).sum()
    count_df = count_df.reindex(count_midx, fill_value=0)
    count_df = count_df.reset_index()
    count_df = count_df.rename(columns=count_renames)

    # Sum again over all id columns only = total
    total_df = block_df.groupby(id_columns).sum()
    total_df = total_df.reset_index()
    total_df['TotalCells'] = total_df['CountCells']

    count_df = count_df.merge(total_df, on=id_columns, how='left', suffixes=('', '_y'))
    count_df['PercentCells'] = count_df['CountCells'] / count_df['TotalCells']*100
    count_df['PercentCells'] = count_df['PercentCells'].fillna(0).replace([np.inf, -np.inf], 0)

    # For all the count columns, also calculate a per-label percent
    id_columns = [column for column in id_columns if column != 'FakeID']
    final_columns = id_columns + ['LabelType', 'LabelID', 'CountCells', 'TotalCells', 'PercentCells']
    for count_total_col in count_renames.values():
        count_percent_col = f'Percent{count_total_col[5:]}'
        count_df[count_percent_col] = count_df[count_total_col] / count_df['TotalCells']*100
        count_df[count_percent_col] = count_df[count_percent_col].fillna(0).replace([np.inf, -np.inf], 0)
        final_columns.extend([count_total_col, count_percent_col])

    count_df['LabelType'] = label_column
    count_df = count_df.rename(columns={label_column: 'LabelID'})
    return count_df[final_columns]


def calc_drop_columns(stats_df: pd.DataFrame,
                      drop_columns: Optional[List[str]] = None,
                      drop_intensity_features: bool = False,
                      drop_norm_intensity_features: bool = False,
                      drop_shell_core_features: bool = False,
                      drop_centile_features: bool = False,
                      drop_regression_stat_features: bool = False,
                      drop_distance_features: bool = False,
                      drop_volume_features: bool = False,
                      drop_contain_features: bool = False,
                      drop_channels: Optional[List[int]] = None) -> List[str]:
    """ Calculate which columns to drop

    :param DataFrame stats_df:
        The dataframe to look at for column information
    :param List[str] drop_columns:
        A list of columns to drop manually
    :param bool drop_intensity_features:
        If True, drop all columns with raw intensity measurements
    :param bool drop_norm_intensity_features:
        If True, drop all columns with normalized intensity measurements
    :param bool drop_shell_core_features:
        If True, drop the core and shell intensity features
    :param bool drop_centile_features:
        If True, drop the 25th and 75th centile intensity features
    :param bool drop_regression_stat_features:
        If True, drop the regression coefficient features (R**2 values)
    :param bool drop_distance_features:
        If True, drop columns that have distance measurements
    :param bool drop_volume_features:
        If True, drop columns that have volumetric stats
    :param list[int] drop_channels:
        A list of 1-indexed channel numbers to drop
    :returns:
        A list of column names to drop
    """

    if drop_columns is None:
        drop_columns = []
    elif isinstance(drop_columns, str):
        drop_columns = [drop_columns]
    drop_columns.extend([
        'CenterX', 'CenterY', 'CenterZ',
        'MinX', 'MinY', 'MinZ',
        'MaxX', 'MaxY', 'MaxZ',
        'NumVoxels', 'NumSurfaceVoxels', 'NumCoreVoxels', 'SurfaceRatio',
    ])
    drop_columns.extend(c for c in stats_df.columns
                        if c.startswith(('IntensityShellNumVoxels_', 'IntensityCoreNumVoxels_')))
    drop_columns.extend(c for c in stats_df.columns
                        if 'Min' in c or 'Max' in c or 'Pct95' in c or 'Pct05' in c)
    drop_columns.extend(c for c in stats_df.columns if c.startswith('Is'))
    if drop_intensity_features:
        drop_columns.extend(c for c in stats_df.columns if c.startswith('Intensity'))
    if drop_norm_intensity_features:
        drop_columns.extend(c for c in stats_df.columns if c.startswith('NormIntensity'))
    if drop_shell_core_features:
        drop_columns.extend(c for c in stats_df.columns if c.startswith(('IntensityShell', 'IntensityCore')))
    if drop_centile_features:
        drop_columns.extend(c for c in stats_df.columns
                            if 'Intensity' in c and ('Pct25' in c or 'Pct75' in c))
    if drop_distance_features:
        drop_columns.extend(c for c in stats_df.columns
                            if c.startswith('Dist') and '_Ch=' in c)
    if drop_regression_stat_features:
        drop_columns.extend(c for c in stats_df.columns if c.endswith('R2') or 'R2_Ch=' in c)
    if drop_volume_features:
        drop_columns.extend(c for c in stats_df.columns if c in ('Volume', 'BBoxVolume', 'ConvexVolume'))
        drop_columns.extend(c for c in stats_df.columns if c in ('SurfaceArea', 'ConvexSurfaceArea'))
        drop_columns.extend(c for c in stats_df.columns if c.endswith('Radius'))
        drop_columns.extend(c for c in stats_df.columns if c.startswith('Num') and c.endswith('Voxels'))
        drop_columns.extend(c for c in stats_df.columns if c.startswith(('Ellipse', 'BBox')) and c.endswith('Axis'))
        drop_columns.extend(c for c in stats_df.columns if c.endswith('BranchLen'))
    if drop_contain_features:
        drop_columns.extend(c for c in stats_df.columns if c.startswith('Contain'))

    # Drop any channels the user requests
    if drop_channels is None:
        drop_channels = []
    elif isinstance(drop_channels, (int, float, str)):
        drop_channels = [drop_channels]
    drop_channels = [int(c) for c in drop_channels]

    for channel_idx in drop_channels:
        drop_columns.extend(c for c in stats_df.columns if c.endswith(f'_Ch={channel_idx}'))
    return drop_columns


def plot_all_image_dists(image_df: pd.DataFrame,
                         plotdir: pathlib.Path,
                         var_column: str,
                         prefix: str = '',
                         suffix: str = '.png',
                         var_palette: str = 'Set1',
                         animal_column: Optional[str] = None,
                         timepoint_column: Optional[str] = None,
                         order: Optional[List[str]] = None,
                         xticklabels: Optional[List[str]] = None,
                         xticklabel_rotation: float = 0.0,
                         plot_style: str = 'boxes',
                         plot_err_capsize: float = 0.2,
                         force_zero_ylim: bool = False,
                         comparisons: Optional[List[Tuple[int]]] = None,
                         channel_names: Optional[Dict[str, str]] = None,
                         edgecolor: str = EDGECOLOR,
                         linewidth: float = LINEWIDTH,
                         fillcolor: str = FILLCOLOR):
    """ Plot all the whole image distributions

    :param DataFrame image_df:
        The whole image data frame
    :param Path plotdir:
        The directory to write plots to
    :param str var_column:
        Column containing the categorical variable
    :param str timepoint_column:
        If not None, column containing the time variable
    :param list[str] order:
        If not None, order for the categorical variable
    :param list[str] xticklabels:
        If not None, labels for the categorical variable
    :param float xticklabel_rotation:
        How much to rotate the x tick labels
    :param list[tuple[int]] comparisons:
        If not None, the list of index pairs to compare
    :param dict[str, str] channel_names:
        If not None, the dictionary mapping channel numbers to names (e.g. 'Ch=3': 'Iba1')
    """
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)
    if channel_names is None:
        channel_names = {}

    total_voxels = image_df['TotalVoxels']
    mean_columns = [c for c in image_df.columns if c.startswith('VolumeIntensityMean_')]
    all_channels = [c.split('_', 1)[1] for c in mean_columns]

    plot_channels = []
    plot_labels = []

    # Back out the percent of each signal in each compartment
    for mean_column in mean_columns:
        mean_intensity = image_df[mean_column]
        dst_channel = mean_column.rsplit('_', 1)[1]
        dst_channel_name = channel_names.get(dst_channel, dst_channel)

        plot_channels.append(mean_column)
        plot_labels.append(f'Mean {dst_channel_name} (AU)')

        total_column = f'VolumeIntensitySum_{dst_channel}'
        image_df[total_column] = mean_intensity*total_voxels

        plot_channels.append(total_column)
        plot_labels.append(f'Total {dst_channel_name} (AU)')

        for src_channel in all_channels:
            column = f'IntensityMean_{dst_channel}In_{src_channel}'
            if column not in image_df.columns:
                continue

            # Calculate percent signal in this compartment
            src_mean_intensity = image_df[column]
            src_total_voxels = image_df[f'MaskVoxels_{src_channel}']
            intensity_ratio = src_mean_intensity/mean_intensity
            volume_ratio = src_total_voxels/total_voxels

            out_column = f'Percent_{dst_channel}In_{src_channel}'
            image_df[out_column] = intensity_ratio*volume_ratio*100

            src_channel_name = channel_names.get(src_channel, src_channel)

            # Add the percent plot to the plot list
            out_column_name = f'% {dst_channel_name} in {src_channel_name} Surface'
            plot_channels.append(out_column)
            plot_labels.append(out_column_name)

            # Add a percent of the image plot
            fraction_column = f'PercentOfVolume_{src_channel}'
            if fraction_column not in image_df.columns:
                image_df[fraction_column] = volume_ratio*100
                plot_channels.append(fraction_column)
                plot_labels.append(f'% {src_channel_name} Surface in Volume')

            # Add mean and total plots to the plot list
            plot_channels.append(column)
            plot_labels.append(f'Mean {dst_channel_name} in {src_channel_name} Surface (AU)')

            total_column = f'Total_{dst_channel}In_{src_channel}'
            image_df[total_column] = src_mean_intensity*src_total_voxels

            plot_channels.append(total_column)
            plot_labels.append(f'Total {dst_channel_name} in {src_channel_name} Surface (AU)')

    # Make a set of per-animal plots
    for ycolumn, ylabel in zip(plot_channels, plot_labels):
        outname = reNORM.sub('_', ycolumn).lower().strip('_')
        animal_df = image_df[[animal_column, ycolumn]].groupby(
            [animal_column], as_index=False).mean()
        animal_order = list(sorted(image_df[animal_column].unique()))
        if force_zero_ylim:
            ylim = [0, None]
        else:
            ylim = None
        plot_utils.plot_boxes(animal_df, plotfile=plotdir / f'{prefix}animal-{outname}.png',
                              var_name=animal_column,
                              value_name=ycolumn,
                              ylabel=ylabel,
                              ylim=ylim,
                              order=animal_order,
                              xticklabel_rotation=90,
                              plot_style='bars',
                              palette='tab20b',
                              capsize=plot_err_capsize,
                              edge_color=edgecolor,
                              linewidth=linewidth,
                              fill_color=fillcolor,
                              showfliers=True,
                              err_bar_join=False,
                              suffix=suffix)

    # Make one set of summary plots over all animals
    assert len(plot_channels) == len(plot_labels)
    for ycolumn, ylabel in zip(plot_channels, plot_labels):
        outname = reNORM.sub('_', ycolumn).lower().strip('_')
        mean_df = image_df[[var_column, animal_column, ycolumn]].groupby(
            [var_column, animal_column], as_index=False).mean()

        if force_zero_ylim:
            ylim = [0, None]
        else:
            ylim = None

        plot_utils.plot_boxes(mean_df, plotfile=plotdir / f'{prefix}{outname}.png',
                              var_name=var_column,
                              value_name=ycolumn,
                              ylabel=ylabel,
                              ylim=ylim,
                              order=order,
                              xticklabels=xticklabels,
                              xticklabel_rotation=xticklabel_rotation,
                              pvalue_comparisons=comparisons,
                              plot_style=plot_style,
                              palette=var_palette,
                              capsize=plot_err_capsize,
                              edge_color=edgecolor,
                              linewidth=linewidth,
                              fill_color=fillcolor,
                              showfliers=True,
                              err_bar_join=False,
                              suffix=suffix)

    # Timepoint stratified data
    if timepoint_column is None:
        num_timepoint_columns = 1
    else:
        num_timepoint_columns = len(np.unique(image_df[timepoint_column]))

    if num_timepoint_columns > 1:
        for ycolumn, ylabel in zip(plot_channels, plot_labels):
            outname = reNORM.sub('_', ycolumn).lower().strip('_')

            if force_zero_ylim:
                ylim = [0, None]
            else:
                ylim = None

            plot_utils.plot_boxes(image_df, plotfile=plotdir / f'tp-{prefix}{outname}.png',
                                  var_name=timepoint_column,
                                  hue_var_name=var_column,
                                  palette=var_palette,
                                  value_name=ycolumn,
                                  title=ylabel,
                                  ylabel=ylabel,
                                  ylim=ylim,
                                  xlabel='Timepoint',
                                  hue_order=order,
                                  plot_style='lines',
                                  showfliers=False,
                                  hide_legend=True,
                                  suffix=suffix)


def run_single_cell_pipeline(stats_df: pd.DataFrame,
                             outdir: pathlib.Path,
                             var_column: str,
                             image_df: Optional[pd.DataFrame] = None,
                             image_norm_method: str = 'div',
                             animal_column: str = 'Animal ID',
                             var_palette: str = 'Set1',
                             distance_palette: str = 'inferno_r',
                             timepoint_palette: str = 'inferno',
                             timepoint_column: Optional[str] = None,
                             distance_column: Optional[str] = None,
                             distance_bins: Optional[List[float]] = None,
                             image_column: Optional[str] = None,
                             batch_column: Optional[str] = None,
                             drop_channels: Optional[List[int]] = None,
                             drop_columns: Optional[List[str]] = None,
                             drop_intensity_features: bool = False,
                             drop_norm_intensity_features: bool = False,
                             drop_centile_features: bool = False,
                             drop_shell_core_features: bool = False,
                             drop_volume_features: bool = False,
                             drop_regression_stat_features: bool = True,
                             drop_distance_features: bool = False,
                             drop_contain_features: bool = False,
                             bad_columns: Optional[List[str]] = None,
                             mean_columns: Optional[List[str]] = None,
                             sum_columns: Optional[List[str]] = None,
                             num_kmeans_clusters: int = 5,
                             additional_top_k_features: int = 3,
                             max_k_accuracy_terms: int = 10,
                             normalization_type: str = 'quantile',
                             reduction_type: str = 'pca',
                             num_reduction_components: int = 8,
                             reduction_kwargs: Optional[Dict[str, object]] = None,
                             cluster_coord_column: Optional[str] = None,
                             projection_coord_column: Optional[str] = None,
                             projection_type: str = 'umap',
                             projection_kwargs: Optional[Dict[str, object]] = None,
                             comparisons: Optional[List[Tuple[int]]] = None,
                             order: Optional[List[str]] = None,
                             xticklabels: Optional[List[str]] = None,
                             xticklabel_rotation: float = 0.0,
                             volcano_kwargs: Optional[Dict[str, object]] = None,
                             prefix: str = '',
                             suffix: str = '.png',
                             plot_style: str = 'boxes',
                             plot_err_capsize: float = 0.2,
                             force_zero_ylim: bool = False,
                             linewidth: float = LINEWIDTH,
                             markersize: float = MARKERSIZE,
                             ylims: Optional[Dict[str, Tuple[float]]] = None,
                             ylabels: Optional[Dict[str, str]] = None,
                             channel_names: Optional[Dict[str, str]] = None,
                             cluster_labels: Optional[Dict[str, str]] = None,
                             overwrite: bool = False,
                             skip_feature_plots: bool = False,
                             skip_whole_image_plots: bool = False) -> SingleCellMorphoPipeline:
    """ Run a standardized version of the analysis

    :param DataFrame stats_df:
        The data frame containing all surface features for microglia
    :param Path outdir:
        The directory to write the output results to
    :param str var_column:
        The column containing the independent variable (treatment, genotype, etc)
    :param DataFrame image_df:
        If not None, the data frame containing all the whole image features
    :param str animal_column:
        The column containing the animal ids
    :param str timepoint_column:
        If not None, the column containing timepoints to analyze
    :param str distance_column:
        If not None, the column containing distance measurements to analyze
    :param list[int] drop_channels:
        If not None, drop any intensity data for these channels (e.g. drop DAPI)
    :param str normalization_type:
        The method to use for data normalization ('standard', 'robust', 'quantile', 'power')
    :param str reduction_type:
        What kind of dimensionality reduction to use (pca/ica)
    :param int num_reduction_components:
        Number of PCA/ICA components to use
    :param int num_kmeans_clusters:
        Number of clusters to generate by kmeans
    :param int additional_top_k_features:
        In addition to the selected mean and sum features, also include this many
        unbiased features from the one-vs-rest analysis
    :param int max_k_accuracy_terms:
        Maximum number of combinatorial terms to try when explaining cluster and
        treatment accuracy
    :param str cluster_coord_column:
        The coordinate column to use to calculate clusters
    :param list[tuple[int]] comparisons:
        If not None, the set of comparisons to generate
    :param list[str] order:
        Order to plot the var column elements in
    :param list[str] xticklabels:
        Ticklabels corresponding to the elements in order
    :param str plot_style:
        Style to use when generating the categorical plots
    :param str projection_coord_column:
        Which column to use to generate the non-linear projection to 2D
    :param str projection_type:
        Which type of non-linear projection to use ('umap' or 'tsne')
    :param dict projection_kwargs:
        Additional keyword arguments to pass to the projection method
    :returns:
        The pipeline allowing for any additional analysis
    """
    # Load the default kwargs for reduction/projection
    if projection_kwargs is None:
        projection_kwargs = {}
    if reduction_kwargs is None:
        reduction_kwargs = {}
    if cluster_labels is None:
        cluster_labels = {}
    if channel_names is None:
        channel_names = {}
    if volcano_kwargs is None:
        volcano_kwargs = {'label_top_k': 10}
    if 'approach' not in volcano_kwargs:
        volcano_kwargs['approach'] = '1-vs-rest'
    if ylims is None:
        ylims = {}
    if ylabels is None:
        ylabels = {}

    # By default, project and cluster in reduction space
    if cluster_coord_column is None:
        cluster_coord_column = reduction_type
    if projection_coord_column is None:
        projection_coord_column = reduction_type

    # Generate nice names for the y axis
    ylabels = qc_utils.load_ylabels(channel_names=channel_names, ylabels=ylabels)

    # Create the output directory
    outdir = pathlib.Path(outdir)
    if overwrite and outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check the animal ID list and the treatment list
    all_animal_ids = list(sorted(np.unique(stats_df[animal_column])))
    print(f'Animal IDs ({len(all_animal_ids)}): {all_animal_ids}')

    stats_df = qc_utils.calc_extended_values(
        stats_df, image_df=image_df, norm_method=image_norm_method)
    categories = set(np.unique(stats_df[var_column]))
    category_animal_map = {}
    for category in categories:
        category_df = stats_df[stats_df[var_column] == category]
        category_animal_map[category] = list(sorted(np.unique(category_df[animal_column])))

    print(f'Got categories ({len(categories)}): {categories}')
    print(f'Category animal map: {category_animal_map}')

    # If we got a timepoint column, also look through the unique columns for that
    if timepoint_column is not None:
        timepoints = set(np.unique(stats_df[timepoint_column]))
        timepoint_animal_map = {}
        for timepoint in timepoints:
            timepoint_df = stats_df[stats_df[timepoint_column] == timepoint]
            timepoint_animal_map[timepoint] = list(sorted(np.unique(timepoint_df[animal_column])))

        print(f'Got timepoints ({len(timepoints)}): {timepoints}')
        print(f'Timepoint animal map: {timepoint_animal_map}')

    # If we got a distance column
    if distance_column is not None:
        drop_distance_features = True
        stats_df['DistanceBin'] = qc_utils.bin_by_dist(
            stats_df[distance_column].values,
            bin_edges=distance_bins)

        distance_order = []
        orig_distance_column = distance_column
        distance_column = 'DistanceBin'
        distance_xticklabels = []
        for i, (dist_st, dist_ed) in enumerate(zip(distance_bins[:-1], distance_bins[1:])):
            distance_xticklabels.append(f'{int(dist_st)}-{int(dist_ed)}')
            distance_order.append(i)
        distance_xticklabels.append(f'>{int(dist_ed)}')
        distance_order.append(len(distance_xticklabels)-1)
        print(f'Distance column "{orig_distance_column}" binned into groups: {distance_xticklabels}')
    else:
        distance_order = None
        distance_xticklabels = None

    norm_var_column = reNORM.sub('_', var_column).lower().strip('_')

    # Validate that order and xticklabels are sane
    if order is None:
        order = list(sorted(categories))
    if xticklabels is None:
        xticklabels = order

    if set(order) != categories:
        raise ValueError(f'Got order categories {order} but actually have {categories}')
    if len(order) != len(xticklabels):
        raise ValueError(f'Got order {order} but xticklabels {xticklabels}')

    # Run some whole-image analysis
    if image_df is not None and not skip_whole_image_plots:
        plot_all_image_dists(
            image_df=image_df,
            plotdir=outdir / 'whole_image_plots',
            var_column=var_column,
            var_palette=var_palette,
            prefix=f'wi-{prefix}',
            suffix=suffix,
            animal_column=animal_column,
            timepoint_column=timepoint_column,
            order=order,
            xticklabels=xticklabels,
            xticklabel_rotation=xticklabel_rotation,
            comparisons=comparisons,
            plot_style=plot_style,
            force_zero_ylim=force_zero_ylim,
            plot_err_capsize=plot_err_capsize,
            channel_names=channel_names)

    # Run the single cell analysis
    plotdir = outdir / 'single_cell_plots'
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    # Drop columns that don't make sense to analyze, or are too noisy
    drop_columns = calc_drop_columns(
        stats_df,
        drop_columns=drop_columns,
        drop_intensity_features=drop_intensity_features,
        drop_norm_intensity_features=drop_norm_intensity_features,
        drop_shell_core_features=drop_shell_core_features,
        drop_centile_features=drop_centile_features,
        drop_regression_stat_features=drop_regression_stat_features,
        drop_volume_features=drop_volume_features,
        drop_distance_features=drop_distance_features,
        drop_contain_features=drop_contain_features,
        drop_channels=drop_channels,
    )

    # Categories are at least the independent variable, and the animal ID
    category_columns = [var_column, animal_column]
    if batch_column is not None:
        category_columns.append(batch_column)
        num_batch_columns = len(np.unique(stats_df[batch_column]))
    else:
        num_batch_columns = 1
    if image_column is not None:
        category_columns.append(image_column)
    if timepoint_column is not None:
        category_columns.append(timepoint_column)
        num_timepoint_columns = len(np.unique(stats_df[timepoint_column]))
    else:
        num_timepoint_columns = 1
    if distance_column is not None:
        category_columns.append(distance_column)

    print(f'Final category columns: {category_columns}')
    drop_columns = [c for c in drop_columns if c not in category_columns]
    print(f'Dropping final columns: {drop_columns}')

    # Save the arguments to this function
    try:
        all_arguments = to_json_types({k: v for k, v in locals().items()
                                       if not isinstance(v, pd.DataFrame)})
        argument_file = outdir / 'all_params.json'
        with argument_file.open('wt') as fp:
            json.dump(all_arguments, fp, sort_keys=True, indent=4, default=to_json_types)
        print(f'Saved local parameters to {argument_file}')
    except Exception:
        print(f'Failed to save local parameters to {argument_file}')
        traceback.print_exc()
        print('Trying to continue anyway...')

    # Load the pipeline
    proc = SingleCellMorphoPipeline(stats_df)
    proc.drop_columns(drop_columns)

    # Cluster the data to look for treatment blocks based on single values
    proc.set_value_columns(category_columns=category_columns,
                           bad_columns=bad_columns)
    proc.calc_palette_column(column=animal_column, palette='tab20b')
    proc.set_palette_column(column=var_column, palette=var_palette,
                            order=order, xticklabels=xticklabels)
    if batch_column is not None:
        proc.calc_palette_column(column=batch_column, palette='Set1')
    if image_column is not None:
        proc.calc_palette_column(column=image_column, palette='Set1')
    if timepoint_column is not None:
        proc.calc_palette_column(column=timepoint_column, palette=timepoint_palette)
    if distance_column is not None:
        proc.calc_palette_column(column=distance_column, palette=distance_palette,
                                 order=distance_order, xticklabels=distance_xticklabels)

    proc.plot_legend(column=var_column, plotfile=plotdir / f'{prefix}{norm_var_column}-legend{suffix}')
    proc.plot_legend(column=animal_column, plotfile=plotdir / f'{prefix}animal-legend{suffix}')
    if num_batch_columns > 1:
        proc.plot_legend(column=batch_column, plotfile=plotdir / f'{prefix}batch-legend{suffix}')
    if num_timepoint_columns > 1:
        proc.plot_legend(column=timepoint_column, plotfile=plotdir / f'{prefix}timepoint-legend{suffix}')
    if distance_column is not None:
        distance_labels = {str(k): str(v) for k, v in zip(distance_order, distance_xticklabels)}
        proc.plot_legend(column=distance_column, plotfile=plotdir / f'{prefix}distance-legend{suffix}',
                         cluster_labels=distance_labels)

    # Transform the data and look at per-feature redundancy
    proc.normalize_data(transformer=normalization_type,
                        outfile=plotdir / f'{prefix}normalized.xlsx')
    proc.write_column_descriptions(plotdir / f'{prefix}feature-desc.xlsx',
                                   ylabels=ylabels)

    proc.plot_clustermap(plotdir / f'{prefix}{norm_var_column}_heatmap{suffix}',
                         row_order=order)

    if num_batch_columns > 1:
        batch_order = list(sorted(np.unique(stats_df[batch_column])))
        proc.plot_clustermap(plotdir / f'{prefix}batch_heatmap{suffix}',
                             palette_column=batch_column,
                             row_order=batch_order)

    if num_timepoint_columns > 1:
        timepoint_order = list(sorted(np.unique(stats_df[timepoint_column])))
        proc.plot_clustermap(plotdir / f'{prefix}timepoint_heatmap{suffix}',
                             palette_column=timepoint_column,
                             row_order=timepoint_order)

    if distance_column is not None:
        proc.plot_clustermap(plotdir / f'{prefix}distance_heatmap{suffix}',
                             palette_column=distance_column,
                             row_order=distance_order)

    # Reduce dimensionality
    if reduction_type == 'pca':
        proc.project_pca(n_components=num_reduction_components, **reduction_kwargs)
    elif reduction_type == 'ica':
        proc.project_ica(n_components=num_reduction_components, **reduction_kwargs)
    elif reduction_type == 'sparse_pca':
        proc.project_sparse_pca(n_components=num_reduction_components, **reduction_kwargs)
    elif reduction_type == 'kernel_pca':
        proc.project_kernel_pca(n_components=num_reduction_components, **reduction_kwargs)
    elif reduction_type == 'ca':
        proc.project_ca(n_components=num_reduction_components, **reduction_kwargs)
    else:
        raise ValueError(f'Unknown reduction type: {reduction_type}')

    # Calculate measures of how good the projection is
    proc.plot_scatter(plotdir / f'{prefix}{reduction_type}{suffix}',
                      coord_column=reduction_type,
                      figsize=(24, 24),
                      linewidth=linewidth,
                      markersize=markersize)

    proc.calc_explained_variance(coord_column=reduction_type)

    proc.plot_reduction_elbow(
        plotfile=plotdir / f'{prefix}{reduction_type}_elbow{suffix}',
        coord_column=reduction_type,
        plot_percent_explained=False,
    )
    proc.plot_reduction_elbow(
        plotfile=plotdir / f'{prefix}{reduction_type}_pct_elbow{suffix}',
        coord_column=reduction_type,
        plot_percent_explained=True,
    )

    proc.calc_reduction_loadings(coord_column=reduction_type)

    # Project onto low dimensional space
    if projection_type == 'umap':
        proc.project_umap(coord_column=projection_coord_column, **projection_kwargs)
    elif projection_type == 'tsne':
        proc.project_tsne(coord_column=projection_coord_column, **projection_kwargs)
    else:
        raise ValueError(f'Unknown projection type: {projection_type}')

    # Plot the UMAP clusters by variable
    proc.plot_scatter(plotdir / f'{prefix}{projection_type}-{norm_var_column}{suffix}',
                      coord_column=projection_type,
                      palette_column=var_column,
                      cache_plot_range=True,
                      linewidth=linewidth,
                      markersize=markersize)
    proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-{norm_var_column}{suffix}',
                      coord_column=projection_type,
                      merge_column=var_column,
                      linewidth=linewidth,
                      markersize=markersize)

    # Plot the UMAP clusters by animal
    proc.plot_scatter(plotdir / f'{prefix}{projection_type}-animal{suffix}',
                      coord_column=projection_type,
                      palette_column=animal_column,
                      linewidth=linewidth,
                      markersize=markersize)
    proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-animal{suffix}',
                      coord_column=projection_type,
                      merge_column=animal_column,
                      linewidth=linewidth,
                      markersize=markersize)

    # Plot the UMAP clusters by batch
    if num_batch_columns > 1:
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-batch{suffix}',
                          coord_column=projection_type,
                          palette_column=batch_column,
                          linewidth=linewidth,
                          markersize=markersize)
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-batch{suffix}',
                          coord_column=projection_type,
                          merge_column=batch_column,
                          linewidth=linewidth,
                          markersize=markersize)

    # Plot the UMAP clusters by timepoint
    if num_timepoint_columns > 1:
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-timepoint{suffix}',
                          coord_column=projection_type,
                          palette_column=timepoint_column,
                          linewidth=linewidth,
                          markersize=markersize)
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-timepoint{suffix}',
                          coord_column=projection_type,
                          merge_column=timepoint_column,
                          linewidth=linewidth,
                          markersize=markersize)
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-animal-timepoint{suffix}',
                          coord_column=projection_type,
                          merge_column=animal_column,
                          palette_column=timepoint_column,
                          linewidth=linewidth,
                          markersize=markersize)

    # Plot the UMAP clusters by distance
    if distance_column is not None:
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-distance{suffix}',
                          coord_column=projection_type,
                          palette_column=distance_column,
                          linewidth=linewidth,
                          markersize=markersize)
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-distance{suffix}',
                          coord_column=projection_type,
                          merge_column=distance_column,
                          linewidth=linewidth,
                          markersize=markersize)
        proc.plot_scatter(plotdir / f'{prefix}{projection_type}-merge-animal-distance{suffix}',
                          coord_column=projection_type,
                          merge_column=animal_column,
                          palette_column=distance_column,
                          linewidth=linewidth,
                          markersize=markersize)

    # Cluster using kmeans
    category_columns = [var_column]
    if timepoint_column is not None:
        category_columns.append(timepoint_column)
    if distance_column is not None:
        category_columns.append(distance_column)
    if batch_column is not None:
        category_columns.append(batch_column)

    # Make a silhouette plot
    proc.plot_clustering_scores(coord_column=cluster_coord_column,
                                max_num_clusters=max([num_kmeans_clusters, 10]),
                                plotfile=plotdir / f'{prefix}{cluster_coord_column}_clustering_scores{suffix}')

    # Generate clusters
    cluster_coord_key = f'LabelKMeans{cluster_coord_column.capitalize()}'
    proc.cluster_kmeans(n_clusters=num_kmeans_clusters, coord_column=cluster_coord_column)
    proc.calc_cluster_counts(id_column=animal_column, category_columns=category_columns)
    proc.plot_scatter(plotdir / f'{prefix}{projection_type}-cluster{suffix}',
                      coord_column=projection_type,
                      palette_column=cluster_coord_key,
                      palette_labels=cluster_labels,
                      add_palette_labels=True,
                      linewidth=linewidth,
                      markersize=markersize)
    proc.write_cluster_counts(plotdir / f'{prefix}cluster_counts.xlsx')
    proc.plot_legend(column=cluster_coord_key, plotfile=plotdir / f'{prefix}cluster-legend{suffix}',
                     cluster_labels=cluster_labels)

    # For each cluster, look at how the treatments split
    cluster_plotdir = plotdir / 'cluster_counts'
    proc.plot_all_cluster_counts(
        label_type=cluster_coord_key,
        var_column=var_column,
        var_palette=var_palette,
        animal_column=animal_column,
        timepoint_column=timepoint_column,
        distance_column=distance_column,
        distance_order=distance_order,
        distance_xticklabels=distance_xticklabels,
        distance_palette=distance_palette,
        batch_column=batch_column,
        plotdir=cluster_plotdir,
        prefix=prefix,
        suffix=suffix,
        order=order,
        plot_style=plot_style,
        capsize=plot_err_capsize,
        xticklabels=xticklabels,
        xticklabel_rotation=xticklabel_rotation,
        cluster_labels=cluster_labels,
        comparisons=comparisons,
        force_zero_ylim=force_zero_ylim,
    )

    proc.write_cluster_prism(plotdir / f'{prefix}prism_cluster_counts.pzfx',
                             label_type=cluster_coord_key,
                             var_column=var_column,
                             animal_column=animal_column,
                             order=order)

    # Calculate the single feature accuracy
    cluster_acc_df = proc.plot_cluster_term_accuracy(
        label_type=cluster_coord_key,
        plotfile=plotdir / f'{prefix}cluster_accuracy{suffix}',
        max_k=max_k_accuracy_terms)
    cluster_acc_df.to_excel(plotdir / f'{prefix}cluster_accuracy.xlsx', index=False)

    treatment_acc_df = proc.plot_cluster_term_accuracy(
        label_type=var_column,
        plotfile=plotdir / f'{prefix}{norm_var_column}_accuracy{suffix}',
        max_k=max_k_accuracy_terms)
    treatment_acc_df.to_excel(plotdir / f'{prefix}{norm_var_column}_accuracy.xlsx', index=False)

    # Calculate the top features for both clusters and the variable
    print(f'Plotting {var_column} volcanos')
    proc.plot_all_volcanos(
        var_column,
        plotdir=plotdir / f'{norm_var_column}_volcanos',
        order=order,
        prefix=prefix,
        label_renames=ylabels,
        suffix=suffix,
        **volcano_kwargs)

    print('Plotting cluster volcanos')
    proc.plot_all_volcanos(
        cluster_coord_key,
        plotdir=plotdir / 'cluster_volcanos',
        prefix=prefix,
        label_renames=ylabels,
        cluster_labels=cluster_labels,
        suffix=suffix,
        **volcano_kwargs)

    if distance_column is not None:
        print('Plotting distance volcanos')
        proc.plot_all_volcanos(
            distance_column,
            plotdir=plotdir / 'distance_volcanos',
            order=distance_order,
            prefix=prefix,
            label_renames=ylabels,
            cluster_labels=distance_labels,
            suffix=suffix,
            **volcano_kwargs)

    # Generate the marker tables for top hits
    treatment_df = proc.find_all_markers(
        label_type=var_column,
        approach='1-vs-rest',
        pvalue_threshold=0.05,
        effect_size_threshold=0.2)
    treatment_df.to_excel(plotdir / f'{prefix}1-vs-rest_{norm_var_column}.xlsx', index=False)

    cluster_df = proc.find_all_markers(
        label_type=cluster_coord_key,
        approach='1-vs-rest',
        pvalue_threshold=0.05,
        effect_size_threshold=0.2)
    cluster_df.to_excel(plotdir / f'{prefix}1-vs-rest_cluster.xlsx', index=False)

    # Find all the top feature columns from the clusters and treatments
    if mean_columns is None:
        mean_columns = []
    elif isinstance(mean_columns, str):
        mean_columns = [mean_columns]
    if sum_columns is None:
        sum_columns = []
    elif isinstance(sum_columns, str):
        sum_columns = [sum_columns]
    mean_columns.extend(proc.merge_top_features(
        [treatment_df, cluster_df],
        include_standard_features=True,
        top_k=additional_top_k_features,
    ))
    print(f'Plotting feature columns: {mean_columns}')

    # Scatter plots for the feature plots in norm space
    if not skip_feature_plots:
        scatter_plotdir = outdir / 'norm_feature_scatter'
        if scatter_plotdir.is_dir():
            shutil.rmtree(scatter_plotdir)
        scatter_plotdir.mkdir(parents=True, exist_ok=True)
        print('Plotting normalized feature scatter plots')
        for column in mean_columns:
            ylabel = ylabels.get(column, column)
            outname = reNORM.sub('_', column).lower().strip('_')
            if not ylabel.startswith('Norm'):
                ylabel = f'Normalized {ylabel}'
            proc.plot_feature_scatter(column,
                                      value_type='norm',
                                      plotfile=scatter_plotdir / f'fs-norm-{prefix}{outname}{suffix}',
                                      coord_column=projection_type,
                                      vmin=-2.5,
                                      vmax=2.5,
                                      title=ylabel,
                                      linewidth=linewidth,
                                      markersize=markersize)

    # Scatter plots in raw values
    if not skip_feature_plots:
        scatter_plotdir = outdir / 'raw_feature_scatter'
        if scatter_plotdir.is_dir():
            shutil.rmtree(scatter_plotdir)
        scatter_plotdir.mkdir(parents=True, exist_ok=True)
        print('Plotting raw feature scatter plots')
        for column in mean_columns:
            ylabel = ylabels.get(column, column)
            vmin, vmax = ylims.get(column, (None, None))
            if force_zero_ylim and vmin is None:
                vmin = 0.0
            outname = reNORM.sub('_', column).lower().strip('_')
            proc.plot_feature_scatter(column,
                                      value_type='raw',
                                      plotfile=scatter_plotdir / f'fs-raw-{prefix}{outname}{suffix}',
                                      coord_column=projection_type,
                                      vmin=vmin,
                                      vmax=vmax,
                                      palette='inferno',
                                      title=ylabel,
                                      linewidth=linewidth,
                                      markersize=markersize)

    # Now collapse everything down to animals
    print('Plotting per animal plots')
    proc.write_animal_data(outdir / 'per_animal_plots',
                           mean_columns=mean_columns,
                           sum_columns=sum_columns,
                           var_column=var_column,
                           animal_column=animal_column,
                           ylabels=ylabels,
                           ylims=ylims,
                           force_zero_ylim=force_zero_ylim,
                           order=order,
                           prefix=prefix,
                           suffix=suffix,
                           plot_style=plot_style,
                           capsize=plot_err_capsize,
                           xticklabels=xticklabels,
                           xticklabel_rotation=xticklabel_rotation,
                           comparisons=comparisons,
                           palette=var_palette,
                           overwrite=True)

    # Look at batch effect
    if num_batch_columns > 1:
        print('Plotting per batch plots')
        proc.write_animal_data(outdir / 'per_batch_plots',
                               mean_columns=mean_columns,
                               sum_columns=sum_columns,
                               var_column=batch_column,
                               animal_column=animal_column,
                               ylabels=ylabels,
                               ylims=ylims,
                               force_zero_ylim=force_zero_ylim,
                               plot_style='bars',
                               suffix=suffix,
                               capsize=plot_err_capsize,
                               prefix=f'batch-{prefix}',
                               overwrite=True)

    # Look at the distance data
    if distance_column is not None:
        print('Plotting per distance plots')
        proc.write_animal_data(outdir / 'per_distance_plots',
                               mean_columns=mean_columns,
                               sum_columns=sum_columns,
                               var_column=distance_column,
                               animal_column=animal_column,
                               ylabels=ylabels,
                               ylims=ylims,
                               xlabel='Distance $(\\mu m)$',
                               xticklabels=distance_xticklabels,
                               xticklabel_rotation=0,
                               force_zero_ylim=force_zero_ylim,
                               plot_style='boxes',
                               palette=distance_palette,
                               suffix=suffix,
                               capsize=plot_err_capsize,
                               prefix=f'distance-{prefix}',
                               overwrite=True)
        # Make counts/distance plots
        distance_outdir = outdir / 'per_distance_count_plots'
        if distance_outdir.is_dir():
            shutil.rmtree(distance_outdir)
        distance_outdir.mkdir(parents=True, exist_ok=True)

        count_df = calc_count_df(proc.df,
                                 label_column=distance_column,
                                 id_columns=[animal_column, var_column])
        count_df.to_excel(distance_outdir / 'counts.xlsx', index=False)

        for value_name in ['CountCells', 'PercentCells']:
            ylim = (0, None)
            outname = value_name.lower()

            plot_utils.plot_boxes(
                df=count_df, var_name='LabelID', hue_var_name=var_column,
                hue_order=order,
                value_name=value_name,
                xlabel='Distance $(\\mu m)$',
                ylim=ylim,
                ylabel={'CountCells': 'Total Cells', 'PercentCells': '% Cells'}.get(value_name),
                palette=var_palette,
                min_samples_per_bin=3,
                plot_style='lines', suffix=suffix, showfliers=False,
                plotfile=distance_outdir / f'dist-{prefix}{outname}{suffix}',
            )

        distance_outdir = outdir / 'per_distance_line_plots'
        if distance_outdir.is_dir():
            shutil.rmtree(distance_outdir)
        distance_outdir.mkdir(parents=True, exist_ok=True)
        print('Plotting per distance line plots')

        for column in mean_columns:
            ylabel = ylabels.get(column, column)
            ylim = ylims.get(column)
            if force_zero_ylim and ylim is None:
                ylim = (0, None)

            outname = reNORM.sub('_', column).lower().strip('_')
            proc.plot_feature(x=distance_column, hue=var_column,
                              y=column, ylabel=ylabel, xlabel='Distance $(\\mu m)$',
                              hue_order=order, ylim=ylim, min_samples_per_bin=3,
                              force_zero_ylim=force_zero_ylim, palette=var_palette,
                              plot_style='lines', suffix=suffix, showfliers=False,
                              plotfile=distance_outdir / f'dist-{prefix}{outname}{suffix}')
            proc.plot_feature(x=distance_column, hue=var_column, merge_column=animal_column,
                              y=column, ylabel=ylabel, xlabel='Distance $(\\mu m)$',
                              hue_order=order, ylim=ylim, xticklabels=distance_xticklabels,
                              force_zero_ylim=force_zero_ylim,
                              plot_style='err_bars', capsize=plot_err_capsize, palette=var_palette,
                              suffix=suffix, showfliers=False, min_samples_per_bin=3,
                              plotfile=distance_outdir / f'dist-{prefix}{outname}-animal{suffix}')

    # Look at the timepoint data
    if num_timepoint_columns > 1:
        timepoint_outdir = outdir / 'per_timepoint_plots'
        if timepoint_outdir.is_dir():
            shutil.rmtree(timepoint_outdir)
        timepoint_outdir.mkdir(parents=True, exist_ok=True)
        print('Plotting per timepoint plots')

        for column in mean_columns:
            ylabel = ylabels.get(column, column)
            ylim = ylims.get(column)
            if force_zero_ylim and ylim is None:
                ylim = (0, None)

            outname = reNORM.sub('_', column).lower().strip('_')
            proc.plot_feature(x=timepoint_column, hue=var_column,
                              y=column, ylabel=ylabel, xlabel='Timepoint',
                              hue_order=order, ylim=ylim,
                              force_zero_ylim=force_zero_ylim, palette=var_palette,
                              plot_style='lines', suffix=suffix, showfliers=False,
                              plotfile=timepoint_outdir / f'tp-{prefix}{outname}{suffix}')
            proc.plot_feature(x=timepoint_column, hue=var_column, merge_column=animal_column,
                              y=column, ylabel=ylabel, xlabel='Timepoint',
                              hue_order=order, ylim=ylim,
                              force_zero_ylim=force_zero_ylim,
                              plot_style='err_bars', capsize=plot_err_capsize, palette=var_palette,
                              suffix=suffix, showfliers=False,
                              plotfile=timepoint_outdir / f'tp-{prefix}{outname}-animal{suffix}')
    return proc
