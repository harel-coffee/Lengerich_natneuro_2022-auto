""" Dimensionality reduction tools

Classes:

* :py:class:`CorrespondenceAnalysis`: Transformer similar to PCA, but for correspondence analysis

Functions:

* :py:func:`calc_pearson_residuals`: Calculate the pearson residuals

"""
# Imports
from typing import Optional

# 3rd party imports
import numpy as np

from sklearn.decomposition import TruncatedSVD

# Classes


class CorrespondenceAnalysis(object):
    """ Implement a correspondence analysis transform

    :param int n_components:
        How many components in the truncated svd
    :param str pearson_transform:
        Pre-conditioning transform for the pearson coefficients
    :param str pearson_residuals:
        Which kind of residuals to calculate
    """

    def __init__(self, n_components: int = 2,
                 pearson_transform: str = 'none',
                 pearson_residuals: str = 'corral'):
        self.n_components = n_components
        self.pearson_transform = pearson_transform
        self.pearson_residuals = pearson_residuals

        self._svd = TruncatedSVD(n_components=n_components)

    def fit(self, x, y=None):
        x = calc_pearson_residuals(x, residuals=self.pearson_residuals,
                                   transform=self.pearson_transform)
        self._svd.fit(x, y)
        return self

    def fit_transform(self, x, y=None):
        x = calc_pearson_residuals(x, residuals=self.pearson_residuals,
                                   transform=self.pearson_transform)
        return self._svd.fit_transform(x, y)

# Functions


def calc_pearson_residuals(data: np.array,
                           transform: Optional[str] = 'none',
                           residuals: str = 'standardized') -> np.array:
    """ Calculate the residuals

    :param ndarray data:
        The (n cells) x (m features) matrix to calculate residuals for
    :param str transform:
        If not None, which variance stabilization to apply before transforming
        (one of 'sqrt', 'anscombe', 'freeman-tukey')
    :param str residuals:
        Which kind of residuals to calculate (one of 'raw', 'standardized',
        'corral', 'freeman-tukey')
    :returns:
        An (n cells) x (m features) matrix of residuals
    """
    if transform == 'sqrt':
        data = np.sqrt(data)
    elif transform == 'anscombe':
        data = 2*np.sqrt(data + 3/8)
    elif transform == 'freeman-tukey':
        data = np.sqrt(data) + np.sqrt(data + 1)
    # FIXME: Should I also implement the power deflation method?

    grand_total = np.sum(data)

    data_prob = data / grand_total
    row_prob = np.sum(data_prob, axis=0)[:, np.newaxis]
    col_prob = np.sum(data_prob, axis=1)[np.newaxis, :]

    # Outer product to get the expected total and expected probabilities
    expected_prob = (row_prob @ col_prob).T

    if residuals == 'raw':
        expected_total = expected_prob * grand_total
        return (data - expected_total) / np.sqrt(expected_total)
    elif residuals == 'standardized':
        expected_total = expected_prob * grand_total
        inv_expected_prob = ((1.0 - row_prob) @ (1.0 - col_prob)).T
        return (data - expected_total) / np.sqrt(expected_total*inv_expected_prob)
    elif residuals == 'corral':
        # Definition from Hsu and Culhane, 2021. It's different from the standard definition
        return (data_prob - expected_prob) / np.sqrt(expected_prob)
    elif residuals == 'freeman-tukey':
        # Definition from Hsu and Culhane, 2021
        return np.sqrt(data_prob) + np.sqrt(data_prob + 1/grand_total) - np.sqrt(4*expected_prob + 1/grand_total)
    else:
        raise KeyError(f'Unknown residual mode: "{residuals}"')
