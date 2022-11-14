# Imports
import unittest

# 3rd party
import numpy as np

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt

# Our own imports
from atv_trem2_morpho import qc_utils

# Helpers


class BaseBatchEffect(unittest.TestCase):

    def plot_diff(self, res, exp, cmap: str = 'viridis'):
        vmin = min([np.min(res), np.min(exp)])
        vmax = max([np.max(res), np.max(exp)])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(res, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.set_title('Result')
        ax2.imshow(exp, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.set_title('Expected')
        ax3.imshow(res - exp, cmap=cmap)
        ax3.set_title('Difference')
        plt.show()

# Tests


class TestFindMutualNeighbors(unittest.TestCase):

    def test_finds_perfect_matches_pairs(self):

        coords = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
        ])
        assert coords.shape[0] == 6

        tree = BallTree(coords)
        inds = tree.query(coords, k=1, return_distance=False)

        assert inds.shape[0] == 6
        assert inds.shape[1] == 1

        matches = qc_utils.find_mutual_neighbors(inds, inds)

        exp_matches = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ], dtype=[('right', 'int64'), ('left', 'int64')])

        np.testing.assert_allclose(matches['right'], exp_matches['right'])
        np.testing.assert_allclose(matches['left'], exp_matches['left'])

    def test_finds_perfect_matches_lattice(self):

        coords = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ])
        assert coords.shape[0] == 6

        tree = BallTree(coords)
        inds = tree.query(coords, k=3, return_distance=False)

        assert inds.shape[0] == 6
        assert inds.shape[1] == 3

        matches = qc_utils.find_mutual_neighbors(inds, inds)

        exp_matches = np.array([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 2),
            (2, 3),
            (2, 1),
            (3, 3),
            (3, 4),
            (3, 2),
            (4, 4),
            (4, 5),
            (4, 3),
            (5, 5),
            (5, 4),
        ], dtype=[('right', 'int64'), ('left', 'int64')])
        np.testing.assert_allclose(matches['right'], exp_matches['right'])
        np.testing.assert_allclose(matches['left'], exp_matches['left'])


class TestRemoveBatchEffectMeans(BaseBatchEffect):

    def test_matches_matrix_two_blocks_identical(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))

        values = np.concatenate([block1, block1], axis=0)
        labels = np.zeros((128, ))
        labels[64:] = 1

        assert values.shape == (128, 20)
        assert labels.shape == (128, )

        res = qc_utils.remove_batch_effect_means(labels, values)

        np.testing.assert_allclose(res, values, atol=1e-3)

    def test_matches_matrix_two_blocks_offset(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))

        values = np.concatenate([block1, block1 + 2.0, block1 + 2.0], axis=0)
        labels = np.zeros((192, ))
        labels[64:] = 1

        assert values.shape == (192, 20)
        assert labels.shape == (192, )

        res = qc_utils.remove_batch_effect_means(labels, values)

        # A constant offset should become evenly spread over the data
        offset = 4/3
        exp = np.concatenate([block1 + offset, block1 + offset, block1 + offset], axis=0)

        np.testing.assert_allclose(res, exp, atol=1e-3)

    def test_matches_matrix_two_blocks_scale(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))

        values = np.concatenate([block1, block1*2.0, block1*2.0], axis=0)
        labels = np.zeros((192, ))
        labels[64:] = 1

        assert values.shape == (192, 20)
        assert labels.shape == (192, )

        res = qc_utils.remove_batch_effect_means(labels, values)

        # A constant scale should also get evenly spread over the data
        scale = 5/3
        offset = 0.0
        exp = np.concatenate([
            block1*scale + offset,
            block1*scale + offset,
            block1*scale + offset], axis=0)
        np.testing.assert_allclose(res, exp, atol=1e-3)

    def test_matches_matrix_two_blocks_scale_median(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))

        values = np.concatenate([block1, block1*2.0, block1*2.0], axis=0)
        labels = np.zeros((192, ))
        labels[64:] = 1

        assert values.shape == (192, 20)
        assert labels.shape == (192, )

        res = qc_utils.remove_batch_effect_means(labels, values, mode='median')

        # A constant scale should also get evenly spread over the data
        scale = 5/3
        offset = 0.0
        exp = np.concatenate([
            block1*scale + offset,
            block1*scale + offset,
            block1*scale + offset], axis=0)
        np.testing.assert_allclose(res, exp, atol=1e-3)


class TestRemoveBatchEffectMNN(BaseBatchEffect):

    def test_matches_matrix_two_blocks_identical(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))

        values = np.concatenate([block1, block1], axis=0)
        labels = np.zeros((128, ))
        labels[64:] = 1

        assert values.shape == (128, 20)
        assert labels.shape == (128, )

        res = qc_utils.remove_batch_effect_mnn(labels, values, num_neighbors=1)

        np.testing.assert_allclose(res, values)

    # @unittest.skip('Still not working...')
    def test_matches_matrix_two_blocks_offset(self):

        gen = np.random.default_rng(12345)

        block1 = gen.integers(0, 255, size=(64, 20))
        block2 = block1.copy()
        block2[:, [1, 3, 5, 7, 9]] += 50

        values = np.concatenate([block1, block2, block2], axis=0)
        labels = np.zeros((192, ))
        labels[64:] = 1

        assert values.shape == (192, 20)
        assert labels.shape == (192, )

        res = qc_utils.remove_batch_effect_mnn(labels, values, num_neighbors=5,
                                               sigma=0.01)

        exp = np.concatenate([block2, block2, block2], axis=0)

        # self.plot_diff(res, exp)

        np.testing.assert_allclose(res, exp, atol=10)
