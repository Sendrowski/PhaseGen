"""
Test spectrum module.
"""

from testing import TestCase

import numpy as np

import phasegen as pg


class SpectrumTestCase(TestCase):
    """
    Test spectrum module.
    """

    def test_plot_2sfs_two_elements(self):
        """
        Test plotting a 2D SFS with two elements.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        with self.assertLogs(level='WARNING', logger=pg.spectrum.logger) as cm:
            sfs.plot()
            sfs.plot_surface()

    def test_plot_2sfs_three_elements(self):
        """
        Test plotting a 2D SFS with three elements.
        """
        sfs = pg.SFS2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        sfs.plot()
        sfs.plot_surface()

    def test_plot_2sfs_log_scale(self):
        """
        Test plotting a 2D SFS with a log scale.
        """
        sfs = pg.SFS2(np.random.randint(1, 100, (12, 12)).astype(float))

        sfs.plot(log_scale=True)

    def test_2sfs_passing_not_2d_array_raises_value_error(self):
        """
        Test that passing a non-2D array raises a ValueError.
        """
        with self.assertRaises(ValueError):
            pg.SFS2([1, 2, 3])

        with self.assertRaises(ValueError):
            pg.SFS2([[1, 2], [3, 4], [5, 6]])

    def test_2sfs_passing_non_square_array_raises_value_error(self):
        """
        Test that passing a non-square array raises a ValueError.
        """
        with self.assertRaises(ValueError):
            pg.SFS2([[1, 2, 1], [3, 4, 5]])

    def test_2sfs_recover_from_file(self):
        """
        Test recovering a 2D SFS from a file.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])
        sfs.to_file('scratch/test_sfs.json')

        sfs2 = pg.SFS2.from_file('scratch/test_sfs.json')

        np.testing.assert_array_equal(sfs.data, sfs2.data)

    def test_2sfs_add(self):
        """
        Test adding two 2D SFSs.
        """
        sfs1 = pg.SFS2([[1, 2], [3, 4]])
        sfs2 = pg.SFS2([[5, 6], [7, 8]])

        sfs3 = sfs1 + sfs2

        np.testing.assert_array_equal(sfs3.data, np.array([[6, 8], [10, 12]]))

    def test_2sfs_subtract(self):
        """
        Test subtracting two 2D SFSs.
        """
        sfs1 = pg.SFS2([[1, 2], [3, 4]])
        sfs2 = pg.SFS2([[5, 6], [7, 8]])

        sfs3 = sfs1 - sfs2

        np.testing.assert_array_equal(sfs3.data, np.array([[-4, -4], [-4, -4]]))

    def test_2sfs_multiply(self):
        """
        Test multiplying two 2D SFSs.
        """
        sfs1 = pg.SFS2([[1, 2], [3, 4]])
        sfs2 = pg.SFS2([[5, 6], [7, 8]])

        sfs3 = sfs1 * sfs2

        np.testing.assert_array_equal(sfs3.data, np.array([[5, 12], [21, 32]]))

    def test_2sfs_floor_divide(self):
        """
        Test floor dividing two 2D SFSs.
        """
        sfs1 = pg.SFS2([[1, 2], [3, 4]])
        sfs2 = pg.SFS2([[5, 6], [7, 8]])

        sfs3 = sfs1 // sfs2

        np.testing.assert_array_equal(sfs3.data, np.array([[0, 0], [0, 0]]))

    def test_2sfs_true_divide(self):
        """
        Test true dividing two 2D SFSs.
        """
        sfs1 = pg.SFS2([[1, 2], [3, 4]])
        sfs2 = pg.SFS2([[5, 6], [7, 8]])

        sfs3 = sfs1 / sfs2

        np.testing.assert_array_equal(sfs3.data, np.array([[1 / 5, 2 / 6], [3 / 7, 4 / 8]]))

    def test_2sfs_add_scalar(self):
        """
        Test adding a scalar to a 2D SFS.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs + 1

        np.testing.assert_array_equal(sfs2.data, np.array([[2, 3], [4, 5]]))

    def test_2sfs_subtract_scalar(self):
        """
        Test subtracting a scalar from a 2D SFS.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs - 1

        np.testing.assert_array_equal(sfs2.data, np.array([[0, 1], [2, 3]]))

    def test_2sfs_multiply_scalar(self):
        """
        Test multiplying a 2D SFS by a scalar.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs * 2

        np.testing.assert_array_equal(sfs2.data, np.array([[2, 4], [6, 8]]))

    def test_2sfs_floor_divide_scalar(self):
        """
        Test floor dividing a 2D SFS by a scalar.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs // 2

        np.testing.assert_array_equal(sfs2.data, np.array([[0, 1], [1, 2]]))

    def test_2sfs_true_divide_scalar(self):
        """
        Test true dividing a 2D SFS by a scalar.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs / 2

        np.testing.assert_array_equal(sfs2.data, np.array([[0.5, 1], [1.5, 2]]))

    def test_2sfs_power_scalar(self):
        """
        Test raising a 2D SFS to a power.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs ** 2

        np.testing.assert_array_equal(sfs2.data, np.array([[1, 4], [9, 16]]))

    def test_2sfs_copy(self):
        """
        Test copying a 2D SFS.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs.copy()

        np.testing.assert_array_equal(sfs.data, sfs2.data)

    def test_2sfs_symmetrize(self):
        """
        Test symmetrizing a 2D SFS.
        """
        sfs = pg.SFS2([[1, 2], [3, 4]])

        sfs2 = sfs.symmetrize()

        np.testing.assert_array_equal(sfs2.data, np.array([[1, 2.5], [2.5, 4]]))

    def test_2sfs_fill_monomorphic(self):
        """
        Test filling a monomorphic 2D SFS.
        """
        sfs = pg.SFS2(np.ones((8, 8))).fill_monomorphic(0)

        np.testing.assert_array_equal(sfs.data[0], np.zeros(8))
        np.testing.assert_array_equal(sfs.data[-1], np.zeros(8))
        np.testing.assert_array_equal(sfs.data[:, 0], np.zeros(8))
        np.testing.assert_array_equal(sfs.data[:, -1], np.zeros(8))

        np.testing.assert_array_equal(sfs.data[1:-1, 1:-1], np.ones((6, 6)))

    def test_mask_upper(self):
        """
        Test masking the upper triangle of a 2D SFS.
        """
        sfs = pg.SFS2(np.random.randint(1, 100, (12, 12)).astype(float))

        sfs = sfs.mask_upper()

        self.assertTrue(np.all(np.isnan(sfs.data[np.tril_indices(12, k=-1)])))

    def test_mask_diagonal(self):
        """
        Test masking both diagonals of a 2D SFS and ensure the rest of the matrix is unchanged.
        """
        original_data = np.random.randint(1, 100, (12, 12)).astype(float)
        sfs = pg.SFS2(original_data)

        masked_sfs = sfs.mask_diagonal()

        # Assert primary diagonal is masked
        self.assertTrue(np.all(np.isnan(masked_sfs.data[np.diag_indices(12)])))

        # Assert secondary diagonal is masked
        flipped_indices = (np.arange(12), np.arange(12)[::-1])
        self.assertTrue(np.all(np.isnan(masked_sfs.data[flipped_indices])))

        # Assert rest of the matrix is unchanged
        diag_mask = np.zeros_like(original_data, dtype=bool)
        np.fill_diagonal(diag_mask, True)  # Mask primary diagonal
        diag_mask = np.logical_or(diag_mask, np.fliplr(diag_mask))  # Add secondary diagonal
        self.assertTrue(np.array_equal(original_data[~diag_mask], masked_sfs.data[~diag_mask]))


    def test_joint_sfs_container(self):
        """
        Exercise the JointSFS container: shape/array/iteration/indexing, plotting and serialization.
        """
        data = np.arange(9).reshape(3, 3).astype(float)
        j = pg.JointSFS(data)

        self.assertEqual(j.n_pops, 2)
        self.assertEqual(tuple(j.shape), (3, 3))
        self.assertEqual(np.asarray(j).shape, (3, 3))
        self.assertEqual(len(list(iter(j))), 3)
        self.assertTrue(np.array_equal(np.asarray(j[0]), data[0]))
        self.assertTrue(np.array_equal(np.asarray(j.copy()), data))

        # plotting (Agg backend, non-interactive)
        j.plot(show=False)
        j.plot_surface(show=False)

        # serialization round-trips
        j.to_file('scratch/test_jsfs.json')
        self.assertTrue(np.allclose(np.asarray(pg.JointSFS.from_file('scratch/test_jsfs.json')), data))
        self.assertTrue(np.allclose(np.asarray(pg.JointSFS.from_json(j.to_json())), data))

    def test_joint_sfs_marginalize(self):
        """
        Marginalize a three-population joint SFS down to two populations and plot it.
        """
        data = np.arange(27).reshape(3, 3, 3).astype(float)
        j = pg.JointSFS(data)

        self.assertEqual(j.n_pops, 3)

        marg = j.marginalize(pops=[0, 1])
        self.assertEqual(marg.n_pops, 2)
        self.assertTrue(np.allclose(np.asarray(marg), data.sum(axis=2)))

        marg.plot(pops=(0, 1), show=False)

        with self.assertRaises(ValueError):
            j.marginalize(pops=[0, 5])

    def test_two_locus_sfs_fold_and_surface(self):
        """
        Fold a TwoLocusSFS and plot it as a surface.
        """
        t = pg.TwoLocusSFS(np.arange(25).reshape(5, 5).astype(float))

        folded = t.fold()
        self.assertEqual(tuple(folded.data.shape), (5, 5))

        t.plot_surface(show=False)
