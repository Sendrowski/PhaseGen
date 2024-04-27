"""
Test spectrum module.
"""

from unittest import TestCase

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

    def test_plot_2sfs_fill_diagonal_entries(self):
        """
        Test plotting a monomorphic 2D SFS.
        """
        sfs = pg.SFS2(np.ones((12, 12)))

        sfs.plot(fill_diagonal_entries=True)
        sfs.plot_surface(fill_diagonal_entries=True)

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
