"""
Test spectrum module.
"""

from unittest import TestCase

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
