"""
Tests of the test setup itself.
"""
from pathlib import Path

import phasegen as pg
import testing
from testing import TestCase


class SetupTestCase(TestCase):
    """The suite tests the package of the checkout it is run from."""

    def test_phasegen_is_imported_from_the_tested_checkout(self):
        """
        Regression: ``testing/__init__.py`` moved the working directory behind the installed package in ``sys.path``,
        so a run from a worktree or another checkout silently tested the editable install's checkout.
        """
        self.assertEqual(
            Path(pg.__file__).resolve().parent.parent,
            Path(testing.__file__).resolve().parent.parent
        )
