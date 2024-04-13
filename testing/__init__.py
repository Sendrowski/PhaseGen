"""
Initialization for the testing module.
"""
import logging
import sys
from pathlib import Path
from unittest import TestCase as BaseTestCase

import pytest
from matplotlib import pyplot as plt


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path = [p for p in sys.path if p != cwd]
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


# run before importing phasegen
prioritize_installed_packages()

import phasegen as pg

# register the expm backend
pg.Backend.register(pg.TensorFlowExpmBackend())

logger = logging.getLogger('phasegen')

logger.info(sys.version)
logger.info(f"Running tests for {pg.__file__}")
logger.info(f"phasegen version: {pg.__version__}")
logger.info(f"Exponentiation backend: {pg.expm.Backend.backend.__class__.__name__}")


class TestCase(BaseTestCase):
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        plt.close('all')
