"""
Initialization for the testing module.
"""
import logging
import os
import sys
from pathlib import Path
from unittest import TestCase as BaseTestCase

# Pin the BLAS/numba thread pools to one thread per process. Under ``pytest -n auto`` every xdist worker would
# otherwise open a pool sized to the whole machine, and the workers spend their time contending rather than working
# (the suite takes 21:30 instead of 5:20). This must happen before numpy is imported, because BLAS reads the
# variables when its shared library loads; the root ``conftest`` gets there first, and this repeats it for the case
# where the suite is run with a rootdir that does not pick that conftest up. ``blas_pinned_late`` records the case
# the variables cannot fix -- numpy already resident, so its pool is already sized -- which ``conftest`` reports.
blas_pinned_late = 'numpy' in sys.modules and not os.environ.get('PHASEGEN_ALLOW_THREADS')

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import matplotlib

# force a non-interactive backend so figures don't pop up during normal test runs; set
# PHASEGEN_SHOW_PLOTS=1 to leave the backend untouched and view plots (e.g. in PyCharm's SciView)
if not (os.environ.get('PHASEGEN_SHOW_PLOTS') or os.environ.get('MPLBACKEND')):
    matplotlib.use('Agg')


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

# register the expm backend. The coalescent statistics issue many small expm calls (e.g. one per SFS bin), for
# which SciPy is fastest (TensorFlow's and Jax's per-call overhead makes them slower despite being better for
# large matrices / GPUs).
pg.Backend.register(pg.SciPyExpmBackend())

logger = logging.getLogger('phasegen')

logger.info(sys.version)
logger.info(f"Running tests for {pg.__file__}")
logger.info(f"phasegen version: {pg.__version__}")
logger.info(f"Exponentiation backend: {pg.expm.Backend.backend.__class__.__name__}")

# create scratch directory if it doesn't exist
if not os.path.exists('scratch'):
    os.makedirs('scratch')


class TestCase(BaseTestCase):
    """
    Common base class for all test cases
    """
    pass
