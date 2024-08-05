"""
PhaseGen package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-04-09"

__version__ = '1.0.0'

import logging
import os
import sys

import jsonpickle.ext.numpy as jsonpickle_numpy
from tqdm import tqdm

# lower the verbosity of TensorFlow
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# register handlers
jsonpickle_numpy.register_handlers()


class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that uses TQDM to display log messages.
    """

    def __init__(self, level=logging.NOTSET):
        """
        Initialize the handler.

        :param level:
        """
        super().__init__(level)

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            msg = self.format(record)

            # we write to stderr to avoid as the progress bar
            # to make the two work together
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the formatter.
        """
        super().__init__(*args, **kwargs)

        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[31m",  # Red
        }

        self.reset = "\033[0m"

    def format(self, record):
        """
        Format the record.
        """
        color = self.colors.get(record.levelname, self.reset)

        formatted = super().format(record)

        # remove package name
        formatted = formatted.replace(record.name, record.name.split('.')[-1])

        return f"{color}{formatted}{self.reset}"


# configure logger
logger = logging.getLogger('phasegen')

# don't propagate to the root logger
logger.propagate = False

# set to INFO by default
logger.setLevel(logging.INFO)

# let TQDM handle the logging
handler = TqdmLoggingHandler()

# define a Formatter with colors
formatter = ColoredFormatter('%(levelname)s:%(name)s: %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)

from .distributions import PhaseTypeDistribution

from .distributions import Coalescent

from .demography import (
    Demography,
    Epoch,
    DiscreteRateChanges,
    PopSizeChanges,
    PopSizeChange,
    MigrationRateChanges,
    MigrationRateChange,
    SymmetricMigrationRateChanges,
    PopulationSplit,
    DiscretizedRateChanges,
    DiscretizedRateChange,
    ExponentialPopSizeChanges,
    ExponentialRateChanges
)

from .coalescent_models import (
    CoalescentModel,
    StandardCoalescent,
    BetaCoalescent,
    DiracCoalescent
)

from .state_space import (
    StateSpace,
    LineageCountingStateSpace,
    BlockCountingStateSpace
)

from .rewards import (
    Reward,
    TreeHeightReward,
    TotalTreeHeightReward,
    TotalBranchLengthReward,
    TotalBranchLengthLocusReward,
    UnfoldedSFSReward,
    FoldedSFSReward,
    LineageReward,
    CustomReward,
    ProductReward,
    SumReward,
    CombinedReward,
    DemeReward,
    LocusReward
)

from .spectrum import (
    SFS,
    Spectra,
    SFS2
)

from .inference import Inference

from .lineage import LineageConfig

from .locus import LocusConfig

from .norms import (
    LNorm,
    L1Norm,
    L2Norm,
    LInfNorm,
    PoissonLikelihood
)

from .state_space_old import StateSpace as OldStateSpace

from .expm import (
    ExpmBackend,
    Backend,
    SciPyExpmBackend,
    TensorFlowExpmBackend,
    JaxExpmBackend
)

from .utils import (
    take_n,
    takewhile_inclusive
)

__all__ = [
    'PhaseTypeDistribution',
    'Coalescent',
    'Demography',
    'Epoch',
    'PopSizeChanges',
    'PopSizeChange',
    'MigrationRateChanges',
    'MigrationRateChange',
    'SymmetricMigrationRateChanges',
    'PopulationSplit',
    'ExponentialPopSizeChanges',
    'ExponentialRateChanges',
    'DiscreteRateChanges',
    'DiscretizedRateChange',
    'DiscretizedRateChanges',
    'StandardCoalescent',
    'BetaCoalescent',
    'DiracCoalescent',
    'SFS2',
    'SFS',
    'Spectra',
    'Inference',
    'LNorm',
    'L1Norm',
    'L2Norm',
    'LInfNorm',
    'PoissonLikelihood',
    'Reward',
    'TreeHeightReward',
    'TotalTreeHeightReward',
    'TotalBranchLengthReward',
    'TotalBranchLengthLocusReward',
    'UnfoldedSFSReward',
    'FoldedSFSReward',
    'LineageReward',
    'CustomReward',
    'ProductReward',
    'SumReward',
    'DemeReward',
    'LocusReward',
    'CombinedReward',
    'StateSpace',
    'LineageCountingStateSpace',
    'BlockCountingStateSpace',
    'CoalescentModel',
    'LineageConfig',
    'LocusConfig',
    'Backend',
    'ExpmBackend',
    'SciPyExpmBackend',
    'TensorFlowExpmBackend',
    'JaxExpmBackend'
]
