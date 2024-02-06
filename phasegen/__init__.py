"""
PhaseGen package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-04-09"

__version__ = 'alpha'

import logging
import sys

import jsonpickle.ext.numpy as jsonpickle_numpy
from tqdm import tqdm

# register handlers
jsonpickle_numpy.register_handlers()


class TqdmLoggingHandler(logging.Handler):
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
        log_color = self.colors.get(record.levelname, self.reset)

        formatted_record = super().format(record)

        return f"{log_color}{formatted_record}{self.reset}"


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
    DiscretizedRateChanges,
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
    DefaultStateSpace,
    BlockCountingStateSpace
)

from .rewards import (
    Reward,
    DefaultReward,
    NonDefaultReward,
    TreeHeightReward,
    TotalTreeHeightReward,
    TotalBranchLengthReward,
    SFSReward,
    CustomReward,
    ProductReward,
    CombinedReward,
    DemeReward
)

from .spectrum import (
    SFS,
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

__all__ = [
    'PhaseTypeDistribution',
    'Coalescent',
    'Demography',
    'Epoch',
    'PopSizeChanges',
    'PopSizeChange',
    'MigrationRateChanges',
    'MigrationRateChange',
    'ExponentialPopSizeChanges',
    'ExponentialRateChanges',
    'DiscreteRateChanges',
    'DiscretizedRateChanges',
    'StandardCoalescent',
    'BetaCoalescent',
    'DiracCoalescent',
    'SFS2',
    'SFS',
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
    'SFSReward',
    'CustomReward',
    'ProductReward',
    'DemeReward',
    'DefaultReward',
    'NonDefaultReward',
    'DefaultStateSpace',
    'BlockCountingStateSpace',
    'CoalescentModel',
    'LineageConfig',
    'LocusConfig',
]
