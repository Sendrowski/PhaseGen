"""
PhaseGen package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-04-09"

__version__ = 'alpha'

import jsonpickle
import numpy as np

from .serialization import NumpyArrayHandler

# register custom json handlers
jsonpickle.handlers.registry.register(np.ndarray, NumpyArrayHandler)

from .distributions import ConstantPopSizeDistribution, VariablePopSizeDistribution

from .distributions import ConstantPopSizeCoalescent, VariablePopSizeCoalescent, MsprimeCoalescent

from .demography import Demography, PiecewiseConstantDemography

from .coalescent_models import CoalescentModel, StandardCoalescent, BetaCoalescent

from .comparison import Comparison

__all__ = [
    'ConstantPopSizeDistribution',
    'VariablePopSizeDistribution',
    'ConstantPopSizeCoalescent',
    'VariablePopSizeCoalescent',
    'MsprimeCoalescent',
    'PiecewiseConstantDemography',
    'StandardCoalescent',
    'BetaCoalescent',
    'Comparison'
]
