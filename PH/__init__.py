"""
PH package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-04-09"

__version__ = 'alpha'

from .PH import CoalescentDistribution, VariablePopulationSizeCoalescentDistribution

from .PH import Demography, PiecewiseConstantDemography

from .PH import CoalescentModel, StandardCoalescent, BetaCoalescent

from .PH import set_precision

from .simulator import Simulator

__all__ = [
    'CoalescentDistribution',
    'VariablePopulationSizeCoalescentDistribution',
    'Demography',
    'PiecewiseConstantDemography',
    'CoalescentModel',
    'StandardCoalescent',
    'BetaCoalescent',
    'set_precision'
    'Simulator',
]
