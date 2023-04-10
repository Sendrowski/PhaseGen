"""
PH package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-04-09"

__version__ = 'alpha'

from .distributions import ConstantPopSizeDistribution, VariablePopSizeDistribution

from .distributions import ConstantPopSizeCoalescent, VariablePopSizeCoalescent

from .demography import Demography, PiecewiseConstantDemography

from .coalescent_models import CoalescentModel, StandardCoalescent, BetaCoalescent

from .distributions import set_precision

from .comparison import Comparison

__all__ = [
    'ConstantPopSizeDistribution',
    'VariablePopSizeDistribution',
    'ConstantPopSizeCoalescent',
    'VariablePopSizeCoalescent',
    'PiecewiseConstantDemography',
    'StandardCoalescent',
    'BetaCoalescent',
    'set_precision',
    'Comparison'
]
