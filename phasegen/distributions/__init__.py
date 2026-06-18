"""
Probability distributions.

This package is split into submodules (:mod:`base`, :mod:`phase_type`, :mod:`spectra`, :mod:`coalescent`,
:mod:`empirical`); all public classes are re-exported here so ``from phasegen.distributions import X`` keeps working.
"""
from .base import (
    ProbabilityDistribution,
    MomentAwareDistribution,
    MarginalDistributions,
    MarginalLocusDistributions,
    MarginalDemeDistributions,
    DensityAwareDistribution,
    DistributionFunction,
    DensityFunction,
    CumulativeDistributionFunction,
    QuantileFunction,
    MarginalDensity,
    MarginalCDF,
    MarginalQuantileFunction,
    JointDensity,
    JointCDF,
    ConditionalDensity,
    ConditionalCDF,
    ConditionalQuantileFunction,
)
from .phase_type import (
    PhaseTypeDistribution,
    TreeHeightDistribution,
    TotalBranchLengthDistribution,
)
from .reward import (
    RewardDistribution,
    JointRewardDistribution,
)
from .spectra import (
    SFSDistribution,
    TajimaSFSMixin,
    UnfoldedSFSDistribution,
    FoldedSFSDistribution,
    JointSFSDistribution,
    TwoLocusSFSDistribution,
)
from .coalescent import (
    AbstractCoalescent,
    Coalescent,
)
from .empirical import (
    EmpiricalJointSFSDistribution,
    EmpiricalDistribution,
    EmpiricalSFSDistribution,
    DictContainer,
    EmpiricalPhaseTypeDistribution,
    EmpiricalPhaseTypeSFSDistribution,
    EmpiricalTwoLocusSFSDistribution,
    MsprimeCoalescent,
)

__all__ = [
    "ProbabilityDistribution",
    "MomentAwareDistribution",
    "MarginalDistributions",
    "MarginalLocusDistributions",
    "MarginalDemeDistributions",
    "DensityAwareDistribution",
    "DistributionFunction",
    "DensityFunction",
    "CumulativeDistributionFunction",
    "QuantileFunction",
    "MarginalDensity",
    "MarginalCDF",
    "MarginalQuantileFunction",
    "JointDensity",
    "JointCDF",
    "ConditionalDensity",
    "ConditionalCDF",
    "ConditionalQuantileFunction",
    "PhaseTypeDistribution",
    "TreeHeightDistribution",
    "TotalBranchLengthDistribution",
    "RewardDistribution",
    "JointRewardDistribution",
    "SFSDistribution",
    "TajimaSFSMixin",
    "UnfoldedSFSDistribution",
    "FoldedSFSDistribution",
    "JointSFSDistribution",
    "TwoLocusSFSDistribution",
    "AbstractCoalescent",
    "Coalescent",
    "EmpiricalJointSFSDistribution",
    "EmpiricalDistribution",
    "EmpiricalSFSDistribution",
    "DictContainer",
    "EmpiricalPhaseTypeDistribution",
    "EmpiricalPhaseTypeSFSDistribution",
    "EmpiricalTwoLocusSFSDistribution",
    "MsprimeCoalescent",
]
