.. _modules.distributions:

Distributions
-------------

Phase-type distributions. The :class:`~phasegen.distributions.Coalescent` class serves as an entry point for accessing all other distributions.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~phasegen.distributions.Coalescent
   ~phasegen.distributions.PhaseTypeDistribution
   ~phasegen.distributions.TreeHeightDistribution
   ~phasegen.distributions.TotalBranchLengthDistribution
   ~phasegen.distributions.FoldedSFSDistribution
   ~phasegen.distributions.UnfoldedSFSDistribution
   ~phasegen.distributions.JointSFSDistribution
   ~phasegen.distributions.TwoLocusSFSDistribution
   ~phasegen.distributions.MarginalLocusDistributions
   ~phasegen.distributions.MarginalDemeDistributions
   ~phasegen.distributions.RewardDistribution
   ~phasegen.distributions.ConditionalRewardDistribution
   ~phasegen.distributions.JointRewardDistribution
   ~phasegen.distributions.DistributionFunction
   ~phasegen.distributions.DensityFunction
   ~phasegen.distributions.CumulativeDistributionFunction
   ~phasegen.distributions.QuantileFunction
   ~phasegen.distributions.SFSDensity
   ~phasegen.distributions.SFSCDF
   ~phasegen.distributions.SFSQuantileFunction
   ~phasegen.distributions.JointDensity
   ~phasegen.distributions.JointCDF
   ~phasegen.distributions.JointSFSDensity
   ~phasegen.distributions.JointSFSCDF
   ~phasegen.distributions.JointSFSQuantileFunction
   ~phasegen.distributions.ConditionalDensity
   ~phasegen.distributions.ConditionalCDF
   ~phasegen.distributions.ConditionalQuantileFunction
   ~phasegen.distributions.MsprimeCoalescent
   ~phasegen.distributions.SampledCoalescent
   ~phasegen.distributions.EmpiricalDistribution
   ~phasegen.distributions.EmpiricalPhaseTypeDistribution
   ~phasegen.distributions.EmpiricalPhaseTypeSFSDistribution
   ~phasegen.distributions.EmpiricalJointDistribution
   ~phasegen.distributions.EmpiricalSFSDistribution
   ~phasegen.distributions.EmpiricalJointSFSDistribution
   ~phasegen.distributions.EmpiricalTwoLocusSFSDistribution

.. autoclass:: phasegen.distributions.Coalescent

.. autoclass:: phasegen.distributions.PhaseTypeDistribution

.. autoclass:: phasegen.distributions.TreeHeightDistribution

.. autoclass:: phasegen.distributions.TotalBranchLengthDistribution

.. autoclass:: phasegen.distributions.FoldedSFSDistribution

.. autoclass:: phasegen.distributions.UnfoldedSFSDistribution

.. autoclass:: phasegen.distributions.JointSFSDistribution

.. autoclass:: phasegen.distributions.TwoLocusSFSDistribution
   :exclude-members: cdf, pdf, quantile, plot_cdf, bin, loci, demes

.. autoclass:: phasegen.distributions.MarginalLocusDistributions

.. autoclass:: phasegen.distributions.MarginalDemeDistributions

.. autoclass:: phasegen.distributions.RewardDistribution

.. autoclass:: phasegen.distributions.ConditionalRewardDistribution
   :exclude-members: __init__, plot_cdf

.. autoclass:: phasegen.distributions.JointRewardDistribution
   :exclude-members: quantile, plot_cdf

.. autoclass:: phasegen.distributions.DistributionFunction
   :members:

.. autoclass:: phasegen.distributions.DensityFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.CumulativeDistributionFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.QuantileFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.SFSDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.SFSCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.SFSQuantileFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.JointDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.JointCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.JointSFSDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.JointSFSCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.JointSFSQuantileFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.ConditionalDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.ConditionalCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.ConditionalQuantileFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.MsprimeCoalescent

.. autoclass:: phasegen.distributions.SampledCoalescent

.. autoclass:: phasegen.distributions.EmpiricalDistribution

.. autoclass:: phasegen.distributions.EmpiricalPhaseTypeDistribution

.. autoclass:: phasegen.distributions.EmpiricalPhaseTypeSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalJointDistribution

.. autoclass:: phasegen.distributions.EmpiricalSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalJointSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalTwoLocusSFSDistribution
