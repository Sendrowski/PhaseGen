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
   ~phasegen.distributions.JointRewardDistribution
   ~phasegen.distributions.DistributionFunction
   ~phasegen.distributions.DensityFunction
   ~phasegen.distributions.CumulativeDistributionFunction
   ~phasegen.distributions.QuantileFunction
   ~phasegen.distributions.MarginalDensity
   ~phasegen.distributions.MarginalCDF
   ~phasegen.distributions.SFSDensity
   ~phasegen.distributions.SFSCDF
   ~phasegen.distributions.MarginalQuantileFunction
   ~phasegen.distributions.JointDensity
   ~phasegen.distributions.JointCDF
   ~phasegen.distributions.JointSFSDensity
   ~phasegen.distributions.JointSFSCDF
   ~phasegen.distributions.ConditionalDensity
   ~phasegen.distributions.ConditionalCDF
   ~phasegen.distributions.ConditionalQuantileFunction
   ~phasegen.distributions.MsprimeCoalescent
   ~phasegen.distributions.SampledCoalescent
   ~phasegen.distributions.EmpiricalDistribution
   ~phasegen.distributions.EmpiricalPhaseTypeDistribution
   ~phasegen.distributions.EmpiricalPhaseTypeSFSDistribution
   ~phasegen.distributions.EmpiricalJointRewardDistribution
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

.. autoclass:: phasegen.distributions.MarginalLocusDistributions

.. autoclass:: phasegen.distributions.MarginalDemeDistributions

.. autoclass:: phasegen.distributions.RewardDistribution

.. autoclass:: phasegen.distributions.JointRewardDistribution

The ``pdf`` / ``cdf`` / ``quantile`` properties return these callable-and-plottable distribution-function objects (call to evaluate, ``.plot()`` to draw, and -- for a bivariate joint -- ``.plot_surface()``), in plain, marginal (per-bin spectrum), joint (bivariate) and conditional flavours. Their public methods are documented as members below (the inversion machinery itself lives on these objects for the analytic / Laplace-transform distributions).

.. autoclass:: phasegen.distributions.DistributionFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.DensityFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.CumulativeDistributionFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.QuantileFunction
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.MarginalDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.MarginalCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.SFSDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.SFSCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.MarginalQuantileFunction
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

.. autoclass:: phasegen.distributions.ConditionalDensity
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.ConditionalCDF
   :members:
   :special-members: __call__

.. autoclass:: phasegen.distributions.ConditionalQuantileFunction
   :members:
   :special-members: __call__

The same statistics can be estimated empirically from simulated genealogies -- via msprime (:class:`~phasegen.distributions.MsprimeCoalescent`) or PhaseGen's own trajectory sampler (:class:`~phasegen.distributions.SampledCoalescent`) -- with the containers below computing the statistics from the sampled realisations.

.. autoclass:: phasegen.distributions.MsprimeCoalescent

.. autoclass:: phasegen.distributions.SampledCoalescent

.. autoclass:: phasegen.distributions.EmpiricalDistribution

.. autoclass:: phasegen.distributions.EmpiricalPhaseTypeDistribution

.. autoclass:: phasegen.distributions.EmpiricalPhaseTypeSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalJointRewardDistribution

.. autoclass:: phasegen.distributions.EmpiricalSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalJointSFSDistribution

.. autoclass:: phasegen.distributions.EmpiricalTwoLocusSFSDistribution
