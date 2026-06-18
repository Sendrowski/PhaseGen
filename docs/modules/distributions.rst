.. _modules.distributions:

Distributions
-------------

Phase-type distributions. The :class:`~phasegen.distributions.Coalescent` class serves as an entry point for accessing all other distributions.

.. autoclass:: phasegen.distributions.Coalescent
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.PhaseTypeDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.TreeHeightDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.TotalBranchLengthDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.FoldedSFSDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.UnfoldedSFSDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointSFSDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.TwoLocusSFSDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalLocusDistributions
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalDemeDistributions
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.RewardDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointRewardDistribution
   :inherited-members:
   :undoc-members:
   :show-inheritance:

The ``pdf`` / ``cdf`` / ``quantile`` properties return these callable-and-plottable distribution-function objects (call to evaluate, ``.plot()`` to draw), in plain, marginal (per-bin spectrum), joint (bivariate) and conditional flavours.

.. autoclass:: phasegen.distributions.DistributionFunction
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.DensityFunction
   :show-inheritance:

.. autoclass:: phasegen.distributions.CumulativeDistributionFunction
   :show-inheritance:

.. autoclass:: phasegen.distributions.QuantileFunction
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalDensity
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalCDF
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalQuantileFunction
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointDensity
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointCDF
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalDensity
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalCDF
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalQuantileFunction
   :show-inheritance:
