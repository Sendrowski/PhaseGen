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
   ~phasegen.distributions.SFSCDF
   ~phasegen.distributions.MarginalQuantileFunction
   ~phasegen.distributions.JointDensity
   ~phasegen.distributions.JointCDF
   ~phasegen.distributions.JointSFSCDF
   ~phasegen.distributions.ConditionalDensity
   ~phasegen.distributions.ConditionalCDF
   ~phasegen.distributions.ConditionalQuantileFunction

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
