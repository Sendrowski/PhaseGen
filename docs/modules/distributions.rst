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

The ``pdf`` / ``cdf`` / ``quantile`` properties return these callable-and-plottable distribution-function objects (call to evaluate, ``.plot()`` to draw, and -- for a bivariate joint -- ``.plot_surface()``), in plain, marginal (per-bin spectrum), joint (bivariate) and conditional flavours. Their public methods are documented as members below (the inversion machinery itself lives on these objects for the analytic / Laplace-transform distributions).

.. autoclass:: phasegen.distributions.DistributionFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.DensityFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.CumulativeDistributionFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.QuantileFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalDensity
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalCDF
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.MarginalQuantileFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointDensity
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.JointCDF
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalDensity
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalCDF
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:

.. autoclass:: phasegen.distributions.ConditionalQuantileFunction
   :members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:
