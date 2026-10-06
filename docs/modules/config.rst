.. _modules.config:

Configurations
--------------

Configuration of the sampled lineages, of the loci and of the mutational configurations. A :class:`~phasegen.lineage.LineageConfig` specifies the number of lineages sampled in each population, and a :class:`~phasegen.locus.LocusConfig` specifies the number of loci, the recombination rate between them and the number of lineages that are initially unlinked. An :class:`~phasegen.initial.InitialDistribution` weights several lineage or locus configurations from which the coalescent starts.

A :class:`~phasegen.distributions.MutationLayout` defines the bins of a spectrum in which mutations are counted, each bin merging one or more frequency classes, for example the classes of the unfolded spectrum or their folded pairs. A :class:`~phasegen.distributions.MutationConfig` is one outcome on such a layout: the number of mutations observed in each of its bins. The layout is shared by all configurations whose probabilities are computed, while each configuration holds a single vector of counts and refers back to its layout.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~phasegen.lineage.LineageConfig
   ~phasegen.locus.LocusConfig
   ~phasegen.initial.InitialDistribution
   ~phasegen.distributions.MutationLayout
   ~phasegen.distributions.MutationConfig

.. autoclass:: phasegen.lineage.LineageConfig

.. autoclass:: phasegen.locus.LocusConfig

.. autoclass:: phasegen.initial.InitialDistribution

.. autoclass:: phasegen.distributions.MutationLayout

.. autoclass:: phasegen.distributions.MutationConfig
   :exclude-members: count, index
