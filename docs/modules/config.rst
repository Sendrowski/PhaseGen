.. _modules.config:

Lineage & locus config
----------------------

Configuration of the sampled lineages and of the loci. A :class:`~phasegen.lineage.LineageConfig` specifies the number of lineages sampled in each population, and a :class:`~phasegen.locus.LocusConfig` specifies the number of loci, the recombination rate between them and the number of lineages that are initially unlinked. An :class:`~phasegen.initial.InitialDistribution` weights several lineage or locus configurations from which the coalescent starts.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~phasegen.lineage.LineageConfig
   ~phasegen.locus.LocusConfig
   ~phasegen.initial.InitialDistribution

.. autoclass:: phasegen.lineage.LineageConfig

.. autoclass:: phasegen.locus.LocusConfig

.. autoclass:: phasegen.initial.InitialDistribution
