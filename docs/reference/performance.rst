.. _reference.performance:

Runtime performance
===================

State space
-----------
The size of the state space can grow rapidly with the complexity of the demographic scenario, i.e. the number of lineages, demes and loci as shown below.

.. image:: ../images/state_space_sizes.png
   :alt: State space sizes
   :width: 100%
   :align: center

Constructing it (enumerating the states and assembling the rate matrix) is accelerated with |numba|_.

Exact computation
-----------------
The runtime of exact moments is governed by the size of the state space, which enters through linear solves with the transient block of the last epoch, matrix exponentials over finite epochs, and the order of the moment. The evaluation strategies and the settings that select them are described in :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`. Below we can see the total runtime in seconds for computing the mean tree height, the mean SFS, and the mean two-locus SFS under a 1-epoch standard coalescent over a range of different numbers of lineages and loci.

.. image:: ../images/execution_times.png
   :alt: Execution times
   :width: 100%
   :align: center

Empirical estimation
--------------------
Where the exact computation becomes too costly, the statistics can instead be estimated from ``phasegen``'s own vectorised sampler (:meth:`~phasegen.distributions.Coalescent.to_empirical`). Its cost grows with the number of samples and the number of jumps per trajectory, while the state space is still constructed as for the exact computation (see :meth:`PhaseTypeDistribution.sample() <phasegen.distributions.PhaseTypeDistribution.sample>`). The figure below shows the runtime for drawing 100,000 samples, on the same colour scale as above. Sampling is considerably faster even for the largest case, which is the slowest to compute exactly in the figure above.

.. image:: ../images/sampling_times.png
   :alt: Sampling times
   :width: 100%
   :align: center

.. |numba| replace:: ``numba``
.. _numba: https://numba.pydata.org/
