.. _reference.performance.runtime:

Runtime
=======

Exact
-----
To obtain moments we need to exponentiate matrices whose size equals the state space size times ``k+1`` where ``k`` is the order of the moment. Matrix exponentiation in general has a cubic runtime (depending on the state space's sparseness), which makes the runtime very sensitive to the size of the state space. In addition, the runtime is linear in the number of epochs introduced. For large state spaces the moments are instead obtained from the *action* of the matrix exponential on a vector (threaded through the epochs), which exploits the sparsity of the rate matrix and avoids forming the dense exponential, giving a substantial speedup for large/high-order/multi-epoch computations (the threshold is controlled by :attr:`~phasegen.settings.Settings.expm_action_min_dim`). Several further optimizations cut the runtime where they apply: the block-counting state space of the single-population standard-coalescent SFS is *flattened* onto the much smaller lineage-counting space, the final unbounded epoch is solved in closed form rather than by exponentiating over the estimated absorption time (:attr:`~phasegen.settings.Settings.closed_form_last_epoch`), and the per-bin solves of a whole spectrum are *batched* into one shared computation. Below we can see the total runtime in seconds for computing the mean tree height, the mean SFS, and the mean two-locus SFS under a 1-epoch standard coalescent over a range of different numbers of lineages and loci.

.. image:: ../../images/execution_times.png
   :alt: Execution times
   :width: 60%
   :align: center

Empirical
---------
Where the exact computation becomes too costly, the statistics can instead be estimated from PhaseGen's own vectorised sampler (:meth:`~phasegen.distributions.Coalescent.to_empirical`). Its cost grows with the number of samples rather than the size of the state space, so it stays roughly flat across the scenarios above. The figure below shows the runtime for drawing 100,000 samples, on the same colour scale as above; the largest case that takes several seconds to compute exactly is sampled in under a tenth of a second.

.. image:: ../../images/sampling_times.png
   :alt: Sampling times
   :width: 60%
   :align: center
