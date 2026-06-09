.. _reference.performance:

Performance
===========

State Space Size
----------------
The size of the state space can grow rapidly with the complexity of the demographic scenario, i.e. the number of lineages, demes and loci as shown below.

.. image:: https://github.com/Sendrowski/PhaseGen/blob/master/docs/images/state_space_sizes.png?raw=true
   :alt: Execution times

State Space Construction
------------------------
Constructing the state space (enumerating the states and assembling the rate matrix) is accelerated with `numba <https://numba.pydata.org/>`__, speeding it up by one to several orders of magnitude for larger state spaces (e.g. roughly 650x for a three-deme joint-SFS state space). The acceleration is used automatically when numba is available and falls back to a pure-Python implementation otherwise; it can be disabled by setting :attr:`~phasegen.settings.Settings.use_numba` to ``False``.

Runtime
-------
To obtain moments we need to exponentiate matrices whose size equals the state space size times ``k+1`` where ``k`` is the order of the moment. Matrix exponentiation in general has a cubic runtime (depending on the state space's sparseness), which makes the runtime very sensitive to the size of the state space. In addition, the runtime is linear in the number of epochs introduced. For large state spaces the moments are instead obtained from the *action* of the matrix exponential on a vector (threaded through the epochs), which exploits the sparsity of the rate matrix and avoids forming the dense exponential, giving a substantial speedup for large/high-order/multi-epoch computations (the threshold is controlled by :attr:`~phasegen.settings.Settings.expm_action_min_dim`). Below we can see the total runtime in seconds for computing the tree mean height and mean SFS under a 1-epoch standard coalescent over a range of different numbers of lineages and loci.

.. image:: https://github.com/Sendrowski/PhaseGen/blob/master/docs/images/execution_times.png?raw=true
   :alt: Execution times
   :width: 60%
   :align: center


