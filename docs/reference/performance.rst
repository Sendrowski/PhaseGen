.. _performance:

Performance
===========

State Space Size
----------------
The size of the state space can grow rapidly with the complexity of the demographic scenario, i.e. the number of lineages, demes and loci as shown below.

.. image:: https://github.com/Sendrowski/fastDFE/tree/master/docs/images/state_space_sizes.png?raw=true
   :alt: Execution times

Runtime
-------
To obtain moments we need to exponentiate matrices whose size equals the state space size times ``k+1`` where ``k`` is the order of the moment. Matrix exponentiation in general has a cubic runtime (depending on the state space's sparseness), which makes the runtime very sensitive to the size of the state space. In addition, the runtime is linear in the number of epochs introduced. Below we can see the total runtime in seconds for computing the tree mean height and mean SFS under a 1-epoch standard coalescent over a range of different numbers of lineages and loci.

.. image:: https://github.com/Sendrowski/fastDFE/tree/master/docs/images/executions_times.png?raw=true
   :alt: Execution times

