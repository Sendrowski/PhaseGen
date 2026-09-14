.. _introduction:

Introduction
============
``phasegen`` is a population genetics coalescent simulator and parameter inference framework that leverages phase-type theory to provide exact solutions for various population genetic scenarios. ``phasegen`` supports multiple demes, varying population sizes and migration rates, multiple-merger coalescents, and recombination between two loci. To ensure correctness, ``phasegen`` has been extensively tested against `msprime <https://tskit.dev/msprime/docs/stable/intro.html>`_ for a wide variety of demographic scenarios and statistics.

Motivation
----------
Coalescent simulators such as `msprime <https://tskit.dev/msprime/docs/stable/intro.html>`_, while being very fast and flexible, provide stochastic solutions. This necessitates the use of Approximate Bayesian Computation (ABC) for parameter estimation, which can be computationally expensive. A set of tools that do, in principle, provide exact solutions are forward simulators, such as `dadi <https://dadi.readthedocs.io/en/latest>`_ and `moments <https://moments.readthedocs.io/en/latest/index.html>`_. However, forward simulators, while having the great advantage of being able to incorporate selection, have different caveats associated with model initialization, choice of run times, and they tend to be overall less efficient than backward simulations. ``phasegen`` is particularly useful in settings where exact solutions of the coalescent are required. The availability of exact solutions furthermore lends itself to gradient-based parameter estimation, such as maximum likelihood estimation (MLE), which can be more efficient than ABC in some cases.

.. toctree::
   :caption: User Guide
   :maxdepth: 2
   :hidden:

   reference/installation
   reference/quickstart
   reference/distribution_functions
   reference/spectra
   reference/multiple_merger_coalescents
   reference/rewards
   reference/demography
   reference/mutation_configs
   reference/empirical_distributions
   reference/inference
   reference/performance
   reference/miscellaneous

.. toctree::
   :caption: API Reference
   :maxdepth: 1
   :hidden:

   modules/distributions
   modules/coalescent_models
   modules/demography
   modules/rewards
   modules/inference
   modules/config
   modules/state_space
   modules/spectrum
   modules/expm
   modules/settings

.. toctree::
   :caption: Miscellaneous
   :maxdepth: 1
   :hidden:

   modules/citing
   modules/changelog