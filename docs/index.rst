.. _introduction:

Introduction
============
``phasegen`` is a population genetic coalescent simulator and parameter inference framework that leverages phase-type theory to provide exact solutions for various population genetic scenarios. ``phasegen`` supports multiple demes, varying population sizes and migration rates, multiple-merger coalescents, and recombination between two loci. To ensure correctness, ``phasegen`` has been extensively tested against `msprime <https://tskit.dev/msprime/docs/stable/intro.html>`_ for a wide variety of demographic scenarios and statistics.

Motivation
----------
Coalescent simular such as `msprime <https://tskit.dev/msprime/docs/stable/intro.html>`_, while being very fast and flexible, provide stochastic solutions. This necessitates the use of Approximate Bayesian Computation (ABC) for parameter estimation, which can be computationally expensive. A set of tools that do, in principle, provide exact solutions are forward simulators, such as `dadi <https://dadi.readthedocs.io/en/latest>`_ and `moments <https://moments.readthedocs.io/en/latest/index.html>`_. However, forward simulators, while having the great advantage of being able to incorporate selection, have different caveats associated with model initialization, choice of run times, and they tend to be overall less efficient than backward simulations. ``phasegen`` is particularly useful in settings where exact solutions of the coalescent are required. The availability of exact solutions furthermore lends itself to gradient-based parameter estimation, such as maximum likelihood estimation (MLE), which can be more efficient than ABC in some cases.

Contents
--------

.. toctree::
   :caption: Python Reference
   :maxdepth: 2

   reference/Python/installation
   reference/Python/quickstart
   reference/Python/coalescent
   reference/Python/rewards
   reference/Python/demography
   reference/Python/mutation_configs
   reference/Python/inference
   reference/Python/exponentiation_backend
   reference/performance
   reference/Python/miscellaneous

.. toctree::
   :caption: R Reference
   :maxdepth: 2

   reference/R/installation
   reference/R/quickstart
   reference/R/coalescent
   reference/R/rewards
   reference/R/demography
   reference/R/mutation_configs
   reference/R/exponentiation_backend
   reference/performance
   reference/R/miscellaneous

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   modules/distributions
   modules/coalescent_models
   modules/demography
   modules/rewards
   modules/inference
   modules/config
   modules/state_space
   modules/norms
   modules/spectrum