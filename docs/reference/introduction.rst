.. _introduction:

Introduction
============
PhaseGen is a novel population genetic coalescent simulator that leverages phase-type theory to address the limitations of existing tools and provide exact solutions for various population genetic scenarios. Developed in Python, PhaseGen can simulate multiple demes and migration between them, support recombination between a number of loci, and handle variable population sizes and migration rates over time.

Motivation
----------
Population genetic coalescent simulators are essential tools in evolutionary biology, allowing researchers to explore complex genetic scenarios and develop hypotheses regarding evolutionary processes. Popular simulators are msprime and δaδi. However, msprime doesn't provide exact solutions, necessitating the use of approximate Bayesian computation (ABC) for parameter estimation, while δaδi relies on forward simulations, which has its own drawbacks. PhaseGen has been developed to fill a gap inbetween these two approaches.

How it works
------------
PhaseGen leverages phase-type theory to provide exact solutions for the coalescent. This development builds upon the foundation laid by Hobolth et al. (2018) but extends and adapts phase-type theory for use in a coalescent context and for time-inhomogeneous models (Hobolth & Jensen, 2011).

PhaseGen is implemented in Python and will be made available through the conda package manager. It will also provide an R interface, enabling integration with the R ecosystem and distribution through the Comprehensive R Archive Network (CRAN). To ensure the accuracy and reliability of PhaseGen, all results are rigorously tested against those obtained from msprime.

Features
--------
PhaseGen offers several key features, including:

* Simulation of multiple demes and migration between them
* Support for recombination between a number of loci
* Compatibility with various coalescent models, such as the Beta coalescent
* Capability to simulate variable population sizes and migration rates over time
* Integration with an efficient maximum likelihood estimation (MLE) framework for parameter inference
