.. _introduction:

Introduction
============
PhaseGen is a population genetic coalescent simulator that leverages phase-type theory to provide exact solutions for various population genetic scenarios. PhaseGen can simulate multiple demes and migration between them, supports recombination between two loci, and handle variable population sizes and migration rates over time. To ensure correctness, PhaseGen has been extensively tested against `msprime <https://tskit.dev/msprime/docs/stable/intro.html> for wide variety of demographic scenarios and statistics.

Motivation
----------
PhaseGen is particularly useful for scenarios where exact solutions are required, such as for maximum likelihood estimation of demographic parameters.