.. _modules.changelog:

Changelog
=========

[Unreleased]
^^^^^^^^^^^^
- Create demographies from :class:`msprime.Demography` and :class:`demes.Graph` objects with :meth:`Demography.from_msprime() <phasegen.demography.Demography.from_msprime>` and :meth:`Demography.from_demes() <phasegen.demography.Demography.from_demes>`, and convert to a :class:`demes.Graph` with :meth:`Demography.to_demes() <phasegen.demography.Demography.to_demes>`.
- Add an :paramref:`~phasegen.demography.Demography.plot.alpha` argument to the demography plots.
- Keep the population names of the joint SFS in its standard deviation and in all sampled spectra.
- Estimate F\ :sub:`ST` and the f-statistics on :class:`~phasegen.distributions.SampledCoalescent`.
- Add the third and fourth raw moments (:attr:`PhaseTypeDistribution.m3 <phasegen.distributions.PhaseTypeDistribution.m3>`, :attr:`PhaseTypeDistribution.m4 <phasegen.distributions.PhaseTypeDistribution.m4>`) to all exact distributions, the bin correlation :attr:`JointSFSDistribution.corr <phasegen.distributions.JointSFSDistribution.corr>` and the cross-locus covariance :attr:`TwoLocusSFSDistribution.cov <phasegen.distributions.TwoLocusSFSDistribution.cov>`, matching their sampled counterparts.
- Speed up joint densities on multi-epoch demographies by evaluating the transform grid in large batches.
- Add a User Guide page of real-world examples, starting with the out-of-Africa model of Gutenkunst et al. (2009).
- Convert the admixture events of an :class:`msprime.Demography` to pulses and a population split in :meth:`Demography.from_msprime() <phasegen.demography.Demography.from_msprime>`.
- Report populations of the demography without sampled lineages at the info level rather than as a warning.

[2.0.0] - 2026-10-06
^^^^^^^^^^^^^^^^^^^^
- Expose the full distribution of any accumulated reward as callable, plottable ``pdf`` / ``cdf`` / ``quantile`` objects, and the joint distribution of two rewards (:class:`~phasegen.distributions.JointRewardDistribution`) with its marginal and conditional distributions.
- Substantially speed up the moments of large state spaces with sparse linear solves and matrix-exponential actions, by an order of magnitude on the joint and two-locus spectra.
- Support time-inhomogeneous (multi-epoch) demographies for mutational block configurations (:meth:`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`), and add them for the joint and two-locus spectra and for folded and deme-resolved layouts, described by :class:`~phasegen.distributions.MutationLayout`.
- Add a fast vectorised trajectory sampler (:meth:`PhaseTypeDistribution.to_empirical() <phasegen.distributions.PhaseTypeDistribution.to_empirical>`, :class:`~phasegen.distributions.SampledCoalescent`) as a sampled counterpart of every phase-type distribution.
- Add admixture pulses with :class:`~phasegen.demography.Pulse`.
- Start the coalescent from a weighted mixture of lineage or locus configurations with :class:`~phasegen.initial.InitialDistribution`.
- Improve parameter inference with :class:`~phasegen.inference.Inference`.
- Make numerous smaller improvements and bug fixes throughout.

[1.2.0] - 2026-06-13
^^^^^^^^^^^^^^^^^^^^
- Add the cross-locus correlation of the two-locus SFS via :attr:`TwoLocusSFSDistribution.corr <phasegen.distributions.TwoLocusSFSDistribution.corr>`.
- Build large rate matrices sparsely with an explicit state-space size cap (:attr:`Settings.dense_rate_matrix_max_states <phasegen.settings.Settings.dense_rate_matrix_max_states>`, :attr:`Settings.max_state_space_size <phasegen.settings.Settings.max_state_space_size>`).
- Evaluate the final unbounded epoch in closed form by default and batch the per-bin spectrum solves, substantially speeding up SFS/jSFS/2-SFS moments (:attr:`Settings.closed_form_last_epoch <phasegen.settings.Settings.closed_form_last_epoch>`).
- Compute the single-population standard-coalescent SFS flattening weights in closed form, avoiding the partition-sized block-counting state space, so large-``n`` SFS (and SFS-based inference) is much faster.
- Raise a clear error for demographies that never absorb (isolated demes or blocked migration) instead of returning a meaningless value.

[1.1.1] - 2026-06-10
^^^^^^^^^^^^^^^^^^^^
- Relax the ``numpy`` upper bound (``>=1.26.4``) to allow numpy 2, so phasegen can coexist with current, numpy-2-built msprime/tskit.

[1.1.0] - 2026-06-10
^^^^^^^^^^^^^^^^^^^^
- Add the joint (multi-population) site-frequency spectrum via :attr:`Coalescent.jsfs <phasegen.distributions.Coalescent.jsfs>`.
- Add the two-locus site-frequency spectrum under recombination via :attr:`Coalescent.sfs2 <phasegen.distributions.Coalescent.sfs2>`, with support for multiple-merger coalescents.
- Add summary statistics: Hudson's :attr:`Coalescent.fst <phasegen.distributions.Coalescent.fst>`, Patterson's f-statistics (:meth:`Coalescent.f2() <phasegen.distributions.Coalescent.f2>`, :meth:`Coalescent.f3() <phasegen.distributions.Coalescent.f3>`, :meth:`Coalescent.f4() <phasegen.distributions.Coalescent.f4>`), Tajima's :attr:`UnfoldedSFSDistribution.tajimas_d <phasegen.distributions.UnfoldedSFSDistribution.tajimas_d>` with the :attr:`UnfoldedSFSDistribution.theta_pi <phasegen.distributions.UnfoldedSFSDistribution.theta_pi>` and :attr:`UnfoldedSFSDistribution.theta_w <phasegen.distributions.UnfoldedSFSDistribution.theta_w>` estimators, and cross-locus linkage via the correlation of coalescence times.
- Accelerate state-space construction with `numba <https://numba.pydata.org/>`__, which is now a required dependency.
- Compute moments of large state spaces from the sparse action of the matrix exponential (threaded over epochs), controlled by :attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`.
- Validate the new statistics against msprime/tskit ground truth, including within the scenario-comparison workflow (Kingman and multiple-merger models, and beyond the two-lineage case).
- Use the ``spawn`` start method for the worker pool on macOS to avoid fork/numba deadlocks.
- Documentation: add a dedicated *Spectra & summary statistics* reference page and drop the exponentiation-backend page.

[1.0.2] - 2025-07-14
^^^^^^^^^^^^^^^^^^^^
- Speed up single population single locus Kingman coalescent SFS computations by flattening block counting state space.
- Rescale rate matrix instead of recomputing it for new epochs in the one population, one locus case.
- Relocate phase-type settings to :class:`~phasegen.settings.Settings` class.

[1.0.1] - 2025-02-17
^^^^^^^^^^^^^^^^^^^^
- Minor improvements in logging and first release archived in Zenodo.

[1.0.0] - 2024-08-05
^^^^^^^^^^^^^^^^^^^^
- First stable release
