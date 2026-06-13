.. _modules.changelog:

Changelog
=========

[1.2.0] - 2026-06-13
^^^^^^^^^^^^^^^^^^^^
- Add the cross-locus correlation of the two-locus SFS via :meth:`TwoLocusSFSDistribution.corr() <phasegen.distributions.TwoLocusSFSDistribution.corr>`.
- Build large rate matrices sparsely with an explicit state-space size cap (:attr:`Settings.dense_rate_matrix_max_states <phasegen.settings.Settings.dense_rate_matrix_max_states>`, :attr:`Settings.max_state_space_size <phasegen.settings.Settings.max_state_space_size>`).
- Evaluate the final unbounded epoch in closed form by default and batch the per-bin spectrum solves, substantially speeding up SFS/jSFS/2-SFS moments (:attr:`Settings.closed_form_last_epoch <phasegen.settings.Settings.closed_form_last_epoch>`).
- Compute the single-population standard-coalescent SFS flattening weights in closed form, avoiding the partition-sized block-counting state space, so large-``n`` SFS (and SFS-based inference) is much faster.
- Raise a clear error for demographies that never absorb (isolated demes or blocked migration) instead of returning a meaningless value.

[1.1.1] - 2026-06-10
^^^^^^^^^^^^^^^^^^^^
- Relax the ``numpy`` upper bound (``>=1.26.4``) to allow numpy 2, so phasegen can coexist with current, numpy-2-built msprime/tskit.

[1.1.0] - 2026-06-10
^^^^^^^^^^^^^^^^^^^^
- Add the joint (multi-population) site-frequency spectrum via :meth:`Coalescent.jsfs() <phasegen.distributions.Coalescent.jsfs>`.
- Add the two-locus site-frequency spectrum under recombination via :meth:`Coalescent.sfs2() <phasegen.distributions.Coalescent.sfs2>`, with support for multiple-merger coalescents.
- Add summary statistics: Hudson's :meth:`Coalescent.fst() <phasegen.distributions.Coalescent.fst>`, Patterson's f-statistics (:meth:`Coalescent.f2() <phasegen.distributions.Coalescent.f2>`, :meth:`Coalescent.f3() <phasegen.distributions.Coalescent.f3>`, :meth:`Coalescent.f4() <phasegen.distributions.Coalescent.f4>`), Tajima's :meth:`UnfoldedSFSDistribution.tajimas_d() <phasegen.distributions.UnfoldedSFSDistribution.tajimas_d>` with the :meth:`UnfoldedSFSDistribution.theta_pi() <phasegen.distributions.UnfoldedSFSDistribution.theta_pi>` and :meth:`UnfoldedSFSDistribution.theta_w() <phasegen.distributions.UnfoldedSFSDistribution.theta_w>` estimators, and cross-locus linkage via the correlation of coalescence times.
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
