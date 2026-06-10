.. _modules.changelog:

Changelog
=========

[1.1.0] - 2026-06-10
^^^^^^^^^^^^^^^^^^^^
- Add the joint (multi-population) site-frequency spectrum via :meth:`~phasegen.distributions.Coalescent.jsfs`.
- Add the two-locus site-frequency spectrum under recombination via :meth:`~phasegen.distributions.Coalescent.sfs2`, with support for multiple-merger coalescents.
- Add summary statistics: Hudson's :meth:`~phasegen.distributions.Coalescent.fst`, Patterson's f-statistics (:meth:`~phasegen.distributions.Coalescent.f2`, :meth:`~phasegen.distributions.Coalescent.f3`, :meth:`~phasegen.distributions.Coalescent.f4`), Tajima's :meth:`~phasegen.distributions.UnfoldedSFSDistribution.tajimas_d` with the :meth:`~phasegen.distributions.UnfoldedSFSDistribution.theta_pi` and :meth:`~phasegen.distributions.UnfoldedSFSDistribution.theta_w` estimators, and cross-locus linkage via the correlation of coalescence times.
- Accelerate state-space construction with `numba <https://numba.pydata.org/>`__, which is now a required dependency.
- Compute moments of large state spaces from the sparse action of the matrix exponential (threaded over epochs), controlled by :attr:`~phasegen.settings.Settings.expm_action_min_dim`.
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
