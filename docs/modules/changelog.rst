.. _modules.changelog:

Changelog
=========

[2.0.0] - 2026-07-10
^^^^^^^^^^^^^^^^^^^^
- Expose the full distribution of any accumulated reward as callable, plottable ``pdf`` / ``cdf`` / ``quantile`` objects, and the :class:`joint distribution <phasegen.distributions.JointRewardDistribution>` of two rewards with its :meth:`marginal <phasegen.distributions.JointRewardDistribution.marginal>` and :meth:`conditional <phasegen.distributions.JointRewardDistribution.conditional>` slices.
- Add a vectorised trajectory sampler (:meth:`to_empirical() <phasegen.distributions.PhaseTypeDistribution.to_empirical>`, :class:`SampledCoalescent <phasegen.distributions.SampledCoalescent>`) as a sampled counterpart of every phase-type distribution.
- Support time-inhomogeneous (multi-epoch) demographies for :meth:`mutational block configurations <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.
- Substantially speed up the closed-form last-epoch moments with a block-triangular sparse LU (strongly-connected-component ordering, ``NATURAL`` column order), giving order-of-magnitude speedups on jSFS and two-locus spectra by lowering the dense/sparse crossover to a few hundred states (:attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`).
- Make the tree-height density exact (the exit-rate reading of the propagated vector, rather than a finite difference of the CDF) and the quantile a vectorised inverse interpolation of the shared hazard grid, rather than a per-level bisection of the CDF.
- Propagate the tree-height cdf / pdf / quantile through the same dense / sparse / matrix-exponential-action machinery as the moments (:attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`), so a large state space is no longer densified into a ``k x k`` propagator.
- Return a public :class:`ConditionalRewardDistribution <phasegen.distributions.ConditionalRewardDistribution>` from :meth:`JointRewardDistribution.conditional() <phasegen.distributions.JointRewardDistribution.conditional>`, carrying its own :attr:`var <phasegen.distributions.ConditionalRewardDistribution.var>` and :meth:`moment() <phasegen.distributions.ConditionalRewardDistribution.moment>`.
- Raise :class:`ModelError <phasegen.errors.ModelError>`, a subclass of ``ValueError``, when the model cannot be evaluated at its parameters, such as a zero population size in an epoch the computation reaches. :class:`Inference <phasegen.inference.Inference>` treats it as an invalid region of the parameter space.
- Make :attr:`JointRewardDistribution.cov <phasegen.distributions.JointRewardDistribution.cov>`, :attr:`JointRewardDistribution.corr <phasegen.distributions.JointRewardDistribution.corr>`, :attr:`EmpiricalJointDistribution.cov <phasegen.distributions.EmpiricalJointDistribution.cov>` and :attr:`EmpiricalJointDistribution.corr <phasegen.distributions.EmpiricalJointDistribution.corr>` properties, as on the other distributions.
- Return one for moments of order zero (:meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`).
- Restrict :class:`~phasegen.rewards.LineageReward` to single-locus coalescents.
- Require the ``value`` and ``half_width`` arguments of :meth:`JointRewardDistribution.window_average() <phasegen.distributions.JointRewardDistribution.window_average>`, with ``half_width`` positive.
- Take the evaluation grid of distribution-function plots as ``t`` (``q`` for quantile functions), as in :meth:`DensityFunction.plot() <phasegen.distributions.DensityFunction.plot>`.
- Draw plots called without ``ax`` on a new figure.
- Treat ``end_time=inf`` as no end time in :class:`~phasegen.distributions.Coalescent` and :class:`~phasegen.distributions.TreeHeightDistribution`.
- Reject non-finite population sizes, migration rates and model parameters, and migration from a population to itself (:class:`~phasegen.demography.Demography`).
- Seed the simulation batches of :class:`~phasegen.distributions.MsprimeCoalescent` from children of a ``numpy.random.SeedSequence`` spawned from its seed, so a given seed yields different replicates than in 1.2.0.
- Require ``record_migration=True`` for the per-deme statistics of a :class:`~phasegen.distributions.MsprimeCoalescent` with more than one deme.
- Return empirical spectrum cdf and pdf values with shape ``(len(t), n + 1)`` (points, bins).
- Put the time axis first in the arrays returned by :meth:`UnfoldedSFSDistribution.accumulate() <phasegen.distributions.UnfoldedSFSDistribution.accumulate>`, :meth:`FoldedSFSDistribution.accumulate() <phasegen.distributions.FoldedSFSDistribution.accumulate>` and :meth:`JointSFSDistribution.accumulate() <phasegen.distributions.JointSFSDistribution.accumulate>`.
- Rename ``SFS2`` to :class:`~sfsutils.spectrum.TwoSFS` and replace the ``fastdfe`` dependency with ``sfsutils-popgen``, whose :mod:`sfsutils` module provides the spectrum classes.
- Replace ``TotalBranchLengthLocusReward`` with :class:`~phasegen.rewards.RestrictedReward`.
- Remove ``Settings.cache_epochs`` and ``Inference.loss_runs``, and raise ``AttributeError`` when assigning an undeclared name on :class:`~phasegen.settings.Settings`.

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
- Add summary statistics: Hudson's :attr:`Coalescent.fst <phasegen.distributions.Coalescent.fst>`, Patterson's f-statistics (:meth:`Coalescent.f2() <phasegen.distributions.Coalescent.f2>`, :meth:`Coalescent.f3() <phasegen.distributions.Coalescent.f3>`, :meth:`Coalescent.f4() <phasegen.distributions.Coalescent.f4>`), Tajima's :meth:`UnfoldedSFSDistribution.tajimas_d() <phasegen.distributions.UnfoldedSFSDistribution.tajimas_d>` with the :meth:`UnfoldedSFSDistribution.theta_pi() <phasegen.distributions.UnfoldedSFSDistribution.theta_pi>` and :meth:`UnfoldedSFSDistribution.theta_w() <phasegen.distributions.UnfoldedSFSDistribution.theta_w>` estimators, and cross-locus linkage via the correlation of coalescence times.
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
