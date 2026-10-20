.. _modules.changelog:

Changelog
=========

[Unreleased]
^^^^^^^^^^^^
- Require Python 3.11 or newer and support Python 3.14, pandas 3 and jsonpickle 4.
- Differentiate inference losses exactly with respect to the parameters of the demography, the coalescent model and the recombination rate, at a cost independent of the number of parameters (:attr:`Inference.gradient <phasegen.inference.Inference.gradient>`), and expose the derivatives of moments of any order and of the mean two-locus SFS through :meth:`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>`, :meth:`Coalescent.gradient() <phasegen.distributions.Coalescent.gradient>` and :class:`~phasegen.gradient.Gradient`.
- Add standard errors from the observed Fisher information with :meth:`Inference.fisher_information() <phasegen.inference.Inference.fisher_information>`, :meth:`Inference.covariance() <phasegen.inference.Inference.covariance>` and :meth:`Inference.standard_errors() <phasegen.inference.Inference.standard_errors>`, or from the Godambe sandwich covariance with ``method='godambe'``, which holds for linked sites, and propagate them to the population size and migration rate trajectories with :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` and ``interval='fisher'`` or ``interval='godambe'`` in the trajectory plots, where ``interval`` replaces ``include_bootstraps``.
- Add a penalty on the parameters to the loss with :attr:`Inference.penalty <phasegen.inference.Inference.penalty>`, which keeps the exact gradient, and compare models with :meth:`Inference.aic() <phasegen.inference.Inference.aic>` and :meth:`Inference.likelihood_ratio_test() <phasegen.inference.Inference.likelihood_ratio_test>`.
- Draw the ground truth in the inference plots with ``truth`` in :meth:`Inference.plot_bootstraps() <phasegen.inference.Inference.plot_bootstraps>`, :meth:`Inference.plot_pop_sizes() <phasegen.inference.Inference.plot_pop_sizes>`, :meth:`Inference.plot_migration() <phasegen.inference.Inference.plot_migration>` and :meth:`Inference.plot_demography() <phasegen.inference.Inference.plot_demography>`.
- Optimize parameters as their logarithms with :attr:`Inference.log_scale <phasegen.inference.Inference.log_scale>`, drawing their start points log-uniformly.
- Make multi-start optimizations with gradient methods step back from parameters at which the model is invalid, and warn where restarts reach the same loss at different estimates. ``method_mle='trust-constr'`` is no longer accepted, since it could stop in the invalid region.
- Speed up variances and higher moments by up to about 1.5x by evaluating the dense epoch exponentials in stacks, and the large ones block-wise, and models with very fast rates, such as a Beta coalescent with ``alpha`` close to 2, by orders of magnitude by ending matrix-exponential actions once the state distribution is stationary.
- Bin each parameter of :meth:`Inference.plot_bootstraps() <phasegen.inference.Inference.plot_bootstraps>` over its own range.
- Run functions defined in scripts and notebooks in parallel workers started by ``spawn``, as on macOS, keep every sample of the empirical joint SFS, warn about a large state space once per type and size, and limit the BLAS threads while moments, their gradients and inferences are evaluated (:attr:`Settings.blas_threads <phasegen.settings.Settings.blas_threads>`), since the NumPy and SciPy wheels each load an OpenBLAS whose thread pools stall each other. This makes `threadpoolctl <https://github.com/joblib/threadpoolctl>`__ a required dependency.
- Access the state spaces of a coalescent by key in :attr:`Coalescent.state_spaces <phasegen.distributions.Coalescent.state_spaces>`, which builds each one on first access. The accessor properties such as ``Coalescent.block_counting_state_space`` are deprecated.

[2.1.0] - 2026-10-12
^^^^^^^^^^^^^^^^^^^^
- Create demographies from :class:`msprime.Demography` and :class:`demes.Graph` objects with :meth:`Demography.from_msprime() <phasegen.demography.Demography.from_msprime>` and :meth:`Demography.from_demes() <phasegen.demography.Demography.from_demes>`, and convert to a :class:`demes.Graph` with :meth:`Demography.to_demes() <phasegen.demography.Demography.to_demes>`.
- Move the lineages of a :class:`~phasegen.demography.PopulationSplit` within a short window in the order of the events, as for a :class:`~phasegen.demography.Pulse`. This removes a bias of about 1% from models with splits and corrects a pulse at or shortly after a split, at up to about twice the runtime.
- Add the third and fourth raw moments (:attr:`PhaseTypeDistribution.m3 <phasegen.distributions.PhaseTypeDistribution.m3>`, :attr:`PhaseTypeDistribution.m4 <phasegen.distributions.PhaseTypeDistribution.m4>`), :attr:`JointSFSDistribution.corr <phasegen.distributions.JointSFSDistribution.corr>` and :attr:`TwoLocusSFSDistribution.cov <phasegen.distributions.TwoLocusSFSDistribution.cov>` to the exact distributions, and F\ :sub:`ST` and the f-statistics to :class:`~phasegen.distributions.SampledCoalescent`.
- Accept descendant configurations of the joint SFS by population name in :class:`~phasegen.rewards.JointSFSReward` and the bins of the joint spectra, and keep the population names in all joint spectra.
- Speed up joint densities on multi-epoch demographies by evaluating the transform grid in large batches.

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
