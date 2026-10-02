"""
Run the code presented in the PhaseGen manuscript (Genetics, doi:10.1093/genetics/iyaf135) against the current API.

Each test mirrors one manuscript code block or figure script in ``scripts/`` (``manuscript_code_blocks.py``,
``plot_manuscript_*.py``, ``run_manuscript_*.py``): the same calls with the same arguments, with fewer evaluation
points, optimization runs and bootstrap replicates where the original is expensive, figures drawn without being shown
and no files written.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phasegen as pg


def _assert_sfs_consistent(coal: pg.Coalescent, n: int) -> None:
    """
    Check the expected SFS and the SFS correlation matrix of a coalescent with ``n`` lineages for shape and sanity.

    :param coal: Coalescent.
    :param n: Total number of lineages.
    """
    sfs = coal.sfs.mean
    corr = coal.sfs.corr

    assert sfs.data.shape == (n + 1,)
    assert np.all(np.isfinite(sfs.data))
    assert np.all(sfs.polymorphic > 0)

    # the polymorphic bins partition the total branch length
    assert sfs.polymorphic.sum() == pytest.approx(coal.total_branch_length.mean, rel=1e-8)

    assert corr.data.shape == (n + 1, n + 1)
    poly = corr.data[1:n, 1:n]
    assert np.all(np.isfinite(poly))
    assert np.all(np.abs(poly) <= 1 + 1e-8)
    assert np.allclose(np.diag(poly), 1, atol=1e-8)


def _assert_tree_height_pdf(coal: pg.Coalescent) -> np.ndarray:
    """
    Check the tree-height density on a grid up to its 99% quantile.

    :param coal: Coalescent.
    :return: The evaluation grid.
    """
    q = coal.tree_height.quantile(0.99)
    t = np.linspace(0, q, 100)

    assert np.isfinite(q) and q > 0
    assert coal.tree_height.cdf(q) == pytest.approx(0.99, abs=1e-6)

    pdf = coal.tree_height.pdf(t)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= -1e-10)

    return t


def _manuscript_inference() -> pg.Inference:
    """
    The inference block of ``manuscript_code_blocks.py`` with one optimization run, two bootstrap replicates and no
    parallelization.

    :return: Inference object, not yet run.
    """
    return pg.Inference(
        coal=lambda t, Ne: pg.Coalescent(
            n=10,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1, t: Ne}}
            )
        ),
        observation=pg.SFS(
            [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
        ),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic,
            modelled=(
                coal.sfs.mean.polymorphic / coal.sfs.mean.Theta * obs.Theta
            )
        ),
        bounds=dict(t=(0, 4), Ne=(0.1, 10)),
        resample=lambda sfs, _: sfs.resample(),
        do_bootstrap=True,
        parallelize=False,
        n_runs=1,
        n_bootstraps=2,
        pbar=False,
        seed=0
    )


@pytest.fixture(scope='module')
def manuscript_inference() -> pg.Inference:
    """
    The manuscript inference, run once for the tests of this module.

    :return: Inference object.
    """
    inf = _manuscript_inference()
    inf.run()

    return inf


def test_code_block_inference(manuscript_inference):
    """
    The inference block of ``manuscript_code_blocks.py`` runs and infers parameters within their bounds.
    """
    inf = manuscript_inference

    assert set(inf.params_inferred) == {'t', 'Ne'}
    assert 0 <= inf.params_inferred['t'] <= 4
    assert 0.1 <= inf.params_inferred['Ne'] <= 10
    assert np.isfinite(inf.loss_inferred)
    assert len(inf.bootstraps) == 2
    assert np.all(np.isfinite(inf.bootstraps[['t', 'Ne']].to_numpy(dtype=float)))


def test_plot_manuscript_inference_example(manuscript_inference, tmp_path):
    """
    ``plot_manuscript_inference_example.py`` runs: function-evaluation counts, fitted against observed SFS, the
    population-size and bootstrap plots, and serialization.
    """
    pg.Backend.register(pg.SciPyExpmBackend())

    inf = manuscript_inference

    nfev_runs = inf.runs['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))
    nfev_bootstraps = inf.bootstraps['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))

    assert nfev_runs.mean() > 0
    assert nfev_bootstraps.mean() > 0

    spectra = pg.Spectra.from_spectra(dict(
        fitted=inf.dist_inferred.sfs.mean /
               (inf.dist_inferred.sfs.mean.theta * inf.dist_inferred.sfs.mean.n_sites) *
               (inf.observation.theta * inf.observation.n_sites),
        observed=inf.observation
    ))

    # the fitted SFS is scaled to the observed number of segregating sites
    assert spectra['fitted'].n_polymorphic == pytest.approx(inf.observation.n_polymorphic, rel=1e-8)

    _, axs = plt.subplots(2, 2, figsize=(6, 5))

    spectra.plot(ax=axs[0, 0], show=False, title='SFS comparison')
    inf.plot_pop_sizes(ax=axs[0, 1], show=False)
    inf.plot_bootstraps(ax=axs[1], show=False, kwargs={'bins': 30}, title=['Marginal distribution'] * 2)

    plt.tight_layout()

    inf.to_file(str(tmp_path / 'inference.json'))

    assert pg.Inference.from_file(str(tmp_path / 'inference.json')).params_inferred == inf.params_inferred


def test_plot_manuscript_kingman_example():
    """
    The Kingman code block of ``plot_manuscript_kingman_example.py`` and its figure panels.
    """
    # 8 lineages, pop size of 1
    coal = pg.Coalescent(n=8)

    # tree height PDF, SFS and 2-SFS
    pdf = coal.tree_height.pdf
    sfs = coal.sfs.mean
    sfs2 = coal.sfs.corr

    # standard Kingman results: E[T_MRCA] = 2 (1 - 1/n), E[xi_i] = 2 / i
    assert coal.tree_height.mean == pytest.approx(2 * (1 - 1 / 8), rel=1e-10)
    assert sfs.polymorphic == pytest.approx(2 / np.arange(1, 8), rel=1e-10)
    assert callable(pdf)
    assert sfs2.data.shape == (9, 9)

    _assert_sfs_consistent(coal, 8)
    _assert_tree_height_pdf(coal)

    _, axs = plt.subplots(2, 2)

    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)


def test_plot_manuscript_2_epoch_example():
    """
    The bottleneck code block of ``plot_manuscript_2_epoch_example.py`` and its figure panels.
    """
    # bottleneck at time 1
    coal = pg.Coalescent(
        n={'pop0': 8},
        demography=pg.Demography(
            pop_sizes={
                'pop0': {0: 1, 1: .2}
            }
        )
    )

    _assert_sfs_consistent(coal, 8)
    _assert_tree_height_pdf(coal)

    # the bottleneck shortens the tree relative to the constant-size Kingman coalescent
    assert coal.tree_height.mean < pg.Coalescent(n=8).tree_height.mean

    _, axs = plt.subplots(2, 2)

    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)


def test_plot_manuscript_mmc_example():
    """
    The Beta-coalescent code block of ``plot_manuscript_mmc_example.py`` and its figure panels.
    """
    # beta coalescent, pop size of 1
    coal = pg.Coalescent(
        n=8,  # 8 lineages
        model=pg.BetaCoalescent(1.4)
    )

    _assert_sfs_consistent(coal, 8)
    _assert_tree_height_pdf(coal)

    _, axs = plt.subplots(2, 2)

    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)


def test_plot_manuscript_migration_example():
    """
    The two-deme code block of ``plot_manuscript_migration_example.py`` and its figure panels.
    """
    # migration from pop0 to pop1 back in time
    coal = pg.Coalescent(
        n={'pop0': 4, 'pop1': 4},
        demography=pg.Demography(
            pop_sizes={'pop0': 1, 'pop1': 2},
            migration_rates={
                ('pop0', 'pop1'): 0.3
            }
        )
    )

    _assert_sfs_consistent(coal, 8)
    _assert_tree_height_pdf(coal)

    # the per-deme spectra partition the total spectrum
    demes = coal.sfs.demes
    assert demes['pop0'].mean.data + demes['pop1'].mean.data == pytest.approx(coal.sfs.mean.data, rel=1e-8)

    _, axs = plt.subplots(2, 2)

    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)

    pg.Spectra(dict(total=coal.sfs.mean, pop0=coal.sfs.demes['pop0'].mean, pop1=coal.sfs.demes['pop1'].mean)).plot(
        ax=axs[1, 0], show=False, title='Expected SFS'
    )

    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)


def test_plot_manuscript_recombination_example():
    """
    The two-locus code block of ``plot_manuscript_recombination_example.py`` and its figure panels on coarser grids.
    """

    def get_cov(r, N1):
        return pg.Coalescent(
            n={'pop0': 2},
            loci=pg.LocusConfig(
                n=2, recombination_rate=r / 2
            ),
            demography=pg.Demography(
                pop_sizes={'pop0': {0: 1, 1: N1}}
            )
        ).tree_height.loci.cov[0, 1]

    def get_coal(r: float, N1: float) -> pg.Coalescent:
        return pg.Coalescent(
            n={'pop0': 2},
            loci=pg.LocusConfig(
                n=2, recombination_rate=r / 2
            ),
            demography=pg.Demography(
                pop_sizes={'pop0': {0: 1, 1: N1}}
            )
        )

    Ns = np.logspace(-1, 1, 2)

    _, axs = plt.subplots(2, 2)

    rs = np.linspace(0, 3, 50)
    for N1 in Ns:
        pdf = get_coal(10, N1).tree_height.pdf(rs)
        assert np.all(np.isfinite(pdf)) and np.all(pdf >= -1e-10)
        axs[0, 1].plot(rs, pdf)

    rs = np.logspace(-1, 1, 3)
    for N1 in Ns:
        cov = np.array([get_cov(r, N1) for r in rs])
        corr = np.array([get_coal(r, N1).tree_height.loci.corr[0, 1] for r in rs])

        # linkage decays with recombination
        assert np.all(cov > 0) and np.all(np.diff(cov) < 0)
        assert np.all((corr > 0) & (corr <= 1)) and np.all(np.diff(corr) < 0)

        axs[1, 0].plot(rs, cov)
        axs[1, 1].plot(rs, corr)


@pytest.mark.slow
def test_plot_manuscript_complex_example():
    """
    Both code blocks of ``plot_manuscript_complex_example.py`` and the summary statistics listed after them.
    """
    # first code block in the manuscript
    coal = pg.Coalescent(
        n=pg.LineageConfig({'pop0': 3, 'pop1': 5}),
        model=pg.BetaCoalescent(alpha=1.7),
        demography=pg.Demography(
            pop_sizes={
                'pop0': {0: 1.0},
                'pop1': {0: 1.2, 5: 0.1, 5.5: 0.8}
            },
            migration_rates={
                ('pop0', 'pop1'): {0: 0.2, 8: 0.3},
                ('pop1', 'pop0'): {0: 0.5}
            }
        )
    )

    # second code block in the manuscript
    _, axs = plt.subplots(2, 2, figsize=(7, 6))
    t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

    coal.demography.plot(ax=axs[0, 0], show=False, t=t)
    axs[0, 0].legend(prop={'size': 6}, loc='center left')
    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations')

    plt.tight_layout()

    mean = coal.tree_height.mean
    var = coal.total_branch_length.var

    q = coal.tree_height.quantile(0.99)
    pdf = coal.tree_height.pdf(np.linspace(0, q, 100))
    cdf = coal.tree_height.cdf(np.linspace(0, q, 100))

    sfs = coal.sfs.mean
    sfs2 = coal.sfs.cov

    sfs_pop0 = coal.sfs.demes['pop0'].mean

    m3 = coal.moment(3, (pg.TotalBranchLengthReward(),) * 3)

    assert np.isfinite(mean) and mean > 0
    assert np.isfinite(var) and var > 0
    assert np.all(np.isfinite(pdf)) and np.all(pdf >= -1e-10)
    assert np.all(np.diff(cdf) >= -1e-10) and cdf[0] == pytest.approx(0, abs=1e-10)
    assert cdf[-1] == pytest.approx(0.99, abs=1e-6)
    assert sfs2.data.shape == (9, 9) and np.all(np.isfinite(sfs2.data))
    assert np.all(sfs_pop0.data <= sfs.data + 1e-10)
    assert np.isfinite(m3)

    _assert_sfs_consistent(coal, 8)


def test_plot_manuscript_2_epoch_computation_example():
    """
    ``plot_manuscript_2_epoch_computation_example.py``: the singleton-doubleton covariance from uncentered,
    unpermuted cross moments equals the centered and permuted moment, and the figure curves on a coarse grid.
    """
    pg.Settings.regularize = False

    def get_coal(N1: float, N2: float, t: float) -> pg.Coalescent:
        return pg.Coalescent(
            n=3,
            demography=pg.Demography(
                pop_sizes={
                    'pop_0': {0: N1, t: N2}
                }
            )
        )

    def get_moment(coal: pg.Coalescent) -> float:
        return coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)])

    N1 = 1
    N2 = 0.5
    t = 2

    coal = get_coal(N1=N1, N2=N2, t=t)

    m12 = coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)], center=False, permute=False)
    m21 = coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(1)], center=False, permute=False)
    m1 = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)], center=False, permute=False)
    m2 = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(2)], center=False, permute=False)

    # center and permute
    m = (m12 + m21) / 2 - m1 * m2

    m_expected = get_moment(coal)

    assert m == pytest.approx(m_expected, rel=1e-12)
    assert np.isfinite(m)

    t_max = 100
    x = np.linspace(0.01, t_max, 5)

    curves = [
        [get_moment(get_coal(N1=N1, N2=N2, t=t)) for N1 in x],
        [get_moment(get_coal(N1=N1, N2=N2, t=t)) for N2 in x],
        [get_moment(get_coal(N1=N1, N2=N2, t=t)) for t in x],
        coal.accumulate(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)], end_times=x)
    ]

    for curve in curves:
        assert np.shape(curve) == x.shape
        assert np.all(np.isfinite(curve))
        plt.plot(x, curve)

    # accumulating to a time far beyond absorption recovers the full covariance
    assert curves[3][-1] == pytest.approx(m_expected, rel=1e-8)


def test_run_manuscript_1_epoch_computation_example():
    """
    ``run_manuscript_1_epoch_computation_example.py``: moments of a truncated coalescent.
    """
    pg.Settings.regularize = False

    coal = pg.Coalescent(n=3, end_time=3)

    m = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)])
    tree_height = coal.tree_height.mean

    # truncation at time 3 bounds the accumulated rewards from above
    full = pg.Coalescent(n=3)
    assert 0 < m < full.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)])
    assert 0 < tree_height < min(3, full.tree_height.mean)
