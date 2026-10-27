"""
Smoke tests replaying the code published in the PhaseGen manuscript.

Each test reruns a code listing shown in a figure of the manuscript, together with the calls of the script its
caption cites (``scripts/plot_manuscript_*.py``, ``scripts/infer_mmc_kingman_2sfs.py``), or the computation of an
appendix example, with smaller samples and fewer runs, bootstraps and grid points. The tests check that the code runs,
not the values it returns.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

import phasegen as pg


def _plot_overview(coal: pg.Coalescent):
    """
    The tree height density, expected SFS and SFS correlations, as plotted in the panels of the example figures.
    """
    _, axs = plt.subplots(2, 2)

    coal.tree_height.pdf.plot(ax=axs[0, 1], show=False)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)


def test_kingman_example():
    """
    Figure 1: tree height density, expected SFS and SFS correlations under the Kingman coalescent.
    """
    # 8 lineages, pop size of 1
    coal = pg.Coalescent(n=8)

    # tree height PDF, SFS and 2-SFS
    pdf = coal.tree_height.pdf
    sfs = coal.sfs.mean
    sfs2 = coal.sfs.corr

    _plot_overview(coal)


def test_two_epoch_example():
    """
    Figure 2: a bottleneck at time 1.
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

    _plot_overview(coal)


def test_migration_example():
    """
    Figure 3: two demes with migration, and the SFS of each deme.
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

    _plot_overview(coal)

    pg.Spectra(dict(total=coal.sfs.mean, pop0=coal.sfs.demes['pop0'].mean, pop1=coal.sfs.demes['pop1'].mean)).plot(
        show=False, title='Expected SFS'
    )


def test_mmc_example():
    """
    Figure 4: the Beta coalescent.
    """
    # beta coalescent, pop size of 1
    coal = pg.Coalescent(
        n=8,  # 8 lineages
        model=pg.BetaCoalescent(1.4)
    )

    _plot_overview(coal)


def test_recombination_example():
    """
    Figure 5: covariance and correlation of the tree heights of two loci, and the tree height density.
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

    def get_coal(r, N1):
        return pg.Coalescent(
            n={'pop0': 2},
            loci=pg.LocusConfig(n=2, recombination_rate=r / 2),
            demography=pg.Demography(pop_sizes={'pop0': {0: 1, 1: N1}})
        )

    for N1 in [0.1, 10]:
        get_coal(10, N1).tree_height.pdf(np.linspace(0, 3, 20))

        for r in [0.1, 10]:
            get_cov(r, N1)
            get_coal(r, N1).tree_height.loci.corr[0, 1]


def test_inference_example():
    """
    Figure 6 and the inference result of Figure 7: a 2-epoch inference with bootstraps, and its plots.
    """
    inf = pg.Inference(
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
        pbar=False
    )

    inf.run()

    spectra = pg.Spectra.from_spectra(dict(
        fitted=inf.dist_inferred.sfs.mean /
               (inf.dist_inferred.sfs.mean.theta * inf.dist_inferred.sfs.mean.n_sites) *
               (inf.observation.theta * inf.observation.n_sites),
        observed=inf.observation
    ))

    _, axs = plt.subplots(2, 2)

    spectra.plot(ax=axs[0, 0], show=False, title='SFS comparison')
    inf.plot_pop_sizes(ax=axs[0, 1], show=False)
    inf.plot_bootstraps(ax=axs[1], show=False, kwargs={'bins': 30}, title=['Marginal distribution'] * 2)


def test_sfs_correlation_inference_example():
    """
    Figure 8: inference of a Beta coalescent from the SFS and the correlation of two of its bins.
    """

    def get_corr(coal: pg.Coalescent) -> float:
        return coal.sfs.get_corr(4, 5)

    dist_inferred = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}}))
    sfs = dist_inferred.sfs.mean
    tree_height = dist_inferred.tree_height.mean
    m2 = get_corr(dist_inferred)

    def loss(coal: pg.Coalescent, obs: (pg.SFS, float)) -> float:
        sfs, m2 = obs

        loss_sfs = pg.MultinomialLikelihood().compute(
            observed=sfs.normalize().polymorphic,
            modelled=coal.sfs.mean.normalize().polymorphic
        )

        loss_m2 = np.abs(
            m2 - get_corr(coal)
        )

        return loss_sfs + loss_m2

    inf_m12 = pg.Inference(
        coal=lambda t, Ne, alpha: pg.Coalescent(
            n=6,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1, t: Ne}}
            ),
            model=pg.BetaCoalescent(alpha=alpha, scale_time=False)
        ),
        observation=(sfs, m2),
        loss=loss,
        bounds=dict(t=(0, tree_height), Ne=(0.01, 10), alpha=(1.00001, 1.99999)),
        resample=lambda obs, _: (obs[0].resample(), obs[1]),
        do_bootstrap=True,
        n_bootstraps=2,
        n_runs=1,
        parallelize=False,
        pbar=False
    )

    inf_m12.run()


def test_complex_example(caplog):
    """
    Figures S5 to S7: a two-deme Beta coalescent with a bottleneck and changing migration, its plots and statistics.
    """
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

    _, axs = plt.subplots(2, 2, figsize=(6, 5))
    t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

    coal.demography.plot(ax=axs[0, 0], show=False, t=t)
    axs[0, 0].legend(prop={'size': 6}, loc='center left')
    coal.tree_height.plot_pdf(ax=axs[0, 1], show=False)
    assert any('plot_pdf() is deprecated' in r.getMessage() for r in caplog.records)
    coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='SFS')
    coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='2-SFS')

    plt.tight_layout()

    coal.tree_height.mean
    coal.total_branch_length.var
    coal.tree_height.cdf(t)
    coal.sfs.cov
    coal.sfs.demes['pop0'].mean
    coal.moment(3, (pg.TotalBranchLengthReward(),) * 3)


def test_computation_examples():
    """
    Appendix figures S1 to S4: the SFS moments of the one-epoch and two-epoch computation examples.
    """
    coal = pg.Coalescent(n=3, end_time=3)

    coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)])
    coal.tree_height.mean

    coal = pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 2: 0.5}}))
    rewards = [pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)]

    m12 = coal.moment(k=2, rewards=rewards, center=False, permute=False)
    m21 = coal.moment(k=2, rewards=rewards[::-1], center=False, permute=False)
    m1 = coal.moment(k=1, rewards=rewards[:1], center=False, permute=False)
    m2 = coal.moment(k=1, rewards=rewards[1:], center=False, permute=False)

    np.testing.assert_allclose((m12 + m21) / 2 - m1 * m2, coal.moment(k=2, rewards=rewards), rtol=1e-12)

    coal.accumulate(k=2, rewards=rewards, end_times=np.linspace(0.01, 100, 20))
