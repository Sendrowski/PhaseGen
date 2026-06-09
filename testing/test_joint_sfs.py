"""
Test the joint (multi-population) site-frequency spectrum.

Fast, simulation-free invariants (state-space structure, marginal consistency, the ``JointSFS`` container, automatic
reward routing) are plain tests; the msprime comparisons are marked ``slow`` so they can be deselected with
``-m "not slow"``.
"""
import json
from math import prod
from pathlib import Path

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent
from phasegen.state_space import JointBlockCountingStateSpace

# directory holding the independent ``moments`` joint-SFS references (see ``scripts/generate_jsfs_reference.py``)
MOMENTS_REFERENCE_DIR = Path(__file__).resolve().parent.parent / 'results' / 'jsfs_reference'

# configs for which a ``moments`` equilibrium-island reference is generated
MOMENTS_REFERENCE_CONFIGS = [
    '1_epoch_2_pops_n_4_jsfs',
    '1_epoch_2_pops_n_6_jsfs',
    '1_epoch_2_pops_n_6_asym_jsfs',
    '1_epoch_2_pops_n_8_jsfs',
]


@pytest.fixture(scope="module")
def two_pop_coalescent(symmetric_demography):
    """
    A small two-population coalescent reused (built once) across the fast tests.
    """
    return pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.5}, migration_rate=0.75)
    )


# sample configurations used for the simulation-free marginal-consistency check
MARGINAL_CONFIGS = [
    {'pop_0': 2, 'pop_1': 2},
    {'pop_0': 3, 'pop_1': 1},
    {'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
]

# msprime comparison cases: (n, pop_sizes, migration_rate, model, seed)
MSPRIME_CASES = [
    pytest.param({'pop_0': 2, 'pop_1': 2}, {'pop_0': 1.0, 'pop_1': 1.5}, 0.75, None, 42, id="2pop_n4"),
    pytest.param({'pop_0': 3, 'pop_1': 1}, {'pop_0': 1.0, 'pop_1': 0.7}, 1.0, None, 43, id="2pop_n4_asym"),
    pytest.param({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}, {'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0}, 1.0, None, 44,
                 id="3pop_n3"),
    pytest.param({'pop_0': 2, 'pop_1': 1}, {'pop_0': 1.0, 'pop_1': 1.0}, 1.0, pg.BetaCoalescent(alpha=1.5), 45,
                 id="beta_2pop_n3"),
]


def test_state_space_structure(symmetric_demography):
    """
    Test the basic structure of the joint block-counting state space.
    """
    n = {'pop_0': 2, 'pop_1': 1}
    s = JointBlockCountingStateSpace(
        lineage_config=pg.LineageConfig(n),
        epoch=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0}).get_epoch(0)
    )

    # number of block types is prod(n_p + 1) - 1 (excluding the all-zero vector)
    assert len(s.block_configs) == prod(v + 1 for v in n.values()) - 1

    # the initial state is unique and alpha is a probability distribution
    assert s.alpha.sum() == pytest.approx(1.0)
    assert (s.alpha > 0).sum() == 1

    # the intensity matrix has zero row sums (valid generator) and at least one absorbing state
    assert np.allclose(s.S.sum(axis=1), 0)
    assert any(state.is_absorbing() for state in s.states)


def test_multiple_loci_not_implemented(symmetric_demography):
    """
    The joint SFS only supports a single locus; using multiple loci (i.e. recombination) must raise a clear
    ``NotImplementedError`` from every entry point.
    """
    n = {'pop_0': 2, 'pop_1': 2}
    demography = symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})

    # constructing the state space directly with more than one locus
    with pytest.raises(NotImplementedError):
        JointBlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n),
            locus_config=pg.LocusConfig(n=2),
            epoch=demography.get_epoch(0)
        )

    # via the public joint SFS distribution
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=n, demography=demography, loci=2, recombination_rate=1.0).jsfs.mean

    # via automatic reward routing
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=n, demography=demography, loci=2, recombination_rate=1.0).moment(
            k=1, rewards=[pg.JointSFSReward((1, 0))]
        )


def test_mean_is_jointsfs_container(two_pop_coalescent):
    """
    The joint SFS mean is a :class:`~phasegen.spectrum.JointSFS` container of the right shape, with non-negative
    entries and zero monomorphic corners.
    """
    mean = two_pop_coalescent.jsfs.mean

    assert isinstance(mean, pg.JointSFS)
    assert mean.n_pops == 2
    assert mean.shape == (3, 3)
    assert mean[0, 0] == 0
    assert mean[2, 2] == 0
    assert np.all(mean.data >= 0)


def test_jointsfs_container_2d_and_3d(symmetric_demography):
    """
    The joint SFS is a 2-D array for two populations (plottable directly) and a genuine 3-D array for three
    populations (marginalizable to 2-D for plotting).
    """
    # two populations -> 2-D
    jsfs2 = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})
    ).jsfs.mean
    assert jsfs2.n_pops == 2
    jsfs2.plot(show=False)
    jsfs2.plot_surface(show=False)

    # three populations -> 3-D, marginalize to 2-D
    jsfs3 = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0})
    ).jsfs.mean
    assert jsfs3.n_pops == 3
    assert jsfs3.shape == (3, 2, 2)

    marginal = jsfs3.marginalize((0, 1))
    assert marginal.shape == (3, 2)

    # marginalizing onto a single population reproduces the per-deme totals (summing the rest)
    assert np.allclose(marginal.data, jsfs3.data.sum(axis=2))

    jsfs3.plot(pops=(0, 1), show=False)
    jsfs3.plot_surface(pops=(0, 1), show=False)


@pytest.mark.parametrize("n", MARGINAL_CONFIGS)
def test_marginal_consistency_with_single_population_sfs(symmetric_demography, n):
    """
    The pooled joint SFS (summing over all configurations with the same total allele count) must reproduce the
    single-population SFS under the same demography. This holds analytically and requires no simulation.
    """
    coal = pg.Coalescent(
        n=n,
        demography=symmetric_demography({pop: 1.0 + i * 0.3 for i, pop in enumerate(n)})
    )

    jsfs = coal.jsfs.mean
    sfs = coal.sfs.mean.data

    # collapse the joint SFS onto the total allele count
    pooled = np.zeros(coal.lineage_config.n + 1)
    for config in np.ndindex(jsfs.shape):
        pooled[sum(config)] += jsfs[config]

    np.testing.assert_allclose(pooled, sfs, atol=1e-10)


def test_single_population_jsfs_equals_sfs():
    """
    For a single population the joint SFS is just the (1-D) SFS.
    """
    coal = pg.Coalescent(n=5)

    np.testing.assert_allclose(np.asarray(coal.jsfs.mean), coal.sfs.mean.data, atol=1e-10)


def test_moment_accumulate_auto_routing(two_pop_coalescent):
    """
    Passing a :class:`JointSFSReward` to :meth:`Coalescent.moment` or :meth:`Coalescent.accumulate` must
    automatically route to the joint state space and agree with the explicit ``jsfs`` distribution.
    """
    mean = two_pop_coalescent.jsfs.mean

    for config in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
        m = two_pop_coalescent.moment(k=1, rewards=[pg.JointSFSReward(config)])
        assert m == pytest.approx(mean[config], abs=1e-9)

    # accumulate at a large time approaches the bin mean
    acc = two_pop_coalescent.accumulate(k=1, end_times=[100.0], rewards=[pg.JointSFSReward((1, 1))])
    assert float(acc[0]) == pytest.approx(mean[(1, 1)], abs=1e-6)


def test_jsfs_accumulate(two_pop_coalescent):
    """
    ``jsfs.accumulate`` returns the whole spectrum over time, converging to the bin means and agreeing per bin with
    ``Coalescent.accumulate``; ``plot_accumulation`` must also run.
    """
    jsfs = two_pop_coalescent.jsfs
    mean = jsfs.mean

    end_times = [0.5, 2.0, 100.0]
    acc = jsfs.accumulate(k=1, end_times=end_times)

    # shape is the spectrum shape plus a trailing time axis
    assert acc.shape == jsfs.shape + (len(end_times),)

    # accumulation at a large end time converges to the bin means
    np.testing.assert_allclose(acc[..., -1], np.asarray(mean), atol=1e-9)

    # each bin's accumulation matches Coalescent.accumulate for that JointSFSReward
    for config in [(1, 0), (1, 1), (2, 0)]:
        single = two_pop_coalescent.accumulate(k=1, end_times=[2.0], rewards=[pg.JointSFSReward(config)])
        assert acc[config + (1,)] == pytest.approx(float(single[0]), abs=1e-9)

    # centered second-moment accumulation and plotting run without error
    assert jsfs.accumulate(k=2, end_times=[2.0], center=True).shape == jsfs.shape + (1,)
    jsfs.plot_accumulation(k=1, end_times=np.linspace(0, 3, 20), show=False)


def test_jsfs_composes_with_existing_rewards(two_pop_coalescent):
    """
    Weighting a joint SFS bin by :class:`DemeReward` partitions it by deme of residence; summing over demes
    recovers the full bin mean. Also checks the cross-bin covariance (diagonal equals variance, symmetric).
    """
    from phasegen.rewards import CombinedReward

    mean = two_pop_coalescent.jsfs.mean

    for config in [(1, 0), (1, 1), (2, 0)]:
        per_deme = sum(
            two_pop_coalescent.moment(k=1, rewards=[CombinedReward([pg.DemeReward(pop), pg.JointSFSReward(config)])])
            for pop in ['pop_0', 'pop_1']
        )
        assert per_deme == pytest.approx(mean[config], abs=1e-9)

    # the cross-bin covariance matches the variance on its diagonal and is symmetric
    cov = two_pop_coalescent.jsfs.cov
    assert cov.shape == (3, 3, 3, 3)
    assert cov[(1, 0) + (1, 0)] == pytest.approx(two_pop_coalescent.jsfs.var[1, 0], abs=1e-9)
    assert cov[(1, 0) + (0, 1)] == pytest.approx(cov[(0, 1) + (1, 0)], abs=1e-9)


def test_jsfs_incompatible_reward_stacking_raises(two_pop_coalescent):
    """
    Stacking a ``JointSFSReward`` with a reward based on a different, incompatible state space (e.g. a
    single-population SFS reward) must raise a clear error rather than silently misroute.
    """
    from phasegen.rewards import CombinedReward

    for other in [pg.UnfoldedSFSReward(1), pg.LocusReward(0), pg.TotalTreeHeightReward()]:
        with pytest.raises(ValueError):
            two_pop_coalescent.moment(k=1, rewards=[CombinedReward([other, pg.JointSFSReward((1, 0))])])


@pytest.mark.slow
@pytest.mark.parametrize("n, pop_sizes, migration_rate, model, seed", MSPRIME_CASES)
def test_jsfs_mean_matches_msprime(symmetric_demography, n, pop_sizes, migration_rate, model, seed):
    """
    Compare the analytical joint SFS mean against msprime across demographies, initializations and coalescent models.
    """
    demography = symmetric_demography(pop_sizes, migration_rate)
    model_kwargs = {} if model is None else dict(model=model)

    ana = pg.Coalescent(n=n, demography=demography, **model_kwargs).jsfs.mean
    ms = MsprimeCoalescent(
        n=n, demography=demography, num_replicates=100000, n_threads=1, seed=seed, **model_kwargs
    ).jsfs.mean

    np.testing.assert_allclose(np.asarray(ana), np.asarray(ms), atol=0.05, err_msg=f"Mismatch for config {n}")


@pytest.mark.slow
def test_jsfs_second_moment_matches_msprime(symmetric_demography):
    """
    Compare the analytical second (non-central) moment of the joint SFS against msprime.
    """
    n = {'pop_0': 2, 'pop_1': 1}
    demography = symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})

    ana = pg.Coalescent(n=n, demography=demography).jsfs.moment(k=2, center=False)
    ms = MsprimeCoalescent(n=n, demography=demography, num_replicates=100000, n_threads=1, seed=46).jsfs.m2

    np.testing.assert_allclose(np.asarray(ana), np.asarray(ms), rtol=0.05, atol=0.3)


@pytest.mark.slow
@pytest.mark.parametrize("name", MOMENTS_REFERENCE_CONFIGS)
def test_jsfs_matches_moments(name):
    """
    Compare the analytical joint SFS against an independent ``moments`` reference (precomputed via snakemake;
    skipped if absent). Both spectra are normalized, so the comparison probes the spectrum shape.
    """
    reference_file = MOMENTS_REFERENCE_DIR / f'{name}.json'

    if not reference_file.exists():
        pytest.skip(
            f"moments reference {reference_file} not generated; "
            f"run `snakemake results/jsfs_reference/{name}.json`"
        )

    with open(reference_file) as f:
        reference = json.load(f)

    # rebuild the PhaseGen coalescent directly from the (self-contained) reference metadata
    demography = pg.Demography(
        pop_sizes=reference['pop_sizes'],
        migration_rates={(src, dest): rate for src, dest, rate in reference['migration_rates']}
    )
    jsfs = np.asarray(pg.Coalescent(n=reference['n'], demography=demography).jsfs.mean)
    jsfs = jsfs / jsfs.sum()

    # moments is a diffusion approximation, so allow a small absolute tolerance on the normalized spectrum
    np.testing.assert_allclose(jsfs, np.array(reference['jsfs']), atol=0.01, err_msg=f"Mismatch for config {name}")
