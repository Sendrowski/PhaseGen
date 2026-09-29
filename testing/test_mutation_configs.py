"""
Tests for the mutational-configuration probabilities of the spectra (:class:`~phasegen.distributions.MutationConfig`,
:class:`~phasegen.distributions.MutationLayout`): exact identities between layouts, and agreement with the mutation
counts simulated by :class:`~phasegen.distributions.MsprimeCoalescent`.
"""
import itertools
import pickle

import jsonpickle
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions.empirical import MsprimeCoalescent

#: Migration rates of the two-deme scenarios.
MIGRATION = {('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 0.2}

#: Population sizes of the two-deme scenarios with one and with two epochs.
TWO_DEME_SIZES = ({'pop_0': {0: 0.5}, 'pop_1': {0: 2}}, {'pop_0': {0: 0.5, 0.4: 1.5}, 'pop_1': {0: 2}})

#: Single-deme demographies with one and with two epochs.
ONE_DEME = (None, pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}}))

#: Number of standard errors of the simulated frequency a configuration probability may deviate by.
N_SE = 4


def _frequencies(layout: pg.MutationLayout, counts: np.ndarray) -> dict:
    """
    Relative frequency of each configuration among replicates.

    :param layout: The layout.
    :param counts: Per-replicate spectrum-shaped counts, of shape ``(N,) + layout.shape``.
    :return: Dictionary from configuration to relative frequency.
    """
    binned = np.stack([sum(counts[(slice(None),) + layout.positions[label]] for label in b) for b in layout.bins], 1)
    rows, n = np.unique(binned, axis=0, return_counts=True)

    return {layout.config(r): c / len(binned) for r, c in zip(rows, n)}


def _assert_matches(dist, layout: pg.MutationLayout, freqs: dict, theta: float, n: int, n_top: int) -> None:
    """
    Assert that the ``n_top`` most probable configurations have probabilities within :data:`N_SE` standard errors of
    their simulated frequencies.

    :param dist: The spectrum.
    :param layout: The layout.
    :param freqs: Simulated frequencies.
    :param theta: The mutation rate.
    :param n: The number of replicates.
    :param n_top: The number of configurations to check.
    """
    for config, p in itertools.islice(dist.get_mutation_configs(theta, layout=layout), n_top):
        se = np.sqrt(p * (1 - p) / n)
        assert abs(freqs.get(config, 0) - p) < N_SE * se, (layout, config, p, freqs.get(config, 0), se)


@pytest.mark.parametrize('sizes', TWO_DEME_SIZES)
def test_joint_pooled_equals_sfs(sizes):
    """
    Merging the descendant vectors of the joint spectrum by their total gives the pooled configuration probabilities,
    and descendant vectors that no genealogy carries together have probability zero. The descending generator starts
    at a configuration of positive probability, where the rounded mean branch lengths give a structural zero.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(pop_sizes=sizes,
                                                                             migration_rates=MIGRATION))
    jsfs = coal.jsfs
    configs = jsfs._get_configs()
    pooled = pg.MutationLayout([tuple(c for c in configs if sum(c) == i) for i in (1, 2, 3)],
                               {c: c for c in configs}, jsfs.shape, ('pop_0', 'pop_1'))

    for m in [(0, 0, 0), (1, 0, 0), (2, 1, 0), (0, 1, 1)]:
        np.testing.assert_allclose(jsfs.get_mutation_config(pooled.config(m), 0.7),
                                   coal.sfs.get_mutation_config(m, 0.7), rtol=1e-12)

    layout = jsfs.mutation_layout()
    incompatible = layout.config([1 if b[0] in ((2, 1), (1, 2)) else 0 for b in layout.bins])
    assert jsfs.get_mutation_config(incompatible, 0.7) == 0

    config, p = next(jsfs.get_mutation_configs(3.0))
    assert p > 0
    assert config.layout == layout


@pytest.mark.parametrize('dem', ONE_DEME)
def test_two_locus_locus_marginal_equals_sfs(dem):
    """
    The configuration probabilities of one locus of the two-locus spectrum equal those of the single-locus spectrum.
    """
    sfs = pg.Coalescent(n=3, demography=dem).sfs
    sfs2 = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0), demography=dem).sfs2

    for locus in (0, 1):
        layout = sfs2.mutation_layout(loci=(locus,))
        for m in [(0, 0), (1, 0), (2, 1)]:
            np.testing.assert_allclose(sfs2.get_mutation_config(layout.config(m), 0.5),
                                       sfs.get_mutation_config(m, 0.5), rtol=1e-12)


@pytest.mark.parametrize('loci', [(), (0, 0), (2,)])
def test_two_locus_layout_rejects_invalid_loci(loci):
    """The loci of a two-locus layout are distinct entries of (0, 1)."""
    with pytest.raises(ValueError):
        pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).sfs2.mutation_layout(loci=loci)


@pytest.mark.parametrize('n', [4, 5])
@pytest.mark.parametrize('dem', ONE_DEME)
def test_folded_spectrum_uses_folded_layout(n, dem):
    """
    The folded spectrum's layout is the folded layout of the unfolded spectrum, and both give the same
    probabilities, which are the sums of the unfolded probabilities over the unfolded configurations that fold onto
    the configuration.
    """
    coal = pg.Coalescent(n=n, demography=dem)
    layout = coal.sfs.mutation_layout(folded=True)

    assert coal.fsfs.mutation_layout() == layout

    for m in [(0, 0), (1, 0), (0, 1), (2, 1)]:
        p = coal.fsfs.get_mutation_config(m, 0.8)
        np.testing.assert_allclose(coal.sfs.get_mutation_config(layout.config(m), 0.8), p, rtol=1e-14)

        unfolded = coal.sfs.mutation_layout()
        ref = sum(coal.sfs.get_mutation_config(c, 0.8) for c in unfolded.configs(sum(m))
                  if layout.from_array(c.to_array()) == m)
        np.testing.assert_allclose(p, ref, rtol=1e-12)


@pytest.mark.parametrize('sizes', TWO_DEME_SIZES)
def test_deme_layout_sums_to_pooled(sizes):
    """
    Summing the configurations resolved by the deme in which the mutation occurs over the splits of each count
    gives the pooled configuration probability.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(pop_sizes=sizes,
                                                                             migration_rates=MIGRATION))
    layout = coal.sfs.mutation_layout(demes=True)

    assert len(layout) == 4

    for m in [(0, 0), (1, 0), (1, 2), (2, 1)]:
        ref = coal.sfs.get_mutation_config(m, 0.6)
        p = sum(coal.sfs.get_mutation_config(layout.config((a, b, m[0] - a, m[1] - b)), 0.6)
                for a in range(m[0] + 1) for b in range(m[1] + 1))
        np.testing.assert_allclose(p, ref, rtol=1e-12)


def test_single_deme_layout_equals_default():
    """With one deme, the deme-resolved layout gives the probabilities of the default layout."""
    sfs = pg.Coalescent(n=4).sfs
    layout = sfs.mutation_layout(demes=True)

    for m in [(0, 0, 0), (2, 1, 0), (1, 1, 1)]:
        assert sfs.get_mutation_config(layout.config(m), 1.1) == pytest.approx(sfs.get_mutation_config(m, 1.1),
                                                                                rel=1e-14)


def test_mutation_config_object():
    """
    A configuration compares and hashes equal to the tuple of its counts, exposes the counts by bin and as a
    spectrum-shaped array, and survives pickling with its layout.
    """
    sfs = pg.Coalescent(n=4).sfs
    c, p = next(sfs.get_mutation_configs(theta=1))

    assert isinstance(c, pg.MutationConfig)
    assert c == tuple(c) and hash(c) == hash(tuple(c))
    assert sfs.get_mutation_config(tuple(c), 1) == p
    np.testing.assert_array_equal(c.to_array()[1:4], list(c))
    assert c.count_of(2) == c[1]
    assert c.total == sum(c)

    restored = pickle.loads(pickle.dumps(c))
    assert restored == c and restored.layout == c.layout

    with pytest.raises(ValueError):
        c.layout.config([1, 0])

    with pytest.raises(ValueError):
        c.layout.config([1, -1, 0])


def test_msprime_mutation_configs_keyed_by_mutation_config():
    """
    The simulated configuration frequencies are keyed by configurations of the spectrum's default layout, which
    look up equally by plain tuple, and survive a jsonpickle round trip.
    """
    ms = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)

    for sfs, layout in ((ms.sfs, pg.Coalescent(n=4).sfs.mutation_layout()),
                        (ms.fsfs, pg.Coalescent(n=4).fsfs.mutation_layout())):
        configs = dict(sfs.mutation_configs)
        key = next(iter(configs))

        assert isinstance(key, pg.MutationConfig) and key.layout == layout
        assert sfs.get_mutation_config(tuple(key)) == configs[key]
        assert next(sfs.get_mutation_configs())[0].layout == layout

        restored = jsonpickle.decode(jsonpickle.encode(sfs, keys=True), keys=True)
        assert restored.get_mutation_config(key) == pytest.approx(configs[key])


def _two_deme_msprime(n: dict, sizes: dict, theta: float, num_replicates: int, parallelize: bool, seed: int):
    """
    The exact and the simulated two-deme coalescent.

    :return: The coalescent, the msprime coalescent and its number of replicates.
    """
    dem = pg.Demography(pop_sizes=sizes, migration_rates=MIGRATION)
    ms = MsprimeCoalescent(n=n, demography=dem, num_replicates=num_replicates, n_threads=1 if not parallelize else 16,
                           parallelize=parallelize, mutation_rate=theta, simulate_mutations=True,
                           record_migration=True, seed=seed)
    ms.simulate()

    return pg.Coalescent(n=n, demography=dem), ms, ms.n_total


def _check_two_deme(coal, ms, theta: float, n: int, n_top: int) -> None:
    """Check the joint, folded joint and deme-resolved layouts against the simulated counts."""
    for layout in (coal.jsfs.mutation_layout(), coal.jsfs.mutation_layout(folded=True)):
        _assert_matches(coal.jsfs, layout, _frequencies(layout, ms.jsfs_mutations), theta, n, n_top)

    layout = coal.sfs.mutation_layout(demes=True)
    counts = np.moveaxis(ms.deme_mutations.sum(axis=0), 1, 0)
    _assert_matches(coal.sfs, layout, _frequencies(layout, counts), theta, n, n_top)


def _check_two_locus(dem, theta: float, num_replicates: int, parallelize: bool, seed: int, n_top: int) -> None:
    """Check the two-locus and folded two-locus layouts against the simulated counts."""
    loci = pg.LocusConfig(n=2, recombination_rate=1.0)
    ms = MsprimeCoalescent(n=3, loci=loci, demography=dem, num_replicates=num_replicates,
                           n_threads=1 if not parallelize else 16, parallelize=parallelize, mutation_rate=theta,
                           simulate_mutations=True, seed=seed)
    ms.simulate()
    sfs2 = pg.Coalescent(n=3, loci=loci, demography=dem).sfs2
    counts = np.moveaxis(ms.mutations[:, 0], 1, 0)

    for layout in (sfs2.mutation_layout(), sfs2.mutation_layout(folded=True)):
        _assert_matches(sfs2, layout, _frequencies(layout, counts), theta, ms.n_total, n_top)


def test_joint_and_deme_configs_match_msprime():
    """
    The joint, folded joint and deme-resolved configuration probabilities of a two-deme sample agree with msprime
    within four standard errors.
    """
    coal, ms, n = _two_deme_msprime({'pop_0': 2, 'pop_1': 2}, TWO_DEME_SIZES[0], 0.5, 4000, False, 3)
    _check_two_deme(coal, ms, 0.5, n, 6)


def test_two_locus_configs_match_msprime():
    """The two-locus and folded two-locus configuration probabilities agree with msprime within four standard errors."""
    _check_two_locus(None, 0.5, 4000, False, 4, 6)


@pytest.mark.slow
@pytest.mark.parametrize('theta', [0.5, 1.0])
@pytest.mark.parametrize('epochs', [0, 1])
def test_joint_and_deme_configs_match_msprime_slow(theta, epochs):
    """
    The joint, folded joint and deme-resolved configuration probabilities of three samples per deme agree with
    msprime within four standard errors, over one and two epochs.
    """
    coal, ms, n = _two_deme_msprime({'pop_0': 3, 'pop_1': 3}, TWO_DEME_SIZES[epochs], theta, 128000, True, 5)
    _check_two_deme(coal, ms, theta, n, 10)


@pytest.mark.slow
@pytest.mark.parametrize('theta', [0.5, 1.0])
@pytest.mark.parametrize('dem', ONE_DEME)
def test_two_locus_configs_match_msprime_slow(theta, dem):
    """
    The two-locus and folded two-locus configuration probabilities agree with msprime within four standard errors,
    over one and two epochs.
    """
    _check_two_locus(dem, theta, 128000, True, 6, 10)
