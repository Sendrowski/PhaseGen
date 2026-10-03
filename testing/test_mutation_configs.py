"""
Tests for the mutational-configuration probabilities of the spectra (:class:`~phasegen.distributions.MutationConfig`,
:class:`~phasegen.distributions.MutationLayout`): exact identities between layouts, and agreement with the mutation
counts simulated by :class:`~phasegen.distributions.MsprimeCoalescent`.
"""
import itertools
import math
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

#: Two-deme population sizes whose second epoch changes the configuration probabilities beyond the sampling error of
#: a few thousand replicates.
SHARP_TWO_DEME_SIZES = {'pop_0': {0: 0.5, 0.3: 3}, 'pop_1': {0: 2, 0.3: 0.3}}

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
    and descendant vectors that no genealogy carries together have probability zero. The descending generator yields
    a configuration of positive probability first.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(pop_sizes=sizes,
                                                                             migration_rates=MIGRATION))
    jsfs = coal.jsfs
    configs = jsfs._get_configs()
    pooled = jsfs.mutation_layout().rebin([tuple(c for c in configs if sum(c) == i) for i in (1, 2, 3)])

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


@pytest.mark.parametrize('n, folded', [(2, False), (2, True), (3, True)])
def test_one_bin_config_accepts_scalar(n, folded):
    """
    A one-bin configuration is accepted as a single count, which is how R passes a configuration of length one, for
    the exact and the simulated spectrum.
    """
    coal = pg.Coalescent(n=n)
    sfs = coal.fsfs if folded else coal.sfs
    ms = MsprimeCoalescent(n=n, num_replicates=50, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)
    ms_sfs = ms.fsfs if folded else ms.sfs

    for m in (0, 1, 2.0, np.int64(3)):
        assert sfs.get_mutation_config(m, 1.1) == sfs.get_mutation_config((int(m),), 1.1)
        assert ms_sfs.get_mutation_config(m) == ms_sfs.get_mutation_config((int(m),))


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


@pytest.mark.parametrize('sizes', [TWO_DEME_SIZES[0], SHARP_TWO_DEME_SIZES])
def test_joint_and_deme_configs_match_msprime(sizes):
    """
    The joint, folded joint and deme-resolved configuration probabilities of a two-deme sample with migration agree
    with msprime within four standard errors, over one and two epochs.
    """
    coal, ms, n = _two_deme_msprime({'pop_0': 2, 'pop_1': 2}, sizes, 0.5, 4000, False, 3)
    _check_two_deme(coal, ms, 0.5, n, 6)


@pytest.mark.parametrize('dem', ONE_DEME)
def test_two_locus_configs_match_msprime(dem):
    """
    The two-locus and folded two-locus configuration probabilities agree with msprime within four standard errors,
    over one and two epochs.
    """
    _check_two_locus(dem, 0.5, 4000, False, 4, 6)


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


def test_empirical_mutation_config_lookup():
    """
    The simulated frequency of a configuration no replicate shows is 0, fresh, after persisting and after a
    jsonpickle round trip, a malformed configuration raises, and lookups leave the stored frequencies unchanged.
    """
    ms = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)
    sfs = ms.sfs
    unsampled = (50, 0, 0)

    assert sfs.get_mutation_config(unsampled) == 0

    for config in [(1, 2), (-1, 0, 0), (1, 0, 0, 0)]:
        with pytest.raises(ValueError):
            sfs.get_mutation_config(config)

    configs = sfs.mutation_configs
    assert unsampled not in configs and sum(configs.values()) == pytest.approx(1)

    key = next(iter(configs))
    for config in (key, tuple(key), list(key), np.array(key)):
        assert sfs.get_mutation_config(config) == configs[key]

    ms._touch()
    ms._drop()

    for dist in (sfs, jsonpickle.decode(jsonpickle.encode(sfs, keys=True), keys=True)):
        assert dist.get_mutation_config(unsampled) == 0
        assert sum(p for _, p in itertools.islice(dist.get_mutation_configs(), 3000)) == pytest.approx(1)
        assert len(dist.mutation_configs) == len(configs)


@pytest.mark.parametrize('dem', ONE_DEME)
@pytest.mark.parametrize('folded', [False, True])
def test_descending_generator_climbs_from_empty_configuration(dem, folded, monkeypatch):
    """
    The descending generator climbs from the empty configuration and yields the most probable configuration first.
    """
    coal = pg.Coalescent(n=4, demography=dem)
    sfs = coal.fsfs if folded else coal.sfs
    get = type(sfs).get_mutation_config
    calls = []

    def spy(self, config, theta):
        calls.append(tuple(config))
        return get(self, config, theta)

    monkeypatch.setattr(type(sfs), 'get_mutation_config', spy)
    config, p = next(sfs.get_mutation_configs(theta=1.0))

    assert sum(calls[0]) == 0
    assert p == pytest.approx(max(q for _, q in itertools.islice(sfs.get_mutation_configs(1.0, order='count'), 200)),
                              rel=1e-12)


@pytest.mark.parametrize('theta', [np.inf, np.nan, -1])
@pytest.mark.parametrize('name', ['sfs', 'fsfs', 'jsfs'])
def test_descending_generator_rejects_invalid_theta(theta, name):
    """The descending generator raises ValueError for a negative or non-finite theta."""
    dist = getattr(pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes=TWO_DEME_SIZES[0], migration_rates=MIGRATION)), name)

    with pytest.raises(ValueError):
        next(dist.get_mutation_configs(theta=theta))


def test_generator_rejects_invalid_order():
    """The generator raises ValueError for an order other than 'probability' and 'count'."""
    with pytest.raises(ValueError):
        next(pg.Coalescent(n=4).sfs.get_mutation_configs(theta=1, order='mass'))


def test_generator_orders_yield_same_configurations():
    """Both orders yield the same configurations with the same probabilities, the count order by total."""
    sfs = pg.Coalescent(n=4).sfs
    by_count = list(itertools.islice(sfs.get_mutation_configs(theta=1, order='count'), 35))
    by_prob = dict(itertools.islice(sfs.get_mutation_configs(theta=1), 2000))

    assert [c.total for c, _ in by_count] == sorted(c.total for c, _ in by_count)
    for c, p in by_count:
        assert by_prob[c] == pytest.approx(p, rel=1e-12)


@pytest.mark.parametrize('sparse', [False, True])
def test_layout_without_reward_in_stalled_epoch(sparse, monkeypatch):
    """
    A layout that leaves states without reward in a finite epoch in which those states cannot move gives the
    probabilities of a coarser layout summed over the counts of the bins it leaves out, on the dense and the sparse
    path.
    """
    monkeypatch.setattr(pg.Settings, 'closed_form_sparse_min_states', 1 if sparse else 10 ** 9)
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 1}},
                        migration_rates={('pop_0', 'pop_1'): {0: 0, 0.5: 1}, ('pop_1', 'pop_0'): {0: 0, 0.5: 1}})
    sfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).sfs
    default = sfs.mutation_layout()
    sparse = default.rebin([(1,)])
    coarse = default.rebin([(1,), (2, 3)])
    theta, k_max = 0.1, 20

    for m in (2, 1, 0):
        p = sfs.get_mutation_config(sparse.config((m,)), theta)
        ref = sum(sfs.get_mutation_config(coarse.config((m, k)), theta) for k in range(k_max + 1))
        assert np.isfinite(p)
        np.testing.assert_allclose(p, ref, rtol=1e-10)


def test_folded_joint_layout_merges_complements():
    """
    The folded joint layout merges each descendant vector with its complement, and its probabilities are the sums of
    the unfolded probabilities over the unfolded configurations that fold onto the configuration.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(pop_sizes=TWO_DEME_SIZES[1],
                                                                             migration_rates=MIGRATION))
    unfolded = coal.jsfs.mutation_layout()
    folded = coal.jsfs.mutation_layout(folded=True)

    assert sorted(folded.bins) == [((0, 1), (2, 0)), ((1, 0), (1, 1))]

    for m in [(0, 0), (1, 0), (0, 1), (2, 1)]:
        ref = sum(coal.jsfs.get_mutation_config(c, 0.8) for c in unfolded.configs(sum(m))
                  if folded.from_array(c.to_array()) == m)
        np.testing.assert_allclose(coal.jsfs.get_mutation_config(folded.config(m), 0.8), ref, rtol=1e-12)

    even = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(pop_sizes=TWO_DEME_SIZES[0],
                                                                             migration_rates=MIGRATION))

    assert sorted(even.jsfs.mutation_layout(folded=True).bins) == [
        ((0, 1), (2, 1)), ((0, 2), (2, 0)), ((1, 0), (1, 2)), ((1, 1),)
    ]


def test_folded_two_locus_layout_merges_complements():
    """
    The folded two-locus layout merges the classes i and n - i of each locus, and its probabilities are the sums of
    the unfolded probabilities over the unfolded configurations that fold onto the configuration.
    """
    sfs2 = pg.Coalescent(n=4, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).sfs2
    unfolded = sfs2.mutation_layout()
    folded = sfs2.mutation_layout(folded=True)

    assert folded.bins == (((0, 1), (0, 3)), ((0, 2),), ((1, 1), (1, 3)), ((1, 2),))

    for m in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 1, 0), (1, 0, 1, 1)]:
        ref = sum(sfs2.get_mutation_config(c, 0.5) for c in unfolded.configs(sum(m))
                  if folded.from_array(c.to_array()) == m)
        np.testing.assert_allclose(sfs2.get_mutation_config(folded.config(m), 0.5), ref, rtol=1e-12)


def test_layout_rejects_inconsistent_positions_and_shape():
    """A layout raises ValueError for a class without a position within its shape and for an array of another shape."""
    layout = pg.Coalescent(n=4).sfs.mutation_layout()

    for positions in ({1: (1,), 2: (2,)}, {1: (1,), 2: (2,), 3: (5,)}, {1: (1,), 2: (2,), 3: (0, 3)}):
        with pytest.raises(ValueError):
            pg.MutationLayout(layout.bins, positions, layout.shape, layout.axes, layout.lineage_config,
                              layout.locus_config)

    for counts in (np.arange(4), np.arange(10), np.zeros((2, 5))):
        with pytest.raises(ValueError):
            layout.from_array(counts)


@pytest.mark.parametrize('folded', [False, True])
def test_from_array_rejects_entries_that_are_not_counts(folded):
    """
    An array with a negative or fractional entry raises ValueError, also where a merged bin sums it to a valid count
    and at the monomorphic classes no bin reads.
    """
    coal = pg.Coalescent(n=4)
    layout = (coal.fsfs if folded else coal.sfs).mutation_layout()

    for counts in ([0, -1, 0, 2, 0], [0, 0.5, 0, 0.5, 0], [-5, 1, 0, 0, 7], [0, 1, np.nan, 0, 0]):
        with pytest.raises(ValueError):
            layout.from_array(np.array(counts))

    assert layout.from_array(np.array([3, 1, 0, 0, 2.0])) == ((1, 0) if folded else (1, 0, 0))


def test_empirical_mutation_config_rejects_other_layout():
    """
    A configuration of another layout raises ValueError on the simulated spectrum, which stores its frequencies in
    its own layout, rather than reading the frequency of the configuration with the same counts there.
    """
    ms = MsprimeCoalescent(n=4, num_replicates=50, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)
    own = ms.sfs.mutation_layout()
    permuted = own.rebin(own.bins[::-1])

    for config in (permuted.config((0, 0, 2)), ms.fsfs.mutation_layout().config((1, 0))):
        with pytest.raises(ValueError):
            ms.sfs.get_mutation_config(config)

    assert ms.sfs.get_mutation_config(own.config((0, 0, 0))) == ms.sfs.get_mutation_config((0, 0, 0))


@pytest.mark.parametrize('rates', [{0: 0}, {0: 1, 0.5: 0}])
@pytest.mark.parametrize('theta', [0, 0.5])
def test_mutation_configs_raise_on_a_demography_that_does_not_absorb(rates, theta):
    """
    Demes disconnected in the last epoch leave absorption uncertain, so the configuration probabilities raise
    ModelError, as the moments do, rather than returning a defective distribution whose descending iterator never
    accumulates the requested mass.
    """
    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): rates, ('pop_1', 'pop_0'): rates})
    sfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).sfs

    with pytest.raises(pg.ModelError):
        sfs.get_mutation_config((0, 0, 0), theta)

    with pytest.raises(pg.ModelError):
        next(sfs.get_mutation_configs(theta))

    with pytest.raises(pg.ModelError):
        next(sfs.get_mutation_configs(theta, order='count'))


def test_multi_epoch_mutation_configs_follow_theta():
    """
    The multi-epoch configuration probabilities are cached per layout and theta, so a second theta on the same
    distribution must give the probabilities of a fresh coalescent, not those cached for the first.
    """
    dem = ONE_DEME[1]
    sfs = pg.Coalescent(n=4, demography=dem).sfs

    for config in ((1, 0, 0), (2, 1, 0)):
        sfs.get_mutation_config(config, 0.5)
        np.testing.assert_allclose(sfs.get_mutation_config(config, 2.0),
                                   pg.Coalescent(n=4, demography=dem).sfs.get_mutation_config(config, 2.0), rtol=1e-14)


@pytest.mark.parametrize('sparse', [False, True])
def test_multi_epoch_expm_action_matches_dense_exponential(sparse, monkeypatch):
    """
    The multi-epoch configuration probabilities propagated by the matrix-exponential action agree with those of the
    dense exponential, on the dense and the sparse generator.
    """
    monkeypatch.setattr(pg.Settings, 'closed_form_sparse_min_states', 1 if sparse else 10 ** 9)
    configs = ((0, 0, 0), (1, 0, 0), (2, 1, 0), (0, 1, 2))

    monkeypatch.setattr(pg.Settings, 'expm_action_min_dim', 10 ** 9)
    dense = [pg.Coalescent(n=4, demography=ONE_DEME[1]).sfs.get_mutation_config(c, 1.3) for c in configs]

    monkeypatch.setattr(pg.Settings, 'expm_action_min_dim', 1)
    action = [pg.Coalescent(n=4, demography=ONE_DEME[1]).sfs.get_mutation_config(c, 1.3) for c in configs]

    np.testing.assert_allclose(action, dense, rtol=1e-12)


def test_rebin_rejects_unknown_class():
    """Rebinning raises ValueError for a class label that is not one of the layout."""
    with pytest.raises(ValueError):
        pg.Coalescent(n=4).sfs.mutation_layout().rebin([(1,), (4,)])


def test_layout_records_lineages_and_loci():
    """The layouts of every spectrum record the lineages and loci of the coalescent, also of a mixture."""
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes=TWO_DEME_SIZES[0], migration_rates=MIGRATION))

    for layout in (coal.sfs.mutation_layout(), coal.fsfs.mutation_layout(demes=True), coal.jsfs.mutation_layout()):
        assert layout.lineage_config == coal.lineage_config
        assert layout.locus_config.n == 1
        assert "n={'pop_0': 2, 'pop_1': 1}, loci=1" in repr(layout)

    assert repr(coal.sfs.mutation_layout().rebin([(1, 2)]).config([3])) == \
           "MutationConfig({(1, 2): 3}, n={'pop_0': 2, 'pop_1': 1}, loci=1)"

    two = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1)).sfs2.mutation_layout()
    assert two.locus_config.n == 2 and two.lineage_config.n == 3

    mixture = pg.InitialDistribution([(1, {'pop_0': 2, 'pop_1': 1}), (3, {'pop_0': 1, 'pop_1': 2})])
    layout = pg.Coalescent(n=mixture, demography=pg.Demography(
        pop_sizes=TWO_DEME_SIZES[0], migration_rates=MIGRATION)).sfs.mutation_layout()
    assert layout.lineage_config == mixture
    assert repr(layout).startswith("MutationLayout(n=[(0.25, {'pop_0': 2, 'pop_1': 1}), (0.75,")


def test_foreign_layouts_are_rejected():
    """A configuration or layout of another spectrum raises ValueError, while those of the spectrum itself pass."""
    s4, s5 = pg.Coalescent(n=4).sfs, pg.Coalescent(n=5).sfs
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes=TWO_DEME_SIZES[0], migration_rates=MIGRATION)).jsfs

    with pytest.raises(ValueError):
        s4.get_mutation_config(s5.mutation_layout().config([0, 0, 0, 1]), theta=1)

    with pytest.raises(ValueError):
        next(s4.get_mutation_configs(theta=1, layout=s5.mutation_layout().rebin([(1,), (2,)])))

    with pytest.raises(ValueError):
        jsfs.get_mutation_config(s4.mutation_layout().config([1, 0, 0]), theta=1)

    assert s4.mutation_layout().rebin([(1,), (2,)]) != s5.mutation_layout().rebin([(1,), (2,)])
    assert s4.get_mutation_config(s4.mutation_layout().rebin([(1,), (2, 3)]).config([1, 0]), theta=1) > 0


@pytest.mark.parametrize('name', ['sfs', 'fsfs', 'jsfs', 'sfs2'])
def test_count_order_yields_every_configuration_once(name):
    """The count order yields every configuration with at most three mutations exactly once, by ascending total."""
    if name == 'sfs2':
        dist = pg.Coalescent(n=3, loci=2, recombination_rate=1.0).sfs2
    else:
        dist = getattr(pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
            pop_sizes=TWO_DEME_SIZES[0], migration_rates=MIGRATION)), name)

    J = len(dist.mutation_layout())
    expected = math.comb(3 + J, J)  # configurations of J bins with total at most 3
    configs = [c for c, _ in itertools.islice(dist.get_mutation_configs(theta=0.5, order='count'), expected)]

    assert len(set(configs)) == expected
    assert all(sum(c) <= 3 for c in configs)
    assert [sum(c) for c in configs] == sorted(sum(c) for c in configs)


@pytest.mark.parametrize('name', ['sfs', 'fsfs', 'jsfs', 'sfs2'])
def test_zero_theta_puts_all_mass_on_the_empty_configuration(name):
    """
    Without mutation, every spectrum over two epochs gives probability 1 to the empty configuration and 0 to any
    other, and the descending iterator yields the empty configuration alone with the full mass.
    """
    if name == 'sfs2':
        dist = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0), demography=ONE_DEME[1]).sfs2
    else:
        dist = getattr(pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
            pop_sizes=TWO_DEME_SIZES[1], migration_rates=MIGRATION)), name)

    layout = dist.mutation_layout()
    J = len(layout)

    assert dist.get_mutation_config(layout.config((0,) * J), theta=0) == pytest.approx(1, abs=1e-14)
    assert dist.get_mutation_config(layout.config((1,) + (0,) * (J - 1)), theta=0) == pytest.approx(0, abs=1e-14)
    assert list(dist.get_mutation_configs(theta=0)) == [(layout.config((0,) * J), 1.0)]
    assert dist.generated_mass == 1


@pytest.mark.parametrize('dem', ONE_DEME)
def test_two_locus_without_recombination_thins_one_genealogy(dem):
    """
    Without recombination both loci share one genealogy, so the mutations of each bin are those of the single-locus
    spectrum at twice the mutation rate, each placed on either locus with probability one half:
    ``P(m0, m1; theta) = P(m0 + m1; 2 theta) * prod_j C(m0_j + m1_j, m0_j) / 2^(m0_j + m1_j)``.
    """
    sfs = pg.Coalescent(n=3, demography=dem).sfs
    sfs2 = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=0), demography=dem).sfs2
    layout = sfs2.mutation_layout()

    for m0, m1 in itertools.product([(0, 0), (1, 0), (1, 1), (2, 0)], [(0, 0), (0, 1), (1, 1)]):
        m = tuple(a + b for a, b in zip(m0, m1))
        ref = sfs.get_mutation_config(m, 1.0) * np.prod([math.comb(k, a) / 2 ** k for k, a in zip(m, m0)])
        np.testing.assert_allclose(sfs2.get_mutation_config(layout.config(m0 + m1), 0.5), ref, rtol=1e-12)


@pytest.mark.parametrize('dem', ONE_DEME)
def test_two_locus_with_free_recombination_factorises(dem):
    """
    As the recombination rate grows the loci become independent, and the two-locus probabilities approach the
    product of the single-locus ones, with an error that falls like the inverse recombination rate (about 4e-5 at
    ``r = 1e4``).
    """
    sfs = pg.Coalescent(n=3, demography=dem).sfs
    sfs2 = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1e4), demography=dem).sfs2
    layout = sfs2.mutation_layout()

    for m0, m1 in [((0, 0), (0, 0)), ((1, 0), (0, 1)), ((2, 1), (1, 0))]:
        ref = sfs.get_mutation_config(m0, 0.5) * sfs.get_mutation_config(m1, 0.5)
        np.testing.assert_allclose(sfs2.get_mutation_config(layout.config(m0 + m1), 0.5), ref, rtol=2e-4)
