"""
Tests for the marginal containers and argument validation of the empirical distributions of
:class:`~phasegen.distributions.MsprimeCoalescent`.
"""
import itertools

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions.empirical import MsprimeCoalescent

#: Two-deme demography with migration.
DEMOGRAPHY = pg.Demography(pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2}},
                           migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 0.5})


@pytest.mark.parametrize('kwargs, attr', [
    (dict(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY, record_migration=True), 'demes'),
    (dict(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)), 'loci'),
])
def test_marginals_get_cov_and_get_corr(kwargs, attr):
    """
    The per-deme and per-locus marginals of the scalar and spectrum statistics give the entries of their covariance
    and correlation matrices by key, the variances of the marginals on the diagonal, and raise ValueError for an
    unknown key.
    """
    ms = MsprimeCoalescent(num_replicates=500, n_threads=1, parallelize=False, seed=1, **kwargs)

    for dist in (ms.total_branch_length, ms.sfs, ms.fsfs):
        marginals = getattr(dist, attr)
        keys = list(marginals)

        for i, a in enumerate(keys):
            var = marginals[a].var
            np.testing.assert_allclose(marginals.get_cov(a, a), getattr(var, 'data', var), rtol=1e-10, atol=1e-12)

            for j, b in enumerate(keys):
                np.testing.assert_array_equal(marginals.get_cov(a, b), np.asarray(marginals.cov)[i, j])
                np.testing.assert_array_equal(marginals.get_corr(a, b), np.asarray(marginals.corr)[i, j])

        with pytest.raises(ValueError):
            marginals.get_cov('unknown', keys[0])


def test_jsfs_of_mixture_with_differing_configurations_raises_before_simulating():
    """
    The joint SFS of an initial distribution whose lineage configurations differ raises ValueError without
    simulating.
    """
    ms = MsprimeCoalescent(n=pg.InitialDistribution([(1, [2, 1]), (1, [1, 2])]), demography=DEMOGRAPHY,
                           num_replicates=10, n_threads=1, parallelize=False, seed=1)

    with pytest.raises(ValueError, match='single lineage configuration'):
        _ = ms.jsfs

    assert ms.heights is None


def test_two_locus_sfs_without_simulated_mutations_has_no_mutation_configs():
    """
    ``MsprimeCoalescent.sfs2`` without simulated mutations passed all-zero counts, so the empty configuration had
    frequency one. Its configuration accessors raise as those of ``sfs`` and ``fsfs`` do.
    """
    ms = MsprimeCoalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0), num_replicates=50, n_threads=1,
                           parallelize=False, seed=1)

    with pytest.raises(ValueError, match="no mutation counts"):
        ms.sfs2.get_mutation_config([0, 0, 0, 0])


def test_empirical_moments_validate_the_order():
    """
    The empirical moments validate their order as the exact ones do: a negative or fractional order raises
    ValueError, and an integral float is accepted, also by the joint spectrum, which raised TypeError for it.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=DEMOGRAPHY)
    dists = [coal.tree_height.to_empirical(200, seed=1), coal.jsfs.to_empirical(200, seed=1),
             pg.Coalescent(n=3, loci=2, recombination_rate=1.0).sfs2.to_empirical(200, seed=1),
             pg.Coalescent(n=4).sfs.to_empirical(200, seed=1),
             pg.distributions.EmpiricalDistribution(np.linspace(0.1, 2, 20)),
             coal.sfs.to_empirical(200, seed=1).demes['pop_0']]

    for dist in dists:
        for k in (-1, 1.5):
            with pytest.raises(ValueError, match="order k"):
                dist.moment(k)

        np.testing.assert_array_equal(np.asarray(dist.moment(2.0)), np.asarray(dist.moment(2)))


def test_empirical_sfs_moments_and_marginal_covariances_are_spectra():
    """
    The moments of every order and the per-deme and per-locus covariances and correlations of the empirical spectra
    are spectra, as on the exact spectra, with the values of the sample moments.
    """
    ms = MsprimeCoalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY, record_migration=True,
                           num_replicates=200, n_threads=1, parallelize=False, seed=1)
    sampled = pg.Coalescent(n=4).sfs.to_empirical(200, seed=1)

    for sfs in (ms.sfs, ms.fsfs, sampled, ms.sfs.demes['pop_0']):
        plain = pg.distributions.EmpiricalDistribution(sfs.samples)

        for value, want in [(sfs.moment(3), plain.moment(3)), (sfs.moment(2, center=False), plain.m2),
                            (sfs.m3, plain.m3), (sfs.m4, plain.m4)]:
            assert isinstance(value, pg.SFS)
            np.testing.assert_array_equal(value.data, want)

    for marginals in (ms.sfs.demes, ms.sfs.loci):
        keys = list(marginals)
        for a in keys:
            for b in keys:
                assert isinstance(marginals.get_cov(a, b), pg.SFS)
                assert isinstance(marginals.get_corr(a, b), pg.SFS)

    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=DEMOGRAPHY).jsfs.to_empirical(200, seed=1)
    assert isinstance(jsfs.loci.get_cov(0, 0), pg.JointSFS)
    assert isinstance(pg.Coalescent(n=3).total_branch_length.to_empirical(200, seed=1).loci.get_cov(0, 0), float)


def test_coalescent_level_methods_raise_after_drop():
    """
    After ``_drop`` the coalescent-level statistics of ``MsprimeCoalescent`` and ``SampledCoalescent`` raised
    AttributeError on the dropped demography or coalescent. They raise the NotImplementedError of the
    distribution-level accumulation.
    """
    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 1,
                                                                               ('pop_1', 'pop_0'): 1})
    ms = MsprimeCoalescent(n={'pop_0': 2, 'pop_1': 1}, demography=dem, num_replicates=50, n_threads=1,
                           parallelize=False, seed=1)
    sampled = pg.Coalescent(n=3).to_empirical(n_samples=50, seed=1)

    for coal in (ms, sampled):
        coal._touch()
        coal._drop()

        calls = [lambda: coal.moment(1), lambda: coal.accumulate(1, [0.5]),
                 lambda: coal.joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward()),
                 lambda: coal.plot_accumulation(end_times=[0.5], show=False)]

        if coal is ms:
            calls += [lambda: ms.fst, lambda: ms.f2('pop_0', 'pop_1')]

        for call in calls:
            with pytest.raises(NotImplementedError, match="MsprimeCoalescent"):
                call()

        with pytest.raises(NotImplementedError, match="MsprimeCoalescent"):
            coal.tree_height.accumulate(1, [0.5])


@pytest.fixture
def two_deme_mutations():
    """A two-deme msprime simulation with mutations and migration recording."""
    ms = MsprimeCoalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY, num_replicates=300, n_threads=1,
                           parallelize=False, simulate_mutations=True, mutation_rate=1.0, record_migration=True,
                           seed=2)
    ms.simulate()

    return ms


def _frequency(counts: np.ndarray, layout, config) -> float:
    """The fraction of replicates whose per-class counts, of shape ``(N,) + layout.shape``, show ``config``."""
    return float(np.mean([tuple(layout.from_array(c)) == tuple(config) for c in counts]))


def test_empirical_sfs_serves_folded_rebinned_and_deme_resolved_layouts(two_deme_mutations):
    """
    The empirical spectra look up configurations of the folded, rebinned and deme-resolved layouts of the exact
    spectrum from the counts of the simulated replicates, which the empirical spectrum refused, and the unfolded
    spectrum gives the folded spectrum's frequencies in the folded layout.
    """
    ms = two_deme_mutations
    exact = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY).sfs
    by_deme = np.moveaxis(ms.deme_mutations.sum(axis=0), 1, 0)
    by_class = by_deme.sum(axis=1)

    layouts = [(exact.mutation_layout(), by_class), (exact.mutation_layout(folded=True), by_class),
               (exact.mutation_layout().rebin([(1,), (2, 3)]), by_class),
               (exact.mutation_layout(demes=True), by_deme),
               (exact.mutation_layout(folded=True, demes=True), by_deme)]

    for sfs in (ms.sfs, ms.fsfs):
        for layout, counts in layouts:
            configs = [c for c, _ in itertools.islice(sfs.get_mutation_configs(layout), 30)]
            for config in configs:
                assert sfs.get_mutation_config(config) == pytest.approx(_frequency(counts, layout, config),
                                                                         rel=1e-12, abs=1e-15)

    assert ms.sfs.mutation_layout(folded=True) == ms.fsfs.mutation_layout() == pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY).fsfs.mutation_layout()
    assert ms.sfs.mutation_layout(demes=True) == exact.mutation_layout(demes=True)
    assert ms.fsfs.mutation_layout(True) == pg.Coalescent(n={'pop_0': 2, 'pop_1': 2},
                                                          demography=DEMOGRAPHY).fsfs.mutation_layout(True)

    for config, p in itertools.islice(ms.fsfs.get_mutation_configs(), 20):
        assert ms.sfs.get_mutation_config(config) == p


def test_empirical_sfs_layout_boundaries(two_deme_mutations):
    """
    A layout of another spectrum raises, deme-resolved layouts need counts resolved by deme, one deme resolves them
    trivially, and the lookups survive dropping the counts and a jsonpickle round trip.
    """
    import jsonpickle

    ms = two_deme_mutations
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY)

    for layout in (coal.jsfs.mutation_layout(), pg.Coalescent(n=5).sfs.mutation_layout()):
        with pytest.raises(ValueError, match="does not belong"):
            ms.sfs.get_mutation_config(layout.config((0,) * len(layout)))

    unresolved = MsprimeCoalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY, num_replicates=50,
                                   n_threads=1, parallelize=False, simulate_mutations=True, mutation_rate=1.0,
                                   seed=2)
    layout = coal.sfs.mutation_layout(demes=True)
    with pytest.raises(ValueError, match="record_migration"):
        unresolved.sfs.get_mutation_config(layout.config((0,) * len(layout)))
    assert sum(unresolved.sfs.mutation_configs.values()) == pytest.approx(1)

    one = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, simulate_mutations=True,
                            mutation_rate=1.0, seed=1)
    deme_layout = pg.Coalescent(n=4).sfs.mutation_layout(demes=True)
    for config, p in itertools.islice(one.sfs.get_mutation_configs(), 30):
        assert one.sfs.get_mutation_config(deme_layout.config(tuple(config))) == p

    layouts = [coal.sfs.mutation_layout(), coal.sfs.mutation_layout(folded=True), layout]
    expected = [[ms.sfs.get_mutation_config(c) for c, _ in itertools.islice(ms.sfs.get_mutation_configs(l), 30)]
                for l in layouts]

    ms._touch()
    ms._drop()

    for dist in (ms.sfs, jsonpickle.decode(jsonpickle.encode(ms.sfs, keys=True), keys=True)):
        got = [[dist.get_mutation_config(c) for c, _ in itertools.islice(dist.get_mutation_configs(l), 30)]
               for l in layouts]
        assert got == expected


def test_distribution_joint_and_moment_of_rewards_use_the_simulated_genealogies():
    """
    The tree height and total branch length of a simulation provide the joint distribution of two rewards and the
    moments of rewards over a window, read from their genealogies as the coalescent-level methods do. Without
    genealogies, these raise.
    """
    ms = MsprimeCoalescent(n=3, num_replicates=200, n_threads=1, parallelize=False, seed=1)
    a, b = pg.TreeHeightReward(), pg.TotalBranchLengthReward()

    for dist in (ms.tree_height, ms.total_branch_length):
        joint, want = dist.joint(a, b), ms.joint(a, b)
        np.testing.assert_array_equal(joint._a, want._a)
        np.testing.assert_array_equal(joint._b, want._b)
        assert dist.moment(2, rewards=[a, b]) == ms.moment(2, rewards=[a, b])
        assert dist.moment(1, end_time=0.5) == pytest.approx(dist.accumulate(1, [0.5])[0], rel=1e-12)
        assert dist.moment(1, rewards=[dist._accumulator.reward]) == pytest.approx(dist.mean, rel=1e-12)

    sampled = pg.Coalescent(n=4).to_empirical(n_samples=500, seed=1)
    th = sampled.tree_height
    assert th.moment(1, rewards=[a]) == pytest.approx(th.mean, rel=1e-12)
    np.testing.assert_allclose(th.joint(a, b)._a, th.samples, rtol=1e-12)
    np.testing.assert_allclose(sampled.sfs.moment(2, start_time=0.0).data,
                               sampled.sfs.accumulate(2, [np.inf])[0], rtol=1e-12)
    np.testing.assert_allclose(sampled.sfs.moment(1, end_time=np.inf).data, sampled.sfs.mean.data, rtol=1e-12)

    plain = pg.Coalescent(n=3).tree_height.to_empirical(100, seed=1)
    assert plain.moment(2) == pg.distributions.EmpiricalDistribution(plain.samples).moment(2)
    for call in (lambda: plain.joint(a, b), lambda: plain.moment(1, rewards=[a])):
        with pytest.raises(NotImplementedError, match="SampledCoalescent"):
            call()


def test_empirical_joint_sfs_functions_select_bins_by_configs():
    """
    The empirical joint SFS pdf, cdf and quantile take the joint bins as ``configs``, as the exact ones do, label
    them alike and validate them.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=DEMOGRAPHY)
    emp = coal.jsfs.to_empirical(500, seed=1)
    configs = [(1, 0), [0, 1]]

    for kind in ('cdf', 'pdf', 'quantile'):
        got = getattr(emp, kind)._plot_data(configs=configs)
        assert got.labels == getattr(coal.jsfs, kind)._plot_data(configs=configs).labels == ['(1, 0)', '(0, 1)']
        assert len(got.y) == 2

        with pytest.raises(ValueError, match="descendant configuration"):
            getattr(emp, kind)._plot_data(configs=[(2, 1)])

    assert emp.cdf._plot_data().labels == [str(c) for c in emp._polymorphic_bins()]


def test_empirical_spectra_reject_the_layout_of_another_spectrum():
    """
    A layout with the bins of this spectrum's layout but other lineages or loci was accepted when its positions
    indexed stored entries, so a layout of three lineages returned a frequency of the four-lineage simulation. It
    raises ValueError as on the exact spectrum, before and after the own layout is memoized.
    """
    ms = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)
    loci = pg.LocusConfig(n=2, recombination_rate=1.0)
    ms2 = MsprimeCoalescent(n=3, loci=loci, num_replicates=200, n_threads=1, parallelize=False,
                            simulate_mutations=True, mutation_rate=1.0, seed=1)
    other_loci = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=2.0)).sfs2

    cases = [(ms.sfs, pg.Coalescent(n=3).sfs.mutation_layout()),
             (ms.sfs, pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY).sfs.mutation_layout()),
             (ms.fsfs, pg.Coalescent(n=3).fsfs.mutation_layout()),
             (ms2.sfs2, other_loci.mutation_layout())]

    for dist, foreign in cases:
        own = dist.mutation_layout()
        config = foreign.config((0,) * len(foreign))

        with pytest.raises(ValueError, match="does not belong"):
            dist.get_mutation_config(config)

        if foreign.bins == own.bins:
            assert sum(dist.mutation_configs.values()) == pytest.approx(1)

            with pytest.raises(ValueError, match="does not belong"):
                dist.get_mutation_config(config)

        with pytest.raises(ValueError, match="does not belong"):
            next(dist.get_mutation_configs(foreign))

    layout = pg.Coalescent(n=4).sfs.mutation_layout()
    assert ms.sfs.get_mutation_config(layout.config((1, 0, 0))) == ms.sfs.get_mutation_config((1, 0, 0))


def test_empirical_moments_after_drop_serve_the_retained_moments():
    """
    After ``_drop`` the moments of the empirical tree height, total branch length, SFS, folded SFS and two-locus SFS
    raised TypeError or AxisError on the freed samples, and the two-locus SFS raised for order zero. They serve order
    zero and the retained moments, and raise an informative ValueError for the others.
    """
    ms = MsprimeCoalescent(n=3, num_replicates=200, n_threads=1, parallelize=False, seed=1)
    ms2 = MsprimeCoalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0), num_replicates=200, n_threads=1,
                            parallelize=False, seed=1)
    dists = [ms.tree_height, ms.total_branch_length, ms.sfs, ms.fsfs, ms2.sfs2]
    expected = [[np.asarray(d.moment(k, center=c)) for k in range(5) for c in (True, False)] for d in dists]

    for coal in (ms, ms2):
        coal._touch()
        coal._drop()

    for dist, want in zip(dists, expected):
        assert dist.samples is None
        got = [np.asarray(dist.moment(k, center=c)) if (k < 3 or not c) else None for k in range(5)
               for c in (True, False)]

        for g, w in zip(got, want):
            if g is not None:
                np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-15)

        for k in (3, 4):
            with pytest.raises(ValueError, match="dropped"):
                dist.moment(k)

    bare = pg.distributions.EmpiricalDistribution(np.linspace(0.1, 2, 20))
    bare._drop()
    for k in range(3):
        with pytest.raises(ValueError, match="dropped"):
            bare.moment(k)


def test_fixtures_without_simulated_mutations_store_no_configuration_frequencies():
    """
    The serialized comparisons of configurations that simulate no mutations carried a point mass at the
    configuration without mutations, which a regenerated fixture does not store.
    """
    from pathlib import Path

    fixtures = sorted(Path('results/comparisons/serialized').glob('*.json'))
    assert fixtures

    for fixture in fixtures:
        config = Path('resources/configs') / f'{fixture.stem}.yaml'
        if 'simulate_mutations: true' not in config.read_text():
            assert '"_count_frequencies"' not in fixture.read_text(), fixture.stem
