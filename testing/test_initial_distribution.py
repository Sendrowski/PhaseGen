"""
Test InitialDistribution, the weighted mixture of starting configurations.
"""

from typing import Callable, List, Tuple

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent

#: Time points at which the distribution functions are compared.
_T = np.linspace(0.05, 6, 9)


def _two_deme_demography() -> pg.Demography:
    """Two demes with asymmetric migration and a size change of ``pop_0`` at time 0.5."""
    return pg.Demography(
        pop_sizes={'pop_0': {0: 1, 0.5: 2}, 'pop_1': {0: 0.5}},
        migration_rates={('pop_0', 'pop_1'): {0: 0.6}, ('pop_1', 'pop_0'): {0: 1.1}}
    )


def _assert_linear(mix: pg.Coalescent, parts: List[Tuple[float, pg.Coalescent]], stats: List[Callable]) -> None:
    """
    Assert that every statistic of the mixture is the weighted sum of the statistics of its components.

    :param mix: Coalescent started from the initial distribution.
    :param parts: Pairs of normalized weight and coalescent started from the component.
    :param stats: Statistics, each mapping a coalescent to a number or an array.
    """
    for stat in stats:
        expected = sum(w * np.asarray(stat(c), dtype=float) for w, c in parts)
        np.testing.assert_allclose(np.asarray(stat(mix), dtype=float), expected, rtol=1e-12, atol=1e-15)


#: Statistics of a single-locus coalescent that are linear in the initial vector.
_SINGLE_LOCUS_STATS = [
    lambda c: [c.tree_height.moment(k, center=False) for k in (1, 2, 3)],
    lambda c: [c.total_branch_length.moment(k, center=False) for k in (1, 2)],
    lambda c: c.tree_height.cdf(_T),
    lambda c: c.tree_height.pdf(_T),
    lambda c: [f(c.distribution(pg.TotalBranchLengthReward()).lst(s))
               for s in (0.3, 1 + 2j) for f in (np.real, np.imag)],
    lambda c: np.real(c.joint_distribution(pg.TreeHeightReward(), pg.TotalBranchLengthReward()).lst(0.5, 0.2)),
    lambda c: c.sfs.mean.data,
    lambda c: c.sfs.moment(2, center=False).data,
    lambda c: c.fsfs.mean.data,
    lambda c: c.sfs.get_mutation_config([1, 0, 0], theta=1),
    lambda c: c.joint_distribution(pg.TreeHeightReward(), pg.TotalBranchLengthReward()).moment(1, 1),
]


def test_single_deme_mixture_of_one_configuration_equals_the_configuration():
    """Components with one deme share their configuration, so the mixture is the configuration itself."""
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.2, 1: 3}})
    mix = pg.Coalescent(n=pg.InitialDistribution([(1, 4), (3, pg.LineageConfig(4))]), demography=demography)
    plain = pg.Coalescent(n=4, demography=demography)

    _assert_linear(mix, [(1.0, plain)], _SINGLE_LOCUS_STATS)


@pytest.mark.parametrize('model', [pg.StandardCoalescent(), pg.BetaCoalescent(alpha=1.6)])
def test_two_deme_mixture_is_the_weighted_sum_of_its_components(model):
    """Raw moments, distribution functions, spectra and mutation configurations are linear in the initial vector."""
    components = [(1, [3, 1]), (3, [1, 3]), (2, [2, 2])]
    demography = _two_deme_demography()
    weights = np.array([w for w, _ in components]) / 6

    mix = pg.Coalescent(n=pg.InitialDistribution(components), demography=demography, model=model)
    parts = [(w, pg.Coalescent(n=c, demography=demography, model=model)) for w, (_, c) in zip(weights, components)]

    _assert_linear(mix, parts, _SINGLE_LOCUS_STATS + [lambda c: c.sfs.demes['pop_1'].mean.data])


def test_two_locus_mixture_over_unlinked_counts_is_the_weighted_sum_of_its_components():
    """Components with different numbers of initially unlinked lineages, over several epochs."""
    components = [(2, pg.LocusConfig(n=2, n_unlinked=0, recombination_rate=0.7)),
                  (1, pg.LocusConfig(n=2, n_unlinked=2, recombination_rate=0.7)),
                  (1, pg.LocusConfig(n=2, n_unlinked=3, recombination_rate=0.7))]
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 0.3, 1.2: 2}})
    weights = [0.5, 0.25, 0.25]

    mix = pg.Coalescent(n=3, loci=pg.InitialDistribution(components), demography=demography)
    parts = [(w, pg.Coalescent(n=3, loci=c, demography=demography)) for w, (_, c) in zip(weights, components)]

    _assert_linear(mix, parts, [
        lambda c: [c.tree_height.moment(k, center=False) for k in (1, 2)],
        lambda c: c.total_branch_length.moment(2, center=False),
        lambda c: c.tree_height.loci[0].mean,
        lambda c: c.tree_height.cdf(_T),
        lambda c: c.sfs2.mean.data,
    ])


def test_mixture_over_lineages_and_loci_combines_both_weights():
    """Lineage and locus mixtures on two demes and two loci combine into the product of their weights."""
    lineages = [(1, [2, 1]), (1, [1, 2])]
    loci = [(3, pg.LocusConfig(n=2, n_unlinked=0, recombination_rate=1)),
            (1, pg.LocusConfig(n=2, n_unlinked=2, recombination_rate=1))]
    demography = _two_deme_demography()

    mix = pg.Coalescent(n=pg.InitialDistribution(lineages), loci=pg.InitialDistribution(loci), demography=demography)
    parts = [(w_a * w_b, pg.Coalescent(n=a, loci=b, demography=demography))
             for w_a, (_, a) in zip([0.5, 0.5], lineages) for w_b, (_, b) in zip([0.75, 0.25], loci)]

    _assert_linear(mix, parts, [
        lambda c: [c.tree_height.moment(k, center=False) for k in (1, 2)],
        lambda c: c.total_branch_length.moment(1, center=False),
        lambda c: c.tree_height.cdf(_T),
    ])


def test_mixture_state_space_is_not_reused_for_other_weights():
    """State spaces of mixtures with different weights differ, so the inference cache does not share them."""
    a = pg.Coalescent(n=pg.InitialDistribution([(1, [2, 1]), (1, [1, 2])]))
    b = pg.Coalescent(n=pg.InitialDistribution([(1, [2, 1]), (2, [1, 2])]))
    c = pg.Coalescent(n=pg.InitialDistribution([(2, [2, 1]), (2, [1, 2])]))

    assert a.lineage_counting_state_space != b.lineage_counting_state_space
    assert a.lineage_counting_state_space == c.lineage_counting_state_space
    assert a.lineage_counting_state_space != pg.Coalescent(n=[2, 1]).lineage_counting_state_space


def test_serialization_round_trip():
    """A coalescent started from an initial distribution keeps it through to_json / from_json."""
    coal = pg.Coalescent(
        n=pg.InitialDistribution([(1, [2, 1]), (2, [1, 2])]),
        loci=pg.InitialDistribution([(1, pg.LocusConfig(n=2, recombination_rate=0.5)),
                                     (1, pg.LocusConfig(n=2, n_unlinked=1, recombination_rate=0.5))]),
        demography=_two_deme_demography()
    )
    mean = coal.tree_height.mean

    restored = pg.Coalescent.from_json(coal.to_json())

    assert restored.lineage_distribution == coal.lineage_distribution
    assert restored.locus_distribution == coal.locus_distribution
    assert restored.tree_height.mean == pytest.approx(mean, rel=1e-12)
    assert pg.Coalescent.from_json(coal.to_json()).tree_height.mean == pytest.approx(mean, rel=1e-12)


@pytest.mark.parametrize('components, error', [
    ([], ValueError),
    ([(0, 3), (1, 3)], ValueError),
    ([(-1, 3), (1, 3)], ValueError),
    ([(np.inf, 3)], ValueError),
    ([(np.nan, 3)], ValueError),
    ([('a', 3)], TypeError),
    ([(1, 3, 4)], TypeError),
    ([3], TypeError),
    ([(1, 3), (1, 4)], ValueError),
    ([(1, [2, 1]), (1, {'a': 2, 'b': 1})], ValueError),
    ([(1, [2, 1]), (1, [2, 1, 0])], ValueError),
    ([(1, pg.LocusConfig(n=2)), (1, pg.LocusConfig(n=1))], ValueError),
    ([(1, pg.LocusConfig(n=2, recombination_rate=1)), (1, pg.LocusConfig(n=2, recombination_rate=2))], ValueError),
    ([(1, 3), (1, pg.LocusConfig(n=2))], TypeError),
])
def test_invalid_components_raise(components, error):
    """Weights are positive and finite, and all components share one state space."""
    with pytest.raises(error):
        pg.InitialDistribution(components)


def test_weights_are_normalized():
    """Weights are any positive numbers, normalized over the components."""
    np.testing.assert_allclose(pg.InitialDistribution([(2, [2, 1]), (6, [1, 2])]).weights, [0.25, 0.75])


def test_invalid_uses_raise():
    """The kind of configuration must match the argument, and the joint SFS and F_ST need one sample."""
    lineages = pg.InitialDistribution([(1, [2, 1]), (1, [1, 2])])

    with pytest.raises(TypeError):
        pg.Coalescent(n=pg.InitialDistribution([(1, pg.LocusConfig(n=2))]))

    with pytest.raises(TypeError):
        pg.Coalescent(n=3, loci=lineages)

    with pytest.raises(ValueError):
        pg.Coalescent(n=2, loci=pg.InitialDistribution([(1, pg.LocusConfig(n=2, n_unlinked=3))]))

    with pytest.raises(ValueError):
        _ = pg.Coalescent(n=lineages).jsfs

    with pytest.raises(ValueError):
        _ = pg.Coalescent(n=lineages).fst


def test_msprime_agrees_with_mixture_over_deme_placements():
    """Each replicate draws its deme placement from the weights, which msprime reproduces within four standard
    errors."""
    components = [(1, [3, 0]), (2, [0, 3]), (1, [2, 1])]
    demography = _two_deme_demography()
    n_rep = 20000

    coal = pg.Coalescent(n=pg.InitialDistribution(components), demography=demography)
    ms = MsprimeCoalescent(n=pg.InitialDistribution(components), demography=demography, num_replicates=n_rep,
                           n_threads=4, parallelize=False, seed=3)

    for exact, sampled in [(coal.tree_height, ms.tree_height), (coal.total_branch_length, ms.total_branch_length)]:
        se = np.std(sampled.samples) / np.sqrt(len(sampled.samples))
        assert abs(sampled.mean - exact.mean) < 4 * se

    sfs = ms.sfs.samples[:, 1:-1]
    se = sfs.std(axis=0) / np.sqrt(len(sfs))
    assert np.all(np.abs(sfs.mean(axis=0) - coal.sfs.mean.data[1:-1]) < 4 * se)

    # the placements of the plain configurations differ, so the mixture is distinguishable from each of them
    assert abs(coal.tree_height.mean - pg.Coalescent(n=[2, 1], demography=demography).tree_height.mean) > 0.05
