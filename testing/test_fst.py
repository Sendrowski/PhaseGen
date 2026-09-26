"""
Test Hudson's F_ST (``Coalescent.fst``).

Fast tests are simulation-free: the symmetric two-deme island model has the closed form ``F_ST = 1 / (1 + 4 m)``,
plus the multi-population guard and MMC support. The msprime comparison is marked ``slow``.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent


def _island(m: float, sizes=(1.0, 1.0)) -> pg.Demography:
    """Symmetric two-deme island model with per-deme sizes ``sizes`` and symmetric migration rate ``m``."""
    return pg.Demography(
        pop_sizes={'pop_0': sizes[0], 'pop_1': sizes[1]},
        migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): m},
    )


@pytest.mark.parametrize("m", [0.25, 0.5, 1.0, 2.0])
def test_symmetric_island_closed_form(m):
    """For the symmetric equal-size two-deme island model, Hudson's F_ST = 1 / (1 + 4 m)."""
    fst = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_island(m)).fst
    assert fst == pytest.approx(1 / (1 + 4 * m), abs=1e-9)


def test_fst_in_unit_interval_and_decreases_with_migration():
    """F_ST lies in (0, 1) and decreases as migration increases (more gene flow -> less differentiation)."""
    fst = [pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_island(m)).fst for m in (0.1, 1.0, 10.0)]
    assert all(0 < f < 1 for f in fst)
    assert fst[0] > fst[1] > fst[2]


def test_requires_two_populations():
    """F_ST requires at least two sampled populations."""
    with pytest.raises(ValueError, match="two sampled populations"):
        pg.Coalescent(n=4).fst


@pytest.mark.parametrize("model", [pg.BetaCoalescent(alpha=1.5), pg.DiracCoalescent(psi=0.5, c=1.0)])
def test_fst_supports_multiple_merger_models(model):
    """F_ST is well-defined under multiple-merger coalescents (computed from two-lineage sub-coalescents)."""
    fst = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_island(0.5), model=model).fst
    assert 0 < fst < 1


def test_independent_of_sample_size_and_loci():
    """F_ST is a pairwise quantity, so it does not depend on the configured sample sizes or number of loci."""
    demo = _island(0.5)
    a = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=demo).fst
    b = pg.Coalescent(n={'pop_0': 5, 'pop_1': 3}, demography=demo).fst
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("m", [0.25, 1.0])
def test_fst_matches_msprime(m):
    """Analytical Hudson's F_ST matches an msprime branch-statistic ground truth."""
    demo = _island(m)
    ana = pg.Coalescent(n={'pop_0': 10, 'pop_1': 10}, demography=demo).fst
    ms = MsprimeCoalescent(n={'pop_0': 10, 'pop_1': 10}, demography=demo, num_replicates=20000, seed=42).fst
    assert ana == pytest.approx(ms, abs=0.01)


# ---- Patterson's f-statistics (linear combinations of pairwise coalescence times) ----

def _island4(sizes, m) -> pg.Demography:
    """Fully-connected four-deme island model (all pairs migrate, so all lineages coalesce)."""
    pops = ['pop_0', 'pop_1', 'pop_2', 'pop_3']
    return pg.Demography(
        pop_sizes={p: s for p, s in zip(pops, sizes)},
        migration_rates={(a, b): m for a in pops for b in pops if a != b},
    )


def _tree4_rates(strong: float, weak: float) -> dict:
    """Symmetric migration for a tree-like structure ((A,B),(C,D)): strong gene flow within each pair
    {pop_0, pop_1} and {pop_2, pop_3}, weak flow between the pairs. Unlike the fully-connected island
    model (whose symmetric topology forces f4 = 0), this gives the antisymmetric structure f4 detects."""
    pops = ['pop_0', 'pop_1', 'pop_2', 'pop_3']
    within = {('pop_0', 'pop_1'), ('pop_1', 'pop_0'), ('pop_2', 'pop_3'), ('pop_3', 'pop_2')}
    return {(a, b): (strong if (a, b) in within else weak) for a in pops for b in pops if a != b}


def _tree4(strong: float = 2.0, weak: float = 0.1) -> pg.Demography:
    """Four demes structured as a population tree ((pop_0, pop_1), (pop_2, pop_3))."""
    pops = ['pop_0', 'pop_1', 'pop_2', 'pop_3']
    return pg.Demography(pop_sizes={p: 1.0 for p in pops}, migration_rates=_tree4_rates(strong, weak))


def test_f4_zero_for_symmetric_model():
    """For a fully symmetric island model there is no treeness, so f4(A,B;C,D) = 0 exactly."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0] * 4, 0.5))
    assert coal.f4('pop_0', 'pop_1', 'pop_2', 'pop_3') == pytest.approx(0.0, abs=1e-9)


def test_f2_symmetric_and_positive():
    """f2 is symmetric in its arguments and positive between distinct populations."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0, 1.5, 0.7, 1.2], 0.4))
    assert coal.f2('pop_0', 'pop_1') == pytest.approx(coal.f2('pop_1', 'pop_0'), abs=1e-12)
    assert coal.f2('pop_0', 'pop_1') > 0


def test_f_statistic_identities():
    """f3 and f4 are fixed linear combinations of f2 (all are linear in pairwise coalescence times), so
    these algebraic identities hold exactly for any demography. This guards f3/f4 against the f2 baseline
    in the fast suite, where otherwise only f2 (and f4 = 0) carry a direct numeric check."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0, 1.5, 0.7, 1.2], 0.4))
    a, b, c, d = 'pop_0', 'pop_1', 'pop_2', 'pop_3'
    assert coal.f3(c, a, b) == pytest.approx((coal.f2(a, c) + coal.f2(b, c) - coal.f2(a, b)) / 2, abs=1e-9)
    assert coal.f4(a, b, c, d) == pytest.approx(
        (coal.f2(a, d) + coal.f2(b, c) - coal.f2(a, c) - coal.f2(b, d)) / 2, abs=1e-9)


def test_f4_detects_tree_topology():
    """f4's purpose is to discriminate tree structure: for a population tree ((A,B),(C,D)) the f4 that pairs
    the true clades vanishes, while the two 'crossing' orderings are equal, nonzero, and antisymmetric under
    swapping a pair. The symmetric island model only ever yields f4 = 0, so this exercises the other regime."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_tree4())
    a, b, c, d = 'pop_0', 'pop_1', 'pop_2', 'pop_3'
    assert coal.f4(a, b, c, d) == pytest.approx(0.0, abs=1e-9)                     # true topology -> 0
    assert coal.f4(a, c, b, d) > 1                                                 # crossing -> clearly nonzero
    assert coal.f4(a, c, b, d) == pytest.approx(coal.f4(a, d, b, c), abs=1e-9)     # both crossings agree
    assert coal.f4(a, c, b, d) == pytest.approx(-coal.f4(c, a, b, d), abs=1e-9)    # antisymmetric in a pair


def test_f_statistics_unknown_population_raises():
    """Referencing a population that does not exist raises a clear error."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0] * 4, 0.5))
    with pytest.raises(ValueError, match="Unknown population"):
        coal.f3('pop_9', 'pop_0', 'pop_1')


@pytest.mark.slow
def test_f_statistics_match_tskit_branch_mode():
    """Analytical f2/f3/f4 match tskit branch-mode f-statistics (which use the same 2x coalescence convention)."""
    import msprime as ms

    sizes = [1.0, 1.5, 0.7, 1.2]
    pops = ['pop_0', 'pop_1', 'pop_2', 'pop_3']
    m = 0.4
    coal = pg.Coalescent(n={p: 2 for p in pops}, demography=_island4(sizes, m))

    d = ms.Demography()
    for p, s in zip(pops, sizes):
        d.add_population(name=p, initial_size=s)
    for a in pops:
        for b in pops:
            if a != b:
                d.set_migration_rate(a, b, m)

    f2, f3, f4 = [], [], []
    for ts in ms.sim_ancestry(samples={p: 6 for p in pops}, demography=d, ploidy=1, sequence_length=1,
                              random_seed=7, num_replicates=20000):
        s = [ts.samples(population=i) for i in range(4)]
        f2.append(ts.f2([s[0], s[1]], mode='branch'))
        f3.append(ts.f3([s[2], s[0], s[1]], mode='branch'))
        f4.append(ts.f4([s[0], s[1], s[2], s[3]], mode='branch'))

    assert coal.f2('pop_0', 'pop_1') == pytest.approx(np.mean(f2), abs=0.05)
    assert coal.f3('pop_2', 'pop_0', 'pop_1') == pytest.approx(np.mean(f3), abs=0.05)
    assert coal.f4('pop_0', 'pop_1', 'pop_2', 'pop_3') == pytest.approx(np.mean(f4), abs=0.05)


@pytest.mark.slow
def test_f4_nonzero_matches_tskit_branch_mode_tree():
    """Validate a genuinely *nonzero* f4 against tskit branch-mode ground truth. The island-model case above
    only checks f4 ~ 0 (its symmetric topology forces this); here a tree-like structure ((A,B),(C,D)) gives
    the antisymmetric, admixture-detecting f4 in the 'crossing' ordering f4(A,C;B,D)."""
    import msprime as ms

    strong, weak = 2.0, 0.1
    pops = ['pop_0', 'pop_1', 'pop_2', 'pop_3']
    rates = _tree4_rates(strong, weak)
    coal = pg.Coalescent(n={p: 2 for p in pops}, demography=_tree4(strong, weak))

    d = ms.Demography()
    for p in pops:
        d.add_population(name=p, initial_size=1.0)
    for (src, dst), r in rates.items():
        d.set_migration_rate(src, dst, r)

    # the 'crossing' ordering (A,C;B,D) is the discriminating, nonzero statistic for this tree
    f4 = []
    for ts in ms.sim_ancestry(samples={p: 6 for p in pops}, demography=d, ploidy=1, sequence_length=1,
                              random_seed=11, num_replicates=20000):
        s = [ts.samples(population=i) for i in range(4)]
        f4.append(ts.f4([s[0], s[2], s[1], s[3]], mode='branch'))  # f4(A, C; B, D)

    ana = coal.f4('pop_0', 'pop_2', 'pop_1', 'pop_3')
    assert ana > 1                                            # genuinely nonzero (guards against trivial pass)
    assert ana == pytest.approx(np.mean(f4), abs=0.05)


def test_fst_respects_start_time():
    """
    The pairwise sub-coalescents behind F_ST and the f-statistics received ``end_time`` but not ``start_time``, so a
    coalescent windowed at ``start_time=1`` returned the unwindowed F_ST of 1/3.
    """
    demo = pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.5})
    c = pg.Coalescent(n={'a': 2, 'b': 2}, demography=demo, start_time=1.0, end_time=2.0)

    t_within = pg.Coalescent(n={'a': 2, 'b': 0}, demography=demo, start_time=1.0, end_time=2.0).tree_height.mean
    t_between = pg.Coalescent(n={'a': 1, 'b': 1}, demography=demo, start_time=1.0, end_time=2.0).tree_height.mean

    assert c.fst == pytest.approx(1 - t_within / t_between, rel=1e-8)
    assert c.f2('a', 'b') == pytest.approx(2 * t_between - 2 * t_within, rel=1e-8)


def test_fst_ignores_an_unsampled_population():
    """F_ST averages over the sampled populations, as the simulated ground truth does. Regression: an isolated,
    unsampled population entered both averages and moved the value from 0.500 to 0.416."""
    sizes = {'a': 1.0, 'b': 1.0}
    rates = {('a', 'b'): 0.5, ('b', 'a'): 0.5}

    plain = pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(pop_sizes=sizes, migration_rates=rates))
    ghost = pg.Coalescent(
        n={'a': 2, 'b': 2, 'c': 0},
        demography=pg.Demography(pop_sizes=sizes | {'c': 1.0}, migration_rates=rates)
    )

    assert ghost.fst == pytest.approx(plain.fst, rel=1e-10)


def test_simulated_fst_rejects_the_layouts_the_exact_one_rejects():
    """The ground truth raises where Coalescent.fst raises. Regression: it returned nan with a RuntimeWarning."""
    from phasegen.distributions.empirical import MsprimeCoalescent

    dem = pg.Demography(pop_sizes={'a': 1.0, 'b': 1.0}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.5})

    for n in ({'a': 3, 'b': 0}, {'a': 1, 'b': 1}):
        with pytest.raises(ValueError):
            MsprimeCoalescent(n=n, demography=dem, num_replicates=10).fst
        with pytest.raises(ValueError):
            pg.Coalescent(n=n, demography=dem).fst


def test_simulated_f_statistics_hold_for_single_lineages_and_repeated_populations():
    """The msprime f-statistics are built from simulated pairwise coalescence times, like the exact ones, so they do
    not depend on the sampled lineages. Regression: tskit's branch statistics on the sampled lineages returned NaN
    for a population with one lineage and -3.20 for f4(a, b; c, a), whose exact value is -1.66. The tolerance is
    about four standard errors at 20,000 replicates per pair."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 3, 'c': 1},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 1, 'c': 1},
                                 events=[pg.SymmetricMigrationRateChanges(pops=['a', 'b', 'c'], rate=0.5)])
    )
    ms = coal.to_msprime(num_replicates=20000, parallelize=False, seed=1)

    assert ms.f2('a', 'c') == pytest.approx(coal.f2('a', 'c'), abs=0.3)
    assert ms.f3('c', 'a', 'b') == pytest.approx(coal.f3('c', 'a', 'b'), abs=0.3)
    assert ms.f4('a', 'b', 'c', 'a') == pytest.approx(coal.f4('a', 'b', 'c', 'a'), abs=0.3)
