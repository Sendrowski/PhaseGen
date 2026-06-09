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
    """F_ST requires at least two populations."""
    with pytest.raises(ValueError, match="two populations"):
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


def test_f4_zero_for_symmetric_model():
    """For a fully symmetric island model there is no treeness, so f4(A,B;C,D) = 0 exactly."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0] * 4, 0.5))
    assert coal.f4('pop_0', 'pop_1', 'pop_2', 'pop_3') == pytest.approx(0.0, abs=1e-9)


def test_f2_symmetric_and_positive():
    """f2 is symmetric in its arguments and positive between distinct populations."""
    coal = pg.Coalescent(n={f'pop_{i}': 2 for i in range(4)}, demography=_island4([1.0, 1.5, 0.7, 1.2], 0.4))
    assert coal.f2('pop_0', 'pop_1') == pytest.approx(coal.f2('pop_1', 'pop_0'), abs=1e-12)
    assert coal.f2('pop_0', 'pop_1') > 0


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
