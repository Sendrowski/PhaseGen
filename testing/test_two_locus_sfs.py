"""
Test the two-locus site-frequency spectrum under recombination.

The fast tests are simulation-free invariants: the recombination limits (``r -> 0`` reproduces the single-locus
cross-moment of the SFS, ``r -> inf`` the outer product of the marginal SFS), symmetry, numba/Python parity, and the
container. The msprime comparisons are marked ``slow``.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings
from phasegen.state_space import TwoLocusBlockCountingStateSpace

MODELS = [
    ("standard", pg.StandardCoalescent()),
    ("beta", pg.BetaCoalescent(alpha=1.5)),
    ("dirac", pg.DiracCoalescent(psi=0.5, c=1.0)),
]


@pytest.fixture(autouse=True)
def _restore_numba():
    prev = Settings.use_numba
    yield
    Settings.use_numba = prev


def _two_sfs(n, r, model=None):
    """Mean two-locus SFS data array."""
    kwargs = {} if model is None else dict(model=model)
    return np.asarray(pg.Coalescent(n=n, loci=2, recombination_rate=r, **kwargs).sfs2.mean.data)


def _single_locus_cross_moment(n, model=None):
    """Single-locus uncentered cross-moment of the SFS, ``E[L_i L_j] = cov[i,j] + mean_i mean_j``."""
    kwargs = {} if model is None else dict(model=model)
    coal = pg.Coalescent(n=n, **kwargs)
    mean = np.asarray(coal.sfs.mean.data)
    return np.asarray(coal.sfs.cov.data) + np.outer(mean, mean)


def test_state_space_structure():
    """Basic structure of the two-locus block-counting state space."""
    n = 3
    ss = TwoLocusBlockCountingStateSpace(
        lineage_config=pg.LineageConfig(n),
        locus_config=pg.LocusConfig(n=2, recombination_rate=1.0),
        epoch=pg.Epoch(),
    )

    # blocks are the two-locus descendant vectors (a0, a1), 0 <= a_l <= n, not both zero
    assert ss.n_blocks == (n + 1) ** 2 - 1
    assert ss.block_index[(1, 1)] >= 0

    # valid generator with a unique initial state and at least one absorbing state
    assert np.allclose(ss.S.sum(axis=1), 0)
    assert ss.alpha.sum() == pytest.approx(1.0)
    assert (ss.alpha > 0).sum() == 1
    assert any(ss._is_absorbing(s) for s in ss.states)


@pytest.mark.parametrize("name, model", MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.parametrize("n", [3, 4])
def test_r_zero_equals_single_locus_cross_moment(name, model, n):
    """At ``r = 0`` (fully linked) the two-locus SFS equals the single-locus cross-moment of the SFS."""
    two = _two_sfs(n, 0.0, model)
    ref = _single_locus_cross_moment(n, model)

    s = slice(1, n)
    np.testing.assert_allclose(two[s, s], ref[s, s], atol=1e-10, err_msg=f"{name} n={n}")


def test_large_r_approaches_independent_standard():
    """For the standard coalescent, as ``r -> inf`` the two loci become independent and the 2-SFS approaches the
    outer product of the marginal SFS. (This factorization does NOT hold for multiple-merger models, where a single
    merger event couples both loci even when they are unlinked, see :func:`test_mmc_stays_correlated_at_large_r`.)"""
    n = 3
    mean = np.asarray(pg.Coalescent(n=n).sfs.mean.data)
    outer = np.outer(mean, mean)

    two = _two_sfs(n, 1e4)

    s = slice(1, n)
    np.testing.assert_allclose(two[s, s], outer[s, s], atol=1e-2)


@pytest.mark.parametrize("name, model", [m for m in MODELS if m[0] != "standard"], ids=["beta", "dirac"])
def test_mmc_stays_correlated_at_large_r(name, model):
    """Under multiple-merger coalescents the two loci remain correlated even at very large recombination (shared
    merger events couple them), so the 2-SFS does NOT converge to the outer product of the marginals."""
    n = 3
    mean = np.asarray(pg.Coalescent(n=n, model=model).sfs.mean.data)
    outer = np.outer(mean, mean)

    two = _two_sfs(n, 1e6, model)

    # converged in r (plateaued) but distinctly above the independent outer product
    assert np.abs(two[1, 1] - _two_sfs(n, 1e8, model)[1, 1]) < 1e-4
    assert two[1, 1] > outer[1, 1] * 1.02


def test_symmetric():
    """The two-locus SFS is symmetric (the two loci are exchangeable)."""
    two = _two_sfs(4, 1.0)
    np.testing.assert_allclose(two, two.T, atol=1e-12)


@pytest.mark.parametrize("name, model", MODELS, ids=[m[0] for m in MODELS])
def test_numba_python_parity(name, model):
    """The numba and pure-Python construction give the same two-locus SFS."""
    Settings.use_numba = True
    numba = _two_sfs(4, 0.75, model)

    Settings.use_numba = False
    python = _two_sfs(4, 0.75, model)

    np.testing.assert_allclose(numba, python, atol=1e-10, err_msg=name)


def test_marginal_consistency():
    """Summing the 2-SFS at r=0 over one locus reproduces (a scaling of) the per-bin second moments; here we check
    the simpler invariant that the diagonal at r=0 equals the single-locus second moments E[L_i^2]."""
    n = 4
    two = _two_sfs(n, 0.0)
    coal = pg.Coalescent(n=n)
    mean = np.asarray(coal.sfs.mean.data)
    second = np.asarray(coal.sfs.cov.data).diagonal() + mean ** 2  # E[L_i^2]

    for i in range(1, n):
        assert two[i, i] == pytest.approx(second[i], abs=1e-9)


def test_container_and_entry_point():
    """``coal.sfs2`` returns a TwoLocusSFS of the right shape; plotting runs."""
    import matplotlib
    matplotlib.use('Agg')

    coal = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    sfs2 = coal.sfs2.mean

    assert isinstance(sfs2, pg.TwoLocusSFS)
    assert sfs2.data.shape == (4, 4)
    assert np.all(sfs2.data >= 0)
    sfs2.plot(show=False)


def test_requires_two_loci_single_population():
    """``sfs2`` requires exactly two loci and a single population."""
    with pytest.raises(ValueError):
        pg.Coalescent(n=4).sfs2.mean  # one locus

    with pytest.raises(NotImplementedError):
        pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 2}, loci=2, recombination_rate=1.0,
            demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
                                     migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0})
        ).sfs2.mean  # multiple populations


def _msprime_two_locus_sfs(n, r, ms_model, reps, seed):
    """Two-locus SFS via msprime: two sites at recombination distance r, the per-bin branch-length cross product."""
    import msprime as ms

    out = np.zeros((n + 1, n + 1))
    for ts in ms.sim_ancestry(samples=n, sequence_length=2, recombination_rate=r, ploidy=1,
                              population_size=1, model=ms_model, num_replicates=reps, random_seed=seed):
        t0, t1 = ts.at(0.5), ts.at(1.5)
        left = np.zeros(n + 1)
        right = np.zeros(n + 1)
        for nd in t0.nodes():
            if t0.parent(nd) != -1:
                left[t0.num_samples(nd)] += t0.branch_length(nd)
        for nd in t1.nodes():
            if t1.parent(nd) != -1:
                right[t1.num_samples(nd)] += t1.branch_length(nd)
        out += np.outer(left, right)

    return out / reps


@pytest.mark.slow
@pytest.mark.parametrize("name, model, ms_model", [
    ("standard", pg.StandardCoalescent(), "standard"),
    ("beta", pg.BetaCoalescent(alpha=1.5), "beta"),
    ("dirac", pg.DiracCoalescent(psi=0.5, c=1.0), "dirac"),
], ids=["standard", "beta", "dirac"])
def test_msprime_two_locus_sfs(name, model, ms_model):
    """The analytical two-locus SFS matches msprime two-locus simulations (standard and multiple-merger models, with
    recombination mapping directly between the two)."""
    import msprime as ms

    ms_models = {"standard": ms.StandardCoalescent(), "beta": ms.BetaCoalescent(alpha=1.5),
                 "dirac": ms.DiracCoalescent(psi=0.5, c=1.0)}

    n, r = 3, 1.0
    ana = _two_sfs(n, r, model)
    sim = _msprime_two_locus_sfs(n, r, ms_models[ms_model], reps=200000, seed=42)

    s = slice(1, n)
    np.testing.assert_allclose(ana[s, s], sim[s, s], atol=0.05, err_msg=name)
