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


def _two_locus_third_cross_moment(n, i, j, k, r, model=None):
    """Two-locus third cross-moment ``E[L^0_i · L^0_j · L^1_k]`` (two rewards at locus 0, one at locus 1)."""
    from phasegen.distributions import PhaseTypeDistribution
    from phasegen.rewards import CombinedReward

    kwargs = {} if model is None else dict(model=model)
    dist = pg.Coalescent(n=n, loci=2, recombination_rate=r, **kwargs).sfs2
    return PhaseTypeDistribution.moment(
        dist, k=3, permute=False, center=False,
        rewards=(
            CombinedReward([dist.reward, pg.TwoLocusSFSReward(0, i)]),
            CombinedReward([dist.reward, pg.TwoLocusSFSReward(0, j)]),
            CombinedReward([dist.reward, pg.TwoLocusSFSReward(1, k)]),
        )
    )


def _single_locus_third_cross_moment(n, i, j, k, model=None):
    """Single-locus third cross-moment ``E[L_i · L_j · L_k]`` on the single-locus block-counting space."""
    from phasegen.distributions import PhaseTypeDistribution
    from phasegen.rewards import CombinedReward

    kwargs = {} if model is None else dict(model=model)
    dist = pg.Coalescent(n=n, **kwargs).sfs
    return PhaseTypeDistribution.moment(
        dist, k=3, permute=False, center=False,
        rewards=tuple(CombinedReward([dist.reward, dist._get_sfs_reward(b)]) for b in (i, j, k))
    )


@pytest.mark.parametrize("name, model", MODELS, ids=[m[0] for m in MODELS])
def test_higher_order_moment_r_zero_equals_single_locus(name, model):
    """Beyond the (second-moment) 2-SFS, a genuine third cross-moment ``E[L^0_i L^0_j L^1_k]`` at ``r = 0`` (fully
    linked, both loci share one tree) must equal the single-locus third cross-moment ``E[L_i L_j L_k]``."""
    n = 3
    for i in (1, 2):
        for j in (1, 2):
            for k in (1, 2):
                two = _two_locus_third_cross_moment(n, i, j, k, 0.0, model)
                ref = _single_locus_third_cross_moment(n, i, j, k, model)
                assert two == pytest.approx(ref, abs=1e-10), f"{name} ({i},{j},{k})"


def test_higher_order_moment_numba_python_parity():
    """The third cross-moment is identical whether the two-locus state space is built with numba or pure Python."""
    n, r = 3, 1.0
    Settings.use_numba = True
    numba = _two_locus_third_cross_moment(n, 1, 2, 2, r)
    Settings.use_numba = False
    python = _two_locus_third_cross_moment(n, 1, 2, 2, r)
    assert numba == pytest.approx(python, abs=1e-12)


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


@pytest.mark.parametrize("n_unlinked", [0, 1, 2])
def test_starting_config_n_unlinked(n_unlinked):
    """Different starting linkage (``n_unlinked`` lineages start unlinked across the loci) is supported and gives
    numba/Python-consistent results; starting more lineages unlinked reduces the inter-locus correlation."""
    n, r = 4, 0.75

    def two(use_numba):
        Settings.use_numba = use_numba
        coal = pg.Coalescent(n=n, loci=pg.LocusConfig(n=2, n_unlinked=n_unlinked, recombination_rate=r))
        return np.asarray(coal.sfs2.mean.data)

    numba = two(True)
    python = two(False)
    np.testing.assert_allclose(numba, python, atol=1e-10)


def test_more_unlinked_reduces_correlation():
    """Starting more lineages unlinked moves the off-diagonal of the 2-SFS toward the independent (smaller) value."""
    n, r = 4, 0.75
    off = [
        np.asarray(pg.Coalescent(n=n, loci=pg.LocusConfig(n=2, n_unlinked=u, recombination_rate=r)).sfs2.mean.data)[1, 1]
        for u in (0, 1, 2)
    ]
    assert off[0] > off[1] > off[2]


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


def test_reward_state_space_guards():
    """The two-locus rewards declare support for exactly the two-locus state space (mirroring the joint-SFS guards),
    and incompatible reward/state-space combinations raise rather than silently computing the wrong thing. This
    matters because the two-locus state space subclasses the joint one, so without explicit guards a joint-SFS reward
    would evaluate on it (and vice versa)."""
    from phasegen.state_space import (BlockCountingStateSpace, JointBlockCountingStateSpace,
                                       TwoLocusBlockCountingStateSpace, LineageCountingStateSpace)

    two = pg.TwoLocusSFSReward(0, 1)
    joint = pg.JointSFSReward((1, 1))

    # supports() is a clean diagonal: each reward supports only its own state space
    assert two.supports(TwoLocusBlockCountingStateSpace)
    assert not two.supports(JointBlockCountingStateSpace)
    assert not two.supports(BlockCountingStateSpace)
    assert not joint.supports(TwoLocusBlockCountingStateSpace)

    # a two-locus reward no longer masquerades as one requiring the (population) joint state space
    assert not pg.Reward.requires_joint_state_space([two])

    ss = TwoLocusBlockCountingStateSpace(
        lineage_config=pg.LineageConfig(3),
        locus_config=pg.LocusConfig(n=2, recombination_rate=1.0),
        epoch=pg.Epoch(),
    )

    # a joint-SFS reward must NOT silently evaluate on the two-locus space (it subclasses the joint one)
    with pytest.raises(NotImplementedError):
        joint._get(ss)

    # and a two-locus reward must reject a non-two-locus space
    single = pg.Coalescent(n=3).block_counting_state_space
    with pytest.raises(NotImplementedError):
        two._get(single)


def _msprime_two_locus_sfs(n, r, ms_model, reps, seed, ms_demography=None):
    """Two-locus SFS via msprime: two sites at recombination distance r, the per-bin branch-length cross product."""
    import msprime as ms

    sim_kwargs = dict(samples=n, sequence_length=2, recombination_rate=r, ploidy=1, model=ms_model,
                      num_replicates=reps, random_seed=seed)
    if ms_demography is None:
        sim_kwargs['population_size'] = 1
    else:
        sim_kwargs['demography'] = ms_demography

    out = np.zeros((n + 1, n + 1))
    for ts in ms.sim_ancestry(**sim_kwargs):
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


def _ms_model(name):
    import msprime as ms
    return {"standard": ms.StandardCoalescent(), "beta": ms.BetaCoalescent(alpha=1.5),
            "dirac": ms.DiracCoalescent(psi=0.5, c=1.0)}[name]


# (name, phasegen model, msprime model name, n, r) — varying coalescent model and number of lineages
MSPRIME_CASES = [
    ("standard n=3", pg.StandardCoalescent(), "standard", 3, 1.0),
    ("standard n=4", pg.StandardCoalescent(), "standard", 4, 0.5),
    ("beta n=3", pg.BetaCoalescent(alpha=1.5), "beta", 3, 1.0),
    ("dirac n=3", pg.DiracCoalescent(psi=0.5, c=1.0), "dirac", 3, 1.0),
]


@pytest.mark.slow
@pytest.mark.parametrize("name, model, ms_model, n, r", MSPRIME_CASES, ids=[c[0] for c in MSPRIME_CASES])
def test_msprime_two_locus_sfs(name, model, ms_model, n, r):
    """The analytical two-locus SFS matches msprime two-locus simulations across models and sample sizes, with
    recombination mapping directly between the two."""
    ana = _two_sfs(n, r, model)
    sim = _msprime_two_locus_sfs(n, r, _ms_model(ms_model), reps=200000, seed=42)

    s = slice(1, n)
    np.testing.assert_allclose(ana[s, s], sim[s, s], atol=0.05, err_msg=name)


@pytest.mark.slow
def test_msprime_two_locus_sfs_two_epoch():
    """The two-locus SFS matches msprime under a two-epoch (time-varying population size) demography."""
    import msprime as ms

    n, r = 3, 1.0
    # population halves at time 1.0
    ana = _two_sfs_demography(n, r)

    dem = ms.Demography()
    dem.add_population(initial_size=1.0)
    dem.add_population_parameters_change(time=1.0, initial_size=0.5)
    sim = _msprime_two_locus_sfs(n, r, ms.StandardCoalescent(), reps=200000, seed=43, ms_demography=dem)

    s = slice(1, n)
    np.testing.assert_allclose(ana[s, s], sim[s, s], atol=0.05)


def _two_sfs_demography(n, r):
    """Analytical two-locus SFS under a two-epoch demography (size halves at t=1)."""
    coal = pg.Coalescent(n=n, loci=2, recombination_rate=r,
                         demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 1.0: 0.5}}))
    return np.asarray(coal.sfs2.mean.data)
