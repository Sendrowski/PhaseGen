"""
Tests for :class:`phasegen.distributions.reward.RewardDistribution` — the full distribution (CDF/PDF/quantile)
of an accumulated reward, obtained from the Laplace-Stieltjes transform and its numerical (de Hoog) inversion.

The references are *exact*, not simulated: for a single epoch the accumulated reward is phase-type with the
reward-transformed generator ``diag(1/r) T`` (with the zero-reward states censored), and the multi-epoch tree
height equals PhaseGen's own (matrix-exponential) ``tree_height.cdf``. The sparse (complex ``expm_multiply`` +
block-triangular LU) path is pinned against the dense path. msprime ground truth is exercised separately through
the comparison scenarios (``total_branch_length`` CDF/PDF in the configs).
"""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import phasegen as pg
from phasegen.settings import Settings
from phasegen.rewards import UnfoldedSFSReward


@pytest.fixture(autouse=True)
def _restore_settings():
    saved = (Settings.closed_form_sparse_min_states, Settings.flatten_block_counting)
    yield
    Settings.closed_form_sparse_min_states, Settings.flatten_block_counting = saved


def _single_epoch_reference_cdf(dist, reward):
    """
    Exact single-epoch CDF: the accumulated reward is phase-type with generator ``diag(1/r) G``, where ``G`` is the
    transient generator with the zero-reward states censored (folded in via ``T_PP + T_PZ (-T_ZZ)^-1 T_ZP``).
    """
    ss = dist.state_space
    S = np.asarray(ss.S.todense()) if sp.issparse(ss.S) else np.asarray(ss.S)
    idx = np.where(~ss.absorbing)[0]
    T = S[np.ix_(idx, idx)]
    alpha = np.asarray(ss.alpha)[idx].astype(float)
    r = np.asarray(reward._get(ss))[idx].astype(float)

    P = np.where(r > 0)[0]
    Z = np.where(r == 0)[0]
    if len(Z):
        neg_zinv = sla.inv(-T[np.ix_(Z, Z)])
        G = T[np.ix_(P, P)] + T[np.ix_(P, Z)] @ neg_zinv @ T[np.ix_(Z, P)]
        beta = alpha[P] + alpha[Z] @ neg_zinv @ T[np.ix_(Z, P)]
    else:
        G = T[np.ix_(P, P)]
        beta = alpha[P]

    A = np.diag(1.0 / r[P]) @ G
    ones = np.ones(len(P))
    return lambda x: float(1 - beta @ sla.expm(A * x) @ ones)


# ----------------------------------------------------------------------------------------------------------------
# exact references
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 6, 8])
def test_single_epoch_total_branch_length_matches_reward_transform(n):
    """Total branch length (all-positive reward): inverted CDF equals the exact reward-transform phase-type CDF."""
    dist = pg.Coalescent(n=n).total_branch_length
    rd = dist.distribution()
    ref = _single_epoch_reference_cdf(dist, dist.reward)

    for x in [0.5, 1.0, 2.0, 4.0, 7.0]:
        assert abs(rd.cdf(x) - ref(x)) < 1e-5, (n, x, rd.cdf(x), ref(x))


@pytest.mark.parametrize("i", [1, 2, 3])
def test_single_epoch_sfs_bin_matches_censored_reward_transform(i):
    """An SFS bin reward has zero-reward states: inverted CDF equals the censored reward-transform CDF (exact)."""
    n = 7
    dist = pg.Coalescent(n=n).sfs
    reward = UnfoldedSFSReward(i)
    rd = dist.distribution(reward=reward)
    ref = _single_epoch_reference_cdf(dist, reward)

    for x in [0.3, 0.8, 1.5, 3.0]:
        assert abs(rd.cdf(x) - ref(x)) < 1e-5, (i, x, rd.cdf(x), ref(x))


def test_multi_epoch_tree_height_matches_phasegen():
    """The inverted tree-height CDF equals PhaseGen's matrix-exponential ``tree_height.cdf`` (3 epochs), away from
    the epoch boundaries, where numerical Laplace inversion has a small Gibbs-type error at the CDF's kink."""
    demo = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.4: 0.25, 1.0: 2.0}})
    coal = pg.Coalescent(n=6, demography=demo)
    rd = coal.tree_height.distribution()

    for x in [0.2, 0.7, 1.3, 2.0, 3.5]:  # deliberately not 0.4 / 1.0 (epoch boundaries)
        assert abs(rd.cdf(x) - float(coal.tree_height.cdf(x))) < 1e-5, (x, rd.cdf(x), float(coal.tree_height.cdf(x)))


# ----------------------------------------------------------------------------------------------------------------
# implementation paths and invariants
# ----------------------------------------------------------------------------------------------------------------
def test_sparse_path_matches_dense():
    """Forcing the sparse path (complex ``expm_multiply`` + complex block-triangular LU) matches the dense path."""
    def cdf_values():
        return np.array([pg.Coalescent(n=8).total_branch_length.distribution().cdf(x) for x in [1.0, 3.0, 6.0]])

    Settings.closed_form_sparse_min_states = 10 ** 9
    dense = cdf_values()
    Settings.closed_form_sparse_min_states = 0
    sparse = cdf_values()

    np.testing.assert_allclose(sparse, dense, atol=1e-9)


def test_multi_epoch_total_branch_length_sparse_matches_dense():
    """Sparse vs dense also agree on a multi-epoch model (the finite-epoch complex action path)."""
    demo = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}})

    def cdf_values():
        return np.array([pg.Coalescent(n=7, demography=demo).total_branch_length.distribution().cdf(x)
                         for x in [1.0, 3.0, 6.0]])

    Settings.closed_form_sparse_min_states = 10 ** 9
    dense = cdf_values()
    Settings.closed_form_sparse_min_states = 0
    sparse = cdf_values()

    np.testing.assert_allclose(sparse, dense, atol=1e-9)


def test_quantile_roundtrip():
    """``cdf(quantile(q)) == q`` for the accumulated-reward distribution."""
    rd = pg.Coalescent(n=6).total_branch_length.distribution()
    for q in [0.1, 0.5, 0.9]:
        assert abs(rd.cdf(rd.quantile(q)) - q) < 1e-4, q


def test_sfs_bin_quantile_roundtrip():
    """Regression: an SFS-bin distribution's quantile must work (its host's ``mean`` is a spectrum, not a scalar,
    so the bracket seed must come from the LST, not ``host.moment``)."""
    rd = pg.Coalescent(n=7).sfs.distribution(reward=UnfoldedSFSReward(2))
    for q in [0.25, 0.5, 0.9]:
        assert abs(rd.cdf(rd.quantile(q)) - q) < 1e-4, q


def test_cos_curve_matches_dehoog():
    """The fast COS curve (used for plotting) matches the exact per-point de Hoog inversion to plotting accuracy."""
    for coal, reward in [
        (pg.Coalescent(n=8), None),  # total branch length below
        (pg.Coalescent(n=8), UnfoldedSFSReward(2)),
    ]:
        dist = coal.total_branch_length if reward is None else coal.sfs
        rd = dist.distribution(reward=reward)
        xs = np.linspace(0.1, rd._range(scale=8), 40)
        # CDF is accurate; the PDF is plotting-grade (a cosine series has a small boundary/Gibbs error near the
        # support edge and any atom, invisible on a plot)
        np.testing.assert_allclose(rd.cdf_curve(xs), rd.cdf(xs), atol=3e-3)
        np.testing.assert_allclose(rd.pdf_curve(xs), rd.pdf(xs), atol=3e-2)


def test_cos_curve_recovers_atom():
    """For an SFS bin that may be empty, the COS CDF starts at the atom ``P(R=0) = phi(inf)``."""
    rd = pg.Coalescent(n=7).sfs.distribution(reward=UnfoldedSFSReward(3))
    p0 = rd.lst(1e8).real
    assert p0 > 0.01  # this bin is empty with non-negligible probability
    # the CDF just above 0 is essentially the atom
    assert abs(rd.cdf_curve(np.array([1e-6]))[0] - p0) < 5e-3


def test_shared_epoch_data_across_bins():
    """All bins of a spectrum share the (reward-independent) per-epoch generators built once on the host."""
    dist = pg.Coalescent(n=6).sfs
    a = dist.distribution(reward=UnfoldedSFSReward(1))._setup
    b = dist.distribution(reward=UnfoldedSFSReward(2))._setup
    # same shared epoch data object, different reward vectors
    assert a['T_epochs'] is b['T_epochs']
    assert not np.array_equal(a['r'], b['r'])


def test_pdf_matches_cdf_finite_difference():
    """The inverted PDF matches a central finite difference of the inverted CDF."""
    rd = pg.Coalescent(n=6).total_branch_length.distribution()
    h = 1e-3
    for x in [1.5, 3.0, 5.0]:
        fd = (rd.cdf(x + h) - rd.cdf(x - h)) / (2 * h)
        assert abs(rd.pdf(x) - fd) < 1e-3, (x, rd.pdf(x), fd)


def test_cdf_is_monotone_and_bounded():
    """The CDF is non-decreasing on [0, inf) and bounded in [0, 1]."""
    rd = pg.Coalescent(n=6).total_branch_length.distribution()
    xs = np.linspace(0, 20, 25)
    F = rd.cdf(xs)
    assert np.all(F >= -1e-9) and np.all(F <= 1 + 1e-9)
    assert np.all(np.diff(F) >= -1e-6)
    assert F[-1] > 0.99  # essentially absorbed by x = 20


def test_array_input():
    """CDF and PDF accept array arguments and return arrays."""
    rd = pg.Coalescent(n=5).total_branch_length.distribution()
    xs = np.array([1.0, 2.0, 3.0])
    assert rd.cdf(xs).shape == xs.shape
    assert rd.pdf(xs).shape == xs.shape
    assert rd.cdf(2.0) == pytest.approx(rd.cdf(xs)[1])


def test_sfs_cdf_pdf_quantile_are_per_bin_vectors():
    """A spectrum's CDF/PDF/quantile are vector-valued over bins (like ``mean``), not the tree-height footgun."""
    coal = pg.Coalescent(n=6)
    sfs = coal.sfs
    F = sfs.cdf(2.0)

    # same shape as the mean spectrum, and the polymorphic entries equal the per-bin distributions
    from phasegen.spectrum import SFS
    assert isinstance(F, SFS)
    assert np.asarray(F.data).shape == np.asarray(sfs.mean.data).shape
    for i in sfs._get_indices():
        bin_cdf = sfs.distribution(reward=UnfoldedSFSReward(i)).cdf(2.0)
        assert np.asarray(F.data)[i] == pytest.approx(bin_cdf, abs=1e-9)

    # not the tree-height distribution (the previous silent footgun)
    assert np.asarray(F.data)[1] != pytest.approx(float(coal.tree_height.cdf(2.0)), abs=1e-6)

    # quantile and pdf are vectors too; arrays (curves) are rejected (use plot_cdf/plot_pdf)
    assert np.asarray(sfs.quantile(0.5).data).shape == np.asarray(sfs.mean.data).shape
    assert np.asarray(sfs.pdf(1.0).data).shape == np.asarray(sfs.mean.data).shape
    with pytest.raises(ValueError):
        sfs.cdf(np.array([1.0, 2.0]))


def test_jsfs_cdf_is_per_bin_matrix():
    """The joint SFS CDF is a per-bin :class:`JointSFS` matrix matching the mean's shape."""
    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs
    F = jsfs.cdf(1.0)
    assert np.asarray(F.data).shape == np.asarray(jsfs.mean.data).shape
    assert np.all(np.asarray(F.data) >= -1e-9) and np.all(np.asarray(F.data) <= 1 + 1e-9)


def test_two_locus_sfs_has_no_univariate_distribution():
    """A 2-SFS entry is a cross-moment (product of two rewards), so CDF/PDF/quantile/plots must raise clearly."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    for method in ('cdf', 'pdf', 'quantile', 'plot_cdf', 'plot_pdf'):
        with pytest.raises(NotImplementedError):
            getattr(sfs2, method)(1.0)


@pytest.mark.slow
def test_sfs_bin_distributions_vs_msprime():
    """Ground truth: each SFS bin's accumulated-branch-length CDF matches a fresh msprime simulation. The cached
    comparison scenarios cannot validate this (their CDF grid is tree-height-scaled, far wider than a single bin's
    support), so we simulate directly and compare the per-bin empirical CDF on a bin-appropriate grid."""
    from phasegen.comparison import Comparison

    c = Comparison(n=5, num_replicates=150000, pop_sizes={'pop_0': {0: 1.0}},
                   parallelize=True, seed=7, comparisons={'tolerance': {}})
    samples = np.asarray(c.ms.sfs.samples)  # (replicates, bins) of branch length subtending i samples

    for i in c.ph.sfs._get_indices():
        rd = c.ph.sfs.distribution(reward=UnfoldedSFSReward(i))
        t = np.linspace(0.02, float(rd.quantile(0.9)), 25)
        ph_cdf = rd.cdf_curve(t)
        ms_cdf = (samples[:, i][:, None] <= t[None, :]).mean(axis=0)  # empirical per-bin CDF
        assert np.abs(ph_cdf - ms_cdf).max() < 0.01, (i, np.abs(ph_cdf - ms_cdf).max())


@pytest.mark.slow
def test_joint_distribution_cross_moment_and_cdf_vs_msprime():
    """Ground truth via the scenario infrastructure: the joint reward distribution's cross-moment ``E[L_i L_j]``
    and joint CDF match a fresh msprime simulation, compared through the empirical SFS's cross-moment / joint-CDF
    tracking (``EmpiricalPhaseTypeSFSDistribution.cross_moment`` / ``joint_cdf``)."""
    from phasegen.comparison import Comparison

    c = Comparison(n=6, num_replicates=200000, pop_sizes={'pop_0': {0: 1.0}},
                   parallelize=True, seed=11, comparisons={'tolerance': {}})
    ph, ms = c.ph, c.ms

    for i, j in [(1, 2), (2, 3), (1, 4), (3, 3)]:
        jd = ph.sfs.joint_distribution(i, j)
        empirical_cross = ms.sfs.cross_moment(i, j)
        assert abs(jd.moment(1, 1) - empirical_cross) < 0.03 * empirical_cross + 0.01  # E[L_i L_j]

        for qa, qb in [(0.4, 0.6), (0.7, 0.5)]:
            x = float(jd.marginal('a').quantile(qa))
            y = float(jd.marginal('b').quantile(qb))
            assert abs(jd.cdf(x, y) - ms.sfs.joint_cdf(i, j, x, y)) < 0.02  # P(L_i <= x, L_j <= y)


@pytest.mark.slow
def test_jsfs_bin_distributions_self_consistent_with_mean():
    """Each jSFS per-config distribution is consistent with the (msprime-validated) jSFS mean: ``E[L_c] = mean[c]``.
    The empirical jSFS keeps only moments (not per-config branch-length samples), so direct msprime validation of
    the *distribution* is covered transitively — the per-config machinery is identical to the single-locus SFS bin
    distributions, which are validated against msprime in ``test_sfs_bin_distributions_vs_msprime``."""
    from phasegen.rewards import JointSFSReward

    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs
    mean = np.asarray(jsfs.mean.data)

    for cfg in jsfs._get_configs():
        rd = jsfs.distribution(reward=JointSFSReward(cfg))
        mean_from_lst = rd._cumulants()[0]  # E[R] = -phi'(0), central difference
        assert abs(mean_from_lst - mean[cfg]) < 1e-5, (cfg, mean_from_lst, mean[cfg])


def test_batched_accumulation_matches_serial():
    """The batched mean accumulation (shared occupation grid) equals the per-bin serial accumulation, where it
    engages (no flattening): jSFS, and a Beta-coalescent SFS."""
    from phasegen.distributions.phase_type import PhaseTypeDistribution
    from phasegen.rewards import CombinedReward, JointSFSReward
    et = np.linspace(0.0, 4.0, 12)

    # jSFS (multi-population, no flattening)
    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs
    batched = jsfs.accumulate(1, et)
    for cfg in jsfs._get_configs():
        serial = PhaseTypeDistribution.accumulate(
            jsfs, k=1, end_times=et, rewards=(CombinedReward([jsfs.reward, JointSFSReward(cfg)]),))
        np.testing.assert_allclose(batched[cfg], serial, atol=1e-10)

    # Beta-coalescent SFS (MMC does not flatten)
    sfs = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs
    batched_sfs = sfs.accumulate(1, et)
    for i in sfs._get_indices():
        np.testing.assert_allclose(batched_sfs[i], sfs.get_accumulation(1, i, et), atol=1e-10)


def test_joint_reward_distribution_within_tree():
    """The joint distribution of two SFS bins recovers the within-tree cross-moment, the marginals, and is
    consistent with the univariate LST (``lst(s, 0) == marginal_a.lst(s)``)."""
    coal = pg.Coalescent(n=5)
    cov = np.asarray(coal.sfs.cov.data)
    mean = np.asarray(coal.sfs.mean.data)
    for i, j in [(1, 1), (1, 2), (2, 3)]:
        jd = coal.sfs.joint_distribution(i, j)
        assert jd.moment(1, 1) == pytest.approx(cov[i, j] + mean[i] * mean[j], abs=1e-9)  # E[L_i L_j]
        assert jd.marginal('a')._cumulants()[0] == pytest.approx(mean[i], abs=1e-7)        # E[L_i]
        assert jd.lst(0.6, 0.0) == pytest.approx(jd.marginal('a').lst(0.6), abs=1e-12)     # combined-shift consistency
        assert jd.cov() == pytest.approx(cov[i, j], abs=1e-8)


def test_joint_distribution_2d_density_and_cdf():
    """The 2D joint density recovers the cross-moment and integrates to the continuous mass; the joint CDF
    saturates to 1 and starts at the joint atom; plotting runs. Validated on a no-atom pair (tight) and an
    atom-bearing pair (the continuous-continuous part)."""
    import matplotlib
    matplotlib.use('Agg')

    # no-atom pair: every tree has external (singleton) branches, so L_1 > 0 a.s.
    jd = pg.Coalescent(n=6).sfs.joint_distribution(1, 1)
    st = jd._cos2d

    # the joint CDF is bounded, monotone, saturates to 1, and starts at the joint atom
    xs = np.linspace(0, st['ba'], 25)
    ys = np.linspace(0, st['bb'], 25)
    G = jd._cdf_grid(xs, ys)
    assert G.min() > -2e-3 and G.max() < 1 + 2e-3
    # essentially monotone (small COS/Gibbs wiggles allowed)
    assert np.all(np.diff(G, axis=0) > -3e-3) and np.all(np.diff(G, axis=1) > -3e-3)
    assert jd.cdf(st['ba'], st['bb']) == pytest.approx(1.0, abs=1e-3)
    assert jd.cdf(0.0, 0.0) == pytest.approx(0.0, abs=1e-3)  # no atom for bin 1

    # the joint CDF's marginal equals the 1D marginal CDF exactly (the strong 2D-accuracy check)
    for y in [0.8, 1.5, 3.0]:
        assert jd.cdf(st['ba'] * 5, y) == pytest.approx(jd.marginal('b').cdf(y), abs=2e-3)

    # atom-bearing pair: bin 3 of n=6 is empty with positive probability; its atom is recovered exactly, and the
    # cross-moment is read off the density (the product kills the boundary)
    jd2 = pg.Coalescent(n=6).sfs.joint_distribution(2, 3)
    assert jd2._atoms['b0'] > 0.05
    assert jd2.cdf(jd2._cos2d['ba'] * 5, 1e-9) == pytest.approx(jd2._atoms['b0'], abs=3e-3)  # P(L_3 = 0)
    x2 = np.linspace(0, jd2._cos2d['ba'], 220)
    y2 = np.linspace(0, jd2._cos2d['bb'], 220)
    f2 = jd2._density(x2, y2)
    cross = (np.outer(x2, y2) * f2).sum() * (x2[1] - x2[0]) * (y2[1] - y2[0])
    assert abs(cross - jd2.moment(1, 1)) < 1e-2  # E[L_2 L_3]

    jd.plot_pdf(show=False)
    jd.plot_cdf(show=False, n_points=15)


def test_joint_reward_distribution_two_locus():
    """The two-locus joint distribution recovers the 2-SFS entry ``E[L^0_i L^1_j]`` and its correlation."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    mean = np.asarray(sfs2.mean.data)
    corr = np.asarray(sfs2.corr.data)
    for i, j in [(1, 1), (1, 2), (2, 2)]:
        jd = sfs2.joint_distribution(i, j)
        assert jd.moment(1, 1) == pytest.approx(mean[i, j], abs=1e-8)
        assert jd.corr() == pytest.approx(corr[i, j], rel=1e-4)


@pytest.mark.parametrize("dist_name, n_bins", [("sfs", 6), ("fsfs", 3)])
def test_plot_all_bins_pdf_cdf(dist_name, n_bins):
    """``plot_pdf`` / ``plot_cdf`` draw one curve per bin on a single axes (unfolded and folded)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dist = getattr(pg.Coalescent(n=7), dist_name)

    _, ax = plt.subplots()
    dist.plot_pdf(ax=ax, show=False)
    assert len(ax.get_lines()) == n_bins

    _, ax = plt.subplots()
    dist.plot_cdf(ax=ax, bins=[1, 2], show=False)
    assert len(ax.get_lines()) == 2
    plt.close('all')


def test_phasetype_distribution_exposes_cdf_pdf_quantile():
    """``PhaseTypeDistribution`` (e.g. total_branch_length) exposes ``cdf``/``pdf``/``quantile`` directly, like
    the tree height, delegating to the reward distribution."""
    dist = pg.Coalescent(n=6).total_branch_length
    rd = dist.distribution()
    assert dist.cdf(3.0) == pytest.approx(rd.cdf(3.0))
    assert dist.pdf(3.0) == pytest.approx(rd.pdf(3.0))
    assert dist.quantile(0.5) == pytest.approx(rd.quantile(0.5), abs=1e-6)


# ----------------------------------------------------------------------------------------------------------------
# validation / errors
# ----------------------------------------------------------------------------------------------------------------
def test_vector_reward_raises():
    """A non-scalar (per-state vector-valued) reward is rejected with a clear error."""
    from phasegen.rewards import Reward

    class _VectorReward(Reward):
        def _get(self, state_space):
            return np.ones((2, state_space.k))  # 2 values per state -> not a scalar reward

        def _supports(self, state_space):
            return True

    dist = pg.Coalescent(n=5).total_branch_length
    with pytest.raises(NotImplementedError):
        dist.distribution(reward=_VectorReward()).cdf(1.0)


def test_negative_reward_raises():
    """A negative reward is rejected."""
    from phasegen.rewards import Reward

    class _NegReward(Reward):
        def _get(self, state_space):
            return -np.ones(state_space.k)

        def _supports(self, state_space):
            return True

    dist = pg.Coalescent(n=5).total_branch_length
    with pytest.raises(ValueError):
        dist.distribution(reward=_NegReward()).cdf(1.0)
