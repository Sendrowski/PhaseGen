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

    # quantile and pdf are vectors too at a scalar argument
    assert np.asarray(sfs.quantile(0.5).data).shape == np.asarray(sfs.mean.data).shape
    assert np.asarray(sfs.pdf(1.0).data).shape == np.asarray(sfs.mean.data).shape

    # an array of evaluation points is vectorized -> a (len(t), n + 1) stack of per-bin spectra (e.g. sfs.pdf([1, 2]))
    for fn, pts in [(sfs.cdf, [1.0, 2.0]), (sfs.pdf, [1.0, 2.0, 3.0]), (sfs.quantile, [0.3, 0.6])]:
        arr = np.asarray(fn(pts))
        assert arr.shape == (len(pts), coal.n + 1)
        assert np.allclose(arr[0], np.asarray(fn(pts[0]).data))  # row 0 == scalar evaluation


def test_jsfs_cdf_is_per_bin_matrix():
    """The joint SFS CDF is a per-bin :class:`JointSFS` matrix matching the mean's shape."""
    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs
    F = jsfs.cdf(1.0)
    assert np.asarray(F.data).shape == np.asarray(jsfs.mean.data).shape
    assert np.all(np.asarray(F.data) >= -1e-9) and np.all(np.asarray(F.data) <= 1 + 1e-9)

    # an array of points is vectorized to a (len(t),) + shape stack
    arr = np.asarray(jsfs.cdf([1.0, 2.0]))
    assert arr.shape == (2,) + jsfs.mean.data.shape
    assert np.allclose(arr[0], np.asarray(F.data))


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

    jd.pdf.plot(show=False)
    jd.cdf.plot(show=False, n_points=15)


def test_joint_reward_distribution_two_locus():
    """The two-locus joint distribution recovers the 2-SFS entry ``E[L^0_i L^1_j]`` and its correlation."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    mean = np.asarray(sfs2.mean.data)
    corr = np.asarray(sfs2.corr.data)
    for i, j in [(1, 1), (1, 2), (2, 2)]:
        jd = sfs2.joint_distribution(i, j)
        assert jd.moment(1, 1) == pytest.approx(mean[i, j], abs=1e-8)
        assert jd.corr() == pytest.approx(corr[i, j], rel=1e-4)


def test_self_pair_joint_distribution_reduces_to_marginal():
    """A bin paired with itself is degenerate on the diagonal (``L_a = L_b`` a.s.): the joint CDF equals the
    marginal CDF at ``min(x, y)`` exactly -- including the atom at 0 -- and the 2D density is singular (raises)."""
    coal = pg.Coalescent(n=6)
    # an atom-free bin (singletons, L_1 > 0 a.s.) and an atom-bearing bin (bin 3 is empty with positive probability)
    for i in (1, 3):
        jd = coal.sfs.joint_distribution(i, i)
        assert jd._is_diagonal
        m = jd.marginal('a')
        for x, y in [(0.5, 1.3), (1.3, 0.5), (0.9, 0.9), (2.0, 0.2)]:
            assert jd.cdf(x, y) == pytest.approx(m.cdf(min(x, y)), abs=1e-9)
        # at min(x, y) == 0 the CDF is the atom P(L_i = 0), not the (unreliable) de Hoog inversion at t = 0
        assert jd.cdf(0.0, 1.0) == pytest.approx(jd._atoms['both0'], abs=1e-12)
        # the joint law lives on the diagonal and has no 2D density
        with pytest.raises(NotImplementedError):
            jd.pdf(1.0, 1.0)

    # the atom-bearing self-pair actually carries mass at 0
    assert coal.sfs.joint_distribution(3, 3)._atoms['both0'] > 0.05


def test_jsfs_joint_distribution_recovers_marginals_and_cross_moment():
    """``JointSFSDistribution.joint_distribution`` is the within-tree bivariate object behind the multi-population
    SFS cross-moment: its marginals match the joint SFS mean, its ``(1, 1)`` moment is a positive cross-moment, and
    a config paired with itself is the singular diagonal."""
    dem = pg.Demography(
        pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.5}},
        migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
    )
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).jsfs
    mean = np.asarray(jsfs.mean.data)

    ca, cb = (1, 0), (0, 1)
    jd = jsfs.joint_distribution(ca, cb)
    assert jd.marginal('a')._cumulants()[0] == pytest.approx(mean[ca], rel=1e-6)
    assert jd.marginal('b')._cumulants()[0] == pytest.approx(mean[cb], rel=1e-6)
    assert jd.moment(1, 1) > 0
    assert -1.0 <= jd.corr() <= 1.0
    assert jsfs.joint_distribution(ca, ca)._is_diagonal


def test_conditional_distribution():
    """``JointRewardDistribution.conditional`` returns a proper, callable-and-plottable 1D distribution; the law of
    total expectation ``E[R_b] = ∫ E[R_b | R_a = x] f_a(x) dx`` is recovered, and the self-pair / atom edge cases
    behave."""
    import matplotlib
    matplotlib.use('Agg')
    from phasegen.distributions.base import DistributionFunction

    jd = pg.Coalescent(n=6).sfs.joint_distribution(1, 2)
    ma, mb = jd.marginal('a'), jd.marginal('b')

    # a proper distribution: CDF monotone from ~0 to 1, quantile is its inverse, callable + plottable
    c = jd.conditional('a', 1.0)
    assert isinstance(c.cdf, DistributionFunction)
    grid = np.linspace(0, c._x_max, 50)
    F = c.cdf(grid)
    assert np.all(np.diff(F) > -1e-6) and F[0] == pytest.approx(c.atom0, abs=2e-3) and F[-1] == pytest.approx(1, abs=2e-3)
    assert c.cdf(c.quantile(0.5)) == pytest.approx(0.5, abs=1e-3)
    c.pdf.plot(show=False)
    c.cdf.plot(show=False)
    c.quantile.plot(show=False)

    # law of total expectation: integrate the conditional mean against the marginal density of R_a
    xs = np.linspace(0.05, ma._range(scale=8), 60)
    cond_means = [np.trapezoid(np.linspace(0, jd.conditional('a', x)._x_max, 400)
                               * jd.conditional('a', x)._pdf(np.linspace(0, jd.conditional('a', x)._x_max, 400)),
                               np.linspace(0, jd.conditional('a', x)._x_max, 400)) for x in xs]
    e_recon = np.trapezoid(np.array(cond_means) * ma.pdf(xs), xs)
    assert e_recon == pytest.approx(mb._cumulants()[0], rel=0.05)

    # a self-pair conditional is a point mass at ``value`` -> not representable, raises
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=6).sfs.joint_distribution(2, 2).conditional('a', 1.0)


@pytest.mark.parametrize("dist_name, n_bins", [("sfs", 6), ("fsfs", 3)])
def test_plot_all_bins_pdf_cdf(dist_name, n_bins):
    """``pdf.plot`` / ``cdf.plot`` / ``quantile.plot`` draw one curve per bin on a single axes (unfolded/folded)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dist = getattr(pg.Coalescent(n=7), dist_name)

    _, ax = plt.subplots()
    dist.pdf.plot(ax=ax, show=False)
    assert len(ax.get_lines()) == n_bins

    _, ax = plt.subplots()
    dist.cdf.plot(ax=ax, bins=[1, 2], show=False)
    assert len(ax.get_lines()) == 2

    _, ax = plt.subplots()
    dist.quantile.plot(ax=ax, bins=[1, 2], show=False)
    assert len(ax.get_lines()) == 2
    plt.close('all')


def test_empirical_sfs_distribution_functions_plot():
    """Regression: the empirical (msprime) SFS exposes the same callable-and-plottable pdf/cdf/quantile as the
    analytic one (one curve per polymorphic bin) -- e.g. ``MsprimeCoalescent(...).sfs.pdf.plot()`` -- including on a
    multi-epoch decline demography. This used to raise because the SFS samples are per-bin (2-D)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phasegen.distributions.empirical import MsprimeCoalescent

    coal = MsprimeCoalescent(parallelize=False, n=10,
                             demography=pg.Demography(pop_sizes={0: 1, 1: 0.01}), num_replicates=200)

    for kind in ('pdf', 'cdf', 'quantile'):
        _, ax = plt.subplots()
        getattr(coal.sfs, kind).plot(ax=ax, show=False)
        assert len(ax.get_lines()) == coal.n - 1  # one curve per polymorphic bin (MsprimeCoalescent exposes ``n``)
    plt.close('all')

    # the empirical density must not spike from over-fine histogram binning (a sane, sample-adaptive bin count)
    _, ax = plt.subplots()
    pg_coal = MsprimeCoalescent(parallelize=False, n=10, demography=pg.Demography(pop_sizes={0: 1}),
                                num_replicates=500)
    pg_coal.sfs.pdf.plot(ax=ax, show=False)
    assert max(line.get_ydata().max() for line in ax.get_lines()) < 50  # not the ~1200 of 10000-bin histograms
    plt.close('all')


def test_cos_inversion_imprecision_warning():
    """The COS plotting inversion warns when it is likely imprecise (support window too small / Gibbs ringing), and
    stays silent on well-behaved curves."""
    import warnings

    d = pg.Coalescent(n=6).total_branch_length.distribution()

    # well-behaved curves must not warn (otherwise the warning is noise on every plot)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        d.cdf_curve(np.linspace(0, d._range(), 100))
        d.pdf_curve(np.linspace(0, d._range(), 100))

    # a deliberately tiny support window truncates the tail -> imprecision warning
    with pytest.warns(UserWarning, match="COS inversion looks imprecise"):
        d._cos(np.linspace(0, 1, 20), 'pdf', n_terms=16, scale=0.5)


def test_pdf_curve_via_cdf_differentiation_is_smooth():
    """pdf_curve differentiates the (stable) COS CDF instead of summing the raw cosine density, so it stays smooth
    even when the direct density rings (wide support range / heavy tail). Validated on a strong-expansion
    demography and against the per-point de Hoog density."""
    d = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).total_branch_length.distribution()
    b = d._range()
    x = np.linspace(0, b, 300)
    pdf = d.pdf_curve(x)          # derivative of the COS CDF
    raw = d._cos(x, 'pdf')        # raw cosine density (rings)
    peak = pdf.max()

    # the differentiated PDF undershoots far less than the raw cosine density (ripples integrated out)
    assert pdf.min() / peak > -0.005
    assert pdf.min() / peak > raw.min() / raw.max()

    # and it still agrees with the exact per-point de Hoog density
    xs = np.linspace(0.05 * b, 0.5 * b, 8)
    assert np.abs(np.interp(xs, x, pdf) - d.pdf(xs)).max() < 0.03 * peak


def test_parallelize_spawn_guard_message(monkeypatch):
    """When the worker pool cannot bootstrap (e.g. a non-import-safe entry point under the macOS 'spawn' start
    method), ``parallelize`` replaces the opaque multiprocessing error with actionable guidance."""
    from phasegen import utils

    class _FakePool:
        def __enter__(self):
            raise RuntimeError("An attempt has been made to start a new process before the current process has "
                               "finished its bootstrapping phase.")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(utils.mp, 'get_context', lambda *a, **k: type('Ctx', (), {'Pool': lambda self: _FakePool()})())

    with pytest.raises(RuntimeError, match="import-safe"):
        utils.parallelize(func=lambda x: x, data=[1, 2, 3], parallelize=True, pbar=False)


def test_distribution_functions_are_callable_and_plottable():
    """``pdf``/``cdf``/``quantile`` are :class:`DistributionFunction`s: calling them evaluates (unchanged), and they
    expose ``.plot()``. The former ``plot_pdf``/``plot_cdf`` still work but warn (deprecated)."""
    import matplotlib
    matplotlib.use('Agg')
    from phasegen.distributions.base import DistributionFunction

    coal = pg.Coalescent(n=5)

    # spectra: callable returns the per-bin spectrum, identical to the underlying evaluation
    assert isinstance(coal.sfs.pdf, DistributionFunction)
    assert np.allclose(np.asarray(coal.sfs.cdf(1.3).data), np.asarray(coal.sfs._cdf(1.3).data))

    # univariate (tree height): callable scalar + plottable
    assert isinstance(coal.tree_height.quantile, DistributionFunction)
    assert coal.tree_height.cdf(1.0) == pytest.approx(coal.tree_height._cdf(1.0))
    coal.tree_height.pdf.plot(show=False)
    coal.tree_height.quantile.plot(show=False)

    # 2D joint: callable (x, y) + heatmap plot
    jd = coal.sfs.joint_distribution(1, 2)
    assert isinstance(jd.cdf, DistributionFunction)
    assert jd.cdf(1.0, 1.0) == pytest.approx(jd._cdf(1.0, 1.0))
    jd.pdf.plot(show=False)

    # the two-locus distribution intentionally has no univariate cdf/pdf/quantile
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2.cdf(1.0)

    # deprecated aliases still work but warn
    with pytest.warns(DeprecationWarning):
        coal.sfs.plot_pdf(show=False)
    with pytest.warns(DeprecationWarning):
        jd.plot_cdf(show=False, n_points=12)


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
