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


@pytest.mark.slow
def test_cos_curve_matches_dehoog():
    """The fast COS plotting curves (cdf_curve / pdf_curve with method='cos') match the exact per-point de Hoog
    inversion across the cases that most stress the COS path: a smooth distribution, an SFS bin, a multiple-merger bin
    with an atom at 0 (Beta and Dirac), and a skewed/heavy-tailed expansion (the support-matched two-pass window)."""
    cases = [
        (pg.Coalescent(n=8), None),                                                  # total branch length (smooth)
        (pg.Coalescent(n=8).sfs, UnfoldedSFSReward(2)),                              # Kingman SFS bin
        (pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs, UnfoldedSFSReward(3)),       # Beta, atom at 0
        (pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.7, c=50)).sfs, UnfoldedSFSReward(3)),  # Dirac, atom at 0
        (pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs,
         UnfoldedSFSReward(3)),                                                       # heavy-tailed expansion
    ]
    for host, reward in cases:
        rd = host.total_branch_length.distribution() if reward is None else host.distribution(reward=reward)
        # compare within the bulk: from above the immediate x=0 boundary (a localized cosine artifact for strongly
        # shifted bins) up to the 0.99 quantile (the COS window is matched to ~the 0.9995 quantile)
        xs = np.linspace(0.15 * rd.quantile(0.99), rd.quantile(0.99), 40)
        peak = float(np.max(rd.pdf(xs)))
        np.testing.assert_allclose(rd.cdf.curve(xs, method='cos'), rd.cdf(xs), atol=5e-3)
        # the PDF is plotting-grade (derived from CDF differences); allow a small boundary/atom error relative to peak
        np.testing.assert_allclose(rd.pdf.curve(xs, method='cos'), rd.pdf(xs), atol=max(2e-2, 0.05 * peak))


@pytest.mark.slow
@pytest.mark.parametrize("demography", [
    None,                                                                          # single-epoch Kingman
    pg.Demography(pop_sizes={'pop_0': {0.0: 1.0, 0.5: 0.3, 1.5: 2.0}}),            # time-inhomogeneous (3-epoch)
])
def test_2d_cos_matches_dehoog(demography):
    """The fast 2D cosine joint inversion (``method='cos'``) agrees with the accurate nested de Hoog
    (``method='dehoog'``). The scenario suite validates the 2D cosine against msprime but never directly against de
    Hoog; this pins the two analytic inversions to each other. The CDF matches closely everywhere; the density matches
    in the bulk (the cosine series is biased only at the near-origin edge -- see the surface comparison, which likewise
    drops the edge)."""
    jd = pg.Coalescent(n=6, demography=demography).sfs.joint_distribution(1, 3)  # off-diagonal -> continuous 2D density
    ma, mb = jd.marginal('a'), jd.marginal('b')

    # CDF: a small grid spanning the support (incl. near the origin), where cosine is accurate everywhere
    xs, ys = np.linspace(0.0, ma.quantile(0.95), 6), np.linspace(0.0, mb.quantile(0.95), 6)
    cdf_cos = np.asarray(jd.cdf(xs, ys, method='cos'), dtype=float)
    cdf_dh = np.asarray(jd.cdf(xs, ys, method='dehoog'), dtype=float)
    assert np.abs(cdf_cos - cdf_dh).max() < 2e-2  # bounded in [0, 1] -> absolute tolerance

    # density: compare in the bulk (above the near-origin edge), relative to the de Hoog peak
    xs, ys = np.linspace(ma.quantile(0.3), ma.quantile(0.9), 6), np.linspace(mb.quantile(0.3), mb.quantile(0.9), 6)
    pdf_cos = np.asarray(jd.pdf(xs, ys, method='cos'), dtype=float)
    pdf_dh = np.asarray(jd.pdf(xs, ys, method='dehoog'), dtype=float)
    assert np.abs(pdf_cos - pdf_dh).max() < 0.1 * max(float(pdf_dh.max()), 1e-300)


def test_cos_curve_recovers_atom():
    """For an SFS bin that may be empty, the COS CDF starts at the atom ``P(R=0) = phi(inf)``."""
    rd = pg.Coalescent(n=7).sfs.distribution(reward=UnfoldedSFSReward(3))
    p0 = rd.lst(1e8).real
    assert p0 > 0.01  # this bin is empty with non-negligible probability
    # the CDF just above 0 is essentially the atom
    assert abs(rd.cdf.curve(np.array([1e-6]))[0] - p0) < 5e-3


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


def test_cdf_right_continuous_at_atom():
    """The exact per-point CDF is right-continuous at the atom: ``F(0) = P(R = 0)`` (the point mass), matching
    the de Hoog curve, rather than the left limit ``F(0-) = 0``. An SFS bin whose class has no subtending branch
    in some genealogies carries such an atom; a continuous reward (tree height) has none, so its ``F(0) = 0``."""
    bin_dist = pg.Coalescent(n=6).sfs.bin(3)  # P(L_3 = 0) > 0
    atom = float(bin_dist.cdf.curve([0.0])[0])
    assert atom > 0.05  # this bin really does have an atom
    assert float(bin_dist.cdf(0.0)) == pytest.approx(atom, abs=1e-6)  # scalar matches the curve, not 0

    th = pg.Coalescent(n=6).tree_height.distribution()  # continuous -> no atom
    assert float(th.cdf(0.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(th.cdf(-1.0)) == 0.0


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
        ph_cdf = rd.cdf.curve(t)
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

    # the fast cosine box underlies the dense CDF *plot* grid; it is bounded, monotone, and saturates to 1
    xs = np.linspace(0, st['ba'], 25)
    ys = np.linspace(0, st['bb'], 25)
    G = jd._cdf_grid(xs, ys, dehoog=False)
    assert G.min() > -2e-3 and G.max() < 1 + 2e-3
    # essentially monotone (small COS/Gibbs wiggles allowed)
    assert np.all(np.diff(G, axis=0) > -3e-3) and np.all(np.diff(G, axis=1) > -3e-3)
    assert jd.cdf(st['ba'], st['bb']) == pytest.approx(1.0, abs=3e-3)  # scale-5 window holds ~99.87% of mass
    assert jd.cdf(0.0, 0.0) == pytest.approx(0.0, abs=1e-3)  # no atom for bin 1

    # the callable CDF uses the accurate nested-de-Hoog box: its marginal equals the 1D marginal CDF (the strong
    # 2D-accuracy check, which the fixed-window cosine box fails for skewed multi-epoch rewards)
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
    assert abs(cross - jd2.moment(1, 1)) < 2e-2  # E[L_2 L_3] (FD-density grid integration, ~3% error)

    jd.pdf.plot(show=False)
    jd.cdf.plot(show=False, n_points=15)


def test_joint_plot_surface():
    """A joint distribution can be drawn as a 3D surface (``pdf.plot_surface()`` / ``cdf.plot_surface()``) as well as
    a heatmap; a univariate distribution has no surface plot (raises)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    jd = pg.Coalescent(n=6).sfs.joint_distribution(1, 2)
    for fn in (jd.pdf, jd.cdf):
        ax = fn.plot_surface(show=False)
        assert ax.name == '3d'
    plt.close('all')

    # a univariate distribution has no surface plot -- the method is simply not exposed
    assert not hasattr(pg.Coalescent(n=6).sfs.bin(2).pdf, 'plot_surface')


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


def test_bin_returns_callable_plottable_1d_distribution():
    """``sfs.bin(i)`` / ``jsfs.bin(i, j)`` return the bin's 1D branch-length distribution as a callable-and-plottable
    RewardDistribution, consistent with the per-bin spectrum; the two-locus SFS has no such per-entry 1D law."""
    import matplotlib
    matplotlib.use('Agg')
    from phasegen.distributions.base import DistributionFunction
    from phasegen.distributions.reward import RewardDistribution

    coal = pg.Coalescent(n=6)
    b = coal.sfs.bin(2)
    assert isinstance(b, RewardDistribution)
    assert isinstance(b.cdf, DistributionFunction)
    # consistent with the per-bin spectrum value, and callable + plottable
    assert b.cdf(1.3) == pytest.approx(np.asarray(coal.sfs.cdf(1.3).data)[2], abs=1e-9)
    assert b.quantile(0.5) == pytest.approx(np.asarray(coal.sfs.quantile(0.5).data)[2], abs=1e-6)
    b.pdf.plot(show=False)
    b.cdf.plot(show=False)
    b.quantile.plot(show=False)

    # the joint SFS bin takes one index per population (e.g. jsfs.bin(1, 0))
    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jb = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).jsfs.bin(1, 0)
    assert isinstance(jb, RewardDistribution) and jb.cdf(1.0) >= 0

    # the two-locus SFS entry (i, j) is a cross-moment, not a single 1D distribution
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2.bin(1, 1)


def test_bin_mean_var_std_return_scalars():
    """``sfs.bin(i).mean/var/std`` must return the scalar bin moments, not crash. Regression for the spectrum host
    overriding ``moment`` to return a whole SFS, which broke ``float()`` in RewardDistribution.mean/var."""
    coal = pg.Coalescent(n=5)
    means = np.asarray(coal.sfs.mean.data)

    for i in (1, 2, 3, 4):
        b = coal.sfs.bin(i)
        assert isinstance(b.mean, float)
        # the bin mean is exactly the per-bin spectrum mean
        assert b.mean == pytest.approx(means[i], rel=1e-9)
        assert isinstance(b.var, float) and b.var >= 0
        assert b.std == pytest.approx(b.var ** 0.5)

    # also works for a joint SFS bin (multi-population spectrum host)
    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    jb = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).jsfs.bin(1, 0)
    assert isinstance(jb.mean, float) and jb.mean >= 0
    assert isinstance(jb.var, float) and jb.var >= 0


def test_conditional_distribution():
    """``JointRewardDistribution.conditional`` returns a proper, callable-and-plottable 1D distribution; the law of
    total expectation ``E[R_b] = ∫ E[R_b | R_a = x] f_a(x) dx`` is recovered, and the self-pair / atom edge cases
    behave."""
    import matplotlib
    matplotlib.use('Agg')
    from phasegen.distributions.base import DistributionFunction

    jd = pg.Coalescent(n=6).sfs.joint_distribution(1, 2)
    ma, mb = jd.marginal('a'), jd.marginal('b')

    # a proper 1D distribution (the nested-inversion conditional is a RewardDistribution): CDF monotone 0 -> 1,
    # quantile its inverse, callable + plottable
    c = jd.conditional('a', 1.0)
    assert isinstance(c.cdf, DistributionFunction)
    grid = np.linspace(0, c.quantile(0.99), 50)
    F = c.cdf(grid)
    assert np.all(np.diff(F) > -1e-6) and F[0] == pytest.approx(0.0, abs=2e-3) and F[-1] == pytest.approx(0.99, abs=2e-2)
    assert c.cdf(c.quantile(0.5)) == pytest.approx(0.5, abs=2e-3)
    c.pdf.plot(show=False)
    c.cdf.plot(show=False)
    c.quantile.plot(show=False)

    # law of total expectation: E[R_b] = ∫ E[R_b | R_a = x] f_a(x) dx (a few conditioning points; each conditional is
    # a real nested inversion, so keep the grid small)
    xs = np.linspace(0.3, ma._range(scale=4), 6)
    cond_means = [jd.conditional('a', float(x))._cumulants()[0] for x in xs]
    e_recon = np.trapezoid(np.array(cond_means) * ma.pdf(xs), xs)
    assert e_recon == pytest.approx(mb._cumulants()[0], rel=0.15)

    # a self-pair conditional is a point mass at ``value`` -> not representable, raises
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=6).sfs.joint_distribution(2, 2).conditional('a', 1.0)


@pytest.mark.parametrize("i, j, value", [(1, 2, 1.0), (2, 4, 0.5)])
def test_conditional_support_window_covers_distribution(i, j, value):
    """A conditional sizes its support window by bracketing the exact CDF, not from finite-difference cumulants whose
    **variance collapses** for the noisy nested transform. Regression guard: that collapse used to shrink the window to
    near the mean (e.g. ``cdf(b) ~ 0.67``), truncating the distribution -- which made the cosine curve fabricate
    reaching 1 and the high quantiles wrong. The window must now span (almost) the whole support, the curve must match
    the exact de Hoog at an interior point, and the high quantile must round-trip."""
    jd = pg.Coalescent(n=6).sfs.joint_distribution(i, j)
    c = jd.conditional('a', value)

    b = c._range(12.0)
    assert float(c.cdf(b)) >= 0.999  # window spans the support (was ~0.67 with the collapsed-variance estimate)

    # the de Hoog spline curve matches the exact per-point de Hoog away from the atom
    x = 0.5 * b
    assert c.cdf.curve(np.array([x]))[0] == pytest.approx(float(c.cdf(x)), abs=5e-3)

    # the high quantile is now accurate (curve reaches it / falls back correctly): F(q_0.99) ~ 0.99
    assert c.cdf(c.quantile(0.99)) == pytest.approx(0.99, abs=1e-2)


# small state spaces (n<=4), but a spread of regimes: single-epoch, time-inhomogeneous (3-epoch) and a
# multiple-merger (Beta) model -- the nested-inversion conditional is most stressed away from the simple Kingman case
_CONDITIONAL_SCENARIOS = {
    '1epoch': lambda: pg.Coalescent(n=4),
    '3epoch': lambda: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0.0: 1.0, 0.3: 0.2, 1.0: 1.5}})),
    'beta': lambda: pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5)),
}


@pytest.mark.parametrize("scenario", list(_CONDITIONAL_SCENARIOS))
def test_conditional_law_of_total_expectation(scenario):
    """``JointRewardDistribution.check_total_expectation`` recovers ``E[R_other] = E[E[R_other|R_on]]`` for both
    conditioning axes, across single-epoch / time-inhomogeneous / multiple-merger regimes. Exercises the runtime guard
    (which logs a warning past its tolerance) and asserts the conditional means integrate back to the marginal mean."""
    jd = _CONDITIONAL_SCENARIOS[scenario]().sfs.joint_distribution(1, 2)
    rel = jd.check_total_expectation(n_points=8, tol=0.1)
    assert rel and max(rel.values()) < 0.1




def test_inversion_detectors_warn(caplog):
    """The numerical-inversion guards (``_warn_if_negative`` / ``_warn_if_nonmonotone`` methods on the distribution)
    log a warning on a substantially negative density or a non-monotone CDF (Gibbs ringing), but stay silent for
    noise-level deviations -- so a clipped/flattened curve is surfaced, not hidden."""
    import logging
    d = pg.Coalescent(n=4).sfs.bin(2)  # any distribution exposing the (inherited) guard methods
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)  # the phasegen logger does not propagate; capture it (and its children) directly
    try:
        def warned(method, arr):
            caplog.clear()
            method(np.asarray(arr, dtype=float), 'test')
            return any(r.levelno >= logging.WARNING for r in caplog.records)

        # substantial negative density -> warns; noise-level negative -> silent
        assert warned(d._warn_if_negative, [0.0, 1.0, -0.5])
        assert not warned(d._warn_if_negative, [0.0, 1.0, -1e-9])
        # non-monotone CDF (real downward step) -> warns; noise-level wiggle -> silent
        assert warned(d._warn_if_nonmonotone, [0.0, 0.5, 0.3, 1.0])
        assert not warned(d._warn_if_nonmonotone, [0.0, 0.5, 0.5 - 1e-9, 1.0])

        # the Settings.check_inversions flag silences both detectors
        pg.Settings.check_inversions = False
        try:
            assert not warned(d._warn_if_negative, [0.0, 1.0, -0.5])
            assert not warned(d._warn_if_nonmonotone, [0.0, 0.5, 0.3, 1.0])
        finally:
            pg.Settings.check_inversions = True
    finally:
        log.removeHandler(caplog.handler)


def test_clean_distribution_emits_no_inversion_warning(caplog):
    """A well-behaved distribution's CDF/PDF curves route through the detectors without false-positive warnings."""
    import logging
    log = logging.getLogger('phasegen')
    marg = pg.Coalescent(n=4).sfs.joint_distribution(1, 2).marginal('a')
    grid = np.linspace(0.0, marg._range(8.0), 50)
    log.addHandler(caplog.handler)  # the phasegen logger does not propagate; capture it directly
    try:
        caplog.clear()
        marg.cdf.curve(grid)
        marg.pdf.curve(grid)
        assert not [r for r in caplog.records if 'imprecise' in r.getMessage()]
    finally:
        log.removeHandler(caplog.handler)


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


def test_cos_inversion_imprecision_warning(caplog):
    """The COS plotting inversion (cdf_curve with method='cos') warns (via the logger) when it is likely imprecise
    (ringing it cannot resolve / window too small), and stays silent on well-behaved curves."""
    import logging

    # the package logger does not propagate to root (where caplog listens), so capture it directly
    pg_logger = logging.getLogger('phasegen')
    pg_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger='phasegen')
    try:
        d = pg.Coalescent(n=6).total_branch_length.distribution()

        # well-behaved curves must not warn (otherwise the warning is noise on every plot)
        d.cdf.curve(np.linspace(0, d._range(), 100), method='cos')
        d.pdf.curve(np.linspace(0, d._range(), 100), method='cos')
        assert 'residual ripple' not in caplog.text

        # a genuinely under-resolved case (a heavy-tailed bin whose spread-out bulk the cosine series cannot fully
        # resolve even with the support-matched window) warns (via the shared non-monotonicity guard)
        e = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs
        e.distribution(reward=e._get_sfs_reward(5)).cdf.curve(np.linspace(0, 50, 100), method='cos')
        assert 'residual ripple' in caplog.text
    finally:
        pg_logger.removeHandler(caplog.handler)


def test_plot_n_grid_setting_controls_grid_size():
    """``Settings.plot_n_grid`` sets the number of points on the default 1D plot grids (cdf / pdf / quantile),
    including the exact (de Hoog) path."""
    import matplotlib
    matplotlib.use('Agg')

    c = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10}))
    prev = pg.Settings.plot_n_grid
    try:
        for kind in ('cdf', 'pdf', 'quantile'):
            for n in (12, 31):
                pg.Settings.plot_n_grid = n
                ax = getattr(c.sfs, kind).plot(show=False)
                assert ax.lines[0].get_xdata().shape[0] == n
                ax.figure.clf()
        # the exact (de Hoog) curve uses the same grid setting
        pg.Settings.plot_n_grid = 17
        ax = c.sfs.pdf.plot(show=False, exact=True)
        assert ax.lines[0].get_xdata().shape[0] == 17
        ax.figure.clf()
    finally:
        pg.Settings.plot_n_grid = prev


def test_plot_exact_de_hoog_matches_per_point():
    """The ``exact=True`` plotting path draws the per-point de Hoog values (cdf / pdf), not the fast COS curve."""
    import matplotlib
    matplotlib.use('Agg')

    c = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10}))
    d = c.sfs.bin(2)  # a single 1D bin distribution

    x = np.linspace(0.1, d.quantile(0.9), 15)
    ax = d.cdf.plot(x=x, show=False, exact=True)
    assert np.allclose(ax.lines[-1].get_ydata(), d.cdf(x), atol=1e-8)
    ax.figure.clf()

    ax = d.pdf.plot(x=x, show=False, exact=True)
    assert np.allclose(ax.lines[-1].get_ydata(), d.pdf(x), atol=1e-8)
    ax.figure.clf()


def test_inversion_method_dehoog_spline_default():
    """The default curve representation (``method='dehoog'``) is the accurate de Hoog + monotone-spline: cdf_curve /
    pdf_curve match the per-point de Hoog closely (better than cosine) on a heavy-tailed bin that makes cosine ring,
    the CDF is monotone, and the quantile inverts the curve. The ``method='cos'`` path is still available per call."""
    d = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs.bin(5)
    xs = np.linspace(0.1 * d.quantile(0.95), d.quantile(0.95), 60)

    F = d.cdf.curve(xs)  # default method='dehoog'
    assert np.all(np.diff(F) >= -1e-9)                       # monotone CDF
    assert np.abs(F - d.cdf(xs)).max() < 2e-3               # accurate vs de Hoog (cosine rings ~1e-2 here)
    assert d.pdf.curve(xs).min() >= -1e-9                    # non-negative density (CDF-derivative)

    # the quantile inverts the cached (default de Hoog) curve and is consistent with it
    assert d.cdf.curve(np.array([d.quantile(0.5)]))[0] == pytest.approx(0.5, abs=2e-3)

    # the cosine path is still available per call
    assert np.all(np.diff(d.cdf.curve(xs, method='cos')) >= -1e-6)


@pytest.mark.slow
def test_cos_two_pass_window_monotone_and_plotting_accurate():
    """For a heavy-tailed distribution (strong size expansion) the default COS window (mean+12 std) is far wider than
    the bulk and would ring; the two-pass support-matched window keeps the plotted cdf_curve monotone and within
    plotting accuracy of the per-point de Hoog CDF, with a non-negative pdf_curve (PDF from CDF differences)."""
    sfs = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs
    for i in (1, 3, 5, 9):
        d = sfs.distribution(reward=sfs._get_sfs_reward(i))
        # evaluate within the bulk (away from the immediate x=0 boundary, where the cosine series has a localized
        # artifact for strongly shifted bins, and below the 0.99 quantile)
        x = np.linspace(0.1 * d.quantile(0.99), d.quantile(0.99), 300)
        F = d.cdf.curve(x, method='cos')
        assert np.all(np.diff(F) >= -1e-9)                  # CDF monotone
        assert np.abs(F - d.cdf(x)).max() < 1.5e-2          # plotting-grade vs de Hoog
        assert d.pdf.curve(x, method='cos').min() >= -1e-9  # PDF (from CDF differences) non-negative


def test_pdf_curve_via_cdf_differentiation_is_smooth():
    """pdf_curve (method='cos') differentiates the (stable) COS CDF instead of summing the raw cosine density, so it
    stays smooth even when the direct density rings (wide support range / heavy tail). Validated on a strong-expansion
    demography and against the per-point de Hoog density."""
    d = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).total_branch_length.distribution()
    b = d._range()
    x = np.linspace(0, b, 300)
    pdf = d.pdf.curve(x, method='cos')  # derivative of the COS CDF
    raw = d.cdf._cos(x, 'pdf')        # raw cosine density (rings)
    peak = pdf.max()

    # the differentiated PDF undershoots far less than the raw cosine density (ripples integrated out)
    assert pdf.min() / peak > -0.005
    assert pdf.min() / peak > raw.min() / raw.max()

    # and it still agrees with the exact per-point de Hoog density
    xs = np.linspace(0.05 * b, 0.5 * b, 8)
    assert np.abs(np.interp(xs, x, pdf) - d.pdf(xs)).max() < 0.03 * peak


def test_plot_endpoint_tracks_quantile_setting():
    """The default cdf/pdf plot endpoint is ``Settings.plot_endpoint_quantile``-quantile, so a heavy upper tail does
    not stretch the view to mean + many std; the setting controls it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phasegen.settings import Settings

    coal = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={0: 1, 1: 10}))
    tbl = coal.total_branch_length
    prev = Settings.plot_endpoint_quantile
    try:
        Settings.plot_endpoint_quantile = 0.95

        # spectrum (reward-curve path): endpoint ~ max bin 0.95-quantile, far below the mean+12std support range
        q = float(np.asarray(coal.sfs.quantile(0.95).data).max())
        stretched = max(coal.sfs.distribution(reward=coal.sfs._get_sfs_reward(i))._range()
                        for i in coal.sfs._get_indices())
        _, ax = plt.subplots()
        coal.sfs.pdf.plot(ax=ax, show=False)
        xmax = max(line.get_xdata().max() for line in ax.get_lines())
        assert xmax == pytest.approx(q, rel=0.1)
        assert xmax < 0.5 * stretched

        # univariate (exact path) endpoint = the configured quantile
        _, ax = plt.subplots()
        tbl.cdf.plot(ax=ax, show=False)
        assert ax.get_lines()[0].get_xdata().max() == pytest.approx(tbl.quantile(0.95), rel=0.1)
        plt.close('all')

        # raising the setting extends the view further into the tail
        endpoint_95 = tbl.quantile(0.95)
        Settings.plot_endpoint_quantile = 0.999
        _, ax = plt.subplots()
        tbl.cdf.plot(ax=ax, show=False)
        assert ax.get_lines()[0].get_xdata().max() > endpoint_95
        plt.close('all')
    finally:
        Settings.plot_endpoint_quantile = prev


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

    # spectra: the callable returns the per-bin spectrum, each bin matching its own RewardDistribution
    assert isinstance(coal.sfs.pdf, DistributionFunction)
    sfs_cdf = np.asarray(coal.sfs.cdf(1.3).data)
    assert sfs_cdf[2] == pytest.approx(float(coal.sfs.bin(2).cdf(1.3)))

    # univariate (tree height): callable scalar + plottable. The exact (expm) CDF on the function object agrees
    # with the LST reward-distribution CDF of the same (tree-height) reward.
    assert isinstance(coal.tree_height.quantile, DistributionFunction)
    assert float(coal.tree_height.cdf(1.0)) == pytest.approx(float(coal.tree_height.distribution().cdf(1.0)), abs=2e-3)
    coal.tree_height.pdf.plot(show=False)
    coal.tree_height.quantile.plot(show=False)

    # 2D joint: callable (x, y) + heatmap plot
    jd = coal.sfs.joint_distribution(1, 2)
    assert isinstance(jd.cdf, DistributionFunction)
    # the default (de Hoog) and the fast cosine box agree on the joint CDF
    assert jd.cdf(1.0, 1.0) == pytest.approx(jd.cdf(1.0, 1.0, method='cos'), abs=2e-2)
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


def test_coalescent_distribution_accessors():
    """``Coalescent.distribution(reward)`` / ``joint_distribution(ra, rb)`` are cached accessors returning the 1D /
    2D accumulated-reward distribution objects, with the state space inferred from the rewards (as for ``moment``)."""
    from phasegen.rewards import TreeHeightReward
    from phasegen.state_space import LineageCountingStateSpace, BlockCountingStateSpace

    c = pg.Coalescent(n=5)

    # 1D: houses mean / var / std + cdf / pdf / quantile; the mean equals the moment engine, the state space is the
    # lineage-counting one (tree height), and the accessor is cached per reward
    r = TreeHeightReward()
    d = c.distribution(r)
    assert c.distribution(r) is d
    assert isinstance(d.state_space, LineageCountingStateSpace)
    assert float(d.mean) == pytest.approx(float(c.moment(1, rewards=[r], center=False)))
    assert float(d.var) == pytest.approx(float(c.moment(2, rewards=[r, r], center=True)))
    assert 0.0 <= float(d.cdf(1.0)) <= 1.0
    assert float(d.pdf(1.0)) >= 0.0
    assert float(d.quantile(0.5)) == pytest.approx(float(c.tree_height.quantile(0.5)), abs=2e-3)

    # 2D: a singleton joint -- houses the marginal means, the cross-moments / cov / corr and the joint cdf / pdf;
    # SFS rewards route to the block-counting state space
    j = c.joint_distribution(UnfoldedSFSReward(1), UnfoldedSFSReward(2))
    assert c.joint_distribution(UnfoldedSFSReward(1), UnfoldedSFSReward(2)) is not None
    assert isinstance(j._host.state_space, BlockCountingStateSpace)
    assert np.shape(j.mean) == (2,)
    assert j.mean[0] == pytest.approx(c.moment(1, rewards=[UnfoldedSFSReward(1)], center=False))
    assert float(j.cov()) == pytest.approx(j.moment(1, 1) - j.moment(1, 0) * j.moment(0, 1))
    assert 0.0 <= float(j.cdf(1.0, 1.0)) <= 1.0
