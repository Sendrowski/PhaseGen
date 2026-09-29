"""
Tests for :class:`phasegen.distributions.reward.RewardDistribution` — the full distribution (CDF/PDF/quantile)
of an accumulated reward, obtained from the Laplace-Stieltjes transform and its numerical (de Hoog) inversion.

The references are *exact*, not simulated: for a single epoch the accumulated reward is phase-type with the
reward-transformed generator ``diag(1/r) T`` (with the zero-reward states censored), and the multi-epoch tree
height equals PhaseGen's own (matrix-exponential) ``tree_height.cdf``. The sparse path (block-triangular LU of the
last-epoch solve) is pinned against the dense path. msprime ground truth is exercised separately through
the comparison scenarios (``total_branch_length`` CDF/PDF in the configs).
"""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import phasegen as pg
from phasegen.settings import Settings
from phasegen.rewards import UnfoldedSFSReward


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


def _single_epoch_reference_pdf(dist, reward):
    """Exact single-epoch density of the same reward-transformed phase-type law: ``f(x) = beta exp(A x) a``, with the
    exit vector ``a = -A 1`` (the derivative of :func:`_single_epoch_reference_cdf`)."""
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
    exit_rates = -A @ np.ones(len(P))
    return lambda x: float(beta @ sla.expm(A * x) @ exit_rates)


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
        got = rd.cdf._cdf_point(x)  # the exact per-point inversion is what is pinned against the reference
        assert abs(got - ref(x)) < 1e-5, (n, x, got, ref(x))


@pytest.mark.parametrize("i", [1, 2, 3])
def test_single_epoch_sfs_bin_matches_censored_reward_transform(i):
    """An SFS bin reward has zero-reward states: inverted CDF equals the censored reward-transform CDF (exact)."""
    n = 7
    dist = pg.Coalescent(n=n).sfs
    reward = UnfoldedSFSReward(i)
    rd = dist.distribution(reward=reward)
    ref = _single_epoch_reference_cdf(dist, reward)

    for x in [0.3, 0.8, 1.5, 3.0]:
        got = rd.cdf._cdf_point(x)
        assert abs(got - ref(x)) < 1e-5, (i, x, got, ref(x))


@pytest.mark.parametrize("n", [4, 6])
def test_single_epoch_density_matches_reward_transform(n):
    """The per-point de Hoog *density* equals the exact reward-transform phase-type density (it is the reference the
    cosine density is validated against, so it needs an independent reference of its own). Checked for an
    all-positive reward and for an SFS bin, whose zero-reward states are censored."""
    for dist, reward in [(pg.Coalescent(n=n).total_branch_length, None),
                         (pg.Coalescent(n=n).sfs, UnfoldedSFSReward(2))]:
        reward = dist.reward if reward is None else reward
        rd = dist.distribution() if isinstance(dist, pg.distributions.phase_type.PhaseTypeDistribution) \
            and reward is dist.reward else dist.distribution(reward=reward)
        ref = _single_epoch_reference_pdf(dist, reward)

        for x in [0.4, 1.0, 2.0, 4.0]:
            got = float(rd.pdf._pdf_point(x))
            assert abs(got - ref(x)) < 1e-5, (n, x, got, ref(x))


def test_multi_epoch_tree_height_matches_phasegen():
    """The inverted tree-height CDF equals PhaseGen's matrix-exponential ``tree_height.cdf`` (3 epochs), away from
    the epoch boundaries, where numerical Laplace inversion has a small Gibbs-type error at the CDF's kink."""
    demo = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.4: 0.25, 1.0: 2.0}})
    coal = pg.Coalescent(n=6, demography=demo)
    rd = coal.tree_height.distribution()

    for x in [0.2, 0.7, 1.3, 2.0, 3.5]:  # deliberately not 0.4 / 1.0 (epoch boundaries)
        got = rd.cdf._cdf_point(x)
        assert abs(got - float(coal.tree_height.cdf(x))) < 1e-5, (x, got, float(coal.tree_height.cdf(x)))


# ----------------------------------------------------------------------------------------------------------------
# implementation paths and invariants
# ----------------------------------------------------------------------------------------------------------------
def test_sparse_path_matches_dense():
    """Forcing the sparse path (complex block-triangular LU of the last-epoch solve) matches the dense path."""
    def cdf_values():
        return np.array([pg.Coalescent(n=8).total_branch_length.distribution().cdf(x) for x in [1.0, 3.0, 6.0]])

    Settings.closed_form_sparse_min_states = 10 ** 9
    dense = cdf_values()
    Settings.closed_form_sparse_min_states = 0
    sparse = cdf_values()

    np.testing.assert_allclose(sparse, dense, atol=1e-9)


def test_multi_epoch_total_branch_length_sparse_matches_dense():
    """Sparse vs dense also agree on a multi-epoch model, whose finite epochs are exponentiated densely either way."""
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
def test_cos_matches_dehoog():
    """The default (cosine) cdf / pdf match the exact per-point de Hoog inversion across the cases that most stress
    the cosine path: a smooth distribution, an SFS bin, a multiple-merger bin
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
        peak = float(np.max([rd.pdf._pdf_point(float(x)) for x in xs]))
        np.testing.assert_allclose(rd.cdf(xs), [rd.cdf._cdf_point(float(x)) for x in xs], atol=5e-3)
        # the PDF is derived from CDF differences; allow a small boundary/atom error relative to the peak
        np.testing.assert_allclose(rd.pdf(xs), [rd.pdf._pdf_point(float(x)) for x in xs],
                               atol=max(2e-2, 0.05 * peak))


def test_cos_curve_recovers_atom():
    """For an SFS bin that may be empty, the COS CDF starts at the atom ``P(R=0) = phi(inf)``."""
    rd = pg.Coalescent(n=7).sfs.distribution(reward=UnfoldedSFSReward(3))
    p0 = rd.lst(np.inf).real
    assert p0 > 0.01  # this bin is empty with non-negligible probability
    # the CDF just above 0 is essentially the atom
    assert abs(float(rd.cdf(1e-6)) - p0) < 5e-3


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
    atom = float(bin_dist.cdf(0.0))
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


def test_sfs_bin_combines_spectrum_reward_and_is_cached():
    """``sfs.bin(i)`` accumulated the bin reward alone, so on a marginal view such as ``sfs.demes['pop_0']`` it
    returned the all-deme bin while the per-bin ``cdf`` and the moments of that view were deme-restricted, and every
    call built a new distribution and cosine fit. The bin combines the reward of the spectrum, agrees with the per-bin
    functions of the view, differs from the all-deme bin and is cached per bin."""
    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    sfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).sfs
    deme = sfs.demes['pop_0']
    x, q, i = np.array([0.3, 1.0, 2.5]), np.array([0.2, 0.5, 0.9]), 1

    b = deme.bin(i)
    assert deme.bin(i) is b

    np.testing.assert_allclose(b.cdf(x), np.asarray(deme.cdf(x))[:, i], atol=1e-12)
    np.testing.assert_allclose(b.quantile(q), np.asarray(deme.quantile(q))[:, i], atol=1e-12)
    assert b.mean == pytest.approx(float(deme.mean.data[i]), rel=1e-8)

    full = sfs.bin(i)
    assert abs(b.mean - full.mean) > 1e-2
    assert np.abs(b.cdf(x) - full.cdf(x)).max() > 1e-2


def test_jsfs_functions_cache_bin_distributions_and_vectorise():
    """The per-bin ``cdf``, ``pdf`` and ``quantile`` of the joint SFS built a new distribution for every bin on every
    call, so each call refitted every cosine expansion, and they evaluated an array point by point. The per-bin
    distributions are cached on the spectrum and shared with ``bin()``, and an array evaluation equals the pointwise
    evaluation on a fresh spectrum."""
    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    make = lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs
    jsfs = make()
    x, q = [0.5, 1.5], [0.2, 0.8]

    F = np.asarray(jsfs.cdf(x))
    cached = dict(jsfs.__dict__['_bin_distributions'])
    assert set(cached) == {tuple(int(v) for v in c) for c in jsfs._get_configs()}

    Q = np.asarray(jsfs.quantile(q))
    jsfs.pdf(x)
    assert all(jsfs.__dict__['_bin_distributions'][c] is d for c, d in cached.items())
    assert all(jsfs.bin(*c) is d for c, d in cached.items())

    fresh = make()
    np.testing.assert_allclose(F, [np.asarray(fresh.cdf(v).data) for v in x], atol=1e-10)
    np.testing.assert_allclose(Q, [np.asarray(fresh.quantile(v).data) for v in q], atol=1e-10)


def test_two_locus_sfs_has_no_univariate_distribution():
    """A 2-SFS entry is a cross-moment (product of two rewards), so CDF/PDF/quantile/plots must raise clearly.
    Regression: ``sfs2.cdf.plot()`` raised AttributeError."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    for method in ('cdf', 'pdf', 'quantile', 'plot_cdf', 'plot_pdf'):
        with pytest.raises(NotImplementedError):
            getattr(sfs2, method)(1.0)

    for kind in ('cdf', 'pdf', 'quantile'):
        with pytest.raises(NotImplementedError):
            getattr(sfs2, kind).plot(show=False)


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
        ph_cdf = rd.cdf(t)
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
        np.testing.assert_allclose(batched[(slice(None),) + cfg], serial, atol=1e-10)

    # Beta-coalescent SFS (MMC does not flatten)
    sfs = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs
    batched_sfs = sfs.accumulate(1, et)
    for i in sfs._get_indices():
        np.testing.assert_allclose(batched_sfs[:, i], sfs.get_accumulation(1, i, et), atol=1e-10)


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
        assert jd.cov == pytest.approx(cov[i, j], abs=1e-8)


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
    G = jd._cdf_grid(xs, ys)
    assert G.min() > -2e-3 and G.max() < 1 + 2e-3
    # essentially monotone (small COS/Gibbs wiggles allowed)
    assert np.all(np.diff(G, axis=0) > -3e-3) and np.all(np.diff(G, axis=1) > -3e-3)
    assert jd.cdf(st['ba'], st['bb']) == pytest.approx(1.0, abs=3e-3)  # scale-5 window holds ~99.87% of mass
    assert jd.cdf(0.0, 0.0) == pytest.approx(0.0, abs=1e-3)  # no atom for bin 1

    # the callable CDF's marginal (one axis pushed past the window) equals the 1D marginal CDF
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

    # the density plot needs a non-diagonal pair: bins (1, 1) are the same reward, so the law is singular on the
    # diagonal and has no 2D density (jd.pdf raises). The CDF is well-defined there and still plots.
    jd2.pdf.plot(show=False)
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
        assert jd.corr == pytest.approx(corr[i, j], rel=1e-4)


def test_self_pair_joint_distribution_reduces_to_marginal():
    """A bin paired with itself is degenerate on the diagonal (``L_a = L_b`` a.s.): the joint CDF equals the
    marginal CDF at ``min(x, y)`` exactly -- including the atom at 0 -- and the 2D density is singular (raises)."""
    coal = pg.Coalescent(n=6)
    # an atom-free bin (singletons, L_1 > 0 a.s.) and an atom-bearing bin (bin 3 is empty with positive probability)
    for i in (1, 3):
        jd = coal.sfs.joint_distribution(i, i)
        assert jd._ratio == 1.0
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
    assert -1.0 <= jd.corr <= 1.0
    assert jsfs.joint_distribution(ca, ca)._ratio == 1.0


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
    assert float(c.cdf(x)) == pytest.approx(float(c.cdf._cdf_point(x)), abs=5e-3)

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


def test_nonmonotone_detector_catches_a_cumulative_sag(caplog):
    """A CDF sagging below its running maximum through many small downward steps warns although no single step
    passes the noise band. Regression: the detector tested only the largest single step, so a ripple spread over the
    grid of the cosine expansion was flattened by the monotone clamp without notice."""
    import logging
    d = pg.Coalescent(n=4).sfs.bin(2)
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)
    try:
        # twenty steps of 2e-4 each: every step is below rtol = 1e-3 of the range, the sag of 4e-3 is above
        cdf = np.concatenate([np.linspace(0.0, 0.5, 50), 0.5 - 2e-4 * np.arange(1, 21), np.linspace(0.5, 1.0, 50)])
        assert -np.diff(cdf).min() < 1e-3
        caplog.clear()
        d._warn_if_nonmonotone(cdf, 'test')
        assert any('sag' in r.getMessage() for r in caplog.records)
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
        marg.cdf(grid)
        marg.pdf(grid)
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
    """The cosine inversion warns (via the logger) when it is likely imprecise (ringing, or a body too narrow
    against the window for the terms summed), and stays silent on well-behaved curves."""
    import logging

    # the package logger does not propagate to root (where caplog listens), so capture it directly
    pg_logger = logging.getLogger('phasegen')
    pg_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger='phasegen')
    try:
        d = pg.Coalescent(n=6).total_branch_length.distribution()

        # well-behaved curves must not warn (otherwise the warning is noise on every plot)
        d.cdf(np.linspace(0, d._range(), 100))
        d.pdf(np.linspace(0, d._range(), 100))
        assert 'residual ripple' not in caplog.text
        assert 'truncation' not in caplog.text

        # a heavy-tailed bin whose body is narrow against the window the tail forces, which the terms summed at the
        # default cannot resolve
        pg.Settings.cos_terms = 64  # the autouse fixture restores it
        e = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs
        e.distribution(reward=e._get_sfs_reward(5)).cdf(np.linspace(0, 50, 100))
        assert 'truncation' in caplog.text
        assert 'Settings.cos_terms' in caplog.text
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
    finally:
        pg.Settings.plot_n_grid = prev


def test_plot_draws_the_function_it_evaluates():
    """A plotted curve is the very function the caller evaluates -- there is no second, plot-only approximation.

    The quantile is the one that used to break this: its plot inverted a private 512-point CDF curve over the cumulant
    window, which is neither the grid the cosine fit uses nor the de Hoog far tail, so ``quantile.plot()`` drew a
    different function from the one ``quantile(q)`` returns.
    """
    import matplotlib
    matplotlib.use('Agg')

    c = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10}))
    d = c.sfs.bin(2)

    x = np.linspace(0.1, float(d.quantile(0.9)), 15)
    for kind in ('cdf', 'pdf'):
        ax = getattr(d, kind).plot(t=x, show=False)
        assert np.allclose(ax.lines[-1].get_ydata(), getattr(d, kind)(x), rtol=1e-12, atol=1e-12)
        ax.figure.clf()

    q = np.linspace(0.1, 0.9, 15)
    ax = d.quantile.plot(q=q, show=False)
    assert np.allclose(ax.lines[-1].get_ydata(), d.quantile(q), rtol=1e-12, atol=1e-12)
    ax.figure.clf()

    # and the same through the spectrum-level (aggregate) plot path, which had its own quantile implementation
    ax = c.sfs.quantile.plot(q=q, show=False)
    drawn = {round(float(line.get_ydata()[0]), 12) for line in ax.lines}
    expected = {round(float(c.sfs.bin(i).quantile(q[0])), 12) for i in (1, 2, 3)}
    assert expected <= drawn
    ax.figure.clf()


def test_cos_default_matches_per_point_dehoog():
    """The default inversion (cosine) matches the exact per-point de Hoog on a heavy-tailed bin that stresses it: the
    CDF is monotone and accurate, the density is non-negative, and the quantile inverts the same representation the
    CDF is read from (so cdf and quantile are mutually consistent)."""
    d = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs.bin(5)
    xs = np.linspace(0.1 * d.quantile(0.95), d.quantile(0.95), 60)

    F = d.cdf(xs)
    assert np.all(np.diff(F) >= -1e-9)                       # monotone CDF
    assert np.abs(F - [d.cdf._cdf_point(float(x)) for x in xs]).max() < 2e-3
    assert d.pdf(xs).min() >= -1e-9                          # non-negative density (CDF-derivative)

    assert float(d.cdf(float(d.quantile(0.5)))) == pytest.approx(0.5, abs=2e-3)


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
        F = d.cdf(x)
        assert np.all(np.diff(F) >= -1e-9)                     # CDF monotone
        assert np.abs(F - [d.cdf._cdf_point(float(v)) for v in x]).max() < 1.5e-2
        assert d.pdf(x).min() >= -1e-9                         # PDF (from CDF differences) non-negative


def test_pdf_via_cdf_differentiation_is_smooth():
    """The cosine pdf differentiates the (stable) cosine CDF instead of summing the raw cosine density, so it
    stays smooth even when the direct density rings (wide support range / heavy tail). Validated on a strong-expansion
    demography and against the per-point de Hoog density."""
    d = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).total_branch_length.distribution()
    b = d._range()
    x = np.linspace(0, b, 300)
    pdf = d.pdf(x)  # derivative of the COS CDF
    fit = d.cdf._cos_coeffs
    raw = fit['fk'] @ np.cos(np.outer(fit['w'], np.minimum(x, fit['b'])))  # raw cosine density (rings)
    assert fit['p0'] <= 1e-9  # no atom to split off
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
    assert float(j.cov) == pytest.approx(j.moment(1, 1) - j.moment(1, 0) * j.moment(0, 1))
    assert 0.0 <= float(j.cdf(1.0, 1.0)) <= 1.0


@pytest.mark.parametrize('scale', [1e-6, 1e7])
def test_conditional_on_atom_is_scale_invariant(scale):
    """Rescaling every population size and epoch boundary by a constant leaves every dimensionless quantity
    unchanged: it is the same coalescent in different units. Conditioning on the atom {R_i = 0} takes the limit
    s -> inf of the sub-transform, which must not depend on the time scale. Regression: a probe at a fixed large s
    was not in the limit on a small-N demography, and the two demographies below disagreed by 0.75%.
    """
    def dimensionless(s: float) -> float:
        demography = pg.Demography(pop_sizes={'pop_0': {0: 1.0 * s, 1.0 * s: 5.0 * s}})
        joint = pg.Coalescent(n=5, demography=demography).sfs.joint_distribution(4, 1)

        # P(R_4 = 0) = 0.5 for n = 5, so the atom conditional is a real path, not a corner case
        assert float(joint._atoms['a0']) == pytest.approx(0.5, abs=1e-6)

        # E[R_1 | R_4 = 0], made dimensionless by the unconditional mean of the same reward
        return float(joint.conditional('a', 0.0).mean) / abs(float(joint.marginal('b').mean))

    assert dimensionless(scale) == pytest.approx(dimensionless(1.0), rel=1e-6)


@pytest.mark.parametrize('label, coal', [
    ('1epoch', lambda: pg.Coalescent(n=5)),
    ('3epoch', lambda: pg.Coalescent(n=5, demography=pg.Demography(
        pop_sizes={'pop_0': {0.0: 1.0, 0.3: 0.2, 1.0: 1.5}}))),
    ('beta', lambda: pg.Coalescent(n=5, model=pg.BetaCoalescent(alpha=1.5))),
])
def test_atom_conditional_matches_the_sampler_exactly(label, coal):
    """
    Conditioning on the atom is the one conditional a sampler validates **exactly**: ``{R_a = 0}`` is a
    positive-probability event, so the replicates with an empty bin *are* the conditioning set -- no window, no
    bandwidth, none of the O(h) bias that makes a sampled ``R_b | R_a = v`` a rough check at best (see
    :class:`~phasegen.distributions.EmpiricalJointDistribution`).

    Worth pinning because nothing else does: every scenario's conditional check places its conditioning points at
    ``quantile(p0 + (1 - p0) u)``, strictly *above* the atom, so ``value = 0`` -- a different class
    (``_AtomConditional``, whose transform is an exact closed-form ratio rather than a nested inversion) -- is never
    exercised there.
    """
    coal = coal()
    jd = coal.sfs.joint_distribution(4, 1)  # P(R_4 = 0) is substantial for n = 5
    cond = jd.conditional('a', 0.0)

    samples = np.asarray(coal.sfs.sample(500_000))
    empty = samples[:, 4] == 0.0  # the atom event {R_4 = 0}, selected exactly
    other = samples[empty, 1]

    # the atom's mass itself
    assert float(jd._atoms['a0']) == pytest.approx(empty.mean(), abs=5e-3)

    # E[R_1 | R_4 = 0] against the sample mean over the empty-bin replicates
    se = other.std() / np.sqrt(other.size)
    assert float(cond.mean) == pytest.approx(other.mean(), abs=max(6 * se, 1e-3 * other.mean()))

    # the variance, which on the atom is the *only* route (the derivative identity divides by the continuous
    # conditioning density and cannot be evaluated at 0), so the sample is the one thing that pins it
    assert float(cond.var) == pytest.approx(other.var(), rel=0.02)
    assert float(cond.moment(2)) == pytest.approx((other ** 2).mean(), rel=0.02)

    # the whole CDF, not just the mean: the exact atom conditional against the empirical one over the same replicates
    grid = np.linspace(0, float(cond.quantile(0.95)), 25)
    ecdf = (other[:, None] <= grid[None, :]).mean(axis=0)
    assert np.abs(np.asarray(cond.cdf(grid)) - ecdf).max() < 0.02


def test_atom_conditional_refuses_the_derivative_identity():
    """The derivative identity normalises by the conditioning marginal's *continuous* density, which at 0 is not the
    atom's mass, so it does not describe the atom conditional. It must refuse rather than quietly divide by whatever
    the density inverts to there."""
    jd = pg.Coalescent(n=5).sfs.joint_distribution(4, 1)
    cond = jd.conditional('a', 0.0)

    with pytest.raises(NotImplementedError, match='atom'):
        cond._raw_moments(k=2)

    # and the higher raw moments are likewise unavailable there, while the first two are not
    with pytest.raises(NotImplementedError):
        cond.moment(3)

    assert cond.moment(2) == pytest.approx(float(cond.var) + float(cond.mean) ** 2)


def test_windowed_conditional_mean_cancels_the_window_bias():
    """The empirical conditional keeps the replicates in a window around the conditioning value, so weighting them
    equally (Nadaraya-Watson) is ``O(h)``-biased wherever the conditional mean has slope in ``v``: the conditioning
    values are not symmetric inside the window. The local-linear fit cancels that term.

    Constructed so the truth is known: ``E[R_b | R_a = v] = v`` exactly, with ``R_a`` drawn from a distribution whose
    density falls off steeply, so a symmetric window holds far more replicates below ``v`` than above and the plain
    window mean must undershoot.
    """
    rng = np.random.default_rng(42)
    a = rng.exponential(1.0, 400_000)
    b = a + rng.normal(0.0, 0.1, a.size)  # E[R_b | R_a = v] = v

    jd = pg.distributions.EmpiricalJointDistribution(a, b)

    v, h = 0.5, 0.3
    cond = jd.conditional('a', v, window=h)

    plain = float(np.mean(cond.samples))  # what a Nadaraya-Watson estimate would give
    assert abs(plain - v) > 0.01  # the window bias is real and not negligible at this bandwidth
    assert float(cond.mean) == pytest.approx(v, abs=0.005)  # the local-linear intercept removes it


def test_conditional_moments_live_on_the_conditional():
    """The conditional's mean has two independent routes -- the central difference of its own (nested) transform, and
    the derivative identity on the joint transform, which the mean reports -- and they must agree. Both are reached
    from the conditional itself."""
    jd = pg.Coalescent(n=5).sfs.joint_distribution(4, 1)
    v = float(jd.marginal('a').quantile(0.5 + 0.5 * float(jd._atoms['a0'])))
    cond = jd.conditional('a', v)

    assert float(cond._cumulants()[0]) == pytest.approx(float(cond.mean), rel=1e-3)

    m1, m2 = cond._raw_moments(k=2)
    assert float(cond.moment(2)) == pytest.approx(m2, rel=1e-9)


def test_conditional_variance_uses_the_mean_it_reports():
    """``ConditionalRewardDistribution.var`` subtracted the squared first moment of the derivative identity while
    ``mean`` and ``moment(1)`` reported the cumulant of the nested transform, so ``var`` differed from
    ``moment(2) - moment(1) ** 2`` by the gap between the two first-moment routes. The invariant must hold on a value
    and on the atom."""
    jd = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={0: 1, 0.5: 0.2})).sfs.joint_distribution(4, 1)
    v = float(jd.marginal('a').quantile(0.5 + 0.5 * float(jd._atoms['a0'])))

    for cond in (jd.conditional('a', v), jd.conditional('a', 0.0)):
        assert float(cond.moment(1)) == float(cond.mean)
        assert float(cond.var) == pytest.approx(cond.moment(2) - cond.moment(1) ** 2, rel=1e-12, abs=1e-15)


def test_joint_cdf_vanishes_below_the_origin():
    """``JointCDF`` integrated the cosine box at negative thresholds, whose antiderivatives are negative there, so
    ``pg.Coalescent(n=4).sfs.joint_distribution(1, 2).cdf(-1.0, 1.0)`` returned -0.147. For identical rewards the
    reduced CDF returned the atom ``P(R = 0)`` at every threshold at or below zero. The CDF is zero wherever either
    threshold is negative, on both paths, and the density is zero there too."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    grid = np.asarray(jd.cdf([-1.0, -1e-9, 0.5, 2.0], [-0.5, 0.5, 2.0]))

    assert np.all(grid[:2, :] == 0.0)
    assert np.all(grid[:, 0] == 0.0)
    assert np.all(grid[2:, 1:] > 0.0)
    assert np.all(np.asarray(jd.pdf([-1.0, 0.5], [-1.0, 0.5]))[0, :] == 0.0)

    # identical rewards, with a substantial atom P(R_3 = 0) for n = 5
    diag = pg.Coalescent(n=5).sfs.joint_distribution(3, 3)
    assert diag._atoms['both0'] > 0.05
    assert diag.cdf(-1.0, 1.0) == 0.0
    assert diag.cdf(1.0, -1.0) == 0.0
    assert diag.cdf(0.0, 1.0) == pytest.approx(diag._atoms['both0'])


def test_joint_marginal_rejects_unknown_reward():
    """``JointRewardDistribution.marginal`` returned the marginal of ``R_b`` for any argument other than ``'a'``, so
    ``marginal('c')`` succeeded silently, unlike ``conditional`` and the empirical ``marginal``."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)

    with pytest.raises(ValueError):
        jd.marginal('c')

    assert jd.marginal('b').mean == pytest.approx(jd.mean[1])


def test_joint_distribution_members_honour_the_cache_setting():
    """``reward.py`` used ``functools.cached_property``, which stores every value, so ``Settings.cache = False`` did not
    stop the joint's atoms, cosine coefficients or the conditional variance from being memoised."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)

    prev = Settings.cache
    Settings.cache = False
    try:
        _ = jd._atoms
        _ = jd.mean
        assert '_atoms' not in jd.__dict__
        assert 'mean' not in jd.__dict__
    finally:
        Settings.cache = prev

    _ = jd._atoms
    assert '_atoms' in jd.__dict__


@pytest.mark.parametrize('window', [dict(start_time=0.5), dict(end_time=0.5)])
def test_joint_and_conditional_distribution_functions_raise_on_a_windowed_coalescent(window):
    """The joint transform ignored the accumulation window, so ``Coalescent(n=4, end_time=0.5).sfs.joint_distribution(1,
    2)`` returned the to-absorption CDF, density and conditionals next to windowed mixed moments. Every transform,
    distribution function and conditional of a joint raises on a windowed coalescent, while the moments keep the
    window."""
    jd = pg.Coalescent(n=4, **window).sfs.joint_distribution(1, 2)

    for call in (lambda: jd.lst(0.1, 0.2), lambda: jd.lst_batch([0.1], [0.2]), lambda: jd.lst_taylor(0.1),
                 lambda: jd.cdf(1.0, 1.0), lambda: jd.pdf(1.0, 1.0), lambda: jd.conditional('a', 1.0),
                 lambda: jd.conditional('a', 0.0)):
        with pytest.raises(NotImplementedError):
            call()

    assert jd.moment(1, 1) < pg.Coalescent(n=4).sfs.joint_distribution(1, 2).moment(1, 1)


def _refusing_joint(threshold_level: float):
    """A joint distribution whose conditionals refuse, with ``ValueError``, every conditioning value above the given
    level of the continuous part of the conditioning reward, standing in for values below inversion resolution."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    build = jd.conditional

    thresholds = {}
    for on in ('a', 'b'):
        p0 = float(jd._atoms['a0' if on == 'a' else 'b0'])
        thresholds[on] = float(jd.marginal(on).quantile(p0 + (1.0 - p0) * threshold_level))

    def conditional(on='a', value=0.0):
        if value > thresholds[on]:
            raise ValueError("The density is below the resolution of the inversion.")
        return build(on, value)

    jd.conditional = conditional
    return jd


def test_conditional_checks_skip_levels_that_cannot_be_constructed(caplog):
    """The conditional checks handled a conditional that could not be constructed in two ways. The two moment checks
    returned ``inf`` as soon as a single level refused, while the two tower checks dropped the node from the quadrature
    without renormalising, so its weight turned into a spurious deficit. All four skip the level, warn once naming it,
    and compute their error over the remaining levels, and only a check without any constructed level is infinite."""
    import logging

    jd = _refusing_joint(0.6)

    # the package logger does not propagate to root, where caplog listens, so capture it directly
    pg_logger = logging.getLogger('phasegen')
    pg_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger='phasegen')
    try:
        moments = jd.check_conditional_moments(quantiles=[0.3, 0.9], tol=1.0)
    finally:
        pg_logger.removeHandler(caplog.handler)
    assert np.isfinite(list(moments.values())).all()
    assert any("could not be constructed" in r.getMessage() and "0.9" in r.getMessage() for r in caplog.records)

    reference = _refusing_joint(1.0).check_conditional_moments(quantiles=[0.3], tol=1.0)
    assert moments == pytest.approx(reference, rel=1e-12)

    grid = jd.check_conditional_grid_moments(quantiles=[0.3, 0.9], tol=1.0)
    assert grid == pytest.approx(_refusing_joint(1.0).check_conditional_grid_moments(quantiles=[0.3], tol=1.0))

    # a renormalised quadrature over the remaining nodes stays close to the full one, a dropped weight would not
    full = _refusing_joint(1.0).check_total_expectation(n_points=6, tol=1.0)
    skipped = _refusing_joint(0.9).check_total_expectation(n_points=6, tol=1.0)
    assert all(skipped[on] < 0.05 for on in skipped) and all(np.isfinite(list(full.values())))

    assert np.isfinite(list(_refusing_joint(0.9).check_total_probability(n_points=3, n_y=3, tol=1.0).values())).all()

    # nothing constructed: infinite
    none = _refusing_joint(0.0)
    assert all(np.isinf(none.check_conditional_moments(n_points=2, tol=1.0)[on]) for on in ('a', 'b'))
    assert all(np.isinf(none.check_total_expectation(n_points=3, tol=1.0)[on]) for on in ('a', 'b'))


def test_window_average_has_the_shape_of_the_statistic():
    """``JointRewardDistribution.window_average`` wrapped every statistic in ``np.atleast_1d``, so a scalar statistic
    such as the conditional mean came back as a length-one array and callers had to index ``[0]``. A scalar statistic
    returns a float, an array-valued one an array of its shape."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    ys = np.array([0.5, 1.0, 2.0])

    mean = jd.window_average(lambda c: c.mean, 'a', 1.0, 0.05, n_nodes=2)
    cdf = jd.window_average(lambda c: c.cdf(ys), 'a', 1.0, 0.05, n_nodes=2)

    assert isinstance(mean, float)
    assert mean == pytest.approx(float(jd.conditional('a', 1.0).mean), rel=0.05)
    assert isinstance(cdf, np.ndarray) and cdf.shape == ys.shape


def test_window_average_requires_a_positive_half_width():
    """``JointRewardDistribution.window_average`` takes the window explicitly and refuses a non-positive half-width.
    Regression: its defaults value=0 and half_width=0 always raised, and half_width=0 built n_nodes identical
    conditionals."""
    jd = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)

    with pytest.raises(TypeError):
        jd.window_average(lambda c: c.mean)

    for half_width in (0.0, -0.05):
        with pytest.raises(ValueError, match='half-width of the conditioning window must be positive'):
            jd.window_average(lambda c: c.mean, 'a', 1.0, half_width)


def _round_trip_cases() -> list:
    """(label, distribution) pairs spanning the 1D representations the cdf/quantile pair has to hold across: a plain
    bin, a heavy upper tail (the case the cosine window truncates), several epochs, a multiple-merger model, a
    non-bin reward, and a conditional (whose transform is itself a nested inversion)."""
    expansion = pg.Demography(pop_sizes={0: 1, 1: 10})

    cases = [
        ('sfs bin 2, constant', pg.Coalescent(n=4).sfs.bin(2)),
        ('sfs bin 2, heavy tail', pg.Coalescent(n=4, demography=expansion).sfs.bin(2)),
        ('sfs bin 1, 3 epochs', pg.Coalescent(
            n=5, demography=pg.Demography(pop_sizes={0: 1, 0.5: 0.1, 2: 5})).sfs.bin(1)),
        ('sfs bin 2, beta MMC', pg.Coalescent(n=5, model=pg.BetaCoalescent(alpha=1.5)).sfs.bin(2)),
        ('total branch length, n=10', pg.Coalescent(n=10).total_branch_length),
    ]

    # a conditional: its lst is a nested inversion, so it exercises the same tail rule on a far noisier transform
    joint = pg.Coalescent(n=4, demography=expansion).sfs.joint_distribution(1, 2)
    v = float(joint.marginal('a').quantile(0.5))
    cases.append(('R_2 | R_1 = median', joint.conditional('a', v)))

    return cases


@pytest.mark.parametrize('label, d', _round_trip_cases(), ids=lambda x: x if isinstance(x, str) else '')
def test_cdf_of_quantile_is_the_identity(label, d):
    """``cdf(quantile(q)) == q``: the 1D cdf and quantile must be mutual inverses, on both sides of the tail cut.

    Both read one grid -- the cosine fit below :attr:`Settings.dehoog_tail_quantile`, exact de Hoog nodes above -- and
    an interpolation of a monotone function and its inverse round-trip exactly, whichever side of the cut they land
    on. The quantile used to bisect the exact inversion out there while the cdf kept interpolating a cosine fit that
    force-normalises to 1 at the end of its window, so on the heavy-tailed bin ``cdf(quantile(0.999))`` came back as
    exactly 1.0: a survival of zero where the truth is 1e-3.
    """
    cut = Settings.dehoog_tail_quantile

    for q in (0.25, 0.5, 0.9, cut - 1e-3, cut, cut + 1e-3, 0.99, 0.999):
        x = float(d.quantile(q))
        assert float(d.cdf(x)) == pytest.approx(q, abs=1e-5), f"{label}: round trip broken at q = {q}"


def test_atom_larger_than_the_cut():
    """An atom can carry the CDF past the cut in a single jump, and the grid must still be the distribution's.

    The grid joins the fit's half to the exact half at ``x_cut``, the point where the CDF reaches the cut, and the
    node there carries the fit's value -- which is normally the cut itself. An atom breaks that: if ``P(R = 0)``
    already exceeds the cut then ``x_cut`` is 0 and the value there is the *atom*, well above the cut. Stamping the
    cut on it shifted every node of the grid by the difference, putting a 2e-2 error in the CDF of a Dirac bin whose
    atom is 0.99 -- against a tolerance of 1e-3.
    """
    d = pg.Coalescent(n=5, model=pg.DiracCoalescent(psi=0.99999999, c=50)).sfs.bin(3)  # a near-total merger
    atom = d.cdf._cdf_point(0.0)

    assert atom > Settings.dehoog_tail_quantile, f"the atom ({atom:.4f}) no longer exceeds the cut; pick another case"
    assert float(d.cdf(0.0)) == pytest.approx(atom, abs=1e-6), "the grid lost the atom at the origin"

    for x in (1e-6, 0.01, 0.1, 1.0):
        assert float(d.cdf(x)) == pytest.approx(d.cdf._cdf_point(x), abs=1e-3), f"the grid is shifted at x = {x}"


@pytest.mark.parametrize('cut', [0.0, 0.5, 0.98])
def test_dehoog_cut_spans_its_range(cut):
    """``Settings.dehoog_tail_quantile`` is a knob over the whole range, not just a tail switch: it is the CDF level
    at or above which the grid's nodes take their value from the exact inversion rather than the cosine fit. At 0 the
    grid is entirely exact, at 1 entirely the fit, and in between each node takes whichever is trusted at its level.
    Nothing else depends on it -- the interpolation rule is the same everywhere.

    It used to invert at the bottom: ``cut = 0`` asks for an all-exact grid and produced an all-*cosine* one, because
    the cut's own quantile is then 0 and a guard read that as "no tail wanted". So the setting silently did the
    opposite of what it said at exactly the end a user would reach for to validate the fit.
    """
    Settings.dehoog_tail_quantile = cut
    d = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs.bin(2)

    x = float(d.quantile(0.999))
    nodes, _ = d.cdf._cdf_grid(q_max=0.999)
    n_exact = len(d.cdf._shared('cdf_exact', list))

    assert n_exact > 0, f"cut = {cut} built no exact nodes"
    assert (n_exact == len(nodes)) == (cut == 0.0), f"cut = {cut}: expected an all-exact grid only at 0"

    # wherever the exact inversion supplies the grid, the far quantile must match a bisection on it
    lo, hi = 0.0, 2 * x
    while d.cdf._cdf_point(hi) < 0.999:
        hi *= 2
    while hi - lo > 1e-7 * x:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if d.cdf._cdf_point(mid) < 0.999 else (lo, mid)

    assert x == pytest.approx(0.5 * (lo + hi), rel=1e-4), f"cut = {cut}: far quantile is not the exact one"


def test_grid_answers_do_not_depend_on_call_order():
    """The lazily grown tail must not change an answer the grid has already given.

    The cdf, pdf and quantile share one grid whose exact de Hoog tail is materialised only when a query reaches past
    :attr:`Settings.dehoog_tail_quantile`, and grown only as far as that query needs. So the node set depends on what
    has been *asked*, and the danger is that it feeds back into the values: interpolate a point, let some later call
    extend or fill in the ladder, and the same point comes back different. Tolerances are tuned on those numbers, so
    an answer that depends on the caller's history is a defect even when it is a small one.

    Two things rule it out. The ladder only ever appends (its step is fixed by the distribution's own decay length,
    never by the query), so no bracket already interpolated can move. And everything below the cut is read off the
    cosine nodes alone -- including the cell that straddles the cut, which is why the tail is anchored there carrying
    the cosine's own value.
    """
    make = lambda: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs.bin(2)  # noqa: E731

    x = np.linspace(0.1, float(make().quantile(0.97)), 50)  # below the cut, so a fresh grid has no tail at all
    virgin = make()
    cold = np.array([virgin.cdf(x), virgin.pdf(x)])

    warm = make()
    warm.quantile(0.999)  # reaches into the tail, so the ladder now exists
    assert np.array_equal(np.array([warm.cdf(x), warm.pdf(x)]), cold), "the tail changed the answers below the cut"

    # and a deeper query must not disturb what the shallower one already returned
    deep = make()
    shallow = float(deep.quantile(0.99))
    deep.quantile(0.999999)  # grows the ladder further
    assert float(deep.quantile(0.99)) == shallow, "growing the tail moved an answer already given"

    # the grid a caller sees is the cache, never a slice of it trimmed to the span they happened to ask for: the span
    # decides only whether the ladder is *extended*. Trimming is what let a query at exactly the cut clamp to the node
    # below it, because the grid handed back stopped just short of the cut.
    d = make()
    d.quantile(0.999)  # the ladder now exists
    wide, _ = d.cdf._cdf_grid(x_max=1e9)
    narrow, _ = d.cdf._cdf_grid(x_max=0.0)  # a query that reaches nowhere near it must still see it
    assert np.array_equal(narrow, wide), "the grid was trimmed to the requested span"


def test_far_tail_is_exact_not_saturated():
    """Past the cosine window the cdf, pdf and quantile must still be the distribution's, not the window's.

    The cosine fit force-normalises to 1 at the end of its window, so interpolating it clamps there: the cdf came back
    as exactly 1.0 and the density as exactly 0 for every point beyond, reporting no mass at all on a bin whose true
    survival at ``quantile(0.999)`` is 1e-3. The grid therefore carries exact de Hoog nodes above
    :attr:`Settings.dehoog_tail_quantile`, joined in log-survival: a chord in ``F`` would leave the quantile 1e-3
    long, systematically, the tail being concave.

    All three functions read those nodes, so all three are checked against the per-point inversion here -- the
    quantile against a bisection on it, the one reference the grid cannot be its own judge of.
    """
    d = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10})).sfs.bin(2)
    cdf_point, pdf_point = d.cdf._cdf_point, d.cdf._pdf_point

    for q in (0.99, 0.999, 0.9999):
        x = float(d.quantile(q))

        lo, hi = 0.0, 2 * x
        while cdf_point(hi) < q:
            hi *= 2
        while hi - lo > 1e-7 * x:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cdf_point(mid) < q else (lo, mid)
        exact = 0.5 * (lo + hi)

        assert x == pytest.approx(exact, rel=1e-4), f"quantile({q}) = {x:.6g}, exact {exact:.6g}"
        assert float(d.cdf(x)) == pytest.approx(cdf_point(x), abs=1e-6), f"cdf past the window at q = {q}"
        assert float(d.pdf(x)) == pytest.approx(pdf_point(x), rel=1e-2), f"density past the window at q = {q}"


def test_density_is_continuous_at_the_tail_join():
    """The density read off the grid must not step where the de Hoog tail joins the cosine fit.

    The tail is anchored at the fit's value at the cut. On the total branch length of an extreme bottleneck, the de
    Hoog CDF sits 1.1e-3 below the fit there, and a tail made of the raw de Hoog values put that difference into the
    first tail segment of width 6e-3: the served density fell from 0.87 to 0.66 across the cut and the pdf missed the
    msprime reference beyond its tolerance. Shifting the de Hoog hazard onto the anchor over a band above the cut
    keeps the density within about ten percent across the join.
    """
    d = pg.Coalescent(
        n=5, demography=pg.Demography(pop_sizes={0: 1, 0.3: 0.01, 1: 1})
    ).total_branch_length._reward_distribution

    xs, cdf = d.cdf._cos_cdf_grid
    x_cut = float(np.interp(Settings.dehoog_tail_quantile, cdf, xs))
    below, above = float(d.pdf(x_cut - 1e-9)), float(d.pdf(x_cut + 1e-9))

    assert above == pytest.approx(below, rel=0.15), f"the density steps from {below:.4f} to {above:.4f} at the join"
    assert float(d.cdf(x_cut + 1e-9)) == pytest.approx(float(d.cdf(x_cut - 1e-9)), abs=1e-6)


def test_windowed_reward_distribution_functions_raise():
    """A reward accumulated over a bounded window (start_time > 0 or a finite end_time) has windowed moments but a
    to-absorption LST inversion, so its cdf / pdf / quantile would silently disagree with its own mean. The guard
    raises on the distribution-function path while the moments stay windowed; the default (no window) is unaffected."""
    # default: no window, distribution functions work
    d0 = pg.Coalescent(n=4).tree_height.distribution()
    assert 0.0 <= float(d0.cdf(1.0)) <= 1.0

    for kw in (dict(start_time=0.5), dict(end_time=1.0)):
        d = pg.Coalescent(n=4, **kw).tree_height.distribution()

        # the windowed moments are still computed (they honour the window)
        assert d.mean > 0 and d.var > 0

        # the distribution functions raise (not implemented for a window) rather than return a to-absorption law
        # inconsistent with the mean
        with pytest.raises(NotImplementedError):
            d.cdf(1.0)
        with pytest.raises(NotImplementedError):
            d.pdf(1.0)
        with pytest.raises(NotImplementedError):
            d.quantile(0.5)


@pytest.mark.parametrize('window', [dict(start_time=0.5), dict(end_time=0.5)])
def test_windowed_coalescent_distribution_functions_raise_on_every_host(window):
    """On a windowed coalescent the window lives on the tree-height distribution, not on the host of a reward
    distribution. The guard read it from the host, which carries no window for ``total_branch_length``,
    ``Coalescent.distribution()`` or an SFS bin, so ``Coalescent(n=4, end_time=0.5).total_branch_length.cdf(1.0)``
    returned the to-absorption law next to a windowed mean. The tree-height quantile grid stopped at ``end_time``, so
    ``tree_height.quantile(0.99)`` returned 0.5 while ``tree_height.cdf(0.5)`` stayed far below 0.99. Every 1D pdf,
    cdf and quantile raises on a windowed coalescent, while the moments keep honouring the window."""
    coal = pg.Coalescent(n=4, **window)

    hosts = [
        coal.tree_height,
        coal.total_branch_length,
        coal.distribution(),
        coal.distribution(pg.TotalBranchLengthReward()),
        coal.sfs.bin(1),
        coal.sfs,
        coal.tree_height.demes['pop_0'],
    ]

    for host in hosts:
        with pytest.raises(NotImplementedError):
            host.cdf(1.0)
        with pytest.raises(NotImplementedError):
            host.pdf(1.0)
        with pytest.raises(NotImplementedError):
            host.quantile(0.5)

    # the moments keep the window
    assert coal.total_branch_length.mean < pg.Coalescent(n=4).total_branch_length.mean
    assert coal.tree_height.mean < pg.Coalescent(n=4).tree_height.mean

    # the default moment-accumulation grid does not depend on a quantile function
    assert len(coal.tree_height._default_end_times()) == Settings.plot_n_grid


@pytest.mark.parametrize("name, transform, inverse, points", [
    ("exponential CDF", lambda s: 1 / (s * (s + 1)), lambda t: 1 - np.exp(-t), [0.05, 0.5, 2, 8, 20]),
    ("gamma density", lambda s: 1 / (s + 1) ** 3, lambda t: t ** 2 * np.exp(-t) / 2, [0.05, 0.5, 2, 8, 20]),
    ("Levy density", lambda s: np.exp(-np.sqrt(s)), lambda t: np.exp(-1 / (4 * t)) / (2 * np.sqrt(np.pi) * t ** 1.5),
     [0.1, 1, 5]),
    ("sine", lambda s: 1 / (s ** 2 + 1), np.sin, [0.5, 3, 10]),
])
def test_dehoog_inversion_matches_closed_form_inverses(name, transform, inverse, points):
    """The double-precision de Hoog inversion recovers closed-form inverse Laplace transforms, including a branch
    point (Levy) and an oscillating inverse (sine), to within 1e-9 at the default degree."""
    from phasegen.distributions.reward import _dehoog_invert

    for t in points:
        assert abs(_dehoog_invert(transform, t, Settings.dehoog_degree) - inverse(t)) < 1e-9, (name, t)


@pytest.mark.parametrize("name, transform, inverse, points", [
    ("exponential CDF", lambda s: 1 / (s * (s + 1)), lambda t: 1 - np.exp(-t), [0.05, 0.2, 1, 3, 10]),
    ("gamma density", lambda s: 1 / (s + 1) ** 3, lambda t: t ** 2 * np.exp(-t) / 2, [0.05, 0.5, 2, 8]),
])
def test_dehoog_inversion_converges_below_the_default_degree(name, transform, inverse, points):
    """The accuracy of the inversion must improve with the degree over the whole range of degrees, not only at the
    default. Regression: the improved remainder divided by ``h`` where the period-2 tail of the continued fraction
    requires ``h ** 2``, costing one to two digits. At the default degree the aliasing floor of the contour hides it
    entirely, so only a lower degree, which ``Settings.dehoog_degree`` is free to take, separates the two forms."""
    from phasegen.distributions.reward import _dehoog_invert

    # the correct remainder gives 3.1e-5, 7.0e-7 and 4.2e-10 on the exponential CDF at these degrees, the wrong one
    # 5.9e-4, 1.9e-5 and 1.7e-8, and 1.3e-4, 2.0e-6 and 3.1e-9 on the gamma density
    for degree, tol in ((5, 1e-4), (6, 3e-6), (8, 2e-9)):
        error = max(abs(_dehoog_invert(transform, t, degree) - inverse(t)) for t in points)
        assert error < tol, (name, degree, error)


def test_dehoog_tail_resolves_the_rise_after_a_bottleneck():
    """Across the extreme bottleneck of ``3_epoch_extreme_bottleneck_n_5`` the total branch length has its 0.99
    quantile on a steep rise at 1.56, where lineages that did not coalesce before the bottleneck coalesce almost at
    once. The contour ``T = 2t`` whose abscissa grew with the degree extrapolated this series through an
    ill-conditioned Pade approximant, giving 0.98982 against 0.9901197 at degree 15. The reference is the transform in
    80-digit arithmetic inverted by de Hoog at degree 100 on the contour ``T = t``, ``eps = 1e-40``."""
    d = pg.Coalescent(
        n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.01, 1: 1}})
    ).total_branch_length.distribution()

    assert abs(d.cdf._cdf_point(1.561995) - 0.9901197135048715) < 2e-5


def test_dehoog_points_evaluated_together_match_single_points():
    """An array of points is inverted in one batched transform evaluation, point by point equal to single points,
    with the atom at 0 and 0 below it, for the CDF and the density."""
    d = pg.Coalescent(n=4).sfs._bin_distribution(2)
    xs = np.array([-1.0, 0.0, 0.3, 1.2, 4.0, 1.2])
    single = [d.cdf._cdf_point(float(x)) for x in xs]
    d.__dict__.pop('_lst_curve_cache', None)

    np.testing.assert_allclose(d.cdf._cdf_point(xs), single, rtol=1e-13, atol=1e-15)
    assert single[0] == 0.0 and single[1] == pytest.approx(d.lst(np.inf).real, abs=1e-15)
    np.testing.assert_allclose(d.pdf._pdf_point(xs[2:]), [d.pdf._pdf_point(float(x)) for x in xs[2:]], rtol=1e-13)
    assert np.isnan(d.cdf._cdf_point(np.nan))


@pytest.mark.parametrize('ne_ancestral', [5e3, 2e4])
def test_window_and_corr_use_exact_moments_with_a_slow_ancient_epoch(ne_ancestral):
    """
    A present-day size of 1 set the time scale to 1 while an ancient epoch of size ``ne_ancestral`` made the slowest
    tail rate of the reward far smaller than the cumulant step ``1e-4``. ``phi(-h)`` was then evaluated past its pole:
    the cumulant mean came out negative and the variance clamped to its floor, so the cosine window collapsed and every
    quantile was negative (``-7.2`` at ``2e4``), and ``corr`` reported ``4e20``. Near the threshold (``5e3``) the window
    and ``corr`` (0.889 against 0.853) were biased.
    """
    from phasegen.rewards import TreeHeightReward

    coal = pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={0: 1.0, 0.5: ne_ancestral}))
    rd = coal.distribution(TreeHeightReward())
    q = np.array([0.5, 0.9])

    assert rd._range() == pytest.approx(rd.mean + 12.0 * rd.std, rel=1e-12)
    assert np.all(rd.quantile(np.array([0.1, 0.5, 0.9])) >= 0.0)
    np.testing.assert_allclose(rd.quantile(q), coal.tree_height.quantile(q), rtol=1e-2)

    c1, c2 = rd._cumulants()
    assert c1 == pytest.approx(rd.mean, rel=1e-6)
    assert c2 == pytest.approx(rd.var, rel=1e-3)

    j = coal.sfs.joint_distribution(1, 2)
    exact = j.cov / np.sqrt(j.marginal('a').var * j.marginal('b').var)
    assert j.corr == pytest.approx(exact, rel=1e-12)
    assert 0.8 < j.corr < 0.9


def test_transform_drops_states_the_initial_vector_cannot_reach():
    """
    A demography naming a deme without samples and without migration produced migration targets of rate zero, such as
    one lineage in each deme. They never absorb, so the last-epoch system was singular at ``s = 0``: ``lst(0)`` was
    NaN and ``total_branch_length.cdf`` raised ``ValueError: array must not contain infs or NaNs``.
    """
    coal = pg.Coalescent(n={'a': 2, 'b': 0}, demography=pg.Demography(pop_sizes={'a': {0: 1.0}, 'b': {0: 1.0}}))
    reference = pg.Coalescent(n=2).total_branch_length

    assert coal.total_branch_length._reward_distribution.lst(0.0) == pytest.approx(1.0, abs=1e-12)
    assert coal.total_branch_length.cdf(1.0) == pytest.approx(reference.cdf(1.0), abs=1e-9)
    assert coal.total_branch_length.quantile(0.5) == pytest.approx(reference.quantile(0.5), rel=1e-6)


def test_time_scale_ignores_an_unsampled_deme():
    """
    The time scale was the mean size over all demes at time 0, including an unsampled source deme of size ``1e7``.
    The atom probe ``1e8 / tau = 20`` then reported a spurious atom (``cdf(0) = 0.048`` for bin 2, whose atom is 0,
    and ``quantile(0.02) = 0``), and the cumulant step became roundoff, collapsing the cosine window at the mean so that
    the CDF at the exact quantiles 0.1, 0.5 and 0.9 read 0.17, 0.78 and 0.98.
    """
    from phasegen.rewards import TreeHeightReward

    demography = pg.Demography(pop_sizes={'a': 1.0, 'b': 1e7}, migration_rates={('b', 'a'): 1.0, ('a', 'b'): 0.0})
    coal = pg.Coalescent(n={'a': 3, 'b': 0}, demography=demography)
    rd = coal.distribution(TreeHeightReward())
    q = np.array([0.1, 0.5, 0.9])

    assert rd._time_scale == 1.0
    np.testing.assert_allclose(rd.cdf(coal.tree_height.quantile(q)), q, atol=1e-3)

    bin2 = coal.sfs.bin(2)
    assert bin2.cdf(0.0) < 1e-6
    assert bin2.quantile(0.02) > 0.0


def test_exact_node_march_stays_local(monkeypatch):
    """
    The march of the de Hoog nodes jumped a whole cosine window when its density estimate was not positive. With
    ``dehoog_tail_quantile = 0`` the anchor sits at 0, where the tree-height density vanishes for ``n = 4``, and the
    second node landed at the window end, giving a median of 0.73 against 1.23. Behind a sharp bottleneck one
    non-monotone de Hoog node gave a negative secant and the march jumped from 1.503 to 3.06, putting the 0.999
    quantile at 1.76 against 1.505.
    """
    from phasegen.rewards import TreeHeightReward

    coal = pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={0: 1, 1.5: 1e-3}))
    q = np.array([0.99, 0.999])
    np.testing.assert_allclose(coal.distribution(TreeHeightReward()).quantile(q), coal.tree_height.quantile(q),
                               rtol=1e-2)

    monkeypatch.setattr(Settings, 'dehoog_tail_quantile', 0.0)
    coal = pg.Coalescent(n=4)
    rd = coal.distribution(TreeHeightReward())
    assert rd.quantile(0.5) == pytest.approx(coal.tree_height.quantile(0.5), rel=1e-4)
    assert rd.cdf(1.0) == pytest.approx(coal.tree_height.cdf(1.0), abs=1e-4)


def test_joint_density_accepts_unsorted_and_repeated_points():
    """The joint density evaluated its spline on the caller's points, which must be strictly increasing, so an
    unsorted query such as ``pdf([2, 1], [1, 0.5])`` raised ``ValueError``."""
    j = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    xs, ys = np.array([1.0, 2.0]), np.array([0.5, 1.0])
    ref = j.pdf(xs, ys)

    np.testing.assert_allclose(j.pdf(xs[::-1], ys[::-1]), ref[::-1, ::-1], rtol=1e-12)
    np.testing.assert_allclose(j.pdf(np.array([2.0, 1.0, 2.0]), ys), ref[[1, 0, 1], :], rtol=1e-12)


def test_conditioning_on_a_zero_probability_atom_raises():
    """The guard refused atoms below ``1e-9``, beneath the ``O(1e-8)`` bias of the atom probe for a reward with a
    positive density at 0, so conditioning on ``L_2 = 0`` for ``n = 4`` (every tree has a cherry) or on ``L_2 = 0`` for
    ``n = 3`` built a conditional on an event of probability zero."""
    with pytest.raises(ValueError, match='zero probability'):
        pg.Coalescent(n=4).sfs.joint_distribution(2, 3).conditional('a', 0.0)
    with pytest.raises(ValueError, match='zero probability'):
        pg.Coalescent(n=3).sfs.joint_distribution(1, 2).conditional('b', 0.0)

    # a real atom still conditions: P(L_3 = 0) = 1/3 for n = 4
    assert pg.Coalescent(n=4).sfs.joint_distribution(2, 3).conditional('b', 0.0).mean > 0


def test_proportional_rewards_are_singular():
    """
    Only identical reward vectors were recognised as a law on a line. For ``n = 2`` the total branch length is twice
    the tree height, yet the joint density returned a finite ridge of 1.63 and the conditional built a nested inversion
    of a point mass.
    """
    from phasegen.rewards import TreeHeightReward, TotalBranchLengthReward

    j = pg.Coalescent(n=2).joint_distribution(TotalBranchLengthReward(), TreeHeightReward())
    assert j._ratio == pytest.approx(2.0)

    with pytest.raises(NotImplementedError):
        j.pdf([2.0], [0.3, 1.0])
    with pytest.raises(NotImplementedError):
        j.conditional('a', 2.0)
    assert j.check_total_expectation() == {}

    tbl = j.marginal('a')
    for x, y in [(1.0, 2.0), (3.0, 0.5), (2.0, 1.0)]:
        assert j.cdf(x, y) == pytest.approx(tbl.cdf(min(x, 2.0 * y)), abs=1e-9)


def test_density_integrates_to_the_cdf_increments_near_a_jump():
    """The density of a reward distribution was the hazard slope from ``np.gradient``, interpolated between the grid's
    nodes, which mixes the slopes of neighbouring segments. Next to a near-discontinuity, where a flat stretch of nodes
    meets a steep one, it put far more mass into a segment than the CDF rises there (total branch length of a
    2-epoch rapid decline, ``n = 2``: 0.59 against 0.006 on one segment), and roundoff-level changes to the
    transform, such as the sparse instead of the dense LU, halved or doubled the scenario's pdf metric. The density
    must integrate to the CDF increment on every segment of the grid, under either solver."""
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 0.001}})
    x_max = 2.1
    gl_x, gl_w = np.polynomial.legendre.leggauss(64)

    for sparse_min in (256, 0):
        Settings.closed_form_sparse_min_states = sparse_min
        d = pg.Coalescent(n=2, demography=demography).total_branch_length._reward_distribution

        # the tail grid grows lazily with the largest query, so fix it first and test the segments inside the range
        d.cdf(2 * x_max)
        nodes, _ = d.cdf._cdf_grid(x_max=2 * x_max)
        nodes = nodes[(nodes >= 1.9) & (nodes <= x_max)]
        lo, hi = nodes[:-1], nodes[1:]
        keep = hi > lo
        lo, hi = lo[keep], hi[keep]

        # Gauss-Legendre inside each segment, where the density is smooth
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        pts = mid[:, None] + half[:, None] * gl_x[None, :]
        integral = half * (np.asarray(d.pdf(pts.ravel())).reshape(pts.shape) @ gl_w)

        np.testing.assert_allclose(integral, np.asarray(d.cdf(hi)) - np.asarray(d.cdf(lo)), atol=1e-10)


def test_joint_cdf_surface_stays_within_the_unit_interval():
    """The atom and axis terms of the joint CDF cancel to roundoff at the origin, which left values such as -5e-37 on
    the plotted surface of the bottleneck example in the User Guide and made R's ``persp`` warn that the surface
    extends beyond its box. The joint CDF must lie within ``[0, 1]`` everywhere."""
    coal = pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}}))
    z = np.asarray(coal.sfs.joint_distribution(1, 2).cdf._plot_data(surface=True).z)

    assert z.min() >= 0.0 and z.max() <= 1.0


def test_joint_density_does_not_depend_on_the_other_points_of_the_query():
    """The finite-difference grid of the joint density spanned the queried range with a fixed node count, so the width
    of the cell the density is averaged over grew with the largest queried point. For ``n = 4`` the density of the
    (tree height, total branch length) pair at ``(1, 3)`` was 0.3176 evaluated alone, 0.3068 when ``(3, 8)`` shared the
    call and 0.0162 when ``(100, 300)`` did, against a mixed central difference of the CDF of 0.3185."""
    from phasegen.rewards import TreeHeightReward, TotalBranchLengthReward

    j = pg.Coalescent(n=4).joint_distribution(TreeHeightReward(), TotalBranchLengthReward())

    alone = float(j.pdf(1.0, 3.0))
    xs, ys = np.array([0.5, 1.0, 3.0, 100.0]), np.array([2.0, 3.0, 8.0, 300.0])
    within_a_batch = float(np.asarray(j.pdf(xs, ys))[1, 1])

    assert alone == within_a_batch

    # the mixed central difference of the CDF, which is evaluated per point and so cannot depend on the query
    h = 0.02
    F = np.asarray(j.cdf(np.array([1.0 - h, 1.0 + h]), np.array([3.0 - h, 3.0 + h])))
    reference = (F[1, 1] - F[1, 0] - F[0, 1] + F[0, 0]) / (4 * h * h)

    assert alone == pytest.approx(reference, rel=0.02)

    # every point of the batch agrees with its own single-point evaluation, including the near-origin one whose
    # cell width the companion point (100, 300) used to inflate by a factor of 100
    for i, x in enumerate(xs[:-1]):
        for k, y in enumerate(ys[:-1]):
            assert float(np.asarray(j.pdf(xs, ys))[i, k]) == float(j.pdf(float(x), float(y)))


def test_atom_probe_is_equivariant_under_scaling_the_reward():
    """The atom probe was ``1e8 / tau`` with ``tau`` built from the rates alone, so it ignored the magnitude of the
    reward. Multiplying the ``n = 4`` tree-height reward by ``1e-8`` made the probe report an atom of 0.32 for a
    continuous law: the CDF was off by up to 0.28, the density was identically zero at the lower points and
    ``quantile(0.1)`` was 0. Scaling a reward by ``c > 0`` must leave the law unchanged up to the same scaling,
    ``F_{cR}(ct) = F_R(t)``."""
    from phasegen.rewards import CustomReward, TreeHeightReward

    coal = pg.Coalescent(n=4)
    unscaled = coal.tree_height.distribution(TreeHeightReward())
    ts = np.array([0.3, 0.8, 1.5, 3.0])
    probs = np.array([0.1, 0.5, 0.9])
    cdf_ref = np.array([unscaled.cdf(t) for t in ts])
    pdf_ref = np.array([unscaled.pdf(t) for t in ts])
    quantile_ref = np.array([unscaled.quantile(p) for p in probs])

    for c in (1e3, 1e-3, 1e-6, 1e-8, 1e-9):
        scaled = coal.tree_height.distribution(CustomReward(lambda ss, m=c: m * TreeHeightReward()._get(ss)))

        assert scaled.lst(np.inf) == 0.0  # the tree height has no atom
        np.testing.assert_allclose([scaled.cdf(t * c) for t in ts], cdf_ref, atol=1e-9)
        np.testing.assert_allclose([scaled.pdf(t * c) * c for t in ts], pdf_ref, rtol=1e-6)
        np.testing.assert_allclose([scaled.quantile(p) / c for p in probs], quantile_ref, rtol=1e-6)

    # a genuinely atomic reward keeps its atom: bin 3 of n = 4 is empty unless the tree is a caterpillar
    assert pg.Coalescent(n=4).sfs.bin(3).lst(np.inf).real == pytest.approx(1 / 3, rel=1e-14)


def test_blocked_final_epoch_raises_instead_of_returning_nan():
    """A final epoch in which some lineages can never coalesce makes the shifted final-epoch system singular at
    s = 0. The transform and every inversion built on it must say so. Regression: the transient states were selected
    by forward reachability from the initial vector, which cannot detect a state that can no longer reach absorption,
    so the solve returned nan and surfaced as an opaque 'array must not contain infs or NaNs' from the cosine fit."""
    dem = pg.Demography(
        pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
        migration_rates={('pop_0', 'pop_1'): {0: 1.0, 50.0: 0.0},
                         ('pop_1', 'pop_0'): {0: 1.0, 50.0: 0.0}}
    )
    coal = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=dem)

    with pytest.raises(ValueError, match="does not absorb"):
        coal.distribution(pg.TreeHeightReward()).lst(0.0)

    with pytest.raises(ValueError, match="does not absorb"):
        coal.total_branch_length.cdf(1.0)

    with pytest.raises(ValueError, match="does not absorb"):
        coal.sfs.bin(1).cdf([1.0])


def test_migration_barrier_in_a_bounded_epoch_still_works():
    """The guard keys on the final epoch only: a barrier in a bounded epoch leaves absorption certain afterwards, so
    the transform must still be evaluated rather than refused."""
    dem = pg.Demography(
        pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
        migration_rates={('pop_0', 'pop_1'): {0: 0.0, 1.0: 1.0},
                         ('pop_1', 'pop_0'): {0: 0.0, 1.0: 1.0}}
    )
    coal = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=dem)

    assert np.isclose(coal.distribution(pg.TreeHeightReward()).lst(0.0).real, 1.0)
    assert 0.0 <= coal.total_branch_length.cdf(5.0) <= 1.0


# the sampler draws trajectories through the CTMC, so it is independent of the transform and inversion machinery;
# the bound is Dvoretzky-Kiefer-Wolfowitz at 1 - 1e-3, sqrt(log(2 / 1e-3) / (2 N)), which is 0.0138 at N = 20000
_ECDF_SAMPLES = 20000
_ECDF_BOUND = float(np.sqrt(np.log(2 / 1e-3) / (2 * _ECDF_SAMPLES)))


@pytest.mark.parametrize("label, build", [
    ("standard", lambda: pg.Coalescent(n=4)),
    ("beta", lambda: pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5))),
    ("dirac", lambda: pg.Coalescent(n=4, model=pg.DiracCoalescent(psi=0.5, c=1.0))),
    ("two epochs", lambda: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}}))),
    ("two demes", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 2.0},
                                 migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}))),
    ("two loci", lambda: pg.Coalescent(n=3, loci=2, recombination_rate=1.0)),
])
def test_per_point_cdf_matches_the_sampler(label, build):
    """The per-point de Hoog CDF has no comparison config exercising it, so it is pinned against the trajectory
    sampler, which shares none of the transform machinery."""
    dist = build().total_branch_length
    draws = np.asarray(dist.sample(_ECDF_SAMPLES, seed=1))
    cdf = dist.distribution().cdf

    for t in np.quantile(draws, [0.1, 0.25, 0.5, 0.75, 0.9]):
        empirical = float(np.mean(draws <= t))
        assert abs(empirical - cdf._cdf_point(float(t))) < _ECDF_BOUND, (label, float(t), empirical)


@pytest.mark.parametrize("i", [1, 2])
def test_per_point_cdf_of_an_sfs_bin_matches_the_sampler(i):
    """An SFS bin carries zero-reward states and an atom at zero, which the total branch length does not."""
    coal = pg.Coalescent(n=5)
    draws = np.asarray(coal.sfs.sample(_ECDF_SAMPLES, seed=1))[:, i]
    cdf = coal.sfs.bin(i).cdf

    for t in np.quantile(draws, [0.25, 0.5, 0.75, 0.9]):
        empirical = float(np.mean(draws <= t))
        assert abs(empirical - cdf._cdf_point(float(t))) < _ECDF_BOUND, (i, float(t), empirical)


def test_exact_march_tolerates_a_step_below_the_float_spacing():
    """The march sizes its step from the secant of the last two nodes, and the step can fall below the spacing of
    the floats at that point, leaving the node where it was. Regression: the repeated node then divided by a zero
    span and a public cdf / pdf / quantile raised ZeroDivisionError."""
    coal = pg.Coalescent(
        n=5,
        model=pg.DiracCoalescent(psi=0.5, c=3.0),
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.1: 0.02}})
    )
    dist = coal.sfs.bin(1)

    # the march must continue past the stretch where the step collapses: the law is complete well before 0.6
    for x in (0.6, 1.0, 3.0):
        assert float(dist.cdf(x)) == pytest.approx(1.0, abs=1e-4)

    assert np.isfinite(float(dist.quantile(0.99)))


@pytest.mark.parametrize("n, sizes", [(8, {0: 1.0, 0.3: 0.05, 1.2: 2.0}), (15, {0: 1.0, 0.5: 0.1})])
def test_exact_march_respects_its_increment_from_an_anchor_at_the_origin(n, sizes):
    """With the tail cut at zero the anchor sits at the origin, where a distribution with no mass there makes the
    secant an average over the flat region and far below the density at the node. Regression: the step then
    overshot the increment it is meant to respect and the CDF between the nodes was interpolated across the gap,
    leaving the served curve up to 0.49 from the exact per-point inversion."""
    original = Settings.dehoog_tail_quantile
    Settings.dehoog_tail_quantile = 0.0
    try:
        dist = pg.Coalescent(n=n, demography=pg.Demography(pop_sizes={'pop_0': dict(sizes)})).total_branch_length
        point = dist.distribution().cdf

        for x in np.linspace(0.05, float(dist.mean) * 3, 25):
            assert abs(float(dist.cdf(float(x))) - float(point._cdf_point(float(x)))) < 0.05, x
    finally:
        Settings.dehoog_tail_quantile = original


def test_conditional_on_linked_loci_carries_the_diagonal_atom():
    """Two rewards equal on a set of paths with positive probability, as the tree heights of linked loci, give the
    conditional an atom at the conditioning value. Regression: the law was treated as continuous, the density was
    biased by 35 to 49% everywhere and spiked at the value, and no atom was reported. Reference values from 400,000
    sampled trajectories: an atom of 0.2467 +- 0.0035 at v = 1.3, and quantiles 0.843, 1.299 and 1.967 at 0.2, 0.5
    and 0.8."""
    joint = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).tree_height.loci.joint_distribution(0, 1)
    v = 1.3
    cond = joint.conditional('a', v)

    assert cond._p == pytest.approx(0.2467, abs=0.014)

    # the atom is a step of the CDF at v, and the levels it covers return v
    assert float(cond.cdf(v)) - float(cond.cdf(v - 1e-9)) == pytest.approx(cond._p, abs=1e-6)
    assert float(cond.quantile(0.5)) == pytest.approx(v)

    for q, ref in ((0.2, 0.843), (0.8, 1.967)):
        assert float(cond.quantile(q)) == pytest.approx(ref, rel=0.02), q

    # the density is that of the continuous part and stays bounded at v
    assert float(cond.pdf(v)) < 1.0


def test_line_atom_conditional_expands_its_continuous_part_on_its_own_window():
    """The continuous part of a line-atom conditional is expanded on a window that contains it. At a low recombination
    rate the diagonal atom holds 99.6% of the mass, and the window of the conditional with the atom collapses onto the
    atom. Regression: the continuous part was expanded on that window, serving a CDF 200 standard errors from the
    sampler. Reference values from 2e8 sampled trajectories, conditioning within 2% of v: 1.18e-4 +- 0.9e-5 at v / 2
    and 0.99717 +- 4.5e-5 at 2 v."""
    joint = pg.Coalescent(n=2, loci=pg.LocusConfig(n=2, recombination_rate=0.01)).tree_height.loci.joint_distribution(
        0, 1)
    v = float(joint.marginal('a').quantile(0.2))
    cond = joint.conditional('a', v)

    assert cond._p == pytest.approx(0.996, abs=1e-3)
    assert cond._continuous.cdf._cos_coeffs['b'] > 5 * v
    assert float(cond.cdf(v / 2)) == pytest.approx(1.18e-4, abs=4e-5)
    assert float(cond.cdf(2 * v)) == pytest.approx(0.99717, abs=2e-4)


@pytest.mark.slow
def test_line_atom_masses_follow_the_refined_inner_truncation():
    """The atom masses of a line-atom conditional are taken at the inner truncation of the expansion, which the
    refinement raises from 60 to 480 on two loci under three epochs. Regression: the masses and the atoms subtracted
    from the continuous part stayed at the calibration truncation, and the CDF was off by 7.3e-3 at 1.2 v, 36 standard
    errors of the sampler. Reference values from 1e8 sampled trajectories, conditioning within 2% of the median v:
    0.05498 +- 1.1e-4 at v / 2, 0.76359 +- 2.1e-4 at 1.2 v and 0.95804 +- 1.0e-4 at 2 v."""
    joint = pg.Coalescent(
        n=2, loci=pg.LocusConfig(n=2, recombination_rate=0.5),
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 3, 1: 0.5}})
    ).tree_height.loci.joint_distribution(0, 1)
    v = float(joint.marginal('a').quantile(0.5))
    cond = joint.conditional('a', v)

    for x, ref, se in ((0.5, 0.05498, 1.1e-4), (1.2, 0.76359, 2.1e-4), (2.0, 0.95804, 1.0e-4)):
        assert float(cond.cdf(x * v)) == pytest.approx(ref, abs=4 * se), x

    nested = cond._nested
    f = joint._line_density(1.0, 'a', v, (nested._N0,))[0]
    assert nested._N0 > 60
    assert cond._atom_masses[0] == pytest.approx(f / nested._G0, rel=1e-12)


def test_conditional_carries_the_atom_of_a_sloped_line():
    """A triple merger of three lineages under a Beta coalescent keeps the total branch length at three times the tree
    height, so the joint law places mass on the line R_a = R_b / 3 and the conditional on the total branch length has
    an atom at a third of its value. Regression: only the diagonal was detected, and the law on this line was treated
    as continuous. Reference values from 400,000 sampled trajectories, conditioning within 2% of the median of R_b:
    an atom of 0.0553 +- 0.0026, and quantiles 1.428, 1.663 and 1.838 at 0.2, 0.5 and 0.8."""
    coal = pg.Coalescent(n=3, model=pg.BetaCoalescent(alpha=1.5))
    joint = coal.joint_distribution(pg.rewards.TreeHeightReward(), pg.rewards.TotalBranchLengthReward())

    assert joint._lines == pytest.approx((1 / 3,))

    v = float(joint.marginal('b').quantile(0.5))
    cond = joint.conditional('b', v)

    np.testing.assert_allclose(cond._atom_values, [v / 3], rtol=1e-12)
    assert cond._p == pytest.approx(0.0553, abs=0.0104)

    y = v / 3
    assert float(cond.cdf(y)) - float(cond.cdf(y - 1e-9)) == pytest.approx(cond._p, abs=1e-6)

    np.testing.assert_allclose(cond.quantile([0.2, 0.5, 0.8]), [1.428, 1.663, 1.838], rtol=0.02)


def test_joint_near_origin_check_probes_the_continuous_part(caplog):
    """The near-origin check of the joint cosine expansion probes above the atom of the conditioning reward. Regression:
    with P(R_a = 0) >= 0.4 all probes sat at 0, where the check returned P(R_a = 0, R_b = 0) (0.133 for Kingman n = 6
    bins 5 and 3) and warned, while the real gap is about 2e-4."""
    joint = pg.Coalescent(n=6).sfs.joint_distribution(5, 3)

    with caplog.at_level('WARNING'):
        assert joint._cos2d_wiggle_check < 0.01

    assert not [r for r in caplog.records if 'under-resolves near the origin' in r.getMessage()]


def test_joint_cosine_expansion_follows_the_term_count():
    """Changing Settings.cos_terms_2d on a live joint distribution rebuilds its expansion. Regression: the first
    expansion was cached, so the 16-term values were served after switching to 128 terms."""
    def joint():
        return pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}})).sfs.joint_distribution(1, 2)

    x, y = np.array([0.5, 1.5]), np.array([0.3, 1.0])

    pg.Settings.cos_terms_2d = 16
    live = joint()
    coarse = live.cdf(x, y)

    pg.Settings.cos_terms_2d = 128
    np.testing.assert_allclose(live.cdf(x, y), joint().cdf(x, y), rtol=1e-12)
    assert np.abs(live.cdf(x, y) - coarse).max() > 1e-6


def test_joint_window_leaving_out_mass_warns(caplog, monkeypatch):
    """A joint whose cosine window leaves out more than the threshold of a marginal's mass logs a warning, since the
    expansion folds that mass back and the joint CDF is too high near the window end. Kingman n = 12 bins 1 and 11
    leave out 8.6e-3 of R_b, the locus tree heights of n = 3 about 2e-3, so a threshold of 5e-3 separates them."""
    from phasegen.distributions import reward
    monkeypatch.setattr(reward, '_COS2D_TAIL_WARN', 5e-3)

    with caplog.at_level('WARNING'):
        pg.Coalescent(n=12).sfs.joint_distribution(1, 11).cdf(1.0, 1.0)

    assert any('leaves out a mass' in r.getMessage() and 'R_b' in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level('WARNING'):
        pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).tree_height.loci.joint_distribution(
            0, 1).cdf(1.0, 1.0)

    assert not [r for r in caplog.records if 'leaves out a mass' in r.getMessage()]


def _locus_jump_joint():
    """The joint of the tree heights at two linked loci under a tenfold decline at 0.5, where the density of either
    height jumps."""
    return pg.Coalescent(
        n=2, loci=pg.LocusConfig(n=2, recombination_rate=1),
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.1}})
    ).tree_height.loci.joint_distribution(0, 1)


def test_conditioning_across_an_epoch_jump(caplog):
    """The density of a locus tree height jumps by 5.46 at the epoch time 0.5, from about 0.6 to 6.0. The jump is
    subtracted from the inner inversion, so conditionals just below it, on it and just above it are built without a
    warning about the jump, the marginal density steps by the jump height, and the conditional on the jump is the
    limit from the right. The means agree with those of 4e7 sampled paths in windows of half-width 2.5e-4, 0.51247
    +- 0.00112 at 0.499 and 0.51493 +- 0.00036 at 0.501. Regression: values within 5% of the jump were refused or
    warned about."""
    joint = _locus_jump_joint()
    x0, heights = joint._density_jumps('a')
    assert x0.tolist() == [0.5]
    assert heights[0, 0].real == pytest.approx(5.45877594, rel=1e-6)

    with caplog.at_level('WARNING'):
        below, on, above = (joint.conditional('a', v) for v in (0.5 * (1 - 1e-7), 0.5, 0.5 * (1 + 1e-7)))
    assert not [r for r in caplog.records if 'jump' in r.getMessage()]

    assert on._nested._G0 - below._nested._G0 == pytest.approx(heights[0, 0].real, rel=2e-2)
    assert on._nested._G0 == pytest.approx(above._nested._G0, rel=1e-5)
    assert on.mean == pytest.approx(above.mean, rel=1e-5)

    assert joint.conditional('a', 0.499).mean == pytest.approx(0.51247, abs=4 * 0.00112)
    assert joint.conditional('a', 0.501).mean == pytest.approx(0.51493, abs=4 * 0.00036)


def test_euler_step_error_of_a_unit_step():
    """The Euler series of a unit step converges to the midpoint on the step, half a unit below the value 1 taken
    there, to half the damped weight of an image point ``(2j + 1) t`` on the step, and to the exact value elsewhere."""
    from phasegen.distributions.reward import _euler_step_error

    err = _euler_step_error(1.0, np.array([1.0, 3.0, 0.3, 6.0]), (60, 480))

    assert err[0] == pytest.approx([-0.5, -0.5], abs=2e-2)
    assert np.abs(err[1:]).max() < 1e-5
    assert np.abs(err[1]).max() < 5e-9  # well below half the image weight, exp(-16) / 2 = 5.6e-8


def test_epoch_jump_requires_initial_mass_on_the_positive_states():
    """A reward accrued at one rate has a density jump at an epoch time only if the paths that stay in its positive
    states from time zero leave them for good at a rate that changes there. The doubleton length at n = 3 is zero on
    the initial state, so its density is continuous at the epoch time 0.5, and the tree height at one of two linked
    loci is positive on the initial state and jumps there."""
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.1}})
    sfs = pg.Coalescent(n=3, demography=demography).sfs.joint_distribution(1, 2)

    assert sfs._density_jumps('b')[0].size == 0
    marginal = sfs.marginal('b')
    assert float(marginal.pdf(0.499)) == pytest.approx(float(marginal.pdf(0.501)), rel=0.02)

    assert _locus_jump_joint()._density_jumps('a')[0].tolist() == [0.5]
    locus = _locus_jump_joint().marginal('a')
    assert float(locus.pdf(0.52)) > 5 * float(locus.pdf(0.48))


@pytest.mark.parametrize('label, joint, jump, height', [
    ('Beta singletons', lambda d: pg.Coalescent(n=3, model=pg.BetaCoalescent(alpha=1.5), demography=d)
     .sfs.joint_distribution(1, 2), 1.5, 0.02950655),
    ('Dirac singletons', lambda d: pg.Coalescent(n=3, model=pg.DiracCoalescent(psi=0.7, c=5), demography=d)
     .sfs.joint_distribution(1, 2), 1.5, 0.43121633),
    ('Kingman singletons', lambda d: pg.Coalescent(n=3, demography=d).sfs.joint_distribution(1, 2), None, None),
    ('two-locus total tree height', lambda d: pg.Coalescent(
        n=2, loci=pg.LocusConfig(n=2, recombination_rate=1.0), demography=d).joint_distribution(
        pg.rewards.TotalTreeHeightReward(), pg.rewards.TreeHeightReward()), 1.0, 0.55138603),
])
def test_epoch_jump_of_a_reward_with_several_rates(label, joint, jump, height):
    """A reward with several positive rates jumps at ``c t0`` when the paths that stay in the states of rate ``c``
    from time zero stop accruing reward at a rate that changes at the epoch time ``t0 = 0.5``. The singletons at
    ``n = 3`` accrue at rate 3 on the initial state, which a multiple merger leaves straight into absorption, and at
    rate 1 afterwards. Under Kingman no path leaves the initial state for good, so the density is continuous. The jumps
    0.0295 (Beta), 0.431 (Dirac) and 0.551 (two loci) agree with Monte Carlo samples of 2e7 paths. The Taylor
    coefficients of the jump of the joint section in the other argument are the finite differences of the jump."""
    joint = joint(pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2}}))
    x0, heights = joint._density_jumps('a', order=2)

    if jump is None:
        assert x0.size == 0
        return

    assert x0.tolist() == [jump]
    assert heights[0, 0].real == pytest.approx(height, rel=1e-6)

    e = 1e-4
    plus, minus = (joint._density_jumps('a', s)[1][0, 0] for s in (e, -e))
    assert heights[0, 1] == pytest.approx((plus - minus) / (2 * e), abs=1e-8)
    assert heights[0, 2] == pytest.approx((plus - 2 * heights[0, 0] + minus) / (2 * e ** 2), abs=1e-5)


def test_moments_next_to_a_jump_start_at_the_calibrated_truncation():
    """The tree height under the extreme bottleneck jumps at 0.3, and the conditional mean of the total branch length
    at 0.303 agrees with that of 4e7 sampled paths in a window of half-width 1.5e-4, 1.1455 +- 0.00024. The moments at
    the truncations 30 and 60 agree to 1e-3 by chance at 1.1408, below the truncation 240 at which the marginal
    density settles."""
    joint = pg.Coalescent(
        n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.01, 1: 1}})
    ).joint_distribution(pg.rewards.TreeHeightReward(), pg.rewards.TotalBranchLengthReward())

    assert joint.conditional('a', 0.303).mean == pytest.approx(1.1455, abs=4 * 0.00024)


def test_bottleneck_density_has_no_jump_near_the_bulk_edge():
    """Under the extreme bottleneck the density of the singleton length rises and falls within a few thousandths
    around 1.53, but it has no jump there: every path leaves the initial state, whose singleton rate 5 gives the
    epoch time 0.3 the location 1.5, into states that still carry singletons. The conditional mean in the window
    [1.5305, 1.5310] agrees with that of 4e7 sampled paths in a window of half-width 5e-4, 0.00842 +- 0.00006 at
    1.5307."""
    joint = _bottleneck_joint()

    assert joint._density_jumps('a')[0].size == 0
    for v in (1.5305, 1.5307, 1.5310):
        assert joint.conditional('a', v).mean == pytest.approx(0.00842, abs=4 * 0.00006)


def test_truncation_warning_reports_the_estimated_error(caplog):
    """The truncation warning compares an error estimated from the convergence order with its bar, not the raw
    movement of the last half of the terms, which overstates the error about threefold. Regression: the documented
    bottleneck-recovery conditional warned at a movement of 1.7e-3, whose actual error is 5.1e-4. With few terms the
    expansion is unresolved and still warns."""
    joint = pg.Coalescent(
        n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}})
    ).sfs.joint_distribution(1, 2)

    with caplog.at_level('WARNING'):
        joint.conditional('a', 0.5).cdf(1.0)

    assert not [r for r in caplog.records if 'truncation' in r.getMessage()]

    caplog.clear()
    pg.Settings.cos_terms = 16
    with caplog.at_level('WARNING'):
        pg.Coalescent(
            n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}})
        ).sfs.joint_distribution(1, 2).conditional('a', 0.5).cdf(1.0)

    assert any('estimated truncation error' in r.getMessage() for r in caplog.records)


def test_line_atom_conditional_warns_above_the_inner_cutoff(caplog):
    """A line-atom conditional warns when its cosine expansion reaches frequencies at which the inner inversion no
    longer resolves the atom, where the continuous part carries the atom's negative. The linked locus heights stay
    below the cutoff at the default terms and cross it with four times as many. The Beta tree height given the total
    branch length crossed it at the default terms at the truncation 60, serving a CDF 9 standard errors from the
    sampler, and the refinement of the truncation on the CDF raises the cutoff above the expansion. The conditioning
    values are fixed literals at which the refined truncation is stable under relative changes of 1e-6. At the Beta
    median a change of three ulps stops the refinement at the truncation 120, at which the expansion ends just below
    the cutoff."""
    def loci():
        return pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).tree_height.loci.joint_distribution(0, 1)

    def warned(joint, value):
        caplog.clear()
        with caplog.at_level('WARNING'):
            joint.conditional('a', value).cdf(1.0)
        return any('inner inversion resolves the atom' in r.getMessage() for r in caplog.records)

    beta = pg.Coalescent(n=3, model=pg.BetaCoalescent(alpha=1.5)).joint_distribution(
        pg.rewards.TreeHeightReward(), pg.rewards.TotalBranchLengthReward())
    assert not warned(beta, 1.0)

    assert not warned(loci(), 1.2)

    pg.Settings.cos_terms = 4 * pg.Settings.cos_terms
    assert warned(loci(), 1.2)


def _bottleneck_joint():
    """The joint of the first two SFS bins under the extreme bottleneck of ``3_epoch_extreme_bottleneck_n_5``."""
    return pg.Coalescent(
        n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.01, 1: 1}})
    ).sfs.joint_distribution(1, 2)


@pytest.mark.parametrize('label, joint', [
    ('sfs', lambda: pg.Coalescent(n=4).sfs.joint_distribution(1, 2)),
    ('linked loci', lambda: pg.Coalescent(
        n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)).tree_height.loci.joint_distribution(0, 1)),
])
def test_inner_truncation_error_estimate_needs_no_extra_transform_evaluations(label, joint):
    """The coarser truncation of the refinement weights the nodes of the finer one, so it equals the Euler inversion
    at that truncation from the same transform evaluations. The expansion reuses the values of the refinement's
    locating pass, so past the window search it evaluates the transform once per argument of the locating pass, at
    both truncations, and once per nonzero frequency of the second pass. A line-atom conditional refines on the locating
    pass of its continuous part, on the window of that part, so its expansion reuses the same values. The moments do
    not trigger the refinement."""
    from phasegen.distributions.reward import _euler_invert

    joint = joint()
    cond = joint.conditional('a', float(joint.marginal('a').quantile(0.5)))
    nested, served = (cond._nested, cond._continuous) if label == 'linked loci' else (cond, cond)
    _ = cond.mean
    assert nested._G_rough is None

    s = -2.0j
    pair = nested._inner(s, (nested._N0, nested._N0 // 2))
    for got, n0 in zip(pair, (nested._N0, nested._N0 // 2)):
        want = _euler_invert(lambda u: joint.lst_batch(u, np.full(len(u), s)), nested._value, N0=n0)
        assert got == pytest.approx(want, rel=1e-12)

    served.cdf._range(served.cdf._cos_rough_scale)
    calls = []
    inner = nested._inner
    nested._inner = lambda s, truncations: calls.append(len(truncations)) or inner(s, truncations)
    _ = served.cdf._cos_coeffs
    assert calls.count(2) == served.cdf._cos_terms_rough + 1
    assert calls.count(1) == served.cdf._cos_terms - 1
    assert nested._N0 == 60


def test_unresolved_inner_truncation_warns(caplog):
    """The refinement warns when the CDF of the locating pass still moves by more than its bar at the largest
    truncation tried."""
    cond = pg.Coalescent(n=4).sfs.joint_distribution(1, 2).conditional('a', 0.5)
    cond.cdf._cos_truncation_tol = 0.0

    with caplog.at_level('WARNING'):
        cond._refine(n_max=cond._N0)

    assert any('inner inversion is unresolved' in r.getMessage() for r in caplog.records)


def test_unresolved_inner_truncation_warning_follows_check_inversions(caplog):
    """``Settings.check_inversions = False`` silences the warning of an unresolved inner inversion, as it does every
    other inversion check. Regression: it was logged regardless."""
    Settings.check_inversions = False
    cond = pg.Coalescent(n=4).sfs.joint_distribution(1, 2).conditional('a', 0.5)
    cond.cdf._cos_truncation_tol = 0.0

    with caplog.at_level('WARNING'):
        cond._refine(n_max=cond._N0)

    assert not [r for r in caplog.records if 'inner inversion is unresolved' in r.getMessage()]


@pytest.mark.slow
def test_bottleneck_conditional_matches_a_high_truncation_reference():
    """The conditional of the first SFS bin given the second at its 0.9 quantile under an extreme bottleneck, against
    the same expansion with the inner truncation at 1920. ``G(0)`` converges at the truncation 60, at which the served
    CDF was off by 3.9e-2, as ``G`` at the frequencies of the expansion was not converged. Refining the truncation on
    the CDF of the locating pass brings it within 2e-3."""
    joint = _bottleneck_joint()
    p0 = float(joint._atoms['a0'])
    cond = joint.conditional('a', float(joint.marginal('a').quantile(p0 + (1 - p0) * 0.9)))
    cdf = cond.cdf
    fit = cdf._cos_coeffs
    assert cond._N0 > 60

    b, w = fit['b'], fit['w']
    G = np.array([cond._inner(complex(-1j * wk), (1920,))[0] for wk in w])
    atom = (cond._inner(complex(np.inf), (1920,))[0] / G[0]).real
    xs = np.linspace(0.0, b, 4096)
    ref = np.maximum.accumulate(cdf._eval_cos_cdf(cdf._cos_fit_from(b, w, G / G[0], atom), xs))
    body = ref < 0.98

    assert np.abs(np.asarray(cdf(xs[body])) - ref[body]).max() < 2e-3


def test_quantile_passes_nan_through():
    """A NaN level gives a NaN quantile, leaves the other levels unchanged and builds no de Hoog nodes beyond those
    the finite levels need. Regression: a NaN level marched the node ladder to its cap."""
    d = pg.Coalescent(n=4).total_branch_length.distribution()

    out = d.quantile([0.5, np.nan])

    assert np.isnan(out[1]) and np.isnan(d.quantile(np.nan))
    assert out[0] == pytest.approx(d.quantile(0.5))
    assert len(d.__dict__['_lst_curve_cache']['cdf_exact']) == 1

    q1 = d.quantile(0.99)
    d.quantile([0.99, np.nan])
    assert len(d.__dict__['_lst_curve_cache']['cdf_exact']) < d.quantile._max_exact_nodes
    assert d.quantile(0.99) == pytest.approx(q1, rel=1e-8)


def test_joint_axis_terms_follow_live_cos_terms():
    """The axis terms of the joint CDF are rebuilt with the term count of a changed ``Settings.cos_terms``, as the
    marginal is. Regression: they kept the term count of their first evaluation."""
    joint = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    Settings.cos_terms = 48
    joint.cdf(0.5, 0.5)
    assert len(joint._cos_axis_coeffs['a']['w']) == 48

    Settings.cos_terms = 96
    fresh = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    assert len(joint._cos_axis_coeffs['a']['w']) == 96
    assert float(joint.cdf(0.5, 0.5)) == pytest.approx(float(fresh.cdf(0.5, 0.5)), abs=1e-12)


def test_joint_2d_expansion_survives_a_change_of_the_1d_terms():
    """The 2D coefficients and the density grid depend on ``Settings.cos_terms_2d`` only, so a change of
    ``Settings.cos_terms`` keeps them and rebuilds the axis terms. Regression: every cosine value was discarded."""
    joint = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)
    joint.pdf(0.5, 0.5)
    cos2d, grid, axis = joint._cos2d, joint._density_grid, joint._cos_axis_coeffs

    Settings.cos_terms = 2 * Settings.cos_terms
    joint.cdf(0.5, 0.5)

    assert joint._cos2d is cos2d and joint._density_grid is grid
    assert joint._cos_axis_coeffs is not axis

    Settings.cos_terms_2d = 2 * Settings.cos_terms_2d
    assert joint._cos2d is not cos2d


def test_joint_axis_expansion_reports_truncation(caplog):
    """The axis terms of the joint CDF run the truncation check of the marginal CDF, and span the atom
    ``P(R_a = 0, R_b = 0)`` at 0 to the axis mass at the window end. Regression: an unresolved axis term was not
    reported."""
    import logging
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)
    try:
        joint = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs.joint_distribution(2, 3)
        joint.cdf(0.5, 0.5)
        assert not any('axis g_' in r.getMessage() for r in caplog.records)

        atoms = joint._atoms
        assert atoms['both0'] > 1e-2
        for which, total in (('a', atoms['a0']), ('b', atoms['b0'])):
            b = joint._cos_axis_coeffs[which]['b']
            assert joint._cos_axis(which, np.array([-1.0, 0.0]))[1] == atoms['both0']
            assert joint._cos_axis(which, np.array([2 * b]))[0] == pytest.approx(total, abs=1e-3)

        Settings.cos_terms = 8
        pg.Coalescent(n=5).sfs.joint_distribution(1, 3).cdf(0.5, 0.5)
        assert any('axis g_b (truncation)' in r.getMessage() for r in caplog.records)
    finally:
        log.removeHandler(caplog.handler)


@pytest.mark.parametrize('orders', [(-1, 1), (1, -1), (2, -2), (0.5, 1), ('1', 1)])
def test_joint_moment_rejects_invalid_orders(orders):
    """The orders of a cross-moment are validated as those of every other moment. Regression: orders summing to zero
    returned 1 and others raised internal errors."""
    with pytest.raises((ValueError, TypeError), match='order k'):
        pg.Coalescent(n=4).sfs.joint_distribution(1, 2).moment(*orders)


def test_joint_moment_accepts_integral_float_orders():
    """An integral float order is accepted, as by ``_validate_order``, and the orders 0 and 0 give 1."""
    joint = pg.Coalescent(n=4).sfs.joint_distribution(1, 2)

    assert joint.moment(2.0, 1.0) == joint.moment(2, 1)
    assert joint.moment(0, 0) == 1.0


@pytest.mark.parametrize('call', [
    lambda j: j.conditional('a', np.nan),
    lambda j: j.conditional('a', np.inf),
    lambda j: j.window_average(lambda c: c.mean, 'a', 1.0, np.nan),
    lambda j: j.window_average(lambda c: c.mean, 'a', 1.0, np.inf),
    lambda j: j.window_average(lambda c: c.mean, 'a', np.nan, 0.1),
    lambda j: j.window_average(lambda c: c.mean, 'a', np.inf, 0.1),
])
def test_conditional_rejects_non_finite_arguments(call):
    """A NaN or infinite conditioning value or half-width raises ValueError. Regression: NaN values and an infinite
    window centre failed with LinAlgError in the inversion."""
    with pytest.raises(ValueError, match='finite'):
        call(pg.Coalescent(n=4).sfs.joint_distribution(1, 2))


def _dirac_five_epoch_joint():
    """The SFS joint of bins 1 and 2 of a Dirac coalescent of 10 lineages over five epochs."""
    return pg.Coalescent(n=10, model=pg.DiracCoalescent(psi=0.7, c=5), demography=pg.Demography(
        pop_sizes={'pop_0': {0: 2, 1.1: 0.3, 3.5: 0.5, 4.2: 8, 7.5: 2.3}})).sfs.joint_distribution(1, 2)


def test_joint_atom_matches_zero_reward_restriction():
    """The atom ``P(R_b = 0) = Phi(0, inf)`` of a five-epoch Dirac joint equals the absorption probability of the
    process restricted to the states without reward b, and the scalar and batched transforms agree. Regression: the
    atom was the transform at a large finite argument, which lost 1.5e-6 in the batched and 9e-8 in the scalar
    transform against a 40-digit reference."""
    joint = _dirac_five_epoch_joint()
    st = joint._setup
    zero = st['rb'] == 0

    # absorption without ever entering a state of positive reward b, epoch by epoch
    vec = np.append(st['alpha'][zero], 0.0)
    for T, t0, t1 in st['T_epochs'][:-1]:
        Td = T.toarray() if sp.issparse(T) else np.asarray(T)
        Q = np.zeros((zero.sum() + 1,) * 2)
        Q[:-1, :-1] = Td[np.ix_(zero, zero)]
        Q[:-1, -1] = -Td[zero].sum(1)
        vec = vec @ sla.expm(Q * (t1 - t0))
    Tm = st['T_epochs'][-1][0]
    Tm = (Tm.toarray() if sp.issparse(Tm) else np.asarray(Tm))
    ref = vec[-1] + vec[:-1] @ np.linalg.solve(-Tm[np.ix_(zero, zero)], -Tm[zero].sum(1))

    assert joint.lst(0.0, np.inf).real == pytest.approx(ref, rel=1e-13)
    assert joint.lst_batch([0.0], [np.inf])[0] == joint.lst(0.0, np.inf)


@pytest.mark.parametrize('pop_sizes', [{0: 1, 0.5: 0.2, 1.5: 2}, {0: 1}])
def test_joint_grid_matches_pointwise_transform(pop_sizes):
    """The grid of the joint transform, evaluated batched along its longer axis over several epochs and by one QZ
    decomposition per node over one, equals the pointwise transform, at infinite arguments included."""
    joint = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': pop_sizes})).sfs.joint_distribution(1, 2)
    sa = np.array([-0.5j, 0.3, -2j, np.inf])
    sb = np.array([-1j, 1j, 0.0, 0.7, -3j, np.inf])

    for x, y in ((sa, sb), (sb, sa)):
        ref = np.array([[joint.lst(a, b) for b in y] for a in x])
        np.testing.assert_allclose(joint._lst_grid(x, y), ref, rtol=1e-12, atol=1e-15)


def test_multi_epoch_lst_keeps_the_relative_precision_of_small_values():
    """The transform of two lineages kept apart by a migration barrier over a first epoch of length 100, where every
    transient state decays, keeps its relative precision far below one. Regression: the squarings of the exponential
    returned its diagonal as one plus a difference, which rounded the total branch length transform at 0.3 to 0 and
    left 3-4 digits of the tree height transform. References from a 40-digit mpmath evaluation."""
    coal = pg.Coalescent(
        n=pg.LineageConfig({'pop_0': 1, 'pop_1': 1}),
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 1}},
                                 migration_rates={('pop_0', 'pop_1'): {0: 0, 100: 1}, ('pop_1', 'pop_0'): {0: 0, 100: 1}})
    )
    height = coal.distribution(pg.TreeHeightReward())
    length = coal.distribution(pg.TotalBranchLengthReward())

    assert height.lst(0.3) == pytest.approx(5.213160428323223e-14, rel=1e-12, abs=0)
    assert height.lst(2 + 5j) == pytest.approx(3.811875917193463e-89 + 4.6738734372751355e-89j, rel=1e-12, abs=0)
    assert length.lst(0.3) == pytest.approx(3.267354762200201e-27, rel=1e-12, abs=0)


@pytest.mark.parametrize('pop_sizes', [{0: 1}, {0: 1, 0.5: 0.2, 1.5: 2}])
def test_atoms_are_exact(pop_sizes):
    """For n = 4 every tree has a doubleton branch, so ``P(L_2 = 0) = 0``, and the third bin is empty unless the tree
    is a caterpillar, with probability 1/3 whatever the demography. Regression: the atoms were transforms at a large
    finite argument, which left ``P(L_2 = 0)`` of order 1e-8."""
    coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': pop_sizes}))
    joint = coal.sfs.joint_distribution(2, 3)

    assert coal.sfs.bin(2).lst(np.inf) == 0.0
    assert joint._atoms['a0'] == 0.0 and joint._atoms['both0'] == 0.0
    assert joint._atoms['b0'] == pytest.approx(1 / 3, rel=1e-14)
    assert joint.conditional('b', 0.0).lst(np.inf) == 0.0


def test_quantile_below_the_atom_is_the_first_node():
    """Levels at or below the probability at the first node map to that node, also when the hazard starts with a run
    of equal values. Regression: the end of that run was returned, a positive quantile at level 0."""
    f = pg.Coalescent(n=4).total_branch_length.distribution().quantile
    nodes, hazard = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_array_equal(f._interp_quantile(np.array([0.0]), nodes, hazard), [0.0])
    assert f._interp_quantile(np.array([0.5]), nodes, hazard)[0] == pytest.approx(2.0 - np.log1p(-0.5))


def test_reward_pdf_at_nan_is_nan_and_leaves_the_grid_alone():
    """The reward density is NaN at NaN, like the CDF and quantile, and a NaN point does not extend the exact tail
    nodes. Regression: the density returned 0 and NaN marched the nodes to the 1 - 1e-12 target."""
    d = pg.Coalescent(n=4).total_branch_length.distribution()
    ref = pg.Coalescent(n=4).total_branch_length.distribution()
    ref.cdf(0.0)

    assert np.isnan(d.pdf(np.nan)) and np.isnan(d.cdf(np.nan))
    assert len(d.cdf._shared('cdf_exact', list)) == len(ref.cdf._shared('cdf_exact', list))

    out = d.pdf(np.array([np.nan, 1.0]))
    assert np.isnan(out[0]) and out[1] > 0


@pytest.mark.parametrize('cut', [None, 1.0])
def test_reward_zero_almost_surely_without_de_hoog_nodes(cut):
    """A reward that is 0 almost surely has a CDF of 1 and a quantile of 0 when the grid takes no de Hoog nodes.
    Regression: the grid was empty and np.interp raised ValueError."""
    Settings.dehoog_tail_quantile = cut
    coal = pg.Coalescent(n={'a': 2, 'b': 0}, demography=pg.Demography(pop_sizes={'a': 1, 'b': 1}))
    d = coal.total_branch_length.demes['b'].distribution()

    assert d.cdf(1.0) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(d.cdf(np.array([0.5, 2.0])), 1.0, atol=1e-9)
    assert d.quantile(0.5) == 0.0
    np.testing.assert_array_equal(d.pdf(np.array([0.5, 2.0])), 0.0)


def test_locating_pass_does_not_report_ringing(monkeypatch):
    """The discarded locating fit of the cosine expansion reports neither ringing nor truncation. Regression: its
    residual ripple was logged under the distribution's label."""
    d = pg.Coalescent(n=4).total_branch_length.distribution()
    calls = []
    monkeypatch.setattr(type(d), '_warn_if_nonmonotone', lambda self, *args, **kwargs: calls.append(args), raising=True)

    d.cdf._fit_cos(d._range(20.0), 64, warn=False)
    assert calls == []

    d.cdf._fit_cos(d._range(20.0), 64)
    assert len(calls) == 1


def test_far_tail_quantile_warns(caplog):
    """A reward quantile above 1 - 1e-9 logs that it is approximate, and a body quantile does not."""
    import logging
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)
    try:
        d = pg.Coalescent(n=4).total_branch_length.distribution()
        d.quantile(0.5)
        assert not any('approximate' in r.getMessage() for r in caplog.records)

        d.quantile(1 - 1e-10)
        assert any('approximate' in r.getMessage() for r in caplog.records)
    finally:
        log.removeHandler(caplog.handler)


def test_density_evaluators_reject_unknown_keywords():
    """A misspelt keyword to a density raises TypeError, as it does for the CDF. Regression: the densities accepted
    and discarded any keyword."""
    coal = pg.Coalescent(n=4)

    for pdf in (coal.total_branch_length.pdf, coal.total_branch_length.distribution().pdf,
                coal.total_branch_length.to_empirical(50, seed=0).pdf):
        with pytest.raises(TypeError):
            pdf([0.5, 1.0], foo=1)

    with pytest.raises(TypeError):
        coal.total_branch_length.cdf([0.5, 1.0], foo=1)


def test_spectrum_cdf_is_one_at_bins_that_are_zero_almost_surely():
    """The monomorphic and folded-away bins are zero almost surely, so their CDF is one from zero on, as for
    MsprimeCoalescent. Regression: the analytic spectrum CDFs returned zero there."""
    t = np.array([-1.0, 0.0, 1.5])
    point_mass = np.array([0.0, 1.0, 1.0])

    coal = pg.Coalescent(n=5)
    for sfs, zero in [(coal.sfs, [0, 5]), (coal.fsfs, [0, 3, 4, 5])]:
        F = np.asarray(sfs.cdf(t))
        for i in zero:
            np.testing.assert_array_equal(F[:, i], point_mass)
        assert np.asarray(sfs.cdf(1.5).data)[zero[0]] == 1

    mig = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1},
                        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})
    F = np.asarray(pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs.cdf(t))
    for config in [(0, 0), (2, 2)]:
        np.testing.assert_array_equal(F[(slice(None),) + config], point_mass)



def test_line_atom_density_rejects_unknown_keywords():
    """The density of a line-atom conditional raises TypeError on an unknown keyword. Regression: it was ignored."""
    joint = pg.Coalescent(n=2, loci=pg.LocusConfig(n=2, recombination_rate=1)).tree_height.loci.joint_distribution(
        0, 1)

    with pytest.raises(TypeError):
        joint.conditional('a', 1.0).pdf(1.0, foo=1)


def test_multi_epoch_conditional_mean_follows_the_derivative_identity():
    """On three epochs the conditional mean at v = 0.5 is within 0.1% of 0.80360104, the derivative identity inverted
    by de Hoog in 135-digit arithmetic at degree 70 (a sampler of 1.6e9 replicates gives 0.80330 +- 0.00036).
    Regression: it was the central difference of the transform at the calibration truncation, 0.26% off."""
    joint = pg.Coalescent(n=2, loci=pg.LocusConfig(n=2, recombination_rate=1), demography=pg.Demography(
        pop_sizes={'pop_0': {0: 2, 0.1: 0.3, 0.4: 1.3}})).tree_height.loci.joint_distribution(0, 1)

    cond = joint.conditional('a', 0.5)

    assert cond.mean == pytest.approx(0.8036010412, rel=1e-3)


@pytest.mark.parametrize('sizes, n, v, truth', [
    ({0: 1.2, 0.3: 10, 1: 0.8, 1.4: 10}, 4, 4.38, 7.7075),
    ({0: 1.0, 0.15: 2.0, 0.3: 0.5, 0.45: 2.0, 0.6: 0.5, 0.75: 2.0, 0.9: 0.5}, 3, 0.54, 0.2873024),
])
def test_multi_epoch_conditional_mean_is_stable_in_the_conditioning_value(sizes, n, v, truth):
    """The conditional mean E[R_2 | R_1 = v] of the SFS on 4_epoch_up_down_n_4 and 7_epoch_oscillating_n_3 does not
    move under shifts of v by a few ulp, and is within 0.1% of the derivative identity inverted by de Hoog in 135-digit
    arithmetic at degree 70. Regression: the identity was inverted by de Hoog in float64, whose quotient-difference
    recurrence amplified roundoff, so the mean swung between 7.735 and 7.990 over four ulp at v = 4.38, and was 0.9%
    off at v = 0.54."""
    joint = pg.Coalescent(n=n, demography=pg.Demography(pop_sizes={'pop_0': sizes})).sfs.joint_distribution(1, 2)
    eps = np.finfo(float).eps

    means = [joint.conditional('a', v * (1 + k * eps)).mean for k in (-1, 0, 1, 4)]

    assert means == pytest.approx([truth] * 4, rel=1e-3)
    assert np.ptp(means) <= 1e-9 * truth


def test_conditional_moments_warn_when_unresolved(caplog, monkeypatch):
    """The conditional moments under the extreme bottleneck at v = 1.2 converge in the truncation only at N0 = 480,
    where they move by less than 1e-3 when it is halved. Capped at 60, they move by 2.8e-2 and a warning is logged."""
    import logging
    from phasegen.distributions import reward
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)

    try:
        _ = _bottleneck_joint().conditional('a', 1.2).mean
        assert not any('moments are unresolved' in r.getMessage() for r in caplog.records)

        monkeypatch.setattr(reward, '_MOMENT_N0_MAX', 60)
        _ = _bottleneck_joint().conditional('a', 1.2).mean
        assert any('moments are unresolved' in r.getMessage() for r in caplog.records)
    finally:
        log.removeHandler(caplog.handler)


def test_conditional_moments_converge_across_epoch_jumps(caplog, monkeypatch):
    """The density of a locus tree height jumps at the epoch times 0.1 and 0.4 below the conditioning value 0.5. With
    the jumps subtracted, the conditional moments converge at the first truncation, and the mean agrees with that of
    4e7 sampled paths in a window of half-width 2.5e-3, 0.79910 +- 0.00408. Regression: the moments moved by 8.6e-3
    at the truncation 60 and converged like the inverse truncation, reaching 1.3e-3 only at 480."""
    import logging
    from phasegen.distributions import reward
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)

    def joint():
        return pg.Coalescent(n=2, loci=pg.LocusConfig(n=2, recombination_rate=1), demography=pg.Demography(
            pop_sizes={'pop_0': {0: 2, 0.1: 0.3, 0.4: 1.3}})).tree_height.loci.joint_distribution(0, 1)

    try:
        mean = joint().conditional('a', 0.5).mean
        monkeypatch.setattr(reward, '_MOMENT_N0_MAX', 60)
        capped = joint().conditional('a', 0.5).mean
        assert not any('moments are unresolved' in r.getMessage() for r in caplog.records)
    finally:
        log.removeHandler(caplog.handler)

    assert capped == pytest.approx(mean, rel=1e-6)
    assert mean == pytest.approx(0.79910, abs=4 * 0.00408)


@pytest.mark.parametrize('sparse', [False, True])
def test_batched_taylor_coefficients_match_single_points(sparse):
    """The batched Taylor coefficients of the joint transform equal those of ``lst_taylor`` at each point, on several
    epochs and on the dense and sparse last-epoch solve."""
    if sparse:
        Settings.closed_form_sparse_min_states = 1
    joint = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.2, 0.3: 10, 1: 0.8}})).sfs \
        .joint_distribution(1, 2)
    s = np.array([0.3, 1.0 + 2.0j, 5.0 - 7.0j])

    batch = joint._lst_taylor_batch(s, 'b', 2)

    np.testing.assert_allclose(batch, [joint.lst_taylor(x, 'b', order=2) for x in s], rtol=1e-12)
    assert joint._setup['sparse'] == sparse


def test_cosine_window_short_of_a_recent_crash_warns(caplog):
    """On a recent crash backward in time the window of the expansion ends before the long tail, and the CDF at the
    join with the de Hoog nodes is off by more than the bar, which is logged. A standard coalescent logs nothing, nor
    does the crash on a grid without a join. Regression: the CDF was off by 2.8e-2 with no warning."""
    import logging
    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)

    def warned(demography, cut=0.98) -> bool:
        caplog.clear()
        Settings.dehoog_tail_quantile = cut
        d = pg.Coalescent(n=10, demography=demography).total_branch_length.distribution()
        d.cdf(1.0)
        return any('COS CDF (window)' in r.getMessage() for r in caplog.records)

    try:
        crash = pg.Demography(pop_sizes={0: 0.01, 0.05: 1})
        assert warned(crash)
        assert not warned(pg.Demography())
        assert not warned(crash, cut=None)
        assert not warned(crash, cut=0.0)
    finally:
        log.removeHandler(caplog.handler)
