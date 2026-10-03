"""
Parity of the empirical joint distribution, spectra and marginal containers with their exact counterparts: the same
members, signatures, return types and plots, and values that agree within Monte Carlo error. Every tolerance is four
standard errors of the empirical estimate, plus a small allowance for the numerical error of the exact inversion where
the exact side is an inversion.
"""
import warnings

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import EmpiricalDistribution, EmpiricalJointDistribution, MsprimeCoalescent

#: Number of trajectories of the sampled distributions.
N = 40000

#: Seed of every sampler.
SEED = 7

#: Allowance for the numerical error of the exact joint CDF and density.
INVERSION_TOL = 3e-3

TWO_DEMES = pg.Demography(pop_sizes={'a': 1, 'b': 1.5}, migration_rates={('a', 'b'): 0.75, ('b', 'a'): 0.75})

#: Members of the exact classes without a sampled counterpart: time accumulation and the distribution of arbitrary
#: rewards need the Markov jump process, as do the transforms, the checks of the inversions and the samplers.
EXACT_ONLY = {
    'accumulate', 'get_accumulation', 'plot_accumulation', 'distribution', 'sample', 'sample_per_locus',
    'to_empirical', 'lst', 'lst_batch', 'lst_taylor', 'check_conditional_grid_moments', 'check_conditional_moments',
    'check_total_expectation', 'check_total_probability', 't_max', 'p_absorption', 'max_iter'
}

#: Further exact-only members per pair, with the reason.
EXACT_ONLY_BY_PAIR = {
    # the joint distribution of two arbitrary rewards needs the jump process
    'tree_height': {'joint'},
    'total_branch_length': {'joint'},
    'total_branch_length.demes[a]': {'joint', 'demes', 'loci'},
    'tree_height.loci[0]': {'joint', 'demes', 'loci'},
    # a per-deme marginal records neither nested marginals nor its deme-restricted mutation counts
    'sfs.demes[a]': {'demes', 'loci', 'mutation_layout', 'get_mutation_config', 'get_mutation_configs',
                     'generated_mass'},
}


@pytest.fixture(scope='module')
def one_deme():
    """The exact coalescent of five lineages and its sampled spectra."""
    coal = pg.Coalescent(n=5)

    return coal, coal.sfs.to_empirical(N, seed=SEED), coal.fsfs.to_empirical(N, seed=SEED + 1)


@pytest.fixture(scope='module')
def two_deme():
    """The exact coalescent of three lineages in two demes and its sampled joint spectrum."""
    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES)

    return coal, coal.jsfs.to_empirical(N, seed=SEED)


@pytest.fixture(scope='module')
def two_locus():
    """The exact coalescent of three lineages at two loci and its sampled two-locus spectrum."""
    coal = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)

    return coal, coal.sfs2.to_empirical(N, seed=SEED)


def _se_mean(x: np.ndarray) -> float:
    """Standard error of the sample mean of ``x``."""
    return float(np.std(x) / np.sqrt(len(x)))


# ---- joint distribution -----------------------------------------------------------------------------------------


def test_empirical_joint_cdf_is_a_function_object_matching_the_exact_one(one_deme):
    """The empirical joint CDF is called as the exact one, on the outer grid, and agrees within Monte Carlo error."""
    coal, emp, _ = one_deme
    exact, sampled = coal.sfs.joint(1, 2), emp.joint(1, 2)
    xs, ys = np.array([0.5, 1.0, 2.0, 0.5]), np.array([0.3, 0.8, 1.6])

    got = sampled.cdf(xs, ys)
    a, b = sampled._a, sampled._b
    brute = np.array([[np.mean((a <= x) & (b <= y)) for y in ys] for x in xs])

    assert isinstance(sampled.cdf, pg.distributions.JointCDF)
    assert got.shape == (4, 3)
    np.testing.assert_array_equal(got, brute)
    assert isinstance(sampled.cdf(1.0, 0.8), float)

    want = exact.cdf(xs, ys)
    se = np.sqrt(want * (1 - want) / N)
    assert np.all(np.abs(got - want) <= 4 * se + INVERSION_TOL)


@pytest.mark.parametrize('n_x, n_y', [(1, 1), (4, 4), (5, 4), (25, 25)])
def test_empirical_joint_cdf_counts_below_each_threshold_on_small_and_large_grids(one_deme, n_x, n_y):
    """
    The empirical joint CDF is the fraction of replicates below each pair of thresholds, whether the replicates are
    counted per point or binned against the grid, also for unsorted and repeated thresholds. A NaN threshold gives
    NaN, as on the exact joint CDF.
    """
    coal, emp, _ = one_deme
    sampled = emp.joint(1, 2)
    a, b = sampled._a, sampled._b
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(-0.5, 4, n_x), rng.uniform(-0.5, 3, n_y)
    xs[-1], ys[0] = np.inf, xs[0]

    brute = np.array([[np.mean((a <= x) & (b <= y)) for y in ys] for x in xs])
    np.testing.assert_array_equal(sampled.cdf._grid_values(xs, ys), brute)

    xs[0] = np.nan
    got = sampled.cdf._grid_values(xs, ys)
    assert np.isnan(got[0]).all()
    np.testing.assert_array_equal(got[1:], brute[1:])
    assert np.isnan(sampled.cdf(np.nan, 1.0)) and np.isnan(coal.sfs.joint(1, 2).cdf(np.nan, 1.0))
    assert np.isnan(sampled.cdf(1.0, [0.5, np.nan])[0, 1])

    assert sampled.cdf._grid_values(np.array([]), ys).shape == (0, n_y)

    # the cached ground truth is the CDF on its grid
    xs, ys, cdf, _ = sampled._surface(25, 0.95)
    np.testing.assert_array_equal(cdf, sampled.cdf(xs, ys))


def test_empirical_joint_pdf_is_the_cell_average_of_the_exact_density(one_deme):
    """
    The empirical joint density is the fraction of replicates in each cell over its area, which estimates the exact
    density averaged over the same cell, here the exact box probability from the joint CDF over the cell area.
    """
    coal, emp, _ = one_deme
    exact, sampled = coal.sfs.joint(1, 2), emp.joint(1, 2)
    xs, ys = np.linspace(0.3, 2.1, 4), np.linspace(0.2, 1.4, 4)
    area = (xs[1] - xs[0]) * (ys[1] - ys[0])

    got = sampled.pdf(xs, ys)

    edges_x, edges_y = np.append(xs, 2 * xs[-1] - xs[-2]), np.append(ys, 2 * ys[-1] - ys[-2])
    G = exact.cdf(edges_x, edges_y)
    box = G[1:, 1:] - G[:-1, 1:] - G[1:, :-1] + G[:-1, :-1]
    se = np.sqrt(box * (1 - box) / N) / area

    assert got.shape == (4, 4)
    assert np.all(np.abs(got - box / area) <= 4 * se + INVERSION_TOL / area)


def test_empirical_joint_pdf_excludes_the_axes_and_needs_two_points():
    """The cells hold the mass off the axes only, and a grid of fewer than two points is refused."""
    rng = np.random.default_rng(1)
    a, b = rng.exponential(size=1000), rng.exponential(size=1000)
    a[:300] = 0
    jd = EmpiricalJointDistribution(a, b)
    xs, ys = np.linspace(0, a.max() + 1e-9, 5), np.linspace(0, b.max() + 1e-9, 5)

    mass = jd.pdf(xs, ys) * np.outer(np.diff(np.append(xs, 2 * xs[-1] - xs[-2])),
                                     np.diff(np.append(ys, 2 * ys[-1] - ys[-2])))

    assert mass.sum() == pytest.approx(0.7)

    with pytest.raises(ValueError, match="at least two points"):
        jd.pdf(1.0, 1.0)


def test_empirical_joint_plots_as_the_exact_one(one_deme):
    """The joint CDF and density draw heatmaps and surfaces, the density at its cell centres, and plot_cdf works."""
    import matplotlib.pyplot as plt

    _, emp, _ = one_deme
    sampled = emp.joint(1, 2)

    sampled.cdf.plot(show=False)
    sampled.pdf.plot(show=False)
    sampled.pdf.plot_surface(show=False)
    with pytest.warns(DeprecationWarning):
        sampled.plot_cdf(show=False)
    plt.close('all')

    cdf, pdf = sampled.cdf._plot_data(n_points=12), sampled.pdf._plot_data(n_points=12)

    assert cdf.x[0] == 0 and cdf.title == "Joint CDF SFS bins (1, 2)"
    assert pdf.x[0] == pytest.approx((pdf.x[1] - pdf.x[0]) / 2)
    assert pdf.z.shape == (12, 12)


def test_empirical_joint_has_no_quantile(one_deme):
    """A joint distribution has no quantile function, exact or empirical."""
    coal, emp, _ = one_deme

    for joint in (coal.sfs.joint(1, 2), emp.joint(1, 2)):
        with pytest.raises(NotImplementedError, match="no quantile"):
            _ = joint.quantile


def test_empirical_window_average_matches_the_exact_one(one_deme):
    """
    The empirical window average of the conditional mean, the plain mean over the window, estimates the exact
    density-weighted window average.
    """
    coal, emp, _ = one_deme
    exact, sampled = coal.sfs.joint(1, 2), emp.joint(1, 2)
    value, h = 1.0, 0.25

    got = sampled.window_average(lambda c: c.mean, 'a', value, h)
    want = exact.window_average(lambda c: c.mean, 'a', value, h)
    window = sampled._b[np.abs(sampled._a - value) <= h]

    assert isinstance(got, float)
    assert abs(got - want) <= 4 * _se_mean(window) + 1e-3

    ys = np.array([0.5, 1.0])
    assert sampled.window_average(lambda c: c.cdf(ys), 'a', value, h).shape == (2,)

    for joint in (exact, sampled):
        with pytest.raises(ValueError, match="reaches 0"):
            joint.window_average(lambda c: c.mean, 'a', 0.1, 0.2)


# ---- standard deviations ----------------------------------------------------------------------------------------


def test_empirical_std_matches_the_exact_one(one_deme, two_deme, two_locus):
    """The standard deviation is the square root of the variance, of its type, and agrees with the exact one."""
    coal, sfs, _ = one_deme
    coal2, jsfs = two_deme
    coal3, sfs2 = two_locus
    th = coal.tree_height.to_empirical(N, seed=SEED)

    pairs = [(coal.tree_height, th), (coal.sfs, sfs), (coal2.jsfs, jsfs), (coal3.sfs2, sfs2),
             (coal.sfs.bin(2), sfs.bin(2)), (coal.sfs.joint(1, 2).marginal('b'), sfs.joint(1, 2).marginal('b'))]

    for exact, emp in pairs:
        std, var = emp.std, emp.var
        assert type(std) is type(exact.std) or (np.ndim(std) == 0 and np.ndim(exact.std) == 0)

        got = np.asarray(std.data if hasattr(std, 'data') else std, dtype=float)
        want = np.asarray(exact.std.data if hasattr(exact.std, 'data') else exact.std, dtype=float)
        np.testing.assert_allclose(got ** 2, np.asarray(var.data if hasattr(var, 'data') else var), rtol=1e-12)

        # delta method: se(std) = sqrt(mu_4 - var^2) / (2 std sqrt(N)), with the central moment of the stored samples
        m4 = np.asarray(EmpiricalDistribution(emp.samples).moment(4), dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            se = np.nan_to_num(np.sqrt(np.maximum(m4 - got ** 4, 0) / len(emp.samples)) / (2 * got))
        assert np.all(np.abs(got - want) <= 4 * se + 1e-9 * want), (type(exact).__name__, got, want)


# ---- spectra ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('folded', [False, True])
def test_empirical_sfs_get_cov_get_corr_and_bin_match_the_exact_ones(one_deme, folded):
    """``get_cov``, ``get_corr`` and ``bin`` of the empirical spectrum agree with the exact ones and validate alike."""
    coal, sfs, fsfs = one_deme
    exact, emp = (coal.fsfs, fsfs) if folded else (coal.sfs, sfs)
    s = emp.samples

    for i, j in [(1, 1), (1, 2), (2, 2)]:
        x, y = s[:, i] - s[:, i].mean(), s[:, j] - s[:, j].mean()
        assert emp.get_cov(i, j) == pytest.approx(float(emp.cov.data[i, j]))
        assert abs(emp.get_cov(i, j) - exact.get_cov(i, j)) <= 4 * _se_mean(x * y)
        assert abs(emp.get_corr(i, j) - exact.get_corr(i, j)) <= 0.03

    outside = 4 if folded else 0
    assert emp.get_cov(outside, 1) == exact.get_cov(outside, 1) == 0
    assert emp.get_corr(outside, 1) == exact.get_corr(outside, 1) == 0

    for d in (exact, emp):
        with pytest.raises(ValueError, match="integer from 0 to 5"):
            d.get_cov(6, 1)
        with pytest.raises(ValueError, match="polymorphic class"):
            d.bin(outside)

    b = emp.bin(2)
    t = np.array([0.2, 0.6, 1.5])
    want = np.asarray(exact.bin(2).cdf(t), dtype=float)

    assert isinstance(b, EmpiricalDistribution)
    assert abs(b.mean - exact.bin(2).mean) <= 4 * _se_mean(s[:, 2])
    assert np.all(np.abs(b.cdf(t) - want) <= 4 * np.sqrt(want * (1 - want) / N) + 1e-3)


def test_folded_tajima_estimators_equal_the_unfolded_ones_and_the_sampled_ones(one_deme):
    """
    The weights of the diversity estimators are symmetric in ``i`` and ``n - i``, so the exact folded estimators
    equal the unfolded ones, and the empirical folded spectrum estimates them.
    """
    coal, _, fsfs = one_deme
    n = 5
    i = np.arange(1, n)
    w_pi, w_w = 2 * i * (n - i) / (n * (n - 1)), np.full(n - 1, 1 / np.sum(1 / i))

    for name in ('theta_pi', 'theta_w', 'tajimas_d'):
        assert getattr(coal.fsfs, name) == pytest.approx(getattr(coal.sfs, name), rel=1e-10)

    s = fsfs.samples[:, 1:n]
    assert abs(fsfs.theta_pi - coal.fsfs.theta_pi) <= 4 * _se_mean(s @ w_pi)
    assert abs(fsfs.theta_w - coal.fsfs.theta_w) <= 4 * _se_mean(s @ w_w)
    assert abs(fsfs.tajimas_d - coal.fsfs.tajimas_d) <= 0.02


def test_empirical_joint_sfs_matches_the_exact_layout_shape_and_covariance(two_deme):
    """Shape, layouts, covariance and the per-locus marginal of the empirical joint SFS match the exact ones."""
    coal, emp = two_deme
    s = emp.samples

    assert emp.shape == coal.jsfs.shape
    for folded in (False, True):
        assert emp.mutation_layout(folded=folded) == coal.jsfs.mutation_layout(folded=folded)

    for a, b in [((1, 0), (1, 0)), ((1, 0), (0, 1)), ((2, 0), (1, 1))]:
        x = s[(slice(None),) + a] - s[(slice(None),) + a].mean()
        y = s[(slice(None),) + b] - s[(slice(None),) + b].mean()
        assert abs(emp.get_cov(a, b) - coal.jsfs.get_cov(a, b)) <= 4 * _se_mean(x * y)

    for d in (coal.jsfs, emp):
        with pytest.raises(ValueError, match="descendant configuration"):
            d.get_cov((0, 0), (1, 0))
        with pytest.raises(ValueError, match="descendant configuration"):
            d.bin(2, 1)

    assert abs(emp.bin(1, 1).mean - coal.jsfs.bin(1, 1).mean) <= 4 * _se_mean(s[:, 1, 1])

    assert list(emp.loci) == list(coal.jsfs.loci) == [0]
    assert emp.loci[0] is emp
    np.testing.assert_array_equal(emp.loci.cov, np.asarray(emp.var.data)[None, None])

    with pytest.raises(NotImplementedError, match="deme"):
        _ = emp.demes


def test_empirical_joint_sfs_mutation_configs_match_the_exact_ones():
    """The configuration frequencies simulated with msprime estimate the exact probabilities, in the exact layout."""
    theta, n_rep = 0.5, 4000
    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES)
    ms = MsprimeCoalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES, num_replicates=n_rep, n_threads=1,
                           parallelize=False, mutation_rate=theta, simulate_mutations=True, seed=SEED)

    assert ms.jsfs.mutation_layout() == coal.jsfs.mutation_layout()

    it = ms.jsfs.get_mutation_configs()
    for _ in range(12):
        config, p = next(it)
        q = coal.jsfs.get_mutation_config(config, theta=theta)
        assert ms.jsfs.get_mutation_config(tuple(config)) == p
        assert abs(p - q) <= 4 * np.sqrt(q * (1 - q) / n_rep) + 1 / n_rep

    assert 0 < ms.jsfs.generated_mass <= 1


def test_empirical_two_locus_sfs_matches_the_exact_layout_and_refusals(two_locus):
    """Shape and layouts match the exact two-locus spectrum, which has no univariate distributions or marginals."""
    coal, emp = two_locus

    assert emp.shape == coal.sfs2.shape
    for loci, folded in [((0, 1), False), ((0, 1), True), ((1,), False)]:
        assert emp.mutation_layout(loci, folded) == coal.sfs2.mutation_layout(loci, folded)

    for d in (coal.sfs2, emp):
        for name in ('cdf', 'pdf', 'quantile', 'loci', 'demes'):
            with pytest.raises(NotImplementedError):
                getattr(d, name)
        for name in ('bin', 'plot_cdf'):
            with pytest.raises(NotImplementedError):
                getattr(d, name)(1)

    jd = emp.joint(1, 2)
    assert jd.label == coal.sfs2.joint(1, 2).label
    with pytest.raises(ValueError, match="polymorphic class"):
        emp.joint(0, 1)


def test_empirical_two_locus_mutation_configs_match_the_exact_ones():
    """The two-locus configuration frequencies simulated with msprime estimate the exact probabilities."""
    theta, n_rep = 0.5, 4000
    loci = pg.LocusConfig(n=2, recombination_rate=1.0)
    coal = pg.Coalescent(n=3, loci=loci)
    ms = MsprimeCoalescent(n=3, loci=loci, num_replicates=n_rep, n_threads=1, parallelize=False,
                           mutation_rate=theta, simulate_mutations=True, seed=SEED)

    assert ms.sfs2.mutation_layout() == coal.sfs2.mutation_layout()

    it = ms.sfs2.get_mutation_configs()
    for _ in range(12):
        config, p = next(it)
        q = coal.sfs2.get_mutation_config(config, theta=theta)
        assert abs(p - q) <= 4 * np.sqrt(q * (1 - q) / n_rep) + 1 / n_rep


# ---- marginal containers ----------------------------------------------------------------------------------------


def test_empirical_marginal_containers_name_and_join_their_marginals():
    """The deme and locus containers expose ``demes`` and ``loci``, and the locus container the joint across loci."""
    coal = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    emp = coal.total_branch_length.to_empirical(N, seed=SEED)
    exact = coal.total_branch_length

    assert list(emp.loci.loci) == list(exact.loci.loci) == [0, 1]
    assert emp.loci.loci[1] is emp.loci[1]

    jd, jd_exact = emp.loci.joint(0, 1), exact.loci.joint(0, 1)
    a, b = jd._a - jd._a.mean(), jd._b - jd._b.mean()
    assert abs(jd.cov - jd_exact.cov) <= 4 * _se_mean(a * b)

    want = jd_exact.cdf(2.0, 2.0)
    assert abs(jd.cdf(2.0, 2.0) - want) <= 4 * np.sqrt(want * (1 - want) / N) + INVERSION_TOL

    for d in (exact.loci, emp.loci):
        with pytest.raises(ValueError, match="does not exist"):
            d.joint(0, 2)

    coal2 = pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES)
    demes = coal2.total_branch_length.to_empirical(N, seed=SEED).demes

    assert list(demes.demes) == list(coal2.total_branch_length.demes.demes) == ['a', 'b']
    assert abs(demes.demes['a'].mean - coal2.total_branch_length.demes['a'].mean) <= 4 * _se_mean(
        demes.demes['a'].samples)

    sfs_loci = coal.sfs2.to_empirical(100, seed=SEED)
    with pytest.raises(NotImplementedError):
        _ = sfs_loci.loci


# ---- boundaries -------------------------------------------------------------------------------------------------


def test_empirical_parity_members_at_their_boundaries():
    """
    Thresholds at the ends of the support, dropped samples and a joint spectrum restored without its lineages: the
    statistics retained by a drop still serve, the members that need the samples refuse, and the layout falls back to
    one deme per axis.
    """
    coal = pg.Coalescent(n=4)
    emp = coal.sfs.to_empirical(2000, seed=SEED)
    jd = emp.joint(1, 2)

    assert jd.cdf(np.inf, np.inf) == 1.0
    assert jd.cdf(-1.0, np.inf) == 0.0
    np.testing.assert_array_equal(jd.cdf([np.inf, -1.0], [0.0]), [[np.mean(jd._b <= 0)], [0.0]])

    cov, corr = emp.get_cov(1, 2), emp.get_corr(1, 2)
    emp._touch(np.linspace(0, 3, 10))
    emp._drop()

    assert emp.get_cov(1, 2) == cov and emp.get_corr(1, 2) == corr
    assert np.isfinite(emp.tajimas_d) and emp.std.data[1] > 0
    for f in (lambda: emp.bin(1), lambda: emp.joint(1, 2)):
        with pytest.raises(ValueError, match="dropped"):
            f()

    loci = pg.Coalescent(n=3, loci=2, recombination_rate=1.0).tree_height.to_empirical(500, seed=SEED)
    loci._touch(np.linspace(0, 3, 10))
    loci._drop()
    with pytest.raises(ValueError, match="dropped"):
        loci.loci.joint(0, 1)

    jsfs = pg.distributions.EmpiricalJointSFSDistribution(moments=np.zeros((3, 3, 2)))
    assert jsfs.shape == (3, 2)
    assert jsfs.mutation_layout().axes == ('pop_0', 'pop_1')
    assert len(jsfs.mutation_layout()) == 4


# ---- member parity ----------------------------------------------------------------------------------------------


def _public(obj) -> set:
    """The public members of the class of ``obj``."""
    return {k for k in dir(type(obj)) if not k.startswith('_')}


def test_empirical_classes_offer_the_members_of_the_exact_ones(one_deme, two_deme, two_locus):
    """
    Every public member of an exact class, except the exact-only ones, exists on its empirical counterpart, so that a
    member added to the exact class without its sampled counterpart fails here.
    """
    coal, sfs, fsfs = one_deme
    coal2, jsfs = two_deme
    coal3, sfs2 = two_locus
    th2 = coal3.tree_height.to_empirical(1000, seed=SEED)
    tbl = coal2.total_branch_length.to_empirical(1000, seed=SEED)
    sfs_demes = pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES).sfs.to_empirical(1000, seed=SEED)

    pairs = {
        'tree_height': (coal.tree_height, coal.tree_height.to_empirical(1000, seed=SEED)),
        'total_branch_length': (coal2.total_branch_length, tbl),
        'sfs': (coal.sfs, sfs),
        'fsfs': (coal.fsfs, fsfs),
        'jsfs': (coal2.jsfs, jsfs),
        'sfs2': (coal3.sfs2, sfs2),
        'sfs.bin': (coal.sfs.bin(1), sfs.bin(1)),
        'sfs.joint': (coal.sfs.joint(1, 2), sfs.joint(1, 2)),
        'jsfs.joint': (coal2.jsfs.joint((1, 0), (0, 1)), jsfs.joint((1, 0), (0, 1))),
        'sfs2.joint': (coal3.sfs2.joint(1, 2), sfs2.joint(1, 2)),
        'sfs.joint.cdf': (coal.sfs.joint(1, 2).cdf, sfs.joint(1, 2).cdf),
        'sfs.joint.pdf': (coal.sfs.joint(1, 2).pdf, sfs.joint(1, 2).pdf),
        'sfs.joint.marginal': (coal.sfs.joint(1, 2).marginal('a'), sfs.joint(1, 2).marginal('a')),
        'sfs.joint.conditional': (coal.sfs.joint(1, 2).conditional('a', 1.0), sfs.joint(1, 2).conditional('a', 1.0)),
        'total_branch_length.demes': (coal2.total_branch_length.demes, tbl.demes),
        'total_branch_length.demes[a]': (coal2.total_branch_length.demes['a'], tbl.demes['a']),
        'tree_height.loci': (coal3.tree_height.loci, th2.loci),
        'tree_height.loci[0]': (coal3.tree_height.loci[0], th2.loci[0]),
        'sfs.demes': (pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES).sfs.demes, sfs_demes.demes),
        'sfs.demes[a]': (pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES).sfs.demes['a'],
                         sfs_demes.demes['a']),
    }

    missing = {
        name: sorted(_public(exact) - _public(emp) - EXACT_ONLY - EXACT_ONLY_BY_PAIR.get(name, set()))
        for name, (exact, emp) in pairs.items()
    }

    assert not {k: v for k, v in missing.items() if v}
