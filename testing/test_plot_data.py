"""
Tests for the plot-data API: the ``_plot_data`` methods of distribution functions, ``_plot_accumulation_data``,
:meth:`Demography._plot_data`, and the bootstrap accessors of :class:`Inference`. The data must be exactly what the
corresponding ``plot`` methods draw, with default grids taken from :class:`Settings`.
"""
import inspect

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import phasegen as pg
from phasegen.distributions.empirical import EmpiricalDistribution, EmpiricalSFSDistribution
from phasegen.settings import Settings


def test_curve_plot_data_is_the_evaluated_function_and_what_plot_draws():
    """For the exact tree height, an accumulated reward and a single SFS bin, the curve is the function evaluated on
    the data grid, and ``plot`` draws that curve."""
    coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={0: 1, 1: 10}))

    for f in (coal.tree_height.pdf, coal.total_branch_length.cdf, coal.sfs.bin(2).quantile):
        data = f._plot_data(n_points=15)

        assert data.x.shape == (15,) and data.y.shape == (1, 15)
        np.testing.assert_allclose(data.y[0], f(data.x), rtol=1e-12, atol=1e-12)

        _, ax = plt.subplots()
        f.plot(ax=ax, show=False, n_points=15)
        assert len(ax.lines) == 1
        np.testing.assert_allclose(ax.lines[0].get_ydata(), data.y[0], rtol=1e-12, atol=1e-12)
        plt.close('all')


def test_plot_signatures_select_data_and_style_the_curves():
    """The grid, ``n_points`` and ``bins`` / ``configs`` are explicit parameters of each function's ``plot``, the
    remaining keyword arguments style every drawn curve, and a misspelled data argument reaches matplotlib, which
    rejects it."""
    coal = pg.Coalescent(n=5)
    jsfs = pg.Coalescent(n={'a': 1, 'b': 1}, demography=pg.Demography(pop_sizes={'a': 1, 'b': 1},
                                                                        migration_rates={('a', 'b'): 1})).jsfs

    def names(f):
        return list(inspect.signature(f.plot).parameters)[:4]

    assert names(coal.tree_height.pdf) == ['ax', 't', 'n_points', 'show']
    assert names(coal.sfs.bin(1).cdf) == ['ax', 't', 'n_points', 'show']
    assert names(coal.sfs.bin(1).quantile) == ['ax', 'q', 'n_points', 'show']
    assert names(coal.sfs.pdf) == ['ax', 't', 'bins', 'n_points']
    assert names(coal.sfs.quantile) == ['ax', 'q', 'bins', 'n_points']
    assert names(jsfs.cdf) == ['ax', 't', 'configs', 'n_points']

    ax = coal.sfs.pdf.plot(bins=[1, 3], n_points=9, show=False, lw=3, alpha=0.4)
    assert len(ax.lines) == 2
    assert all(line.get_linewidth() == 3 and line.get_alpha() == 0.4 for line in ax.lines)

    with pytest.raises(AttributeError, match='bin'):
        coal.sfs.pdf.plot(bin=[1], n_points=9, show=False)


def test_curve_plot_data_default_grid_from_settings():
    """The default grid has ``Settings.plot_n_grid`` points and ends at the ``Settings.plot_endpoint_quantile``
    quantile, or spans the central probability range for a quantile function."""
    Settings.plot_n_grid = 17
    Settings.plot_endpoint_quantile = 0.8
    th = pg.Coalescent(n=5).tree_height

    cdf = th.cdf._plot_data()
    np.testing.assert_allclose(cdf.x, np.linspace(0, th.quantile(0.8), 17))
    assert (cdf.xlabel, cdf.ylabel, cdf.title, cdf.labels) == ('t', 'F(t)', 'CDF', [''])

    np.testing.assert_allclose(th.quantile._plot_data().x, np.linspace(0.2, 0.8, 17))


def test_sfs_plot_data_selects_bins():
    """A spectrum has one curve per requested bin, labelled by bin, each the bin's own function, and the default grid
    ends at the largest requested bin's endpoint quantile."""
    Settings.plot_n_grid = 21
    Settings.plot_endpoint_quantile = 0.8
    sfs = pg.Coalescent(n=5).sfs

    data = sfs.pdf._plot_data(bins=[1, 3])

    assert data.labels == ['1', '3'] and data.legend_title == 'bin'
    assert data.y.shape == (2, 21)
    np.testing.assert_allclose(data.y[1], sfs.bin(3).pdf(data.x), rtol=1e-10, atol=1e-12)
    assert data.x[-1] == pytest.approx(max(sfs.bin(1).quantile(0.8), sfs.bin(3).quantile(0.8)), rel=0.02)

    assert sfs.cdf._plot_data().labels == ['1', '2', '3', '4']
    assert pg.Coalescent(n=5).fsfs.cdf._plot_data(n_points=4).labels == ['1', '2']


def test_empirical_plot_data_scalar_cell_average_density():
    """The empirical density of a sample vector is the cell average over cells coarsened with the sample size, drawn
    at the cell centres."""
    samples = np.random.default_rng(0).exponential(size=900)
    d = EmpiricalDistribution(samples)

    data = d.pdf._plot_data()
    n_cells = 30  # sqrt(900), within [20, 100]
    edges = np.linspace(0, np.quantile(samples, Settings.plot_endpoint_quantile), n_cells)
    width = edges[1] - edges[0]

    np.testing.assert_allclose(data.x, edges + width / 2)
    counts, _ = np.histogram(samples, bins=np.append(edges, edges[-1] + width))
    np.testing.assert_allclose(data.y[0], counts / samples.size / width)
    assert (data.labels, data.title, data.legend_title) == ([''], 'PDF', None)


def test_empirical_plot_data_per_bin():
    """A replicate-by-bin sample matrix has one curve per polymorphic bin (or per requested bin), each the empirical
    function of that bin's column."""
    rng = np.random.default_rng(1)
    samples = np.zeros((500, 5))
    samples[:, 1:4] = rng.exponential(scale=[1.0, 2.0, 3.0], size=(500, 3))
    d = EmpiricalSFSDistribution(samples)

    quantile = d.quantile._plot_data(n_points=9)
    assert quantile.labels == ['1', '2', '3'] and quantile.title == 'SFS bin quantile functions'
    np.testing.assert_allclose(quantile.y[2], np.quantile(samples[:, 3], np.linspace(0.1, 0.9, 9)))

    cdf = d.cdf._plot_data(bins=[2], n_points=9)
    assert cdf.labels == ['2']
    np.testing.assert_allclose(cdf.x[-1], np.quantile(samples[:, 2], 0.9))
    np.testing.assert_allclose(cdf.y[0], np.interp(cdf.x, np.sort(samples[:, 2]), np.arange(1, 501) / 500, left=0))


def test_joint_plot_data_resolution_from_settings_and_drawn_values():
    """The joint grid resolution comes from the settings for a density heatmap, a density surface and a CDF, the
    values are the joint function on the grid, and the heatmap draws them."""
    Settings.plot_joint_pdf_n_grid = 9
    Settings.plot_joint_pdf_surface_n_grid = 7
    Settings.plot_joint_cdf_n_grid = 5
    jd = pg.Coalescent(n=5).sfs.joint(1, 2)

    assert jd.pdf._plot_data().z.shape == (9, 9)
    assert jd.pdf._plot_data(surface=True).z.shape == (7, 7)

    cdf = jd.cdf._plot_data()
    assert cdf.z.shape == (5, 5) and (cdf.vmin, cdf.vmax) == (0.0, 1.0)
    np.testing.assert_allclose(cdf.z, jd.cdf(cdf.x, cdf.y), atol=1e-12)
    assert cdf.x[-1] <= jd.marginal('a').quantile(Settings.plot_endpoint_quantile) + 1e-12
    assert cdf.title == 'Joint CDF SFS bins (1, 2)'

    ax = jd.cdf.plot(show=False)
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_array()).ravel(), cdf.z.T.ravel())


def test_accumulation_plot_data():
    """The accumulation data is the moment accumulated at the end times, one row per polymorphic bin for a spectrum,
    with the default end times up to the tree-height endpoint quantile."""
    coal = pg.Coalescent(n=5)
    t = np.linspace(0, 2, 6)

    sfs = coal.sfs._plot_accumulation_data(1, t)
    assert sfs.labels == ['1', '2', '3', '4'] and sfs.title == 'SFS Moment accumulation (Unit)'
    np.testing.assert_allclose(sfs.y, coal.sfs.accumulate(1, t).T[1:5], rtol=1e-10, atol=1e-12)
    assert sfs.y[:, -1].sum() > sfs.y[:, 1].sum() > 0

    Settings.plot_n_grid = 11
    th = coal.tree_height._plot_accumulation_data(2)
    np.testing.assert_allclose(th.x, np.linspace(0, coal.tree_height.quantile(Settings.plot_endpoint_quantile), 11))
    np.testing.assert_allclose(th.y[0], coal.tree_height.accumulate(2, th.x), rtol=1e-10, atol=1e-12)
    assert th.title == 'Moment accumulation (TreeHeight, TreeHeight)'

    ax = coal.sfs.plot_accumulation(end_times=t, show=False)
    np.testing.assert_allclose(ax.lines[2].get_ydata(), sfs.y[2])


def test_demography_plot_data():
    """Population sizes and migration rates are evaluated at the given times and named by population and by pair,
    with the default times from the settings."""
    d = pg.Demography(
        pop_sizes={'a': {0: 1, 1: 3}, 'b': {0: 2}},
        migration_rates={('a', 'b'): {0: 0.1, 2: 0.5}}
    )
    t = np.array([0.5, 1.5, 2.5])

    data = d._plot_data(t)
    assert data.labels == ['a', 'b', 'a->b', 'b->a']
    np.testing.assert_allclose(data.y, [[1, 3, 3], [2, 2, 2], [0.1, 0.1, 0.5], [0, 0, 0]])

    assert d._plot_data(t, kind='pop_sizes').labels == ['a', 'b']
    assert d._plot_data(t, kind='migration').title == 'Migration rate trajectory'

    with pytest.raises(ValueError):
        d._plot_data(t, kind='sizes')

    Settings.plot_demography_end_time = 3.0
    Settings.plot_demography_n_grid = 11
    np.testing.assert_allclose(d._plot_data().x, np.linspace(0, 3, 11))

    ax = d.plot_pop_sizes(t=t, show=False)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 3, 3])


def test_inference_bootstrap_plot_data():
    """The bootstrap values are the parameter columns as floats, each bootstrap demography is the ``coal`` callback's
    demography at that row, and the default times end at the inferred tree-height quantile of the settings."""
    inf = pg.Inference(
        x0=dict(t=0.5, Ne=0.5),
        bounds=dict(t=(0, 2), Ne=(0.1, 1)),
        coal=lambda t, Ne: pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})),
        loss=lambda coal, observation: 0.0,
        parallelize=False
    )

    with pytest.raises(RuntimeError):
        inf._plot_demography_data()

    inf.dist_inferred = inf.get_coal(t=0.5, Ne=0.5)
    inf.bootstraps = pd.DataFrame([[0.3, 0.2, 1.0, ''], [1.2, 0.7, 2.0, '']], columns=['t', 'Ne', 'loss', 'result'])

    values = inf._bootstrap_values
    assert values.dtype == float
    np.testing.assert_array_equal(values, [[0.3, 0.2], [1.2, 0.7]])

    demographies = inf._bootstrap_demographies
    times = np.array([0.1, 0.8, 1.5])
    np.testing.assert_allclose(demographies[0]._plot_data(times, 'pop_sizes').y, [[1, 0.2, 0.2]])
    np.testing.assert_allclose(demographies[1]._plot_data(times, 'pop_sizes').y, [[1, 1, 0.7]])

    Settings.plot_inference_quantile = 0.5
    Settings.plot_inference_n_grid = 13
    inferred, bootstraps = inf._plot_demography_data(kind='pop_sizes')
    np.testing.assert_allclose(inferred.x, np.linspace(0, inf.dist_inferred.tree_height.quantile(0.5), 13))
    assert len(bootstraps) == 2 and bootstraps[1].labels == ['pop_0']
    np.testing.assert_allclose(bootstraps[1].y, demographies[1]._plot_data(inferred.x, 'pop_sizes').y)

    assert inf._plot_demography_data(include_bootstraps=False)[1] == []

    ax = inf.plot_pop_sizes(show=False)
    assert len(ax.lines) == 3


def test_inference_demography_plot_colours_each_series_and_lists_it_once():
    """Every series of a multi-population demography has its own colour, shared by its bootstrap trajectories, and
    one legend entry. Regression: all series were drawn in C0 and each bootstrap added its own legend entries."""
    inf = pg.Inference(
        x0=dict(m=0.5),
        bounds=dict(m=(0.1, 1)),
        coal=lambda m: pg.Coalescent(
            n={'a': 2, 'b': 2},
            demography=pg.Demography(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): m, ('b', 'a'): m})
        ),
        loss=lambda coal, observation: 0.0,
        parallelize=False
    )
    inf.dist_inferred = inf.get_coal(m=0.5)
    inf.bootstraps = pd.DataFrame([[0.3, 1.0, ''], [0.7, 2.0, '']], columns=['m', 'loss', 'result'])

    ax = inf.plot_demography(show=False)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(labels) == len(set(labels))

    colours = {}
    for line in ax.lines:
        colours.setdefault(line.get_label() if not line.get_label().startswith('_') else None, set()).add(line.get_color())

    named = {k: v for k, v in colours.items() if k is not None}
    assert len({c for v in named.values() for c in v}) == len(named) > 1
    assert colours[None] <= {c for v in named.values() for c in v}


def test_plot_onto_a_passed_ax_saves_and_returns_that_ax(tmp_path):
    """Plotting onto axes of a figure that is not current saves that figure and returns those axes. Regression: the
    current figure was saved, blank here, and plt.gca() was returned."""
    fig, ax = plt.subplots()
    plt.figure()

    out = pg.Coalescent(n=3).tree_height.cdf.plot(ax=ax, show=False, file=str(tmp_path / 'cdf.png'), n_points=15)

    assert out is ax
    assert len(ax.lines) > 0
    assert plt.imread(tmp_path / 'cdf.png')[..., :3].std() > 0
    plt.close('all')


def test_univariate_plots_share_the_grid_keyword_t():
    """Every univariate function takes its grid as ``t``, so an exact curve and its empirical reference are drawn with
    the same call. Regression: the reward and spectrum functions took ``x`` and raised on ``t``, while the tree height
    and the empirical functions took ``t`` and raised on ``x``."""
    coal = pg.Coalescent(n=4)
    grid = np.linspace(0.1, 3, 20)

    functions = (
        coal.tree_height.cdf, coal.tree_height.pdf, coal.total_branch_length.cdf,
        coal.distribution(pg.TotalBranchLengthReward()).pdf, coal.sfs.bin(1).cdf,
        coal.tree_height.to_empirical(200, seed=0).cdf
    )
    for f in functions:
        ax = f.plot(t=grid, show=False)
        np.testing.assert_allclose(ax.lines[-1].get_xdata(), grid)
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), f(grid), rtol=1e-12, atol=1e-12)
        plt.close('all')

    ax = coal.sfs.cdf.plot(t=grid, show=False)
    assert len(ax.lines) == 3
    np.testing.assert_allclose(ax.lines[0].get_xdata(), grid)
    plt.close('all')


def test_single_reward_plot_has_no_legend():
    """A single accumulated reward draws one unlabelled curve without a legend. Regression: the curve of a
    ``PhaseTypeDistribution`` was labelled with its kind, ``'cdf'``, ``'pdf'`` or ``'quantile'``."""
    tbl = pg.Coalescent(n=4).total_branch_length

    for f in (tbl.cdf, tbl.pdf, tbl.quantile):
        assert f._plot_data(n_points=9).labels == ['']

        ax = f.plot(n_points=9, show=False)
        assert ax.get_legend() is None
        plt.close('all')


def test_labelled_multi_curve_plots_name_each_curve():
    """A label passed to a plot of several curves names each curve as ``'<label> (<legend title> <curve label>)'``,
    as the R package does, and a single curve takes the label alone. Regression: every curve of a spectrum got the
    same legend entry."""
    coal = pg.Coalescent(n=4)

    def legend(ax):
        return [text.get_text() for text in ax.get_legend().get_texts()]

    ax = coal.sfs.pdf.plot(n_points=9, show=False, label='exact')
    coal.sfs.to_empirical(200, seed=0).pdf.plot(n_points=9, ax=ax, show=False, label='sampled')
    assert legend(ax) == [f'{source} (bin {i})' for source in ('exact', 'sampled') for i in (1, 2, 3)]
    assert ax.get_legend().get_title().get_text() == ''
    plt.close('all')

    ax = coal.sfs.plot_accumulation(end_times=np.linspace(0, 2, 5), show=False, label='exact')
    assert legend(ax) == ['exact (bin 1)', 'exact (bin 2)', 'exact (bin 3)']
    plt.close('all')

    ax = coal.tree_height.pdf.plot(n_points=9, show=False, label='exact')
    assert legend(ax) == ['exact']
    plt.close('all')


def test_empirical_folded_sfs_plots_its_polymorphic_bins():
    """The empirical folded SFS draws the bins ``1, ..., n // 2`` of the analytic one. Regression: its default curves
    ran over ``1, ..., n - 1`` and drew the empty upper bins as constant curves."""
    fsfs = pg.Coalescent(n=5).fsfs

    for kind in ('cdf', 'pdf', 'quantile'):
        assert getattr(fsfs.to_empirical(300, seed=0), kind)._plot_data().labels == ['1', '2']

    assert pg.Coalescent(n=5).sfs.to_empirical(300, seed=0).cdf._plot_data().labels == ['1', '2', '3', '4']


def test_joint_plots_draw_on_one_fresh_axes():
    """Without ``ax`` each joint plot draws onto a fresh figure, and a surface replaces passed 2D axes by 3D ones.
    Regression: repeated heatmaps stacked meshes and colorbars on one axes, every surface opened a figure that was
    never closed, and a 2D ``ax`` raised."""
    Settings.plot_joint_cdf_n_grid = 5
    jd = pg.Coalescent(n=5).sfs.joint(1, 2)
    plt.close('all')

    for _ in range(2):
        ax = jd.cdf.plot(show=False)
    assert len(ax.collections) == 1 and len(ax.figure.axes) == 2  # the heatmap and its colorbar

    plt.close('all')
    axes = [jd.cdf.plot_surface(show=False) for _ in range(2)]
    assert all(ax.name == '3d' and len(ax.figure.axes) == 1 for ax in axes)
    assert len(plt.get_fignums()) == 2 and axes[0].figure is not axes[1].figure

    fig, (left, right) = plt.subplots(1, 2)
    ax = jd.cdf.plot_surface(ax=right, show=False)
    assert ax.name == '3d' and ax.figure is fig and len(fig.axes) == 2 and left in fig.axes
    plt.close('all')


def test_proportional_joint_plot_skips_the_2d_expansion():
    """The plotting grid of a pair with one reward a multiple of the other takes the window ends without building the
    2D cosine expansion, which its CDF does not use and its density refuses."""
    Settings.plot_joint_cdf_n_grid = 5
    jd = pg.Coalescent(n=5).sfs.joint(1, 1)
    assert jd._ratio == 1.0

    data = jd.cdf._plot_data()
    with pytest.raises(NotImplementedError):
        jd.pdf.plot(show=False)

    assert Settings.cache and 'cos2d' not in jd.__dict__.get('_cos_cache', {})
    assert data.x[-1] <= jd._cos2d_window('a')

    other = pg.Coalescent(n=5).sfs.joint(1, 2)
    assert (other._cos2d_window('a'), other._cos2d_window('b')) == (other._cos2d['ba'], other._cos2d['bb'])
    assert 'cos2d' in other.__dict__['_cos_cache']
    plt.close('all')


def test_plots_without_ax_leave_other_figures_alone():
    """Without ``ax`` a plot draws on a new figure and leaves the open figures as they are. Regression: the current
    figure was closed and the plot went onto the current axes of another open figure, so a joint heatmap was drawn
    onto a 3D axes and a curve plot raised TypeError from ``Axes3D.fill_between``."""
    coal = pg.Coalescent(n=4)
    Settings.plot_joint_cdf_n_grid = 5
    jd = coal.sfs.joint(1, 2)
    inf = pg.Inference(
        x0=dict(m=0.5),
        bounds=dict(m=(0.1, 1)),
        coal=lambda m: pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={0: 1, 1: m})),
        loss=lambda coal, observation: 0.0,
        parallelize=False
    )
    inf.dist_inferred = inf.get_coal(m=0.5)
    plt.close('all')

    other = plt.figure().add_subplot(projection='3d').figure
    current = plt.subplots()[0]

    plots = (
        lambda: jd.cdf.plot(show=False),
        lambda: coal.tree_height.pdf.plot(show=False),
        lambda: coal.sfs.plot_accumulation(k=1, show=False),
        lambda: coal.demography.plot(show=False),
        lambda: inf.plot_pop_sizes(show=False, include_bootstraps=False),
        lambda: inf.plot_demography(show=False, include_bootstraps=False)
    )
    for plot in plots:
        plt.figure(current.number)
        ax = plot()
        assert ax.name != '3d' and ax.figure not in (other, current)
        assert plt.fignum_exists(other.number) and plt.fignum_exists(current.number)
        assert len(other.axes) == 1 and not current.axes[0].has_data()

    plt.close('all')


def test_plots_with_clear_false_draw_onto_the_current_axes():
    """Without ``ax`` and with ``clear=False`` a plot draws onto the current axes and opens no figure."""
    coal = pg.Coalescent(n=4)
    plt.close('all')

    ax = coal.tree_height.pdf.plot(show=False, n_points=9)
    for plot in (
            lambda: coal.tree_height.cdf.plot(show=False, clear=False, n_points=9),
            lambda: coal.sfs.bin(1).cdf.plot(show=False, clear=False, n_points=9)
    ):
        assert plot() is ax

    assert len(ax.lines) == 3 and plt.get_fignums() == [ax.figure.number]
    plt.close('all')


def test_spectrum_functions_put_the_grid_on_the_first_axis():
    """The exact and empirical per-bin cdf, pdf and quantile and the moment accumulation of a spectrum return
    ``(len(t), n + 1)``. Regression: the empirical cdf and pdf and the accumulation returned the transpose."""
    coal = pg.Coalescent(n=4)
    t = np.linspace(0.1, 2, 7)
    q = np.linspace(0.2, 0.8, 3)
    sampled = coal.sfs.to_empirical(2000, seed=0)

    for sfs in (coal.sfs, sampled):
        assert np.asarray(sfs.cdf(t)).shape == (7, 5)
        assert np.asarray(sfs.pdf(t)).shape == (7, 5)
        assert np.asarray(sfs.quantile(q)).shape == (3, 5)

    acc = coal.sfs.accumulate(1, t)
    assert acc.shape == (7, 5)
    np.testing.assert_allclose(acc[-1], coal.sfs.moment(1, end_time=t[-1]).data, rtol=1e-10)

    # the empirical cdf of each bin is that of its own column of samples
    np.testing.assert_array_equal(np.asarray(sampled.cdf(t))[:, 2],
                                  EmpiricalDistribution(sampled.samples[:, 2]).cdf(t))


def test_accumulation_plot_takes_the_batched_mean_path():
    """The mean accumulation that ``plot_accumulation`` draws takes the batched path for the spectrum's own reward,
    also where the flattening does not apply. Regression: the plot passed that reward explicitly and was evaluated
    per bin, 6 to 19 times slower."""
    sfs = pg.Coalescent(n=5, model=pg.BetaCoalescent(alpha=1.5)).sfs
    assert not sfs._flattening_applies(1)

    t = np.linspace(0, 2, 5)
    batched = sfs._accumulate_batched(1, sfs._get_indices(), t, (sfs.reward,), None)
    assert batched is not None
    np.testing.assert_allclose(sfs._plot_accumulation_data(1, t).y, batched, rtol=1e-12)
    np.testing.assert_allclose(batched, [sfs.get_accumulation(1, i, t) for i in sfs._get_indices()], rtol=1e-10,
                               atol=1e-14)


def test_spectrum_functions_keep_the_shape_of_the_points():
    """The per-bin functions of a spectrum return ``t.shape`` followed by the spectrum's shape for any array ``t``.
    Regression: a 2-D ``t`` raised a broadcast ValueError."""
    coal = pg.Coalescent(n={'a': 1, 'b': 1}, demography=pg.Demography(pop_sizes={'a': 1, 'b': 1},
                                                                        migration_rates={('a', 'b'): 1}))
    t = np.array([[0.3, 0.8, 1.5], [2.0, 2.5, 3.0]])
    q = np.array([[0.2], [0.7]])

    for spectrum, shape in ((pg.Coalescent(n=4).sfs, (5,)), (coal.jsfs, (2, 2))):
        for f, x in ((spectrum.cdf, t), (spectrum.pdf, t), (spectrum.quantile, q)):
            out = f(x)
            assert out.shape == x.shape + shape
            np.testing.assert_array_equal(out.reshape((-1,) + shape), f(x.ravel()))


@pytest.mark.parametrize('kind', ['cdf', 'pdf', 'quantile'])
def test_exact_and_empirical_curves_share_titles_and_axes(kind):
    """
    The exact and the empirical curves of a conditional, a marginal, the tree height and its deme view, and a
    spectrum carry the same title and axis labels: the label of a conditional leads the title, and the x-axis is ``t``
    for the tree height and ``x`` for any other reward. Regression: the empirical curves were always titled without a
    label and put ``t`` on the x-axis, and the exact deme views named it the accumulated branch length.
    """
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
        pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}))
    emp = coal.to_empirical(n_samples=2000)

    def dists(c):
        joint = c.joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward())
        return [joint.conditional(value=1.0), joint.marginal('a'), c.tree_height, c.tree_height.demes['pop_0'], c.sfs]

    for exact, sampled in zip(dists(coal), dists(emp)):
        a, b = getattr(exact, kind)._plot_data(), getattr(sampled, kind)._plot_data()
        assert (a.title, a.xlabel, a.ylabel) == (b.title, b.xlabel, b.ylabel)

    cond = getattr(coal.joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward()).conditional(value=1.0), kind)
    assert cond._plot_data().title.startswith('R_b | R_a = 1')
    assert getattr(coal.tree_height.demes['pop_0'], kind)._plot_data().xlabel == ('q' if kind == 'quantile' else 't')
