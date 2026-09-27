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
    assert names(coal.sfs.bin(1).cdf) == ['ax', 'x', 'n_points', 'show']
    assert names(coal.sfs.bin(1).quantile) == ['ax', 'q', 'n_points', 'show']
    assert names(coal.sfs.pdf) == ['ax', 'x', 'bins', 'n_points']
    assert names(coal.sfs.quantile) == ['ax', 'q', 'bins', 'n_points']
    assert names(jsfs.cdf) == ['ax', 'x', 'configs', 'n_points']

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
    jd = pg.Coalescent(n=5).sfs.joint_distribution(1, 2)

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
    np.testing.assert_allclose(sfs.y, coal.sfs.accumulate(1, t)[1:5], rtol=1e-10, atol=1e-12)
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
