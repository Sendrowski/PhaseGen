"""
Unit tests for the :class:`~phasegen.comparison.Comparison` machinery.

These cover the *simulation-free* plumbing -- tolerance-tree parsing/expansion, the difference metrics, the
pairwise-surface pair extraction, and the degenerate-surface guard -- without the cost of the end-to-end scenario
suite (``testing/test_scenarios.py``, which validates accuracy against the analytical results, and which also exercises
the ``mutation_configs`` comparison via the ``*_mu_*`` configs). They run in milliseconds and pin the contracts of the
config-driven dispatch that the scenario fixtures rely on.
"""
import logging

import numpy as np

from testing import TestCase

import phasegen as pg
from phasegen.comparison import Comparison
from phasegen.distributions import MsprimeCoalescent
from phasegen.distributions.empirical import EmpiricalDistribution


class ComparisonHelpersTestCase(TestCase):
    """Pure, simulation-free unit tests for the static/config-only comparison helpers."""

    def test_rel_diff_scalar_array_and_zero(self):
        """``rel_diff`` is element-wise, holds the 0/0 case at 0, and returns a scalar for scalar input."""
        # both zero -> 0 (no division blow-up); scalar in -> scalar out
        self.assertEqual(Comparison.rel_diff(0.0, 0.0), 0.0)
        # |1 - 3| / ((|1| + |3|) / 2) = 2 / 2 = 1
        self.assertEqual(Comparison.rel_diff(1.0, 3.0), 1.0)
        # array path, element-wise, with a both-zero entry held at 0
        rd = np.asarray(Comparison.rel_diff([0.0, 2.0, 1.0], [0.0, 2.0, 3.0]))
        self.assertEqual(rd.tolist(), [0.0, 0.0, 1.0])

    def test_diff_label(self):
        """The difference-metric label maps the cdf (max abs) specially; the pdf and the mutation configurations both
        use the total-variation distance; everything else is a worst relative difference."""
        for stat in ('cdf', 'pairwise_cdf', 'loci_pairwise_cdf'):
            self.assertEqual(Comparison._diff_label(stat), 'max abs')
        for stat in ('pdf', 'pairwise_pdf', 'loci_pairwise_pdf', 'mutation_configs'):
            self.assertEqual(Comparison._diff_label(stat), 'total variation')
        self.assertEqual(Comparison._diff_label('quantile'), 'rel. Wasserstein')
        self.assertEqual(Comparison._diff_label('mean'), 'max rel')

    def test_parse_collection_key(self):
        """Only quoted list/set literals (and bare-identifier collections) are collection keys; a bare stat or tuple
        key is not."""
        self.assertIsNone(Comparison._parse_collection_key('cdf'))
        self.assertIsNone(Comparison._parse_collection_key('(1, 2)'))  # a bare tuple stays a single pair key
        self.assertEqual(Comparison._parse_collection_key('[1, 3, 9]'), [1, 3, 9])
        self.assertEqual(Comparison._parse_collection_key('[(1, 2), (1, 9)]'), [(1, 2), (1, 9)])
        # bare-identifier collection (broadcast a sub-spec over several keys), which ``ast.literal_eval`` rejects
        self.assertEqual([s.strip() for s in Comparison._parse_collection_key('[cosine, mean]')],
                         ['cosine', 'mean'])

    def test_expand_keys_broadcasts_and_deep_copies(self):
        """A list key broadcasts its sub-spec over the elements (ints -> bin keys, tuples -> ``"(i, j)"`` keys); a bare
        tuple stays single; broadcasting deep-copies; expansion recurses into nested dicts."""
        self.assertEqual(Comparison._expand_keys({'[1, 3]': {'pdf': 0.01}}),
                         {1: {'pdf': 0.01}, 3: {'pdf': 0.01}})

        out = Comparison._expand_keys({'[(1, 2), (1, 9)]': {'cdf': 0.02}, '(2, 3)': {'cdf': 0.03}})
        self.assertEqual(out['(1, 2)'], {'cdf': 0.02})
        self.assertEqual(out['(1, 9)'], {'cdf': 0.02})
        self.assertEqual(out['(2, 3)'], {'cdf': 0.03})
        # the broadcast must be a deep copy: mutating one expansion leaves the others untouched
        out['(1, 2)']['cdf'] = 99
        self.assertEqual(out['(1, 9)']['cdf'], 0.02)

        self.assertEqual(Comparison._expand_keys({'sfs': {'[1, 2]': {'pdf': 0.1}}}),
                         {'sfs': {1: {'pdf': 0.1}, 2: {'pdf': 0.1}}})

    def test_pairwise_surface_pairs(self):
        """Only the pair keys (not the legacy ``cdf``/``pdf`` aggregates) are collected for surface caching, de-duped,
        and only for distributions that request a surface."""
        c = Comparison.__new__(Comparison)
        c.comparisons = {'tolerance': {
            'sfs': {'pairwise': {'(1, 2)': {'cdf': 0.1, 'pdf': 0.1},
                                 '(1, 4)': {'cdf': 0.1, 'pdf': 0.1}}},
            'sfs2': {'pairwise': {'(1, 1)': {'cdf': 0.1, 'pdf': 0.1}}},
            'jsfs': {'mean': 0.01},  # no pairwise group -> absent from the result
        }}
        pairs = c._pairwise_surface_pairs()
        self.assertEqual(pairs['sfs'], [(1, 2), (1, 4)])
        self.assertEqual(pairs['sfs2'], [(1, 1)])
        self.assertNotIn('jsfs', pairs)

    def test_pairwise_surface_pairs_expands_list_keys(self):
        """A list-of-pairs key under ``pairwise`` is expanded before the pairs are collected."""
        c = Comparison.__new__(Comparison)
        c.comparisons = {'tolerance': {
            'sfs': {'pairwise': {'[(1, 2), (2, 3)]': {'cdf': 0.1, 'pdf': 0.1}}},
        }}
        self.assertEqual(c._pairwise_surface_pairs()['sfs'], [(1, 2), (2, 3)])


class CurveStatRegressionTestCase(TestCase):
    """Regressions for two comparison-machinery bugs on the un-moded (``mode=None``) statistic paths: the spectrum-wide
    SFS pdf cell average and the ``std`` statistic on an empirical operand that exposes ``var`` but not ``std``."""

    def test_spectrum_wide_sfs_pdf_cell_average_does_not_crash(self):
        """A spectrum-wide, ``mode=None`` SFS pdf comparison averages an :class:`SFSDensity` over the grid cells.
        The density returns ``(len(grid), n_bins)`` (grid on axis 0), but the quadrature reshape assumed the grid on
        the trailing axis, so pre-fix this raised ``ValueError: cannot reshape array of size 800 into shape
        (160, 20, 8)``. Post-fix the grid axis is moved to the back before the reshape, so the cell average returns
        the per-bin densities without error."""
        coal = pg.Coalescent(n=4)
        dens = coal.sfs.pdf  # SFSDensity: returns (len(grid), n + 1), grid on axis 0
        t = np.linspace(0, float(np.max(coal.sfs.quantile(0.99).data)), 20)

        # the density's own orientation is grid-first -- the exact shape the pre-fix reshape mis-handled
        self.assertEqual(np.asarray(dens(t)).shape, (len(t), 5))

        avg = Comparison._cell_average(dens, t, coal.sfs.cdf)  # pre-fix: ValueError from the reshape

        # oriented to the (n_bins, len(grid)) cell-average contract, finite and non-negative
        self.assertEqual(avg.shape, (5, len(t)))
        self.assertTrue(np.all(np.isfinite(avg)))
        self.assertTrue(np.all(avg >= 0))

        # the polymorphic bins carry continuous mass in [0, 1]; the monomorphic edge bins are the zero atom
        widths = np.diff(np.append(t, 2 * t[-1] - t[-2]))
        masses = (avg * widths).sum(axis=1)
        self.assertTrue(np.all(masses <= 1.0 + 1e-9))
        self.assertTrue(np.all(masses[1:-1] > 0))
        self.assertEqual(masses[0], 0.0)
        self.assertEqual(masses[-1], 0.0)


class WindowedConditionalConfigTestCase(TestCase):
    """The windowed-conditional CDF check reads its ``cdf_axes`` from the config; a value outside the axis labels
    ``{'a', 'b'}`` (e.g. a user writing index-based ``[0, 1]``) would otherwise make the check a silent no-op that
    passes with diff 0.0, so it must be rejected."""

    def test_invalid_cdf_axes_raises(self):
        """An unrecognised ``cdf_axes`` value raises a clear ValueError instead of silently passing."""
        c = Comparison.__new__(Comparison)
        pair = (1, 2)
        # a cached entry for the pair so the check reaches the cdf_axes validation (only the first two fields matter)
        entry = (pair[0], pair[1], 'a', 0.5, 0.1, 100, 1.0, 0.01, np.linspace(0, 1, 3), np.zeros(3))
        ms = type('Ms', (), {'_windowed_conditional': [entry]})()

        with self.assertRaises(ValueError) as ctx:
            c._compare_windowed_conditional(jd=None, ms=ms, pair=pair,
                                            tols={'cdf': 0.0, 'cdf_axes': [0, 1]}, title='t')
        self.assertIn('cdf_axes', str(ctx.exception))

    @staticmethod
    def _loci_comparison(conditional: dict) -> Comparison:
        """A two-locus comparison whose tree height carries the given ``loci: pairwise: conditional`` block."""
        c = Comparison(
            n=2, n_loci=2, recombination_rate=1, pop_sizes={'pop_0': {0: 1}}, num_replicates=20000, seed=0,
            parallelize=False,
            comparisons={'tolerance': {'tree_height': {'loci': {'pairwise': {'conditional': conditional}}}}}
        )
        c.visualize = False
        return c

    def test_loci_windowed_conditional_is_cached_and_asserted(self):
        """A ``windowed`` block under ``loci: pairwise: conditional`` caches its msprime ground truth over the locus
        pair and asserts the window means against it."""
        c = self._loci_comparison({'windowed': {'mean': 4}})
        c.cache_ground_truth()

        self.assertEqual({(i, j) for i, j, *_ in c.ms.tree_height._loci_windowed_conditional}, {(0, 1)})
        self.assertEqual(getattr(c.ms.tree_height, '_windowed_conditional', []), [])

        c.compare()
        self.assertEqual(c.n_assertions, 1)

    def test_loci_windowed_conditional_raises_on_edited_windows(self):
        """Editing a ``windowed`` block's ``quantiles`` or ``window`` after caching raises at comparison time.
        Regression: the check asserted at the cached windows and passed silently."""
        for edit in ({'quantiles': [0.3]}, {'window': 0.3}):
            with self.subTest(edit=edit):
                c = self._loci_comparison({'windowed': {'mean': 4}})
                c.cache_ground_truth()
                c.comparisons['tolerance']['tree_height']['loci']['pairwise']['conditional']['windowed'].update(edit)

                with self.assertRaises(ValueError) as ctx:
                    c.compare()
                self.assertIn('other windows', str(ctx.exception))

    def test_loci_atom_conditional_raises_at_config_load(self):
        """A per-locus reward has no atom at 0, so an ``atom`` block under ``loci: pairwise: conditional`` is
        rejected when the comparison is created."""
        with self.assertRaises(ValueError) as ctx:
            self._loci_comparison({'atom': {'mass': 0.01}})
        self.assertIn("'atom'", str(ctx.exception))


class DehoogConditionalTestCase(TestCase):
    """The ``dehoog`` conditional check, which needs no msprime operand."""

    def test_dehoog_conditional_asserts_against_its_tolerance(self):
        """A ``dehoog`` block compares the conditional CDF with the de Hoog inversion of its transform and asserts the
        largest difference against ``cdf``, and rejects a key that is neither ``cdf`` nor one of its options."""
        c = Comparison(n=4, pop_sizes={'pop_0': {0: 1}}, comparisons={'tolerance': {}})
        jd = c.ph.sfs.joint(1, 2)
        opts = {'quantiles': [0.5], 'axes': ['b']}

        with self.assertLogs('phasegen', level='INFO') as logs:
            c._compare_dehoog_conditional(jd, (1, 2), {'cdf': 1e-3, **opts}, 't')
        self.assertEqual(c.n_assertions, 1)
        self.assertRegex(logs.output[0],
                         r'#1 t: conditional \(1, 2\) dehoog: cdf: [\d.]+ <= 0\.001 \(max abs, [\d.]+s\)')

        with self.assertRaises(AssertionError):
            c._compare_dehoog_conditional(jd, (1, 2), {'cdf': 1e-12, **opts}, 't')

        for tols in ({'cdf': 1e-3, 'nodes': 2}, {'cdf': 1e-3, 'axes': [0]}, opts):
            with self.subTest(tols=tols), self.assertRaises(ValueError):
                c._compare_dehoog_conditional(jd, (1, 2), tols, 't')

    def test_dehoog_conditional_detects_an_unresolved_expansion(self):
        """The check fails on a conditional expansion with too few cosine terms for its window, which the per-point
        de Hoog reference does not depend on."""
        c = Comparison(n=4, pop_sizes={'pop_0': {0: 1}}, comparisons={'tolerance': {}})
        tols = {'cdf': 2.5e-3, 'quantiles': [0.5], 'axes': ['a']}

        c._compare_dehoog_conditional(c.ph.sfs.joint(1, 2), (1, 2), tols, 't')

        pg.Settings.cos_terms = 16
        with self.assertRaises(AssertionError):
            c._compare_dehoog_conditional(c.ph.sfs.joint(1, 2), (1, 2), tols, 't')


class _NaNJD:
    """A joint distribution whose conditional checks and window averages return NaN, with an identity marginal
    quantile and no atoms, so its windows are ``v = q`` and ``h = window * q``."""

    _atoms = {'a0': 0.0, 'b0': 0.0}

    def __init__(self, check: dict = None):
        self.check = check

    def marginal(self, on):
        return type('Marginal', (), {'quantile': staticmethod(lambda p: p)})()

    def window_average(self, f, on, v, h, n_nodes=None):
        return f(type('Conditional', (), {'mean': np.nan, 'cdf': staticmethod(lambda y: np.full_like(y, np.nan))})())

    def check_total_expectation(self, tol, **kwargs):
        return self.check

    def conditional(self, on, v):
        """A conditional whose served CDF is NaN on axis ``a`` and whose de Hoog reference is NaN on axis ``b``."""
        def nan(y):
            return np.full_like(np.asarray(y, dtype=float), np.nan)

        def zero(y):
            return np.zeros_like(np.asarray(y, dtype=float))

        served, exact = (nan, zero) if on == 'a' else (zero, nan)
        cdf = type('CDF', (), {'__call__': lambda self, y: served(y), '_cdf_point': staticmethod(exact)})()
        return type('Conditional', (), {'quantile': staticmethod(lambda p: p), 'cdf': cdf})()


class ConditionalNaNTestCase(TestCase):
    """A NaN returned by phasegen fails a conditional check. Regression: the worst case was taken with Python's
    ``max``, which drops a NaN that is not its first argument, so the check passed."""

    @staticmethod
    def _comparison() -> Comparison:
        c = Comparison(n=2, pop_sizes={'pop_0': {0: 1}}, num_replicates=10, comparisons={'tolerance': {}})
        c.visualize = False
        return c

    def test_nan_on_either_axis_fails(self):
        """A NaN on the first or second conditioning axis fails the check."""
        for res in ({'a': 0.0, 'b': np.nan}, {'a': np.nan, 'b': 0.0}):
            with self.subTest(res=res):
                with self.assertRaises(AssertionError):
                    self._comparison()._compare_conditional(_NaNJD(res), (1, 2), {'total_expectation': 0.1}, 't')

    def test_nan_windowed_mean_and_cdf_fail(self):
        """A NaN window average fails the windowed mean and cdf checks."""
        pair = (1, 2)
        ys = np.linspace(0, 1, 3)
        ms = type('Ms', (), {'_windowed_conditional': [
            (1, 2, on, 0.5, 0.1, 100, 1.0, 0.01, ys, np.zeros(3)) for on in ('a', 'b')
        ]})()

        for stat in ('mean', 'cdf'):
            with self.subTest(stat=stat):
                tols = {'quantiles': [0.5], 'window': 0.2, stat: 1.0}
                with self.assertRaises(AssertionError):
                    self._comparison()._compare_windowed_conditional(_NaNJD(), ms, pair, tols, 't')

    def test_nan_dehoog_cdf_fails(self):
        """A NaN in the served conditional CDF or in its de Hoog reference fails the ``dehoog`` check."""
        for axes in (['a'], ['b'], ['b', 'a']):
            with self.subTest(axes=axes):
                tols = {'cdf': 1.0, 'quantiles': [0.5], 'levels': [0.5], 'axes': axes}
                with self.assertRaises(AssertionError):
                    self._comparison()._compare_dehoog_conditional(_NaNJD(), (1, 2), tols, 't')


class _ExplodingJD:
    """A joint distribution whose curve inversions raise -- used to prove the degenerate guard returns *before* it
    would evaluate any surface."""

    def cdf(self, *args, **kwargs):
        raise AssertionError("cdf inversion attempted on a degenerate surface")

    def pdf(self, *args, **kwargs):
        raise AssertionError("pdf inversion attempted on a degenerate surface")


class PairwiseSurfaceGuardTestCase(TestCase):
    """The degenerate-surface guard in ``_compare_pairwise_surface`` skips a pair with a warning, rather than asserting
    on it or crashing, when its cached empirical grid has zero-width support or non-finite values -- e.g. a
    high-frequency bin under an extreme multiple-merger (star-like genealogy)."""

    @staticmethod
    def _bare_comparison() -> Comparison:
        c = Comparison.__new__(Comparison)
        c.logger = logging.getLogger('phasegen')
        c.do_assertion = True
        c.n_assertions = 0
        c.visualize = False
        c._comp_index = 0
        return c

    @staticmethod
    def _ms_with_surface(entry) -> object:
        return type('Ms', (), {'_joint_surface': [entry]})()

    def test_zero_width_grid_is_skipped(self):
        """A zero-width support on an axis (``xs[-1] <= xs[0]``) is degenerate: no continuous surface to compare."""
        c = self._bare_comparison()
        xs = np.zeros(25)  # zero-width support
        ys = np.linspace(0.0, 1.0, 25)
        grid = np.zeros((25, 25))
        ms = self._ms_with_surface((1, 2, xs, ys, grid, grid))

        with self.assertLogs('phasegen', level='WARNING') as logs:
            c._compare_pairwise_surface(ph=None, ms=ms, pair=(1, 2), tols={'cdf': 0.0, 'pdf': 0.0},
                                        title='t', name='n', joint_fn=lambda i, j: _ExplodingJD())

        self.assertEqual(c.n_assertions, 0)
        self.assertIn('not asserted', logs.output[0])

    def test_non_finite_grid_is_skipped(self):
        """A non-finite cached CDF (degenerate bin with no off-zero mass) is skipped rather than crashing."""
        c = self._bare_comparison()
        xs = np.linspace(0.0, 1.0, 25)
        ys = np.linspace(0.0, 1.0, 25)
        cdf = np.full((25, 25), np.nan)
        ms = self._ms_with_surface((1, 2, xs, ys, cdf, np.zeros((25, 25))))

        c._compare_pairwise_surface(ph=None, ms=ms, pair=(1, 2), tols={'cdf': 0.0},
                                    title='t', name='n', joint_fn=lambda i, j: _ExplodingJD())

        self.assertEqual(c.n_assertions, 0)

    def test_missing_surface_raises(self):
        """A configured pair with no cached empirical surface is a fixture error, not a silent skip."""
        c = self._bare_comparison()
        ms = self._ms_with_surface((9, 9, np.linspace(0, 1, 25), np.linspace(0, 1, 25),
                                    np.zeros((25, 25)), np.zeros((25, 25))))
        with self.assertRaises(ValueError):
            c._compare_pairwise_surface(ph=None, ms=ms, pair=(1, 2), tols={'cdf': 0.0},
                                        title='t', name='n', joint_fn=lambda i, j: _ExplodingJD())

    def test_no_config_declares_a_duplicate_key(self):
        """A YAML mapping silently keeps only the *last* of two identical keys, so a duplicated bin-pair key under
        ``conditional:`` deletes a whole block of checks without any error. That is invisible in a passing run -- the
        checks simply stop existing -- so guard every config against it."""
        import pathlib

        import yaml as pyyaml
        from yaml.constructor import ConstructorError

        class NoDuplicates(pyyaml.SafeLoader):
            pass

        def construct(loader, node, deep=False) -> dict:
            seen = set()
            for key_node, _ in node.value:
                key = loader.construct_object(key_node, deep=True)
                key = tuple(key) if isinstance(key, list) else key
                if key in seen:
                    raise ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
            return pyyaml.SafeLoader.construct_mapping(loader, node, deep)

        NoDuplicates.add_constructor(pyyaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
        # the multi-population configs key migration rates by a !!python/tuple of deme names
        NoDuplicates.add_constructor('tag:yaml.org,2002:python/tuple',
                                     lambda loader, node: tuple(loader.construct_sequence(node)))

        configs = sorted(pathlib.Path('resources/configs').glob('*.yaml'))
        self.assertGreater(len(configs), 100)

        for path in configs:
            with self.subTest(config=path.name):
                pyyaml.load(path.read_text(), Loader=NoDuplicates)


class CompareOnlyTestCase(TestCase):
    """``--compare-only`` (``Comparison.only``) restricts every block, the coalescent-level statistics included, and a
    restricted block keeps the options it runs at."""

    def test_restrict_keeps_block_options(self):
        """Restricting to a leaf inside a conditional or windowed block keeps that block's option keys. Regression:
        they were dropped, so the restricted run evaluated the check at its default settings."""
        spec = {'sfs': {'conditional': {'(1, 2)': {
            'moments': 0.01, 'quantiles': [0.3], 'curves': 2,
            'windowed': {'mean': 4, 'cdf': 0.01, 'quantiles': [0.5], 'window': 0.1, 'nodes': 3, 'cdf_axes': ['a']},
        }}}, 'tree_height': {'mean': 0.01}}

        out = Comparison._restrict(spec, 'mean')

        self.assertEqual(out, {'sfs': {'conditional': {'(1, 2)': {
            'windowed': {'mean': 4, 'quantiles': [0.5], 'window': 0.1, 'nodes': 3, 'cdf_axes': ['a']},
            'quantiles': [0.3], 'curves': 2,
        }}}, 'tree_height': {'mean': 0.01}})
        self.assertEqual(Comparison._restrict(spec, 'pdf'), {})

    @staticmethod
    def _statistics_comparison(only: str = None) -> Comparison:
        """A bare comparison asserting one cached F_ST, and nothing else."""
        c = Comparison.__new__(Comparison)
        c.logger = logging.getLogger('phasegen')
        c.do_assertion = True
        c.n_assertions = 0
        c.visualize = False
        c.only = only
        c.comparisons = {'tolerance': {}, 'statistics': {'fst': 0.1}}
        c._ms_statistics = {('fst', ()): 0.5}
        c.__dict__['ph'] = type('Ph', (), {'fst': 0.51})()
        return c

    def test_statistics_follow_compare_only(self):
        """A restriction to another key skips the statistics, one to the block or the statistic keeps them.
        Regression: the statistics were asserted under any restriction."""
        for only, n in ((None, 1), ('cosine', 0), ('statistics', 1), ('fst', 1)):
            with self.subTest(only=only):
                c = self._statistics_comparison(only)
                c.compare()
                self.assertEqual(c.n_assertions, n)

    def test_statistic_result_carries_index_and_runtime(self):
        """A statistic is logged like every other comparison, with its index, metric and runtime."""
        c = self._statistics_comparison()

        with self.assertLogs('phasegen', level='INFO') as logs:
            c.compare(title='t')

        self.assertRegex(logs.output[0], r'#1 t: fst: 0\.01980 <= 0\.1 \(max rel, [\d.]+s\)')


class QuantileMetricTestCase(TestCase):
    """The relative Wasserstein metric of ``Comparison._quantile_diff`` on a bin whose reference quantile is 0."""

    def test_zero_reference_fails_on_a_wrong_curve(self):
        """A bin whose reference quantile is 0 on the whole grid passes only where the other curve is 0 as well.
        Regression: any phasegen curve, NaN or large, gave 0 there."""
        q = np.linspace(0.05, 0.95, 5)
        zero = np.zeros((2, len(q)))
        ref = np.vstack([np.zeros(len(q)), np.linspace(1.0, 2.0, len(q))])

        self.assertEqual(Comparison._quantile_diff(zero, zero, q), 0.0)
        self.assertEqual(Comparison._quantile_diff(ref, ref, q), 0.0)

        for bad in (np.nan, 5.0):
            with self.subTest(bad=bad):
                y_ph = ref.copy()
                y_ph[0, 2] = bad
                self.assertFalse(Comparison._quantile_diff(ref, y_ph, q) <= 1.0)


class UncachedCurveTestCase(TestCase):
    """A curve comparison on an empirical operand whose ground truth was never cached."""

    def test_uncached_operand_uses_its_own_grid(self):
        """The cdf and quantile of an operand without a cache are evaluated on the fallback grids. Regression: the
        cache lookup raised ``TypeError`` on the operand's unset cache."""
        c = Comparison.__new__(Comparison)
        c.visualize = False
        ph = pg.Coalescent(n=3).tree_height
        ms = EmpiricalDistribution(np.random.default_rng(0).exponential(4 / 3, 20000))

        for stat in ('cdf', 'quantile'):
            with self.subTest(stat=stat):
                diff, _ = c._diff_and_plot_curve(ph, ms, getattr(ms, stat), stat, None, 'n')
                self.assertTrue(np.isfinite(diff))


class PairwiseKeysTestCase(TestCase):
    """Unknown keys under a pairwise block or a ``loci: pairwise`` block are rejected."""

    def test_unknown_keys_raise(self):
        """Regression: an unknown leaf was dropped and the scenario passed with no assertion, and a ``cdf`` key in
        place of a pair raised an unrelated ``ValueError`` from parsing it."""
        c = PairwiseSurfaceGuardTestCase._bare_comparison()

        for data in ({'pairwise': {'(1, 2)': {'cdf': 0.1, 'pfd': 0.1}}}, {'pairwise': {'cdf': 0.1}}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError) as ctx:
                    c._compare_stat_recursively(ph=None, ms=None, data=data)
                self.assertIn('takes', str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            c._compare_loci_pairwise(ph=None, ms=None, sub={'cdf': 0.1, 'pfd': 0.1}, title='t', name='n')
        self.assertIn('pfd', str(ctx.exception))


class SpectrumGroundTruthTestCase(TestCase):
    """The msprime ground truth of the compared distributions, and only theirs, survives the fixture round trip of
    ``scripts/create_comparison.py``."""

    @staticmethod
    def _round_trip(c: Comparison) -> Comparison:
        """
        Cache the ground truth, drop the simulated data and restore the comparison from its serialization, as
        ``scripts/create_comparison.py`` does.

        :param c: The comparison.
        :return: The restored comparison.
        """
        import os
        import tempfile

        c.cache_ground_truth()
        c.ms._drop()
        c.__dict__.pop('ph', None)

        with tempfile.TemporaryDirectory() as tmp:
            file = os.path.join(tmp, 'c.json')
            c.to_file(file)
            restored = Comparison.from_file(file)

        restored.visualize = False
        return restored

    def test_jsfs_is_cached_and_compared(self):
        """The joint SFS is cached without its samples where compared, and the uncompared distributions are not cached.
        Regression: the msprime joint SFS was neither cached nor dropped, so only a dedicated script could build its
        fixture, and every fixture cached all of the tree height, total branch length and SFS."""
        kwargs = dict(n={'pop_0': 2, 'pop_1': 2}, pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 1.5}},
                      migration_rates={('pop_0', 'pop_1'): {0: 0.75}, ('pop_1', 'pop_0'): {0: 0.75}},
                      num_replicates=2000, seed=0, parallelize=False)

        c = self._round_trip(Comparison(**kwargs, comparisons={'tolerance': {'jsfs': {'mean': 1, 'var': 1}}}))
        self.assertIsNone(c.ms.__dict__['jsfs'].samples)
        self.assertFalse({'tree_height', 'total_branch_length', 'sfs', 'fsfs'} & set(c.ms.__dict__))
        c.compare()
        self.assertEqual(c.n_assertions, 2)

        c = self._round_trip(Comparison(**kwargs, comparisons={'tolerance': {'tree_height': {'mean': 1}}}))
        self.assertEqual({'tree_height'}, set(MsprimeCoalescent._distributions) & set(c.ms.__dict__))

    def test_sfs2_is_cached_and_compared(self):
        """The two-locus SFS is cached without its samples where compared, and the uncompared distributions are not
        cached."""
        kwargs = dict(n=3, n_loci=2, recombination_rate=1, pop_sizes={'pop_0': {0: 1}}, num_replicates=2000, seed=0,
                      parallelize=False)

        c = self._round_trip(Comparison(**kwargs, comparisons={'tolerance': {'sfs2': {'mean': 1, 'var': 1}}}))
        self.assertIsNone(c.ms.__dict__['sfs2'].samples)
        self.assertFalse({'tree_height', 'total_branch_length', 'sfs', 'fsfs'} & set(c.ms.__dict__))
        c.compare()
        self.assertEqual(c.n_assertions, 2)

        c = self._round_trip(Comparison(**kwargs, comparisons={'tolerance': {'tree_height': {'mean': 1}}}))
        self.assertEqual({'tree_height'}, set(MsprimeCoalescent._distributions) & set(c.ms.__dict__))
        self.assertEqual(len(c.ms.tree_height._loci_joint_surface), 1)
        c.compare()
        self.assertEqual(c.n_assertions, 1)


class CellAverageJumpTestCase(TestCase):
    """The exact pdf's cell average integrates a density jump that lies closer to a cell edge than any node."""

    def test_epoch_jump_next_to_a_cell_edge(self):
        """Regression: on the grid ``1.38 k`` the epoch boundary at 1.4 sits 0.02 inside the cell [1.38, 2.76), short
        of the first node of both the 8- and the 4-node rule. The two rules agreed, so the cell was not refined, and
        the tree height's pdf differed from its own cdf's increments by a total variation of about 5e-3."""
        th = pg.Coalescent(n=2, demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1.2, 0.3: 10, 1: 0.8, 1.4: 10}})).tree_height
        t = 1.38 * np.arange(100)
        exact = np.diff(th.cdf(np.append(t, 1.38 * 100))) / 1.38

        ms = EmpiricalDistribution(np.zeros(1))
        ms._cache = {'t': t, 'pdf': exact}

        c = Comparison.__new__(Comparison)
        c.visualize, c.show_title, c.do_assertion, c.n_assertions = False, False, True, 0
        c.logger = logging.getLogger('phasegen.Comparison')

        c.compare_stat(th, ms, 'pdf', tol=2e-4)
