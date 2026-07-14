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

from phasegen.comparison import Comparison


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
        # bare-identifier collection (broadcast a sub-spec over inversion modes), which ``ast.literal_eval`` rejects
        self.assertEqual([s.strip() for s in Comparison._parse_collection_key('[cosine, de_hoog]')],
                         ['cosine', 'de_hoog'])

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


class _ExplodingJD:
    """A joint distribution whose curve inversions raise -- used to prove the degenerate guard returns *before* it
    would evaluate any surface."""

    def cdf(self, *args, **kwargs):
        raise AssertionError("cdf inversion attempted on a degenerate surface")

    def pdf(self, *args, **kwargs):
        raise AssertionError("pdf inversion attempted on a degenerate surface")


class PairwiseSurfaceGuardTestCase(TestCase):
    """The degenerate-surface guard in ``_compare_pairwise_surface`` skips (rather than asserts on / crashes on) a pair
    whose cached empirical grid has zero-width support or non-finite values -- e.g. a high-frequency bin under an
    extreme multiple-merger (star-like genealogy)."""

    @staticmethod
    def _bare_comparison() -> Comparison:
        c = Comparison.__new__(Comparison)
        c.logger = logging.getLogger('phasegen')
        c.do_assertion = True
        c.n_assertions = 0
        c.visualize = False
        c._comp_index = 0
        c.runtimes = {}
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

        c._compare_pairwise_surface(ph=None, ms=ms, pair=(1, 2), tols={'cdf': 0.0, 'pdf': 0.0},
                                    title='t', name='n', joint_fn=lambda i, j: _ExplodingJD())

        self.assertEqual(c.n_assertions, 0)  # nothing asserted -> pair skipped

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
