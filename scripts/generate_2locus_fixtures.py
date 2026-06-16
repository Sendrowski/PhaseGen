"""
Generate a serialized two-locus-SFS comparison fixture used by ``testing/test_scenarios.py``.

For a two-locus config ``resources/configs/{name}_2_locus_sfs.yaml`` this simulates the msprime ground truth for the
two-locus SFS, caches it into the serialized :class:`~phasegen.comparison.Comparison`, verifies the analytical
two-locus SFS agrees within the configured tolerance, and writes the (small) fixture. Driven per-fixture by
snakemake; a single config can also be generated directly with ``python scripts/generate_2locus_fixtures.py``.
"""

import sys

sys.path.append('.')

import numpy as np

import phasegen as pg
from phasegen.comparison import Comparison

# run sequentially: forking after numpy/matplotlib are imported deadlocks on macOS
pg.Settings.parallelize = False

try:
    file = snakemake.input[0]  # noqa: F821
    out = snakemake.output[0]  # noqa: F821
except NameError:
    name = '1_epoch_n_3_2_locus_sfs'
    file = f'resources/configs/{name}.yaml'
    out = f'results/comparisons/serialized/{name}.json'

c = Comparison.from_yaml(file)
c.parallelize = False
c.n_threads = 1

# cache the msprime two-locus SFS ground truth (its own simulation), then null the raw per-replicate samples and the
# demography to keep the fixture small
_ = c.ms.sfs2

# cache the two-locus joint distribution ground truth (cross-moment + joint CDF at marginal-quantile points) from the
# per-replicate locus branch lengths, then drop those (large) samples so only the small ground truth is serialized.
# Only a few *representative* cross-locus bin pairs (low-low / low-high / mid-high) rather than all O(n^2) pairs: the
# all-pairs aggregate is prohibitively slow for larger n (the 2D cosine build per pair on the two-locus state space
# took ~60 min/pass at n=6), while a fixed handful is cheap and exercises the informative regimes.
n = c.ph.lineage_config.n
hi = n - 1
if hi >= 3:
    rep_pairs = [(1, 2), (1, hi), (2, hi)]
elif hi == 2:
    rep_pairs = [(1, 2)]
else:
    rep_pairs = [(1, 1)]
c.ms.sfs2.cache_joint(rep_pairs, [(0.4, 0.6), (0.6, 0.4), (0.7, 0.7)])
c.ms.sfs2.drop()

for attr in ('heights', 'total_branch_lengths', 'sfs_lengths', 'mutations', 'jsfs_moments', 'demography'):
    setattr(c.ms, attr, None)

# verify the analytical two-locus SFS agrees with the cached truth within the configured tolerances (the 'joint' and
# 'pairwise' stats are not ``.data`` spectra -- they are validated at test time via Comparison.compare_stat -- so skip)
for stat, tol in c.comparisons['tolerance']['sfs2'].items():
    if stat in ('joint', 'pairwise'):
        continue
    diff = Comparison.rel_diff(np.array(getattr(c.ms.sfs2, stat).data), np.array(getattr(c.ph.sfs2, stat).data)).max()
    print(f'{stat:>5}: rel_diff.max={diff:.4f} tol={tol} [{"ok" if diff <= tol else "FAIL"}]', flush=True)

# drop the cached analytical coalescent so only the small msprime ground truth is serialized (otherwise the two-locus
# state space would bloat the fixture); the analytical side is recomputed fresh at test time
c.__dict__.pop('ph', None)

c.to_file(out)
