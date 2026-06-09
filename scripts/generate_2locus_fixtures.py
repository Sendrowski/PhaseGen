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
for attr in ('heights', 'total_branch_lengths', 'sfs_lengths', 'mutations', 'jsfs_moments', 'demography'):
    setattr(c.ms, attr, None)

# verify the analytical two-locus SFS agrees with the cached truth within the configured tolerances
for stat, tol in c.comparisons['tolerance']['sfs2'].items():
    diff = Comparison.rel_diff(np.array(getattr(c.ms.sfs2, stat).data), np.array(getattr(c.ph.sfs2, stat).data)).max()
    print(f'{stat:>5}: rel_diff.max={diff:.4f} tol={tol} [{"ok" if diff <= tol else "FAIL"}]', flush=True)

# drop the cached analytical coalescent so only the small msprime ground truth is serialized (otherwise the two-locus
# state space would bloat the fixture); the analytical side is recomputed fresh at test time
c.__dict__.pop('ph', None)

c.to_file(out)
