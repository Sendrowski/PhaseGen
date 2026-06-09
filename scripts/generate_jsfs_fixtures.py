"""
Generate a serialized joint-SFS comparison fixture used by ``testing/test_scenarios.py``.

For a joint-SFS config ``resources/configs/{name}_jsfs.yaml`` this simulates the msprime ground truth, caches it into
the serialized :class:`~phasegen.comparison.Comparison`, verifies the analytical joint SFS agrees within the
configured tolerance, and writes the (small) fixture. Driven per-fixture by snakemake; a single config can also be
generated directly with ``python scripts/generate_jsfs_fixtures.py``.
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
    name = '1_epoch_2_pops_n_4_jsfs'
    file = f'resources/configs/{name}.yaml'
    out = f'results/comparisons/serialized/{name}.json'


def get_stat(dist, stat: str) -> np.ndarray:
    """
    Get the joint-SFS statistic, mirroring how :meth:`Comparison.compare_stat` accesses it.

    :param dist: Joint SFS distribution (analytical or empirical).
    :param stat: Statistic name.
    :return: The statistic array.
    """
    # the analytical distribution exposes higher moments via moment(), the empirical one via attributes
    if stat in ('m3', 'm4') and hasattr(dist, 'moment'):
        return dist.moment(int(stat[1]), center=False)

    return getattr(dist, stat)


c = Comparison.from_yaml(file)
c.parallelize = False
c.n_threads = 1

# cache the msprime joint-SFS ground truth (accumulated within simulate()), then null the raw per-replicate samples
# and the demography to keep the fixture small (the moments are retained by the cached jsfs distribution)
_ = c.ms.jsfs
for attr in ('heights', 'total_branch_lengths', 'sfs_lengths', 'mutations', 'jsfs_moments', 'demography'):
    setattr(c.ms, attr, None)

# verify the analytical joint SFS agrees with the cached truth within the configured tolerances
for stat, tol in c.comparisons['tolerance']['jsfs'].items():
    diff = Comparison.rel_diff(np.array(get_stat(c.ms.jsfs, stat)), np.array(get_stat(c.ph.jsfs, stat))).max()
    print(f'{stat:>5}: rel_diff.max={diff:.4f} tol={tol} [{"ok" if diff <= tol else "FAIL"}]', flush=True)

# drop the cached analytical coalescent so only the small msprime ground truth is serialized (otherwise the joint
# state space would bloat the fixture); the analytical side is recomputed fresh at test time
c.__dict__.pop('ph', None)

c.to_file(out)
