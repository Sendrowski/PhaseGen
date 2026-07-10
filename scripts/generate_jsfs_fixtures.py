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

# split the tolerance into the top-level msprime stats and the nested self-consistency ``empirical`` sub-spec, so a
# config may validate the joint SFS against msprime, the phasegen sampler, or both
tol = c._expand_keys(c.comparisons.get('tolerance', {}))
empirical_spec = tol.get('empirical')
msprime_spec = {k: v for k, v in tol.items() if k != 'empirical'}

# cache the msprime joint-SFS ground truth (accumulated within simulate()) -- the moments are retained by the cached
# jsfs distribution -- then null the raw per-replicate samples and the demography to keep the fixture small. Cache the
# full-grid joint *surface* ground truth for any configured pairwise surface pairs (config-pair keys like
# ``((0, 1), (1, 0))`` under ``jsfs: pairwise: cosine``) before dropping the per-replicate samples (``drop`` only nulls
# the samples, leaving ``_joint_surface`` intact)
if msprime_spec:
    for dist, pairs in c._pairwise_surface_pairs(msprime_spec).items():
        getattr(c.ms, dist).cache_joint_surface(pairs)
    c.ms.jsfs.drop()
    for attr in ('heights', 'total_branch_lengths', 'sfs_lengths', 'mutations', 'jsfs_moments', 'jsfs_samples', 'demography'):
        setattr(c.ms, attr, None)

    # verify the analytical joint SFS agrees with the cached truth within the configured tolerances (the 'joint' and
    # 'pairwise' stats are not ``.data`` spectra -- validated at test time via Comparison.compare_stat -- so skip)
    for stat, tol_ in msprime_spec.get('jsfs', {}).items():
        if stat in ('joint', 'pairwise'):
            continue
        diff = Comparison.rel_diff(np.array(get_stat(c.ms.jsfs, stat)), np.array(get_stat(c.ph.jsfs, stat))).max()
        print(f'{stat:>5}: rel_diff.max={diff:.4f} tol={tol_} [{"ok" if diff <= tol_ else "FAIL"}]', flush=True)

# cache the phasegen-sampler (self-consistency) ground truth for the nested ``empirical`` sub-spec, if present
if empirical_spec:
    c.empirical.touch()
    for dist, pairs in c._pairwise_surface_pairs(empirical_spec).items():
        getattr(c.empirical, dist).cache_joint_surface(pairs)
    c.empirical.drop()

# drop the cached analytical coalescent so only the small msprime ground truth is serialized (otherwise the joint
# state space would bloat the fixture); the analytical side is recomputed fresh at test time
c.__dict__.pop('ph', None)

c.to_file(out)
