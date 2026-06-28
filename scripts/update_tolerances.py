"""
Cheaply re-embed a comparison's tolerances / statistic selection into its **existing** serialized fixture, reusing
the cached msprime ground truth -- so tuning a tolerance (or adding/removing a statistic that needs no new cached
data) does not require re-running the 1e6-replicate simulation.

It aborts (pointing to ``create_comparison``) when a full regeneration is genuinely needed: a changed
simulation-defining parameter (``n``, ``pop_sizes``, model, seed, ...), or a newly requested pairwise *surface* pair
whose empirical grid was never cached.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"

from phasegen.comparison import Comparison

import os

try:
    yaml_file = snakemake.input[0]
    marker = snakemake.output[0]
except NameError:
    # testing / direct invocation
    name = "1_epoch_n_4"
    yaml_file = f"resources/configs/{name}.yaml"
    marker = None

# the fixture is read and rewritten in place -- deliberately *not* a snakemake input, so snakemake does not try to
# (re)build it via create_comparison (a full re-simulation) when the YAML is newer; it must already exist
config_name = os.path.splitext(os.path.basename(yaml_file))[0]
fixture = f"results/comparisons/serialized/{config_name}.json"
if not os.path.exists(fixture):
    raise FileNotFoundError(f"{fixture} does not exist; run the create_comparison rule first to generate it.")

old = Comparison.from_file(fixture)      # the existing fixture (carries the cached ground truth)
new = Comparison.from_yaml(yaml_file)    # the freshly edited config

# the cached ground truth is only valid if the simulation-defining parameters are unchanged (alpha/psi/c capture the
# Beta/Dirac model parameters; the model itself is compared by class, since instances have no value equality).
# NOTE: `seed` is deliberately excluded -- it only selects which random realization was drawn, not the distribution,
# so the existing cached sample stays a valid ground truth for a tolerance sync (we reuse it, we do not regenerate).
sim_attrs = ['n', 'pop_sizes', 'migration_rates', 'num_replicates', 'n_samples', 'n_loci', 'recombination_rate',
             'mutation_rate', 'alpha', 'psi', 'c']
changed = [a for a in sim_attrs if getattr(old, a, None) != getattr(new, a, None)]
if type(getattr(old, 'model', None)) is not type(getattr(new, 'model', None)):
    changed.append('model')
if changed:
    raise ValueError(f"Simulation-defining parameters changed {changed} for {fixture}; the cached ground truth is "
                     f"stale -- run the create_comparison rule to regenerate from scratch.")

# every requested pairwise *surface* pair must already have a cached empirical grid (touch caches the per-statistic
# and pointwise-pairwise data for all bins, so only the explicit surface pairs can be genuinely missing)
for dist, pairs in new._pairwise_surface_pairs().items():
    cached = {(e[0], e[1]) for e in getattr(getattr(old.ms, dist), '_joint_surface', [])}
    missing = [p for p in pairs if tuple(p) not in cached]
    if missing:
        raise ValueError(f"Pairwise surface pair(s) {missing} for '{dist}' are not cached in {fixture}; "
                         f"run the create_comparison rule to cache them.")

# swap in the new tolerances / statistic selection and re-serialize (no simulation)
old.comparisons = new.comparisons
old.to_file(fixture)

if marker is not None:
    with open(marker, 'w') as f:
        f.write(f"synced tolerances from {yaml_file} into {fixture}\n")

pass
