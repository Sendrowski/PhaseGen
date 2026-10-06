"""
Cheaply re-embed a comparison's tolerances / statistic selection into its **existing** serialized fixture, reusing
the cached msprime ground truth -- so tuning a tolerance (or adding/removing a statistic that needs no new cached
data) does not require re-running the 1e6-replicate simulation.

It aborts (pointing to ``create_comparison``) when a full regeneration is genuinely needed: a changed
ground-truth-defining parameter (``n``, ``pop_sizes``, model, ``end_time``, ...), a newly requested pairwise
*surface* pair (msprime or sampler), compared distribution, atom-conditional pair or coalescent-level statistic whose
ground truth was never cached, or windowed-conditional windows other than the cached ones.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"

from phasegen.comparison import Comparison
from phasegen.distributions import MsprimeCoalescent

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

# the fixture is only valid if the parameters defining the simulation and the analytic model are unchanged
# (alpha/psi/c capture the Beta/Dirac model parameters; the model itself is compared by class, since instances have
# no value equality).
# NOTE: `seed` is deliberately excluded -- it only selects which random realization was drawn, not the distribution,
# so the existing cached sample stays a valid ground truth for a tolerance sync (we reuse it, we do not regenerate).
sim_attrs = ['n', 'pop_sizes', 'migration_rates', 'num_replicates', 'n_samples', 'n_loci', 'recombination_rate',
             'n_unlinked', 'mutation_rate', 'record_migration', 'simulate_mutations', 'mass_threshold', 'end_time',
             'alpha', 'psi', 'c']
changed = [a for a in sim_attrs if getattr(old, a, None) != getattr(new, a, None)]
if type(getattr(old, 'model', None)) is not type(getattr(new, 'model', None)):
    changed.append('model')
if changed:
    raise ValueError(f"Ground-truth-defining parameters changed {changed} for {fixture}; the cached ground truth is "
                     f"stale -- run the create_comparison rule to regenerate from scratch.")

tolerance = new.comparisons.get('tolerance', {})
msprime_spec = {k: v for k, v in tolerance.items() if k != 'empirical'}
empirical_spec = tolerance.get('empirical', {})


def require_cached(what: str, missing: list) -> None:
    """Abort when configured checks need ground truth the fixture does not cache."""
    if missing:
        raise ValueError(f"{what} {missing} are not cached in {fixture}. Run the create_comparison rule to cache them.")


# the msprime operand caches only the distributions it compares
cached = getattr(old.__dict__.get('ms'), '__dict__', {})
require_cached("Distributions", [name for name in MsprimeCoalescent._distributions
                                 if name in new._expand_keys(msprime_spec) and name not in cached])

# every requested pairwise *surface* pair must already have a cached empirical grid (_touch caches the per-statistic
# and pointwise-pairwise data for all bins, so only the explicit surface pairs can be genuinely missing), on the
# msprime operand and, for the nested 'empirical' block, on the sampler operand
for operand, spec in (('ms', msprime_spec), ('empirical', empirical_spec)):
    for dist, pairs in new._pairwise_surface_pairs(spec).items():
        cached = {(e[0], e[1]) for e in getattr(getattr(old.__dict__.get(operand), dist, None), '_joint_surface', [])}
        require_cached(f"Pairwise surface pairs of '{dist}' ({operand})", [p for p in pairs if tuple(p) not in cached])

# the atom-conditional ground truth is cached per bin pair
for dist, pairs in new._atom_conditional_pairs(msprime_spec).items():
    cached = {(e[0], e[1]) for e in getattr(getattr(old.ms, dist), '_atom_conditional', [])}
    require_cached(f"Atom-conditional pairs of '{dist}'", [p for p in pairs if tuple(p) not in cached])

# the coalescent-level statistics are cached per name and arguments
statistics = [(stat, tuple(spec.get('args', []) if isinstance(spec, dict) else []))
              for stat, spec in new.comparisons.get('statistics', {}).items()]
require_cached("Statistics", [s for s in statistics if s not in getattr(old, '_ms_statistics', {})])

# the windowed-conditional ground truth is cached at the windows the configured quantiles and window define, for the
# bin pairs and the locus pair separately
for loci in (False, True):
    attr = '_loci_windowed_conditional' if loci else '_windowed_conditional'
    for dist, specs in new._windowed_conditional_specs(msprime_spec, loci=loci).items():
        if Comparison._stale_windows(getattr(getattr(old.ms, dist), attr, []), specs):
            raise ValueError(f"The windowed-conditional windows of '{dist}' differ from those cached in {fixture}. "
                             f"Run the create_comparison rule to cache them.")

# swap in the new tolerances / statistic selection and re-serialize (no simulation)
old.comparisons = new.comparisons
old.to_file(fixture)

if marker is not None:
    with open(marker, 'w') as f:
        f.write(f"synced tolerances from {yaml_file} into {fixture}\n")

pass
