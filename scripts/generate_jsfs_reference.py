"""
Generate an independent joint-SFS reference using the ``moments`` package, for an independent cross-check of the
analytical joint SFS (``moments`` uses a moment-closure ODE, distinct from PhaseGen and msprime).

For each single-epoch, two-population, standard-coalescent config in ``resources/configs/{name}.yaml`` this builds
the migration-drift equilibrium island model in ``moments`` and writes the normalized expected joint SFS to
``results/jsfs_reference/{name}.json`` (normalizing removes the ``theta`` scaling, leaving the spectrum shape).

Migration maps directly (verified to ~2e-4, no scaling factor): ``moments`` ``m[i, j]`` (rate into ``i`` from
``j``) equals PhaseGen's ``migration_rates[(pop_i, pop_j)]``.

Run via snakemake (executes in the ``envs/dev.yaml`` conda env providing ``moments``) or directly with
``python scripts/generate_jsfs_reference.py``.
"""

import sys

sys.path.append('.')

import json
from pathlib import Path

import numpy as np
import yaml

import moments

# sample size used internally by moments' moment-closure integration (the result is projected down to the requested
# sample sizes afterwards; larger values reduce the jackknife closure error), and the equilibrium integration time
n_int = None
tf = 40.0

try:
    config_file = snakemake.input[0]  # noqa: F821
    out = snakemake.output[0]  # noqa: F821
except NameError:
    # direct invocation: regenerate every supported reference
    config_file = None
    out = None

# two-population, single-epoch, standard-coalescent island models for which a moments equilibrium reference is built
SUPPORTED_CONFIGS = [
    '1_epoch_2_pops_n_4_jsfs',
    '1_epoch_2_pops_n_6_jsfs',
    '1_epoch_2_pops_n_6_asym_jsfs',
    '1_epoch_2_pops_n_8_jsfs',
]


def load_scenario(path: str) -> dict:
    """
    Parse the demographic scenario from a PhaseGen joint-SFS config, independently of PhaseGen itself.

    :param path: Path to the config YAML.
    :return: Dictionary with ``name``, ordered ``pops``, ``n``, ``pop_sizes`` and ``migration_rates``.
    :raises ValueError: If the scenario is not a single-epoch, two-population, standard-coalescent model.
    """
    with open(path) as f:
        cfg = yaml.unsafe_load(f)

    pops = list(cfg['n'])

    if len(pops) != 2:
        raise ValueError(f"The moments reference only supports two populations, got {len(pops)}: {pops}.")

    if cfg.get('model', 'standard') != 'standard':
        raise ValueError(f"The moments reference only supports the standard coalescent, got '{cfg.get('model')}'.")

    # require a single epoch (only time 0) for both sizes and migration rates so the island model is at equilibrium
    for p in pops:
        if list(cfg['pop_sizes'][p]) != [0]:
            raise ValueError(f"The moments reference only supports constant (single-epoch) population sizes for {p}.")

    for pair, rates in cfg['migration_rates'].items():
        if list(rates) != [0]:
            raise ValueError(f"The moments reference only supports constant (single-epoch) migration rates for {pair}.")

    return dict(
        name=Path(path).stem,
        pops=pops,
        n={p: int(cfg['n'][p]) for p in pops},
        pop_sizes={p: float(cfg['pop_sizes'][p][0]) for p in pops},
        migration_rates={pair: float(rates[0]) for pair, rates in cfg['migration_rates'].items()},
    )


def moments_island_jsfs(scenario: dict, n_int: int = None, tf: float = 40.0) -> np.ndarray:
    """
    Compute the normalized expected joint SFS of the two-population migration-drift equilibrium island model.

    :param scenario: Scenario as returned by :func:`load_scenario`.
    :param n_int: Internal per-population sample size for integration (moments' jackknife closure needs it large;
        the result is projected down). Defaults to ``max(n) + 14``.
    :param tf: Integration time; large enough to reach the island-model equilibrium.
    :return: Normalized joint SFS array of shape ``(n_0 + 1, n_1 + 1)`` (monomorphic corners zeroed).
    """
    pops = scenario['pops']
    ns = [scenario['n'][p] for p in pops]
    nu = [scenario['pop_sizes'][p] for p in pops]

    # moments m[i, j] = rate into pop i from pop j = PhaseGen migration_rates[(pop_i, pop_j)]
    m = np.zeros((2, 2))
    for i, pi in enumerate(pops):
        for j, pj in enumerate(pops):
            if i != j:
                m[i, j] = scenario['migration_rates'][(pi, pj)]

    if n_int is None:
        n_int = max(ns) + 14

    # ancestral equilibrium, split into the two demes, then integrate to the island-model equilibrium
    fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(2 * n_int))
    fs = moments.Manips.split_1D_to_2D(fs, n_int, n_int)
    fs.integrate(nu, tf, m=m)

    arr = np.array(fs.project(ns).data)

    # zero the monomorphic corners (which carry no joint-SFS information) and normalize to the spectrum shape
    arr[0, 0] = 0.0
    arr[ns[0], ns[1]] = 0.0

    return arr / arr.sum()


def generate(config_file: str, out: str, n_int: int = None, tf: float = 40.0):
    """
    Generate and write a single moments joint-SFS reference.

    :param config_file: Path to the config YAML.
    :param out: Output path for the reference JSON.
    :param n_int: Internal integration sample size (see :func:`moments_island_jsfs`).
    :param tf: Integration time.
    """
    scenario = load_scenario(config_file)
    jsfs = moments_island_jsfs(scenario, n_int=n_int, tf=tf)

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, 'w') as f:
        json.dump(dict(
            name=scenario['name'],
            method=f"moments island-model equilibrium (n_int={n_int or max(scenario['n'].values()) + 14}, tf={tf})",
            n=scenario['n'],
            pop_sizes=scenario['pop_sizes'],
            # serialize the (tuple-keyed) migration rates as a list of [source, dest, rate] triples
            migration_rates=[[pi, pj, rate] for (pi, pj), rate in scenario['migration_rates'].items()],
            jsfs=jsfs.tolist(),
        ), f, indent=2)

    print(f"wrote {out}", flush=True)


if config_file is not None:
    generate(config_file, out, n_int=n_int, tf=tf)
else:
    for name in SUPPORTED_CONFIGS:
        generate(f"resources/configs/{name}.yaml", f"results/jsfs_reference/{name}.json", n_int=n_int, tf=tf)
