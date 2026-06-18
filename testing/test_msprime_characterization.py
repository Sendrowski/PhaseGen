"""
Characterization (golden-master) test pinning the raw per-replicate output of :class:`MsprimeCoalescent.simulate`.

This is the guardrail for refactoring the simulation internals (e.g. splitting the monolithic ``simulate_batch``
loop into per-statistic accumulator components): the simulated arrays -- which are the *ground truth* the entire
analytic test suite is validated against -- must stay **bit-identical** under a behaviour-preserving refactor.

With a fixed ``seed`` and ``n_threads=1, parallelize=False`` the msprime simulation is fully deterministic, so the
arrays are reproducible run-to-run. A committed baseline (``fixtures/msprime_characterization.npz``) is generated
from the current code (``REGENERATE=1 pytest ...`` or delete the file) and the test asserts the live output matches
it exactly. The baseline is tied to the installed msprime version; regenerate it if msprime is upgraded.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent

#: Small, fully deterministic configurations spanning the simulation branches (single/multi-population with the
#: migration-recording path + joint SFS, two loci with recombination, multiple-merger models, and mutations).
CONFIGS = {
    'standard_n4': dict(n=4),
    'beta_n4': dict(n=4, model=pg.BetaCoalescent(alpha=1.5)),
    'dirac_n4': dict(n=4, model=pg.DiracCoalescent(psi=0.5, c=1.0)),
    'two_pop_migration_n2': dict(
        n={'pop_0': 2, 'pop_1': 2},
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.5}},
                                 migration_rates={('pop_0', 'pop_1'): {0: 1.0}, ('pop_1', 'pop_0'): {0: 1.0}}),
        record_migration=True,
    ),
    'two_loci_n3_r1': dict(n=3, loci=2, recombination_rate=1.0),
    'mutations_n4': dict(n=4, simulate_mutations=True, mutation_rate=2.0),
}

#: Result attributes set by :meth:`MsprimeCoalescent.simulate` that a refactor must preserve exactly.
FIELDS = ('heights', 'total_branch_lengths', 'sfs_lengths', 'mutations', 'jsfs_moments', 'jsfs_samples')

BASELINE = Path(__file__).parent / 'fixtures' / 'msprime_characterization.npz'


def _simulate(name: str) -> dict:
    """Run the deterministic simulation for ``name`` and return its raw result arrays."""
    coal = MsprimeCoalescent(num_replicates=200, seed=42, n_threads=1, parallelize=False, **CONFIGS[name])
    coal.simulate()
    out = {}
    for field in FIELDS:
        val = getattr(coal, field)
        out[field] = np.asarray(val, dtype=float) if val is not None else np.array([np.nan])
    return out


def _key(name: str, field: str) -> str:
    return f"{name}__{field}"


def _generate_baseline() -> dict:
    """Compute the full baseline (all configs x fields) from the current code."""
    data = {}
    for name in CONFIGS:
        for field, arr in _simulate(name).items():
            data[_key(name, field)] = arr
    return data


if os.environ.get('REGENERATE') or not BASELINE.exists():
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASELINE, **_generate_baseline())


@pytest.mark.parametrize('name', list(CONFIGS), ids=list(CONFIGS))
def test_simulate_matches_baseline(name):
    """The deterministic simulation output matches the committed baseline bit-for-bit (per result field)."""
    baseline = np.load(BASELINE)
    out = _simulate(name)
    for field, arr in out.items():
        expected = baseline[_key(name, field)]
        assert arr.shape == expected.shape, f"{name}.{field}: shape {arr.shape} != {expected.shape}"
        np.testing.assert_array_equal(arr, expected, err_msg=f"{name}.{field} drifted from the baseline")
