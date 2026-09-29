"""
Characterization (golden-master) test pinning the raw per-replicate output of :class:`MsprimeCoalescent.simulate`.

This is the guardrail for refactoring the simulation internals (e.g. splitting the monolithic ``simulate_batch``
loop into per-statistic accumulator components): the simulated arrays, which are the ground truth the entire
analytic test suite is validated against, must stay identical to within ``RTOL`` under a behaviour-preserving
refactor. The comparison is a relative tolerance rather than an exact one because floating-point reassociation
differs between architectures, so a baseline generated on one machine differs from another in the last few bits
(measured at 3.5e-16 relative, about 1.5 double-precision epsilon, between arm64 macOS and x86-64 Linux), while a
genuine behavioural change moves the arrays far more than that.

With a fixed ``seed`` and ``n_threads=1, parallelize=False`` the msprime simulation is fully deterministic, so the
arrays are reproducible run-to-run. A committed baseline (``fixtures/msprime_characterization.npz``) is generated
from the current code (``REGENERATE=1 pytest ...`` or delete the file) and the test asserts the live output matches
it. The baseline is tied to the installed msprime version; regenerate it if msprime is upgraded.
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

#: Relative tolerance of the baseline comparison, loose enough to absorb cross-architecture rounding and far tighter
#: than any behavioural change.
RTOL = 1e-12


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
    """The deterministic simulation output matches the committed baseline to within ``RTOL`` (per result field)."""
    baseline = np.load(BASELINE)
    out = _simulate(name)
    for field, arr in out.items():
        expected = baseline[_key(name, field)]
        assert arr.shape == expected.shape, f"{name}.{field}: shape {arr.shape} != {expected.shape}"
        np.testing.assert_allclose(arr, expected, rtol=RTOL, atol=0,
                                   err_msg=f"{name}.{field} drifted from the baseline")


def test_simulating_mutations_requires_a_mutation_rate():
    """Regression: ``simulate_mutations=True`` without a mutation rate silently simulated no mutations."""
    with pytest.raises(ValueError, match='mutation rate'):
        MsprimeCoalescent(n=3, simulate_mutations=True)


def test_per_deme_tree_height_is_undefined_for_several_loci():
    """The tree height of several loci is their maximum, which the per-deme heights summed over loci do not
    decompose, so the per-deme tree height raises as for the exact coalescent. The additive total branch length keeps
    its per-deme breakdown. Regression: the per-deme heights were the sums of the per-locus heights."""
    ms = MsprimeCoalescent(n=4, loci=2, recombination_rate=1.0, num_replicates=200, n_threads=2, parallelize=False,
                           seed=1)

    with pytest.raises(NotImplementedError):
        _ = ms.tree_height.demes

    assert ms.total_branch_length.demes['pop_0'].mean == pytest.approx(ms.total_branch_length.mean)

    ms.tree_height._touch(np.linspace(0, 1, 3))
    ms.tree_height._drop()
