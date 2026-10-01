"""
Tests for the marginal containers and argument validation of the empirical distributions of
:class:`~phasegen.distributions.MsprimeCoalescent`.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions.empirical import MsprimeCoalescent

#: Two-deme demography with migration.
DEMOGRAPHY = pg.Demography(pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2}},
                           migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 0.5})


@pytest.mark.parametrize('kwargs, attr', [
    (dict(n={'pop_0': 2, 'pop_1': 2}, demography=DEMOGRAPHY, record_migration=True), 'demes'),
    (dict(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1.0)), 'loci'),
])
def test_marginals_get_cov_and_get_corr(kwargs, attr):
    """
    The per-deme and per-locus marginals of the scalar and spectrum statistics give the entries of their covariance
    and correlation matrices by key, the variances of the marginals on the diagonal, and raise ValueError for an
    unknown key.
    """
    ms = MsprimeCoalescent(num_replicates=500, n_threads=1, parallelize=False, seed=1, **kwargs)

    for dist in (ms.total_branch_length, ms.sfs, ms.fsfs):
        marginals = getattr(dist, attr)
        keys = list(marginals)

        for i, a in enumerate(keys):
            var = marginals[a].var
            np.testing.assert_allclose(marginals.get_cov(a, a), getattr(var, 'data', var), rtol=1e-10, atol=1e-12)

            for j, b in enumerate(keys):
                np.testing.assert_array_equal(marginals.get_cov(a, b), np.asarray(marginals.cov)[i, j])
                np.testing.assert_array_equal(marginals.get_corr(a, b), np.asarray(marginals.corr)[i, j])

        with pytest.raises(ValueError):
            marginals.get_cov('unknown', keys[0])


def test_jsfs_of_mixture_with_differing_configurations_raises_before_simulating():
    """
    The joint SFS of an initial distribution whose lineage configurations differ raises ValueError without
    simulating.
    """
    ms = MsprimeCoalescent(n=pg.InitialDistribution([(1, [2, 1]), (1, [1, 2])]), demography=DEMOGRAPHY,
                           num_replicates=10, n_threads=1, parallelize=False, seed=1)

    with pytest.raises(ValueError, match='single lineage configuration'):
        _ = ms.jsfs

    assert ms.heights is None
