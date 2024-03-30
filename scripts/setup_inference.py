"""
Set up inference object.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-30"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    out = "scratch/inference.json"

import phasegen as pg

# observed (neutral) SFS
observed = pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])


def get_coal(t: float, Ne: float) -> pg.Coalescent:
    """
    Get coalescent distribution.

    :param t: Time of discrete population size change.
    :param Ne: New population size at time ``t``.
    :return: Coalescent distribution.
    """
    return pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    )


def loss(coal: pg.Coalescent, observation: pg.SFS) -> float:
    """
    Calculate loss by using the Poisson likelihood.

    :param coal: Coalescent distribution
    :param observation: Observed SFS
    :return: Loss
    """
    return pg.PoissonLikelihood().compute(
        observed=observation.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    )


# create inference object
inf = pg.Inference(
    x0=dict(t=1, Ne=1),
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=observed,
    resample=lambda sfs, _: sfs.resample(),
    coal=get_coal,
    loss=loss,
    cache=True
)

# perform inference
inf.run()

inf.to_file(out)

pass
