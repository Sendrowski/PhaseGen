import numpy as np
from fastdfe import Spectra
import psutil

import phasegen as pg

# observed (neutral) SFS
observed = pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])

pg.logger.setLevel('DEBUG')


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


def loss(coal: pg.Coalescent) -> float:
    """
    Calculate loss by using the Poisson likelihood.

    :param coal: Coalescent distribution
    :return: Loss
    """
    return pg.PoissonLikelihood().compute(
        observed=observed.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    )


# create inference object
inf = pg.Inference(
    x0=dict(t=1, Ne=1),
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    dist=get_coal,
    loss=loss
)

# perform inference
inf.run()

spectra = Spectra.from_spectra(dict(
    modelled=inf.dist_inferred.sfs.mean.normalize() * observed.n_polymorphic,
    observed=observed
))

spectra.plot()

# plot inferred demography
inf.dist_inferred.demography.plot_pop_sizes(
    t=np.linspace(0, inf.dist_inferred.tree_height.quantile(0.99), 100)
)

pass
