from fastdfe import Spectra

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
    resample=lambda sfs: sfs.resample(),
    do_bootstrap=True,
    parallelize=True,
    n_bootstraps=100,
    coal=get_coal,
    loss=loss,
    cache=True
)

# perform inference
inf.run()

spectra = Spectra.from_spectra(dict(
    modelled=inf.dist_inferred.sfs.mean.normalize() * observed.n_polymorphic,
    observed=observed
))

spectra.plot()

inf.plot_demography()
inf.plot_pop_sizes()
inf.plot_migration()
inf.plot_bootstraps(kind='hist')
inf.plot_bootstraps(kind='kde')

inf.to_file('scratch/inference.json')

pass
