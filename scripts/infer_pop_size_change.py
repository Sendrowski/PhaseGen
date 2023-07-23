from fastdfe import Spectra

import phasegen as pg

# observed SFS
observed = pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])


def get_dist(t: float, Ne: float) -> pg.Coalescent:
    """
    Get coalescent distribution.

    :param t: Time of discrete population size change.
    :param Ne: New population size at time ``t``.
    :return: Coalescent distribution.
    """
    return pg.PiecewiseTimeHomogeneousCoalescent(
        n=10,
        demography=pg.PiecewiseTimeHomogeneousDemography(
            pop_sizes=[1, Ne],
            times=[0, t]
        ),
        pbar=False,
        parallelize=False
    )


def loss(dist: pg.Coalescent) -> float:
    """
    Calculate loss.

    :param dist: Coalescent distribution
    :return: Loss
    """
    # modelled SFS

    # return Poisson likelihood
    return pg.PoissonLikelihood().compute(
        observed=observed.normalize().polymorphic,
        modelled=dist.sfs.mean.normalize().polymorphic
    )


# create inference object
inf = pg.Inference(
    x0=dict(t=1, Ne=1),
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    dist=get_dist,
    loss=loss,
)

# perform inference
inf.run()

s = Spectra.from_spectra(dict(
    modelled=inf.dist_inferred.sfs.mean.normalize() * observed.n_polymorphic,
    observed=observed
))

s.plot()

inf.dist_inferred.demography.plot(t_max=inf.dist_inferred.tree_height.mean)

pass
