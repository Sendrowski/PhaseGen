"""
Code blocks for the manuscript.
"""
# toggle 'show indent guides'

# noinspection all
# ------------------------------------------------------------------
inf = pg.Inference(
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    observation=pg.SFS(
        [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.polymorphic,
        modelled=(
                coal.sfs.mean.polymorphic /
                (coal.sfs.mean.theta * coal.sfs.mean.n_sites) *
                (obs.theta * obs.n_sites)
        )
    ),
    bounds=dict(t=(0, 4), Ne=(0.1, 10)),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=True,
    parallelize=True
)
# ------------------------------------------------------------------
