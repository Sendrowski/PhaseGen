"""
Fit MMC scenario to Kingman SFS.
"""

import phasegen as pg

try:
    testing = False
    n = snakemake.params.n
    do_bootstrap = snakemake.params.get('do_bootstrap', True)
    parallelize = snakemake.params.get('parallelize', True)
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 10
    do_bootstrap = True
    parallelize = True
    out = "scratch/inf_kingman.json"

inf = pg.Inference(
    coal=lambda t1, N1, t2, N2: (
        pg.Coalescent(
            n=n,
            model=pg.BetaCoalescent(alpha=1.7, scale_time=False),
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1, t1: N1, t2: N2}}
            )
        )
    ),
    observation=pg.SFS.standard_kingman(n=n),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    bounds=dict(t1=(0, 5), N1=(0.1, 10), t2=(0, 5), N2=(0.1, 10)),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=do_bootstrap,
    parallelize=parallelize,
    n_runs=10,
    n_bootstraps=100
)

# inf.run()
#inf.to_file(out_inf)

dist = inf.coal(t1=0.5, N1=4.5, t2=4.5, N2=0.1)
dist.to_file(out)
