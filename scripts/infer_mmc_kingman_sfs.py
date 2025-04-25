"""
Infer demographic history from MMC SFS using SFS only.
"""

import phasegen as pg

try:
    testing = False
    inf_kingman = snakemake.input[0]
    n = snakemake.params.n
    n_bootstraps = snakemake.params.get('n_bootstraps', 100)
    n_runs = snakemake.params.get('n_runs', 10)
    do_bootstrap = snakemake.params.get('do_bootstrap', True)
    parallelize = snakemake.params.get('parallelize', True)
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    inf_kingman = "results/inference/MMC_Kingman.json"
    n = 10
    n_bootstraps = 100
    n_runs = 10
    do_bootstrap = True
    parallelize = True
    out = "scratch/inf_sfs.json"

dist_inferred = pg.Coalescent.from_file(inf_kingman)
sfs = dist_inferred.sfs.mean
tree_height = dist_inferred.tree_height.mean

inf_sfs = pg.Inference(
    coal=lambda t, Ne, alpha: pg.Coalescent(
        n=n,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        ),
        model=pg.BetaCoalescent(alpha=alpha, scale_time=False)
    ),
    observation=sfs,
    loss=lambda coal, obs: pg.MultinomialLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    bounds=dict(t=(0, tree_height), Ne=(0.01, 10), alpha=(1.00001, 1.99999)),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=do_bootstrap,
    n_bootstraps=n_bootstraps,
    n_runs=n_runs,
    parallelize=parallelize
)

inf_sfs.run()
inf_sfs.to_file(out)
