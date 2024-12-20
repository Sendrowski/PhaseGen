"""
Infer demographic history from MMC SFS using SFS only.
"""

import numpy as np

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

def get_corr(coal: pg.Coalescent) -> float:
    """
    Get SFS correlation statistic.

    :param coal: Coalescent distribution.
    :return: Correlation coefficient.
    """
    return coal.sfs.get_corr(4, 5)

dist_inferred = pg.Coalescent.from_file(inf_kingman)
sfs = dist_inferred.sfs.mean
tree_height = dist_inferred.tree_height.mean
m2 = get_corr(dist_inferred)


def loss(coal: pg.Coalescent, obs: (pg.SFS, float)) -> float:
    """
    Loss function for inference.

    :param coal: Coalescent distribution
    :param obs: observation
    :return: loss
    """
    sfs, m2 = obs

    loss_sfs = pg.PoissonLikelihood().compute(
        observed=sfs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    )

    loss_m2 = np.abs(
        m2 - get_corr(coal)
    )

    pg.logger.debug(f'loss: {dict(sfs=loss_sfs, m12=loss_m2)}')

    return loss_sfs + loss_m2


inf_m12 = pg.Inference(
    coal=lambda t, Ne, alpha: pg.Coalescent(
        n=n,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        ),
        model=pg.BetaCoalescent(alpha=alpha, scale_time=False)
    ),
    observation=(sfs, m2),
    loss=loss,
    bounds=dict(t=(0, tree_height), Ne=(0.01, 10), alpha=(1.00001, 1.99999)),
    resample=lambda obs, _: (obs[0].resample(), obs[1]),
    do_bootstrap=do_bootstrap,
    n_bootstraps=n_bootstraps,
    n_runs=n_runs,
    parallelize=parallelize
)

inf_m12.run()
inf_m12.to_file(out)
