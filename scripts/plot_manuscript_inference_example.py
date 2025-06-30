"""
Run manuscript inference example.
"""
import re

import matplotlib.pyplot as plt

import phasegen as pg

# set computation backend
pg.Backend.register(pg.SciPyExpmBackend())

#pg.logger.setLevel(pg.logging.DEBUG)

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
    parallelize=False,
)

# perform inference
inf.run()

nfev_runs = inf.runs['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))
nfev_bootstraps = inf.bootstraps['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))

print(f"Mean number of function evaluations (runs): {nfev_runs.mean()}")
print(f"Mean number of function evaluations (bootstraps): {nfev_bootstraps.mean()}")

spectra = pg.Spectra.from_spectra(dict(
    fitted=inf.dist_inferred.sfs.mean /
           (inf.dist_inferred.sfs.mean.theta * inf.dist_inferred.sfs.mean.n_sites) *
           (inf.observation.theta * inf.observation.n_sites),
    observed=inf.observation
))

_, axs = plt.subplots(2, 2, figsize=(6, 5))

spectra.plot(ax=axs[0, 0], show=False, title='SFS comparison')
inf.plot_pop_sizes(ax=axs[0, 1], show=False)
inf.plot_bootstraps(ax=axs[1], show=False, kwargs={'bins': 30}, title=['Marginal distribution'] * 2)

# Add labels A, B, C, D to the plots
for i, ax in enumerate(axs.flat):
    ax.text(-0.05, 1.125, ['A', 'B', 'C', 'D'][i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top',
            ha='right')

plt.tight_layout()
plt.savefig('reports/manuscripts/main/figures/inference_result.png', dpi=400)
plt.show()

inf.to_file('scratch/inference.json')

pass
