"""
Run the inference example from the manuscript.
"""
import fastdfe as fd
import matplotlib.pyplot as plt

import phasegen as pg

# set computation backend
pg.Backend.register(pg.TensorFlowExpmBackend())

inf = pg.Inference(
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    observation=pg.SFS(
        [0, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0]
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    bounds=dict(t=(0, 4), Ne=(0.1, 10)),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=True,
    parallelize=True
)

# perform inference
inf.run()

spectra = fd.Spectra.from_spectra(dict(
    fitted=inf.dist_inferred.sfs.mean.normalize() * inf.observation.n_polymorphic,
    observed=inf.observation
))

_, axs = plt.subplots(2, 2, figsize=(6, 5))

spectra.plot(ax=axs[0, 0], show=False, title='SFS comparison')
inf.plot_pop_sizes(ax=axs[0, 1], show=False)
inf.plot_bootstraps(ax=axs[1], show=False, kwargs={'bins': 30}, title=['Marginal distribution'] * 2)

plt.tight_layout()
plt.savefig('reports/manuscripts/merged/figures/inference_result.png', dpi=400)
plt.show()

inf.to_file('scratch/inference.json')

pass
