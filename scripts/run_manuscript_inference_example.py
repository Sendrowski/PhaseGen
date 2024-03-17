"""
Run the inference example from the manuscript and saves the results to a file.
"""
import fastdfe as fd
import phasegen as pg

inf = pg.Inference(
    x0=dict(t=1, Ne=1),
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=pg.SFS(
        [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
    ),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=True,
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    )
)

# perform inference
inf.run()

# plot results
import matplotlib.pyplot as plt

spectra = fd.Spectra.from_spectra(dict(
    modelled=inf.dist_inferred.sfs.mean.normalize() * inf.observation.n_polymorphic,
    observed=inf.observation
))

_, axs = plt.subplots(3, 1, figsize=(5, 10))

spectra.plot(ax=axs[0], show=False, title='SFS comparison')
inf.plot_pop_sizes(ax=axs[1], show=False)
inf.plot_bootstraps(ax=axs[2], show=False, subplots=False)

plt.tight_layout()
plt.savefig('scratch/manuscript_inference_example.png', dpi=400)
plt.show()

inf.to_file('scratch/inference.json')

pass
