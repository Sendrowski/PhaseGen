from fastdfe import Spectra

import phasegen as pg

pg.logger.setLevel('DEBUG')

# create inference object
inf = pg.Inference(
    x0=dict(t=1, Ne=1),
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
    resample=lambda sfs, _: sfs.resample(),
    do_bootstrap=False,
    parallelize=False,
    n_runs=1,
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    loss=lambda coal, observation: pg.PoissonLikelihood().compute(
        observed=observation.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    cache=False
)

# perform inference
inf.run()

# plot results
import matplotlib.pyplot as plt

spectra = Spectra.from_spectra(dict(
    modelled=inf.dist_inferred.sfs.mean.normalize() * inf.observation.n_polymorphic,
    observed=inf.observation
))

_, axs = plt.subplots(2, 2, figsize=(9, 7))

spectra.plot(ax=axs[0, 0], show=False)
inf.plot_pop_sizes(ax=axs[0, 1], show=False)
inf.plot_bootstraps(ax=axs[1, 0], show=False, subplots=False)

plt.tight_layout()
plt.savefig('scratch/manuscript_inference_example.png', dpi=400)
plt.show()

inf.to_file('scratch/inference.json')

pass
