"""
Compare execution time and record SFS computed by PhaseGen and dadi across lineage and deme configurations.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-09"

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

try:
    import sys

    sys.path.append('.')
    testing = False
    out = snakemake.output[0]
except NameError:
    testing = True
    out = "scratch/dadi_comparison.png"

# PhaseGen
import phasegen as pg

pg.Backend.register(pg.SciPyExpmBackend())


def time_sfs_phasegen(n, d):
    start = time.time()
    coal = pg.Coalescent(
        n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
        loci=pg.LocusConfig()
    )
    sfs = coal.sfs.mean
    runtime = time.time() - start
    return runtime, sfs


def time_sfs_dadi(n, d):
    import dadi
    import time

    sample_sizes = [n] * d

    # use two epoch model to see how changing demography affects dadi runtime
    def two_epoch_model(params, ns, pts):
        nu, T = params
        if d == 1:
            return dadi.Demographics1D.two_epoch(params, ns, pts)
        elif d == 2:
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.Integration.two_pops(phi, xx, T=T, nu1=nu, nu2=nu, m12=10, m21=10)

            return dadi.Spectrum.from_phi(phi, ns, (xx,) * d)

    neutral_model_ex = dadi.Numerics.make_extrap_func(two_epoch_model)

    pts = [n + 20, n + 30, n + 40]
    start = time.time()
    sfs = neutral_model_ex([1, 1], sample_sizes, pts)  # Example: nu=2, T=1
    runtime = time.time() - start
    return runtime, sfs


# Configurations
N = np.arange(2, 11)
D = np.arange(1, 3)
shape = (len(N), len(D))

data_phasegen = np.zeros(shape)
data_dadi = np.zeros(shape)
spectra_phasegen = {}
spectra_dadi = {}

pbar = tqdm(total=2 * len(N) * len(D))
for i, n in enumerate(N):
    for j, d in enumerate(D):
        t_phasegen, sfs_phasegen = time_sfs_phasegen(n, d)
        data_phasegen[i, j] = t_phasegen
        spectra_phasegen[(n, d)] = sfs_phasegen.data
        pbar.update(1)

        t_dadi, sfs_dadi = time_sfs_dadi(n, d)
        data_dadi[i, j] = t_dadi
        spectra_dadi[(n, d)] = sfs_dadi.data
        pbar.update(1)
pbar.close()

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

norm = mpl.colors.LogNorm(vmin=min(data_phasegen.min(), data_dadi.min()),
                          vmax=max(data_phasegen.max(), data_dadi.max()))

sns.heatmap(data_phasegen, annot=True, fmt=".3f", xticklabels=D, yticklabels=N,
            cmap='viridis', norm=norm, ax=ax[0], cbar=False)
ax[0].set_title("Runtime (s) – PhaseGen")
ax[0].set_xlabel("n demes")
ax[0].set_ylabel("n lineages")
ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)

sns.heatmap(data_dadi, annot=True, fmt=".3f", xticklabels=D, yticklabels=N,
            cmap='viridis', norm=norm, ax=ax[1], cbar=False)
ax[1].set_title("Runtime (s) – dadi")
ax[1].set_xlabel("n demes")
ax[1].set_ylabel("n lineages")
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)

fig.tight_layout(pad=2)
plt.savefig(out)

if testing:
    plt.show()

pass
