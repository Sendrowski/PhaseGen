"""
Compare inference runtime between dadi and PhaseGen for 3-epoch SFS with bottleneck
"""

import re
import time
import warnings
from typing import Tuple

import dadi
import fastdfe as fd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

import phasegen as pg

pg.Settings.flatten_block_counting = True

bounds = dict(t=(0, 4), Ne=(0.1, 10))
n_runs = 100
n_bootstraps = 1
sample_sizes = np.arange(5, 26, 2)


def simulate_bottleneck_sfs(n: int) -> pg.SFS:
    """
    Simulate SFS from a 3-epoch bottleneck scenario using PhaseGen.
    """
    demography = pg.Demography(
        pop_sizes={'pop_0': {0: 1, 0.5: 0.2, 1: 1}}
    )
    coal = pg.Coalescent(n=n, demography=demography)
    return coal.sfs.mean


def run_dadi(sfs: dadi.Spectrum) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fit 2-epoch model using dadi.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    ns = sfs.sample_sizes[0]
    pts_l = np.array([ns + 20, ns + 30, ns + 40])
    lower, upper = [bounds['t'][0], bounds['Ne'][0]], [bounds['t'][1], bounds['Ne'][1]]

    def two_epoch(params, ns, pts):
        T, nu = params
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=nu * 2)
        phi = dadi.Integration.one_pop(phi, xx, T * 2, nu=2)
        return dadi.Spectrum.from_phi(phi, (int(ns),), (xx,))

    func_ex = dadi.Numerics.make_extrap_log_func(two_epoch)

    start = time.time()
    runs = []
    for _ in tqdm(range(n_runs), desc="dadi runs"):
        p0 = np.random.uniform(low=lower, high=upper)
        result = dadi.Inference.optimize(
            p0, sfs, func_ex, pts_l,
            lower_bound=lower, upper_bound=upper,
            epsilon=1.5e-8, verbose=0, full_output=True
        )
        popt, ll, n_it = result[0], result[1], result[4]
        runs.append((*popt, ll, n_it))

    runs = pd.DataFrame(runs, columns=["t", "Ne", "loss", "n_it"])
    best_idx = runs["loss"].idxmin()
    best_params = runs.iloc[best_idx, :2].values
    pg.logger.info(f"Inference parameters: {dict(zip(['t', 'Ne'], best_params))}")

    bootstraps = []
    for _ in tqdm(range(n_bootstraps), desc="dadi bootstraps"):
        resampled = dadi.Spectrum(fd.Spectrum(sfs.data).resample().data)
        result = dadi.Inference.optimize(
            best_params, resampled, func_ex, pts_l,
            lower_bound=lower, upper_bound=upper,
            epsilon=1.5e-8, verbose=0, full_output=True
        )
        popt, ll, n_it = result[0], result[1], result[4]
        bootstraps.append((*popt, ll, n_it))

    bootstraps = pd.DataFrame(bootstraps, columns=["t", "Ne", "loss", "n_it"])
    pg.logger.info(
        f"Bootstrap parameters: "
        f"mean: {bootstraps[['t', 'Ne']].mean().to_dict()}, "
        f"std: {bootstraps[['t', 'Ne']].std().to_dict()}"
    )

    return runs, bootstraps, time.time() - start


def run_phasegen(sfs: pg.SFS) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fit 2-epoch model using PhaseGen.
    """
    pg.Backend.register(pg.SciPyExpmBackend())

    inf = pg.Inference(
        coal=lambda t, Ne: pg.Coalescent(
            n=sfs.n,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})
        ),
        observation=sfs,
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic,
            modelled=(
                    coal.sfs.mean.polymorphic /
                    (coal.sfs.mean.theta * coal.sfs.mean.n_sites) *
                    (obs.theta * obs.n_sites)
            )
        ),
        bounds=bounds,
        resample=lambda s, _: s.resample(),
        do_bootstrap=True,
        parallelize=False,
        n_runs=n_runs,
        n_bootstraps=n_bootstraps
    )

    start = time.time()
    inf.run()
    total_runtime = time.time() - start

    runs = inf.runs.copy()
    bootstraps = inf.bootstraps.copy()

    runs["n_it"] = runs['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))
    bootstraps["n_it"] = bootstraps['result'].apply(lambda s: int(re.search(r'nfev:\s(\d+)', s).group(1)))

    return runs, bootstraps, total_runtime


results = []

for n in sample_sizes:
    pg.logger.info(f"Sample size = {n}")
    sfs_phasegen = simulate_bottleneck_sfs(n)

    # dadi
    runs_dadi, bs_dadi, time_dadi = run_dadi(dadi.Spectrum(sfs_phasegen.data))
    runs_pg, bs_pg, time_pg = run_phasegen(sfs_phasegen)

    best_run_dadi = runs_dadi.iloc[runs_dadi.loss.argmin()]
    best_run_pg = runs_pg.iloc[runs_pg.loss.argmin()]

    results.append({
        "n": n,
        "dadi.runtime": time_dadi,
        "phasegen.runtime": time_pg,
        "dadi.n_it": runs_dadi["n_it"].mean(),
        "phasegen.n_it": runs_pg["n_it"].mean(),
        "dadi.loss": best_run_dadi.loss,
        "phasegen.loss": best_run_pg.loss,
        "dadi.t": best_run_dadi.t,
        "phasegen.t": best_run_pg.t,
        "dadi.Ne": best_run_dadi.Ne,
        "phasegen.Ne": best_run_pg.Ne,
    })

results = pd.DataFrame(results)
results.to_csv("scratch/inference_comparison.csv", index=False)

results = pd.read_csv("scratch/inference_comparison.csv")

metric = "runtime"
methods = ["dadi", "phasegen"]

data = results[["n"] + [f"{m}.{metric}" for m in methods]].set_index("n")
data.columns = methods

plt.figure(figsize=(4, 4))
sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", cbar=False)
plt.title("Runtime in seconds")
plt.xlabel("Method")
plt.ylabel("Sample size")
plt.gca().set_yticklabels(plt.gca().get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("scratch/inference_runtime_comparison.png", dpi=400)
plt.show()

pass
