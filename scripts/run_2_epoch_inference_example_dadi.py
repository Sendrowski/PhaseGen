"""
Perform the 2-epoch inference example in the manuscript using dadi.
"""
import time
import warnings

import dadi
import fastdfe as fd
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Data: observed SFS (1D)
sfs = dadi.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
ns = sfs.sample_sizes[0]
pts_l = np.array([ns + 20, ns + 30, ns + 40])

n_runs = 20
n_bootstraps = 100

# bounds
lower = [0, 0.1]
upper = [4, 10]


def two_epoch(params: list, ns: int, pts: int) -> dadi.Spectrum:
    """
    Two epoch model for a single population.

    :param params: (T, nu): time of population size change, and population size in past
    :param ns: Sample size
    :param pts: Number of grid points
    :return: Resulting SFS
    """
    T, nu = params

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx, nu=nu * 2)
    phi = dadi.Integration.one_pop(phi, xx, T * 2, nu=2)

    return dadi.Spectrum.from_phi(phi, (int(ns),), (xx,))


# extrapolate function to larger grids
func_ex = dadi.Numerics.make_extrap_log_func(two_epoch)

# main optimization runs
runs = []
for _ in tqdm(range(n_runs), desc="Optimizing"):
    p0 = np.random.uniform(low=lower, high=upper)

    start = time.time()
    result = dadi.Inference.optimize(
        p0, sfs, func_ex, pts_l, lower_bound=lower, epsilon=1.5e-8,
        upper_bound=upper, verbose=0, full_output=True
    )
    runtime = time.time() - start

    popt, ll_model, n_it = result[0], result[1], result[4]
    model = func_ex(popt, ns, pts_l)

    runs.append((*popt, ll_model, n_it, runtime))

# convert to DataFrame
runs = pd.DataFrame(
    runs,
    columns=["T", "nu", "ll", "n_it", "runtime"]
)

# identify model with best likelihood
best_idx = runs["ll"].idxmin()
best_params = runs.iloc[best_idx, :2].values
best_model = func_ex(best_params, ns, pts_l)

theta = dadi.Inference.optimal_sfs_scaling(best_model, sfs)

# plot modelled vs observed SFS
fd.Spectra(dict(
    observed=sfs,
    modelled=func_ex(best_params, ns, pts_l) * theta
)).plot()

# bootstrap optimization
bootstraps = []
for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
    resampled = dadi.Spectrum(fd.Spectrum(sfs.data).resample().data)

    start = time.time()
    result = dadi.Inference.optimize(
        best_params, resampled, func_ex, pts_l, epsilon=1.5e-8,
        lower_bound=lower, upper_bound=upper, verbose=0, full_output=True
    )
    runtime = time.time() - start

    popt, ll_model, n_it = result[0], result[1], result[4]
    model = func_ex(popt, ns, pts_l)

    bootstraps.append((*popt, ll_model, n_it, runtime))

# convert to DataFrame
bootstraps = pd.DataFrame(
    bootstraps,
    columns=["T", "nu", "ll", "n_it", "runtime"]
)

print(runs.mean())
print(bootstraps.mean())

pass
