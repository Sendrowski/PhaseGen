# Parameter inference
The availability of exact moments lends itself to gradient-based parameter estimation. This is commonly done based on the SFS, but higher-order moments can also be used, provided they can be computed from the data at hand. ``phasegen`` provides a lightweight framework for performing parameter inference which is done by defining an {class}`~phasegen.inference.Inference` object.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
## Running an inference
{class}`~phasegen.inference.Inference` requires a parametrized coalescent distribution, a loss function, and parameter bounds to be specified. More specifically, the `coal` argument is a callable that returns a {class}`~phasegen.distributions.Coalescent` object, based on the parameter values of the current optimization step, and `loss` is a callable specifying the current loss given this coalescent and the observation. By default, 10 independent optimization runs are performed using the L-BFGS-B algorithm, and the best result is returned.

We first simulate an observation. A demography with recent exponential growth that follows a bottleneck is sampled with the empirical coalescent at a million independent loci, each of which carries a Poisson-distributed number of mutations at a population-scaled mutation rate ``theta`` per unit of branch length. The resulting SFS has a sample size of 20.

```{code-cell} python
import numpy as np
import phasegen as pg

theta = 1
n_loci = 1_000_000

true = pg.Demography(
    pop_sizes={'pop_0': {0.2: 0.5, 0.6: 1.5}},
    events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 3},
        growth_rate={'pop_0': 9},
        start_time={'pop_0': 0},
        end_time={'pop_0': 0.2},
        step_size=0.0125
    )]
)

samples = pg.Coalescent(n=20, demography=true).to_empirical(n_samples=n_loci, seed=42).sfs.samples
counts = np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0)
observation = pg.SFS(counts)

observation.plot();
```

```{code-cell} r
library(phasegen)
pg <- load_phasegen()
np <- reticulate::import("numpy", convert = FALSE)

theta <- 1
n_loci <- 1000000L

true <- pg$Demography(
    events = list(
        pg$ExponentialPopSizeChanges(
            initial_size = list(pop_0 = 3),
            growth_rate = list(pop_0 = 9),
            start_time = list(pop_0 = 0),
            end_time = list(pop_0 = 0.2),
            step_size = 0.0125
        ),
        pg$PopSizeChange(pop = "pop_0", time = 0.2, size = 0.5),
        pg$PopSizeChange(pop = "pop_0", time = 0.6, size = 1.5)
    )
)

samples <- pg$Coalescent(n = 20L, demography = true)$to_empirical(n_samples = n_loci, seed = 42L)$sfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$SFS(reticulate::py_to_r(counts))

plot(observation)
```

+++
We approximate this demography by five epochs of constant population size. The durations ``d0`` to ``d3`` of the first four epochs and the population sizes ``Ne0`` to ``Ne4`` of all five epochs are inferred, so that the epochs remain ordered for any parameter values. As the mutation rate is known, the loss is the Poisson likelihood of the observed polymorphic counts given their expected values under the model.

```{code-cell} python
bounds = {f'd{i}': (0.01, 1) for i in range(4)} | {f'Ne{i}': (0.05, 5) for i in range(5)}

def coal(d0, d1, d2, d3, Ne0, Ne1, Ne2, Ne3, Ne4):
    times = np.cumsum([0, d0, d1, d2, d3])
    return pg.Coalescent(
        n=20,
        demography=pg.Demography(
            pop_sizes={'pop_0': dict(zip(times, [Ne0, Ne1, Ne2, Ne3, Ne4]))}
        )
    )

def loss(coal, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.polymorphic,
        modelled=n_loci * theta * coal.sfs.mean.polymorphic
    )

inf = pg.Inference(
    bounds=bounds,
    coal=coal,
    loss=loss,
    observation=observation,
    seed=42
)
```

```{code-cell} r
bounds <- c(
    setNames(rep(list(c(0.01, 1)), 4), paste0("d", 0:3)),
    setNames(rep(list(c(0.05, 5)), 5), paste0("Ne", 0:4))
)

coal <- function(d0, d1, d2, d3, Ne0, Ne1, Ne2, Ne3, Ne4) {
    times <- cumsum(c(d0, d1, d2, d3))
    sizes <- c(Ne1, Ne2, Ne3, Ne4)
    pg$Coalescent(
        n = 20L,
        demography = pg$Demography(
            pop_sizes = list(pop_0 = Ne0),
            events = lapply(1:4, function(i) pg$PopSizeChange(pop = "pop_0", time = times[i], size = sizes[i]))
        )
    )
}

loss <- function(coal, obs) pg$PoissonLikelihood()$compute(
    observed = obs$polymorphic,
    modelled = n_loci * theta * coal$sfs$mean$polymorphic
)

inf <- pg$Inference(
    bounds = bounds,
    coal = coal,
    loss = loss,
    observation = observation,
    seed = 42L
)
```

+++
Upon construction, the inference object is ready to be optimized. We then compare the inferred population size trajectory with the true one.

```{code-cell} python
inf.run()
```

```{code-cell} python
import matplotlib.pyplot as plt

t = np.linspace(0, 1.5, 301)

for label, demography in [('true', true), ('inferred', inf.dist_inferred.demography)]:
    sizes = [demography.get_epoch(u).pop_sizes['pop_0'] for u in t]
    plt.plot(t, sizes, drawstyle='steps-post', label=label)

plt.xlabel('t')
plt.ylabel('population size')
plt.legend();
```

```{code-cell} python
:tags: [remove-cell]
fitted = n_loci * theta * inf.dist_inferred.sfs.mean.polymorphic
assert np.max(np.abs(fitted / observation.polymorphic - 1)) < 0.05
```

```{code-cell} r
inf$run()
```

```{code-cell} r
library(ggplot2)

t <- seq(0, 1.5, length.out = 301)
trajectory <- function(demography, label) data.frame(
    t = t,
    size = sapply(t, function(u) demography$get_epoch(u)$pop_sizes$pop_0),
    label = label
)
sizes <- rbind(trajectory(true, "true"), trajectory(inf$dist_inferred$demography, "inferred"))
sizes$label <- factor(sizes$label, levels = c("true", "inferred"))

ggplot(sizes, aes(t, size, colour = label)) +
    geom_step() +
    labs(y = "population size", colour = NULL)
```

```{code-cell} r
:tags: [remove-cell]
fitted <- n_loci * theta * inf$dist_inferred$sfs$mean$polymorphic
stopifnot(max(abs(fitted / observation$polymorphic - 1)) < 0.05)
```

+++
## Parametric bootstrapping
We may also wish to perform parametric bootstrapping. In order to do this, we provide a callback function that resamples the observation passed to {class}`~phasegen.inference.Inference`, here by drawing each bin from a Poisson distribution, and enable ``do_bootstrap``.

```{code-cell} python
inf = pg.Inference(
    bounds=bounds,
    coal=coal,
    loss=loss,
    observation=observation,
    resample=lambda sfs, rng: sfs.resample(seed=rng),
    do_bootstrap=True,
    n_bootstraps=20,
    seed=42
)
```

```{code-cell} r
inf <- pg$Inference(
    bounds = bounds,
    coal = coal,
    loss = loss,
    observation = observation,
    resample = function(sfs, rng) sfs$resample(seed = rng),
    do_bootstrap = TRUE,
    n_bootstraps = 20L,
    seed = 42L
)
```

+++
We run the inference again and visualize the results.

```{code-cell} python
inf.run()
```

```{code-cell} python
inf.plot_pop_sizes();
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams['figure.figsize'] = [7, 7]
```

```{code-cell} python
inf.plot_bootstraps();
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
assert len(inf.bootstraps) == 20
```

```{code-cell} r
inf$run()
```

```{code-cell} r
plot(inf, which = "pop_sizes")
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 7)
```

```{code-cell} r
plot(inf, which = "bootstraps")
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
stopifnot(nrow(inf$bootstraps) == 20)
```

+++
## Exact gradients for many parameters
By default, {class}`~phasegen.inference.Inference` differentiates the loss exactly with respect to the demographic parameters. The cost of this gradient does not grow with the number of parameters, whereas a finite-difference gradient needs one loss evaluation per parameter, which makes models with many parameters affordable. Losses for which no exact gradient is available are differentiated by finite differences, as noted in the log.

We sample a joint SFS of two demes ``A`` and ``B``, each sampled four times, in the same way as the SFS above, here at 100,000 loci. Deme ``A`` declines exponentially into the past before two size changes, deme ``B`` passes through a bottleneck, and lineages migrate in both directions.

```{code-cell} python
n_loci = 100_000

true = pg.Demography(
    pop_sizes={'A': {0.4: 0.6, 1.0: 1.2}, 'B': {0: 0.8, 0.3: 0.15, 0.5: 1}},
    migration_rates={('A', 'B'): 0.3, ('B', 'A'): 0.6},
    events=[pg.ExponentialPopSizeChanges(
        initial_size={'A': 3},
        growth_rate={'A': 4},
        start_time={'A': 0},
        end_time={'A': 0.4},
        step_size=0.05
    )]
)

samples = pg.Coalescent(n={'A': 4, 'B': 4}, demography=true).to_empirical(n_samples=n_loci, seed=42).jsfs.samples
counts = np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0)
observation = pg.JointSFS(counts.astype(float), pop_names=['A', 'B'])

observation.plot();
```

```{code-cell} r
n_loci <- 100000L

true <- pg$Demography(
    pop_sizes = list(B = 0.8),
    events = list(
        pg$ExponentialPopSizeChanges(
            initial_size = list(A = 3),
            growth_rate = list(A = 4),
            start_time = list(A = 0),
            end_time = list(A = 0.4),
            step_size = 0.05
        ),
        pg$PopSizeChange(pop = "A", time = 0.4, size = 0.6),
        pg$PopSizeChange(pop = "A", time = 1, size = 1.2),
        pg$PopSizeChange(pop = "B", time = 0.3, size = 0.15),
        pg$PopSizeChange(pop = "B", time = 0.5, size = 1),
        pg$MigrationRateChange(source = "A", dest = "B", time = 0, rate = 0.3),
        pg$MigrationRateChange(source = "B", dest = "A", time = 0, rate = 0.6)
    )
)

samples <- pg$Coalescent(n = list(A = 4L, B = 4L), demography = true)$to_empirical(n_samples = n_loci, seed = 42L)$jsfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$JointSFS(np$asarray(counts, dtype = "float64"), pop_names = c("A", "B"))

plot(observation)
```

+++
We approximate this demography by four epochs of constant population size in each deme and constant migration rates, which gives ten parameters: the population sizes ``A0`` to ``A3`` and ``B0`` to ``B3`` of each deme in the four epochs, and the migration rates ``m_AB`` and ``m_BA``. The loss is the Poisson likelihood of the polymorphic bins of the joint SFS.

```{code-cell} python
times = [0, 0.2, 0.6, 1.5]

def coal(A0, A1, A2, A3, B0, B1, B2, B3, m_AB, m_BA):
    return pg.Coalescent(
        n={'A': 4, 'B': 4},
        demography=pg.Demography(
            pop_sizes={
                'A': dict(zip(times, [A0, A1, A2, A3])),
                'B': dict(zip(times, [B0, B1, B2, B3]))
            },
            migration_rates={('A', 'B'): m_AB, ('B', 'A'): m_BA}
        )
    )

def loss(coal, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.data.ravel()[1:-1],
        modelled=n_loci * theta * coal.jsfs.mean.data.ravel()[1:-1]
    )

names = [f'{pop}{i}' for pop in 'AB' for i in range(4)] + ['m_AB', 'm_BA']

inf = pg.Inference(
    bounds={name: (0.05, 5) for name in names},
    x0={name: 1 for name in names},
    coal=coal,
    loss=loss,
    observation=observation,
    n_runs=1,
    seed=42
)
```

```{code-cell} r
times <- c(0, 0.2, 0.6, 1.5)

coal <- function(A0, A1, A2, A3, B0, B1, B2, B3, m_AB, m_BA) {
    A <- c(A0, A1, A2, A3)
    B <- c(B0, B1, B2, B3)
    pg$Coalescent(
        n = list(A = 4L, B = 4L),
        demography = pg$Demography(
            pop_sizes = list(A = A0, B = B0),
            events = c(
                lapply(2:4, function(i) pg$PopSizeChange(pop = "A", time = times[i], size = A[i])),
                lapply(2:4, function(i) pg$PopSizeChange(pop = "B", time = times[i], size = B[i])),
                pg$MigrationRateChange(source = "A", dest = "B", time = 0, rate = m_AB),
                pg$MigrationRateChange(source = "B", dest = "A", time = 0, rate = m_BA)
            )
        )
    )
}

loss <- function(coal, obs) {
    k <- as.numeric(obs$data)
    m <- as.numeric(coal$jsfs$mean$data)
    pg$PoissonLikelihood()$compute(
        observed = k[-c(1, length(k))],
        modelled = n_loci * theta * m[-c(1, length(m))]
    )
}

names <- c(paste0("A", 0:3), paste0("B", 0:3), "m_AB", "m_BA")

inf <- pg$Inference(
    bounds = setNames(rep(list(c(0.05, 5)), length(names)), names),
    x0 = setNames(as.list(rep(1, length(names))), names),
    coal = coal,
    loss = loss,
    observation = observation,
    n_runs = 1L,
    seed = 42L
)
```

+++
The log of the run confirms that the gradient is exact. The inferred population size trajectories approximate the true ones in both demes.

```{code-cell} python
inf.run()
```

```{code-cell} python
:tags: [remove-cell]
assert inf.loss_inferred <= loss(coal(**{name: 1 for name in names}), observation)
```

```{code-cell} python
:tags: [full-width]
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
t = np.linspace(0, 2, 401)

for ax, pop in zip(axes, 'AB'):
    for label, demography in [('true', true), ('inferred', inf.dist_inferred.demography)]:
        sizes = [demography.get_epoch(u).pop_sizes[pop] for u in t]
        ax.plot(t, sizes, drawstyle='steps-post', label=label)
    ax.set_title(pop)
    ax.set_xlabel('t')

axes[0].set_ylabel('population size')
axes[0].legend();
```

```{code-cell} r
inf$run()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$loss_inferred <= loss(do.call(coal, setNames(as.list(rep(1, length(names))), names)), observation))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
t <- seq(0, 2, length.out = 401)
trajectory <- function(demography, label) do.call(rbind, lapply(c("A", "B"), function(pop) data.frame(
    pop = pop,
    t = t,
    size = sapply(t, function(u) demography$get_epoch(u)$pop_sizes[[pop]]),
    label = label
)))
sizes <- rbind(trajectory(true, "true"), trajectory(inf$dist_inferred$demography, "inferred"))
sizes$label <- factor(sizes$label, levels = c("true", "inferred"))

ggplot(sizes, aes(t, size, colour = label)) +
    geom_step() +
    facet_wrap(~pop) +
    labs(y = "population size", colour = NULL)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
The observed and the fitted joint SFS side by side, with their relative difference in each bin.

```{code-cell} python
:tags: [full-width]
from matplotlib.colors import LogNorm

observed = observation.data
fitted = n_loci * theta * inf.dist_inferred.jsfs.mean.data
mean = (observed + fitted) / 2
diff = np.divide(np.abs(observed - fitted), mean, out=np.zeros_like(mean), where=mean > 0)

fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), layout='constrained')
norm = LogNorm(observed[observed > 0].min(), observed.max())

for ax, data, title in zip(axes, [observed, fitted], ['observed', 'fitted']):
    image = ax.imshow(np.ma.masked_equal(data, 0), origin='lower', norm=norm)
    ax.set_title(title)

diff_image = axes[2].imshow(diff, origin='lower', cmap='coolwarm', vmin=0, vmax=0.1)
axes[2].set_title('difference')
fig.colorbar(image, ax=axes[:2], shrink=0.7)
fig.colorbar(diff_image, ax=axes[2], shrink=0.7)

for ax in axes:
    ax.set_xlabel('allele count B')
axes[0].set_ylabel('allele count A');
```

```{code-cell} python
:tags: [remove-cell]
assert diff.max() < 0.15
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 2.6)
```

```{code-cell} r
:tags: [full-width]
observed <- observation$data
fitted <- n_loci * theta * inf$dist_inferred$jsfs$mean$data
diff <- abs(observed - fitted) / ((observed + fitted) / 2)
diff[observed == 0 & fitted == 0] <- 0

grid <- function(m) expand.grid(A = seq_len(nrow(m)) - 1, B = seq_len(ncol(m)) - 1)
counts <- rbind(
    cbind(grid(observed), count = as.numeric(observed), panel = "observed"),
    cbind(grid(fitted), count = as.numeric(fitted), panel = "fitted")
)
counts$panel <- factor(counts$panel, levels = c("observed", "fitted"))
counts$count[counts$count == 0] <- NA

spectra <- ggplot(counts, aes(B, A, fill = count)) +
    geom_tile() +
    facet_wrap(~panel) +
    scale_fill_viridis_c(trans = "log10", na.value = "white") +
    coord_equal() +
    labs(x = "allele count B", y = "allele count A")

differences <- ggplot(cbind(grid(diff), difference = as.numeric(diff)), aes(B, A, fill = difference)) +
    geom_tile() +
    scale_fill_distiller(palette = "RdBu", limits = c(0, 0.1), oob = scales::squish) +
    coord_equal() +
    labs(title = "difference", x = "allele count B", y = NULL)

library(patchwork)
spectra + differences + plot_layout(widths = c(2, 1))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(max(diff) < 0.15)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
Standard errors of the estimates follow from the curvature of the loss at the estimate, as the inverse of the observed Fisher information, which costs two exact gradients per parameter. They approximate the uncertainty locally only. They do not account for other optima, and they understate the uncertainty of weakly identified parameters, such as the ancient population sizes, for which the parametric bootstrap described above is more reliable.

```{code-cell} python
import pandas as pd

pd.DataFrame({'estimate': inf.params_inferred, 'standard error': inf.standard_errors()}).round(3)
```

```{code-cell} python
:tags: [remove-cell]
assert inf.standard_errors().notna().all()
```

```{code-cell} r
se <- inf$standard_errors()

round(data.frame(estimate = unlist(inf$params_inferred[names]), `standard error` = se[names], check.names = FALSE), 3)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(is.finite(se)))
```

+++
## Distributed bootstrapping
For inferences with long runtimes, the bootstrapping process can be distributed by creating bootstrap samples which are {class}`~phasegen.inference.Inference` objects themselves ({meth}`Inference.create_bootstrap() <phasegen.inference.Inference.create_bootstrap>`). These bootstraps can be run in parallel and the results combined afterwards ({meth}`Inference.add_bootstraps() <phasegen.inference.Inference.add_bootstraps>`).

+++ {"tags": ["python-only"]}
Below is an example [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow for distributed bootstrapping: 

``Snakefile``:

```python
rule all:
    input:
        "results/graphs/inference/demography.png",

# setup inference and perform initial run
rule setup_inference:
    output:
        "results/inference/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "setup_inference.py"

# create and run bootstrap
rule run_bootstrap:
    input:
        "results/inference/inference.json"
    output:
        "results/inference/bootstraps/{i}/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "run_bootstrap.py"

# merge bootstraps and visualize
rule merge_bootstraps:
    input:
        inference="results/inference/inference.json",
        bootstraps=expand("results/inference/bootstraps/{i}/inference.json",i=range(100))
    output:
        inference="results/inference/inference.bootstrapped.json",
        demography="results/graphs/inference/demography.png",
        bootstraps="results/graphs/inference/bootstraps.png",
    conda:
        "phasegen.yaml"
    script:
        "merge_bootstraps.py"
```

+++ {"tags": ["python-only"]}
where ``phasegen.yaml`` is a conda environment file with the following content:
```yaml
name: phasegen
channels:
  - defaults
dependencies:
  - python>=3.10,<3.14
  - pip
  - pip:
      - phasegen
```

+++ {"tags": ["python-only"]}
In ``setup_inference.py``, we set up the inference object, perform the initial run and save the results to a file. Note that bootstrapping is disabled here as we are doing it manually.

```python
import phasegen as pg
out = snakemake.output[0]

inf = pg.Inference(
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=pg.SFS(
        [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
    ),
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    resample=lambda sfs, rng: sfs.resample(seed=rng),
    do_bootstrap=False,
    seed=42
)

inf.run()

inf.to_file(out)
```

+++ {"tags": ["python-only"]}
In ``run_bootstrap.py``, we load the inference object from the file, create the bootstrap sample of this job and run the inference for it.

```python
import phasegen as pg

file = snakemake.input[0]
out = snakemake.output[0]

inf = pg.Inference.from_file(file)

bootstrap = inf.create_bootstrap(index=int(snakemake.wildcards.i))

bootstrap.run()

bootstrap.to_file(out)
```

+++ {"tags": ["python-only"]}
In ``merge_bootstraps.py``, we load the inference object and all bootstraps, merge the bootstraps with the main inference object and visualize the results.

```python
import phasegen as pg

inf_file = snakemake.input.inference
bootstraps_file = snakemake.input.bootstraps
out = snakemake.output.inference
out_demography = snakemake.output.demography
out_bootstraps = snakemake.output.bootstraps

inf = pg.Inference.from_file(inf_file)

inf.add_bootstraps([pg.Inference.from_file(f) for f in bootstraps_file])

inf.plot_demography(file=out_demography)
inf.plot_bootstraps(file=out_bootstraps)

inf.to_file(out)
```

+++ {"tags": ["r-only"]}
In a production setting the bootstraps are typically distributed across a cluster, for instance with a [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow. In such a workflow, one rule sets up the inference and performs the initial run ({meth}`Inference.to_file() <phasegen.inference.Inference.to_file>`), a second rule loads it ({meth}`Inference.from_file() <phasegen.inference.Inference.from_file>`), creates a single bootstrap and runs it, and a final rule merges the bootstrap results back into the main inference object. Below we demonstrate the same API in a single notebook, running the bootstraps in a serial loop rather than as parallel cluster jobs.

From R, the `coal`, `loss` and `resample` callbacks must be Python functions, defined via `reticulate::py_run_string`, because `create_bootstrap` and `to_file` serialize them and R closures do not survive that round-trip.

```{code-cell} r
# the model, loss and resampling callbacks, defined in Python so they survive serialization
py <- reticulate::py_run_string('
import phasegen as pg

def coal(t, Ne):
    return pg.Coalescent(
        n=10,
        demography=pg.Demography(pop_sizes={"pop_0": {0: 1, t: Ne}}),
    )

def loss(coal, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic,
    )

def resample(sfs, rng):
    return sfs.resample(seed=rng)
')
```

+++ {"tags": ["r-only"]}
We set up the inference object, perform the initial run and save the result to a file. Bootstrapping is disabled here as we perform it manually.

```{code-cell} r
inf <- pg$Inference(
    bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
    observation = pg$SFS(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)),
    coal = py$coal,
    loss = py$loss,
    resample = py$resample,
    do_bootstrap = FALSE,
    seed = 42L
)

inf$run()

path <- file.path(tempdir(), "inference.json")
inf$to_file(path)
```

+++ {"tags": ["r-only"]}
Each bootstrap is created from the inference object with {meth}`Inference.create_bootstrap() <phasegen.inference.Inference.create_bootstrap>`, which resamples the observation using the provided `resample` callback, and is then run independently. On a cluster each iteration of this loop would instead be a separate job that loads ``inference.json``, creates the bootstrap of its index, runs it and writes its own file. Here we reload the saved object once and run the replicates serially.

```{code-cell} r
:tags: [remove-output]
inf <- pg$Inference$from_file(path)

boots <- lapply(seq_len(20L), function(i) {
    b <- inf$create_bootstrap(index = i)
    b$run()
    b
})
```

+++ {"tags": ["r-only"]}
Finally we merge the bootstraps into the main inference object with {meth}`Inference.add_bootstraps() <phasegen.inference.Inference.add_bootstraps>` and visualize the inferred demography and the bootstrap distribution of the inferred parameters.

```{code-cell} r
inf$add_bootstraps(boots)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(nrow(inf$bootstraps) == 20)
```

```{code-cell} r
plot(inf)
```

```{code-cell} r
plot(inf, which = "bootstraps")
```
