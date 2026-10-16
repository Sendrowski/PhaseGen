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
(single-deme-sfs)=
## Single-deme SFS
{class}`~phasegen.inference.Inference` requires a parametrized coalescent distribution, a loss function, and parameter bounds to be specified. More specifically, the `coal` argument is a callable that returns a {class}`~phasegen.distributions.Coalescent` object, based on the parameter values of the current optimization step, and `loss` is a callable specifying the current loss given this coalescent and the observation. By default, 10 independent optimization runs are performed using the L-BFGS-B algorithm, and the best result is returned.

We first simulate an observation. A demography with recent exponential growth that follows a bottleneck is sampled a million times with the empirical coalescent, and each sample carries a Poisson-distributed number of mutations at a population-scaled mutation rate ``theta`` per unit of branch length. The resulting SFS has a sample size of 20.

```{code-cell} python
:tags: [full-width]
import numpy as np
import matplotlib.pyplot as plt
import phasegen as pg

theta = 1
n_samples = 1_000_000

# exponential growth since a bottleneck, preceded by a larger ancestral population
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

# SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples = pg.Coalescent(n=20, demography=true).to_empirical(n_samples=n_samples, seed=42).sfs.samples
counts = np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0)
observation = pg.SFS(counts)

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
true.plot_pop_sizes(t=np.linspace(0, 1.5, 301), ax=axes[0], show=False)
observation.plot(ax=axes[1], show=False);
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
library(phasegen)
library(patchwork)
pg <- load_phasegen()
np <- reticulate::import("numpy", convert = FALSE)

theta <- 1
n_samples <- 1000000L

# exponential growth since a bottleneck, preceded by a larger ancestral population
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

# SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples <- pg$Coalescent(n = 20L, demography = true)$to_empirical(n_samples = n_samples, seed = 42L)$sfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$SFS(reticulate::py_to_r(counts))

plot(true, which = "pop_sizes", t = seq(0, 1.5, length.out = 301)) + plot(observation)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
We approximate this demography by five epochs of constant population size. The durations ``d0`` to ``d3`` of the first four epochs and the population sizes ``Ne0`` to ``Ne4`` of all five epochs are inferred, so that the epochs remain ordered for any parameter values. As the mutation rate is known, the loss is the Poisson likelihood of the observed polymorphic counts given their expected values under the model.

```{code-cell} python
bounds = {f'd{i}': (0.01, 1) for i in range(4)} | {f'Ne{i}': (0.05, 5) for i in range(5)}

def coal(d0, d1, d2, d3, Ne0, Ne1, Ne2, Ne3, Ne4):
    # start times of the five epochs
    times = np.cumsum([0, d0, d1, d2, d3])
    return pg.Coalescent(
        n=20,
        demography=pg.Demography(
            pop_sizes={'pop_0': dict(zip(times, [Ne0, Ne1, Ne2, Ne3, Ne4]))}
        )
    )

def loss(coal, obs):
    # expected counts are the mean branch lengths times the total mutation rate
    return pg.PoissonLikelihood().compute(
        observed=obs.polymorphic,
        modelled=n_samples * theta * coal.sfs.mean.polymorphic
    )

inf = pg.Inference(
    bounds=bounds,
    coal=coal,
    loss=loss,
    observation=observation,
    parallelize=True,
    seed=42
)
```

```{code-cell} r
bounds <- c(
    setNames(rep(list(c(0.01, 1)), 4), paste0("d", 0:3)),
    setNames(rep(list(c(0.05, 5)), 5), paste0("Ne", 0:4))
)

coal <- function(d0, d1, d2, d3, Ne0, Ne1, Ne2, Ne3, Ne4) {
    # start times of the epochs after the first
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

# expected counts are the mean branch lengths times the total mutation rate
loss <- function(coal, obs) pg$PoissonLikelihood()$compute(
    observed = obs$polymorphic,
    modelled = n_samples * theta * coal$sfs$mean$polymorphic
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
fitted = n_samples * theta * inf.dist_inferred.sfs.mean.polymorphic
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
fitted <- n_samples * theta * inf$dist_inferred$sfs$mean$polymorphic
stopifnot(max(abs(fitted / observation$polymorphic - 1)) < 0.05)
```

+++
### Bootstrapping
We may also wish to perform parametric bootstrapping. In order to do this, we provide a callback function that resamples the observation passed to {class}`~phasegen.inference.Inference`, here by drawing each bin from a Poisson distribution, and enable ``do_bootstrap``.

```{code-cell} python
inf = pg.Inference(
    bounds=bounds,
    coal=coal,
    loss=loss,
    observation=observation,
    resample=lambda sfs, rng: sfs.resample(seed=rng),  # Poisson draw of each bin
    do_bootstrap=True,
    n_bootstraps=20,
    parallelize=True,
    seed=42
)
```

```{code-cell} r
inf <- pg$Inference(
    bounds = bounds,
    coal = coal,
    loss = loss,
    observation = observation,
    resample = function(sfs, rng) sfs$resample(seed = rng),  # Poisson draw of each bin
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
stopifnot(nrow(inf$bootstraps) == 20)
```
