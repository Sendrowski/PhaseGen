```{code-cell} python
:tags: [remove-cell]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phasegen as pg

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
theta = 1
```

```{code-cell} r
:tags: [remove-cell]
library(phasegen)
library(ggplot2)
library(patchwork)
pg <- load_phasegen()
np <- reticulate::import("numpy", convert = FALSE)
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
theta <- 1
```

+++
(joint-sfs)=
## Two-deme joint SFS
The joint SFS of several demes carries information on their population sizes and on the migration between them. We sample a joint SFS of two demes ``A`` and ``B``, with four lineages from each, in the same way as the SFS of one deme in {ref}`single-deme-sfs`, here from 100,000 samples, with a population-scaled mutation rate ``theta`` per sample. Deme ``A`` declines exponentially into the past before two size changes, deme ``B`` passes through a bottleneck, and lineages migrate in both directions.

```{code-cell} python
n_samples = 100_000

# A declines exponentially into the past, B passes through a bottleneck
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
```

```{code-cell} python
:tags: [full-width]
# joint SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples = pg.Coalescent(n={'A': 4, 'B': 4}, demography=true).to_empirical(n_samples=n_samples, seed=42).jsfs.samples
counts = np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0)
observation = pg.JointSFS(counts.astype(float), pop_names=['A', 'B'])

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
true.plot_pop_sizes(t=np.linspace(0, 2, 401), ax=axes[0], show=False)
observation.plot(ax=axes[1], show=False);
```

```{code-cell} r
n_samples <- 100000L

# A declines exponentially into the past, B passes through a bottleneck
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
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
# joint SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples <- pg$Coalescent(n = list(A = 4L, B = 4L), demography = true)$to_empirical(n_samples = n_samples, seed = 42L)$jsfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$JointSFS(np$asarray(counts, dtype = "float64"), pop_names = c("A", "B"))

plot(true, which = "pop_sizes", t = seq(0, 2, length.out = 401)) + plot(observation)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
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
    # all bins except the two monomorphic corners
    return pg.PoissonLikelihood().compute(
        observed=obs.data.ravel()[1:-1],
        modelled=n_samples * theta * coal.jsfs.mean.data.ravel()[1:-1]
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
    # all bins except the two monomorphic corners
    k <- as.numeric(obs$data)
    m <- as.numeric(coal$jsfs$mean$data)
    pg$PoissonLikelihood()$compute(
        observed = k[-c(1, length(k))],
        modelled = n_samples * theta * m[-c(1, length(m))]
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
The log of the run confirms that the gradient is exact (see {ref}`exact-gradients`).

```{code-cell} python
inf.run()
```

```{code-cell} python
:tags: [remove-cell]
assert inf.loss_inferred <= loss(coal(**{name: 1 for name in names}), observation)
```

```{code-cell} r
inf$run()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$loss_inferred <= loss(do.call(coal, setNames(as.list(rep(1, length(names))), names)), observation))
```

+++
The inferred population size trajectories approximate the true ones, drawn dashed, in both demes. The shaded bands are approximate 95% confidence intervals from the Fisher information, which {meth}`Inference.plot_pop_sizes() <phasegen.inference.Inference.plot_pop_sizes>` draws with ``interval='fisher'`` and {meth}`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` returns as a table. The Fisher information is the curvature of the loss at the estimate, and its inverse approximates the covariance of the parameters, at a cost of two exact gradients per parameter (see {ref}`exact-gradients`). The intervals describe the uncertainty locally only. They do not account for other optima, and they understate the uncertainty of weakly identified parameters, such as the ancient population sizes, for which the parametric bootstrap below is more reliable.

```{code-cell} python
t = np.linspace(0, 2, 401)
ax = inf.plot_pop_sizes(t=t, interval='fisher', show=False)

# the true trajectories, dashed in the colour of each deme
for line, pop in zip(list(ax.lines), ['A', 'B']):
    ax.plot(t, [true.get_epoch(u).pop_sizes[pop] for u in t], drawstyle='steps-post', linestyle='--',
            color=line.get_color(), label=f'{pop} (true)')

ax.legend()
ax.set_ylim(0, 5);
```

```{code-cell} r
t <- seq(0, 2, length.out = 401)

# the true trajectories, dashed in the colour of each deme
truth <- do.call(rbind, lapply(c("A", "B"), function(pop) data.frame(
    x = t, y = sapply(t, function(u) true$get_epoch(u)$pop_sizes[[pop]]), series = pop
)))

plot(inf, which = "pop_sizes", t = t, interval = "fisher") +
    geom_step(data = truth, aes(x, y, colour = series, group = series, linetype = "true"), inherit.aes = FALSE) +
    scale_linetype_manual(values = c(true = "dashed"), name = NULL) +
    coord_cartesian(ylim = c(0, 5))
```

+++
The observed and the fitted joint SFS side by side, with their relative difference in each bin.

```{code-cell} python
:tags: [full-width]
from matplotlib.colors import LogNorm

observed = observation.data
fitted = n_samples * theta * inf.dist_inferred.jsfs.mean.data
# absolute difference relative to the mean of both spectra
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
fitted <- n_samples * theta * inf$dist_inferred$jsfs$mean$data
# absolute difference relative to the mean of both spectra
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
### Bootstrapping
As for the SFS of one deme in {ref}`single-deme-sfs`, a parametric bootstrap resamples each bin of the observed joint SFS from a Poisson distribution and reruns the inference from the estimate. Its spread can be compared with the standard errors from the Fisher information, which assume independent bins, as for unlinked loci and for the observation simulated here. The Godambe standard errors of {meth}`Inference.standard_errors() <phasegen.inference.Inference.standard_errors>` with ``method='godambe'`` evaluate the score of the loss at the estimate on observations drawn with the same callback, without refitting, and remain valid for linked sites when the callback draws a block bootstrap over the genome.

```{code-cell} python
:tags: [remove-output]
# Poisson draw of each bin
inf.resample = lambda obs, rng: pg.JointSFS(rng.poisson(obs.data).astype(float), pop_names=obs.pop_names)
inf.n_bootstraps = 20
inf.parallelize = True

inf.bootstrap()
```

```{code-cell} python
import pandas as pd

pd.DataFrame({
    'estimate': inf.params_inferred,
    'standard error (Fisher)': inf.standard_errors(),
    'standard error (Godambe)': inf.standard_errors(method='godambe'),
    'standard deviation (bootstrap)': inf.bootstraps[names].std()
}).round(3)
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
:tags: [remove-output]
# Poisson draw of each bin
inf$resample <- function(obs, rng) pg$JointSFS(rng$poisson(obs$data) * 1, pop_names = obs$pop_names)
inf$n_bootstraps <- 20L

inf$bootstrap()
```

```{code-cell} r
boots <- as.matrix(inf$bootstraps[names])
se <- inf$standard_errors()
se_godambe <- inf$standard_errors(method = "godambe")

round(data.frame(
    estimate = unlist(inf$params_inferred[names]),
    `standard error (Fisher)` = se[names],
    `standard error (Godambe)` = se_godambe[names],
    `standard deviation (bootstrap)` = apply(boots, 2, sd),
    check.names = FALSE
), 3)
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
## Three-deme joint SFS
The joint SFS of three demes ``A``, ``B`` and ``C`` is fitted in the same way, with two lineages sampled from each. Deme ``A`` declines exponentially into the past before two size changes, ``B`` passes through a bottleneck, ``C`` changes size once, and lineages migrate between every pair of demes.

```{code-cell} python
pops = ['A', 'B', 'C']
pairs = [(a, b) for a in pops for b in pops if a != b]

true = pg.Demography(
    pop_sizes={'A': {0.4: 0.6, 1.0: 1.2}, 'B': {0: 0.8, 0.3: 0.15, 0.5: 1}, 'C': {0: 1.5, 0.8: 0.7}},
    migration_rates={('A', 'B'): 0.3, ('B', 'A'): 0.6, ('A', 'C'): 0.1,
                     ('C', 'A'): 0.2, ('B', 'C'): 0.4, ('C', 'B'): 0.15},
    events=[pg.ExponentialPopSizeChanges(
        initial_size={'A': 3},
        growth_rate={'A': 4},
        start_time={'A': 0},
        end_time={'A': 0.4},
        step_size=0.05
    )]
)
```

```{code-cell} r
pops <- c("A", "B", "C")
pairs <- Filter(function(pair) pair[1] != pair[2],
                unlist(lapply(pops, function(a) lapply(pops, function(b) c(a, b))), recursive = FALSE))

true <- pg$Demography(
    pop_sizes = list(B = 0.8, C = 1.5),
    events = c(
        list(pg$ExponentialPopSizeChanges(
            initial_size = list(A = 3),
            growth_rate = list(A = 4),
            start_time = list(A = 0),
            end_time = list(A = 0.4),
            step_size = 0.05
        )),
        unname(Map(function(pop, time, size) pg$PopSizeChange(pop = pop, time = time, size = size),
                   c("A", "A", "B", "B", "C"), c(0.4, 1, 0.3, 0.5, 0.8), c(0.6, 1.2, 0.15, 1, 0.7))),
        unname(Map(function(source, dest, rate) pg$MigrationRateChange(source = source, dest = dest, time = 0, rate = rate),
                   c("A", "B", "A", "C", "B", "C"), c("B", "A", "C", "A", "C", "B"), c(0.3, 0.6, 0.1, 0.2, 0.4, 0.15)))
    )
)
```

+++
We draw 100,000 samples of the joint SFS as before, and show it marginalized onto each pair of demes next to the true population sizes.

```{code-cell} python
:tags: [full-width]
# joint SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples = pg.Coalescent(n=dict.fromkeys(pops, 2), demography=true).to_empirical(n_samples=n_samples, seed=42).jsfs.samples
counts = np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0)
observation = pg.JointSFS(counts.astype(float), pop_names=pops)

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
true.plot_pop_sizes(t=np.linspace(0, 2, 401), ax=axes[0, 0], show=False)

# the joint SFS of each pair of demes, summed over the third
for ax, pair in zip(axes.ravel()[1:], [(0, 1), (0, 2), (1, 2)]):
    observation.plot(pops=pair, ax=ax, show=False)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 5.5)
```

```{code-cell} r
:tags: [full-width]
# joint SFS branch lengths of each sample, then Poisson mutation counts summed over samples
n <- setNames(as.list(rep(2L, 3)), pops)
samples <- pg$Coalescent(n = n, demography = true)$to_empirical(n_samples = n_samples, seed = 42L)$jsfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$JointSFS(np$asarray(counts, dtype = "float64"), pop_names = pops)

# the joint SFS of each pair of demes, summed over the third
plot(true, which = "pop_sizes", t = seq(0, 2, length.out = 401)) +
    plot(observation, pops = c(0, 1)) +
    plot(observation, pops = c(0, 2)) +
    plot(observation, pops = c(1, 2)) +
    plot_layout(ncol = 2)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
We fit the population sizes of each deme in three epochs and the migration rates between every ordered pair of demes, fifteen parameters in all, with the loss of the two-deme example.

```{code-cell} python
times = [0, 0.2, 0.6]

def coal(**params):
    return pg.Coalescent(
        n=dict.fromkeys(pops, 2),
        demography=pg.Demography(
            pop_sizes={p: dict(zip(times, [params[f'{p}{i}'] for i in range(3)])) for p in pops},
            migration_rates={(a, b): params[f'm_{a}{b}'] for a, b in pairs}
        )
    )

names = [f'{p}{i}' for p in pops for i in range(3)] + [f'm_{a}{b}' for a, b in pairs]
```

```{code-cell} r
times <- c(0, 0.2, 0.6)

coal <- function(...) {
    params <- list(...)
    pg$Coalescent(
        n = n,
        demography = pg$Demography(
            pop_sizes = setNames(lapply(pops, function(p) params[[paste0(p, 0)]]), pops),
            events = c(
                unlist(lapply(pops, function(p) lapply(2:3, function(i) {
                    pg$PopSizeChange(pop = p, time = times[i], size = params[[paste0(p, i - 1)]])
                })), recursive = FALSE),
                lapply(pairs, function(pair) pg$MigrationRateChange(
                    source = pair[1], dest = pair[2], time = 0, rate = params[[paste0("m_", pair[1], pair[2])]]
                ))
            )
        )
    )
}

names <- c(paste0(rep(pops, each = 3), 0:2), sapply(pairs, function(pair) paste0("m_", pair[1], pair[2])))
```

+++
Population sizes and migration rates often span orders of magnitude, and optimizing them as their logarithms with ``log_scale=True`` reduces the number of iterations considerably, while the estimates stay on the original scale.

```{code-cell} python
inf = pg.Inference(
    bounds={name: (0.05, 5) for name in names},
    x0={name: 1 for name in names},
    coal=coal,
    loss=loss,
    observation=observation,
    log_scale=True,
    n_runs=1,
    seed=42
)

inf.run()
```

```{code-cell} python
:tags: [remove-cell]
assert inf.loss_inferred <= loss(coal(**{name: 1 for name in names}), observation)
```

```{code-cell} r
inf <- pg$Inference(
    bounds = setNames(rep(list(c(0.05, 5)), length(names)), names),
    x0 = setNames(as.list(rep(1, length(names))), names),
    coal = coal,
    loss = loss,
    observation = observation,
    log_scale = TRUE,
    n_runs = 1L,
    seed = 42L
)

inf$run()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$loss_inferred <= loss(do.call(coal, setNames(as.list(rep(1, length(names))), names)), observation))
```

+++
The inferred population size trajectories of the three demes.

```{code-cell} python
:tags: [full-width]
fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True)
t = np.linspace(0, 2, 401)
epoch = np.searchsorted(times, t, side='right') - 1

for ax, pop in zip(axes, pops):
    ax.plot(t, [true.get_epoch(u).pop_sizes[pop] for u in t], drawstyle='steps-post', label='true')
    ax.plot(t, [inf.params_inferred[f'{pop}{i}'] for i in epoch], drawstyle='steps-post', label='inferred')
    ax.set_title(pop)
    ax.set_xlabel('t')

axes[0].set_ylabel('population size')
axes[0].legend();
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 2.6)
```

```{code-cell} r
:tags: [full-width]
t <- seq(0, 2, length.out = 401)
epoch <- findInterval(t, times)
sizes <- do.call(rbind, lapply(pops, function(pop) rbind(
    data.frame(pop = pop, t = t, label = "true",
               size = sapply(t, function(u) true$get_epoch(u)$pop_sizes[[pop]])),
    data.frame(pop = pop, t = t, label = "inferred",
               size = unlist(inf$params_inferred[paste0(pop, epoch - 1)]))
)))
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
