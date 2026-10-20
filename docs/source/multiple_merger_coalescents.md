# Coalescent models
Supported coalescent models are {class}`~phasegen.coalescent_models.StandardCoalescent`, {class}`~phasegen.coalescent_models.BetaCoalescent` and {class}`~phasegen.coalescent_models.DiracCoalescent`, and can be specified when constructing the {class}`~phasegen.distributions.Coalescent` distribution.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

subplot_defaults = {k: matplotlib.rcParams[k] for k in ('figure.subplot.left', 'figure.subplot.right', 'figure.subplot.wspace')}
matplotlib.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.right': 1, 'figure.subplot.wspace': 0})
```

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt
import phasegen as pg

# compare 2-SFS of Kingman and Beta coalescents
kingman = pg.Coalescent(n=10, model=pg.StandardCoalescent())
beta = pg.Coalescent(n=10, model=pg.BetaCoalescent(alpha=1.5))

_, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={"projection": "3d"})

# beta coalescent shows positive correlations between disparate minor allele frequencies
kingman.sfs.cov.plot_surface(ax=axs[0], show=False, title='Kingman')
beta.sfs.cov.plot_surface(ax=axs[1], title='Beta');
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams.update(subplot_defaults)
```

```{code-cell} python
:tags: [remove-cell]
assert beta.sfs.cov.data[1, 8] > 0 > kingman.sfs.cov.data[1, 8]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
:tags: [full-width]
library(phasegen)
pg <- load_phasegen()

# compare 2-SFS of Kingman and Beta coalescents
kingman <- pg$Coalescent(n = 10L, model = pg$StandardCoalescent())
beta <- pg$Coalescent(n = 10L, model = pg$BetaCoalescent(alpha = 1.5))

par(mfrow = c(1, 2))
# beta coalescent shows positive correlations between disparate minor allele frequencies
persp(kingman$sfs$cov, title = "Kingman")
persp(beta$sfs$cov, title = "Beta")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(beta$sfs$cov$data[2, 9] > 0, kingman$sfs$cov$data[2, 9] < 0)
```

+++
## Inferring strength of multiple mergers
The parameters of a multiple-merger coalescent can be inferred together with the demography, with exact gradients (see {ref}`exact-gradients`). Below we draw 100,000 samples of the SFS of 20 lineages under a Beta coalescent with a population size change, add Poisson mutations with a population-scaled mutation rate ``theta`` per sample, and infer ``alpha`` together with the two population sizes and the time ``t_change`` of the size change, the latter three optimized as their logarithms. The standard errors follow from the Godambe sandwich covariance (see {ref}`bootstrapping`), with the observation resampled by a Poisson draw of each bin.

```{code-cell} python
import numpy as np
import pandas as pd

theta, n_samples = 1, 100_000

true = pg.Coalescent(
    n=20,
    model=pg.BetaCoalescent(alpha=1.6),
    demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 3}})
)

# SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples = true.to_empirical(n_samples=n_samples, seed=42).sfs.samples
observation = pg.SFS(np.random.default_rng(42).poisson(theta * np.asarray(samples)).sum(axis=0))

inf = pg.Inference(
    bounds=dict(alpha=(1.01, 1.99), N0=(0.1, 10), N1=(0.1, 10), t_change=(0.05, 2)),
    coal=lambda alpha, N0, N1, t_change: pg.Coalescent(
        n=20,
        model=pg.BetaCoalescent(alpha=alpha),
        demography=pg.Demography(pop_sizes={'pop_0': {0: N0, t_change: N1}})
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.polymorphic,
        modelled=n_samples * theta * coal.sfs.mean.polymorphic
    ),
    observation=observation,
    resample=lambda sfs, rng: sfs.resample(seed=rng),  # Poisson draw of each bin
    log_scale=['N0', 'N1', 't_change'],
    n_runs=3,
    seed=42
)

inf.run()

pd.DataFrame({'estimate': inf.params_inferred, 'standard error': inf.standard_errors(method='godambe')}).round(3)
```

```{code-cell} python
:tags: [remove-cell]
assert abs(inf.params_inferred['alpha'] - 1.6) < 0.05
```

```{code-cell} r
np <- reticulate::import("numpy", convert = FALSE)

theta <- 1
n_samples <- 100000L

true <- pg$Coalescent(
    n = 20L,
    model = pg$BetaCoalescent(alpha = 1.6),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1),
        events = list(pg$PopSizeChange(pop = "pop_0", time = 0.5, size = 3))
    )
)

# SFS branch lengths of each sample, then Poisson mutation counts summed over samples
samples <- true$to_empirical(n_samples = n_samples, seed = 42L)$sfs$samples
counts <- np$random$default_rng(42L)$poisson(np$multiply(theta, np$ascontiguousarray(samples)))$sum(axis = 0L)
observation <- pg$SFS(reticulate::py_to_r(counts))

inf <- pg$Inference(
    bounds = list(alpha = c(1.01, 1.99), N0 = c(0.1, 10), N1 = c(0.1, 10), t_change = c(0.05, 2)),
    coal = function(alpha, N0, N1, t_change) pg$Coalescent(
        n = 20L,
        model = pg$BetaCoalescent(alpha = alpha),
        demography = pg$Demography(
            pop_sizes = list(pop_0 = N0),
            events = list(pg$PopSizeChange(pop = "pop_0", time = t_change, size = N1))
        )
    ),
    loss = function(coal, obs) pg$PoissonLikelihood()$compute(
        observed = obs$polymorphic,
        modelled = n_samples * theta * coal$sfs$mean$polymorphic
    ),
    observation = observation,
    resample = function(sfs, rng) sfs$resample(seed = rng),  # Poisson draw of each bin
    log_scale = c("N0", "N1", "t_change"),
    n_runs = 3L,
    seed = 42L
)

inf$run()

se <- inf$standard_errors(method = "godambe")
params <- c("alpha", "N0", "N1", "t_change")
round(data.frame(estimate = unlist(inf$params_inferred[params]), `standard error` = se[params], check.names = FALSE), 3)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(inf$params_inferred$alpha - 1.6) < 0.05)
```

+++
The inferred population sizes approximate the true ones, drawn dashed. The shaded band is an approximate 95% confidence interval from the Godambe covariance.

```{code-cell} python
inf.plot_pop_sizes(t=np.linspace(0, 2, 401), interval='godambe', truth=true.demography);
```

```{code-cell} r
plot(inf, which = "pop_sizes", t = seq(0, 2, length.out = 401), interval = "godambe", truth = true$demography)
```

+++
A parametric bootstrap refits the model to 20 resampled observations. The true parameter values are drawn dashed.

```{code-cell} python
:tags: [remove-output]
inf.n_bootstraps = 20

inf.bootstrap()
```

```{code-cell} python
:tags: [remove-cell]
figsize = matplotlib.rcParams['figure.figsize']
matplotlib.rcParams['figure.figsize'] = [7, 7]
```

```{code-cell} python
inf.plot_bootstraps(truth=dict(alpha=1.6, N0=1, N1=3, t_change=0.5));
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams['figure.figsize'] = figsize
assert len(inf.bootstraps) == 20
```

```{code-cell} r
:tags: [remove-output]
inf$n_bootstraps <- 20L

inf$bootstrap()
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 7)
```

```{code-cell} r
plot(inf, which = "bootstraps", truth = list(alpha = 1.6, N0 = 1, N1 = 3, t_change = 0.5))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
stopifnot(nrow(inf$bootstraps) == 20)
```
