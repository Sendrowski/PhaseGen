# Empirical distributions

Every exact distribution `phasegen` computes can also be sampled. {meth}`~phasegen.distributions.PhaseTypeDistribution.to_empirical` draws genealogies from the same phase-type generator and returns an empirical counterpart ({class}`~phasegen.distributions.EmpiricalPhaseTypeDistribution`) that exposes the same interface as the exact {class}`~phasegen.distributions.PhaseTypeDistribution`, only estimated by Monte Carlo rather than the exact matrix computation.

```{versionadded} 2.0
```

This serves two purposes. First, it provides a fast, independent check of the exact results. Second, because the sampler is fully vectorised, it provides a fallback for state spaces too large for the exact computation to remain tractable.

```{code-cell} python
import phasegen as pg
```

```{code-cell} r
:tags: [remove-output]
library(phasegen)
pg <- load_phasegen()
```

+++
## Sampling a spectrum

Consider a two-population demography with a size change and asymmetric migration. {meth}`~phasegen.distributions.JointSFSDistribution.to_empirical` draws genealogies from this model and bins each one's branch lengths into the joint SFS. Its `mean` is the sampled counterpart of the exact {meth}`~phasegen.distributions.Coalescent.jsfs`.

```{code-cell} python
coal = pg.Coalescent(
    n={'pop_0': 4, 'pop_1': 4},
    demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1, 1: 0.3}, 'pop_1': {0: 1.5}},
        migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.2},
    ),
)
```

```{code-cell} python
:tags: [remove-cell]
import matplotlib

subplot_defaults = {k: matplotlib.rcParams[k] for k in ('figure.subplot.left', 'figure.subplot.right', 'figure.subplot.wspace')}
matplotlib.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.right': 1, 'figure.subplot.wspace': 0})
```

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

# the sampled mean joint SFS reproduces the exact surface from 50,000 genealogies
sampled = coal.jsfs.to_empirical(50_000, seed=42)

_, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={'projection': '3d'})
coal.jsfs.mean.plot_surface(ax=axs[0], show=False, title='Exact')
sampled.mean.plot_surface(ax=axs[1], title='Sampled (50,000)');
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams.update(subplot_defaults)
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert abs(sampled.mean.data.sum() - coal.jsfs.mean.data.sum()) < 4 * np.sqrt(coal.total_branch_length.var / 50_000)
```

```{code-cell} r
coal <- pg$Coalescent(
    n = pg$LineageConfig(list(pop_0 = 4, pop_1 = 4)),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1, pop_1 = 1.5),
        events = c(
            pg$PopSizeChange(pop = "pop_0", time = 1, size = 0.3),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 0.5),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.2)
        )
    )
)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
:tags: [full-width]
# the sampled mean joint SFS reproduces the exact surface from 50,000 genealogies
sampled <- coal$jsfs$to_empirical(50000L, seed = 42L)

par(mfrow = c(1, 2))
persp(coal$jsfs$mean, title = "Exact")
persp(sampled$mean, title = "Sampled (50,000)")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(sum(sampled$mean$data) - sum(coal$jsfs$mean$data)) < 4 * sqrt(coal$total_branch_length$var / 50000))
```

+++
Because the empirical object exposes the same interface as the exact distribution, scalar summaries compare directly. A hundred thousand genealogies closely recover the exact tree height and total branch length.

```{code-cell} python
th = coal.tree_height.to_empirical(100_000, seed=42)
tbl = coal.total_branch_length.to_empirical(100_000, seed=42)

print(f"{'':<22}{'exact':>10}{'sampled':>10}")
print(f"{'tree height (mean)':<22}{coal.tree_height.mean:>10.3f}{th.mean:>10.3f}")
print(f"{'tree height (var)':<22}{coal.tree_height.var:>10.3f}{th.var:>10.3f}")
print(f"{'branch length (mean)':<22}{coal.total_branch_length.mean:>10.3f}{tbl.mean:>10.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert abs(th.mean - coal.tree_height.mean) < 4 * np.sqrt(coal.tree_height.var / 100_000)
assert abs(tbl.mean - coal.total_branch_length.mean) < 4 * np.sqrt(coal.total_branch_length.var / 100_000)
assert abs(th.var - coal.tree_height.var) < 0.05 * coal.tree_height.var
```

```{code-cell} r
th <- coal$tree_height$to_empirical(100000L, seed = 42L)
tbl <- coal$total_branch_length$to_empirical(100000L, seed = 42L)

cat(sprintf("%-22s%10s%10s\n", "", "exact", "sampled"))
cat(sprintf("%-22s%10.3f%10.3f\n", "tree height (mean)", coal$tree_height$mean, th$mean))
cat(sprintf("%-22s%10.3f%10.3f\n", "tree height (var)", coal$tree_height$var, th$var))
cat(sprintf("%-22s%10.3f%10.3f\n", "branch length (mean)", coal$total_branch_length$mean, tbl$mean))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(
    abs(th$mean - coal$tree_height$mean) < 4 * sqrt(coal$tree_height$var / 100000),
    abs(tbl$mean - coal$total_branch_length$mean) < 4 * sqrt(coal$total_branch_length$var / 100000),
    abs(th$var - coal$tree_height$var) < 0.05 * coal$tree_height$var
)
```

+++
## External ground truth

{meth}`~phasegen.distributions.Coalescent.to_msprime` returns an `msprime`-backed coalescent ({class}`~phasegen.distributions.MsprimeCoalescent`) with the same interface. Where `to_empirical` is `phasegen`'s own sampler, this is a fully independent implementation: an external ground truth rather than a self-consistency check, at the cost of the slower, non-vectorised `msprime` simulation. `phasegen` is extensively validated against `msprime` across a wide range of scenarios.

```{code-cell} python
ms = coal.to_msprime(num_replicates=5_000, seed=42)

print(f"tree height mean   exact = {coal.tree_height.mean:.3f}   msprime = {ms.tree_height.mean:.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert abs(ms.tree_height.mean - coal.tree_height.mean) < 4 * np.sqrt(coal.tree_height.var / 5_000)
```

```{code-cell} r
ms <- coal$to_msprime(num_replicates = 5000L, seed = 42L)

cat(sprintf("tree height mean   exact = %.3f   msprime = %.3f\n",
            coal$tree_height$mean, ms$tree_height$mean))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(ms$tree_height$mean - coal$tree_height$mean) < 4 * sqrt(coal$tree_height$var / 5000))
```
