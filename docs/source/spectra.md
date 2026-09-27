# Spectra & summary statistics
Beyond scalar moments, the {class}`~phasegen.distributions.Coalescent` provides full spectra, namely the joint (multi-population) and two-locus site-frequency spectra, as well as a set of standard scalar summary statistics. All of these are exact and respect the full demography and coalescent model. See the {doc}`quickstart` for how to configure the underlying model.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = (5, 4)
```

```{code-cell} python
import phasegen as pg
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4)
```

```{code-cell} r
:tags: [remove-output]
library(phasegen)
pg <- load_phasegen()
```

+++
## Joint site-frequency spectrum
For multiple populations, {meth}`~phasegen.distributions.Coalescent.jsfs` gives the joint (multi-population) SFS: the expected branch length subtending each configuration of derived-allele counts per population (the deme of origin). For ``P`` populations it is a ``P``-dimensional {class}`~sfsutils.spectrum.JointSFS` of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, with higher moments available via {meth}`moment(k) <phasegen.distributions.JointSFSDistribution.moment>`, {meth}`var <phasegen.distributions.JointSFSDistribution.var>` and {meth}`cov <phasegen.distributions.JointSFSDistribution.cov>`. It is restricted to a single locus, and the state space grows quickly with the per-population sample sizes, which therefore need to remain small.

```{code-cell} python
# a two-population demography with a population-size change and asymmetric migration
coal = pg.Coalescent(
    n={'pop_0': 5, 'pop_1': 5},
    demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1, 1: 0.3}, 'pop_1': {0: 1.5}},
        migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.2},
    ),
)

# mean joint SFS: expected branch length subtending each (pop_0, pop_1) allele-frequency configuration
coal.jsfs.mean.plot_surface(title='Mean joint SFS');
```

```{code-cell} python
:tags: [remove-cell]
data = coal.jsfs.mean.data
assert abs(data.sum() - data[0, 0] - data[-1, -1] - coal.total_branch_length.mean) < 1e-8
```

```{code-cell} r
# a two-population demography with a population-size change and asymmetric migration
coal <- pg$Coalescent(
    n = list(pop_0 = 5, pop_1 = 5),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1, pop_1 = 1.5),
        events = c(
            pg$PopSizeChange(pop = "pop_0", time = 1, size = 0.3),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 0.5),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.2)
        )
    )
)

# mean joint SFS: expected branch length subtending each (pop_0, pop_1) allele-frequency configuration
persp(coal$jsfs$mean, title = "Mean joint SFS")
```

```{code-cell} r
:tags: [remove-cell]
data <- coal$jsfs$mean$data
stopifnot(abs(sum(data) - data[1, 1] - data[nrow(data), ncol(data)] - coal$total_branch_length$mean) < 1e-8)
```

+++
## Tree height under recombination
For two loci separated by recombination, ``phasegen`` provides the exact distribution of the tree height, covering both the time to the ultimate MRCA across the two loci and the marginal genealogy at each locus.

Two loci at recombination rate 0.1, and the mean time to the ultimate most recent common ancestor of the sample across both loci:

```{code-cell} python
coal = pg.Coalescent(n=8, loci=pg.LocusConfig(recombination_rate=0.1, n=2))

coal.tree_height.mean
```

```{code-cell} python
:tags: [remove-cell]
linked = coal.tree_height.mean
```

```{code-cell} r
coal <- pg$Coalescent(n = 8L, loci = pg$LocusConfig(recombination_rate = 0.1, n = 2L))

coal$tree_height$mean
```

```{code-cell} r
:tags: [remove-cell]
linked <- coal$tree_height$mean
```

+++
The mean marginal tree height of the first locus:

```{code-cell} python
coal.tree_height.loci[0].mean
```

```{code-cell} python
:tags: [remove-cell]
assert abs(coal.tree_height.loci[0].mean - 2 * (1 - 1 / 8)) < 1e-8 and linked > coal.tree_height.loci[0].mean
```

```{code-cell} r
coal$tree_height$loci[[0L]]$mean
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(coal$tree_height$loci[[0L]]$mean - 2 * (1 - 1 / 8)) < 1e-8, linked > coal$tree_height$loci[[0L]]$mean)
```

+++
The same mean when all eight lineages start out unlinked between the two loci:

```{code-cell} python
coal = pg.Coalescent(n=8, loci=pg.LocusConfig(recombination_rate=0.1, n=2, n_unlinked=8))

coal.tree_height.mean
```

```{code-cell} python
:tags: [remove-cell]
assert coal.tree_height.mean > linked
```

```{code-cell} r
coal <- pg$Coalescent(n = 8L, loci = pg$LocusConfig(recombination_rate = 0.1, n = 2L, n_unlinked = 8L))

coal$tree_height$mean
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(coal$tree_height$mean > linked)
```

+++
## Two-locus SFS under recombination
For two loci separated by recombination rate ``r``, {meth}`~phasegen.distributions.Coalescent.sfs2` gives the two-locus SFS, whose entry ``(i, j)`` is the expected product of the branch length subtending ``i`` samples at locus 0 and ``j`` samples at locus 1. Its mean and correlation are {class}`~sfsutils.spectrum.TwoLocusSFS` objects, and it interpolates between the within-tree SFS covariance at ``r = 0`` (fully linked) and independent loci as ``r → ∞`` (for the standard coalescent). The starting linkage is set via the {class}`~phasegen.locus.LocusConfig` ``n_unlinked``. A single population is supported, and the state space grows quickly with the sample size.

The single- and two-locus spectra apply to different locus configurations. {meth}`~phasegen.distributions.Coalescent.sfs2` requires exactly two loci, while the single-locus {meth}`~phasegen.distributions.Coalescent.sfs` requires one. Its marginal mean does not depend on the recombination rate, so the SFS of one of two loci equals that of a coalescent with a single locus.

```{code-cell} python
:tags: [remove-cell]
subplot_defaults = {k: matplotlib.rcParams[k] for k in ('figure.subplot.left', 'figure.subplot.right', 'figure.subplot.wspace')}
matplotlib.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.right': 1, 'figure.subplot.wspace': 0})
```

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

# the two-locus SFS interpolates between tightly linked and independent loci as r grows
_, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={"projection": "3d"})

# r = 0.1: tightly linked -> strong cross-locus structure (close to the within-tree SFS covariance)
pg.Coalescent(n=6, loci=2, recombination_rate=0.1).sfs2.mean.plot_surface(
    ax=axs[0], show=False, title='Tightly linked (r = 0.1)')

# r = 10: nearly independent -> approaches the outer product of the marginal SFS
pg.Coalescent(n=6, loci=2, recombination_rate=10.0).sfs2.mean.plot_surface(
    ax=axs[1], title='Near-independent (r = 10)');
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams.update(subplot_defaults)
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

marginal = pg.Coalescent(n=6).sfs.mean.data[1:6]
outer = np.outer(marginal, marginal)
rel = lambda r: np.max(np.abs(pg.Coalescent(n=6, loci=2, recombination_rate=r).sfs2.mean.data[1:6, 1:6] / outer - 1))
assert rel(10.0) < 0.1 and rel(0.1) > 1
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
:tags: [full-width]
# the two-locus SFS interpolates between tightly linked and independent loci as r grows

par(mfrow = c(1, 2))
# r = 0.1: tightly linked -> strong cross-locus structure (close to the within-tree SFS covariance)
persp(pg$Coalescent(n = 6L, loci = 2L, recombination_rate = 0.1)$sfs2$mean, title = "Tightly linked (r = 0.1)")

# r = 10: nearly independent -> approaches the outer product of the marginal SFS
persp(pg$Coalescent(n = 6L, loci = 2L, recombination_rate = 10.0)$sfs2$mean, title = "Near-independent (r = 10)")
```

```{code-cell} r
:tags: [remove-cell]
marginal <- pg$Coalescent(n = 6L)$sfs$mean$data[2:6]
outer_product <- outer(marginal, marginal)
rel <- function(r) max(abs(pg$Coalescent(n = 6L, loci = 2L, recombination_rate = r)$sfs2$mean$data[2:6, 2:6] / outer_product - 1))
stopifnot(rel(10) < 0.1, rel(0.1) > 1)
```

+++
## Summary statistics
Beyond full spectra, several standard scalar summaries are available directly from the {class}`~phasegen.distributions.Coalescent`, each respecting the full demography and coalescent model:

- Population structure: Hudson's {attr}`Coalescent.fst <phasegen.distributions.Coalescent.fst>` and Patterson's f-statistics ({meth}`~phasegen.distributions.Coalescent.f2`, {meth}`~phasegen.distributions.Coalescent.f3`, {meth}`~phasegen.distributions.Coalescent.f4`), all derived from inter-population pairwise coalescence times.
- Linkage: the correlation of coalescence times between two loci ({meth}`tree_height.loci.get_corr <phasegen.distributions.MarginalLocusDistributions.get_corr>`), which decays towards zero as the recombination rate grows (for the standard coalescent).
- SFS skew: Tajima's {meth}`~phasegen.distributions.UnfoldedSFSDistribution.tajimas_d`, together with the underlying {meth}`~phasegen.distributions.UnfoldedSFSDistribution.theta_pi` and {meth}`~phasegen.distributions.UnfoldedSFSDistribution.theta_w` estimators.

We illustrate them on a relatively complex scenario: a structured three-population demography with asymmetric population sizes and migration.

```{code-cell} python
struct = pg.Coalescent(
    n={'pop_0': 2, 'pop_1': 2, 'pop_2': 2},
    demography=pg.Demography(
        pop_sizes={'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.5},
        migration_rates={
            ('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5,
            ('pop_1', 'pop_2'): 0.2, ('pop_2', 'pop_1'): 0.2,
            ('pop_0', 'pop_2'): 0.2, ('pop_2', 'pop_0'): 0.2,
        },
    ),
)
```

```{code-cell} r
struct <- pg$Coalescent(
    n = list(pop_0 = 2, pop_1 = 2, pop_2 = 2),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1.0, pop_1 = 1.0, pop_2 = 1.5),
        events = c(
            pg$SymmetricMigrationRateChanges(pops = c("pop_0", "pop_1"), rate = 0.5),
            pg$SymmetricMigrationRateChanges(pops = c("pop_1", "pop_2"), rate = 0.2),
            pg$SymmetricMigrationRateChanges(pops = c("pop_0", "pop_2"), rate = 0.2)
        )
    )
)
```

+++
Population structure, as Hudson's F<sub>ST</sub> and Patterson's f-statistics:

```{code-cell} python
print(f"F_ST                    = {struct.fst:.3f}")
print(f"f2(pop_0, pop_2)        = {struct.f2('pop_0', 'pop_2'):.3f}")
print(f"f3(pop_1; pop_0, pop_2) = {struct.f3('pop_1', 'pop_0', 'pop_2'):.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < struct.fst < 1 and struct.f2('pop_0', 'pop_2') > 0
```

```{code-cell} r
cat(sprintf("F_ST                    = %.3f\n", struct$fst))
cat(sprintf("f2(pop_0, pop_2)        = %.3f\n", struct$f2("pop_0", "pop_2")))
cat(sprintf("f3(pop_1; pop_0, pop_2) = %.3f\n", struct$f3("pop_1", "pop_0", "pop_2")))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(struct$fst > 0, struct$fst < 1, struct$f2("pop_0", "pop_2") > 0)
```

+++
Linkage, as the correlation of the coalescence times at two loci, which decays as the recombination rate grows:

```{code-cell} python
for r in [0.1, 1.0, 10.0]:
    corr = pg.Coalescent(n=2, loci=2, recombination_rate=r).tree_height.loci.get_corr(0, 1)
    print(f"corr(T_A, T_B) at r={r:<4} = {corr:.3f}")
```

```{code-cell} python
:tags: [remove-cell]
corrs = [pg.Coalescent(n=2, loci=2, recombination_rate=r).tree_height.loci.get_corr(0, 1) for r in [0.1, 1.0, 10.0]]
assert 1 > corrs[0] > corrs[1] > corrs[2] > 0
```

```{code-cell} r
for (r in c(0.1, 1.0, 10.0)) {
    corr <- pg$Coalescent(n = 2L, loci = 2L, recombination_rate = r)$tree_height$loci$get_corr(0L, 1L)
    cat(sprintf("corr(T_A, T_B) at r=%-4s = %.3f\n", r, corr))
}
```

```{code-cell} r
:tags: [remove-cell]
corrs <- sapply(c(0.1, 1, 10), function(r) pg$Coalescent(n = 2L, loci = 2L, recombination_rate = r)$tree_height$loci$get_corr(0L, 1L))
stopifnot(all(diff(corrs) < 0), corrs[1] < 1, corrs[3] > 0)
```

+++
Skew of the SFS, as Tajima's D under recent population growth, where an excess of low-frequency variants makes D negative:

```{code-cell} python
growth = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.1}}))
print(f"Tajima's D (growth) = {growth.sfs.tajimas_d:.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert growth.sfs.tajimas_d < 0
```

```{code-cell} r
growth <- pg$Coalescent(
    n = 10L,
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1.0),
        events = c(pg$PopSizeChange(pop = "pop_0", time = 0.5, size = 0.1))
    )
)
cat(sprintf("Tajima's D (growth) = %.3f\n", growth$sfs$tajimas_d))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(growth$sfs$tajimas_d < 0)
```
