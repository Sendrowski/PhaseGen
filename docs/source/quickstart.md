# Quickstart
## Defining the coalescent
In order to obtain statistics from coalescent distributions, we first need to define such a distribution. This is done by creating a {class}`~phasegen.distributions.Coalescent` object which serves as an entry point from which all statistics can be obtained. Below is an example of a simple Kingman coalescent distribution with 10 lineages, and a single population of constant size 1.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} python
import phasegen as pg

coal = pg.Coalescent(
    n=10,
    demography=pg.Demography(
        pop_sizes=1
    )
)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

```{code-cell} r
library(phasegen)
pg <- load_phasegen()

coal <- pg$Coalescent(
    n = 10,
    demography = pg$Demography(
        pop_sizes = 1
    )
)
```

+++
## Moments and distribution functions
We can now access various statistics from this distribution, which are made available as cached properties of the component distributions of the {class}`~phasegen.distributions.Coalescent` object. The mean coalescence time, or tree height:

```{code-cell} python
coal.tree_height.mean
```

```{code-cell} python
:tags: [remove-cell]
assert abs(coal.tree_height.mean - 2 * (1 - 1 / 10)) < 1e-10
```

```{code-cell} r
coal$tree_height$mean
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(coal$tree_height$mean - 2 * (1 - 1 / 10)) < 1e-10)
```

+++
The tree height variance:

```{code-cell} python
coal.tree_height.var
```

```{code-cell} python
:tags: [remove-cell]
assert abs(coal.tree_height.var - sum(4 / (k * (k - 1)) ** 2 for k in range(2, 11))) < 1e-10
```

```{code-cell} r
coal$tree_height$var
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(coal$tree_height$var - sum(4 / ((2:10) * (1:9))^2)) < 1e-10)
```

+++
The expected total branch length:

```{code-cell} python
coal.total_branch_length.mean
```

```{code-cell} python
:tags: [remove-cell]
assert abs(coal.total_branch_length.mean - 2 * sum(1 / i for i in range(1, 10))) < 1e-10
```

```{code-cell} r
coal$total_branch_length$mean
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(coal$total_branch_length$mean - 2 * sum(1 / (1:9))) < 1e-10)
```

+++
The expected site-frequency spectrum:

```{code-cell} python
coal.sfs.mean.plot();
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert np.allclose(coal.sfs.mean.data[1:10], 2 / np.arange(1, 10))
```

```{code-cell} r
plot(coal$sfs$mean)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(as.numeric(coal$sfs$mean$data[2:10]), 2 / (1:9))))
```

+++
In fact, ``tree_height``, ``total_branch_length`` and ``sfs`` are all {class}`~phasegen.distributions.PhaseTypeDistribution` objects which can be accessed to obtain statistics on these distributions. In the API reference, these are {class}`~phasegen.distributions.TreeHeightDistribution`, {class}`~phasegen.distributions.TotalBranchLengthDistribution`, and {class}`~phasegen.distributions.UnfoldedSFSDistribution`, respectively. {class}`~phasegen.distributions.PhaseTypeDistribution` instances support the computation of moments and cross-moments of arbitrary order, which is only limited by the computational burden associated with higher-order moments. Every one of them also offers the PDF, CDF and quantile function.

```{code-cell} python
coal.tree_height.quantile(0.95)
```

```{code-cell} python
:tags: [remove-cell]
assert abs(coal.tree_height.cdf(coal.tree_height.quantile(0.95)) - 0.95) < 1e-4
```

```{code-cell} python
coal.tree_height.pdf.plot();
```

```{code-cell} r
coal$tree_height$quantile(0.95)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(coal$tree_height$cdf(coal$tree_height$quantile(0.95)) - 0.95) < 1e-4)
```

```{code-cell} r
plot(coal$tree_height$pdf)
```

+++
## Demography and coalescent models
Before we discuss how to obtain more complex statistics, we first define a more complex coalescent distribution. Here, we define a two-population coalescent using the {class}`~phasegen.coalescent_models.BetaCoalescent` model, where the population sizes and migration rates are time-dependent. The nested mappings passed as ``pop_sizes`` and ``migration_rates`` define the population name and times at which the population sizes and migration rates change.

```{code-cell} python
coal = pg.Coalescent(
    n=pg.LineageConfig({'pop_0': 3, 'pop_1': 5}),
    model=pg.BetaCoalescent(alpha=1.7),
    demography=pg.Demography(
        pop_sizes={
            'pop_1': {0: 1.2, 5: 0.1, 5.5: 0.8},
            'pop_0': {0: 1.0}
        },
        migration_rates={
            ('pop_0', 'pop_1'): {0: 0.2, 8: 0.3},
            ('pop_1', 'pop_0'): {0: 0.5}
        }
    )
)
```

```{code-cell} r
coal <- pg$Coalescent(
    n = pg$LineageConfig(list(pop_0 = 3, pop_1 = 5)),
    model = pg$BetaCoalescent(alpha = 1.7),
    demography = pg$Demography(
        pop_sizes = list(pop_1 = 1.2, pop_0 = 1.0),
        events = c(
            pg$PopSizeChange(pop = "pop_1", time = 5, size = 0.1),
            pg$PopSizeChange(pop = "pop_1", time = 5.5, size = 0.8),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 0.2),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 8, rate = 0.3),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.5)
        )
    )
)
```

+++
Plotting the demography (see {meth}`Demography.plot() <phasegen.demography.Demography.plot>`) shows the population sizes and migration rates over time. ``pop_1`` experiences a bottleneck at time 5, with continuous migration between the two populations.

```{code-cell} python
coal.demography.plot();
```

```{code-cell} r
plot(coal$demography)
```

+++
The density of the underlying {class}`~phasegen.distributions.TreeHeightDistribution`:

```{code-cell} python
coal.tree_height.pdf.plot();
```

```{code-cell} r
plot(coal$tree_height$pdf)
```

+++
We can also compute higher-order moments of the SFS, such as the branch length correlation between branches subtending different numbers of lineages in the coalescent tree.

```{code-cell} python
coal.sfs.corr.plot();
```

```{code-cell} r
plot(coal$sfs$corr)
```

+++
We may also marginalize over a single population. Here we obtain the mean SFS of ``pop_0``, which represents the branch lengths for lineages that subtend ``i`` lineages in the coalescent tree, while spending time in population ``pop_0``.

```{code-cell} python
coal.sfs.demes['pop_0'].mean.plot();
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(sum(coal.sfs.demes[d].mean.data for d in coal.demography.pop_names), coal.sfs.mean.data)
```

```{code-cell} r
plot(coal$sfs$demes$pop_0$mean)
```

```{code-cell} r
:tags: [remove-cell]
demes_sum <- Reduce(`+`, lapply(coal$demography$pop_names, function(d) coal$sfs$demes[[d]]$mean$data))
stopifnot(isTRUE(all.equal(demes_sum, coal$sfs$mean$data)))
```

+++
## Joint distributions
Any two rewards also have a joint distribution, obtained from {meth}`Coalescent.joint_distribution() <phasegen.distributions.Coalescent.joint_distribution>`. Returning to the Kingman coalescent with 10 lineages, the tree height and the total branch length are strongly positively correlated, since a tall tree also tends to have long branches overall.

```{code-cell} python
coal = pg.Coalescent(n=10)

joint = coal.joint_distribution(pg.TreeHeightReward(), pg.TotalBranchLengthReward())

joint.corr
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(joint.mean, [coal.tree_height.mean, coal.total_branch_length.mean])
assert joint.corr > 0.9
```

```{code-cell} r
coal <- pg$Coalescent(n = 10)

joint <- coal$joint_distribution(pg$TreeHeightReward(), pg$TotalBranchLengthReward())

joint$corr
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(as.numeric(joint$mean), c(coal$tree_height$mean, coal$total_branch_length$mean))))
stopifnot(joint$corr > 0.9)
```

+++
The joint density, with the tree height on the horizontal and the total branch length on the vertical axis:

```{code-cell} python
joint.pdf.plot();
```

```{code-cell} r
plot(joint$pdf)
```

+++
The {doc}`distribution_functions` section describes joint, marginal and conditional distributions in detail. The {doc}`rewards` section describes how to obtain more complex moments by specifying rewards. Parameter inference from observed summary statistics is described in the {doc}`inference` section.
