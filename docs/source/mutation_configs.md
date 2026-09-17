# Mutation count probabilities
So far, we have only looked at statistics based on branch lengths of the coalescent tree. However, when dealing with short sequences, we may not have enough mutations to compute stable branch-length-based summary statistics. Instead, we may obtain the distribution of mutational counts. It is particularly informative to consider the SFS computed over small non-recombining blocks.

In the following example we obtain the first 8 unfolded mutational configurations and their probabilities under a three-epoch size-change demography, with a mutation rate of 1 per unit of branch length. Configurations, their probabilities and the order in which {meth}`UnfoldedSFSDistribution.get_mutation_configs() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs>` yields them are described in {meth}`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} python
import phasegen as pg
import pandas as pd
from itertools import islice

# a three-epoch size-change demography (decline then expansion)
coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3, 1.5: 2}}))

pd.DataFrame(islice(coal.sfs.get_mutation_configs(theta=1), 8))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

```{code-cell} r
library(phasegen)
pg <- load_phasegen()

# a three-epoch size-change demography (decline then expansion)
coal <- pg$Coalescent(
    n = 4L,
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1),
        events = c(
            pg$PopSizeChange(pop = "pop_0", time = 0.5, size = 0.3),
            pg$PopSizeChange(pop = "pop_0", time = 1.5, size = 2)
        )
    )
)

do.call(rbind, reticulate::iterate(
    pg$take_n(coal$sfs$get_mutation_configs(theta = 1), 8L)
))
```

+++
The iterator does not terminate, so we consume it until the yielded probability mass, {attr}`UnfoldedSFSDistribution.generated_mass <phasegen.distributions.UnfoldedSFSDistribution.generated_mass>`, exceeds 0.8.

```{code-cell} python
it = coal.sfs.get_mutation_configs(theta=1)

# continue until generated mass is above 0.8
pd.DataFrame(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))
```

```{code-cell} python
:tags: [remove-cell]
assert coal.sfs.generated_mass >= 0.8
```

```{code-cell} r
it <- coal$sfs$get_mutation_configs(theta = 1)

until <- pg$takewhile_inclusive(function(x) {
    return(coal$sfs$generated_mass < 0.8)
}, it)

# continue until generated mass is above 0.8
do.call(rbind, reticulate::iterate(until))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(coal$sfs$generated_mass >= 0.8)
```

+++
The probability of a single configuration is returned by {meth}`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.

```{code-cell} python
coal.sfs.get_mutation_config([1, 1, 0], theta=1)
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < coal.sfs.get_mutation_config([1, 1, 0], theta=1) < 1
```

```{code-cell} r
coal$sfs$get_mutation_config(c(1, 1, 0), theta = 1)
```

```{code-cell} r
:tags: [remove-cell]
p <- coal$sfs$get_mutation_config(c(1, 1, 0), theta = 1)
stopifnot(p > 0, p < 1)
```

+++
Folded configurations are obtained in the same way from {attr}`Coalescent.fsfs <phasegen.distributions.Coalescent.fsfs>`.

```{code-cell} python
df = pd.DataFrame(islice(coal.fsfs.get_mutation_configs(theta=1), 30))
      
df.plot(kind='bar', x=0, legend=False, xlabel='config');
```

```{code-cell} python
:tags: [remove-cell]
assert (df[1] > 0).all() and df[1].sum() <= 1
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.height = 3.8)
```

```{code-cell} r
df <- do.call(rbind, reticulate::iterate(
    pg$take_n(coal$fsfs$get_mutation_configs(theta = 1), 30L)
))

heights <- as.numeric(df[, 2])
labels <- sapply(df[, 1], function(x) paste(unlist(x), collapse = ", "))

par(mar = c(4.5, 4, 1, 1))
barplot(heights, names.arg = labels, las = 2, xlab = "config", cex.names = 0.6, col = "#1f77b4", border = NA)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(heights > 0), sum(heights) <= 1)
```
