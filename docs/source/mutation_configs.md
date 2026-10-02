# Mutation count probabilities
So far, we have only looked at statistics based on branch lengths of the coalescent tree. However, when dealing with short sequences, we may not have enough mutations to compute stable branch-length-based summary statistics. Instead, we may obtain the distribution of mutational counts. It is particularly informative to consider the SFS computed over small non-recombining blocks.

In the following example we obtain the first 8 unfolded mutational configurations and their probabilities under a three-epoch size-change demography, with a mutation rate of 1 per unit of branch length. Configurations, their probabilities and the order in which {meth}`UnfoldedSFSDistribution.get_mutation_configs() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs>` yields them are described in {meth}`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`. Configuration ``(2, 1, 0)`` holds two singletons, one doubleton and no tripletons.

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
By default, the iterator yields the configurations in descending order of probability, and with ``order='count'`` in ascending order of the total number of mutations. It does not terminate, so we consume it until the yielded probability mass, {attr}`UnfoldedSFSDistribution.generated_mass <phasegen.distributions.UnfoldedSFSDistribution.generated_mass>`, exceeds 0.8.

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
## Layouts
The iterator yields {class}`~phasegen.distributions.MutationConfig` objects, which compare equal to the plain tuple of their counts. Each carries the {class}`~phasegen.distributions.MutationLayout` that defines its bins. A bin merges one or more elementary frequency classes, listed in {attr}`MutationLayout.bins <phasegen.distributions.MutationLayout.bins>`, and {meth}`MutationConfig.to_array() <phasegen.distributions.MutationConfig.to_array>` places the counts at the positions of the classes in the spectrum array.

```{code-cell} python
config, p = next(coal.sfs.get_mutation_configs(theta=1))

config.layout.bins, config.to_array()
```

```{code-cell} r
# a configuration converts to an integer vector of its counts, so it is retrieved unconverted
builtins <- reticulate::import_builtins(convert = FALSE)
config <- reticulate::py_get_item(builtins$`next`(coal$sfs$get_mutation_configs(theta = 1)), 0L)

list(config$layout$bins, reticulate::py_to_r(config$to_array()))
```

+++
The iterator and {meth}`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>` accept any layout of the spectrum, and {meth}`MutationLayout.rebin() <phasegen.distributions.MutationLayout.rebin>` groups its classes into other bins. The folded layout of {meth}`UnfoldedSFSDistribution.mutation_layout() <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>` merges the classes ``i`` and ``n - i``. It is the default layout of {attr}`Coalescent.fsfs <phasegen.distributions.Coalescent.fsfs>`, whose configuration ``(2, 1)`` holds two singletons or tripletons and one doubleton.

```{code-cell} python
df = pd.DataFrame(islice(coal.fsfs.get_mutation_configs(theta=1), 30))
      
df.plot(kind='bar', x=0, legend=False, xlabel='config');
```

```{code-cell} python
:tags: [remove-cell]
assert (df[1] > 0).all() and df[1].sum() <= 1
assert coal.fsfs.mutation_layout() == coal.sfs.mutation_layout(folded=True)
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

+++
With several demes, ``demes=True`` resolves each bin by the deme in which the mutation occurs. The bins are ordered by deme and then by frequency class.

```{code-cell} python
# two demes with asymmetric migration
coal2 = pg.Coalescent(
    n={'pop_0': 2, 'pop_1': 2},
    demography=pg.Demography(
        pop_sizes={'pop_0': {0: 0.5}, 'pop_1': {0: 2}},
        migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 0.2},
    ),
)

layout = coal2.sfs.mutation_layout(demes=True)

pd.DataFrame(islice(coal2.sfs.get_mutation_configs(theta=0.5, layout=layout), 6))
```

```{code-cell} python
:tags: [remove-cell]
assert layout.bins[0] == (('pop_0', 1),) and len(layout) == 6
```

```{code-cell} r
# two demes with asymmetric migration
coal2 <- pg$Coalescent(
    n = list(pop_0 = 2L, pop_1 = 2L),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 0.5, pop_1 = 2),
        events = c(
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.2)
        )
    )
)

layout <- coal2$sfs$mutation_layout(demes = TRUE)

do.call(rbind, reticulate::iterate(
    pg$take_n(coal2$sfs$get_mutation_configs(theta = 0.5, layout = layout), 6L)
))
```

+++
The joint spectrum {attr}`Coalescent.jsfs <phasegen.distributions.Coalescent.jsfs>` has one bin per polymorphic descendant vector, the numbers of descendants a mutated branch subtends in each deme. {meth}`JointSFSDistribution.mutation_layout(folded=True) <phasegen.distributions.JointSFSDistribution.mutation_layout>` merges each descendant vector with its complement. Descendant vectors that no genealogy carries together, such as ``(2, 1)`` and ``(1, 2)``, have probability zero.

```{code-cell} python
layout = coal2.jsfs.mutation_layout()

layout.bins
```

```{code-cell} python
pd.DataFrame(islice(coal2.jsfs.get_mutation_configs(theta=0.5), 6))
```

```{code-cell} python
:tags: [remove-cell]
assert coal2.jsfs.get_mutation_config(layout.config([1 if b[0] in ((2, 1), (1, 2)) else 0 for b in layout.bins]), 0.5) == 0
```

```{code-cell} r
layout <- coal2$jsfs$mutation_layout()

reticulate::py_get_attr(layout, "bins")
```

```{code-cell} r
do.call(rbind, reticulate::iterate(
    pg$take_n(coal2$jsfs$get_mutation_configs(theta = 0.5), 6L)
))
```

+++
The two-locus spectrum {attr}`Coalescent.sfs2 <phasegen.distributions.Coalescent.sfs2>` counts the mutations of both loci, with bins labelled ``(locus, i)``, and ``theta`` is the mutation rate per locus. {meth}`TwoLocusSFSDistribution.mutation_layout(loci=(0,)) <phasegen.distributions.TwoLocusSFSDistribution.mutation_layout>` counts the mutations of one locus, whose configuration probabilities equal those of a single-locus coalescent.

```{code-cell} python
coal3 = pg.Coalescent(n=3, loci=2, recombination_rate=1)

pd.DataFrame(islice(coal3.sfs2.get_mutation_configs(theta=0.5), 6))
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

one = coal3.sfs2.mutation_layout(loci=(0,))
assert np.isclose(coal3.sfs2.get_mutation_config(one.config([1, 0]), 0.5),
                  pg.Coalescent(n=3).sfs.get_mutation_config([1, 0], 0.5), rtol=1e-12)
```

```{code-cell} r
coal3 <- pg$Coalescent(n = 3L, loci = 2L, recombination_rate = 1)

do.call(rbind, reticulate::iterate(
    pg$take_n(coal3$sfs2$get_mutation_configs(theta = 0.5), 6L)
))
```
