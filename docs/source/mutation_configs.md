# Mutation count probabilities
So far, we have only looked at statistics based on branch lengths of the coalescent tree. However, when dealing with short sequences, we may not have enough mutations to compute stable branch-length-based summary statistics. Instead, we may like to obtain the distribution of mutational counts. It is particularly informative to consider the SFS computed over small non-recombining blocks.

In the following example we obtain the first 8 unfolded mutational configuration probabilities under a three-epoch size-change demography using a mutation rate of 1. {meth}`~phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs` returns a generator that yields the mutational configurations in descending order of probability, so a target probability mass is reached after evaluating comparatively few of them. Each configuration is a vector of length `n-1` where the `i`th entry denotes the number of mutations with multiplicities `i+1`, and `n` is the number of lineages. For example, `[1, 1, 0]` means that there is one singleton, one doubleton, and no tripleton mutations.

```{code-cell} python
:tags: [remove-cell]
import matplotlib.pyplot as plt

# render figures at 300 dpi, displayed at their nominal size by docs/merge_notebooks.py
%config InlineBackend.figure_format = 'png'
# pad the saved figure, as its tight bounding box leaves out the axis labels of 3D plots
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.3, 'dpi': 300}
%precision %.7g

plt.rcParams['figure.figsize'] = [4.4, 3.3]
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
Sys.setenv(TQDM_DISABLE = "1")
setwd("~/PycharmProjects/PhaseGen/")
reticulate::use_condaenv("/Users/janek/miniforge3/envs/dev-phasegen", required = TRUE)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3, repr.plot.res = 300)
# the ggplot2 theme of the R figures, padded like the Python figures
ggplot2::theme_set(ggplot2::theme_bw() + ggplot2::theme(plot.margin = ggplot2::margin(12, 12, 12, 12)))
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
The number of mutational configurations is infinite since we may have arbitrarily many mutations, albeit with increasingly lower probabilities depending on the mutation rate and coalescent distribution. We may also wish to obtain probabilities until we have reached a certain probability mass threshold. We can do this by consuming the generator while keeping track of the {attr}`~phasegen.distributions.UnfoldedSFSDistribution.generated_mass` attribute. In the following example, we obtain mutational configurations until the generated mass is above 0.8.

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
Alternatively, we can obtain the probability of a specific mutational configuration (cf. {meth}`~phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config`).

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
We can also do the same for folded configurations. In this case, the configurations are vectors of length `n // 2`. For example, configuration `[1, 1]` denotes one singleton or tripleton and one doubleton mutation.

```{code-cell} python
df = pd.DataFrame(islice(coal.fsfs.get_mutation_configs(theta=1), 30))
      
df.plot(kind='bar', x=0, legend=False, xlabel='config');
```

```{code-cell} python
:tags: [remove-cell]
assert (df[1] > 0).all() and df[1].sum() <= 1
```

```{code-cell} r
df <- do.call(rbind, reticulate::iterate(
    pg$take_n(coal$fsfs$get_mutation_configs(theta = 1), 30L)
))

heights <- as.numeric(df[, 2])
labels <- sapply(df[, 1], function(x) paste(unlist(x), collapse = ", "))

barplot(heights, names.arg = labels, las = 2, xlab = "config", cex.names = 0.6)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(heights > 0), sum(heights) <= 1)
```
