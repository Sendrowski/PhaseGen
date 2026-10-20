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
(exact-gradients)=
## Exact gradients
By default, {class}`~phasegen.inference.Inference` computes the gradient of the loss exactly. Unlike a finite-difference gradient, its cost hardly depends on the number of parameters and it carries no step-size error, so that optimizations over many parameters run considerably faster and converge more reliably. The parameters may enter the demography, the coalescent model and the recombination rate between two loci.

The exact gradient is available for losses that read only first moments, as all losses on this page do: the means of {attr}`Coalescent.sfs <phasegen.distributions.Coalescent.sfs>`, {attr}`Coalescent.fsfs <phasegen.distributions.Coalescent.fsfs>`, {attr}`Coalescent.jsfs <phasegen.distributions.Coalescent.jsfs>`, {attr}`Coalescent.tree_height <phasegen.distributions.Coalescent.tree_height>` and {attr}`Coalescent.total_branch_length <phasegen.distributions.Coalescent.total_branch_length>`, and moments of order one, including those of custom rewards through {meth}`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, each with at most one reward and accumulated until absorption. Any other loss is differentiated by finite differences, as noted in the log. Passing ``gradient='finite_differences'`` to {class}`~phasegen.inference.Inference` turns the exact gradient off and uses finite differences throughout.

+++
### Derivatives of moments
The derivatives the exact gradient is built from are also available directly. For a distribution of a coalescent, such as the tree height, the total branch length or a spectrum, {meth}`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>` returns the derivatives of its expected value, or of the expected value of each bin of a spectrum, with respect to the quantities that define the coalescent: the population size of each deme in each epoch, the migration rate between each ordered pair of demes in each epoch, the start time of each epoch, the parameters of the coalescent model and the recombination rate between two loci. They are collected in a {class}`~phasegen.distributions.DemographyGradient`, whose arrays hold one derivative per epoch and deme, or per epoch and pair of demes. All of these derivatives are computed at once, at the same cost, and grouped by kind: {attr}`DemographyGradient.pop_sizes <phasegen.distributions.DemographyGradient.pop_sizes>`, {attr}`DemographyGradient.migration_rates <phasegen.distributions.DemographyGradient.migration_rates>`, {attr}`DemographyGradient.start_times <phasegen.distributions.DemographyGradient.start_times>` and {attr}`DemographyGradient.parameters <phasegen.distributions.DemographyGradient.parameters>`, which holds one entry per continuous parameter of the model, such as ``alpha`` of a Beta coalescent, and the recombination rate of two loci. Below we differentiate the expected tree height of a Beta coalescent in two demes and show the derivatives by the population sizes and by ``alpha``.

```{code-cell} python
coal = pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 2},
    model=pg.BetaCoalescent(alpha=1.5),
    demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1, 0.5: 2}, 'pop_1': 0.5},
        migration_rates={('pop_0', 'pop_1'): 0.3, ('pop_1', 'pop_0'): 0.3}
    )
)

grad = coal.tree_height.gradient()
```

```{code-cell} r
coal <- pg$Coalescent(
    n = list(pop_0 = 3L, pop_1 = 2L),
    model = pg$BetaCoalescent(alpha = 1.5),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1, pop_1 = 0.5),
        events = list(
            pg$PopSizeChange(pop = "pop_0", time = 0.5, size = 2),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 0.3),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.3)
        )
    )
)

grad <- coal$tree_height$gradient()
```

+++
The derivatives by the population sizes form a table with one row per epoch, here before and after the size change of ``pop_0`` at time 0.5, and one column per deme. Each entry is the change in the expected tree height per unit increase of that population size within that epoch. The last column holds the derivatives by the start times of the epochs: zero for the first epoch, which starts at 0, and for the second the change per unit delay of the size change.

```{code-cell} python
pd.DataFrame(np.column_stack([grad.pop_sizes, grad.start_times]), columns=grad.pops + ['start time'])
```

```{code-cell} python
grad.parameters
```

```{code-cell} r
setNames(as.data.frame(cbind(grad$pop_sizes, grad$start_times)), c(unlist(grad$pops), "start time"))
```

```{code-cell} r
grad$parameters
```

+++
For a spectrum, every array of the gradient carries the bins as leading axes, and {meth}`Coalescent.gradient() <phasegen.distributions.Coalescent.gradient>` returns the same derivatives for the expected value of any reward. The parameters of a model, such as those of an inference, usually define these quantities only indirectly, for instance through the time of a size change. {class}`~phasegen.gradient.Gradient` returns the derivatives of a statistic built from first moments, such as a normalized SFS, with respect to the parameters of a coalescent callback, by chaining the derivatives above through the callback, as {class}`~phasegen.inference.Inference` does for its loss. Here we take the derivatives of each bin of the normalized SFS of a two-epoch demography with respect to the time of the size change and the population size after it. The first moments the statistic reads, here the mean SFS, are detected automatically, and a statistic that also reads another quantity, such as a variance, a density or a moment over a time window, is differentiated by finite differences.

```{code-cell} python
gradient = pg.Gradient(
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})
    ),
    statistic=lambda coal: coal.sfs.mean.normalize().polymorphic
)

value, jacobian = gradient.compute(dict(t=0.3, Ne=0.5))

pd.DataFrame(jacobian, columns=['t', 'Ne'], index=range(1, 10)).round(4)
```

```{code-cell} r
gradient <- pg$Gradient(
    coal = function(t, Ne) pg$Coalescent(
        n = 10L,
        demography = pg$Demography(
            pop_sizes = list(pop_0 = 1),
            events = list(pg$PopSizeChange(pop = "pop_0", time = t, size = Ne))
        )
    ),
    statistic = function(coal) coal$sfs$mean$normalize()$polymorphic
)

result <- gradient$compute(list(t = 0.3, Ne = 0.5))

round(setNames(as.data.frame(result[[2]], row.names = 1:9), c("t", "Ne")), 4)
```

+++
The same derivatives yield the observed Fisher information at the estimate of an inference and the scores of resampled observations, from which {meth}`Inference.standard_errors() <phasegen.inference.Inference.standard_errors>` derives the Fisher and Godambe standard errors shown in {ref}`bootstrapping`.
