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
By default, {class}`~phasegen.inference.Inference` differentiates the loss exactly with respect to the parameters. One exact gradient costs about one evaluation of the moments the loss reads and one backward pass through them, whatever the number of parameters, whereas a finite-difference gradient costs one model evaluation per parameter, so that optimizations over many parameters run considerably faster. The moments are never differenced, so the gradient is free of the step-size errors their finite differences incur, which makes the optimization more stable and its convergence more reliable. The parameters may enter the population sizes, migration rates and epoch times of the demography, the parameters of a multiple-merger coalescent and the recombination rate between two loci.

The exact gradient is available for losses that read only first moments, as all losses on this page do: the means of {attr}`Coalescent.sfs <phasegen.distributions.Coalescent.sfs>`, {attr}`Coalescent.fsfs <phasegen.distributions.Coalescent.fsfs>`, {attr}`Coalescent.jsfs <phasegen.distributions.Coalescent.jsfs>`, {attr}`Coalescent.tree_height <phasegen.distributions.Coalescent.tree_height>` and {attr}`Coalescent.total_branch_length <phasegen.distributions.Coalescent.total_branch_length>`, and moments of order one, including those of custom rewards through {meth}`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, each with at most one reward and accumulated until absorption. Any other loss is differentiated by finite differences, as noted in the log. Passing ``gradient='finite_differences'`` to {class}`~phasegen.inference.Inference` turns the exact gradient off and uses finite differences throughout.

+++
### Derivatives of moments
The derivatives of first moments are also available directly. {meth}`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>` returns those with respect to the population sizes, migration rates and start times of the epochs, the parameters of the coalescent model and the recombination rate between two loci, as a {class}`~phasegen.distributions.DemographyGradient`. Below we differentiate the expected tree height of a Beta coalescent in two demes.

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

# derivatives by the population size of each deme, one row per epoch
pd.DataFrame(grad.pop_sizes, columns=grad.pops)
```

```{code-cell} python
grad.parameters
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

# derivatives by the population size of each deme, one row per epoch
setNames(as.data.frame(grad$pop_sizes), unlist(grad$pops))
```

```{code-cell} r
grad$parameters
```

+++
For a spectrum, every array of the gradient carries the bins as leading axes, and {meth}`Coalescent.gradient() <phasegen.distributions.Coalescent.gradient>` differentiates the first moment of any reward. Derivatives with respect to the parameters of a model are obtained with {class}`~phasegen.gradient.Gradient`, which chains these derivatives through a coalescent callback, as {class}`~phasegen.inference.Inference` does, for any statistic built from first moments. Here we take the Jacobian of the normalized SFS of a two-epoch demography with respect to the time of the size change and the population size.

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
The same derivatives yield the observed Fisher information at the estimate of an inference, from which {meth}`Inference.standard_errors() <phasegen.inference.Inference.standard_errors>` derives the standard errors shown in {ref}`joint-sfs`.
