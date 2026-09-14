# Demography
The {class}`~phasegen.distributions.Coalescent` expects a demography object to be passed to it, which can be configured in various ways. When constructing a demography object, you can directly specify the time points at which the population sizes or migration rates change.

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

d = pg.Demography(
    pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2.5, 1: 0.8}},
    migration_rates={
        ('pop_0', 'pop_1'): {0: 1.7, 0.7: 2},
        ('pop_1', 'pop_0'): {0: 3}
    }
)

d.plot();
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

d <- pg$Demography(
    pop_sizes = list(pop_0 = 1, pop_1 = 2.5),
    events = c(
        pg$PopSizeChange(pop = "pop_1", time = 1, size = 0.8),
        pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1.7),
        pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0.7, rate = 2),
        pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 3)
    )
)

plot(d)
```

+++
Alternatively, you can configure the demography object after construction by adding demographic events. Below, we add {class}`~phasegen.demography.PopSizeChange` and {class}`~phasegen.demography.MigrationRateChange` events to an empty demography object and check that it has the same epochs as the demography constructed above.

```{code-cell} python
d_events = pg.Demography()

d_events.add_event(pg.PopSizeChange(pop='pop_0', time=0, size=1))
d_events.add_event(pg.PopSizeChange(pop='pop_1', time=0, size=2.5))
d_events.add_event(pg.PopSizeChange(pop='pop_1', time=1, size=0.8))

d_events.add_event(pg.MigrationRateChange(source='pop_0', dest='pop_1', time=0, rate=1.7))
d_events.add_event(pg.MigrationRateChange(source='pop_0', dest='pop_1', time=0.7, rate=2))
d_events.add_event(pg.MigrationRateChange(source='pop_1', dest='pop_0', time=0, rate=3))

list(d_events.epochs) == list(d.epochs)
```

```{code-cell} python
:tags: [remove-cell]
assert list(d_events.epochs) == list(d.epochs)
```

```{code-cell} r
d_events <- pg$Demography()

d_events$add_event(pg$PopSizeChange(pop = "pop_0", time = 0, size = 1))
d_events$add_event(pg$PopSizeChange(pop = "pop_1", time = 0, size = 2.5))
d_events$add_event(pg$PopSizeChange(pop = "pop_1", time = 1, size = 0.8))

d_events$add_event(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1.7))
d_events$add_event(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0.7, rate = 2))
d_events$add_event(pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 3))

epochs <- function(demography) reticulate::iterate(demography$epochs)
all(mapply(`==`, epochs(d_events), epochs(d)))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(mapply(`==`, epochs(d_events), epochs(d))))
```

+++
This is similar to the [Msprime demography API](https://tskit.dev/msprime/docs/stable/demography.html), and we can easily convert to an [``msprime.Demography``](https://tskit.dev/msprime/docs/stable/api.html#msprime.Demography) object. Note that the reverse, converting an msprime demography to a native {class}`~phasegen.demography.Demography` object, is not currently supported due to ``phasegen``'s inherent restriction to discrete rate changes.

```{code-cell} python
d_msprime = d.to_msprime()
```

```{code-cell} python
:tags: [remove-cell]
assert d_msprime.num_populations == 2
```

```{code-cell} r
d_msprime <- d$to_msprime()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(d_msprime$num_populations == 2)
```

+++
## Discretizing continuous demographies
There are also utilities for discretizing continuous demographies. In the example below, we create a discretized demography by passing a continuous callback function to {class}`~phasegen.demography.DiscretizedRateChange`. You can freely combine this with other demographic events. Note that the total runtime of the computations is linear in the number of epochs, i.e., it is roughly a multiple of the number of epochs.

```{code-cell} python
d = pg.Demography(
    events=[
        pg.DiscretizedRateChange(
            trajectory=lambda t: 1.5 + 0.5 * t,
            pop='pop_0',
            start_time=0,
            end_time=5,
            step_size=0.5
        )
    ]
)

d.plot();
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

sizes = [e.pop_sizes['pop_0'] for e in d.get_epochs(np.array([0.0, 2.0, 4.0]))]
assert sizes[0] < sizes[1] < sizes[2]
```

```{code-cell} r
d <- pg$Demography(
    events = c(
        pg$DiscretizedRateChange(
            trajectory = function(t) 1.5 + 0.5 * t,
            pop = "pop_0",
            start_time = 0,
            end_time = 5,
            step_size = 0.5
        )
    )
)

plot(d)
```

```{code-cell} r
:tags: [remove-cell]
sizes <- sapply(d$get_epochs(reticulate::np_array(c(0, 2, 4))), function(e) e$pop_sizes$pop_0)
stopifnot(all(diff(sizes) > 0))
```

+++
For exponential growth or decline, you can also make use of {class}`~phasegen.demography.ExponentialPopSizeChanges`.

```{code-cell} python
d = pg.Demography(
    events=[
        pg.ExponentialPopSizeChanges(
            initial_size={'pop_0': 1.5},
            growth_rate=0.5,
            start_time=0,
            end_time=8,
            step_size=0.5
        )
    ]
)

d.plot();
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

sizes = [e.pop_sizes['pop_0'] for e in d.get_epochs(np.array([0.0, 2.0, 4.0]))]
assert sizes[0] > sizes[1] > sizes[2]
```

```{code-cell} r
d <- pg$Demography(
    events = c(
        pg$ExponentialPopSizeChanges(
            initial_size = list(pop_0 = 1.5),
            growth_rate = 0.5,
            start_time = 0,
            end_time = 8,
            step_size = 0.5
        )
    )
)

plot(d)
```

```{code-cell} r
:tags: [remove-cell]
sizes <- sapply(d$get_epochs(reticulate::np_array(c(0, 2, 4))), function(e) e$pop_sizes$pop_0)
stopifnot(all(diff(sizes) < 0))
```

+++
## Population splits
Population splits (forwards in time) correspond to population mergers (backwards in time). Since ``phasegen`` does not support deterministic lineage movements due to its inherent structure, we can model a population split by specifying a large unidirectional migration rate from the derived to the ancestral population. Below, we model a population split where ``pop_0`` splits from ``pop_1`` at time 2. This corresponds to a population merger of the two populations at time 2 backwards in time, and we thus need to initialize both populations at time 0 in the present.

```{code-cell} python
d = pg.Demography(
    pop_sizes={'pop_0': 1, 'pop_1': 3},
    events=[
        pg.PopulationSplit(
            derived='pop_0',
            ancestral='pop_1',
            time=2
        )
    ]
)
```

```{code-cell} r
d <- pg$Demography(
    pop_sizes = list(pop_0 = 1, pop_1 = 3),
    events = c(
        pg$PopulationSplit(
            derived = "pop_0",
            ancestral = "pop_1",
            time = 2
        )
    )
)
```

+++
Plotting the migration rates, we see that there is a large migration rate from ``pop_0`` to ``pop_1`` at time 2. All migration rates to the derived population (``pop_0``) are furthermore set to 0 at the time of the split.

```{code-cell} python
d.plot_migration();
```

```{code-cell} r
plot(d, which = "migration")
```

+++
Wrapping in a {class}`~phasegen.distributions.Coalescent` object, by specifying the initial numbers of lineages in each population, we can visualize the tree height distribution. We see that the probability of absorption is 0 after the split (forwards in time). This is because our scenario represents a *clean* population split, where the derived population is completely isolated from the ancestral population after the split, which makes coalescence between the two populations impossible.

```{code-cell} python
coal = pg.Coalescent(
    n={'pop_0': 4, 'pop_1': 4},
    demography=d
)

coal.tree_height.pdf.plot();
```

```{code-cell} python
:tags: [remove-cell]
assert coal.tree_height.cdf(1.9) < 1e-6
```

```{code-cell} r
coal <- pg$Coalescent(
    n = list(pop_0 = 4, pop_1 = 4),
    demography = d
)

plot(coal$tree_height$pdf)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(coal$tree_height$cdf(1.9) < 1e-6)
```

+++
## Population mergers
Population mergers (forwards in time) correspond to population splits (backwards in time). Mergers of populations that were completely isolated prior to the merger are difficult to model in a coalescent framework. This is because, backwards in time, we would end up with isolated populations which would not be able to coalesce. We can thus only model mergers of populations, provided they are in contact again eventually. Below is an example.

```{note}
Migration rates between demes are assumed to be 0 if not specified otherwise.
```

```{code-cell} python
coal = pg.Coalescent(
    n={'pop_0': 8, 'pop_1': 0},
    demography=pg.Demography(
        pop_sizes={'pop_0': 1, 'pop_1': 1},
        events=[
            pg.MigrationRateChanges(
                {
                    ('pop_1', 'pop_0'): {2: 0, 8: 0.1},
                    ('pop_0', 'pop_1'): {0.5: 1, 3: 0, 8: 0.1}
                }
            )
        ]
    )
)

_, axs = plt.subplots(1, 2, figsize=(7, 3))
coal.demography.plot_migration(ax=axs[0])
coal.tree_height.pdf.plot(ax=axs[1]);
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
coal <- pg$Coalescent(
    n = list(pop_0 = 8, pop_1 = 0),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1, pop_1 = 1),
        events = c(
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 2, rate = 0),
            pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 8, rate = 0.1),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0.5, rate = 1),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 3, rate = 0),
            pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 8, rate = 0.1)
        )
    )
)

library(patchwork)

plot(coal$demography, which = "migration") + plot(coal$tree_height$pdf)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

