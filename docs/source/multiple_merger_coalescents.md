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
