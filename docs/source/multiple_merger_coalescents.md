# Coalescent models
Supported coalescent models are {class}`~phasegen.coalescent_models.StandardCoalescent`, {class}`~phasegen.coalescent_models.BetaCoalescent` and {class}`~phasegen.coalescent_models.DiracCoalescent`, and can be specified when constructing the {class}`~phasegen.distributions.Coalescent` distribution.

```{code-cell} python
:tags: [remove-cell]
import matplotlib.pyplot as plt

# render figures at 300 dpi, displayed at their nominal size by docs/merge_notebooks.py
%config InlineBackend.figure_format = 'png'
# pad the saved figure, as its tight bounding box leaves out the axis labels of 3D plots
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.3, 'dpi': 300}
%precision %.7g

plt.rcParams['figure.figsize'] = (5, 4)
```

```{code-cell} python
import phasegen as pg
from matplotlib import pyplot as plt

# compare 2-SFS of Kingman and Beta coalescents
kingman = pg.Coalescent(n=10, model=pg.StandardCoalescent())
beta = pg.Coalescent(n=10, model=pg.BetaCoalescent(alpha=1.5))

fig, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={"projection": "3d"})
for ax in axs:
    ax.set_box_aspect(None, zoom=1.15)
fig.subplots_adjust(left=0, right=1, wspace=0)

# beta coalescent shows positive correlations between disparate minor allele frequencies
kingman.sfs.cov.plot_surface(ax=axs[0], show=False, title='Kingman')
beta.sfs.cov.plot_surface(ax=axs[1], title='Beta');
```

```{code-cell} python
:tags: [remove-cell]
assert beta.sfs.cov.data[1, 8] > 0 > kingman.sfs.cov.data[1, 8]
```

```{code-cell} r
:tags: [remove-cell]
Sys.setenv(TQDM_DISABLE = "1")
setwd("~/PycharmProjects/PhaseGen/")
reticulate::use_condaenv("/Users/janek/miniforge3/envs/dev-phasegen", required = TRUE)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4, repr.plot.res = 300)
# the ggplot2 theme of the R figures, padded like the Python figures
ggplot2::theme_set(ggplot2::theme_bw() + ggplot2::theme(plot.margin = ggplot2::margin(12, 12, 12, 12)))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
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

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4)
```
