# Real-world examples
The following examples apply ``phasegen`` to demographic models inferred from human data.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
## Out of Africa
We consider the three-population out-of-Africa model of [Gutenkunst et al. (2009)](https://doi.org/10.1371/journal.pgen.1000695) for the Yoruba in Ibadan (YRI), Utah residents of northern and western European ancestry (CEU) and Han Chinese in Beijing (CHB). Backward in time, CEU and CHB, which have grown exponentially since their split 848 generations ago, merge into a bottlenecked out-of-Africa population, which merges with YRI 5,600 generations ago. We load the model from [``stdpopsim``](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html#sec_catalog_homsap_models_outofafrica_3g09) and convert it with {meth}`Demography.from_msprime() <phasegen.demography.Demography.from_msprime>`. Population sizes in the model are numbers of diploid individuals, and with a scale of 2N<sub>A</sub>, for the ancestral size N<sub>A</sub> = 7,300, the converted sizes are relative to N<sub>A</sub> and one unit of time is 14,600 generations. Exponential growth is discretized into piecewise constant sizes.

```{code-cell} python
:tags: [full-width]
import numpy as np
import matplotlib.pyplot as plt
import stdpopsim
import phasegen as pg

model = stdpopsim.get_species('HomSap').get_demographic_model('OutOfAfrica_3G09')

N_A = 7300
d = pg.Demography.from_msprime(model.model, scale=2 * N_A)

_, axs = plt.subplots(1, 2, figsize=(7, 3))
t = np.linspace(0, 0.8, 400)
d.plot_pop_sizes(t=t, ax=axs[0], show=False)
d.plot_migration(t=t, ax=axs[1], show=False)
axs[1].set_ylim(0, 5);
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
library(phasegen)
library(patchwork)
pg <- load_phasegen()
stdpopsim <- reticulate::import("stdpopsim")

model <- stdpopsim$get_species("HomSap")$get_demographic_model("OutOfAfrica_3G09")

N_A <- 7300
d <- pg$Demography$from_msprime(model$model, scale = 2 * N_A)

t <- seq(0, 0.8, length.out = 400)
plot(d, which = "pop_sizes", t = t) +
    plot(d, which = "migration", t = t) +
    ggplot2::coord_cartesian(ylim = c(0, 5)) +
    ggplot2::theme(legend.position.inside = c(1, 1), legend.justification = c(1, 1))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
The coalescence time of two lineages carries the imprint of this history. The density for pairs sampled within CEU or within CHB peaks sharply just after the split of CEU and CHB, in their small founding populations, whereas a pair sampled across YRI and CEU coalesces only through migration before the out-of-Africa split. The densities jump where the rate of coalescence changes abruptly, such as at the ancestral expansion. Each density is that of the tree height of a coalescent with two sampled lineages, and populations without samples are given 0 lineages.

```{code-cell} python
pairs = {
    'YRI': {'YRI': 2},
    'CEU': {'CEU': 2},
    'CHB': {'CHB': 2},
    'YRI-CEU': {'YRI': 1, 'CEU': 1},
    'CEU-CHB': {'CEU': 1, 'CHB': 1}
}
unsampled = {pop: 0 for pop in d.pop_names}

_, ax = plt.subplots()
for label, n in pairs.items():
    coal = pg.Coalescent(n=unsampled | n, demography=d)
    coal.tree_height.pdf.plot(
        ax=ax, t=np.linspace(0, 1.2, 400), show=False, label=label, title='Pairwise coalescence time'
    )
```

```{code-cell} python
:tags: [remove-cell]
means = {k: pg.Coalescent(n=unsampled | n, demography=d).tree_height.mean for k, n in pairs.items()}
assert means['YRI-CEU'] > means['CEU-CHB'] > means['CEU'] > means['CHB']
```

```{code-cell} r
pairs <- list(
    YRI = list(YRI = 2),
    CEU = list(CEU = 2),
    CHB = list(CHB = 2),
    `YRI-CEU` = list(YRI = 1, CEU = 1),
    `CEU-CHB` = list(CEU = 1, CHB = 1)
)
unsampled <- setNames(as.list(rep(0, length(d$pop_names))), d$pop_names)

p <- NULL
for (label in names(pairs)) {
    coal <- pg$Coalescent(n = modifyList(unsampled, pairs[[label]]), demography = d)
    p <- plot(coal$tree_height$pdf, t = seq(0, 1.2, length.out = 400), add = p, label = label,
              title = "Pairwise coalescence time")
}
p
```

```{code-cell} r
:tags: [remove-cell]
means <- sapply(pairs, function(n) pg$Coalescent(n = modifyList(unsampled, n), demography = d)$tree_height$mean)
stopifnot(means[["YRI-CEU"]] > means[["CEU-CHB"]], means[["CEU-CHB"]] > means[["CEU"]], means[["CEU"]] > means[["CHB"]])
```

+++
The same pairwise coalescence times give Hudson's F<sub>ST</sub> and Patterson's f-statistics in their branch form, in units of time.

```{code-cell} python
coal = pg.Coalescent(n=unsampled | {'YRI': 2, 'CEU': 2, 'CHB': 2}, demography=d)

print(f"F_ST              = {coal.fst:.3f}")
print(f"f2(YRI, CEU)      = {coal.f2('YRI', 'CEU'):.3f}")
print(f"f2(CEU, CHB)      = {coal.f2('CEU', 'CHB'):.3f}")
print(f"f3(CEU; YRI, CHB) = {coal.f3('CEU', 'YRI', 'CHB'):.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < coal.fst < 1
assert coal.f2('YRI', 'CEU') > coal.f2('CEU', 'CHB') > 0 and coal.f3('CEU', 'YRI', 'CHB') > 0
```

```{code-cell} r
coal <- pg$Coalescent(n = modifyList(unsampled, list(YRI = 2, CEU = 2, CHB = 2)), demography = d)

cat(sprintf("F_ST              = %.3f\n", coal$fst))
cat(sprintf("f2(YRI, CEU)      = %.3f\n", coal$f2("YRI", "CEU")))
cat(sprintf("f2(CEU, CHB)      = %.3f\n", coal$f2("CEU", "CHB")))
cat(sprintf("f3(CEU; YRI, CHB) = %.3f\n", coal$f3("CEU", "YRI", "CHB")))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(coal$fst > 0, coal$fst < 1, coal$f2("YRI", "CEU") > coal$f2("CEU", "CHB"), coal$f2("CEU", "CHB") > 0,
          coal$f3("CEU", "YRI", "CHB") > 0)
```

+++
The joint SFS of three lineages each from CEU and CHB, with YRI unsampled, is marginalized onto the CEU and CHB axes for plotting, the populations being ordered by name. Most of its mass lies on low-frequency alleles private to one of the two populations.

```{code-cell} python
coal = pg.Coalescent(n=unsampled | {'CEU': 3, 'CHB': 3}, demography=d)

coal.jsfs.mean.marginalize([0, 1]).plot_surface(title='Mean joint SFS of CEU and CHB');
```

```{code-cell} python
:tags: [remove-cell]
jsfs = coal.jsfs.mean.marginalize([0, 1]).data
assert abs(jsfs.sum() - jsfs[0, 0] - jsfs[-1, -1] - coal.total_branch_length.mean) < 1e-8
```

```{code-cell} r
coal <- pg$Coalescent(n = modifyList(unsampled, list(CEU = 3, CHB = 3)), demography = d)

persp(coal$jsfs$mean$marginalize(c(0L, 1L)), title = "Mean joint SFS of CEU and CHB")
```

```{code-cell} r
:tags: [remove-cell]
jsfs <- coal$jsfs$mean$marginalize(c(0L, 1L))$data
stopifnot(abs(sum(jsfs) - jsfs[1, 1] - jsfs[nrow(jsfs), ncol(jsfs)] - coal$total_branch_length$mean) < 1e-8)
```

+++
Any two quantities of the same genealogy have a joint distribution. Below we take two lineages each from CEU and YRI, and consider the tree height together with the length of the branches subtending exactly one CEU lineage and no other, along which a mutation gives a CEU-private singleton. Both grow with the depth of the genealogy, so they are positively correlated.

```{code-cell} python
coal = pg.Coalescent(n=unsampled | {'CEU': 2, 'YRI': 2}, demography=d)

joint = coal.joint(pg.TreeHeightReward(), pg.JointSFSReward((1, 0, 0)))

joint.corr
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < joint.corr < 1
```

```{code-cell} r
coal <- pg$Coalescent(n = modifyList(unsampled, list(CEU = 2, YRI = 2)), demography = d)

joint <- coal$joint(pg$TreeHeightReward(), pg$JointSFSReward(c(1L, 0L, 0L)))

joint$corr
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(joint$corr > 0, joint$corr < 1)
```

+++
Inverting the joint density exactly requires the transform at many points for each epoch of the model, so we estimate it from genealogies sampled with {meth}`Coalescent.to_empirical() <phasegen.distributions.Coalescent.to_empirical>` instead. The bright band at short singleton branches comes from genealogies in which the two CEU lineages coalesce in the out-of-Africa bottleneck, so that their private branches stay short however deep the genealogy is. The rays above it come from genealogies in which a CEU lineage remains on its own until close to the root, and the breaks at the out-of-Africa split and the ancestral expansion mark abrupt changes in the rate of coalescence.

```{code-cell} python
sampled = coal.to_empirical(1000000, seed=42)

sampled.joint(pg.TreeHeightReward(), pg.JointSFSReward((1, 0, 0))).pdf.plot(n_points=150);
```

```{code-cell} python
:tags: [remove-cell]
corr = sampled.joint(pg.TreeHeightReward(), pg.JointSFSReward((1, 0, 0))).corr
assert abs(corr - joint.corr) < 4 * (1 - joint.corr ** 2) / np.sqrt(1000000)
```

```{code-cell} r
sampled <- coal$to_empirical(1000000L, seed = 42L)

plot(sampled$joint(pg$TreeHeightReward(), pg$JointSFSReward(c(1L, 0L, 0L)))$pdf, n_points = 150L)
```

```{code-cell} r
:tags: [remove-cell]
corr <- sampled$joint(pg$TreeHeightReward(), pg$JointSFSReward(c(1L, 0L, 0L)))$corr
stopifnot(abs(corr - joint$corr) < 4 * (1 - joint$corr^2) / sqrt(1000000))
```
