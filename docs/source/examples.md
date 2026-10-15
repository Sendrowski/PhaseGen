# Real-world examples
The following examples apply ``phasegen`` to demographic models inferred from great ape and human data.

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
## Human, chimp and gorilla
We take one lineage each from human (H), chimpanzee (C) and gorilla (G) under the speciation history estimated by [Hobolth et al. (2007)](https://doi.org/10.1371/journal.pgen.0030007). Human and chimpanzee split 4.1 million years ago, and their ancestor split from gorilla about 2 million years earlier. The two ancestral populations have effective sizes of 65,000 and 45,000, and the generation time is 25 years. Time is in generations and each population size is given as 2N, twice the number of diploid individuals, so that a pair of lineages coalesces at rate one over the size. With a single lineage per population, the present-day sizes do not matter and are set to that of the human-chimpanzee ancestor.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import phasegen as pg

N_HC, N_HCG, generation_time = 65000, 45000, 25
t_HC, t_HCG = 4.1e6 / generation_time, 6.1e6 / generation_time
print(f"t_HC = {t_HC:,.0f}, t_HCG = {t_HCG:,.0f} generations")

coal = pg.Coalescent(
    n={'C': 1, 'G': 1, 'H': 1},
    demography=pg.Demography(pop_sizes={'C': 2 * N_HC, 'G': 2 * N_HC, 'H': 2 * N_HC}, events=[
        pg.PopulationSplit(time=t_HC, derived='C', ancestral='H'),
        pg.PopSizeChange(pop='H', time=t_HC, size=2 * N_HC),
        pg.PopSizeChange(pop='H', time=t_HCG, size=2 * N_HCG),
        pg.PopulationSplit(time=t_HCG, derived='G', ancestral='H')
    ])
)
```

```{code-cell} r
library(phasegen)
library(patchwork)
pg <- load_phasegen()

N_HC <- 65000
N_HCG <- 45000
generation_time <- 25
t_HC <- 4.1e6 / generation_time
t_HCG <- 6.1e6 / generation_time
cat(sprintf("t_HC = %s, t_HCG = %s generations\n", format(t_HC, big.mark = ","), format(t_HCG, big.mark = ",")))

coal <- pg$Coalescent(
    n = list(C = 1, G = 1, H = 1),
    demography = pg$Demography(pop_sizes = list(C = 2 * N_HC, G = 2 * N_HC, H = 2 * N_HC), events = c(
        pg$PopulationSplit(time = t_HC, derived = "C", ancestral = "H"),
        pg$PopSizeChange(pop = "H", time = t_HC, size = 2 * N_HC),
        pg$PopSizeChange(pop = "H", time = t_HCG, size = 2 * N_HCG),
        pg$PopulationSplit(time = t_HCG, derived = "G", ancestral = "H")
    ))
)
```

+++
Below is the joint density of the tree height and the length of the internal branch ancestral to the human and chimpanzee lineages, both in generations. The branch length is a {class}`~phasegen.rewards.JointSFSReward` with one descendant in each. We invert it with more cosine terms than the default, which resolves its sharp edges. Its shape is set by the two speciation times. No genealogy is shallower than t<sub>HCG</sub>, which gives the vertical edge. The human and chimpanzee lineages coalesce no earlier than t<sub>HC</sub>, so their branch is at most the tree height minus t<sub>HC</sub>, which gives the upper diagonal edge. Above the diagonal ridge at the tree height minus t<sub>HCG</sub>, the two lineages coalesced in their own ancestor, and below it in the ancestor shared with gorilla.

```{code-cell} python
:tags: [remove-cell]
cos_terms = pg.Settings.cos_terms, pg.Settings.cos_terms_2d
```

```{code-cell} python
pg.Settings.cos_terms, pg.Settings.cos_terms_2d = 3072, 1024

HC = pg.JointSFSReward({'H': 1, 'C': 1})
joint = coal.joint(pg.TreeHeightReward(), HC)

_, ax = plt.subplots()
joint.pdf.plot(ax=ax, n_points=225, show=False, title='Joint density')
ax.ticklabel_format(style='sci', scilimits=(0, 0))
ax.set_xlabel('tree height')
ax.set_ylabel('human-chimpanzee branch length');
```

```{code-cell} r
:tags: [remove-cell]
cos_terms <- list(pg$Settings$cos_terms, pg$Settings$cos_terms_2d)
```

```{code-cell} r
pg$Settings$cos_terms <- 3072L
pg$Settings$cos_terms_2d <- 1024L

HC <- pg$JointSFSReward(list(H = 1L, C = 1L))
joint <- coal$joint(pg$TreeHeightReward(), HC)

plot(joint$pdf, n_points = 225L, title = "Joint density") +
    ggplot2::labs(x = "tree height", y = "human-chimpanzee branch length")
```

+++
When the gorilla lineage coalesces first with the human or the chimpanzee lineage, the genealogy disagrees with the species tree and the human-chimpanzee branch has length zero. This incomplete lineage sorting (ILS) is the atom of the branch length at zero. It is not visible in the density above and is given exactly by the CDF at zero.

```{code-cell} python
p_ils = coal.distribution(HC).cdf(0)
print(f"P(ILS) = {p_ils:.4f}")
```

```{code-cell} python
:tags: [remove-cell]
assert np.isclose(p_ils, 2 / 3 * np.exp(-(t_HCG - t_HC) / (2 * N_HC)), rtol=1e-6)
```

```{code-cell} r
p_ils <- coal$distribution(HC)$cdf(0)
cat(sprintf("P(ILS) = %.4f\n", p_ils))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(p_ils / (2 / 3 * exp(-(t_HCG - t_HC) / (2 * N_HC))) - 1) < 1e-6)
```

+++
Conditioning on the human-chimpanzee branch length with {meth}`JointRewardDistribution.conditional() <phasegen.distributions.JointRewardDistribution.conditional>` gives the distribution of the tree height for a given length of this branch. At length zero, this is the tree height under ILS, which rises from zero at t<sub>HCG</sub>. A positive length gives a normalised vertical slice through the joint density above. The tree height is the age of the human-chimpanzee coalescence plus the length of the human-chimpanzee branch, which spans from that coalescence to the root. As this coalescence is older than t<sub>HC</sub>, a longer branch shifts the curve to older tree heights. The density jumps up where the human-chimpanzee coalescence becomes older than t<sub>HCG</sub>, as the ancestor shared with gorilla is smaller and coalescence there is faster.

```{code-cell} python
_, ax = plt.subplots()
t = np.linspace(2e5, 6e5, 600)
for length in (0, 25000, 100000):
    joint.conditional(on='b', value=length).pdf.plot(ax=ax, t=t, show=False, label=f'HC branch length {length:,}')
ax.ticklabel_format(style='sci', scilimits=(0, 0))
ax.set_xlabel('tree height')
ax.set_title('Conditional tree height');
```

```{code-cell} python
:tags: [remove-cell]
assert np.isclose(joint.conditional(on='b', value=0).mean, t_HCG + 4 / 3 * 2 * N_HCG, rtol=1e-5)
pg.Settings.cos_terms, pg.Settings.cos_terms_2d = cos_terms
```

```{code-cell} r
t <- seq(2e5, 6e5, length.out = 600)
p <- NULL
for (length in c(0, 25000, 100000)) {
    p <- plot(joint$conditional(on = "b", value = length)$pdf, t = t, add = p,
              label = paste("HC branch length", format(length, big.mark = ",", scientific = FALSE)),
              title = "Conditional tree height")
}
p + ggplot2::labs(x = "tree height")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(joint$conditional(on = "b", value = 0)$mean / (t_HCG + 4 / 3 * 2 * N_HCG) - 1) < 1e-5)
pg$Settings$cos_terms <- cos_terms[[1]]
pg$Settings$cos_terms_2d <- cos_terms[[2]]
```

+++
A genealogy of three lineages has a single internal branch, so the human-chimpanzee and human-gorilla branches are never both present and their product has mean zero.

```{code-cell} python
HG = pg.JointSFSReward({'H': 1, 'G': 1})
product = coal.moment(2, (HC, HG), center=False)
print(f"E[L_HC L_HG] = {product:.4g}")
```

```{code-cell} r
HG <- pg$JointSFSReward(list(H = 1L, G = 1L))
product <- coal$moment(2L, list(HC, HG), center = FALSE)
cat(sprintf("E[L_HC L_HG] = %.4g\n", product))
```

+++
Their covariance, given by {meth}`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, therefore equals minus the product of their means.

```{code-cell} python
cov = coal.moment(2, (HC, HG))
means = coal.moment(1, (HC,)) * coal.moment(1, (HG,))
print(f"Cov(L_HC, L_HG) = {cov:.4g}, -E[L_HC] E[L_HG] = {-means:.4g}")
```

```{code-cell} python
:tags: [remove-cell]
assert abs(product) < 1e-10 * means and np.isclose(cov, -means, rtol=1e-8)
```

```{code-cell} r
cov <- coal$moment(2L, list(HC, HG))
means <- coal$moment(1L, list(HC)) * coal$moment(1L, list(HG))
cat(sprintf("Cov(L_HC, L_HG) = %.4g, -E[L_HC] E[L_HG] = %.4g\n", cov, -means))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(product) < 1e-10 * means, abs(cov / -means - 1) < 1e-8)
```

+++
## Out of Africa
We consider the three-population out-of-Africa model of [Gutenkunst et al. (2009)](https://doi.org/10.1371/journal.pgen.1000695) for the Yoruba in Ibadan (YRI), Utah residents of northern and western European ancestry (CEU) and Han Chinese in Beijing (CHB). In this model, an ancestral African population expands and persists at constant size as YRI. A small group leaves Africa 5,600 generations ago and stays small, the out-of-Africa bottleneck, until it splits 848 generations ago into the still smaller founding populations of CEU and CHB, which have grown exponentially since. The populations exchange migrants throughout. We load the model from [``stdpopsim``](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html#sec_catalog_homsap_models_outofafrica_3g09) and draw it with [``demesdraw``](https://grahamgower.github.io/demesdraw/), with time in generations on a logarithmic scale. The width of each population is proportional to its size, and the thin arrows show migration.

```{code-cell} python
import demesdraw
import stdpopsim

model = stdpopsim.get_species('HomSap').get_demographic_model('OutOfAfrica_3G09')

_, ax = plt.subplots()
demesdraw.tubes(model.model.to_demes(), ax=ax, log_time=True, seed=42);
```

```{code-cell} r
stdpopsim <- reticulate::import("stdpopsim")
demesdraw <- reticulate::import("demesdraw")
plt <- reticulate::import("matplotlib.pyplot")

model <- stdpopsim$get_species("HomSap")$get_demographic_model("OutOfAfrica_3G09")

ax <- plt$subplots(figsize = c(4.4, 3.3))[[2]]
invisible(demesdraw$tubes(model$model$to_demes(), ax = ax, log_time = TRUE, seed = 42L))
plt$savefig("out_of_africa.png", dpi = 300, bbox_inches = "tight")
```

```{code-cell} r
:tags: [remove-input]
IRdisplay::display_png(file = "out_of_africa.png")
invisible(file.remove("out_of_africa.png"))
```

+++
We convert the model with {meth}`Demography.from_msprime() <phasegen.demography.Demography.from_msprime>`. Population sizes in the model are numbers of diploid individuals, and with a scale of 2N<sub>A</sub>, for the ancestral size N<sub>A</sub> = 7,300, the converted sizes are relative to N<sub>A</sub> and one unit of time is 14,600 generations. Exponential growth is discretized into piecewise constant sizes, as the trajectories below show.

```{code-cell} python
:tags: [full-width]
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
We next compute the density of the coalescence time of two lineages sampled within one population or across two. Each density is that of the tree height of a coalescent with two sampled lineages, and populations without samples are given 0 lineages. The densities plotted below carry the imprint of the history shown above. Those for pairs sampled within CEU or within CHB peak sharply just after the split of CEU and CHB, in their small founding populations, whereas a pair sampled within YRI, whose population stays large, coalesces over a broad range of times. A pair sampled across YRI and CEU coalesces only through migration before the out-of-Africa split. The densities jump where the rate of coalescence changes abruptly, such as at the ancestral expansion.

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
We can also compute the joint SFS, here of three lineages each from CEU and CHB, with YRI unsampled. It is marginalized onto the CEU and CHB axes for plotting, the populations being ordered by name. Most of its mass lies on low-frequency alleles private to one of the two populations.

```{code-cell} python
coal = pg.Coalescent(n=unsampled | {'CEU': 3, 'CHB': 3}, demography=d)

coal.jsfs.mean.marginalize([0, 1]).plot(title='Mean joint SFS of CEU and CHB', log_scale=False);
```

```{code-cell} python
:tags: [remove-cell]
jsfs = coal.jsfs.mean.marginalize([0, 1]).data
assert abs(jsfs.sum() - jsfs[0, 0] - jsfs[-1, -1] - coal.total_branch_length.mean) < 1e-8
```

```{code-cell} r
coal <- pg$Coalescent(n = modifyList(unsampled, list(CEU = 3, CHB = 3)), demography = d)

plot(coal$jsfs$mean$marginalize(c(0L, 1L)), title = "Mean joint SFS of CEU and CHB", log_scale = FALSE)
```

```{code-cell} r
:tags: [remove-cell]
jsfs <- coal$jsfs$mean$marginalize(c(0L, 1L))$data
stopifnot(abs(sum(jsfs) - jsfs[1, 1] - jsfs[nrow(jsfs), ncol(jsfs)] - coal$total_branch_length$mean) < 1e-8)
```

+++
Any two quantities of the same genealogy have a joint distribution. Below we take two lineages each from CEU and YRI, and consider the tree height, the time to the most recent common ancestor of all four lineages, together with the CEU singleton length, the total length of the branches ancestral to a single CEU lineage and to no other sampled lineage. A mutation on such a branch is a singleton private to CEU, and the length is specified by a {class}`~phasegen.rewards.JointSFSReward` with one descendant in CEU and none in the other populations. Both grow with the depth of the genealogy, so they are positively correlated.

```{code-cell} python
coal = pg.Coalescent(n=unsampled | {'CEU': 2, 'YRI': 2}, demography=d)

joint = coal.joint(pg.TreeHeightReward(), pg.JointSFSReward({'CEU': 1}))

joint.corr
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < joint.corr < 1
```

```{code-cell} r
coal <- pg$Coalescent(n = modifyList(unsampled, list(CEU = 2, YRI = 2)), demography = d)

joint <- coal$joint(pg$TreeHeightReward(), pg$JointSFSReward(list(CEU = 1L)))

joint$corr
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(joint$corr > 0, joint$corr < 1)
```

+++
We can also access the full joint distribution of the two quantities. Its density, shown below with the two marginal densities beside it, is estimated here from genealogies sampled with {meth}`Coalescent.to_empirical() <phasegen.distributions.Coalescent.to_empirical>`, which is faster than exact inversion for a model with this many epochs. The bright band at short singleton lengths comes from genealogies in which the two CEU lineages coalesce early, in the small founding population of CEU or soon after in the out-of-Africa bottleneck, so that their private branches stay short however deep the genealogy is. The rays above it come from genealogies in which a CEU lineage remains on its own until close to the root, and the breaks at the out-of-Africa split and the ancestral expansion mark abrupt changes in the rate of coalescence.

```{code-cell} python
:tags: [full-width]
sampled = coal.to_empirical(1000000, seed=42)
sampled_joint = sampled.joint(pg.TreeHeightReward(), pg.JointSFSReward({'CEU': 1}))

_, axs = plt.subplots(1, 2, figsize=(7, 3), width_ratios=[1.25, 1])
sampled_joint.pdf.plot(ax=axs[0], n_points=150, show=False, title='Joint density')
axs[0].set_xlabel('tree height')
axs[0].set_ylabel('CEU singleton length')

x = np.linspace(0, 3, 400)
sampled_joint.marginal('a').pdf.plot(ax=axs[1], t=x, show=False, label='tree height')
sampled_joint.marginal('b').pdf.plot(
    ax=axs[1], t=x, show=False, label='CEU singleton length', title='Marginal densities'
);
```

```{code-cell} python
:tags: [remove-cell]
# four standard errors of the sampled correlation, 0.0014 by batch means over 100 blocks of 10,000 genealogies
assert abs(sampled_joint.corr - joint.corr) < 0.0056
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
sampled <- coal$to_empirical(1000000L, seed = 42L)
sampled_joint <- sampled$joint(pg$TreeHeightReward(), pg$JointSFSReward(list(CEU = 1L)))

x <- seq(0, 3, length.out = 400)
p <- plot(sampled_joint$marginal("a")$pdf, t = x, label = "tree height")
p <- plot(sampled_joint$marginal("b")$pdf, t = x, add = p, label = "CEU singleton length",
          title = "Marginal densities")

plot(sampled_joint$pdf, n_points = 150L, title = "Joint density") +
    ggplot2::labs(x = "tree height", y = "CEU singleton length") +
    p + ggplot2::theme(legend.position.inside = c(1, 1), legend.justification = c(1, 1))
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

```{code-cell} r
:tags: [remove-cell]
# four standard errors of the sampled correlation, 0.0014 by batch means over 100 blocks of 10,000 genealogies
stopifnot(abs(sampled_joint$corr - joint$corr) < 0.0056)
```
