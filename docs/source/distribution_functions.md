# Distribution functions

`phasegen` exposes the full distribution of any accumulated coalescent reward (tree height, total branch length, an individual SFS bin's branch length, or any {doc}`custom reward <rewards>`) as callable, plottable distribution-function objects (`pdf`, `cdf`, `quantile`). They are computed from the model without sampling. The tree height is evaluated by matrix exponentiation, as described at {class}`~phasegen.distributions.TreeHeightDistribution`, and every other reward by numerical inversion of its Laplace transform, as described at {class}`~phasegen.distributions.RewardDistribution`. The inversion can lose precision for extreme demographies.

```{versionadded} 2.0
```

Any two rewards additionally have a joint distribution, from which the {meth}`JointRewardDistribution.marginal <phasegen.distributions.JointRewardDistribution.marginal>` and {meth}`JointRewardDistribution.conditional <phasegen.distributions.JointRewardDistribution.conditional>` distributions of one given the other follow.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = (5, 4)
```

```{code-cell} python
import phasegen as pg
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4)
```

```{code-cell} r
:tags: [remove-output]
library(phasegen)
pg <- load_phasegen()
```

+++
## One-dimensional reward distributions

Consider a single population that passes through a sharp bottleneck. Its {meth}`Coalescent.total_branch_length <phasegen.distributions.Coalescent.total_branch_length>`, the summed length of every branch in the tree, is a distribution whose `pdf`, `cdf` and `quantile` are callable at points and plottable.

```{code-cell} python
coal = pg.Coalescent(
    n=8,
    demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}}),
)

tbl = coal.total_branch_length

print(f"f(1.5)     = {tbl.pdf(1.5):.3f}")
print(f"P(L <= 2)  = {tbl.cdf(2.0):.3f}")
print(f"median     = {tbl.quantile(0.5):.3f}")
```

```{code-cell} python
:tags: [remove-cell]
assert abs(tbl.cdf(tbl.quantile(0.5)) - 0.5) < 1e-4
assert 0 < tbl.cdf(2.0) < 1 and tbl.pdf(1.5) > 0
```

```{code-cell} r
coal <- pg$Coalescent(
    n = 8L,
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 1.0),
        events = c(
            pg$PopSizeChange(pop = "pop_0", time = 0.25, size = 0.08),
            pg$PopSizeChange(pop = "pop_0", time = 0.7, size = 1.0)
        )
    )
)

tbl <- coal$total_branch_length

cat(sprintf("f(1.5)     = %.3f\n", tbl$pdf(1.5)))
cat(sprintf("P(L <= 2)  = %.3f\n", tbl$cdf(2.0)))
cat(sprintf("median     = %.3f\n", tbl$quantile(0.5)))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(tbl$cdf(tbl$quantile(0.5)) - 0.5) < 1e-4, tbl$cdf(2) > 0, tbl$cdf(2) < 1, tbl$pdf(1.5) > 0)
```

+++
The density, the cumulative distribution function and the quantile function of the total branch length:

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

_, axs = plt.subplots(ncols=2, figsize=(7, 3))
tbl.pdf.plot(ax=axs[0], show=False, label='density')
tbl.cdf.plot(ax=axs[0], show=False, clear=False, label='CDF', title='Density and CDF')
tbl.quantile.plot(ax=axs[1], clear=False, title='Quantile function');
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
library(patchwork)

p <- plot(tbl$pdf, label = "density")
p <- plot(tbl$cdf, add = p, label = "CDF", title = "Density and CDF")
p + plot(tbl$quantile, title = "Quantile function")
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4)
```

+++
The `quantile` is the inverse of the `cdf`. Thus `quantile(0.5)` is the median, and `quantile(0.9)` is the branch length exceeded only one time in ten.

+++
## Joint distributions

Any two accumulated rewards have a joint distribution. {meth}`UnfoldedSFSDistribution.joint_distribution <phasegen.distributions.UnfoldedSFSDistribution.joint_distribution>` returns it as a {class}`~phasegen.distributions.JointRewardDistribution` with a 2D `pdf` and `cdf`, computed as described at {class}`JointDensity <phasegen.distributions.JointDensity>` and {class}`JointCDF <phasegen.distributions.JointCDF>` (a joint quantile is not well-defined). As an example, consider two rewards from the bottleneck above, the singleton and doubleton branch lengths (SFS bins 1 and 2). They are not independent. Within a tree, branch length subtending one frequency class reduces that available to the other, so their joint distribution is bimodal and negatively correlated.

```{code-cell} python
joint = coal.sfs.joint_distribution(1, 2)  # singleton and doubleton branch lengths

print(f"P(R_a <= 1.5, R_b <= 0.5) = {joint.cdf(1.5, 0.5):.3f}")
print(f"means (E[R_a], E[R_b])    = {joint.mean.round(3)}")
print(f"correlation               = {joint.corr():.3f}")
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert joint.corr() < 0
assert np.allclose(joint.mean, coal.sfs.mean.data[1:3])
```

```{code-cell} r
joint <- coal$sfs$joint_distribution(1L, 2L)  # singleton and doubleton branch lengths

cat(sprintf("P(R_a <= 1.5, R_b <= 0.5) = %.3f\n", joint$cdf(1.5, 0.5)))
cat("means (E[R_a], E[R_b])    =", paste(round(joint$mean, 3), collapse = " "), "\n")
cat(sprintf("correlation               = %.3f\n", joint$corr()))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(joint$corr() < 0, isTRUE(all.equal(as.numeric(joint$mean), as.numeric(coal$sfs$mean$data[2:3]))))
```

+++
The joint density and the joint cumulative distribution function, drawn as surfaces over the singleton and doubleton branch lengths:

```{code-cell} python
:tags: [remove-cell]
subplot_defaults = {k: matplotlib.rcParams[k] for k in ('figure.subplot.left', 'figure.subplot.right', 'figure.subplot.wspace')}
matplotlib.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.right': 1, 'figure.subplot.wspace': 0})
```

```{code-cell} python
:tags: [full-width]
_, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={'projection': '3d'})
joint.pdf.plot_surface(ax=axs[0], show=False, title='Joint density')
joint.cdf.plot_surface(ax=axs[1], title='Joint CDF');
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams.update(subplot_defaults)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
:tags: [full-width]
par(mfrow = c(1, 2))
persp(joint$pdf, title = "Joint density")
persp(joint$cdf, title = "Joint CDF")
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 5, repr.plot.height = 4)
```

+++
## Marginal distributions

{meth}`JointRewardDistribution.marginal <phasegen.distributions.JointRewardDistribution.marginal>` collapses the joint distribution onto one axis, recovering the ordinary 1D distribution of that reward, here identical to accessing the SFS bin directly via {meth}`UnfoldedSFSDistribution.bin <phasegen.distributions.UnfoldedSFSDistribution.bin>`.

```{code-cell} python
marg = joint.marginal('a')  # branch length of bin 1 (singletons)
print(f"marginal mean = {marg.mean:.4f}   (bin 1 mean = {coal.sfs.bin(1).mean:.4f})")

_, ax = plt.subplots()
marg.pdf.plot(ax=ax, show=False, label='marginal(a)', lw=4, alpha=0.4)
coal.sfs.bin(1).pdf.plot(ax=ax, label='bin(1)', title='Marginal vs. direct bin density', lw=1.5);
```

```{code-cell} python
:tags: [remove-cell]
assert abs(marg.mean - coal.sfs.bin(1).mean) < 1e-6 * coal.sfs.bin(1).mean
```

```{code-cell} r
marg <- joint$marginal("a")  # branch length of bin 1 (singletons)
cat(sprintf("marginal mean = %.4f   (bin 1 mean = %.4f)\n", marg$mean, coal$sfs$bin(1L)$mean))

p <- plot(marg$pdf, label = "marginal(a)", linewidth = 2, alpha = 0.4)
plot(coal$sfs$bin(1L)$pdf, add = p, label = "bin(1)", linewidth = 0.7, title = "Marginal vs. direct bin density")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(marg$mean - coal$sfs$bin(1L)$mean) < 1e-6 * coal$sfs$bin(1L)$mean)
```

+++
## Conditional distributions

{meth}`JointRewardDistribution.conditional <phasegen.distributions.JointRewardDistribution.conditional>` gives the 1D distribution of one reward given the other equals a fixed value, obtained by a nested inversion of the joint Laplace transform as described at {class}`ConditionalRewardDistribution <phasegen.distributions.ConditionalRewardDistribution>`. Because the two rewards are negatively correlated, the doubleton length shifts left as the conditioning singleton length grows. Conditioning on a short singleton branch leaves the doubleton length bimodal, reflecting whether lineages coalesced during or after the bottleneck. Conditioning on a long one collapses it to a single mode.

```{code-cell} python
print("E[R_b | R_a = v]:")
for v in [0.4, 0.9, 1.4]:
    print(f"  v = {v}:  {joint.conditional('a', v).mean:.3f}")
```

```{code-cell} python
:tags: [remove-cell]
means = [joint.conditional('a', v).mean for v in [0.4, 0.9, 1.4]]
assert means[0] > means[1] > means[2]
```

```{code-cell} r
cat("E[R_b | R_a = v]:\n")
for (v in c(0.4, 0.9, 1.4)) {
    cat(sprintf("  v = %.1f:  %.3f\n", v, joint$conditional("a", v)$mean))
}
```

```{code-cell} r
:tags: [remove-cell]
means <- sapply(c(0.4, 0.9, 1.4), function(v) joint$conditional("a", v)$mean)
stopifnot(all(diff(means) < 0))
```

+++
The conditional density of the doubleton branch length, given a short and a long singleton branch:

```{code-cell} python
_, ax = plt.subplots()
for v in [0.5, 1.3]:
    joint.conditional('a', v).pdf.plot(ax=ax, show=False, label=f'R_a = {v}',
                                       title='Conditional density of $R_b$ given $R_a$');
```

```{code-cell} r
p <- NULL
for (v in c(0.5, 1.3)) {
    p <- plot(joint$conditional("a", v)$pdf, add = p, label = sprintf("R_a = %.1f", v),
              title = "Conditional density of R_b given R_a")
}
p
```

+++
## Checking against the sampler

Every one of these objects has an empirical counterpart from {meth}`PhaseTypeDistribution.to_empirical <phasegen.distributions.PhaseTypeDistribution.to_empirical>` (see {doc}`Empirical distributions <empirical_distributions>`), drawn from the same model by Monte Carlo, with the empirical {meth}`EmpiricalPhaseTypeSFSDistribution.joint_distribution <phasegen.distributions.EmpiricalPhaseTypeSFSDistribution.joint_distribution>` exposing the matching {meth}`EmpiricalJointDistribution.marginal <phasegen.distributions.EmpiricalJointDistribution.marginal>` and {meth}`EmpiricalJointDistribution.conditional <phasegen.distributions.EmpiricalJointDistribution.conditional>`. This independently validates the exact results, which is valuable because the Laplace-transform inversion can lose precision for extreme demographies. The sampled conditional is only approximate, since it is estimated by restricting the unconditioned sample to a narrow window around the conditioning value rather than drawn from the conditional law directly, but it is close enough to confirm the exact densities that coincide with it below.

```{code-cell} python
emp = coal.sfs.to_empirical(1_000_000, seed=42).joint_distribution(1, 2)
print(f"correlation:  exact {joint.corr():+.3f}   sampled {emp.corr():+.3f}")

_, ax = plt.subplots()
for v in [0.5, 1.3]:
    joint.conditional('a', v).pdf.plot(ax=ax, show=False, label=f'exact (R_a = {v})')
    emp.conditional('a', v).pdf.plot(ax=ax, show=(v == 1.3), label=f'sampled (R_a = {v})',
                                     title='Conditional density of $R_b$: exact vs sampled')
```

```{code-cell} python
:tags: [remove-cell]
assert abs(emp.corr() - joint.corr()) < 0.01
```

```{code-cell} r
emp <- coal$sfs$to_empirical(1000000L, seed = 42L)$joint_distribution(1L, 2L)
cat(sprintf("correlation:  exact %+.3f   sampled %+.3f\n", joint$corr(), emp$corr()))

p <- NULL
for (v in c(0.5, 1.3)) {
    p <- plot(joint$conditional("a", v)$pdf, add = p, label = sprintf("exact (R_a = %.1f)", v))
    p <- plot(emp$conditional("a", v)$pdf, add = p, label = sprintf("sampled (R_a = %.1f)", v),
              title = "Conditional density of R_b: exact vs sampled")
}
p
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(emp$corr() - joint$corr()) < 0.01)
```
