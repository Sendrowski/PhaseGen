# Custom rewards
In order to compute more complex moments such as higher order (cross)-moments that are not directly made available as cached properties of {class}`~phasegen.distributions.PhaseTypeDistribution`, we can specify our own rewards. A {class}`~phasegen.rewards.Reward` is a means of rewarding or weighting each state so as to obtain the moments of the quantity of interest. Examples of common rewards are {class}`~phasegen.rewards.TreeHeightReward` and {class}`~phasegen.rewards.TotalBranchLengthReward` or {class}`~phasegen.rewards.UnfoldedSFSReward`. We can use {class}`~phasegen.distributions.PhaseTypeDistribution`'s {meth}`~phasegen.distributions.PhaseTypeDistribution.moment`, which requires a tuple of rewards to be specified, whose length equals the order of the moment to be computed.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} python
import numpy as np
import phasegen as pg

coal = pg.Coalescent(n=10)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

```{code-cell} r
library(phasegen)
pg <- load_phasegen()

coal <- pg$Coalescent(n = 10L)
```

+++
We can explicitly compute the moments made available as cached properties. Note that {meth}`~phasegen.distributions.PhaseTypeDistribution.moment` provides centered moments by default.

```{code-cell} python
# mean tree height
coal.tree_height.mean == coal.moment(1, (pg.TreeHeightReward(),))
```

```{code-cell} python
# variance of tree height
coal.moment(2, (pg.TreeHeightReward(),) * 2) == coal.tree_height.var
```

```{code-cell} python
# second non-central moment of total branch length
coal.total_branch_length.m2 == coal.moment(2, (pg.TotalBranchLengthReward(),) * 2, center=False)
```

```{code-cell} python
# mean of the 2nd unfolded SFS entry
np.isclose(coal.sfs.mean.data[2], coal.moment(1, (pg.UnfoldedSFSReward(2),)))
```

```{code-cell} python
:tags: [remove-cell]
assert coal.tree_height.mean == coal.moment(1, (pg.TreeHeightReward(),))
assert coal.moment(2, (pg.TreeHeightReward(),) * 2) == coal.tree_height.var
assert coal.total_branch_length.m2 == coal.moment(2, (pg.TotalBranchLengthReward(),) * 2, center=False)
assert np.isclose(coal.sfs.mean.data[2], coal.moment(1, (pg.UnfoldedSFSReward(2),)))
```

```{code-cell} r
# mean tree height
coal$tree_height$mean == coal$moment(1L, c(pg$TreeHeightReward()))
```

```{code-cell} r
# variance of tree height
coal$moment(2L, c(pg$TreeHeightReward(), pg$TreeHeightReward())) == coal$tree_height$var
```

```{code-cell} r
# second non-central moment of total branch length
coal$total_branch_length$m2 == coal$moment(2L, c(pg$TotalBranchLengthReward(), pg$TotalBranchLengthReward()), center = FALSE)
```

```{code-cell} r
# mean of the 2nd unfolded SFS entry
isTRUE(all.equal(coal$sfs$mean$data[[3]], coal$moment(1L, c(pg$UnfoldedSFSReward(2L)))))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(
    coal$tree_height$mean == coal$moment(1L, c(pg$TreeHeightReward())),
    coal$moment(2L, c(pg$TreeHeightReward(), pg$TreeHeightReward())) == coal$tree_height$var,
    coal$total_branch_length$m2 == coal$moment(2L, c(pg$TotalBranchLengthReward(), pg$TotalBranchLengthReward()), center = FALSE),
    isTRUE(all.equal(coal$sfs$mean$data[[3]], coal$moment(1L, c(pg$UnfoldedSFSReward(2L)))))
)
```

+++
If necessary, we can also compute much higher order (cross)-moments.

```{code-cell} python
# 5th central moment of tree height
coal.moment(5, (pg.TreeHeightReward(),) * 5)
```

```{code-cell} python
# 3rd central cross-moment of 2nd, 3rd and 4th unfolded SFS entries
coal.moment(3, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(3), pg.UnfoldedSFSReward(4)))
```

```{code-cell} r
# 5th central moment of tree height
coal$moment(5L, lapply(1:5, function(x) pg$TreeHeightReward()))
```

```{code-cell} r
# 3rd central cross-moment of 2nd, 3rd and 4th unfolded SFS entries
coal$moment(3L, c(pg$UnfoldedSFSReward(2L), pg$UnfoldedSFSReward(3L), pg$UnfoldedSFSReward(4L)))
```

+++
## Combining rewards
Sometimes we may want to combine multiple rewards. To this end, we can use {class}`~phasegen.rewards.ProductReward` or {class}`~phasegen.rewards.SumReward`. As an example, we compute the mean SFS over the first 2 out of 3 demes, that is, the branch lengths of each frequency class accumulated while the lineages reside in the first two demes.

```{code-cell} python
# 3-deme coalescent with symmetric migration
coal = pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 2, 'pop_2': 0},
    demography=pg.Demography(
        pop_sizes={'pop_0': 3, 'pop_1': 0.5, 'pop_2': 0.1},
        events=[
            pg.SymmetricMigrationRateChanges(
                pops=['pop_0', 'pop_1', 'pop_2'],
                rate=1
            )
        ]
    )
)
```

```{code-cell} python
sfs = coal.sfs.moment(1, (pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')]),))

sfs.plot();
```

```{code-cell} r
# 3-deme coalescent with symmetric migration
coal <- pg$Coalescent(
    n = list(pop_0 = 3L, pop_1 = 2L, pop_2 = 0L),
    demography = pg$Demography(
        pop_sizes = list(pop_0 = 3, pop_1 = 0.5, pop_2 = 0.1),
        events = c(
            pg$SymmetricMigrationRateChanges(
                pops = c('pop_0', 'pop_1', 'pop_2'),
                rate = 1
            )
        )
    )
)
```

```{code-cell} r
sfs <- coal$sfs$moment(1L, c(pg$SumReward(c(pg$DemeReward('pop_0'), pg$DemeReward('pop_1')))))

plot(sfs)
```

+++
Plotting the marginal spectra for each deme, we indeed see that the mean SFS over the first two demes was obtained above.

```{code-cell} python
pg.Spectra({d: coal.sfs.demes[d].mean for d in coal.demography.pop_names}).plot();
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(sfs.data, coal.sfs.demes['pop_0'].mean.data + coal.sfs.demes['pop_1'].mean.data)
```

```{code-cell} r
spectra <- pg$Spectra(setNames(
    lapply(coal$demography$pop_names, function(d) coal$sfs$demes[[d]]$mean),
    coal$demography$pop_names
))

plot(spectra)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(sfs$data, coal$sfs$demes$pop_0$mean$data + coal$sfs$demes$pop_1$mean$data)))
```

+++
Note that we have here used {meth}`~phasegen.distributions.UnfoldedSFSDistribution.moment` of {class}`~phasegen.distributions.UnfoldedSFSDistribution`. This method internally uses a {class}`~phasegen.rewards.CombinedReward` to combine the given reward with {class}`~phasegen.rewards.UnfoldedSFSReward` to obtain the SFS for all possible SFS bins. Unlike a plain {class}`~phasegen.rewards.ProductReward`, it treats deme rewards as a restriction to the demes the lineages of each frequency class reside in. Had we used {meth}`~phasegen.distributions.Coalescent.moment` instead, we would have needed to specify the reward for each SFS bin separately.

```{code-cell} python
demes = pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')])
sfs_bin = pg.UnfoldedSFSReward(2)

# restrict the SFS reward for the second SFS bin to the first two demes
sfs.data[2] == coal.moment(1, (pg.CombinedReward([demes, sfs_bin]),))
```

```{code-cell} python
:tags: [remove-cell]
assert np.isclose(sfs.data[2], coal.moment(1, (pg.CombinedReward([demes, sfs_bin]),)))
```

```{code-cell} r
demes <- pg$SumReward(c(pg$DemeReward('pop_0'), pg$DemeReward('pop_1')))
sfs_bin <- pg$UnfoldedSFSReward(2L)

# restrict the SFS reward for the second SFS bin to the first two demes
sfs$data[[3]] == coal$moment(1L, c(pg$CombinedReward(c(demes, sfs_bin))))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(sfs$data[[3]], coal$moment(1L, c(pg$CombinedReward(c(demes, sfs_bin)))))))
```

+++
## Tracing reward accumulation over time
Since rewards are accumulated over time, we can trace their accumulation over time. This can be useful for debugging purposes or to understand how the reward is accrued over time. To this end, we can use {class}`~phasegen.distributions.PhaseTypeDistribution`'s {meth}`~phasegen.distributions.PhaseTypeDistribution.accumulate` and {meth}`~phasegen.distributions.PhaseTypeDistribution.plot_accumulation`. Note that this is different from the CDF, which instead accumulates probability mass over time.

```{code-cell} python
# accumulation of mean SFS over all demes
coal.sfs.plot_accumulation(k=1);
```

```{code-cell} r
# accumulation of mean SFS over all demes
plot_accumulation(coal$sfs, k = 1L)
```

+++
## Adjusting start and end times of reward accumulation
By default, rewards are accumulated from time 0 until time of almost sure absorption. We can adjust the start and end times of reward accumulation by specifying the `start_time` and `end_time` arguments to
{class}`~phasegen.distributions.Coalescent` or {class}`~phasegen.distributions.PhaseTypeDistribution`'s {meth}`~phasegen.distributions.PhaseTypeDistribution.moment`.

```{code-cell} python
mean1 = pg.Coalescent(n=10, end_time=2).tree_height.mean
mean2 = pg.Coalescent(n=10).tree_height.moment(1, end_time=2)

mean1 == mean2
```

```{code-cell} python
:tags: [remove-cell]
assert mean1 == mean2
assert mean1 < pg.Coalescent(n=10).tree_height.mean
```

```{code-cell} r
mean1 <- pg$Coalescent(n = 10L, end_time = 2)$tree_height$mean
mean2 <- pg$Coalescent(n = 10L)$tree_height$moment(1L, end_time = 2)

mean1 == mean2
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(mean1 == mean2, mean1 < pg$Coalescent(n = 10L)$tree_height$mean)
```

+++
## Distributions of custom rewards

Each reward above was summarised by its moments, but any reward also has a full distribution. {meth}`~phasegen.distributions.Coalescent.distribution` returns the 1D law of a reward, with its `pdf`, `cdf` and `quantile`, and {meth}`~phasegen.distributions.Coalescent.joint` the joint law of two, reaching combinations with no dedicated accessor. See {doc}`Distribution functions <distribution_functions>` for these objects in general.

```{code-cell} python
king = pg.Coalescent(n=8)

# full distribution of the combined singleton and doubleton branch length
combined = king.distribution(pg.SumReward([pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)]))
print(f"mean = {combined.mean:.3f}, median = {combined.quantile(0.5):.3f}")

combined.pdf.plot(title='Combined singleton + doubleton branch length');
```

```{code-cell} python
:tags: [remove-cell]
assert abs(combined.mean - (king.sfs.mean.data[1] + king.sfs.mean.data[2])) < 1e-8
assert abs(combined.cdf(combined.quantile(0.5)) - 0.5) < 1e-4
```

```{code-cell} r
king <- pg$Coalescent(n = 8L)

# full distribution of the combined singleton and doubleton branch length
combined <- king$distribution(pg$SumReward(c(pg$UnfoldedSFSReward(1L), pg$UnfoldedSFSReward(2L))))
cat(sprintf("mean = %.3f, median = %.3f\n", combined$mean, combined$quantile(0.5)))

plot(combined$pdf, title = "Combined singleton + doubleton branch length")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(
    abs(combined$mean - sum(king$sfs$mean$data[2:3])) < 1e-8,
    abs(combined$cdf(combined$quantile(0.5)) - 0.5) < 1e-4
)
```

+++
The joint distribution of two rewards follows the same way. Here the tree height and the singleton branch length, two different statistics of the same genealogy, are positively correlated.

```{code-cell} python
:tags: [remove-cell]
subplot_defaults = {k: matplotlib.rcParams[k] for k in ('figure.subplot.left', 'figure.subplot.right', 'figure.subplot.wspace')}
matplotlib.rcParams.update({'figure.subplot.left': 0, 'figure.subplot.right': 1, 'figure.subplot.wspace': 0})
```

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

joint = king.joint(pg.TreeHeightReward(), pg.UnfoldedSFSReward(1))
print(f"correlation = {joint.corr:.3f}")

_, axs = plt.subplots(ncols=2, figsize=(7, 3.4), subplot_kw={'projection': '3d'})
joint.pdf.plot_surface(ax=axs[0], show=False, title='Joint density')
joint.cdf.plot_surface(ax=axs[1], title='Joint CDF');
```

```{code-cell} python
:tags: [remove-cell]
matplotlib.rcParams.update(subplot_defaults)
```

```{code-cell} python
:tags: [remove-cell]
assert joint.corr > 0
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.4)
```

```{code-cell} r
:tags: [full-width]
joint <- king$joint(pg$TreeHeightReward(), pg$UnfoldedSFSReward(1L))
cat(sprintf("correlation = %.3f\n", joint$corr))

par(mfrow = c(1, 2))
persp(joint$pdf, title = "Joint density")
persp(joint$cdf, title = "Joint CDF")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(joint$corr > 0)
```
