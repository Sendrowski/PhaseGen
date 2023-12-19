import phasegen as pg

# construct coalescent object
coal = pg.Coalescent(
    n=10,
    demography=pg.PiecewiseConstantDemography(
        pop_sizes={0: 1.2, 0.3: 10, 1: 0.8, 1.4: 10}
    )
)

from matplotlib import pyplot as plt

_, axs = plt.subplots(ncols=2, figsize=(8, 4))

coal.tree_height.plot_pdf(ax=axs[0], show=False)
coal.tree_height.plot_cdf(ax=axs[1])

coal.sfs.cov.plot()

coal.sfs.cov.plot_heatmap()

pass
