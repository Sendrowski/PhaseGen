from matplotlib import pyplot as plt

import phasegen as pg

# construct coalescent object
coal = pg.PiecewiseTimeHomogeneousCoalescent(
    n=10,
    demography=pg.PiecewiseTimeHomogeneousDemography(
        pop_sizes=[1.2, 10, 0.8, 10],
        times=[0, 0.3, 1, 1.4]
    )
)

coal.sfs.cov.plot()

coal.sfs.cov.plot_heatmap()

_, axs = plt.subplots(ncols=2, figsize=(10, 5))

coal.tree_height.plot_pdf(ax=axs[0], show=False)
coal.tree_height.plot_cdf(ax=axs[1])

pass