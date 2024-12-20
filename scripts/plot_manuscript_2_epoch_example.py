"""
Plot the 2-epoch example for the manuscript.
"""
from matplotlib.gridspec import GridSpec
import phasegen as pg

# -------------------------------
# bottleneck at time 1
coal = pg.Coalescent(
    n={'pop0': 8},
    demography=pg.Demography(
        pop_sizes={
            'pop0': {0: 1, 1: .2}
        }
    )
)
# -------------------------------


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], width_ratios=[0.9, 1])
t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

axs = np.array([[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(2)])

axs[0, 0].imshow(plt.imread("results/graphs/code/2_epoch_example.png"))
axs[0, 0].axis('off')
axs[0, 0].set_title('Python code')

coal.tree_height.plot_pdf(ax=axs[0, 1], show=False)
coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)

plt.tight_layout(pad=0.5)
plt.savefig('reports/manuscripts/merged/figures/2_epoch_example.png', dpi=400)
plt.show()

pass
