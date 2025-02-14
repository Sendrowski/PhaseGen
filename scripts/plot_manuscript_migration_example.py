"""
Plot the migration example used in the manuscript.
"""
import fastdfe as fd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# code block in the manuscript
# -----------------------------------------
import phasegen as pg

# migration from pop0 to pop1 back in time
coal = pg.Coalescent(
    n={'pop0': 4, 'pop1': 4},
    demography=pg.Demography(
        pop_sizes={'pop0': 1, 'pop1': 2},
        migration_rates={
            ('pop0', 'pop1'): 0.3
        }
    )
)
# -----------------------------------------


fig = plt.figure(figsize=(7, 6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], width_ratios=[0.9, 1])
t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

axs = np.array([[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(2)])

axs[0, 0].imshow(plt.imread("results/graphs/code/migration_example.png"))
axs[0, 0].axis('off')
axs[0, 0].set_title('Python code')

coal.tree_height.plot_pdf(ax=axs[0, 1], show=False)

fd.Spectra(dict(total=coal.sfs.mean, pop0=coal.sfs.demes['pop0'].mean, pop1=coal.sfs.demes['pop1'].mean)).plot(
    ax=axs[1, 0], show=False, title='Expected SFS'
)

coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations', max_abs=1)

# Add labels A, B, C, D to the plots
for i, ax in enumerate(axs.flat):
    ax.text(-0.05, 1.125, ['A', 'B', 'C', 'D'][i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout(pad=0.5)
plt.savefig('reports/manuscripts/merged/figures/migration_example.png', dpi=400)
plt.show()

pass
