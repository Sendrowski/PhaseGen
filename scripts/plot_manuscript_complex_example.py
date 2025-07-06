"""
Plot the complex example used in the manuscript.
"""

import phasegen as pg

# first code block in the manuscript
# ----------------------------------------------------------
coal = pg.Coalescent(
    n=pg.LineageConfig({'pop0': 3, 'pop1': 5}),
    model=pg.BetaCoalescent(alpha=1.7),
    demography=pg.Demography(
        pop_sizes={
            'pop0': {0: 1.0},
            'pop1': {0: 1.2, 5: 0.1, 5.5: 0.8}
        },
        migration_rates={
            ('pop0', 'pop1'): {0: 0.2, 8: 0.3},
            ('pop1', 'pop0'): {0: 0.5}
        }
    )
)
# ----------------------------------------------------------

# second code block in the manuscript
# ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

_, axs = plt.subplots(2, 2, figsize=(7, 6))
t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

coal.demography.plot(ax=axs[0, 0], show=False, t=t)
axs[0, 0].legend(prop={'size': 6}, loc='center left')
coal.tree_height.plot_pdf(ax=axs[0, 1], show=False)
coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='Expected SFS')
coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='SFS correlations')

# Add labels A, B, C, D to the plots
for i, ax in enumerate(axs.flat):
    ax.text(-0.05, 1.125, ['A', 'B', 'C', 'D'][i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
# plt.show()
# ----------------------------------------------------------

plt.savefig('reports/manuscripts/main/figures/complex_example.png', dpi=400)
plt.show()
pass

mean = coal.tree_height.mean
var = coal.total_branch_length.var

q = coal.tree_height.quantile(0.99)
pdf = coal.tree_height.pdf(np.linspace(0, q, 100))
cdf = coal.tree_height.cdf(np.linspace(0, q, 100))

sfs = coal.sfs.mean
sfs2 = coal.sfs.cov

sfs_pop0 = coal.sfs.demes['pop0'].mean

m3 = coal.moment(3, (pg.TotalBranchLengthReward(),) * 3)
