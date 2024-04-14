"""
Run the code example from the manuscript.
"""

from fastdfe import Spectra

import phasegen as pg

coal = pg.Coalescent(
    n=pg.LineageConfig({'pop_0': 3, 'pop_1': 5}),
    model=pg.BetaCoalescent(alpha=1.7),
    demography=pg.Demography(
        pop_sizes={  # dictionary keys are times
            'pop_1': {0: 1.2, 5: 0.1, 5.5: 0.8},
            'pop_0': {0: 1.0}
        },
        migration_rates={
            ('pop_0', 'pop_1'): {0: 0.2, 8: 0.3},
            ('pop_1', 'pop_0'): {0: 0.5}
        }
    )
)

import numpy as np
import matplotlib.pyplot as plt

_, axs = plt.subplots(5, 1, figsize=(3.2, 10))
t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

coal.demography.plot(ax=axs[0], show=False, t=t)
axs[0].legend(prop={'size': 6}, loc='lower left')
coal.tree_height.plot_pdf(ax=axs[1], show=False)

Spectra(dict(
    total=coal.sfs.mean,
    pop_0=coal.sfs.demes['pop_0'].mean,
    pop_1=coal.sfs.demes['pop_1'].mean
)).plot(ax=axs[2], show=False, title='SFS')

coal.sfs.corr.plot(ax=axs[3], show=False, title='2-SFS')

for k in range(1, 4):
    coal.tree_height.plot_accumulation(
        k=k,
        ax=axs[4],
        show=False,
        label=f"$\mu_{{{k}}}$",
        title="Tree height accumulation"
    )

plt.tight_layout()
plt.savefig('scratch/manuscript_basic_example.png', dpi=400)
plt.show()

mean = coal.tree_height.mean
var = coal.total_branch_length.var

q = coal.tree_height.quantile(0.99)
pdf = coal.tree_height.pdf(np.linspace(0, q, 100))
cdf = coal.tree_height.cdf(np.linspace(0, q, 100))

sfs = coal.sfs.mean
sfs2 = coal.sfs.cov

sfs_pop0 = coal.sfs.demes['pop_0'].mean

m3 = coal.moment(3, (pg.TotalBranchLengthReward(),) * 3)

# joint SFS correlation matrix
_ = coal.sfs.demes.corr

pass
