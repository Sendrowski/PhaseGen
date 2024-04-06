"""
Run the code example from the manuscript.
"""

import numpy as np

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
    ),
    parallelize=False
)

import matplotlib.pyplot as plt

_, axs = plt.subplots(2, 2, figsize=(6, 5))
t = np.linspace(0, coal.tree_height.quantile(0.99), 100)

coal.demography.plot(ax=axs[0, 0], show=False, t=t)
axs[0, 0].legend(
    prop={'size': 5.5},
    loc='lower left',
    framealpha=0.35
)

coal.tree_height.plot_pdf(ax=axs[0, 1], show=False)
coal.sfs.mean.plot(ax=axs[1, 0], show=False, title='SFS')
coal.sfs.corr.plot(ax=axs[1, 1], show=False, title='2-SFS')

plt.tight_layout()
plt.savefig('scratch/poster_basic_example.png', dpi=400, transparent=True)
plt.show()

pass
