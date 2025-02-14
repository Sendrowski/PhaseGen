"""
Plot the recombination example for the manuscript.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import phasegen as pg


# code block in the manuscript
# noinspection all
# -------------------------------------------
def get_cov(r, N1):
    return pg.Coalescent(
        n={'pop0': 2},
        loci=pg.LocusConfig(
            n=2, recombination_rate=r / 2
        ),
        demography=pg.Demography(
            pop_sizes={'pop0': {0: 1, 1: N1}}
        )
    ).tree_height.loci.cov[0, 1]


# -------------------------------------------

def get_coal(r: float, N1: float) -> pg.Coalescent:
    """
    Get the coalescent for a 2-epoch model with
    two loci and a given recombination rate.
    """
    return pg.Coalescent(
        n={'pop0': 2},
        loci=pg.LocusConfig(
            n=2, recombination_rate=r / 2
        ),
        demography=pg.Demography(
            pop_sizes={'pop0': {0: 1, 1: N1}}
        )
    )


fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

axs = np.array([[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(2)])

axs[0, 0].imshow(plt.imread("results/graphs/code/recombination_example.png"))
axs[0, 0].axis('off')
axs[0, 0].set_title('Python code')

Ns = np.logspace(-1, 1, 5)
rs = np.linspace(0, 3, 2000)
for N1 in Ns:
    axs[0, 1].plot(rs, get_coal(10, N1).tree_height.pdf(rs), label=f'$N_1=${N1:.1f}'.rstrip('0').rstrip('.'), alpha=0.6)
axs[0, 1].legend(prop={'size': 8})
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title('Coalescent time', pad=6)
axs[0, 1].set_xticks(np.arange(0, 4))
# axs[0, 1].set_box_aspect(1)

rs = np.logspace(-1, 1, 10)
for N1 in Ns:
    axs[1, 0].plot(rs, [get_cov(r, N1) for r in rs], label=f'$N_1=${N1:.1f}'.rstrip('0').rstrip('.'))
axs[1, 0].set_xlabel('$\\rho$', labelpad=0)
axs[1, 0].set_ylabel('cov', labelpad=0)
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')
axs[1, 0].legend(prop={'size': 8})
# axs[1, 0].set_box_aspect(1)
axs[1, 0].set_title('Covariance', pad=6)

rs = np.logspace(-1, 1, 10)
for N1 in Ns:
    axs[1, 1].plot(rs, [get_coal(r, N1).tree_height.loci.corr[0, 1] for r in rs],
                   label=f'$N_1=${N1:.1f}'.rstrip('0').rstrip('.'))
axs[1, 1].set_xlabel('$\\rho$', labelpad=0)
axs[1, 1].set_ylabel('corr', labelpad=0)
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].legend(prop={'size': 8})
# axs[1, 1].set_box_aspect(1)
axs[1, 1].set_title('Correlation', pad=6)

# Add labels A, B, C, D to the plots
for i, ax in enumerate(axs.flat):
    ax.text(-0.05, 1.125, ['A', 'B', 'C', 'D'][i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout(pad=1)
plt.savefig('reports/manuscripts/merged/figures/recombination_example.png', dpi=400)
plt.show()

pass
