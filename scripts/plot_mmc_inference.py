"""
Plot the results of the MMC inference.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import phasegen as pg

try:
    testing = False
    inf_kingman = snakemake.input.inf_kingman
    fit_sfs = snakemake.input.fit_sfs
    fit_2sfs = snakemake.input.fit_2sfs
    n_bins = snakemake.params.n_bins
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    inf_kingman = "results/MMC_inference/Kingman.10.json"
    fit_sfs = "results/MMC_inference/Kingman_SFS.10.50.1000.json"
    fit_2sfs = "results/MMC_inference/Kingman_2SFS.10.50.1000.json"
    n_bins = 30
    out = "reports/manuscripts/main/figures/2sfs_inference.png"

coal_truth = pg.Coalescent.from_file(inf_kingman)


def get_coal(t: float, Ne: float, alpha: float) -> pg.Coalescent:
    """
    Get coalescent object.

    :param t: Time of population size change
    :param Ne: Effective population size
    :param alpha: Alpha parameter of Beta coalescent
    :return: Coalescent object
    """
    return pg.Coalescent(
        n=coal_truth.lineage_config.n,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        ),
        model=pg.BetaCoalescent(alpha=alpha, scale_time=False)
    )


inf_sfs = pg.Inference.from_file(fit_sfs)
inf_2sfs = pg.Inference.from_file(fit_2sfs)

inf_sfs.get_coal = get_coal
inf_2sfs.get_coal = get_coal

# Define the grid with an extra column for labels
fig = plt.figure(figsize=np.array([14, 12]) * 0.7)  # Wider figure to accommodate labels
gs = GridSpec(7, 4, width_ratios=[0.5, 3, 3, 3], height_ratios=[-0.7, 2, 2.5, 2, 1, 1, 1])  # Extra column for labels

# Create axes for plots
axes = [[fig.add_subplot(gs[row, col + 1]) for col in range(3)] for row in range(1, 7)]  # Skip first column for plots

# Plotting
coal_truth.sfs.mean.normalize().plot(ax=axes[0][0], show=False, title=None)
inf_sfs.dist_inferred.sfs.mean.normalize().plot(ax=axes[0][1], show=False, title=None)
inf_2sfs.dist_inferred.sfs.mean.normalize().plot(ax=axes[0][2], show=False, title=None)
[ax.set_xlabel('') for ax in axes[0]]
y_max = max([ax.get_ylim()[1] for ax in axes[0]])
[ax.set_ylim(0, y_max) for ax in axes[0]]

max_abs = 0.3
coal_truth.sfs.corr.plot(ax=axes[1][0], show=False, max_abs=max_abs)
inf_sfs.dist_inferred.sfs.corr.plot(ax=axes[1][1], show=False, max_abs=max_abs)
inf_2sfs.dist_inferred.sfs.corr.plot(ax=axes[1][2], show=False, max_abs=max_abs)

coal_truth.demography.plot_pop_sizes(ax=axes[2][0], show=False)
inf_sfs.plot_demography(ax=axes[2][1], show=False)
inf_2sfs.plot_demography(ax=axes[2][2], show=False)
[ax.set_ylabel('') for ax in axes[2]]
[ax.set_title('') for ax in axes[2]]
y_max = max([ax.get_ylim()[1] for ax in axes[2]])
[ax.set_ylim(0, y_max) for ax in axes[2]]

# Plot alpha
axes[3][0].hist([coal_truth.model.alpha], bins=n_bins, range=(1, 2))
inf_sfs.bootstraps.alpha.hist(ax=axes[3][1], bins=n_bins, grid=False, density=True)
inf_2sfs.bootstraps.alpha.hist(ax=axes[3][2], bins=n_bins, grid=False, density=True)

# Plot Ne
axes[4][0].axis('off')
axes[4][0].text(0.5, 0.5, '$\\times$', fontsize=20, ha='center', va='center')
inf_sfs.bootstraps.Ne.hist(ax=axes[4][1], bins=n_bins, grid=False, density=True)
inf_2sfs.bootstraps.Ne.hist(ax=axes[4][2], bins=n_bins, grid=False, density=True)

# Plot t
axes[5][0].axis('off')
axes[5][0].text(0.5, 0.5, '$\\times$', fontsize=20, ha='center', va='center')
inf_sfs.bootstraps.t.hist(ax=axes[5][1], bins=n_bins, grid=False, density=True)
inf_2sfs.bootstraps.t.hist(ax=axes[5][2], bins=n_bins, grid=False, density=True)

# Add ylabel axes
label_axes = [fig.add_subplot(gs[row, 0]) for row in range(1, 7)]
row_labels = ['SFS', 'SFS corr', '$N_e(t)$', '$\\alpha$', '$N_1$', '$t_1$']
for ax, label in zip(label_axes, row_labels):
    ax.text(0.5, 0.5, label, fontsize=18, ha='center', va='center', rotation=0)
    ax.axis('off')

# Add xlabel axes
label_axes = [fig.add_subplot(gs[0, col + 1]) for col in range(3)]
col_labels = ['ground truth', 'SFS inference', 'SFS + corr inference']
for ax, label in zip(label_axes, col_labels):
    ax.text(0.5, 0.5, label, fontsize=20, ha='center', va='center', rotation=0)
    ax.axis('off')

# Adjust layout
plt.tight_layout(pad=0)
plt.subplots_adjust(top=0.93)

plt.savefig(out)

if testing:
    plt.show()

pass
