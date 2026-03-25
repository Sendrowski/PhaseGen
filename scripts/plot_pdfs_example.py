"""
Plotting the PDFs for the presentation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-25"

import matplotlib.pyplot as plt
import numpy as np

import phasegen as pg

plt.rc('figure', figsize=(3, 2))
plt.rc('xtick', bottom=False, top=False, labelbottom=False)
plt.rc('ytick', left=False, right=False, labelleft=False)

t = np.linspace(0, 4, 100)

def plot():
    """
    Plot the demography and PDF of a coalescent.
    """
    ax = plt.gca()

    # remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    for line in ax.lines:
        line.set_linewidth(3)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()


def plot_demography(coal: pg.Coalescent, title: str):
    """
    Plot the demography of a coalescent.

    :param coal: The coalescent to plot.
    :param title: The title of the plot.
    """

    coal.demography.plot(show=False, title='', t=t)

    plt.gca().set_ylim(0, 2)

    plot()


def plot_pdf_pg(coal: pg.Coalescent):
    """
    Plot the PDF of a coalescent.

    :param coal: The coalescent to plot.
    """

    coal.tree_height.plot_pdf(
        show=False,
        title='',
        t=t
    )

    plot()


def plot_pdf_msprime(coal: pg.Coalescent):
    """
    Plot the PDF of a coalescent.

    :param coal: The coalescent to plot.
    """

    coal.to_msprime(
        num_replicates=100000,
        parallelize=False
    ).tree_height.plot_pdf(show=False, title='')

    plot()


coals = dict(
    constant=pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1}}
        )
    ),
    variable=pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 1: 0.2, 1.3: 1.5}}
        )
    ),
)

for name, coal in coals.items():
    plot_demography(coal, title=name)
    plot_pdf_pg(coal)
    plot_pdf_msprime(coal)

pass
