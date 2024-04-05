"""
Plotting the PDFs for the presentation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-25"

import matplotlib.pyplot as plt

import phasegen as pg

plt.rc('figure', figsize=(3, 2))
plt.rc('xtick', bottom=False, top=False, labelbottom=False)
plt.rc('ytick', left=False, right=False, labelleft=False)


def plot():
    """
    Plot the demography and PDF of a coalescent.

    :param coal: The coalescent to plot.
    """
    # remove axis labels
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')

    for line in plt.gca().lines:
        line.set_linewidth(3)

    plt.show()


def plot_demography(coal: pg.Coalescent, title: str):
    """
    Plot the demography of a coalescent.

    :param coal: The coalescent to plot.
    :param title: The title of the plot.
    """

    coal.demography.plot(show=False, title=title)

    plt.gca().set_ylim(0, 2)

    plot()


def plot_pdf_pg(coal: pg.Coalescent):
    """
    Plot the PDF of a coalescent.

    :param coal: The coalescent to plot.
    """

    coal.tree_height.plot_pdf(show=False, title='Tree height')

    plot()


def plot_pdf_msprime(coal: pg.Coalescent):
    """
    Plot the PDF of a coalescent.

    :param coal: The coalescent to plot.
    """

    coal._to_msprime(
        num_replicates=1000000
    ).tree_height.plot_pdf(show=False, title='Tree height')

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
            pop_sizes={'pop_0': {0: 1, 1: 0.2, 2: 1}}
        )
    ),
)

for name, coal in coals.items():
    plot_demography(coal, title=name)
    plot_pdf_pg(coal)
    plot_pdf_msprime(coal)

pass
