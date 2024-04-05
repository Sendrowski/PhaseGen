"""
Plot discretization schematic.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_population_growth(x, y, plot_type='line'):
    """
    Plot population growth.

    :param x:
    :param y:
    :param plot_type:
    :return:
    """
    plt.figure(figsize=(1.5, 1.5))
    if plot_type == 'line':
        plt.plot(x, y, linewidth=3)
    elif plot_type == 'step':
        plt.step(x, y, linewidth=3)
    plt.xlabel('t')
    plt.xticks([])
    plt.yticks([])
    plt.title('Demography')
    plt.tight_layout()
    plt.show()


def get_pop_size(t: np.ndarray) -> np.ndarray:
    """
    Get the population size at time t.

    :param t:
    :return:
    """
    return -np.exp(t)


x = np.linspace(0, 3, 100)
y = get_pop_size(x)
plot_population_growth(x, -y, 'line')

x = np.linspace(0, 3, 10)
y = get_pop_size(x)
plot_population_growth(x, -y, 'step')
