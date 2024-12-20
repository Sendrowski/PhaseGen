"""
Run manuscript cross moment example.
"""
import numpy as np
from matplotlib import pyplot as plt

import phasegen as pg


def get_coal(N1: float, N2: float, t: float) -> pg.Coalescent:
    """
    Get the 2-epoch coalescent.

    :param N1: First population size.
    :param N2: Second population size.
    :param t: Time of population size change.
    :return: Coalescent.
    """
    return pg.Coalescent(
        n=3,
        demography=pg.Demography(
            pop_sizes={
                'pop_0': {0: N1, t: N2}
            }
        ),
        regularize=False
    )


def get_moment(coal: pg.Coalescent) -> float:
    """
    Get the singleton-doubleton moment.

    :param coal: Coalescent.
    :return: The moment.
    """
    return coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)])



N1 = 1
N2 = 1
t = 2

coal = get_coal(N1=N1, N2=N2, t=t)

m12 = coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)], center=False, permute=False)
m21 = coal.moment(k=2, rewards=[pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(1)], center=False, permute=False)
m1 = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)], center=False, permute=False)
m2 = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(2)], center=False, permute=False)

# center and permute
m = (m12 + m21) / 2 - m1 * m2

m_expected = get_moment(coal)

assert m == m_expected

t_max = 3
alpha = 1
fig = plt.figure(figsize=(3.7, 2.5))
x = np.linspace(0.01, t_max, 100)

plt.plot(x, [get_moment(get_coal(N1=N1, N2=N2, t=t)) for N1 in x], label='$x=N_1$', alpha=alpha)
plt.plot(x, [get_moment(get_coal(N1=N1, N2=N2, t=t)) for N2 in x], label='$x=N_2$', alpha=alpha)
plt.plot(x, [get_moment(get_coal(N1=N1, N2=N2, t=t)) for t in x], label='$x=\delta_1$', alpha=alpha)
plt.plot(
    x,
    coal.accumulate(k=2, rewards=[pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)], end_times=x),
    label='$x=t_{max}$',
    alpha=alpha
)

plt.plot([0, t_max], [0, 0], linestyle='dashed', lw=0.7, color='black', alpha=0.8)
plt.xlabel('x')
plt.ylabel('$\sigma_{12}$', rotation=0, labelpad=12.5)
plt.legend(prop={'size': 8})
plt.title('Singleton/doubleton covariance', fontsize=10)
plt.tight_layout()
plt.savefig('reports/manuscripts/merged/figures/cross_moment_example_plot.png', dpi=400)
plt.show()

pass
