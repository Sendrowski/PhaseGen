"""
Run the code example from the manuscript.
"""
import pandas as pd
from matplotlib import pyplot as plt

import phasegen as pg

coal = pg.Coalescent(
    n=pg.LineageConfig({'pop_0': 3, 'pop_1': 5}),
    model=pg.BetaCoalescent(alpha=1.7),
    demography=pg.Demography(
        pop_sizes={'pop_1': 1, 'pop_0': 2},
        migration_rates={
            ('pop_0', 'pop_1'): 1,
            ('pop_1', 'pop_0'): 2
        }
    )
)

it = coal.sfs.get_mutation_configs(theta=0.1)

df = pd.DataFrame(pg.takewhile_inclusive(
    lambda _: coal.sfs.generated_mass < 0.8, it)
)

df.plot(
    kind='bar', x=0, legend=False, ylabel='mass',
    xlabel='config', figsize=(3, 4)
)

plt.tight_layout()
plt.show()

pass
