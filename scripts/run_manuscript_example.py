"""
Run the example from the manuscript.
"""

import phasegen as pg

coal = pg.Coalescent(
    n=pg.LineageConfig(3),
    demography=pg.Demography(
        pop_sizes={
            'pop_0': {0: 1, 2: 0.5},
        }
    )
)

_ = coal.total_branch_length.var

pass