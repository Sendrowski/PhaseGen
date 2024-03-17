"""
Run manuscript toy example.
"""

import phasegen as pg

coal = pg.Coalescent(
    n=3,
    demography=pg.Demography(
        pop_sizes={
            'pop_0': {0: 1, 2: 0.5}
        }
    )
)

_ = coal.tree_height.mean
_ = coal.total_branch_length.mean

pass
