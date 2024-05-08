"""
Visualize the state space for a number of different models.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-10"

import os

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    name = snakemake.params.name
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    name = "coalescent_5_lineages_lineage_counting"
    out = f"scratch/{name}.png"

import phasegen as pg

configs = dict(
    # 5 lineages, one deme, lineage-counting state space
    coalescent_5_lineages_lineage_counting=dict(
        coal=pg.Coalescent(
            n=5
        ),
        state_space_type="lineage_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, 0, 0]).replace('\n', '')
            )
        )
    ),
    # 4 lineages, one deme, lineage-counting state space
    coalescent_4_lineages_lineage_counting=dict(
        coal=pg.Coalescent(
            n=4
        ),
        state_space_type="lineage_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, 0, 0]).replace('\n', '')
            )
        )
    ),
    # 5 lineages, one deme, block-counting state space
    coalescent_5_lineages_block_counting=dict(
        coal=pg.Coalescent(
            n=5
        ),
        state_space_type="block_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, 0, :]).replace('\n', '')
            )
        )
    ),
    # 2 lineages, 2 demes, lineage-counting state space
    migration_2_lineages_lineage_counting=dict(
        coal=pg.Coalescent(
            n={'pop_0': 1, 'pop_1': 1},
            demography=pg.Demography(
                migration_rates={('pop_0', 'pop_1'): {0: 1}, ('pop_1', 'pop_0'): {0: 1}},
            )
        ),
        state_space_type="lineage_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, :, 0])
            )
        )
    ),
    # 2 lineages, 2 demes, block-counting state space
    migration_3_lineages_block_counting=dict(
        coal=pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 1},
            demography=pg.Demography(
                migration_rates={('pop_0', 'pop_1'): {0: 1}, ('pop_1', 'pop_0'): {0: 1}},
            )
        ),
        state_space_type="block_counting",
        plot=dict(
            format_state=lambda s: (
                    str(s[0][0, 0]).replace('\n', '') + '\n' + str(s[0][0, 1]).replace('\n', '')
            )
        )
    ),
    # 2 loci, 2 lineages, lineage-counting state space
    recombination_2_lineages=dict(
        coal=pg.Coalescent(
            n=2,
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        ),
        state_space_type="lineage_counting",
        plot=dict(
            format_state=lambda s: (
                    str(s[0][:, 0, 0]).replace('\n', '') + '\n' + str(s[1][:, 0, 0]).replace('\n', '')
            ),
            ratio=0.8
        )
    ),
    # 2 demes, 3 lineages, 2 loci
    recombination_2_loci_2_pops_3_lineages_lineage_counting=dict(
        coal=pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 1},
            loci=pg.LocusConfig(n=2, recombination_rate=1),
            demography=pg.Demography(
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
        ),
        state_space_type="lineage_counting",
        plot=dict(
            dpi=20,
            ratio=0.7
        )
    ),
    beta_coalescent_5_lineages_lineage_counting=dict(
        coal=pg.Coalescent(
            n=5,
            model=pg.BetaCoalescent(alpha=1.7)
        ),
        state_space_type="lineage_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, 0, 0]).replace('\n', '')
            ),
            ratio=0.8
        )
    ),
    beta_coalescent_5_lineages_block_counting=dict(
        coal=pg.Coalescent(
            n=5,
            model=pg.BetaCoalescent(alpha=1.7)
        ),
        state_space_type="block_counting",
        plot=dict(
            format_state=lambda s: (
                str(s[0][0, 0, :]).replace('\n', '')
            ),
            ratio=0.8
        )
    ),
)

getattr(configs[name]['coal'], configs[name]['state_space_type'] + "_state_space").plot_rates(
    file=os.path.splitext(out)[0],
    extension=os.path.splitext(out)[1].replace('.', ''),
    view=False,
    cleanup=True,
    #background_color='#b4ccfc',
    **configs[name]['plot']
)

pass
