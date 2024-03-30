"""
Visualize inference.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-30"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    out_demography = snakemake.output.demography
    out_pop_sizes = snakemake.output.pop_sizes
    out_migration = snakemake.output.migration
    out_bootstraps_hist = snakemake.output.bootstraps_hist
    out_bootstraps_kde = snakemake.output.bootstraps_kde

except NameError:
    # testing
    testing = True
    file = f"scratch/inference.bootstrapped.json"
    out_demography = "scratch/demography.png"
    out_pop_sizes = "scratch/pop_sizes.png"
    out_migration = "scratch/migration.png"
    out_bootstraps_hist = "scratch/bootstraps_hist.png"
    out_bootstraps_kde = "scratch/bootstraps_kde.png"

import phasegen as pg

inf = pg.Inference.from_file(file)

inf.plot_demography()
inf.plot_pop_sizes()
inf.plot_migration()
inf.plot_bootstraps(kind='hist')
inf.plot_bootstraps(kind='kde')

pass
