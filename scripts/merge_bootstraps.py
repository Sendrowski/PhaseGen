"""
Merge bootstraps.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-30"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    inf_file = snakemake.input.inference
    bootstraps_file = snakemake.input.bootstraps
    out = snakemake.output.inference
    out_demography = snakemake.output.demography
    out_pop_sizes = snakemake.output.pop_sizes
    out_migration = snakemake.output.migration
    out_bootstraps_hist = snakemake.output.bootstraps_hist
    out_bootstraps_kde = snakemake.output.bootstraps_kde
except NameError:
    # testing
    testing = True
    inf_file = f"scratch/inference.json"
    bootstraps_file = [f"scratch/bootstrap.json"]
    out = f"scratch/inference.bootstrapped.json"
    out_demography = "scratch/demography.png"
    out_pop_sizes = "scratch/pop_sizes.png"
    out_migration = "scratch/migration.png"
    out_bootstraps_hist = "scratch/bootstraps_hist.png"
    out_bootstraps_kde = "scratch/bootstraps_kde.png"

import phasegen as pg

inf = pg.Inference.from_file(inf_file)

inf.add_bootstraps([pg.Inference.from_file(f) for f in bootstraps_file])

inf.plot_demography(file=out_demography, show=testing)
inf.plot_pop_sizes(file=out_pop_sizes, show=testing)
inf.plot_migration(file=out_migration, show=testing)
inf.plot_bootstraps(kind='hist', file=out_bootstraps_hist, show=testing)
inf.plot_bootstraps(kind='kde', file=out_bootstraps_kde, show=testing)

inf.to_file(out)

pass



