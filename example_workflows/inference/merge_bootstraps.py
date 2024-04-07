# TODO test workflow

import phasegen as pg

inf_file = snakemake.input.inference
bootstraps_file = snakemake.input.bootstraps
out = snakemake.output.inference
out_demography = snakemake.output.demography
out_bootstraps = snakemake.output.bootstraps

inf = pg.Inference.from_file(inf_file)

inf.add_bootstraps([pg.Inference.from_file(f) for f in bootstraps_file])

inf.plot_demography(file=out_demography)
inf.plot_bootstraps(file=out_bootstraps)

inf.to_file(out)
