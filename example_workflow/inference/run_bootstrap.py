import phasegen as pg

file = snakemake.input[0]
out = snakemake.output[0]

inf = pg.Inference.from_file(file)

bootstrap = inf.create_bootstrap()

bootstrap.run()

bootstrap.to_file(out)