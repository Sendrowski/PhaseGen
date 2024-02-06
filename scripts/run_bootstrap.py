"""
Run a single bootstrap sample.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-01-29"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = f"scratch/bootstrap.json"
    out = f"scratch/bootstrap.run.json"

import phasegen as pg

inf = pg.Inference.from_file(file)

inf.run()

inf.to_file(out)

pass



