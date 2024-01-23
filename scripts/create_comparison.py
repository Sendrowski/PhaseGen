"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

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
    name = "1_epoch_n_2_test_size"
    file = f"resources/configs/{name}.yaml"
    parallelize = False
    out = f"scratch/{name}.json"

from phasegen.comparison import Comparison

c = Comparison.from_yaml(file)

if testing:
    c.parallelize = parallelize

# touch msprime stats to cache them
c.ms.touch()

# drop simulated data
c.ms.drop()

c.to_file(out)

pass
