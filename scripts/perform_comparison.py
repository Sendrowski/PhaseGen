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
    name = "2_epoch_varying_migration_low_coalescence"
    file = f"results/comparisons/serialized/{name}.json"
    out = f"scratch/{name}.json"

from phasegen.comparison import Comparison

comp = Comparison.from_file(file)

comp.compare(file.split('/')[-1].split('.')[0], do_assertion=False)

pass
