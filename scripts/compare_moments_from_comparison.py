"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-03-11"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    name = snakemake.params.name
    dist = snakemake.params.dist
    stat = snakemake.params.stat
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    name = "1_epoch_migration_disparate_pop_size_one_all_n_2"
    file = f"results/comparisons/serialized/{name}.json"
    dist = 'sfs'
    stat = 'mean'
    out = f"scratch/{name}.json"

from phasegen.comparison import Comparison

comp = Comparison.from_file(file)

comp.comparisons = {
    'types': ['ph'],
    'tolerance': {
        dist: {
            stat: 0.1
        }
    }
}

comp.compare(name, do_assertion=False)

pass
