"""
Visualize the transition matrix of a scenario.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-01-27"

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
    name = "1_epoch_2_loci_2_pops_n_3_r_1"
    file = f"resources/configs/{name}.yaml"
    out = f"scratch/{name}.csv"

from phasegen.comparison import Comparison

c = Comparison.from_yaml(file)

c.ph.default_state_space.plot_rates(out)

pass
