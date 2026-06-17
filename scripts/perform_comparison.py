"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

try:
    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    name = "2_epoch_n_4_decline"
    file = f"results/comparisons/serialized/{name}.json"
    out = f"scratch/{name}.json"

import os

from phasegen.comparison import Comparison

comp = Comparison.from_file(file)

# adhere to the *current* YAML config (tolerances + statistic selection): the fixture's embedded copy lags the YAML
# until synced (update_tolerances rule), so overlay the live config here, reusing the cached ground truth (no
# re-simulation). Requires the simulation params to be unchanged and any requested pairwise surfaces to be cached.
config_name = os.path.basename(file).rsplit('.', 1)[0]
yaml_file = f"resources/configs/{config_name}.yaml"
if os.path.exists(yaml_file):
    comp.comparisons = Comparison.from_yaml(yaml_file).comparisons

comp.do_assertion = False
comp.visualize = True

comp.compare(config_name)

pass
