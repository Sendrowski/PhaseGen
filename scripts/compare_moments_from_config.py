"""
Compare moments of msprime and ph.
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
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "resources/configs/test_moments_height_standard_coalescent.yaml"
    out = "scratch/test_moments_height_standard_coalescent.json"

import json

import yaml

from PH import Comparison
from comp import diff_rel_max_abs

# load config from file
with open(file, 'r') as f:
    config = yaml.safe_load(f)

s = Comparison(**config['config'])

result = {}

# assign results
for metric in ["tree_height", "total_branch_length"]:
    result[metric] = {}

    for stat in ["mean", "var"]:
        result[metric][stat] = {}

        for sim in ["msprime", "ph"]:
            result[metric][stat][sim] = getattr(getattr(s, sim), metric).__getattribute__(stat)

        result[metric][stat]["diff_rel_max_abs"] = diff_rel_max_abs(
            result[metric][stat]["msprime"], result[metric][stat]["ph"]
        )

# save results
with open(out, "w") as f:
    json.dump(dict(config=config, result=result), f, indent=4)

# check results
for k1, v1 in config['tolerance'].items():
    for k2, v2 in v1.items():
        if not result[k1][k2]["diff_rel_max_abs"] < v2:
            raise AssertionError(f"{result[k1][k2]['diff_rel_max_abs']} not less than {v2}.")
