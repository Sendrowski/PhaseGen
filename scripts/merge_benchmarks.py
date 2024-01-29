"""
Merge benchmark results into a single CSV file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-01-22"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    files = snakemake.input
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    files = ["scratch/1_epoch_n_2_test_size.csv", "scratch/1_epoch_n_2_test_size.csv"]
    out = f"scratch/benchmarks.csv"

import pandas as pd

df = pd.concat([pd.read_csv(f) for f in files])

df.to_csv(out, index=False)

pass