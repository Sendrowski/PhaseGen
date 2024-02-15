"""
Merge DPGP3 sequence files into a single file listing
the number of genotypes and minor allele frequencies.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-12-24"

from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    files_in = snakemake.input
    out = snakemake.output[0]
except NameError:
    files_in = [
        'resources/dpgp3/data/2L/ZI104_Chr2L.seq',
        'resources/dpgp3/data/2L/ZI103_Chr2L.seq',
        'resources/dpgp3/data/2L/ZI103_Chr2L.seq',
        'resources/dpgp3/data/2L/ZI103_Chr2L.seq',
        'resources/dpgp3/data/2L/ZI10_Chr2L.seq'
    ]
    out = 'scratch/freqs.csv'

files = [open(f) for f in files_in]

n = len(files)  # number of individuals
n_sites = len(open(files_in[0]).read())

# initialize dataframe
sites = np.zeros((n_sites, 2), dtype=int)

for i in tqdm(range(n_sites)):

    # read site from each file
    site = [f.read(1) for f in files]

    c = Counter(site)

    # number of genotyped sites
    k = n - c['N']
    c.pop('N', None)

    freq = min(c.values()) if len(c) == 2 else 0

    sites[i] = [k, freq]

# convert to dataframe
sites = pd.DataFrame(sites, columns=['k', 'freq'], dtype=int)

sites.to_csv(out, index=False, header=None, sep=' ')
