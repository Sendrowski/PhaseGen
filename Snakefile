import os
from pathlib import Path
from typing import List

import pandas as pd


def get_filenames(path) -> List[str]:
    """
    Get all filenames in a directory.

    :param path: Path to directory
    :return: Filenames without extension
    """
    return [os.path.splitext(file.name)[0] for file in Path(path).glob('*') if file.is_file()]


configs = get_filenames("resources/configs")

wildcard_constraints:
    opts=r'[^/]*'  # match several optional options not separated by /

rule all:
    input:
        (
            #"results/graphs/inference/demography.png",
            #"docs/_build"
            #expand("results/comparisons/serialized/{config}.json",config=configs),
            expand("results/graphs/transitions/{name}.png",name=[
                'coalescent_4_lineages_lineage_counting',
                'coalescent_5_lineages_lineage_counting',
                'coalescent_5_lineages_block_counting',
                'migration_2_lineages_lineage_counting',
                'migration_3_lineages_lineage_counting',
                'migration_3_lineages_block_counting',
                'recombination_2_lineages',
                'recombination_2_loci_2_pops_3_lineages_lineage_counting',
                'beta_coalescent_5_lineages_lineage_counting',
                'beta_coalescent_5_lineages_block_counting',
            ]),
            #"results/graphs/execution_times.png",
            #"results/graphs/state_space_sizes.png",
            #"results/benchmarks/state_space/all.csv",
            #expand("results/drosophila/2sfs/rice/{chr}/d={d}.folded.txt",chr="2L",d=10),
            #expand("results/drosophila/2sfs/{chr}/n={n}.d={d}.folded.txt",chr="2L",n=[10, 20, 40, 100],d=[10, 100]),
            #expand("results/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.txt",
            #    mu=[1e-6],Ne=[1e4],n=[40],L=[1e6],r=[1e-7],folded=["folded"],d=[100],
            #    model=['standard'], replicate=[1,2,3]),
            #expand("results/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.txt",
            #    mu=[3e-6],Ne=[1e4],n=[40],L=[1e6],r=[3e-7],folded=["folded"],d=[100],
            #    model=['beta.1.8'], replicate=[1,2,3]),
            #expand("results/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.txt",
            #    mu=[1e-4],Ne=[1e4],n=[40],L=[1e6],r=[1e-5],folded=["folded"],d=[100],
            #    model=['beta.1.5'], replicate=[1,2,3]),
            #expand("results/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.txt",
            #    mu=[3e-4],Ne=[1e4],n=[40],L=[1e6],r=[3e-5],folded=["folded"],d=[100],
            #    model=['beta.1.25'], replicate=[1,2,3]),
        )

rule create_comparison:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/comparisons/serialized/{config}.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/create_comparison.py"

rule benchmark_state_space_creation:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/benchmarks/state_space/{config}.csv"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/benchmark_scenario.py"

rule merge_benchmarks:
    input:
        expand("results/benchmarks/state_space/{config}.csv",config=configs)
    output:
        "results/benchmarks/state_space/all.csv"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/merge_benchmarks.py"

# update dependencies
rule update_dependencies:
    output:
        base="envs/requirements.txt",
        base_snakemake=".snakemake/conda/requirements.txt",
        testing="envs/requirements_testing.txt",
        testing_snakemake=".snakemake/conda/requirements_testing.txt",
        docs="docs/requirements.txt"
    conda:
        "envs/build.yaml"
    shell:
        """
            poetry update
            poetry export -f requirements.txt --without-hashes -o {output.base}
            poetry export -f requirements.txt --without-hashes -o {output.base_snakemake}
            poetry export --with dev -f requirements.txt --without-hashes -o {output.testing}
            poetry export --with dev -f requirements.txt --without-hashes -o {output.testing_snakemake}
            poetry export --with dev -f requirements.txt --without-hashes -o {output.docs}
            mamba env update -f envs/dev.yaml
            mamba env update -f envs/testing.yaml
            mamba env update -f envs/base.yaml
        """

# download DPGP3 VCF
rule download_DPGP3_VCF:
    output:
        protected("resources/dpgp3/data/dpgp3_sequences.tar")
    params:
        url="http://pooldata.genetics.wisc.edu/dpgp3_sequences.tar.bz2"
    shell:
        "curl {params.url} -o {output} -L"

# extract chromosome archive from archive
rule extract_DPGP3_chrom_from_archive:
    input:
        "resources/dpgp3/data/dpgp3_sequences.tar"
    output:
        temp("resources/dpgp3/data/{chr}.tar")
    shell:
        "tar -O -zxvf {input} dpgp3_sequences/dpgp3_Chr{wildcards.chr}.tar > {output}"

# extract sequence from chromosome archive
rule extract_DPGP3_chrom:
    input:
        "resources/dpgp3/data/{chr_drosophila}.tar"
    output:
        temp("resources/dpgp3/data/{chr_drosophila}/{name}_Chr{chr_drosophila}.seq")
    shell:
        "tar -O -xvf {input} {wildcards.name}_Chr{wildcards.chr_drosophila}.seq > {output}"


def sample_files_dpgp3(chr: str, n: int) -> List[str]:
    """
    Get files for the n first samples of the DPGP3 data.

    :param n: Number of samples
    :param chr: Chromosome
    :return: List of file names
    """
    names = pd.read_csv(f"resources/rice/data/DPGP3/inversions/noninverted_Chr{chr}.txt",header=None).head(n)[0]

    return expand(rules.extract_DPGP3_chrom.output,name=names,allow_missing=True)


# merge component VCFs
rule merge_sequences_DPGP3:
    input:
        lambda w: sample_files_dpgp3(w.chr_drosophila,int(w.n))
    output:
        "results/drosophila/data/{chr_drosophila}_{n}.csv"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/merge_seqs_dpgp3.py"

# calculate 2-SFS from the data that Rice et al. prepared
rule calculate_2sfs_data_rice:
    input:
        counts="resources/rice/data/DPGP3/minor_allele_counts/Chr{chr}.mac.txt.gz",
        fourfold="resources/rice/data/dmel-4Dsites.txt.gz"
    output:
        data="results/drosophila/2sfs/rice/{chr}/d={d}.{folded}.txt",
        image="results/graphs/drosophila/2sfs/rice/{chr}/d={d}.{folded}.png"
    params:
        n_proj=100,
        d=lambda w: int(w.d),
        filter_4fold=True,
        filter_boundaries=True,
        boundaries={
            '2L': (1e6, 17e6),
            '2R': (6e6, 19e6),
            '3L': (1e6, 17e6),
            '3R': (10e6, 26e6)
        },
        chrom="{chr}",
        folded=lambda w: w.folded == 'folded',
    conda:
        "envs/dev.yaml"
    script:
        "scripts/calculate_2sfs.py"

# calculate 2-SFS from the DPGP3 data
rule calculate_2sfs_data_DPGP3:
    input:
        counts="results/drosophila/data/{chr}_{n}.csv",
        fourfold="resources/rice/data/dmel-4Dsites.txt.gz"
    output:
        data="results/drosophila/2sfs/{chr}/n={n}.d={d}.{folded}.txt",
        image="results/graphs/drosophila/2sfs/{chr}/n={n}.d={d}.{folded}.png"
    params:
        n_proj=lambda w: int(w.n),
        d=lambda w: int(w.d),
        filter_4fold=True,
        filter_boundaries=True,
        boundaries={
            '2L': (1e6, 17e6),
            '2R': (6e6, 19e6),
            '3L': (1e6, 17e6),
            '3R': (10e6, 26e6)
        },
        chrom="{chr}",
        folded=lambda w: w.folded == 'folded',
    conda:
        "envs/dev.yaml"
    script:
        "scripts/calculate_2sfs.py"

# simulate sequence
rule simulate_sequence:
    output:
        data="results/simulations/data/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}.txt",
        info="results/simulations/info/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}.yaml"
    params:
        mu=lambda w: float(w.mu),
        Ne=lambda w: float(w.Ne),
        n=lambda w: float(w.n),
        length=lambda w: float(w.L),
        folded=False,# fold later
        model=lambda w: w.model.split('.')[0],
        alpha=lambda w: float(w.model.split('.',1)[1]) if 'beta' in w.model else None,
        recombination_rate=lambda w: float(w.r)
    conda:
        "envs/dev.yaml"
    script:
        "scripts/simulate_sequence.py"

# calculate 2-SFS from the simulated data
rule calculate_2sfs_simulated:
    input:
        counts="results/simulations/data/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}.txt",
    output:
        data="results/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.txt",
        image="results/graphs/2sfs/simulations/{model}/replicate={replicate}/mu={mu}/Ne={Ne}/n={n}/L={L}/r={r}/{folded}/d={d}.png"
    params:
        n_proj=lambda w: int(w.n),
        d=lambda w: int(w.d),
        filter_4fold=False,
        filter_boundaries=False,
        folded=lambda w: w.folded == 'folded',
        chrom="simulated"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/calculate_2sfs.py"

# plot execution time
rule plot_execution_time:
    output:
        "results/graphs/execution_times.png"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/plot_execution_times.py"

# plot state space sizes
rule plot_state_space_sizes:
    output:
        "results/graphs/state_space_sizes.png"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/plot_state_space_sizes.py"

# plot state space transitions
rule plot_transitions:
    output:
        "results/graphs/transitions/{name}.png"
    params:
        name="{name}"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/plot_transitions.py"

# update the documentation
rule update_docs:
    output:
        directory("docs/_build")
    conda:
        "envs/dev.yaml"
    shell:
        "make html -C docs"

# setup inference
rule setup_inference:
    output:
        "results/inference/inference.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/setup_inference.py"

# run bootstrap
rule run_bootstrap:
    input:
        "results/inference/inference.json"
    output:
        "results/inference/bootstraps/{i}/inference.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/run_bootstrap.py"

# merge bootstraps
rule merge_bootstraps:
    input:
        inference="results/inference/inference.json",
        bootstraps=expand("results/inference/bootstraps/{i}/inference.json",i=range(100))
    output:
        inference="results/inference/inference.bootstrapped.json",
        demography="results/graphs/inference/demography.png",
        pop_sizes="results/graphs/inference/pop_sizes.png",
        migration="results/graphs/inference/migration.png",
        bootstraps_hist="results/graphs/inference/bootstraps_hist.png",
        bootstraps_kde="results/graphs/inference/bootstraps_kde.png",
    conda:
        "envs/dev.yaml"
    script:
        "scripts/merge_bootstraps.py"

# get import times
rule get_import_times:
    output:
        "results/import_times.txt"
    conda:
        "envs/dev.yaml"
    shell:
        "python -X importtime -c 'import phasegen' 2> {output} || true"
