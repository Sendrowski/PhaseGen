import os
from pathlib import Path
from typing import List

def get_filenames(path) -> List[str]:
    """
    Get all filenames in a directory.

    :param path: Path to directory
    :return: Filenames without extension
    """
    return [os.path.splitext(file.name)[0] for file in Path(path).glob('*') if file.is_file()]


configs = get_filenames("resources/configs")

wildcard_constraints:
    opts=r'[^/]*',  # match several optional options not separated by /
    page=r'[^/.]+'  # a User Guide page name

rule all:
    input:
        (
            #expand("results/graphs/MMC_inference/Kingman.{n}.{n_runs}.{n_bootstraps}.{n_bins}.png",n=10,n_runs=20,n_bootstraps=100,n_bins=30),
            #"docs/_build"
            expand("results/comparisons/serialized/{config}.json",config=configs),
            #expand("results/graphs/transitions/{name}.png",name=[
            #    'coalescent_4_lineages_lineage_counting',
            #    'coalescent_5_lineages_lineage_counting',
            #    'coalescent_5_lineages_block_counting',
            #    'migration_2_lineages_lineage_counting',
            #    'migration_3_lineages_lineage_counting',
            #    'migration_3_lineages_block_counting',
            #    'recombination_2_lineages',
            #    'recombination_3_lineages',
            #    'recombination_2_loci_2_pops_3_lineages_lineage_counting',
            #    'beta_coalescent_5_lineages_lineage_counting',
            #    'beta_coalescent_5_lineages_block_counting',
            #    'dirac_coalescent_5_lineages_lineage_counting',
            #    'dirac_coalescent_5_lineages_block_counting',
            #]),
            #"results/graphs/execution_times.png",
            #"results/graphs/state_space_sizes.png",
            #"results/benchmarks/state_space/all.csv",
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

# create comparisons
# Rebuild EVERY comparison fixture from scratch:
#
#   snakemake -F -j8 --use-conda --keep-going regenerate_fixtures
#
# Needed whenever the library changes what is *cached* into a fixture (a new cache_* call, or a change to the meaning
# of a cached curve). A fixture declares only its config YAML as an input, so a library change never invalidates it.
#
# `-F` is required: mtime is not a working trigger here (a config newer than its fixture still reports "Nothing to be
# done"), and a shell `touch` on the configs does not help. Use snakemake's own `--touch` to mark outputs current.
#
# Exactly ONE layer of parallelism. The msprime simulation parallelises internally, so either turn that off and fan out
# over fixtures (PG_PARALLELIZE=0 with -j8, several times faster for this mostly-small-n suite), or keep it and build
# one fixture at a time (-j1). Combining -j8 with the internal parallelism thrashes; combining PG_PARALLELIZE=0 with
# -j1 leaves no parallelism at all and runs the whole suite on a single core.
rule regenerate_fixtures:
    input:
        expand("results/comparisons/serialized/{config}.json", config=configs)

rule create_comparison:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/comparisons/serialized/{config}.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/create_comparison.py"

# Cheaply re-embed a config's comparison tolerances / statistic selection into its EXISTING serialized fixture,
# reusing the cached msprime ground truth (no re-simulation). The fixture is an input and is updated in place; the
# touch-marker output makes snakemake re-run this whenever the config YAML changes (the rerun trigger), so a tolerance
# edit is synced with `snakemake results/comparisons/serialized/.<config>.tolerances_synced`. Use create_comparison
# instead when a simulation parameter changed or a new pairwise surface pair must be cached (the script aborts then).
rule update_tolerances:
    input:
        "resources/configs/{config}.yaml"
    output:
        touch("results/comparisons/serialized/.{config}.tolerances_synced")
    conda:
        "envs/dev.yaml"
    script:
        "scripts/update_tolerances.py"

# create joint-SFS comparisons (the jsfs-specific caching that create_comparison cannot handle)
rule create_jsfs_comparison:
    input:
        "resources/configs/{config}_jsfs.yaml"
    output:
        "results/comparisons/serialized/{config}_jsfs.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/generate_jsfs_fixtures.py"

# prefer the jsfs-specific rule for *_jsfs fixtures (both rules match the same output)
ruleorder: create_jsfs_comparison > create_comparison

# create two-locus-SFS comparisons (the sfs2-specific caching that create_comparison cannot handle)
rule create_2locus_comparison:
    input:
        "resources/configs/{config}_2_locus_sfs.yaml"
    output:
        "results/comparisons/serialized/{config}_2_locus_sfs.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/generate_2locus_fixtures.py"

# prefer the two-locus-specific rule for *_2_locus_sfs fixtures (both rules match the same output)
ruleorder: create_2locus_comparison > create_comparison

def get_scan_fixtures(w):
    """
    Serialized fixtures for the non-slow scenario suite (the single source of truth is
    testing.test_scenarios: all ``configs`` minus ``slow_configs``). Imported lazily so an
    unrelated snakemake target does not pay the phasegen import cost.
    """
    from testing.test_scenarios import configs as scen_configs, slow_configs
    return [f"results/comparisons/serialized/{c}.json" for c in scen_configs if c not in slow_configs]

# render every non-slow scenario's diff plots (Agg, low DPI) and a manifest tying each comparison to its PNG
rule render_scenario_scan:
    input:
        get_scan_fixtures
    output:
        "results/comparisons/scan/manifest.json"
    conda:
        "envs/dev.yaml"
    shell:
        "MPLBACKEND=Agg python scripts/render_scenario_scan.py results/comparisons/scan"

# build the self-contained, click-to-inspect HTML comparison-scan report from the manifest + rendered PNGs
rule scenario_scan_report:
    input:
        "results/comparisons/scan/manifest.json"
    output:
        "results/comparisons/scan/report.html"
    conda:
        "envs/dev.yaml"
    shell:
        "python scripts/build_scan_report.py results/comparisons/scan"

# generate an independent joint-SFS reference using the moments package (runs in the dev env which provides moments)
rule generate_jsfs_reference:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/jsfs_reference/{config}.json"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/generate_jsfs_reference.py"

# benchmark state space creation
rule benchmark_state_space_creation:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/benchmarks/state_space/{config}.csv"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/benchmark_scenario.py"

# merge benchmarks
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
            poetry self add poetry-plugin-export
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
        "scripts/plot_heatmap_execution_times.py"

# plot vectorized-sampling time for the same scenarios (for comparison with the exact-computation times)
rule plot_sampling_time:
    output:
        "results/graphs/sampling_times.png"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/plot_heatmap_sampling_times.py"

# copy a generated graph into the docs image directory (so the docs figures are refreshed automatically)
rule copy_graph_to_docs:
    input:
        "results/graphs/{name}.png"
    output:
        "docs/images/{name}.png"
    shell:
        "cp {input} {output}"

# plot state space sizes
rule plot_state_space_sizes:
    output:
        "results/graphs/state_space_sizes.png"
    conda:
        "envs/dev.yaml"
    script:
        "scripts/plot_heatmap_state_space_sizes.py"

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

# User Guide pages, each written as one source (docs/source/{page}.md) holding the prose and the code of both languages
doc_pages = [p.stem for p in Path("docs/source").glob("*.md")]

# resolution of the User Guide figures in dots per inch
DOCS_FIGURE_DPI = 300

# split a User Guide source into its Python and R notebooks
rule split_page:
    input:
        "docs/source/{page}.md"
    output:
        python="results/docs/Python/{page}.ipynb",
        r="results/docs/R/{page}.ipynb"
    params:
        dpi=DOCS_FIGURE_DPI
    conda:
        "envs/docs.yaml"
    script:
        "docs/split_page.py"

# install the repository's Python package into the User Guide env in editable mode
rule install_python_package:
    input:
        "pyproject.toml",
        [str(p) for p in Path("phasegen").rglob("*.py")]
    output:
        touch("results/docs/Python/phasegen.installed")
    conda:
        "envs/docs.yaml"
    shell:
        "python -m pip install --no-deps -e . > /dev/null"

# install the repository's R package into the User Guide env and register its R kernel inside that env
rule install_r_package:
    input:
        "DESCRIPTION",
        "NAMESPACE",
        [str(p) for p in Path("R").glob("*.R")]
    output:
        touch("results/docs/R/phasegen.installed")
    conda:
        "envs/docs.yaml"
    shell:
        """
        R CMD INSTALL --no-docs . > /dev/null
        Rscript -e 'IRkernel::installspec(name = "ir-phasegen", displayname = "R (phasegen)", user = FALSE,
                                          prefix = Sys.getenv("CONDA_PREFIX"))'
        """

# execute the Python notebook of a page, collapsing the stored stream frames
rule execute_python_page:
    input:
        notebook="results/docs/Python/{page}.ipynb",
        python_installed="results/docs/Python/phasegen.installed"
    output:
        "results/docs/Python/{page}.executed.ipynb"
    conda:
        "envs/docs.yaml"
    shell:
        """
        jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
            --output {wildcards.page}.executed.ipynb {input.notebook}
        python docs/coalesce_streams.py {output}
        """

# execute the R notebook of a page on the Python interpreter of the same env, collapsing the stored stream frames
rule execute_r_page:
    input:
        notebook="results/docs/R/{page}.ipynb",
        python_installed="results/docs/Python/phasegen.installed",
        r_installed="results/docs/R/phasegen.installed"
    output:
        "results/docs/R/{page}.executed.ipynb"
    conda:
        "envs/docs.yaml"
    shell:
        """
        export RETICULATE_PYTHON="$CONDA_PREFIX/bin/python"
        jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
            --output {wildcards.page}.executed.ipynb {input.notebook}
        python docs/coalesce_streams.py {output}
        """

# merge the executed Python and R notebooks of a page into the User Guide page with language tabs
rule merge_page:
    input:
        python="results/docs/Python/{page}.executed.ipynb",
        r="results/docs/R/{page}.executed.ipynb"
    output:
        "docs/reference/{page}.ipynb"
    params:
        dpi=DOCS_FIGURE_DPI
    conda:
        "envs/docs.yaml"
    script:
        "docs/merge_notebooks.py"

# write the outputs displayed from the executed notebook of a page in one language to docs/outputs/{page}
rule extract_page_outputs:
    input:
        "results/docs/{language}/{page}.executed.ipynb"
    output:
        touch("results/docs/{language}/{page}.outputs.written")
    conda:
        "envs/docs.yaml"
    script:
        "docs/extract_outputs.py"

# build all User Guide pages from their sources and write their outputs
rule doc_pages:
    input:
        expand("docs/reference/{page}.ipynb", page=doc_pages),
        expand("results/docs/{language}/{page}.outputs.written", language=["Python", "R"], page=doc_pages)

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

# fit MMC scenario to Kingman coalescent
rule fit_MMC_Kingman:
    output:
        "results/MMC_inference/Kingman.{n}.json"
    conda:
        "envs/dev.yaml"
    params:
        n=lambda w: int(w.n),
        parallelize=True,
    script:
        "scripts/fit_mmc_kingman.py"

# infer demographic history from MMC SFS using SFS only
rule infer_MMC_Kingman_SFS:
    input:
        "results/MMC_inference/Kingman.{n}.json"
    output:
        "results/MMC_inference/Kingman_SFS.{n}.{n_runs}.{n_bootstraps}.json"
    conda:
        "envs/dev.yaml"
    params:
        n=lambda w: int(w.n),
        n_runs=lambda w: int(w.n_runs),
        n_bootstraps=lambda w: int(w.n_bootstraps),
        parallelize=False,
    script:
        "scripts/infer_mmc_kingman_sfs.py"

# infer demographic history from MMC SFS using SFS and 2-SFS
rule infer_MMC_Kingman_2SFS:
    input:
        "results/MMC_inference/Kingman.{n}.json"
    output:
        "results/MMC_inference/Kingman_2SFS.{n}.{n_runs}.{n_bootstraps}.json"
    conda:
        "envs/dev.yaml"
    params:
        n=lambda w: int(w.n),
        n_runs=lambda w: int(w.n_runs),
        n_bootstraps=lambda w: int(w.n_bootstraps),
        parallelize=False,
    script:
        "scripts/infer_mmc_kingman_2sfs.py"

# plot MMC inference
rule plot_MMC_inference:
    input:
        inf_kingman="results/MMC_inference/Kingman.{n}.json",
        fit_sfs="results/MMC_inference/Kingman_SFS.{n}.{n_runs}.{n_bootstraps}.json",
        fit_2sfs="results/MMC_inference/Kingman_2SFS.{n}.{n_runs}.{n_bootstraps}.json"
    output:
        "results/graphs/MMC_inference/Kingman.{n}.{n_runs}.{n_bootstraps}.{n_bins}.png"
    conda:
        "envs/dev.yaml"
    params:
        n_bins=lambda w: int(w.n_bins)
    script:
        "scripts/plot_mmc_inference.py"

# run latexdiff for main.tex
rule run_latexdiff_main:
    input:
        old="reports/manuscripts/old/main.tex",
        new="reports/manuscripts/main/main.tex"
    output:
        main="reports/manuscripts/diff/main.tex"
    conda:
        "latexdiff"
    shell:
        'latexdiff --graphics-markup=none {input.old} {input.new} > {output}'

# run latexdiff for appendix.tex
rule run_latexdiff_appendix:
    input:
        old="reports/manuscripts/old/appendix.tex",
        new="reports/manuscripts/main/appendix.tex"
    output:
        main="reports/manuscripts/diff/appendix.tex"
    conda:
        "latexdiff"
    shell:
        'latexdiff --graphics-markup=none {input.old} {input.new} > {output}'
