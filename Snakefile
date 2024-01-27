import os
import re
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


def extract_opt(str: str, name, default_value=None):
    """
    Extract named option from string.
    :param str:
    :param name:
    :param default_value:
    :return:
    """
    # named options have the following signature
    match = re.search(f"[_./-]{name}[_:]([^_./-]*)",str)

    if match:
        return match.groups()[0]

    return default_value


wildcard_constraints:
    opts=r'[^/]*'  # match several optional options not separated by /

rule all:
    input:
        (
            expand("results/comparisons/serialized/{config}.json",config=configs),
            "results/benchmarks/state_space/all.csv"
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

# generate a requirements.txt using poetry
rule generate_requirements_poetry:
    output:
        base="envs/requirements.txt",
        base_snakemake=".snakemake/conda/requirements.txt",
        testing="envs/requirements_testing.txt",
        testing_snakemake=".snakemake/conda/requirements_testing.txt",
        docs="../docs/requirements.txt"
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
            mamba env update -f envs/base.yaml
        """
