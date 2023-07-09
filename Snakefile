import re

configs = [
    'standard_coalescent_ph_n_4',
    'standard_coalescent_ph_const_n_4',
    '2_epoch_n_2',
    'rapid_decline_n_2',
    'rapid_decline_n_5',
    '4_epoch_up_down_n_10',
    '4_epoch_up_down_n_2',
    '3_epoch_extreme_bottleneck_n_5'
]


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
            expand("results/comparisons/serialized/{config}.json",config=configs)
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
        """