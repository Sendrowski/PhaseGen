import re

import numpy as np

configs_comp = [
    'test_moments_height_scenario_1',
    'test_moments_height_scenario_2',
    'test_moments_height_scenario_larger_n',
    'test_moments_height_standard_coalescent',
    'test_moments_height_standard_coalescent_high_Ne',
    'test_moments_height_standard_coalescent_low_Ne'
]

configs_plot = [
    'test_plot_cdf_const_total_branch_length',
    'test_plot_cdf_const_tree_height',
    'test_plot_cdf_const_tree_height_large_Ne',
    'test_plot_cdf_var_total_branch_length',
    'test_plot_cdf_var_total_branch_length_larger_n',
    'test_plot_cdf_var_tree_height',
    'test_plot_cdf_var_tree_height_larger_n',
    'test_plot_pdf_const_total_branch_length',
    'test_plot_pdf_const_tree_height',
    'test_plot_pdf_var_total_branch_length',
    'test_plot_pdf_var_total_branch_length_larger_n',
    'test_plot_pdf_var_tree_height',
    'test_plot_pdf_var_tree_height_n_3',
    'test_plot_pdf_var_tree_height_n_4',
    'test_plot_pdf_var_tree_height_n_5',
    'test_plot_cdf_var_total_branch_length_n_3',
    'test_plot_cdf_var_total_branch_length_n_4',
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
            "results/comp_run.txt",
            "results/comp_plotted.txt"
        )

rule run_comparisons:
    input:
        expand("results/comparisons/configs/{config}.json",config=configs_comp),
    output:
        touch("results/comp_run.txt")

rule run_plots:
    input:
        expand("results/comparisons/graphs/{config}.png",config=configs_plot)
    output:
        touch("results/comp_plotted.txt")

rule compare_moments_from_config:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/comparisons/configs/{config}.json"
    conda:
        "envs/base.yaml"
    script:
        "scripts/compare_moments_from_config.py"

rule compare_moments:
    output:
        "results/comparisons/opts/{opts}.json"
    params:
        n=lambda w: extract_opt(w.opts,"n",2),
        times=lambda w: extract_opt(w.opts,"times",[0]),
        pop_sizes=lambda w: extract_opt(w.opts,"pop_sizes",[1]),
        alpha=lambda w: extract_opt(w.opts,"alpha",np.eye(1,extract_opt(w.opts,"n",100) - 1,0)[0]),
        num_replicates=lambda w: extract_opt(w.opts,"num_replicates",10000),
        n_threads=lambda w: extract_opt(w.opts,"n_threads",100),
        parallelize=lambda w: extract_opt(w.opts,"parallelize",True),
        models=lambda w: w.models.split('_'),
        type="{type}",
        dist="{dist}"
    conda:
        "envs/base.yaml"
    script:
        "scripts/compare_moments.py"

rule plot_comparison:
    input:
        "resources/configs/{config}.yaml"
    output:
        "results/comparisons/graphs/{config}.png"
    conda:
        "envs/base.yaml"
    script:
        "scripts/plot_comparison_from_config.py"
