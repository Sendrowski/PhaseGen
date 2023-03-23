rule all:
    input:
        expand("results/simulations/PH/{n}/moments.json",n=[10, 100]),
        expand("results/simulations/msprime/{n}/moments.json",n=[10, 100]),

# simulate moments using phase-type theory
rule simulate_moments_PH:
    output:
        "results/simulations/PH/{n}/moments.json"
    params:
        n=lambda w: int(w.n)
    conda:
        "envs/base.yaml"
    script:
        "scripts/simulate_moments_PH.py"

# simulate moments using msprime
rule simulate_moments_msprime:
    output:
        "results/simulations/msprime/{n}/moments.json"
    params:
        n=lambda w: int(w.n),
        num_replicates=10000
    conda:
        "envs/base.yaml"
    script:
        "scripts/simulate_moments_msprime.py"

# simulate moments using phase-type theory and msprime
rule simulate_moments:
    output:
        "results/simulations/comp/{n}_variable/moments.json"
    params:
        n=lambda w: int(w.n),
        pop_sizes=[0.12, 1, 0.01, 10],
        times=[0, 0.3, 1, 1.4]
