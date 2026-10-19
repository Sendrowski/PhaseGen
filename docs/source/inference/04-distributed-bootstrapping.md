```{code-cell} python
:tags: [remove-cell]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phasegen as pg

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
theta = 1
```

```{code-cell} r
:tags: [remove-cell]
library(phasegen)
library(ggplot2)
library(patchwork)
pg <- load_phasegen()
np <- reticulate::import("numpy", convert = FALSE)
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
theta <- 1
```

+++
## Distributed bootstrapping
For inferences with long runtimes, the bootstrapping process can be distributed by creating bootstrap samples which are {class}`~phasegen.inference.Inference` objects themselves ({meth}`Inference.create_bootstrap() <phasegen.inference.Inference.create_bootstrap>`). These bootstraps can be run in parallel and the results combined afterwards ({meth}`Inference.add_bootstraps() <phasegen.inference.Inference.add_bootstraps>`). Typically, one job sets up the inference and performs the initial run ({meth}`Inference.to_file() <phasegen.inference.Inference.to_file>`), one job per bootstrap loads it ({meth}`Inference.from_file() <phasegen.inference.Inference.from_file>`), creates its bootstrap and runs it, and a final job merges the bootstraps into the main inference object. Below we demonstrate these steps in a single notebook, running the bootstraps in a serial loop.

+++
We set up the inference object, perform the initial run and save the result to a file. Bootstrapping is disabled here as we perform it manually. {meth}`Inference.to_file() <phasegen.inference.Inference.to_file>` serializes the `coal`, `loss` and `resample` callbacks together with the variables they reference, so in Python they may be any callables, lambdas included. From R they must be Python functions, since R closures do not survive serialization.

```{code-cell} python
import os
import tempfile

inf = pg.Inference(
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    resample=lambda sfs, rng: sfs.resample(seed=rng),
    do_bootstrap=False,
    seed=42
)

inf.run()

path = os.path.join(tempfile.gettempdir(), 'inference.json')
inf.to_file(path)
```

```{code-cell} r
# the model, loss and resampling callbacks, defined in Python so they survive serialization
py <- reticulate::py_run_string('
import phasegen as pg

def coal(t, Ne):
    return pg.Coalescent(
        n=10,
        demography=pg.Demography(pop_sizes={"pop_0": {0: 1, t: Ne}}),
    )

def loss(coal, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic,
    )

def resample(sfs, rng):
    return sfs.resample(seed=rng)
')
```

```{code-cell} r
inf <- pg$Inference(
    bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
    observation = pg$SFS(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)),
    coal = py$coal,
    loss = py$loss,
    resample = py$resample,
    do_bootstrap = FALSE,
    seed = 42L
)

inf$run()

path <- file.path(tempdir(), "inference.json")
inf$to_file(path)
```

+++
Each bootstrap is created from the saved inference object with {meth}`Inference.create_bootstrap() <phasegen.inference.Inference.create_bootstrap>`, which resamples the observation using the provided `resample` callback, and is then run independently. On a cluster, each iteration of this loop is a separate job that writes its own file.

```{code-cell} python
:tags: [remove-output]
inf = pg.Inference.from_file(path)

bootstraps = []
for i in range(1, 21):
    bootstrap = inf.create_bootstrap(index=i)
    bootstrap.run()
    bootstraps.append(bootstrap)
```

```{code-cell} r
:tags: [remove-output]
inf <- pg$Inference$from_file(path)

boots <- lapply(seq_len(20L), function(i) {
    b <- inf$create_bootstrap(index = i)
    b$run()
    b
})
```

+++
Finally we merge the bootstraps into the main inference object with {meth}`Inference.add_bootstraps() <phasegen.inference.Inference.add_bootstraps>` and visualize the inferred demography and the bootstrap distribution of the inferred parameters.

```{code-cell} python
inf.add_bootstraps(bootstraps)
```

```{code-cell} python
:tags: [remove-cell]
assert len(inf.bootstraps) == 20
```

```{code-cell} python
inf.plot_demography();
```

```{code-cell} python
inf.plot_bootstraps();
```

```{code-cell} r
inf$add_bootstraps(boots)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(nrow(inf$bootstraps) == 20)
```

```{code-cell} r
plot(inf)
```

```{code-cell} r
plot(inf, which = "bootstraps")
```

+++
### Example workflow
On a cluster, these steps are well described by a workflow manager such as [Snakemake](https://snakemake.readthedocs.io/en/stable/), with one rule per step and one job per bootstrap replicate. Such a workflow could look as follows, with each rule running a script in the conda environment ``phasegen.yaml``.

+++ {"tags": ["python-only"]}
``Snakefile``:

```python
rule all:
    input:
        "results/graphs/inference/demography.png",

# setup inference and perform initial run
rule setup_inference:
    output:
        "results/inference/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "setup_inference.py"

# create and run bootstrap
rule run_bootstrap:
    input:
        "results/inference/inference.json"
    output:
        "results/inference/bootstraps/{i}/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "run_bootstrap.py"

# merge bootstraps and visualize
rule merge_bootstraps:
    input:
        inference="results/inference/inference.json",
        bootstraps=expand("results/inference/bootstraps/{i}/inference.json",i=range(100))
    output:
        inference="results/inference/inference.bootstrapped.json",
        demography="results/graphs/inference/demography.png",
        bootstraps="results/graphs/inference/bootstraps.png",
    conda:
        "phasegen.yaml"
    script:
        "merge_bootstraps.py"
```

``phasegen.yaml``:

```yaml
name: phasegen
channels:
  - defaults
dependencies:
  - python>=3.10,<3.14
  - pip
  - pip:
      - phasegen
```

+++ {"tags": ["r-only"]}
``Snakefile``:

```python
rule all:
    input:
        "results/graphs/inference/demography.png",

# setup inference and perform initial run
rule setup_inference:
    output:
        "results/inference/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "setup_inference.R"

# create and run bootstrap
rule run_bootstrap:
    input:
        "results/inference/inference.json"
    output:
        "results/inference/bootstraps/{i}/inference.json"
    conda:
        "phasegen.yaml"
    script:
        "run_bootstrap.R"

# merge bootstraps and visualize
rule merge_bootstraps:
    input:
        inference="results/inference/inference.json",
        bootstraps=expand("results/inference/bootstraps/{i}/inference.json",i=range(100))
    output:
        inference="results/inference/inference.bootstrapped.json",
        demography="results/graphs/inference/demography.png",
        bootstraps="results/graphs/inference/bootstraps.png",
    conda:
        "phasegen.yaml"
    script:
        "merge_bootstraps.R"
```

``phasegen.yaml``, into which the ``phasegen`` R package is installed once from GitHub as described in {ref}`reference.installation`:

```yaml
name: phasegen
channels:
  - conda-forge
dependencies:
  - r-base
  - r-reticulate
  - r-ggplot2
  - r-remotes
```

+++
The first script sets up the inference object, performs the initial run and saves the result to a file. Bootstrapping is disabled here as it is performed by the other rules.

+++ {"tags": ["python-only"]}
``setup_inference.py``:

```python
import phasegen as pg

out = snakemake.output[0]

inf = pg.Inference(
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    observation=pg.SFS(
        [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
    ),
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    loss=lambda coal, obs: pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    resample=lambda sfs, rng: sfs.resample(seed=rng),
    do_bootstrap=False,
    seed=42
)

inf.run()

inf.to_file(out)
```

+++ {"tags": ["r-only"]}
``setup_inference.R``:

```r
library(phasegen)
pg <- load_phasegen()

out <- snakemake@output[[1]]

# the model, loss and resampling callbacks, defined in Python so they survive serialization
py <- reticulate::py_run_string('
import phasegen as pg

def coal(t, Ne):
    return pg.Coalescent(
        n=10,
        demography=pg.Demography(pop_sizes={"pop_0": {0: 1, t: Ne}}),
    )

def loss(coal, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic,
    )

def resample(sfs, rng):
    return sfs.resample(seed=rng)
')

inf <- pg$Inference(
    bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
    observation = pg$SFS(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)),
    coal = py$coal,
    loss = py$loss,
    resample = py$resample,
    do_bootstrap = FALSE,
    seed = 42L
)

inf$run()

inf$to_file(out)
```

+++
The second script loads the inference object, creates the bootstrap replicate of its job and runs it.

+++ {"tags": ["python-only"]}
``run_bootstrap.py``:

```python
import phasegen as pg

file = snakemake.input[0]
out = snakemake.output[0]

inf = pg.Inference.from_file(file)

bootstrap = inf.create_bootstrap(index=int(snakemake.wildcards.i))

bootstrap.run()

bootstrap.to_file(out)
```

+++ {"tags": ["r-only"]}
``run_bootstrap.R``:

```r
library(phasegen)
pg <- load_phasegen()

file <- snakemake@input[[1]]
out <- snakemake@output[[1]]

inf <- pg$Inference$from_file(file)

bootstrap <- inf$create_bootstrap(index = as.integer(snakemake@wildcards[["i"]]))

bootstrap$run()

bootstrap$to_file(out)
```

+++
The third script loads the inference object and all bootstraps, merges the bootstraps into the main inference object and visualizes the results.

+++ {"tags": ["python-only"]}
``merge_bootstraps.py``:

```python
import phasegen as pg

inf_file = snakemake.input.inference
bootstraps_file = snakemake.input.bootstraps
out = snakemake.output.inference
out_demography = snakemake.output.demography
out_bootstraps = snakemake.output.bootstraps

inf = pg.Inference.from_file(inf_file)

inf.add_bootstraps([pg.Inference.from_file(f) for f in bootstraps_file])

inf.plot_demography(file=out_demography)
inf.plot_bootstraps(file=out_bootstraps)

inf.to_file(out)
```

+++ {"tags": ["r-only"]}
``merge_bootstraps.R``:

```r
library(phasegen)
library(ggplot2)
pg <- load_phasegen()

inf <- pg$Inference$from_file(snakemake@input[["inference"]])

inf$add_bootstraps(lapply(snakemake@input[["bootstraps"]], pg$Inference$from_file))

ggsave(snakemake@output[["demography"]], plot(inf))
ggsave(snakemake@output[["bootstraps"]], plot(inf, which = "bootstraps"))

inf$to_file(snakemake@output[["inference"]])
```
