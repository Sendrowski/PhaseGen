# Parameter inference
The availability of exact moments lends itself to gradient-based parameter estimation. This is commonly done based on the SFS, but higher-order moments can also be used, provided they can be computed from the data at hand. ``phasegen`` provides a lightweight framework for performing parameter inference which is done by defining an {class}`~phasegen.inference.Inference` object.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
## Running an inference
{class}`~phasegen.inference.Inference` requires a parametrized coalescent distribution, a loss function, and parameter bounds to be specified. More specifically, the `coal` argument is a callable that returns a {class}`~phasegen.distributions.Coalescent` object, based on the parameter values of the current optimization step, and `loss` is a callable specifying the current loss. By default, 10 independent optimization runs are performed using the L-BFGS-B algorithm, and the best result is returned. 

Below we optimize a two-epoch demography where the time of the size change (``t``) and the resulting population size (``Ne``) are variable. The observed summary statistic is an SFS with a sample size of 10, and the loss function is the Poisson likelihood.

```{code-cell} python
import phasegen as pg

observation = pg.SFS(
    [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
)

inf = pg.Inference(
    bounds=dict(t=(0, 4), Ne=(0.1, 1)),
    coal=lambda t, Ne: pg.Coalescent(
        n=10,
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, t: Ne}}
        )
    ),
    loss=lambda coal, _: pg.PoissonLikelihood().compute(
        observed=observation.normalize().polymorphic,
        modelled=coal.sfs.mean.normalize().polymorphic
    ),
    seed=42
)
```

```{code-cell} r
library(phasegen)
pg <- load_phasegen()

observation <- pg$SFS(
    c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)
)

inf <- pg$Inference(
    bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
    coal = function(t, Ne) pg$Coalescent(
        n = 10,
        demography = pg$Demography(
            events = c(
                pg$PopSizeChange(pop = "pop_0", time = 0, size = 1),
                pg$PopSizeChange(pop = "pop_0", time = t, size = Ne)
            )
        )
    ),
    loss = function(coal, ...) pg$PoissonLikelihood()$compute(
        observed = observation$normalize()$polymorphic,
        modelled = coal$sfs$mean$normalize()$polymorphic
    ),
    seed = 42L
)
```

+++
Upon construction, the inference object is ready to be optimized and the result can be visualized.

```{code-cell} python
inf.run()
```

```{code-cell} python
inf.plot_pop_sizes();
```

```{code-cell} python
:tags: [remove-cell]
assert inf.loss_inferred <= inf.loss(inf.get_coal(t=0, Ne=1), None)
```

```{code-cell} r
inf$run()
```

```{code-cell} r
plot(inf, which = "pop_sizes")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$loss_inferred <= inf$loss(inf$get_coal(t = 0, Ne = 1), NULL))
```

+++
## Parametric bootstrapping
We may also wish to perform parametric bootstrapping. In order to do this, we provide a callback function to resample the data, and enable ``do_bootstrap``. Note that we specify {attr}`~phasegen.inference.Inference.observation` to {class}`~phasegen.inference.Inference`, which is necessary for the resampling to work.

```{code-cell} python
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
    do_bootstrap=True,
    n_bootstraps=20,
    seed=42
)
```

```{code-cell} r
inf <- pg$Inference(
    bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
    observation = pg$SFS(
        c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)
    ),
    coal = function(t, Ne) pg$Coalescent(
        n = 10,
        demography = pg$Demography(
            events = c(
                pg$PopSizeChange(pop = "pop_0", time = 0, size = 1),
                pg$PopSizeChange(pop = "pop_0", time = t, size = Ne)
            )
        )
    ),
    loss = function(coal, obs) pg$PoissonLikelihood()$compute(
        observed = obs$normalize()$polymorphic,
        modelled = coal$sfs$mean$normalize()$polymorphic
    ),
    resample = function(sfs, rng) sfs$resample(seed = rng),
    do_bootstrap = TRUE,
    n_bootstraps = 20L,
    seed = 42L
)
```

+++
We run the inference again and visualize the results.

```{code-cell} python
inf.run()
```

```{code-cell} python
inf.plot_pop_sizes();
```

```{code-cell} python
inf.plot_bootstraps();
```

```{code-cell} python
:tags: [remove-cell]
assert len(inf.bootstraps) == 20
```

```{code-cell} r
inf$run()
```

```{code-cell} r
plot(inf, which = "pop_sizes")
```

```{code-cell} r
plot(inf, which = "bootstraps")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(nrow(inf$bootstraps) == 20)
```

+++
## Distributed bootstrapping
For inferences with long runtimes, the bootstrapping process can be distributed by creating bootstrap samples which are {class}`~phasegen.inference.Inference` objects themselves ({meth}`~phasegen.inference.Inference.create_bootstrap`). These bootstraps can be run in parallel and the results combined afterwards ({meth}`~phasegen.inference.Inference.add_bootstraps`).

+++ {"tags": ["python-only"]}
Below is an example [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow for distributed bootstrapping: 

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

+++ {"tags": ["python-only"]}
where ``phasegen.yaml`` is a conda environment file with the following content:
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

+++ {"tags": ["python-only"]}
In ``setup_inference.py``, we set up the inference object, perform the initial run and save the results to a file. Note that bootstrapping is disabled here as we are doing it manually.

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

+++ {"tags": ["python-only"]}
In ``run_bootstrap.py``, we load the inference object from the file, create a bootstrap sample and run the inference for it. This will use the specified resampling function to resample the SFS.

```python
import phasegen as pg

file = snakemake.input[0]
out = snakemake.output[0]

inf = pg.Inference.from_file(file)

bootstrap = inf.create_bootstrap()

bootstrap.run()

bootstrap.to_file(out)
```

+++ {"tags": ["python-only"]}
In ``merge_bootstraps.py``, we load the inference object and all bootstraps, merge the bootstraps with the main inference object and visualize the results.

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
In a production setting the bootstraps are typically distributed across a cluster, for instance with a [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow. In such a workflow, one rule sets up the inference and performs the initial run ({meth}`~phasegen.inference.Inference.to_file`), a second rule loads it ({meth}`~phasegen.inference.Inference.from_file`), creates a single bootstrap and runs it, and a final rule merges the bootstrap results back into the main inference object. Below we demonstrate the same API in a single notebook, running the bootstraps in a serial loop rather than as parallel cluster jobs.

Two points are specific to using `phasegen` from R. First, `create_bootstrap` and `to_file` serialize the `coal`, `loss` and `resample` callbacks, so we define them as Python functions (via `reticulate::py_run_string`) rather than R closures, which do not survive that round-trip. Second, we accumulate the bootstrap objects in memory and merge them directly, rather than reloading each from its file, so their optimizer results stay intact.

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

+++ {"tags": ["r-only"]}
We set up the inference object, perform the initial run and save the result to a file. Bootstrapping is disabled here as we perform it manually.

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

+++ {"tags": ["r-only"]}
Each bootstrap is created from the inference object with {meth}`~phasegen.inference.Inference.create_bootstrap`, which resamples the observation using the provided `resample` callback, and is then run independently. On a cluster each iteration of this loop would instead be a separate job that loads ``inference.json``, creates one bootstrap, runs it and writes its own file. Here we reload the saved object once and run the replicates serially.

```{code-cell} r
:tags: [remove-output]
inf <- pg$Inference$from_file(path)

boots <- lapply(seq_len(20L), function(i) {
    b <- inf$create_bootstrap()
    b$run()
    b
})
```

+++ {"tags": ["r-only"]}
Finally we merge the bootstraps into the main inference object with {meth}`~phasegen.inference.Inference.add_bootstraps` and visualize the inferred demography and the bootstrap distribution of the inferred parameters.

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
