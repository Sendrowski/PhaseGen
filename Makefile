# Convenience targets. The Snakefile holds the full reproducibility workflow; these are shortcuts for common tasks.

.PHONY: docs docs-clean notebooks test test-all

# pytest-xdist worker count (-n) and per-test runtime tracking (--durations=0 reports every test's duration).
# Override the worker count with e.g. `make test PYTEST_NPROCS=4`.
PYTEST_NPROCS ?= auto

# run the fast test suite (excludes slow msprime comparisons / large state spaces) in parallel, with runtime tracking
test:
	pytest -m "not slow" -n $(PYTEST_NPROCS) --durations=0

# run the full suite including the slow comparisons
test-all:
	pytest -n $(PYTEST_NPROCS) --durations=0

# build the HTML documentation into docs/_build/html
docs:
	$(MAKE) -C docs html

# rebuild the documentation from a clean state (picks up new API classes and notebook changes)
docs-clean:
	$(MAKE) -C docs clean html

# re-execute all documentation notebooks in place, embedding fresh outputs (Python in the dev env, R in the R env)
notebooks:
	snakemake --use-conda --cores 1 reexecute_notebooks
