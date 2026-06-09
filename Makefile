# Convenience targets. The Snakefile holds the full reproducibility workflow; these are shortcuts for common tasks.

.PHONY: docs docs-clean notebooks

# build the HTML documentation into docs/_build/html
docs:
	$(MAKE) -C docs html

# rebuild the documentation from a clean state (picks up new API classes and notebook changes)
docs-clean:
	$(MAKE) -C docs clean html

# re-execute all documentation notebooks in place, embedding fresh outputs (Python in the dev env, R in the R env)
notebooks:
	snakemake --use-conda --cores 1 reexecute_notebooks
