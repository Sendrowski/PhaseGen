.. _reference.installation:

Installation
============

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. rubric:: PyPI

      To install ``phasegen``, you can use pip:

      .. code-block:: bash

         pip install phasegen

      ``phasegen`` is compatible with Python 3.10 through 3.13.

      .. rubric:: Conda

      However, to avoid potential conflicts with other packages, it is recommended to install ``phasegen`` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

      To do this, you can run

      .. code-block:: bash

          mamba create -n phasegen -c conda-forge phasegen
          mamba activate phasegen

      Alternatively, to ensure reproducibility, you can create a file ``environment.yml``:

      .. code-block:: yaml

        name: phasegen
        channels:
          - conda-forge
        dependencies:
          - phasegen

      Then run the following commands to create and activate the environment:

      .. code-block:: bash

        mamba env create -f environment.yml
        mamba activate phasegen

      You are now ready to use ``phasegen``:

      .. code-block:: python

          import phasegen as pg

   .. tab-item:: :fab:`r-project` R
      :sync: r

      To install the ``phasegen`` package in R, execute the following command:

      .. code-block:: r

         devtools::install_github("Sendrowski/PhaseGen")

      Once the installation is successfully completed, initiate the package within your R session using:

      .. code-block:: r

         library(phasegen)

      The ``phasegen`` R package serves as a wrapper around the Python library, and draws its figures with ``ggplot2`` through ``plot()`` and ``persp()`` methods. Loading the R package declares the Python requirement, which reticulate resolves into a suitable environment the first time the module is loaded:

      .. code-block:: r

         pg <- load_phasegen()

      ``phasegen`` is compatible with Python 3.10 through 3.13.

      To use an existing Python installation instead, follow the installation instructions under the Python tab and select the environment before loading the module:

      .. code-block:: r

         reticulate::use_condaenv("~/miniforge3/envs/phasegen", required = TRUE)
         pg <- load_phasegen()

      See the R package documentation for more information on the available functions.
