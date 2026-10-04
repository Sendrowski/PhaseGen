.. _reference.installation:

Installation
============

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. rubric:: PyPI

      ``phasegen`` can be installed with ``pip``:

      .. code-block:: bash

         pip install phasegen

      ``phasegen`` is compatible with Python 3.10 through 3.13.

      The ``msprime``-backed distributions (:meth:`Coalescent.to_msprime() <phasegen.distributions.Coalescent.to_msprime>`, :class:`~phasegen.distributions.MsprimeCoalescent`) require ``msprime``, which is installed separately:

      .. code-block:: bash

         pip install msprime

      .. rubric:: Conda

      To avoid potential conflicts with other packages, it is recommended to install ``phasegen`` in an isolated environment. The easiest way to do this is with ``conda`` or ``mamba``:

      .. code-block:: bash

          mamba create -n phasegen -c conda-forge phasegen
          mamba activate phasegen

      Alternatively, for reproducibility, the environment can be defined in a file ``environment.yml``:

      .. code-block:: yaml

        name: phasegen
        channels:
          - conda-forge
        dependencies:
          - phasegen

      The environment is then created and activated with:

      .. code-block:: bash

        mamba env create -f environment.yml
        mamba activate phasegen

      ``phasegen`` is then imported with:

      .. code-block:: python

          import phasegen as pg

   .. tab-item:: :fab:`r-project` R
      :sync: r

      The ``phasegen`` R package is installed from GitHub with:

      .. code-block:: r

         devtools::install_github("Sendrowski/PhaseGen")

      Once the installation has completed, the package is loaded in an R session with:

      .. code-block:: r

         library(phasegen)

      The ``phasegen`` R package serves as a wrapper around the Python library, and draws its figures with ``ggplot2`` through ``plot()`` methods and with base graphics through ``persp()`` methods. Loading the R package declares the Python requirement, which ``reticulate`` resolves into a suitable environment the first time the module is loaded:

      .. code-block:: r

         pg <- load_phasegen()

      ``phasegen`` is compatible with Python 3.10 through 3.13.

      An existing Python installation can be used instead by installing ``phasegen`` as described under the Python tab and selecting its environment before loading the module:

      .. code-block:: r

         reticulate::use_condaenv("~/miniforge3/envs/phasegen", required = TRUE)
         pg <- load_phasegen()

      The R package documentation describes the available functions in more detail.
