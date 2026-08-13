.. _reference.r.installation:

Installation
============

To install the ``phasegen`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/PhaseGen")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(phasegen)

The ``phasegen`` R package serves as a wrapper around the Python library although visualization utilities are not reimplemented. The visualization capabilities of the Python API remain available, with limited customizability. Loading the R package declares the Python requirement, which reticulate resolves into a suitable environment the first time the module is loaded:

.. code-block:: r

   pg <- load_phasegen()

``phasegen`` is compatible with Python 3.10 through 3.13.

To use an existing Python installation instead, follow the `Python installation guide <../Python/installation.html>`_ and select the environment before loading the module:

.. code-block:: r

   reticulate::use_condaenv("~/miniforge3/envs/phasegen", required = TRUE)
   pg <- load_phasegen()

See the R package documentation for more information on the available functions.