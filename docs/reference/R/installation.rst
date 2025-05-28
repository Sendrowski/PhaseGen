.. _reference.r.installation:

Installation
============

To install the ``phasegen`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/PhaseGen")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(phasegen)

The ``phasegen`` R package serves as a wrapper around the Python library although visualization utilities are not reimplemented. You may choose to use the visualization capabilities of the Python API but this will offer limited customizability. The Python package must be installed separately which can be accomplished with:

.. code-block:: r

   install_phasegen()

``phasegen`` is compatible with Python 3.10, 3.11 and 3.12.

Alternatively, you can also follow the instructions in the `Python installation guide <../Python/installation.html>`_ to install the Python package.

After installing the Python package, the ``phasegen`` wrapper module can be loaded into your R environment using the following command:

.. code-block:: r

   pg <- load_phasegen()

See the R package documentation for more information on the available functions.