.. _reference.python.installation:

Installation
============

PyPi
^^^^
To install the ``phasegen``, you can use pip:

.. code-block:: bash

   pip install phasegen

``phasegen`` is compatible with Python 3.10 through 3.12.

Conda
^^^^^
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

Activate the newly created conda environment:

.. code-block:: python

    import phasegen as pg