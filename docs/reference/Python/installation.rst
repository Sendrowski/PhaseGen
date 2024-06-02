.. _reference.python.installation:

Installation
============

PyPi
^^^^
To install the ``phasegen``, you can use pip:

.. code-block:: bash

   pip install phasegen

phasegen is compatible with Python 3.10, 3.11 and 3.12.

Conda
^^^^^
However, to avoid potential conflicts with other packages, it is recommended to install ``phasegen`` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

To do this, you can run

.. code-block:: bash

    mamba create -n phasegen 'python>=3.10,<3.13' pip
    mamba activate phasegen
    pip install phasegen

Alternative, create a new file called ``environment.yml`` with the following content:

.. code-block:: yaml

  name: phasegen
  channels:
    - defaults
  dependencies:
    - python>=3.10,<3.13
    - pip
    - pip:
        - phasegen

Run the following command to create a new `conda` environment using the ``environment.yml`` file:

.. code-block:: bash

  mamba env create -f environment.yml

Activate the newly created conda environment:

.. code-block:: bash

  mamba activate phasegen

You are now ready to use the ``phasegen`` package within the isolated conda environment.

.. code-block:: python

    import phasegen as fd