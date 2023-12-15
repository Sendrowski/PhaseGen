.. _installation_python:

Installation
============

PyPi
^^^^
To install the `phasegen`, you can use pip:

.. code-block:: bash

   pip install phasegen

Conda
^^^^^
However, to avoid potential conflicts with other packages, it is recommended to install `phasegen` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

Create a new file called ``environment.yml`` with the following content:

   .. code-block:: yaml

      name: phasegen
      channels:
        - defaults
      dependencies:
        - python
        - pip
        - pip:
            - phasegen

Run the following command to create a new `conda` environment using the ``environment.yml`` file:

   .. code-block:: bash

      conda env create -f environment.yml

Activate the newly created conda environment:

   .. code-block:: bash

      conda activate phasegen

You are now ready to use the ``phasegen`` package within the isolated conda environment.

   .. code-block:: python

        import phasegen as pg