.. _reference.r.exponentiation_backend:

TODO add support for R

Exponentiation Backend
======================

``phasegen`` makes heavy use of matrix exponentiation to compute quantities of interest. You can choose between different exponentiation backends for this purpose. The default is :class:`~phasegen.expm.SciPyExpmBackend` which uses SciPy's expm function. One alternative, which is often faster, especially when parallelization is enabled, is :class:`~phasegen.expm.TensorFlowExpmBackend`. To switch to the TensorFlow backend, you need to install TensorFlow which is an optional dependency of ``phasegen`` due to its heavy weight. To install everything in one go, you can use the following conda environment file:

.. code-block:: yaml

  name: phasegen
  channels:
    - defaults
  dependencies:
    - python>=3.10,<3.13
    - tensorflow
    - pip
    - pip:
        - phasegen

Only Python-based backends are implemented which necessitates the use of a Python-based package manager for installation.
After installation you can use register the backend as follows:

.. code-block:: r

    pg <- load_phasegen()

    pg$Backend$register(pg$TensorFlowExpmBackend())