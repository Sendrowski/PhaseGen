.. _reference.miscellaneous:

Miscellaneous
=============

Logging
-------

``phasegen`` uses the standard Python :mod:`logging` module for logging. By default, ``phasegen`` logs to the console at the ``INFO`` level. The logging level can be changed, for example to ``DEBUG``, as follows:

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. code-block:: python

          import phasegen as pg

          pg.logger.setLevel("DEBUG")

   .. tab-item:: :fab:`r-project` R
      :sync: r

      .. code-block:: r

          library(phasegen)
          pg <- load_phasegen()

          pg$logger$setLevel("DEBUG")

Debugging
---------

When an unexpected error occurs, disabling parallelization yields a more descriptive stack trace (see :attr:`Settings.parallelize <phasegen.settings.Settings.parallelize>`).

Object-oriented design
----------------------

``phasegen`` follows an object-oriented design. Objects such as :class:`~phasegen.distributions.Coalescent` take their configuration on construction and expose the resulting statistics as properties and methods.
