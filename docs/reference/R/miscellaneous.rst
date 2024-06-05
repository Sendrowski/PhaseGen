.. _reference.r.miscellaneous:

Miscellaneous
=============

Logging
-------

``phasegen`` uses the standard Python :mod:`logging` module for logging. By default, ``phasegen`` logs to the console at the ``INFO`` level. You can change the logging level, to for example ``DEBUG`` as follows::

    pg <- load_phasegen()

    pg$logger$setLevel("DEBUG")

Debugging
---------

If you encounter an unexpected error, you might want to disable parallelization to obtain a more descriptive stack trace (see ``parallelize`` for :class:`~phasegen.distributions.Coalescent` and :class:`~phasegen.inference.Inference`).

Object-Oriented Design
----------------------
The ``phasegen`` Python library is implemented using an object-oriented design which carries over to the R interface. This paradigm may be less familiar to R users.