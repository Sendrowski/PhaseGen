.. _reference.miscellaneous:

Miscellaneous
=============

Logging
-------

PhaseGen uses the standard Python :mod:`logging` module for logging. By default, PhaseGen logs to the console at the ``INFO`` level. You can change the logging level, to for example ``DEBUG`` as follows::

    import phasegen as pg

    pg.logger.setLevel("DEBUG")

Debugging
---------

If you encounter an unexpected error, you might want to disable parallelization to obtain a more descriptive stack trace (see ``parallelize`` for :class:`~phasegen.distributions.Coalescent` and :class:`~phasegen.inference.Inference`).
