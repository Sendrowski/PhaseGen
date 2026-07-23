.. _reference.performance.state_space_construction:

State Space Construction
========================
Constructing the state space (enumerating the states and assembling the rate matrix) is accelerated with `numba <https://numba.pydata.org/>`__, speeding it up by one to several orders of magnitude for larger state spaces. The acceleration is applied automatically (numba is a required dependency); it can be disabled by setting :attr:`~phasegen.settings.Settings.use_numba` to ``False``, in which case construction uses the pure-Python implementation.
