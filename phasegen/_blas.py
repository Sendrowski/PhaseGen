"""
Limit on the threads of the BLAS libraries during an evaluation.
"""
from contextlib import ContextDecorator
from typing import Optional

import threadpoolctl

from .settings import Settings


class BlasThreadLimit(ContextDecorator):
    """
    Context manager and decorator limiting the threads of the loaded BLAS libraries to :attr:`Settings.blas_threads
    <phasegen.settings.Settings.blas_threads>` while it is entered, and restoring them on exit. Nested entries are
    allowed. The libraries are detected once, at the first entry.

    Example::

        @BlasThreadLimit()
        def evaluate():
            ...
    """

    #: The controller of the BLAS libraries loaded at the first entry
    _controller: Optional[threadpoolctl.ThreadpoolController] = None

    def __init__(self) -> None:
        """
        Initialize the limit.
        """
        #: The limits entered and not yet exited, innermost last
        self._entered: list = []

    def __enter__(self) -> 'BlasThreadLimit':
        """
        Limit the threads of the BLAS libraries, unless :attr:`Settings.blas_threads
        <phasegen.settings.Settings.blas_threads>` is ``None``.

        :return: This limit.
        """
        limiter = None
        if Settings.blas_threads is not None:
            if BlasThreadLimit._controller is None:
                BlasThreadLimit._controller = threadpoolctl.ThreadpoolController()
            limiter = BlasThreadLimit._controller.limit(limits=Settings.blas_threads, user_api='blas')
        self._entered.append(limiter)

        return self

    def __exit__(self, *exc) -> None:
        """
        Restore the threads of the BLAS libraries to their number at the matching entry.

        :param exc: The exception information, if any.
        """
        limiter = self._entered.pop()
        if limiter is not None:
            limiter.restore_original_limits()
