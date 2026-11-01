"""Runs of equal steps of a time grid and the propagators reused along them."""
import sys
from math import isfinite
from typing import Callable, Dict, List, Tuple

import numpy as np

#: Number of unit roundoffs of a grid time within which its step continues a run of equal steps.
_STEP_ULPS = 4

#: Largest number of step lengths whose propagators ``_Propagators`` keeps.
_MAX_PROPAGATORS = 8

#: Largest number of steps of a run, bounding the row vectors a run materializes.
_MAX_RUN_STEPS = 2 ** 12


class _StepRuns:
    """
    Ascending times of one epoch as runs of equal steps from the time the propagated vector is at. A run of
    :math:`c` steps :math:`d` from :math:`x_0` advances the vector to the times :math:`x_0 + j d`,
    :math:`j = 1, \\ldots, c`, and holds the times :math:`u_j` it covers when a step exists with
    :math:`|u_j - x_0 - j d| \\le \\epsilon |u_j|` for all :math:`j`, where :math:`\\epsilon` is ``_STEP_ULPS``
    unit roundoffs, the rounding of a time grid. Each run is extended while such a step exists, and is then cut into
    runs of at most ``_MAX_RUN_STEPS`` steps. Times at or before the time reached take a step of zero.
    """

    @staticmethod
    def split(x: float, t: np.ndarray) -> Tuple[List[Tuple[float, int]], float]:
        """
        :param x: The time the vector is at.
        :param t: Ascending times, at least ``x`` up to rounding.
        :return: The runs as pairs of step and number of steps, and the time reached after the last.
        """
        tol = _STEP_ULPS * sys.float_info.epsilon

        # the closed runs, and the open one as its start, the bounds on its step, its number of steps and last time
        runs: List[Tuple[float, int]] = []
        run = None

        for u in t.tolist():
            if run is not None:
                x0, lo, hi, c, last = run

                if isfinite(u) and isfinite(x0):
                    r = tol * abs(u)
                    lo_new, hi_new = max(lo, (u - x0 - r) / (c + 1)), min(hi, (u - x0 + r) / (c + 1))

                    if lo_new <= hi_new:
                        run = (x0, lo_new, hi_new, c + 1, u)
                        continue

                d = _StepRuns._step(*run)
                runs.append((d, c))
                x = x0 + c * d

            d = u - x if u > x else 0.0
            r = tol * abs(u) if isfinite(d) else 0.0
            run = (x, d - r, d + r, 1, u)

        if run is not None:
            d = _StepRuns._step(*run)
            runs.append((d, run[3]))
            x = run[0] + run[3] * d

        return [(d, min(c - s, _MAX_RUN_STEPS)) for d, c in runs for s in range(0, c, _MAX_RUN_STEPS)], x

    @staticmethod
    def _step(x0: float, lo: float, hi: float, c: int, last: float) -> float:
        """The step of a run, that to its last time where it lies within the bounds."""
        d = (last - x0) / c if last > x0 else 0.0

        return max(0.0, min(max(d, lo), hi))


class _Propagators:
    """
    The propagators of one epoch by step length, computed by ``compute(tau)`` and kept for the ``_MAX_PROPAGATORS``
    step lengths used most recently.
    """

    def __init__(self, compute: Callable[[float], np.ndarray]) -> None:
        """
        :param compute: The propagator of a step.
        """
        self._compute = compute
        self._cache: Dict[float, np.ndarray] = {}

    def __call__(self, tau: float) -> np.ndarray:
        """
        :param tau: The step.
        :return: The propagator of ``tau``.
        """
        P = self._cache.pop(tau) if tau in self._cache else self._compute(tau)
        self._cache[tau] = P

        if len(self._cache) > _MAX_PROPAGATORS:
            del self._cache[next(iter(self._cache))]

        return P
