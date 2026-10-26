"""The exponential of a Metzler generator and its integral, applied to a row vector by uniformization."""
from functools import cached_property
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.special import gammaln

#: Largest product of the uniformization rate and the step of one chunk, below which the first Poisson weight stays
#: within the double-precision range.
_MAX_CHUNK = 500.0

#: Absolute truncation floor of the Poisson series relative to the 1-norm of the propagated vector.
_FLOOR = 1e-46

#: Floor relative to the 1-norm of the propagated vector above which the entries seed the states a step of more than
#: one chunk is taken over.
_SUPPORT_FLOOR = 1e-16

#: Factor on the largest exit rate among the states reachable from the entries above ``_SUPPORT_FLOOR``, up to which
#: the nonzero entries below that floor also seed the states a step of more than one chunk is taken over.
_SUPPORT_RATE_RATIO = 10.0


class _Uniformization:
    r"""
    The row vectors :math:`\mathbf{v} e^{\mathbf{A} \tau}` and :math:`\mathbf{v} \int_0^\tau e^{\mathbf{A} s}\,
    \mathrm{d}s` for a sub-generator :math:`\mathbf{A}` with non-negative off-diagonal entries and non-positive row
    sums, and a non-negative :math:`\mathbf{v}`. With :math:`\lambda` the largest exit rate :math:`-A_{ii}`,
    :math:`\mathbf{P} = \mathbf{I} + \mathbf{A} / \lambda` is non-negative and substochastic, and

    .. math::

        e^{\mathbf{A} \tau} = \sum_{k \ge 0} w_k \mathbf{P}^k, \qquad
        \int_0^\tau e^{\mathbf{A} s}\, \mathrm{d}s = \frac{1}{\lambda} \sum_{k \ge 0} Q_k \mathbf{P}^k,

    with :math:`w_k` the Poisson probabilities of :math:`k` at mean :math:`\lambda \tau` and :math:`Q_k = \sum_{j > k}
    w_j` their upper tails. The series is truncated where the remaining Poisson mass falls below ``_FLOOR`` times the
    1-norm of the vector. A step with :math:`\lambda \tau` above ``_MAX_CHUNK`` is taken in equal chunks over the
    states ``_support`` returns, at their largest exit rate, and restarted on the remaining states whenever these
    shrink (adaptive uniformization).
    """

    def __init__(self, A) -> None:
        """
        :param A: The sub-generator, dense or sparse.
        """
        #: The sub-generator
        self.A: sp.csr_matrix = sp.csr_matrix(A)

        #: The exit rates
        self.exits: np.ndarray = np.maximum(-self.A.diagonal(), 0.0)

        #: The uniformization rate
        self.lam: float = float(self.exits.max()) if self.A.shape[0] else 0.0

    @cached_property
    def _targets(self) -> sp.csr_matrix:
        """
        The transposed sparsity pattern of the sub-generator, whose product with a mask of states marks their targets.
        """
        return (self.A != 0).T.tocsr()

    def _support(self, v: np.ndarray) -> np.ndarray:
        """
        The states reachable from the entries of ``v`` above ``_SUPPORT_FLOOR`` times its 1-norm, and from the nonzero
        ones exiting at most ``_SUPPORT_RATE_RATIO`` times as fast as the states reachable from those.

        :param v: The non-negative, nonzero row vector.
        :return: The mask of the states.
        """
        def reachable(seeds: np.ndarray) -> np.ndarray:
            reach = frontier = seeds
            while frontier.any():
                frontier = (self._targets @ frontier) & ~reach
                reach = reach | frontier
            return reach

        strong = v > _SUPPORT_FLOOR * v.sum()
        slow = self.exits <= _SUPPORT_RATE_RATIO * self.exits[reachable(strong)].max()

        return reachable(strong | ((v > 0) & slow))

    @staticmethod
    def _weights(lt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        The Poisson probabilities at mean ``lt`` up to the truncation, normalized to sum to one, and their upper
        tails.

        :param lt: The mean.
        :return: The probabilities and the upper tails.
        """
        # beyond the mean, the tail after k is at most w_{k + 1} / (1 - lt / (k + 2))
        log_floor = np.log(_FLOOR)
        k = int(np.ceil(lt)) + 1
        while -lt + (k + 1) * np.log(lt) - gammaln(k + 2) - np.log1p(-lt / (k + 2)) > log_floor:
            k += max(1, int(np.sqrt(lt)) // 4)

        ks = np.arange(k + 1)
        w = np.exp(-lt + ks * np.log(lt) - gammaln(ks + 1))
        w /= w.sum()
        Q = np.append(np.cumsum(w[:0:-1])[::-1], 0.0)

        return w, Q

    def propagate(self, v: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param v: The non-negative row vector.
        :param tau: The non-negative time.
        :return: The vector advanced by ``tau`` and its integral over ``[0, tau]``.
        """
        v = np.asarray(v, dtype=float)

        if self.lam == 0 or tau == 0 or not v.any():
            return v.copy(), v * tau

        restrict = self.lam * tau > _MAX_CHUNK
        floor = _FLOOR * float(v.sum()) / max(1.0, tau)
        u, occ = v.copy(), np.zeros_like(v)

        while tau > 0 and u.any():
            A, lam, keep = self.A, self.lam, slice(None)
            if restrict:
                keep = np.flatnonzero(self._support(u))
                A, lam = A[keep][:, keep], float(self.exits[keep].max())

            x, y, u = u[keep], occ[keep], np.zeros_like(v)
            if lam == 0:
                u[keep], occ[keep] = x, y + x * tau
                break

            n = A.shape[0]
            chunks = max(1, int(np.ceil(lam * tau / _MAX_CHUNK)))
            h = tau / chunks
            w, Q = self._weights(lam * h)
            Pt = (sp.identity(n, format='csr') + A / lam).T.tocsr()

            for done in range(1, chunks + 1):
                x = _series(Pt.indptr, Pt.indices, Pt.data, x, w, Q / lam, floor, y)
                u[keep] = x
                if not x.any() or restrict and done < chunks and self._support(u).sum() < n:
                    break

            u[keep], occ[keep] = x, y
            tau = (chunks - done) * h

        return u, occ


@njit(cache=True)
def _series(indptr, indices, data, v, w, q, floor, occ):
    """
    ``sum_k w_k x_k`` with ``x_k = P^k v`` for the CSR matrix ``P``, adding ``sum_k q_k x_k`` to ``occ``, until the
    weights end or the 1-norm of ``x_k`` falls to ``floor``.
    """
    n = v.size
    x, y = v.copy(), np.empty(n)
    u = w[0] * v
    for i in range(n):
        occ[i] += q[0] * v[i]

    for k in range(1, w.size):
        total = 0.0
        for i in range(n):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += data[p] * x[indices[p]]
            y[i] = acc
            total += acc
        x, y = y, x

        for i in range(n):
            u[i] += w[k] * x[i]
            occ[i] += q[k] * x[i]

        if total <= floor:
            break

    return u
