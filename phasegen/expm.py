"""
Matrix exponentiation backends.

.. deprecated::
    The backend registry is deprecated and will be removed. The coalescent statistics issue many small matrix
    exponentials, for which the default SciPy backend is faster than the TensorFlow, Jax and PyTorch ones, whose
    per-call overhead dominates at these sizes.

Two operations are exposed: the dense matrix exponential :math:`\\exp(\\mathbf{A})`
(:meth:`ExpmBackend.compute() <phasegen.expm.ExpmBackend.compute>`) and its action
:math:`\\exp(\\mathbf{A})\\mathbf{v}` on a vector or thin matrix
(:meth:`ExpmBackend.compute_action() <phasegen.expm.ExpmBackend.compute_action>`).

A registered backend reaches the Van Loan evaluation of moments, the tree-height distribution functions and the
mutational configurations. The Laplace transform of an accumulated reward always calls SciPy. The occupation times of
spectra call SciPy above :attr:`Settings.closed_form_sparse_min_states
<phasegen.settings.Settings.closed_form_sparse_min_states>` and :attr:`Settings.expm_action_min_dim
<phasegen.settings.Settings.expm_action_min_dim>`, and the registered backend below them.
"""
import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy
import scipy.sparse
import scipy.special

logger = logging.getLogger('phasegen')

#: 1-norm of the first step of :meth:`Backend.expm_multiply() <phasegen.expm.Backend.expm_multiply>` on an argument of
#: larger norm. The steps double in norm and end once the vector is stationary, since the cost of the truncated Taylor
#: action grows linearly in the norm.
_ACTION_STEP_NORM = 1e4

#: Unit roundoff below which the remaining change of an action counts as none.
_UNIT_ROUNDOFF = 2.0 ** -53

#: Multiple of the unit roundoff of ``|A| |v|`` within which an entry of ``A v`` counts as zero, the resolution of the
#: product at a vector that is stationary up to the error of the action that produced it.
_STATIONARY_NOISE = 32


class _GrowthBound:
    r"""
    A bound :math:`G(f) \ge f^{-1} \int_0^f \|\exp(s \mathbf{A})\|_1\, ds` for
    :meth:`Backend._stationary() <phasegen.expm.Backend._stationary>`. With :math:`\mathbf{A} = \mathbf{D} + \mathbf{N}`
    split into its :math:`L` diagonal blocks :math:`\mathbf{D}` and the strictly block triangular rest
    :math:`\mathbf{N}`, the Dyson series of :math:`\exp(s \mathbf{A})` ends after :math:`L` terms, so that

    .. math::
        G(f) = \min_{p \in \{1, \infty\}} c_p \Big( \varphi(f m_p) + e^{f m_p}
        \sum_{j=1}^{L-1} \frac{(f \beta_p)^j}{(j + 1)!} \Big),

    with :math:`m_p = \max(0, \mu_p(\mathbf{D}))` the positive part of the logarithmic :math:`p`-norm,
    :math:`\mu_1(\mathbf{D}) = \max_j (d_{jj} + \sum_{i \ne j} |d_{ij}|)` and :math:`\mu_\infty(\mathbf{D}) =
    \mu_1(\mathbf{D}^\top)`, :math:`\beta_p = \|\mathbf{N}\|_p`, :math:`\varphi(z) = (e^z - 1) / z` and
    :math:`\varphi(0) = 1`, :math:`c_1 = 1` and :math:`c_\infty` the dimension of :math:`\mathbf{A}`, which converts
    the :math:`\infty`-norm bound to the 1-norm. For a transposed generator and :math:`L = 1`, :math:`G = 1`.
    """

    def __init__(self, a, abs_a, blocks: int = 1) -> None:
        r"""
        :param a: Square matrix :math:`\mathbf{A}`.
        :param abs_a: Its entrywise absolute value.
        :param blocks: The number :math:`L` of equal diagonal blocks.
        """
        coo = scipy.sparse.coo_matrix(abs_a)
        dim = a.shape[0]
        d = np.asarray(a.diagonal(), dtype=float)
        inner = coo.row // (dim // blocks) == coo.col // (dim // blocks)

        #: The number of blocks :math:`L`.
        self.blocks: int = blocks

        #: The triples :math:`(c_p, m_p, \beta_p)` for :math:`p = 1, \infty`.
        self.norms: list = []
        for c, axis in ((1.0, coo.col), (float(dim), coo.row)):
            sums = np.bincount(axis[inner], coo.data[inner], dim) - np.abs(d) + d
            off = np.bincount(axis[~inner], coo.data[~inner], dim)
            self.norms.append((c, max(0.0, float(sums.max())), float(off.max())))

    def __call__(self, f: float) -> float:
        r"""
        :param f: The fraction :math:`f \ge 0`.
        :return: :math:`G(f)`, possibly infinite.
        """
        j = np.arange(1, self.blocks)
        out = np.inf
        with np.errstate(over='ignore', invalid='ignore'):
            for c, m, beta in self.norms:
                z = f * m
                phi = np.expm1(z) / z if z > 0 else 1.0
                tail = np.exp(z) * float(np.sum(np.exp(j * np.log(f * beta) - scipy.special.gammaln(j + 2)))) \
                    if j.size and f * beta > 0 else 0.0
                out = min(out, c * (phi + tail))
        return float(out)


class ExpmBackend(ABC):
    """
    Base class for matrix exponentiation backends. A custom backend implements :meth:`compute` and is activated with
    :meth:`Backend.register() <phasegen.expm.Backend.register>`.
    """

    @abstractmethod
    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential :math:`\\exp(\\mathbf{A})`.

        :param m: Square matrix.
        :return: The matrix exponential of ``m``.
        """
        pass

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential on a vector (or thin matrix),
        :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``).

        The default implementation densifies ``a`` and forms the dense exponential via :meth:`compute`, so the action
        uses the backend's own exponentiation. :class:`SciPyExpmBackend` overrides it with the truncated Taylor
        algorithm of Al-Mohy and Higham (2011) in :func:`scipy.sparse.linalg.expm_multiply`, which exploits the
        sparsity of :math:`\\mathbf{A}` without forming the dense exponential. Other backends may override it likewise.

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        a_dense = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)

        return self.compute(a_dense) @ b


class TensorFlowExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using TensorFlow, an optional dependency with the installation and GPU
    notes of :class:`JaxExpmBackend`.
    """

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using TensorFlow.

        :param m: Matrix.
        :return: Matrix exponential
        """
        # noinspection PyUnresolvedReferences
        import tensorflow as tf

        return tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy()


class SciPyExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using SciPy.

    .. note::
        This is the default backend.
    """

    #: Largest 1-norm passed to :func:`scipy.linalg.expm`. Its choice of the number of squarings overflows for 1-norms
    #: above about 1e38, so a larger argument is scaled by a power of two below this bound and the result squared,
    #: each squaring roughly doubling the relative error.
    _max_norm: float = 1e36

    def __init__(self, precision: type | str | np.dtype = np.float64) -> None:
        """
        Initialize the backend.

        :param precision: Floating-point precision of the matrix exponential and its action, as a NumPy floating type
            such as ``np.float32`` or ``np.float64``, or its name such as ``'float32'``. Defaults to double precision.
            A lower precision may be faster but is much more prone to numerical issues. Only single and double
            precision are supported.
        :raises TypeError: If ``precision`` is neither single nor double precision.
        """
        try:
            dtype = np.dtype(precision)
        except TypeError:
            dtype = None

        if dtype is None or dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"Precision must be np.float32 or np.float64, got {precision!r}.")

        #: Precision of the matrix exponential and its action
        self.precision: np.dtype = dtype

    def compute(self, m: np.ndarray) -> np.ndarray:
        r"""
        Compute the matrix exponential using SciPy, as :math:`\exp(\mathbf{A} / 2^k)^{2^k}` with the smallest
        :math:`k \ge 0` that brings the 1-norm of :math:`\mathbf{A} / 2^k` within :attr:`_max_norm`.

        :param m: Matrix
        :return: Matrix exponential
        """
        m = m.astype(self.precision)
        norm = float(np.abs(m).sum(axis=0).max()) if m.size else 0.0
        k = int(np.ceil(np.log2(norm / self._max_norm))) if np.isfinite(norm) and norm > self._max_norm else 0

        e = scipy.linalg.expm(m / 2.0 ** k if k else m)
        for _ in range(k):
            e = e @ e

        return e

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``) with the truncated Taylor algorithm
        of Al-Mohy and Higham (2011) in :func:`scipy.sparse.linalg.expm_multiply`, which exploits the sparsity of
        :math:`\\mathbf{A}` without forming the dense exponential.

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        from scipy.sparse.linalg import expm_multiply

        return expm_multiply(a.astype(self.precision), np.asarray(b, dtype=self.precision))


class JaxExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using Jax.
    Note that jax is an optional dependency and thus needs to be installed separately.
    GPU acceleration may be available depending on the underlying hardware.
    """

    def __init__(self, max_squarings: int = 2 ** 10) -> None:
        """
        Initialize the backend.

        :param max_squarings: Maximum number of squarings (see jax.scipy.linalg.expm).
        """
        import jax

        # enable double precision
        jax.config.update("jax_enable_x64", True)

        #: Maximum number of squarings
        self.max_squarings = max_squarings

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using Jax.

        :param m: Matrix
        :return: Matrix exponential
        """
        import jax

        # casting explicitly to np.float64 to avoid problems with object type
        return jax.scipy.linalg.expm(m.astype(np.float64), max_squarings=self.max_squarings)


class PyTorchExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using PyTorch.
    Note that PyTorch is an optional dependency and thus needs to be installed separately.
    GPU acceleration may be available depending on the underlying hardware.
    """

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using PyTorch.

        :param m: Matrix
        :return: Matrix exponential
        """
        # noinspection PyUnresolvedReferences
        import torch

        # casting explicitly to np.float64 to avoid problems with object type
        return torch.matrix_exp(torch.tensor(m.astype(np.float64), dtype=torch.float64)).numpy()


class Backend(ABC):
    """
    Configure the backend for matrix exponentiation.
    """
    #: Backend for matrix exponentiation
    backend: ExpmBackend = SciPyExpmBackend()

    @classmethod
    @abstractmethod
    def expm(cls, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential :math:`\\exp(\\mathbf{A})`, as a writable NumPy array whatever array type the
        backend returns.
        """
        return cls._writable(cls.backend.compute(m))

    @classmethod
    def expm_multiply(cls, a, b: np.ndarray, action: Callable = None, blocks: int = 1) -> np.ndarray:
        """
        Compute the action of the matrix exponential, :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``), via the
        active backend, as a writable NumPy array. An argument of 1-norm above ``_ACTION_STEP_NORM`` is applied in steps
        of doubling norm (:meth:`Backend._step_fractions() <phasegen.expm.Backend._step_fractions>`) that end once the
        vector is stationary (:meth:`Backend._stationary() <phasegen.expm.Backend._stationary>`).

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :param action: The action ``exp(a) @ b`` of one step, by default that of the active backend.
        :param blocks: The number of equal diagonal blocks of ``a`` outside of which it is strictly block triangular.
        :return: ``exp(a) @ b``.
        """
        action = cls.backend.compute_action if action is None else action
        norm = float(abs(a).sum(axis=0).max()) if a.shape[0] else 0.0

        if not (np.isfinite(norm) and norm > _ACTION_STEP_NORM):
            return cls._writable(action(a, b))

        abs_a, v, done = abs(a), b, 0.0
        growth = _GrowthBound(a, abs_a, blocks)
        for f in cls._step_fractions(norm):
            v = cls._writable(action(a * (f - done), v))
            done = f
            if f < 1 and cls._stationary(a, abs_a, v, 1 - f, growth(1 - f)):
                break

        return v

    @staticmethod
    def _step_fractions(norm: float) -> np.ndarray:
        """
        The cumulative fractions :math:`\\min(1, \\nu_0 (2^j - 1) / \\nu)`, :math:`j = 1, 2, \\ldots`, of an argument
        of 1-norm :math:`\\nu` after each step, with :math:`\\nu_0` the ``_ACTION_STEP_NORM``.

        :param norm: The 1-norm :math:`\\nu`.
        :return: The increasing fractions, the last equal to 1.
        """
        n_steps = int(np.ceil(np.log2(norm / _ACTION_STEP_NORM + 1)))
        return np.minimum(1.0, _ACTION_STEP_NORM * (2.0 ** np.arange(1, n_steps + 1) - 1) / norm)

    @staticmethod
    def _stationary(a, abs_a, v: np.ndarray, remaining: float, growth: float) -> bool:
        """
        Whether the change :math:`\\int_0^f \\exp(s \\mathbf{A}) \\mathbf{A}\\mathbf{v}\\, ds` over the remaining
        fraction :math:`f`, of 1-norm at most :math:`G f \\|\\mathbf{A}\\mathbf{v}\\|_1`, is within the unit roundoff
        :math:`u` relative to :math:`\\|\\mathbf{v}\\|_1` in every column: :math:`G f \\|\\mathbf{r}\\|_1 \\le u
        \\|\\mathbf{v}\\|_1`, where :math:`r_i = \\max(0, |(\\mathbf{A}\\mathbf{v})_i| - c\\, u\\, (|\\mathbf{A}|\\,
        |\\mathbf{v}|)_i)` drops what the rounding of the product and the error of the action that produced
        :math:`\\mathbf{v}` leave unresolved, :math:`c` is ``_STATIONARY_NOISE`` and :math:`|\\cdot|` is entrywise.

        :param a: Matrix :math:`\\mathbf{A}`.
        :param abs_a: Its entrywise absolute value :math:`|\\mathbf{A}|`.
        :param v: Vector or thin matrix :math:`\\mathbf{v}`.
        :param remaining: The fraction :math:`f`.
        :param growth: The factor :math:`G \\ge 1` of :class:`_GrowthBound` at :math:`f`.
        :return: Whether the vector is stationary over the remaining fraction.
        """
        r = np.maximum(0.0, np.abs(a @ v) - _STATIONARY_NOISE * _UNIT_ROUNDOFF * (abs_a @ np.abs(v))).sum(axis=0)

        with np.errstate(invalid='ignore', over='ignore'):
            return bool(np.all((r == 0) | (growth * remaining * r <= _UNIT_ROUNDOFF * np.abs(v).sum(axis=0))))

    @staticmethod
    def _writable(x) -> np.ndarray:
        """
        The array as a writable NumPy array, copied only if it is not one.

        :param x: Array of any array type.
        :return: Writable NumPy array.
        """
        return x if isinstance(x, np.ndarray) and x.flags.writeable else np.array(x)

    @classmethod
    def register(cls, backend: ExpmBackend) -> None:
        """
        Register a backend.

        .. deprecated::
            The backend registry is deprecated and will be removed; see :mod:`phasegen.expm`.
        """
        logger.warning(
            "Backend.register is deprecated and will be removed; phasegen will call SciPy directly."
        )
        cls.backend = backend
