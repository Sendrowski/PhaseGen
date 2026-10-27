"""
Moment evaluation engine: the Van Loan / closed-form / matrix-exponential machinery shared by every
phase-type distribution, mixed into :class:`~phasegen.distributions.phase_type.PhaseTypeDistribution`.
"""

import itertools
import logging
from collections import deque
from ..caching import cache
from math import comb, factorial
from typing import Dict, List, NamedTuple, Tuple, Iterable, Optional, Sequence, TYPE_CHECKING
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csg
from ..coalescent_models import StandardCoalescent
from ..demography import Epoch
from ..errors import ModelError
from .._blas import BlasThreadLimit
from .._block_expm import BlockTriangularExpm
from ..expm import Backend, SciPyExpmBackend
from ..rewards import Reward, CustomReward, UnfoldedSFSReward, FoldedSFSReward, UnitReward, CombinedReward
from ..settings import Settings
from ..state_space import BlockCountingStateSpace

from ._grid import _Propagators, _StepRuns
from ._gradient import _ClosedFormTape, _OccupationStep, _TapedAction, _VanLoanStep
from ._kingman import KingmanBlockKernel
from ._common import _make_hashable, _validate_order, _validate_reward_count, _validate_rewards, \
    _validate_start_time

if TYPE_CHECKING:
    from ..demography import Demography
    from ..lineage import LineageConfig
    from ..locus import LocusConfig
    from ..state_space import StateSpace
    from .phase_type import TreeHeightDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')

#: Sentinel for the ``perm`` argument of ``MomentEvaluator._lu_solver`` meaning "compute the block-triangular ordering
#: from ``A``". Callers factorizing one sparsity pattern at many diagonal shifts precompute the ordering once. It is
#: recognised by value, the ordering itself being an array or ``None``.
_AUTO_PERM = 'auto'

#: Smallest step of the sparse Van Loan action in units of the balancing factor. Block ``j`` of the extended vector
#: scales as the ``j``-th power of this step, so it bounds the range the blocks span.
_MIN_SCALED_STEP = 0.1

#: Floor of the balancing factor relative to the reciprocal geometric mean of the rates. A shorter step contributes
#: reward terms below double resolution, and a smaller factor would overflow the rebased extended vector.
_MIN_BALANCE = np.finfo(float).eps

#: Cost ratio :math:`K` of the dense exponentials against the sparse action of ``_ExpmCost``. With
#: ``_ACTION_STEP_NORM_OFFSET``, ``_TAPED_DENSE_COST`` and ``_MOMENT_COST``, fitted jointly on 3880 passes recorded on
#: each of an arm64 Mac and an x86-64 cluster node (closed-form and windowed moments, occupation times, time grids,
#: the forward pass of the second moments of spectra and gradients, 10 to 490 states), where it brings each kind of
#: pass within 2.5% of always taking the faster side on both machines.
_EXPM_COST_RATIO = 0.17

#: Fixed cost :math:`\gamma` of one sparse action in units of the step norm, the work besides the matrix-vector
#: products that the step norm counts.
_ACTION_STEP_NORM_OFFSET = 20

#: Factor on the cost of the dense exponentials of a pass recorded for its gradient, whose backward pass takes the
#: Fréchet derivative of each dense exponential and reverses each action.
_TAPED_DENSE_COST = 4

#: Dense size from which ``Settings.expm_method = 'auto'`` takes the action whatever the cost, which bounds the memory
#: of the dense exponentials.
_EXPM_ACTION_MIN_DIM = 1500


class _ExpmCost(NamedTuple):
    r"""
    A cost profile of the choice between dense exponentials and sparse actions. With :math:`n_a` actions of
    :math:`B` stacked vectors under operators of dimension :math:`d_a` and step norms :math:`s_i`, against
    :math:`B c` dense exponentials of size :math:`d_d` for each of :math:`n_d` steps, the dense side is cheaper where

    .. math::
        \left(\sum_i s_i + \gamma n_a\right) (B d_a + d_0) > K F F_\mathrm{tape} B c\, n_d\, d_d^3,

    with :math:`\gamma` the ``_ACTION_STEP_NORM_OFFSET``, :math:`K` the ``_EXPM_COST_RATIO``, :math:`F` and
    :math:`d_0` the fields of the profile, and :math:`F_\mathrm{tape}` the ``_TAPED_DENSE_COST`` for a pass recorded
    for its gradient and 1 otherwise.
    """
    #: The factor :math:`F` on the cost of the dense exponentials.
    dense: float

    #: The fixed cost :math:`d_0` of a product of the action in units of a row entry.
    offset: int

    def bound(self, d_action: int, d_dense: int, count: int, n_actions: int, n_dense: int, batch: int = 1,
              taped: bool = False) -> float:
        r"""
        The summed step norm above which the dense exponentials are cheaper, by :attr:`Settings.expm_method
        <phasegen.settings.Settings.expm_method>`: :math:`-\infty` for ``'dense'``, :math:`\infty` for ``'action'``
        and for ``'auto'`` from ``_EXPM_ACTION_MIN_DIM`` in ``d_dense`` on, and the cost model otherwise.

        :param d_action: The dimension :math:`d_a` of the operator of an action.
        :param d_dense: The size :math:`d_d` of each dense exponential.
        :param count: The number :math:`c` of dense exponentials per step and vector.
        :param n_actions: The number :math:`n_a` of actions.
        :param n_dense: The number :math:`n_d` of steps with their own dense exponentials.
        :param batch: The number :math:`B` of stacked vectors.
        :param taped: Whether the pass is recorded for its gradient.
        :return: The bound, negative where the dense exponentials are cheaper at any norm.
        """
        if Settings.expm_method == 'dense':
            return -np.inf

        if Settings.expm_method == 'action' or d_dense >= _EXPM_ACTION_MIN_DIM:
            return np.inf

        factor = _EXPM_COST_RATIO * self.dense * (_TAPED_DENSE_COST if taped else 1)
        return (factor * batch * count * n_dense * d_dense ** 3 / (batch * d_action + self.offset) -
                _ACTION_STEP_NORM_OFFSET * n_actions)


#: The cost profile of the moments, distribution functions and occupation times.
_MOMENT_COST = _ExpmCost(dense=1, offset=10000)

#: The cost profile of a batch of the Laplace transform, propagated as one stacked action of the batched truncated
#: Taylor series and exponentiated as one stack of Pade approximants. Fitted with ``_EXPM_COST_RATIO`` on 6981 batches
#: recorded on an x86-64 cluster node (marginal, joint and conditional transforms on one and two demes, two loci,
#: splits, pulses and out of Africa, 2 to 627 dimensions, 1 to 44 bounded epochs), where it brings each kind of batch
#: within 3% of always taking the faster side.
_TRANSFORM_COST = _ExpmCost(dense=0.36, offset=3000)

#: Smallest dimension from which the dense exponentials of the Van Loan matrices and of the augmented occupation
#: generator take ``BlockTriangularExpm``, one matrix at a time, rather than the generic exponential of the stack.
#: Over matrices of 2 to 4 diagonal blocks of 8 to 128 states each and step norms 0.1 to 100, it picked an exponential
#: within 10% of the faster one for 89 to 100% of them on an arm64 Mac and an x86-64 cluster node, and the
#: structured one was 1.1 to 2.5 times as fast from this dimension on.
_BLOCK_EXPM_MIN_DIM = 96

#: Largest number of entries of a stack of matrices exponentiated at once, about a megabyte, so that it stays in cache.
_MAX_STACK_ENTRIES = 2 ** 17

#: The SciPy backend, whose interval action the occupation grid of spectra takes whichever backend is registered.
_SCIPY_EXPM = SciPyExpmBackend()

#: Largest ratio of the pairwise coalescence and migration rates of an epoch that double precision resolves, see
#: ``MomentEvaluator._rate_spread``.
_MAX_RATE_SPREAD = 1e16


class MomentEvaluator:
    """Moment-evaluation methods of :class:`~phasegen.distributions.PhaseTypeDistribution`, described in its
    ``moment``."""

    # attributes provided by the host PhaseTypeDistribution this mixin is mixed into
    state_space: 'StateSpace'
    _tree_height: 'TreeHeightDistribution'
    demography: 'Demography'
    reward: Reward
    lineage_config: 'LineageConfig'
    locus_config: 'LocusConfig'
    _logger: logging.Logger
    _absorption_certain_cache: Dict[tuple, tuple]
    _alpha_support_cache: Dict[tuple, tuple]
    _last_epoch_reach_cache: Dict[tuple, tuple]
    _epoch_csr_cache: Dict[int, tuple]
    _epochs_cache: Dict[int, Tuple[List[Epoch], bool]]

    @staticmethod
    def _van_loan_matrix(R, S, k: int = 1, sparse: bool = False) -> 'sp.spmatrix | np.ndarray':
        r"""
        The Van Loan matrix of ``PhaseTypeDistribution.moment``, assembled directly as sparse CSR when ``sparse``.

        :param R: List of length k of reward vectors.
        :param S: Intensity matrix (dense or sparse, matching ``sparse``).
        :param k: The order of the moment.
        :param sparse: Whether to build a sparse matrix.
        :return: Van Loan matrix of :math:`(k + 1) \times (k + 1)` blocks.
        """
        if sparse:
            blocks = [[None] * (k + 1) for _ in range(k + 1)]
            for i in range(k + 1):
                blocks[i][i] = S
                if i < k:
                    blocks[i][i + 1] = sp.diags(R[i])
            return sp.bmat(blocks, format='csr')

        O = np.zeros_like(S)
        return np.block([
            [S if i == j else np.diag(R[i]) if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)
        ])

    @staticmethod
    def _block_expm(A: np.ndarray, n: int, scalar: Sequence[bool]) -> np.ndarray:
        """
        The dense exponentials of a stack of Van Loan matrices of
        :meth:`MomentEvaluator._van_loan_matrix() <phasegen.distributions._moments.MomentEvaluator._van_loan_matrix>` or
        of augmented occupation generators, block upper triangular of the structure of ``BlockTriangularExpm``. With the
        default backend, that class evaluates each matrix where ``_block_expm_applies``, and :func:`scipy.linalg.expm`
        the stack otherwise. Other backends evaluate each matrix. The strictly lower blocks are exactly zero. The
        rebases between balancing factors multiply them by powers of the factor ratio, so any rounding left in them
        would reach the moment.

        :param A: Stack of block upper triangular matrices of ``len(scalar)`` blocks of size ``n`` in each dimension,
            times their steps, of shape ``(m, dim, dim)``.
        :param n: The size of one block.
        :param scalar: Whether each diagonal block is a scalar multiple of the identity.
        :return: The exponentials of ``A``.
        """
        backend = Backend.backend
        default = type(backend) is SciPyExpmBackend and backend.precision == np.float64
        if default and MomentEvaluator._block_expm_applies(A.shape[-1]):
            block = BlockTriangularExpm(n, scalar)
            return np.stack([block.compute(a) for a in A])

        if default and np.abs(A).sum(axis=-2).max(initial=0.0) <= backend._max_norm:
            E = sla.expm(A)
        else:
            E = np.stack([expm(a) for a in A])
        for i in range(1, len(scalar)):
            E[:, i * n:(i + 1) * n, :i * n] = 0.0

        return E

    @staticmethod
    def _block_expm_applies(dim: int) -> bool:
        """
        Whether ``BlockTriangularExpm`` is faster than the generic exponential on matrices of dimension ``dim``, see
        ``_BLOCK_EXPM_MIN_DIM``.

        :param dim: The dimension of the matrices.
        :return: Whether to take ``BlockTriangularExpm``.
        """
        return dim >= _BLOCK_EXPM_MIN_DIM

    def _block_expms(
            self,
            builders: Sequence['Callable[[], np.ndarray]'],
            n: int,
            scalar: Sequence[bool]
    ) -> 'Iterator[np.ndarray]':
        """
        The exponentials of the matrices returned by ``builders``, in order, by ``_block_expm`` in stacks of at most
        ``_MAX_STACK_ENTRIES`` entries, or one by one where ``_block_expm_applies``. The epochs of one evaluation do not
        depend on the propagated vector, so their generic exponentials share a stack.

        :param builders: Functions returning the matrices, of ``len(scalar)`` blocks of size ``n`` in each dimension.
        :param n: The size of one block.
        :param scalar: Whether each diagonal block is a scalar multiple of the identity.
        :return: Iterator over the exponentials.
        """
        dim = n * len(scalar)
        size = 1 if self._block_expm_applies(dim) else max(1, _MAX_STACK_ENTRIES // dim ** 2)
        for i in range(0, len(builders), size):
            yield from self._block_expm(np.stack([b() for b in builders[i:i + size]]), n, scalar)

    @staticmethod
    def _block_triangular_order(A) -> Optional[np.ndarray]:
        """
        Permutation reordering ``A`` into block-triangular form by a topological sort (Kahn) of the condensation of
        its strongly connected components, derived from the sparsity pattern alone. On coalescent transient blocks
        most components are single states, since only migration and recombination create cycles, so a
        ``NATURAL``-ordered LU back-substitutes over the components with little fill.

        Returns ``None`` when a single component spans more than half of the states, so the caller keeps the default
        fill-reducing ordering.

        :param A: Square matrix (sparse or dense).
        :return: A permutation array, or ``None`` to fall back to the default ordering.
        """
        # the SCC structure depends only on the sparsity pattern; take magnitudes so a complex generator (the
        # reward-distribution path evaluates ``-T`` at complex reward shifts) is not cast to real by
        # ``connected_components`` — which would warn and could drop edges whose real part is zero.
        pattern = sp.csr_matrix(A).copy()
        pattern.data = np.abs(pattern.data)
        n = pattern.shape[0]

        n_scc, labels = csg.connected_components(pattern, directed=True, connection='strong')
        if n_scc <= 1:
            return None

        members = [[] for _ in range(n_scc)]
        for node, c in enumerate(labels):
            members[c].append(node)

        # a single dominant SCC means little triangular structure: keep the fill-reducing ordering
        if max(len(m) for m in members) > n // 2:
            return None

        # Kahn topological sort of the condensation
        coo = pattern.tocoo()
        succ = [set() for _ in range(n_scc)]
        indeg = np.zeros(n_scc, dtype=int)
        for u, v in zip(coo.row, coo.col):
            cu, cv = labels[u], labels[v]
            if cu != cv and cv not in succ[cu]:
                succ[cu].add(cv)
                indeg[cv] += 1

        queue = deque(c for c in range(n_scc) if indeg[c] == 0)
        order = []
        while queue:
            c = queue.popleft()
            order.append(c)
            for w in succ[c]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    queue.append(w)

        return np.fromiter((node for c in order for node in members[c]), dtype=int, count=n)

    @staticmethod
    def _solve_sparse(n_transient: int) -> bool:
        """
        Whether a linear solve over ``n_transient`` transient states takes the sparse path, at and above
        :attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`.
        The matrix exponentials are chosen independently, by ``_dense_expm_is_faster`` under
        :attr:`Settings.expm_method <phasegen.settings.Settings.expm_method>`.

        :param n_transient: Number of transient states the solve is over.
        :return: Whether to take the sparse path.
        """
        return n_transient >= Settings.closed_form_sparse_min_states

    @staticmethod
    def _lu_solver(A, sparse: bool, perm=_AUTO_PERM) -> 'Callable':
        """
        Factorize ``A`` once (sparse SuperLU or dense LU) and return a callable solving ``A x = b``. The sparse path
        applies the ordering of ``_block_triangular_order`` with ``NATURAL`` column ordering and permutes the
        right-hand side in and out. The ordering depends only on the sparsity pattern, so callers factorizing at many
        diagonal shifts pass it as ``perm``, and ``perm=None`` forces the default ordering.

        :param A: The matrix to factorize (sparse or dense, matching ``sparse``).
        :param sparse: Whether to use the sparse factorization.
        :param perm: The block-triangular permutation, or ``None`` for the default ordering, or ``_AUTO_PERM``
            (default) to compute it from ``A``.
        :return: Callable ``b -> x`` solving ``A x = b``, whose attribute ``T`` solves ``A^T x = b`` with the same
            factorization.
        """
        if sparse:
            if isinstance(perm, str):
                perm = MomentEvaluator._block_triangular_order(A)
            if perm is None:
                lu = spla.splu(sp.csc_matrix(A))
                solve = lambda b: lu.solve(np.asarray(b))
                solve.T = lambda b: lu.solve(np.asarray(b), trans='T')
                return solve

            inv = np.empty_like(perm)
            inv[perm] = np.arange(perm.size)
            lu = spla.splu(sp.csr_matrix(A)[perm][:, perm].tocsc(), permc_spec='NATURAL')
            solve = lambda b: lu.solve(np.asarray(b)[perm])[inv]
            solve.T = lambda b: lu.solve(np.asarray(b)[perm], trans='T')[inv]
            return solve

        lu = sla.lu_factor(A)
        solve = lambda b: sla.lu_solve(lu, b)
        solve.T = lambda b: sla.lu_solve(lu, b, trans=1)
        return solve

    @_make_hashable
    @cache
    def moment(
            self,
            k: int,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        r"""
        The :math:`k`-th moment of accumulated rewards, with the notation of
        :class:`~phasegen.distributions.PhaseTypeDistribution`.

        For rewards :math:`R_1, \dots, R_k` with reward vectors :math:`\mathbf{r}_1, \dots, \mathbf{r}_k`, the raw
        cross-moment :math:`\mathbb{E}[R_1 \cdots R_k]` is computed first. By default it is centered using the raw
        moments of all subsets of the rewards, so that ``moment(2)`` is the variance.

        .. rubric:: Single epoch

        For a time-homogeneous process accumulating rewards until absorption, the Green's matrix
        :math:`\mathbf{U} = (-\mathbf{T})^{-1}` holds in entry :math:`U_{xy}` the expected time spent in transient state
        :math:`y` when starting in :math:`x`. With the reward vectors restricted to the transient states,

        .. math::

            \mathbb{E}[R_1 \cdots R_k] = \sum_{\sigma} \boldsymbol{\alpha}_T
            \prod_{j=1}^{k} \mathbf{U} \operatorname{diag}(\mathbf{r}_{\sigma(j)})\; \mathbf{e}_T,

        where :math:`\sigma` runs over the :math:`k!` orderings of the rewards and the product is ordered by
        increasing :math:`j`. For a single reward this is
        :math:`\mathbb{E}[R^k] = k!\, \boldsymbol{\alpha}_T (\mathbf{U} \operatorname{diag}(\mathbf{r}))^k
        \mathbf{e}_T` (Hobolth et al., 2019).

        .. rubric:: Several epochs

        Within an epoch of duration :math:`\Delta`, the rewards accumulated in a given order are obtained from the
        block matrix of Van Loan (1978),

        .. math::

            \mathbf{V} =
            \begin{pmatrix}
                \mathbf{S} & \operatorname{diag}(\mathbf{r}_1) &        &                                   \\
                           & \mathbf{S}                        & \ddots &                                   \\
                           &                                   & \ddots & \operatorname{diag}(\mathbf{r}_k) \\
                           &                                   &        & \mathbf{S}
            \end{pmatrix}.

        The top-right :math:`|E| \times |E|` block :math:`[\cdot]_{1, k+1}` of its exponential is

        .. math::

            \big[e^{\mathbf{V} \Delta}\big]_{1, k+1}
            = \int_{0 < u_1 < \dots < u_k < \Delta}
            e^{\mathbf{S} u_1} \operatorname{diag}(\mathbf{r}_1) \cdots
            \operatorname{diag}(\mathbf{r}_k)\, e^{\mathbf{S} (\Delta - u_k)}\, \mathrm{d}\mathbf{u},

        so that reward :math:`j` is collected at time :math:`u_j`, with the process evolving under
        :math:`\mathbf{S}` between the collection times. The exponentials of the finite epochs are chained in time, and the unbounded
        last epoch is closed with the single-epoch formula. A start or end time shortens the durations accordingly.

        .. rubric:: Implementation

        - Matrix exponentials are applied to vectors as sparse actions (Al-Mohy and Higham, 2011) or formed densely
          as :attr:`Settings.expm_method <phasegen.settings.Settings.expm_method>` selects, by default by a cost
          comparison, with :math:`(k + 1)|E|` the size of the dense exponential.
        - :math:`\mathbf{U}` is never formed. A single LU factorization of :math:`-\mathbf{T}` serves all solves. It is
          sparse from :attr:`Settings.closed_form_sparse_min_states
          <phasegen.settings.Settings.closed_form_sparse_min_states>` transient states on, with the states ordered by
          the strongly connected components of the transition graph so that the factors stay nearly triangular.
        - The closed form requires :attr:`Settings.closed_form_last_epoch
          <phasegen.settings.Settings.closed_form_last_epoch>`, accumulation until absorption and certain absorption
          from every transient state of the last epoch that can carry mass. Otherwise the last epoch is integrated up to
          :attr:`TreeHeightDistribution.t_max <phasegen.distributions.TreeHeightDistribution.t_max>`.
        - A central moment of a single reward takes all lower raw moments from the closed form of the highest order,
          whose extended vector holds the moment of the last :math:`i` rewards divided by :math:`i!\, \lambda^i` in a
          block of its own, where :math:`\lambda` is the balancing factor of the blocks.
        - Spectra share one computation across bins. The expected occupation times :math:`\mathbf{m}` of the
          transient states, which equal :math:`\boldsymbol{\alpha}_T \mathbf{U}` in a single epoch, give every bin
          mean as :math:`\mathbf{m}\, \mathbf{r}_j`.
        - For one population and one locus under the :class:`~phasegen.coalescent_models.StandardCoalescent`, the
          first moments and the second moments of the SFS bins are computed on the smaller lineage-counting state
          space (:attr:`Settings.flatten_block_counting <phasegen.settings.Settings.flatten_block_counting>`).

        .. rubric:: References

        Al-Mohy, A. H. and Higham, N. J. (2011). Computing the action of the matrix exponential, with an application
        to exponential integrators. SIAM Journal on Scientific Computing 33(2), 488-511.

        Hobolth, A., Siri-Jégousse, A. and Bladt, M. (2019). Phase-type distributions in population genetics.
        Theoretical Population Biology 127, 16-32.

        Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. IEEE Transactions on Automatic
        Control 23(3), 395-404.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :param end_time: The end time :math:`t_\mathrm{end}`. By default, the end time of the distribution, or
            absorption. An infinite end time accumulates until absorption.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: The :math:`k`-th moment.
        :raises TypeError: If ``k`` is not a number or an entry of ``rewards`` is not a
            :class:`~phasegen.rewards.Reward`.
        :raises ValueError: If ``k`` is not a non-negative integer, if a single :class:`~phasegen.rewards.Reward` is
            passed or the number of rewards differs from ``k``, or if the start time is negative, exceeds the end time,
            or lies beyond the time of almost sure absorption.
        :raises ModelError: If the population sizes and migration rates are too far apart for a reliable evaluation,
            or if the moment is not a number.
        """
        k = _validate_order(k)
        _validate_rewards(rewards, k)
        start_time, end_time = self._resolve_window(start_time, end_time)

        # a window starting after 0 is accumulated directly from the entry distribution propagated to its start
        m = float(MomentEvaluator.accumulate(
            self,
            k=k,
            end_times=[end_time],
            rewards=rewards,
            center=center,
            permute=permute,
            start_time=start_time
        )[0])

        if np.isnan(m):
            raise ModelError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return m

    def _resolve_window(self, start_time: float = None, end_time: float = None) -> Tuple[float, float]:
        """
        Resolve and validate the accumulation window of a moment.

        :param start_time: The start time. By default, the start time of the distribution.
        :param end_time: The end time. By default, the end time of the distribution, or infinity, which accumulates
            until absorption.
        :return: The start and end time.
        :raises ValueError: If the start time is negative, exceeds the end time, or lies beyond the time of almost
            sure absorption.
        """
        if start_time is None:
            start_time = self._tree_height.start_time

        if end_time is None:
            end_time = np.inf if self._tree_height.end_time is None else self._tree_height.end_time

        _validate_start_time(start_time)

        if not end_time >= 0:
            raise ValueError(f"End time must be greater than or equal to 0, got {end_time}.")

        if start_time > 0 and np.isinf(end_time):
            t_absorption = self._get_time_to_absorption()

            if start_time > t_absorption:
                raise ValueError(
                    f"The window start time ({start_time:.1f}) lies beyond the time of almost sure absorption "
                    f"({t_absorption:.1f}), so the accumulation window is empty."
                )

        if end_time < start_time:
            raise ValueError("End time must be greater than equal start time.")

        return start_time, end_time

    @staticmethod
    def _get_regularization_factor(S: np.ndarray, duration: float = np.inf, scale: float = 1.0) -> float:
        """
        The balancing factor of the Van Loan matrix, the reciprocal geometric mean of the positive rates of ``S``
        capped at ``duration``, times the reward scale ``scale`` (see :meth:`_reward_scale`), or 1 when
        ``Settings.regularize`` is disabled. Scaling ``S`` by it and the step by its inverse divides the reward blocks
        by the factor, which the callers undo by multiplying the moment by its ``k``-th power. The reward scale brings
        the largest reward to one, since larger reward blocks force the scaling-and-squaring of the exponential into
        more squarings than its generator blocks need. The cap keeps the scaled step at or above one: the block of
        order ``j`` of the exponential scales as the ``j``-th power of the scaled step, and below one the higher orders
        fall under the resolution of double precision. The cap is floored at ``_MIN_BALANCE`` times the factor.

        :param S: Intensity matrix.
        :param duration: Length of the epoch within the accumulation window.
        :param scale: The reward scale of the Van Loan matrix.
        :return: Regularization factor.
        """
        if not Settings.regularize:
            return 1.0

        # obtain positive rates (for a sparse matrix, the positive stored entries)
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        factor = 10 ** - np.log10(rates).mean()

        if duration > 0:
            factor = min(factor, max(duration, _MIN_BALANCE * factor))

        return scale * factor

    @staticmethod
    def _reward_scale(R: Sequence[np.ndarray]) -> float:
        """
        The largest absolute entry of the reward vectors ``R``, or 1 where it is not positive and finite.

        :param R: Reward vectors.
        :return: Reward scale.
        """
        scale = max((float(np.max(np.abs(r), initial=0.0)) for r in R), default=0.0)

        return scale if 0 < scale < np.inf else 1.0

    def _balance(self, epoch: 'Epoch', start: float, end: float, scale: float) -> float:
        """
        The regularization factor of the current rate matrix for the part of ``epoch`` within ``[start, end]``.

        :param epoch: The epoch the state space is set to.
        :param start: Start of the accumulation window.
        :param end: End of the accumulation window.
        :param scale: The reward scale of the Van Loan matrix.
        :return: Regularization factor.
        """
        duration = min(epoch.end_time, end) - max(epoch.start_time, start)

        return self._get_regularization_factor(self.state_space.S, duration, scale)

    @staticmethod
    def _rebase(z: np.ndarray, lamb: float, lamb_new: float, k: int, n: int) -> float:
        """
        Rebase the extended vector from one balancing factor onto another, in place.

        Block ``j`` of ``z`` holds its value divided by ``lamb ** (k - j)``, so changing the factor multiplies block
        ``j`` by ``(lamb / lamb_new) ** (k - j)``. This is an exact diagonal similarity: it leaves the represented
        vector unchanged and only moves which power of the factor each block carries.

        :param z: Extended vector of ``(k + 1)`` blocks of length ``n``, modified in place.
        :param lamb: The factor the blocks are currently stored against.
        :param lamb_new: The factor to store them against.
        :param k: The order of the moment.
        :param n: The number of states, the length of one block.
        :return: ``lamb_new``, the factor now in force.
        """
        if lamb_new == lamb:
            return lamb

        ratio = lamb / lamb_new
        for j in range(k + 1):
            z[j * n:(j + 1) * n] *= ratio ** (k - j)

        return lamb_new

    @staticmethod
    def _rebase_forward(w: np.ndarray, lamb: float, lamb_new: float, k: int, n: int) -> float:
        """
        Rebase the forward extended vector ``w = alpha_ext Q`` of the extended propagator ``Q`` from one balancing
        factor onto another, in place, the row counterpart of :meth:`_rebase`. Block ``j`` of ``w`` is row block 0 of
        ``Q``, so it carries the factor to the power ``-j`` and is multiplied by ``(lamb / lamb_new) ** j``. An exact
        diagonal similarity.

        :param w: Extended row vector of ``(k + 1)`` blocks of length ``n``, modified in place.
        :param lamb: The factor the blocks are currently stored against.
        :param lamb_new: The factor to store them against.
        :param k: The order of the moment.
        :param n: The number of states, the length of one block.
        :return: ``lamb_new``, the factor now in force.
        """
        if lamb_new == lamb:
            return lamb

        ratio = lamb / lamb_new
        for j in range(1, k + 1):
            w[j * n:(j + 1) * n] *= ratio ** j

        return lamb_new

    def _action_operator(self, rewards: Sequence[Reward], k: int, i_epoch: int) -> Tuple:
        """
        The transposed sparse Van Loan matrix of the current epoch for the left vector action, unbalanced, with its
        stored entries split into the generator blocks and the reward blocks, so that :meth:`_advance_action` scales
        it to any step and balancing factor without reassembling it.

        :param rewards: Sequence of k rewards.
        :param k: The order of the moment.
        :param i_epoch: The epoch number, for the stability warning.
        :return: The matrix, its data restricted to the generator blocks and to the reward blocks (zero elsewhere), the
            balancing factor of the epoch before capping and the reward scale it includes.
        """
        n = self.state_space.k
        S = self.state_space.S
        self._check_numerical_stability(S, i_epoch)
        r_vecs = [np.asarray(r._get(state_space=self.state_space), dtype=float) for r in rewards]
        scale = self._reward_scale(r_vecs)
        Vt = self._van_loan_matrix(r_vecs, sp.csr_matrix(S), k, sparse=True).T.tocsr()

        # entries off the block diagonal are the reward blocks
        rows = np.repeat(np.arange(Vt.shape[0]), np.diff(Vt.indptr))
        reward = rows // n != Vt.indices // n

        return (Vt, np.where(reward, 0.0, Vt.data), np.where(reward, Vt.data, 0.0),
                self._get_regularization_factor(S, scale=scale), scale)

    def _advance_action(
            self,
            w: np.ndarray,
            tau: float,
            lamb: float,
            op: Tuple,
            k: int,
            count: int = 1
    ) -> Tuple[np.ndarray, float]:
        """
        Advance the forward extended vector ``count`` times by ``tau`` within the current epoch by the sparse
        matrix-exponential action, several steps by one interval action. Each step is balanced on the factor of the
        epoch capped at the reward scale times ``tau / _MIN_SCALED_STEP``, since block ``j`` of the vector gains the
        ``j``-th power of the scaled step, and floored at ``_MIN_BALANCE`` times the factor.

        :param w: Extended row vector of ``(k + 1)`` blocks, stored against ``lamb`` and rebased in place.
        :param tau: The step, non-positive for none.
        :param lamb: The factor ``w`` is stored against.
        :param op: The operator of :meth:`_action_operator` for the current epoch.
        :param k: The order of the moment.
        :param count: The number of steps.
        :return: The vectors after each step, of shape ``(count, len(w))``, and the factor they are stored against.
        """
        if tau <= 0:
            return np.repeat(w[None], count, axis=0), lamb

        Vt, gen, rew, factor, scale = op
        if Settings.regularize:
            lamb_step = min(factor, max(scale * tau / _MIN_SCALED_STEP, _MIN_BALANCE * factor))
            lamb = self._rebase_forward(w, lamb, lamb_step, k, self.state_space.k)

        A = sp.csr_matrix((tau * gen + (tau / lamb) * rew, Vt.indices, Vt.indptr), shape=Vt.shape)

        if count == 1:
            return Backend.expm_multiply(A, w, blocks=k + 1)[None], lamb

        return Backend.expm_multiply_interval(
            A, w, float(count), count, advance=lambda v, s: Backend.expm_multiply(A * s, v, blocks=k + 1)
        ), lamb

    def _rate_spread(self, epoch: 'Epoch', duration: float = np.inf) -> float:
        r"""
        The ratio of the largest to the smallest of the pairwise coalescence rates and migration rates of ``epoch``.
        The pairwise coalescence rate of a deme of size :math:`N` is :math:`\lambda_{2,2} / \tau(N)`, with
        :math:`\lambda_{2,2}` the rate at which two lineages merge under the coalescent model and :math:`\tau(N)` its
        time scale. Keyed on these rates, not on the rate matrix, whose range multiple-merger models widen
        legitimately. Over a finite ``duration`` :math:`d`, a rate :math:`r` with :math:`r d` at most
        ``1 / _MAX_RATE_SPREAD`` is left out, its events over the epoch lying below double precision.

        :param epoch: The epoch.
        :param duration: The time over which the rates act.
        :return: The ratio, 1 without positive rates.
        """
        model = self.state_space.model

        rates = [model._get_rate(b=2, k=2) / model._get_timescale(v) for v in epoch.pop_sizes.values() if v > 0]
        rates += [v for v in epoch.migration_rates.values() if v > 0]
        rates = [r for r in rates if r > 0 and r * duration > 1 / _MAX_RATE_SPREAD]

        return max(rates) / min(rates) if rates else 1

    def _check_demography_conditioning(self, epochs: Iterable['Epoch']) -> None:
        """
        Fail fast when the rates of an epoch span more than double precision over its duration (see
        ``_rate_spread``), which makes the matrix exponentials of the epoch, the absorption-time search and the
        closed-form solve unreliable.

        :param epochs: The epochs, the last of which may be held until absorption.
        :raises ModelError: if the pairwise coalescence and migration rates of an epoch differ by a factor of more than
            ``_MAX_RATE_SPREAD``.
        """
        for epoch in epochs:
            ratio = self._rate_spread(epoch, epoch.end_time - epoch.start_time)

            if ratio > _MAX_RATE_SPREAD:
                raise ModelError(
                    "The demography is too ill-conditioned to reliably compute the time of almost sure absorption and "
                    f"the moments: the pairwise coalescence and migration rates of epoch {epoch.index} differ by a "
                    f"factor of {ratio:.1e}. Use less extreme parameters, or set the end time manually (see "
                    "``Coalescent.end_time``)."
                )

    def _check_numerical_stability(self, S: np.ndarray, epoch: int) -> None:
        """
        Warn about potential numerical instability when the inverse time scales the exponentials and solves resolve
        differ by more than 10 orders of magnitude, once per epoch. These are the total exit rates of the transient
        states and, for each communicating class of several states, the largest total rate out of the class from one of
        its states, which fast transitions within the class, such as migration, hide from the exit rates. A small
        transition rate beside larger ones out of the same state, such as a rare multiple merger, leaves them unchanged.

        :param S: (Regularized) intensity matrix of the current state space.
        :param epoch: Epoch number.
        """
        warned = self.__dict__.setdefault('_stability_warned', set())
        if epoch in warned:
            return

        # the spread does not change with a scalar multiple of the rates, so each epoch is checked once per state space
        memo = self._state_space_memo('_stability_checked')
        ss = self.state_space
        if id(ss) not in memo or memo[id(ss)][0] is not ss:
            memo[id(ss)] = (ss, set())

        checked = memo[id(ss)][1]
        if epoch in checked:
            return
        checked.add(epoch)

        exit_rates = -np.asarray(S.diagonal()).ravel()[~self.state_space.absorbing]
        rates = exit_rates[exit_rates > 0]

        # only migration and recombination create cycles, so a single population at a single locus has classes of one
        # state each
        if self.lineage_config.n_pops > 1 or self.locus_config.n > 1:
            if sp.issparse(S):
                A = S.tocoo()
                off = (A.row != A.col) & (A.data > 0)
                rows, cols, data = A.row[off], A.col[off], A.data[off]
            else:
                S = np.asarray(S)
                rows, cols = np.nonzero(S > 0)
                off = rows != cols
                rows, cols = rows[off], cols[off]
                data = S[rows, cols]

            n = S.shape[0]
            n_classes, labels = csg.connected_components(
                sp.csr_matrix((data, (rows, cols)), shape=S.shape), directed=True, connection='strong'
            )

            if n_classes < n:
                leaving = labels[rows] != labels[cols]
                out = np.bincount(rows[leaving], weights=data[leaving], minlength=n)
                escape = np.zeros(n_classes)
                np.maximum.at(escape, labels, out)
                rates = np.concatenate([rates, escape[escape > 0]])

        if rates.size and rates.min() / rates.max() < 1e-10:
            warned.add(epoch)
            self._logger.warning(
                "Intensity matrix in epoch %d has total exit rates of its states and communicating classes that "
                "differ by more than 10 orders of magnitude: min: %g, max: %g. This may lead to numerical "
                "instability, despite matrix regularization.",
                epoch, rates.min(), rates.max()
            )

    @BlasThreadLimit()
    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r""" The :math:`k`-th moment accumulated from the start time :math:`t_\mathrm{start}` to each end time
        :math:`t_\mathrm{end}` in ``end_times``, as described in :meth:`PhaseTypeDistribution.moment()
        <phasegen.distributions.PhaseTypeDistribution.moment>`. An end time at or before
        :math:`t_\mathrm{start}` accumulates no reward.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :return: The moment at each end time.
        :raises ValueError: If ``k`` is not a non-negative integer, if the number of rewards differs from it, if the
            start time is negative, or if it lies beyond the time of almost sure absorption while an end time is
            infinite.
        """
        k = _validate_order(k)
        end_times = tuple(end_times)

        if start_time is None:
            start_time = self._tree_height.start_time

        _validate_start_time(start_time)

        if start_time > 0 and np.isinf(end_times).any():
            self._resolve_window(start_time, np.inf)

        if rewards is None:
            rewards = [self.reward] * k

        _validate_reward_count(rewards, k)

        # the raw moments of all orders of one reward until absorption come from the closed form of the highest
        if center and k > 1 and self._powers_apply(k, rewards, end_times):
            self._logger.debug("accumulate (k=%d): centering with the raw moments of one closed form", k)
            raw = self._raw_powers(k, rewards[0], start_time)
            central = sum(comb(k, i) * raw[i] * (-raw[1]) ** (k - i) for i in range(k + 1))

            return np.full(len(end_times), central)

        # center moments around the mean
        if center and k > 1:
            self._logger.debug("accumulate (k=%d): centering (subtracting lower-order moment products)", k)

            components = []

            # first order moments
            means = [self._accumulate_raw(1, (rewards[i],), end_times, True, start_time) for i in range(k)]

            for i in range(k + 1):
                # iterate over all possible subsets of rewards of size i
                for indices in itertools.combinations(range(k), i):
                    # joint moment
                    mu_i = self._accumulate_raw(i, tuple(rewards[j] for j in indices), end_times, permute, start_time)

                    # product of means of remaining rewards
                    mu1 = np.prod([means[j] for j in range(k) if j not in indices], axis=0)

                    components += [(-1) ** (k - i) * mu_i * mu1]

            return np.sum(components, axis=0)

        return self._accumulate_raw(k, rewards, end_times, permute, start_time)

    def _powers_apply(self, k: int, rewards: Sequence[Reward], end_times: Sequence[float]) -> bool:
        """
        Whether the raw moments up to order ``k`` are read from one closed form by :meth:`_raw_powers`: where the
        rewards are equal, the end times infinite, :attr:`Settings.closed_form_last_epoch
        <phasegen.settings.Settings.closed_form_last_epoch>` set, the moment not flattened and absorption certain in
        the last epoch.

        :param k: The order.
        :param rewards: The rewards.
        :param end_times: The end times.
        :return: Whether they are.
        """
        return (
                all(r == rewards[0] for r in rewards[1:]) and
                bool(np.isinf(end_times).all()) and
                Settings.closed_form_last_epoch and
                not self._flattening_applies(k, rewards) and
                self._absorption_certain_in_last_epoch(k)
        )

    @cache
    def _last_epoch_solver(self, space: int, n_epochs: int, states: bytes, sparse: bool) -> 'Callable':
        """
        The factorization of ``-T`` of the last epoch on a set of transient states, as ``_lu_solver``, memoized by
        state space, epoch list and states. The caller sets the last epoch.

        :param space: The identity of the state space.
        :param n_epochs: The length of the epoch list (see ``_get_epochs_until_unbounded``), which identifies it.
        :param states: The transient states, as the bytes of their indices.
        :param sparse: Whether to factorize sparsely.
        :return: The solver.
        """
        idx_t = np.frombuffer(states, dtype=np.intp)

        return self._lu_solver(-self._transient_block(idx_t, sparse=sparse), sparse)

    @_make_hashable
    @cache
    def _raw_powers(self, k: int, reward: Reward, start_time: float) -> np.ndarray:
        r"""
        The raw moments :math:`\mathbb{E}[R^i]` for :math:`i = 0, \dots, k` of a reward accumulated from the start time
        until absorption, from the trailing blocks of the closed form of order ``k``.

        :param k: The highest order.
        :param reward: The reward.
        :param start_time: The start time.
        :return: The moments.
        :raises ModelError: If a moment is not a number.
        """
        Reward._check_accumulable(self.state_space, (reward,) * k)
        raw = self._accumulate_closed_form(k, (reward,) * k, start_time, suffixes=True)

        if np.isnan(raw).any():
            raise ModelError(
                "NaN value encountered when computing moment. This is likely due to an ill-conditioned rate matrix."
            )

        return raw

    def _accumulate_raw(
            self,
            k: int,
            rewards: Sequence[Reward],
            end_times: Iterable[float],
            permute: bool,
            start_time: float
    ) -> np.ndarray:
        r"""
        The raw :math:`k`-th cross-moment of ``accumulate``, for any order :math:`k \ge 0`.

        :param k: The order of the moment, 0 giving one.
        :param rewards: Sequence of :math:`k` rewards.
        :param end_times: The end times at which to evaluate the moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :param start_time: The start time.
        :return: The raw moment at each end time.
        """
        if k == 0:
            return np.ones_like(list(end_times))

        # every ordering of identical rewards is the same ordering
        if permute and any(r != rewards[0] for r in rewards[1:]):
            # get all possible permutations of rewards
            permutations = list(itertools.permutations(rewards))

            # compute average over all permutations
            return np.sum(
                [self._accumulate(k, tuple(end_times), r, start_time=start_time) for r in permutations], axis=0
            ) / len(permutations)

        return self._accumulate(k, tuple(end_times), rewards, start_time=start_time)

    @_make_hashable
    @cache
    def _accumulate_flattened(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None,
            start_time: float = 0.0
    ) -> np.ndarray:
        r"""
        Evaluate the kth (non-central) moment at different end times using the lineage counting state space. The
        second moment of the bin rewards ``a`` and ``b`` collected in this order is
        :math:`2 \sum_{i \in a, j \in b} X_{ij}`, see ``KingmanBlockKernel.contract``.

        :param k: The order of the moment, 1 or 2.
        :param end_times: Sequence of end times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time from which to start accumulation.
        :return: The moment accumulated at the specified times or time.
        :raises ValueError: If the moment is not flattened, see ``_flattening_applies``.
        """
        rewards = (self.reward,) * k if rewards is None else tuple(rewards)

        if not self._flattening_applies(k, rewards):
            raise ValueError(
                "Flattened accumulation is only supported for the first moment and for the second moment of SFS bins "
                "on the block-counting state space of a single population and a single locus under the standard "
                "coalescent."
            )

        if k == 2:
            n = self.lineage_config.n
            a, b = (np.asarray(self._sfs_reward_block_sizes(r, n)) - 1 for r in rewards)
            cross = self._tree_height._level_cross_moments(tuple(end_times), float(start_time))[1]

            return 2 * cross[:, a][:, :, b].sum(axis=(1, 2))

        weights = self._flattened_weights(rewards[0])

        # Create a custom reward that returns the weights.
        weighted_reward = CustomReward(lambda _: weights)

        return self._tree_height._accumulate(
            k=k, end_times=end_times, rewards=(weighted_reward,), start_time=start_time
        )

    @_make_hashable
    @cache
    def _level_cross_moments(self, end_times: Sequence[float], start_time: float) -> np.ndarray:
        r"""
        The ordered cross-moments :math:`X_{ij}` of ``KingmanBlockKernel.contract`` of the branch lengths subtending
        :math:`i` and :math:`j` samples, from the ordered two-point occupation :math:`M_{ab} = \mathbb{E}
        \int\!\!\int_{t_\mathrm{start} < s < u < t} \mathbb{1}\{X_s = a\} \mathbb{1}\{X_u = b\}\, \mathrm{d}s\,
        \mathrm{d}u` of the transient states :math:`a, b` of the lineage-counting state space of this distribution, with
        :math:`X_s` the state at time :math:`s` and :math:`t` an end time. ``_bin_second_moments`` of the indicator
        rewards of the states gives :math:`M + M^\top`, and :math:`M` is upper triangular, the states being ordered by
        decreasing number of lineages.

        :param end_times: The end times.
        :param start_time: The start time.
        :return: The expected occupation of each transient state, of shape ``(len(end_times), n - 1)``, and
            :math:`X_{ij}`, of shape ``(len(end_times), n - 1, n - 1)``.
        """
        idx_t = np.where(~self.state_space.absorbing)[0]
        levels = [CustomReward(lambda space, a=a: (np.arange(space.k) == a).astype(float)) for a in idx_t]
        kernel = KingmanBlockKernel(self.lineage_config.n)

        means = np.zeros((len(end_times), len(idx_t)))
        cross = np.zeros((len(end_times), len(idx_t), len(idx_t)))
        for i, end_time in enumerate(end_times):
            if end_time > start_time:
                means[i], occupation = self._bin_second_moments(levels, start_time, end_time)
                occupation = np.triu(occupation)
                occupation[np.diag_indices_from(occupation)] /= 2
                cross[i] = kernel.contract(occupation)

        return means, cross

    def _flattened_weights(self, reward: Reward) -> np.ndarray:
        """
        The reward of :meth:`_accumulate_flattened` on the lineage-counting state space, indexed by ``n - k`` for
        ``k`` lineages: the expected reward of the block-counting states with ``k`` lineages.

        :param reward: The reward on the block-counting state space.
        :return: Weight vector over the lineage-counting states.
        """
        n = self.lineage_config.n

        # Prefer the closed-form Kingman block-size weights, which never build the (p(n)-state) block-counting space.
        weights = self._flattened_sfs_weights(reward, n)
        if weights is not None:
            self._logger.debug(
                "flattening block-counting onto the lineage-counting state space (%d states) via the closed-form "
                "Kingman block-size weights", self._tree_height.state_space.k
            )
        else:
            # fall back to weighting by the per-lineage state probabilities of the block-counting space
            r = reward._get(self.state_space)
            probs = self.state_space._state_probs
            weights = np.zeros(n)
            for i, s in enumerate(self.state_space.states):
                weights[n - s.lineages.sum()] += probs[i] * r[i]
            self._logger.debug(
                "flattening block-counting state space (%d states) onto the lineage-counting state space (%d states)",
                len(self.state_space.states), self._tree_height.state_space.k
            )

        return weights

    def _flattened_sfs_weights(self, reward: Reward, n: int) -> Optional[np.ndarray]:
        """
        Flattening weights of ``PhaseTypeDistribution.moment`` per lineage count (indexed by ``n - k``), from the
        uniform distribution of Kingman block sizes over compositions, without building the block-counting space.
        Returns ``None`` for a reward that is not a unit multiple of an unfolded or folded SFS bin reward.

        :param reward: The reward whose flattened weights to compute.
        :param n: The number of lineages.
        :return: Weight vector indexed by ``n - k``, or ``None`` if the reward is unsupported.
        """
        block_sizes = self._sfs_reward_block_sizes(reward, n)
        if block_sizes is None:
            return None

        weights = np.zeros(n)
        for k in range(2, n + 1):  # k = 1 (the grand MRCA, a single size-n block) carries no polymorphic SFS reward
            denom = comb(n - 1, k - 1)
            weights[n - k] = sum(
                k * comb(n - b - 1, k - 2) / denom for b in block_sizes if 0 <= k - 2 <= n - b - 1
            )
        return weights

    @staticmethod
    def _sfs_reward_block_sizes(reward: Reward, n: int) -> Optional[List[int]]:
        """
        The block sizes an SFS reward counts: ``[i]`` for an unfolded bin ``i``, ``[i, n - i]`` for a folded bin
        (just ``[i]`` when ``i == n - i``). A :class:`~phasegen.rewards.CombinedReward` is unwrapped when it is a
        product of unit rewards and a single SFS reward (as built by the SFS moment path). Returns ``None`` for
        anything else, including the bins without polymorphic blocks, which fall back to the block-counting traversal.

        :raises ValueError: if the SFS bin index lies outside ``0, ..., n``.
        """
        if isinstance(reward, CombinedReward):
            non_unit = [r for r in reward.rewards if not isinstance(r, UnitReward)]
            if len(non_unit) != 1:
                return None
            reward = non_unit[0]

        if isinstance(reward, (FoldedSFSReward, UnfoldedSFSReward)):
            return reward._block_sizes(n) or None

        return None
    @_make_hashable
    @cache
    def _accumulate(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None,
            start_time: float = 0.0
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times.

        :param k: The order of the moment.
        :param end_times: Sequence of end times or end time when to evaluate the moment. A NaN end time gives a NaN
            moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time from which to start accumulation. By default, ``0`` (accumulation from the origin).
        :return: The moment accumulated at the specified times or time.
        """
        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        elif len(rewards) != k:
            raise ValueError(f"Number of rewards must be {k}.")

        end_times = np.array(end_times, dtype=float)

        undefined = np.isnan(end_times)
        if undefined.any():
            moments = np.full(end_times.shape, np.nan)
            if not undefined.all():
                moments[~undefined] = self._accumulate(k, tuple(end_times[~undefined]), rewards, start_time)
            return moments

        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        # flattening takes precedence over the closed form (it shrinks the state space, which dominates the cost) and
        # re-enters this method on the lineage-counting state space, where the flattened reward is checked
        if self._flattening_applies(k, rewards):
            self._logger.debug("accumulate (k=%d): flattened block-counting", k)
            return self._accumulate_flattened(k, end_times, rewards, start_time)

        Reward._check_accumulable(self.state_space, rewards)

        # infinite end times accumulate until absorption, in closed form when absorption is certain in the last
        # epoch and otherwise over the estimated absorption time
        infinite = np.isinf(end_times)
        if infinite.any():
            if Settings.closed_form_last_epoch and self._absorption_certain_in_last_epoch(k):
                self._logger.debug("accumulate (k=%d): closed-form last epoch", k)
                moments = np.empty(end_times.shape)
                moments[infinite] = self._accumulate_closed_form(k, rewards, start_time)
                if not infinite.all():
                    moments[~infinite] = self._accumulate(k, tuple(end_times[~infinite]), rewards, start_time)
                return moments

            end_times = np.where(infinite, self._get_time_to_absorption(), end_times)

        return self._accumulate_windowed(k, float(start_time), end_times, rewards)

    def _propagate_plain(self, p: np.ndarray, tau: float, use_action: bool) -> np.ndarray:
        r"""
        Propagate a state distribution ``p`` forward by ``tau`` under the *current* epoch's plain generator
        :math:`\mathbf{S}`: :math:`\mathbf{p} \mapsto \mathbf{p}\,\exp(\mathbf{S}\tau)`. Used to carry the entry
        distribution to the start of a moment window before the windowed Van Loan accumulation. Returns ``p``
        unchanged for a non-positive ``tau``.

        :param p: State distribution (row vector).
        :param tau: Elapsed time within the current epoch.
        :param use_action: Whether to apply the sparse matrix-exponential action instead of forming the dense
            exponential.
        :return: The propagated distribution.
        """
        if tau <= 0:
            return p

        S = self.state_space.S
        if use_action:
            # p exp(S tau) = (exp((S tau)^T) p^T)^T, so apply the action to the transposed generator
            St = (S * tau).T.tocsc() if sp.issparse(S) else sp.csc_matrix(np.asarray(S) * tau).T.tocsc()
            return Backend.expm_multiply(St, p)

        return p @ expm(self._dense_rate_matrix() * tau)

    def _dense_van_loan(self, R: Sequence[np.ndarray], lamb: float, k: int, i_epoch: int) -> np.ndarray:
        """
        The dense Van Loan matrix of the current epoch with the intensity matrix balanced by ``lamb``.

        :param R: The k reward vectors.
        :param lamb: The balancing factor.
        :param k: The order of the moment.
        :param i_epoch: The epoch number, for the stability warning.
        :return: The Van Loan matrix.
        """
        S = self._dense_rate_matrix() * lamb
        self._check_numerical_stability(S, i_epoch)

        return self._van_loan_matrix(R, S, k)

    def _accumulate_windowed(
            self,
            k: int,
            start_time: float,
            end_times: np.ndarray,
            rewards: Sequence[Reward]
    ) -> np.ndarray:
        """
        Raw moment of a single reward ordering over the window ``[start_time, t]`` for each finite ``t`` in
        ``end_times``: propagates ``alpha`` to ``start_time`` with the plain generator, then threads the row vector
        holding it in its first block through the epochs by the Van Loan exponential and reads off its product with
        ``e`` in the last block at each end time. The exponential is dense or a sparse action as
        ``_dense_expm_is_faster`` decides from ``_grid_steps``. Within an epoch, the end times are
        taken as runs of equal steps (``_StepRuns``), with one dense exponential per step length or one interval action
        per run. For ``k >= 2`` the difference of two moments accumulated from 0 would omit the cross terms.

        :param k: The order of the moment.
        :param start_time: The non-negative window start time.
        :param end_times: The finite window end times.
        :param rewards: Sequence of k rewards (a single ordering).
        :return: The windowed moment accumulated over ``[start_time, t]`` for each ``t`` in ``end_times``.
        """
        t_sorted: np.ndarray = np.sort(end_times)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)
        self.state_space.update_epoch(epoch)
        n = self.state_space.k

        dim = (k + 1) * n
        use_action = not self._dense_expm_is_faster(dim, dim, 1, *self._grid_steps(start_time, t_sorted))

        # --- propagate the entry distribution to the window start via the plain generator ---
        alpha_start = np.asarray(self.state_space.alpha, dtype=float)
        u_prev = 0.0
        while start_time > epoch.end_time:
            alpha_start = self._propagate_plain(alpha_start, epoch.end_time - u_prev, use_action)
            u_prev = epoch.end_time
            i_epoch, epoch = next(epochs)
            self.state_space.update_epoch(epoch)
        alpha_start = self._propagate_plain(alpha_start, start_time - u_prev, use_action)
        u_prev = start_time

        self._logger.debug(
            "accumulate (k=%d): from t=%.3g via the %s Van Loan exponential (dim %d)",
            k, start_time, "sparse action of the" if use_action else "dense", (k + 1) * n
        )

        # w = alpha_start in the first block and e_ext = e in the last block, so that w @ Q @ e_ext is
        # alpha_start @ Q[:n, -n:] @ e for the propagator Q. Block j of w is stored divided by lamb ** j.
        w = np.zeros((k + 1) * n)
        w[:n] = alpha_start
        e_ext = np.zeros((k + 1) * n)
        e_ext[-n:] = self.state_space.e

        # rewards are epoch-invariant (they depend on the states, not the rates), so only the intensity matrix, and
        # hence the Van Loan matrix, is refreshed per epoch
        R = [r._get(state_space=self.state_space) for r in rewards]
        scale = self._reward_scale(R)

        if use_action:
            op, lamb = self._action_operator(rewards, k, i_epoch), 1.0
        else:
            lamb = self._balance(epoch, start_time, t_sorted[-1], scale)
            op = self._dense_van_loan(R, lamb, k, i_epoch)

        def propagators(op: np.ndarray, lamb: float) -> _Propagators:
            """The dense exponentials of the Van Loan matrix ``op`` balanced by ``lamb``, by step."""
            return _Propagators(lambda tau: self._block_expm((op * tau / lamb)[None], n, [False] * (k + 1))[0])

        expms = None if use_action else propagators(op, lamb)

        def advance(w, tau, lamb, count=1) -> Tuple[np.ndarray, float]:
            if use_action:
                return self._advance_action(w, tau, lamb, op, k, count)

            W = np.empty((count, len(w)))
            for c in range(count):
                w = W[c] = w @ expms(tau) if tau > 0 else w

            return W, lamb

        moments = np.zeros_like(t_sorted, dtype=float)

        # an empty (or reversed) window accumulates no reward
        x, i = start_time, int(np.searchsorted(t_sorted, start_time, side='right'))

        # The Van Loan exponential is evaluated over the absorption time, which scales with Ne (the doubling search
        # in ``_get_absorption_time`` deliberately spans many orders of magnitude). For a large time the dense
        # ``expm`` can transiently over/underflow inside scipy's scaling-squaring on some BLAS builds, even though
        # the regularized result (corrected by ``lamb ** k``) is finite. The benign intermediate over/divide/invalid
        # is silenced here and the *output* is checked for finiteness below, so a genuine blow-up still surfaces.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            while i < len(t_sorted):
                # the end times up to the end of the epoch, as runs of equal steps
                j = int(np.searchsorted(t_sorted, epoch.end_time, side='right'))
                runs, x_end = _StepRuns.split(x, t_sorted[i:j])

                for step, count in runs:
                    W, lamb = advance(w, step, lamb, count)
                    moments[i:i + count] = factorial(k) * lamb ** k * (W @ e_ext)
                    w, i = W[-1].copy(), i + count

                if i == len(t_sorted):
                    break

                # advance to the end of the epoch
                w, lamb = advance(w, epoch.end_time - x_end, lamb)
                w, x = w[-1], epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

                if use_action:
                    op = self._action_operator(rewards, k, i_epoch)
                else:
                    # balance each epoch on its own rates (see ``_rebase``)
                    lamb_new = self._balance(epoch, start_time, t_sorted[-1], scale)
                    lamb = self._rebase_forward(w, lamb, lamb_new, k, n)
                    op = self._dense_van_loan(R, lamb, k, i_epoch)
                    expms = propagators(op, lamb)

        # restore the original (unsorted) order
        moments = moments[np.argsort(np.argsort(end_times))]

        # the suppressed intermediate over/underflow must not have corrupted the (finite) result
        if not np.isfinite(moments).all():
            self._logger.warning(
                "Non-finite values encountered when computing moments. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def _get_time_to_absorption(self) -> float:
        """
        The estimated time of almost sure absorption, over which an infinite end time is integrated when the closed
        form does not apply.

        :return: The time of almost sure absorption.
        """
        if self._tree_height.end_time is None:
            return self._tree_height.t_max

        return self._tree_height._get_absorption_time()

    def _get_epochs_until_unbounded(self, k: int = 1) -> List[Epoch]:
        """
        Materialize the demographic epochs up to and including the one taken to hold until absorption. The iteration
        stops at an epoch with an infinite end time, or at the first epoch beginning at or after the time of almost
        sure absorption (``TreeHeightDistribution.t_max`` without an accumulation window) whose extension is
        negligible under its own rates and those of every later epoch, at most
        ``TreeHeightDistribution._max_extension_epochs`` epochs past that time. The rates of that epoch are then
        extended over the remaining time. Negligible refers to the moment of order ``k``, whose share of the reward
        accumulated past that time grows with the order.

        :param k: The order of the moment.
        :return: List of epochs, the last of which is unbounded.
        """
        # the epochs depend only on the tree height, which fixes the demography and the absorption time, and on the
        # order, while the closed form queries them once per moment and an SFS evaluates many bins
        th = self._tree_height
        memo = th.__dict__.get('_epochs_cache')
        if not isinstance(memo, dict):
            memo = th.__dict__['_epochs_cache'] = {}

        if k in memo:
            return memo[k][0]

        # the order matters only where an epoch is held past the time of almost sure absorption
        if k > 1:
            self._get_epochs_until_unbounded(1)
            if not memo[1][1]:
                memo[k] = memo[1]
                return memo[k][0]

        epochs, t_absorption, survival, scale, extra, extended = [], None, None, None, 0, False

        # the epochs before the time of almost sure absorption, that time, and the survival and scale there are those
        # of the first order, whose search stored them, so a higher order resumes the search at that time
        search = th.__dict__.get('_epochs_search')
        if k > 1 and isinstance(search, tuple):
            prefix, t_absorption, survival, scale = search
            epochs = list(prefix)

        # the state distribution propagated to the start of each finite epoch. Where the CDF there is still below the
        # absorption level, the epoch starts before the time of almost sure absorption, whose search is then spared
        w, prev = None, None

        # the transient mass reaching each epoch after the time of almost sure absorption, by epoch index, from a
        # propagation shared across orders that advances with the search, which visits the epochs in turn
        reach = th.__dict__.setdefault('_extension_reach', {})

        def mass(e: Epoch) -> float:
            if e.index not in reach:
                w_e, t_e, epoch_e = reach.get('cursor', (None, np.inf, None))
                if t_e > e.start_time:
                    w_e, t_e, epoch_e = np.asarray(th.state_space.alpha, dtype=float), 0.0, self.demography.get_epoch(0)

                w_e = th._sweep_to(w_e, t_e, e.start_time, epoch_e)
                reach['cursor'] = (w_e, e.start_time, e)
                reach[e.index] = float(w_e @ th._e / w_e.sum())

            return reach[e.index]

        def holds(first: Epoch) -> bool:
            if not th._extension_is_negligible(first, survival, scale, k, t_absorption):
                return False

            # the search ends where the transient mass underflows, or at a population size of zero, which absorbs it
            # at once
            try:
                for e in itertools.islice(
                        self.demography.epochs, first.index + 1, first.index + th._max_extension_epochs - extra):
                    if not th._extension_is_negligible(e, mass(e), scale, k, t_absorption):
                        return False
                    if mass(e) == 0:
                        break
            except ModelError:
                pass

            return True

        for epoch in itertools.islice(self.demography.epochs, len(epochs), None):

            if epoch.end_time == np.inf:
                epochs.append(epoch)
                break

            if t_absorption is None:
                # across the previous epoch, which ends where this one starts
                if prev is None:
                    w = np.asarray(th.state_space.alpha, dtype=float)
                else:
                    w = th._sweep_to(w, prev.start_time, epoch.start_time, prev)
                prev = epoch

                if th._cum(w) < th.p_absorption:
                    epochs.append(epoch)
                    continue

                # the search is evaluated only once an epoch may start after absorption, and the survival and scale
                # only once one does
                t_absorption = self._get_time_to_absorption()
                prefix = list(epochs)

            if epoch.start_time >= t_absorption and survival is None:
                survival = self._tree_height._survival(t_absorption)
                scale = self._tree_height._get_absorption_scale()

            # an epoch reached after absorption is almost sure stands in for every epoch after it, so it may only be
            # held where the reward that substitution misplaces is negligible under its own rates and under those of
            # every later epoch the transient mass reaches. The count bounds the search, a demography having
            # infinitely many epochs by design.
            if epoch.start_time >= t_absorption and extra < th._max_extension_epochs and not holds(epoch):
                epochs.append(epoch)
                extra += 1
                continue

            if epoch.start_time >= t_absorption:
                last = Epoch(
                    start_time=epoch.start_time,
                    end_time=np.inf,
                    pop_sizes=epoch.pop_sizes,
                    migration_rates=epoch.migration_rates
                )
                last.index = epoch.index
                epochs.append(last)
                extended = True
                break

            epochs.append(epoch)

        memo[k] = (epochs, extended)

        if k == 1 and t_absorption is not None:
            th.__dict__['_epochs_search'] = (prefix, t_absorption, survival, scale)

        return epochs

    def _absorption_certain_in_last_epoch(self, k: int = 1) -> bool:
        """
        Whether every transient state that can carry mass (see ``_alpha_support``) can reach an absorbing state in the
        final epoch of ``_get_epochs_until_unbounded`` (see ``_absorbs_in``), so that ``-T`` restricted to those
        states is non-singular and the closed form applies. When ``False``, callers use the matrix exponential up to
        the absorption-time estimate, whose search raises for a demography that does not absorb.

        :param k: The order of the moment, which selects the epochs (see ``_get_epochs_until_unbounded``).
        :return: Whether absorption is certain from every transient state of the last epoch that can carry mass.
        """
        # the result depends only on the (fixed) last-epoch structure, so memoize it: the closed form queries this
        # once per moment, and an SFS/jSFS evaluates many bins, so recomputing the reachability each time dominated.
        # One host serves several state spaces (the lineage-counting one for a tree height, the block-counting one
        # for a spectrum), so the memo is per state space and epoch list. The lists of all orders share their finite
        # epochs and differ in where they end, so the length identifies the list.
        cache = self._state_space_memo('_absorption_certain_cache')
        ss = self.state_space
        key = (id(ss), len(self._get_epochs_until_unbounded(k)))

        if key in cache and cache[key][0] is ss:
            return cache[key][1]

        # the memoized reachability and support, which leave the state space in the last epoch
        reach = self._reaches_absorption_in_last_epoch(k)
        certain = bool(reach.all()) or bool(reach[self._alpha_support(k) & ~ss.absorbing].all())
        cache[key] = (ss, certain)
        return certain

    def _absorbs_in(self, epochs: Sequence[Epoch]) -> bool:
        """
        Whether every transient state that can carry mass over ``epochs`` (see ``_forward_closure``) reaches an
        absorbing state in the last of them. A state the initial vector never reaches, such as a deme declared with no
        samples and no migration into it, has no bearing on the answer. Leaves the state space in the last epoch.

        :param epochs: The epochs in order, the last of which is held until absorption.
        :return: Whether absorption is certain.
        """
        self.state_space.update_epoch(epochs[-1])
        _, reach = self._reaches_absorption()

        if reach.all():
            return True

        support = self._forward_closure(epochs)
        self.state_space.update_epoch(epochs[-1])

        return bool(reach[support & ~self.state_space.absorbing].all())

    def _forward_closure(self, epochs: Sequence[Epoch]) -> np.ndarray:
        """
        The states that can carry probability mass over ``epochs``: the forward closure of the initial support under
        each epoch's transitions in turn. Leaves the state space in the last epoch.

        :param epochs: The epochs in order.
        :return: Boolean mask over the states.
        """
        ss = self.state_space
        support = np.asarray(ss.alpha) > 0

        for epoch in epochs:
            ss.update_epoch(epoch)
            support = self._close_forward(support, ss.S)

        return support

    @staticmethod
    def _close_forward(support: np.ndarray, S) -> np.ndarray:
        """
        The states reachable from ``support`` through the transitions of the generator ``S``, including ``support``.

        :param support: Boolean mask over the states of ``S``.
        :param S: The generator or a diagonal block of it, dense or sparse.
        :return: Boolean mask over the states of ``S``.
        """
        adj = ((S if sp.issparse(S) else sp.csr_matrix(np.asarray(S))) != 0).T.tocsr()

        while True:
            nxt = support | (adj @ support > 0)
            if np.array_equal(nxt, support):
                return support
            support = nxt

    def _state_space_memo(self, name: str) -> dict:
        """
        A memo of the host keyed by state space, or by state space and epoch list, holding ``(state_space, value)`` so
        that an entry is used only for the very state space it was computed on, which a key reused by another object
        after deserialization is not.

        :param name: Attribute name of the memo.
        :return: The memo.
        """
        return self.__dict__.setdefault(name, {})

    def _alpha_support(self, k: int = 1) -> np.ndarray:
        """
        The states that can carry probability mass over the epochs of ``_get_epochs_until_unbounded``, see
        ``_forward_closure``. Memoized per state space and epoch list, one host serving several state spaces.

        :param k: The order of the moment, which selects the epochs (see ``_get_epochs_until_unbounded``).
        :return: Boolean mask over the states of the current state space.
        """
        ss = self.state_space
        cache = self._state_space_memo('_alpha_support_cache')
        key = (id(ss), len(self._get_epochs_until_unbounded(k)))

        if key in cache and cache[key][0] is ss:
            return cache[key][1]

        # where every state reaches absorption the restriction changes no solve, so the closure is skipped
        if self._absorbs_from_every_state(k):
            support = np.ones(ss.k, dtype=bool)
            cache[key] = (ss, support)
            return support

        support = self._forward_closure(self._get_epochs_until_unbounded(k))
        cache[key] = (ss, support)
        return support

    def _reaches_absorption_in_last_epoch(self, k: int = 1) -> np.ndarray:
        """
        The states that reach absorption in the final epoch, memoized per state space and epoch list. Leaves the
        state space in the final epoch.

        :param k: The order of the moment, which selects the epochs (see ``_get_epochs_until_unbounded``).
        :return: Boolean mask over the states, including the absorbing ones.
        """
        ss = self.state_space
        cache = self._state_space_memo('_last_epoch_reach_cache')
        ss.update_epoch(self._get_epochs_until_unbounded(k)[-1])
        key = (id(ss), len(self._get_epochs_until_unbounded(k)))

        if key in cache and cache[key][0] is ss:
            return cache[key][1]

        _, reach = self._reaches_absorption()
        cache[key] = (ss, reach)
        return reach

    def _absorbs_from_every_state(self, k: int = 1) -> bool:
        """
        Whether every state reaches absorption in the final epoch, so that absorption is certain whatever the initial
        vector.

        :param k: The order of the moment, which selects the epochs (see ``_get_epochs_until_unbounded``).
        :return: Whether every state reaches absorption.
        """
        return bool(self._reaches_absorption_in_last_epoch(k).all())

    def _reaches_absorption(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward reachability over the rate graph of the current epoch: a state reaches absorption if it is absorbing
        or has a positive rate to a state that does, propagated with a sparse adjacency matrix.

        :return: ``(absorbing, reach)`` boolean masks over the states; ``reach`` includes the absorbing states.
        """
        absorbing = self.state_space.absorbing

        S = self.state_space.S
        if sp.issparse(S):
            adj = S.tocsr(copy=True)
            adj.setdiag(0)
            adj.eliminate_zeros()
        else:
            adj = sp.csr_matrix((S - np.diag(np.diag(S))) > 0)

        reach = absorbing.copy()
        while True:
            nxt = absorbing | (adj @ reach > 0)
            if np.array_equal(nxt, reach):
                break
            reach = nxt

        return absorbing, reach

    def _assert_absorbs(self, epochs: Sequence[Epoch] = None) -> None:
        """
        Raise unless absorption is certain from every transient state that can carry mass, see ``_absorbs_in``.

        :param epochs: The epochs in order, the last of which is held until absorption. By default, those of
            ``_get_epochs_until_unbounded``.
        :raises ModelError: if some state carrying mass can never reach a common ancestor.
        """
        certain = self._absorption_certain_in_last_epoch() if epochs is None else self._absorbs_in(epochs)

        if not certain:
            raise ModelError(
                "The demography does not absorb: some states carrying probability mass can never reach a common "
                "ancestor in the final (unbounded) epoch, so there is no almost-sure absorption time. This typically "
                "means a deme is isolated or migration is one-way/blocked in the last epoch, leaving lineages that "
                "can never coalesce. Check the migration structure of the last epoch."
            )

    def _accumulate_closed_form(
            self,
            k: int,
            rewards: Sequence[Reward],
            start_time: float = 0.0,
            tape: _ClosedFormTape = None,
            suffixes: bool = False
    ) -> np.ndarray:
        """
        Raw moment of a single reward ordering from ``start_time`` to absorption with the closed-form last epoch of
        ``PhaseTypeDistribution.moment``: the backward recursion with one LU of ``-T`` (``_lu_solver``), then the
        finite epochs from ``start_time`` on applied backwards to the extended vector, and last the entry distribution
        propagated to ``start_time``.

        :param k: The order of the moment, at least 1.
        :param rewards: Sequence of k rewards (a single ordering).
        :param start_time: The non-negative start time.
        :param tape: If given, receives the record that ``CrossMomentGradient`` differentiates, for a zero start time.
            The sparse actions are then applied by ``_TapedAction``.
        :param suffixes: Whether to return the raw moments of the trailing rewards ``rewards[k - i:]`` for ``i = 0,
            ..., k``, read from the trailing blocks of the extended vector, one for ``i = 0``.
        :return: The kth moment accumulated until absorption, as an array of one entry, or the moments of the trailing
            rewards.
        :raises ValueError: If ``tape`` is given with a positive start time.
        """
        if tape is not None and start_time > 0:
            raise ValueError("A closed form is recorded from time zero only.")

        epochs = self._get_epochs_until_unbounded(k)
        self._check_demography_conditioning(epochs)
        n = self.state_space.k

        # --- final, unbounded epoch: limit vector z ---
        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, len(epochs) - 1)
        absorbing = self.state_space.absorbing

        # restrict to the states that can carry mass: a transient state outside the support contributes nothing and
        # makes ``-T`` singular when it cannot reach absorption. Mass leaves the support nowhere, the support being
        # a forward closure, so the restricted solve is exact on it.
        support = self._alpha_support(k)
        idx_t = np.where(~absorbing & support)[0]
        idx_a = np.where(absorbing)[0]
        e = np.asarray(self.state_space.e)

        # The closed form factors the transient sub-generator ``T`` (size = number of transient states), whose
        # dense-LU vs sparse-LU crossover sits at :attr:`closed_form_sparse_min_states` transient states. The
        # finite-epoch Van Loan exponentials take the sparse action except on the stiff generators of
        # ``_dense_expm_is_faster``, whatever the factorization.
        sparse = self._solve_sparse(len(idx_t))
        dim = (k + 1) * n
        use_action = not self._dense_expm_is_faster(
            dim, dim, 1, self._closed_form_steps(epochs, start_time), taped=tape is not None
        )

        if sparse:
            self._logger.debug(
                "closed form (k=%d): sparse LU (splu) of T (n_t=%d >= %d), %s exponential, %d finite epoch(s)",
                k, len(idx_t), Settings.closed_form_sparse_min_states, "sparse" if use_action else "dense",
                len(epochs) - 1
            )
        else:
            self._logger.debug(
                "closed form (k=%d): dense LU of T (n_t=%d), %d finite epoch(s)", k, len(idx_t), len(epochs) - 1
            )
        solve = self._last_epoch_solver(id(self.state_space), len(epochs), idx_t.astype(np.intp).tobytes(), sparse)

        # the j-th block of the extended vector is stored divided by lamb ** (k - j), which balances the Van Loan
        # exponentials of the finite epochs, so the moment is multiplied by lamb ** k at the end
        r_all = [np.asarray(r._get(self.state_space), dtype=float) for r in rewards]
        scale = self._reward_scale(r_all)
        lamb = self._get_regularization_factor(self.state_space.S, scale=scale)

        # reward diagonals restricted to the transient states (the off-diagonal Van Loan reward blocks are diagonal)
        r_t = [r[idx_t] for r in r_all]

        nu = [None] * (k + 1)
        nu[k] = e[idx_t]
        for j in range(k - 1, 0, -1):
            nu[j] = solve(r_t[j] * nu[j + 1]) / lamb

        z = np.zeros((k + 1) * n)
        z[idx_t] = solve(r_t[0] * nu[1]) / lamb
        for j in range(1, k + 1):
            z[j * n + idx_t] = nu[j]
        z[k * n + idx_a] = e[idx_a]

        if tape is not None:
            tape.epochs, tape.idx_t, tape.solve, tape.lamb = epochs, idx_t, solve, lamb
            tape.rewards = r_all
            tape.blocks = [z[j * n + idx_t].copy() for j in range(k)]

        # --- preceding finite epochs, backward, via the (sparse or dense) full Van Loan matrix exponential ---
        # Each epoch is balanced on its own rates. ``V * tau`` carries the epoch's generator on the diagonal and its
        # reward blocks divided by the factor, so a factor drawn from one epoch leaves the reward blocks of an epoch
        # with a different rate scale far from one, and the scaling-and-squaring of the exponential loses their
        # cancellation. Rebasing the stored vector is the exact diagonal similarity that permits it. The factors,
        # steps and Van Loan matrices do not depend on the extended vector, so they are gathered first and the dense
        # exponentials evaluated in stacks.
        steps = []
        R = [r._get(self.state_space) for r in rewards]
        for i_epoch, epoch in reversed(list(enumerate(epochs[:-1]))):
            if epoch.end_time <= start_time:
                break

            self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, i_epoch)
            lamb_e = self._balance(epoch, start_time, np.inf, scale)
            duration = epoch.end_time - max(epoch.start_time, start_time)
            tau = duration / lamb_e

            if use_action:
                r_vecs = [np.asarray(r, dtype=float) for r in R]
                V = self._van_loan_matrix(r_vecs, self._epoch_csr(i_epoch) * lamb_e, k, sparse=True)
                steps.append((i_epoch, duration, lamb_e, V * tau))
            else:
                steps.append((i_epoch, duration, lamb_e, (self._dense_rate_matrix() * lamb_e, tau)))

        if not use_action:
            exps = self._block_expms([
                lambda S=S, tau=tau: self._van_loan_matrix(R, S, k) * tau for _, _, _, (S, tau) in steps
            ], n, [False] * (k + 1))

        for i_epoch, duration, lamb_e, op in steps:
            if tape is None:
                lamb = self._rebase(z, lamb, lamb_e, k, n)
                z = Backend.expm_multiply(op, z, blocks=k + 1) if use_action else next(exps) @ z
                continue

            rebase = np.ones_like(z)
            self._rebase(rebase, lamb, lamb_e, k, n)
            lamb = self._rebase(z, lamb, lamb_e, k, n)

            if use_action:
                step = _VanLoanStep(i_epoch, epochs[i_epoch], duration, rebase, z.copy(), op.tocsr())
                step.action = _TapedAction(step.A, blocks=k + 1)
                z, step.checkpoints = step.action.apply(z)
            else:
                S, tau = op
                step = _VanLoanStep(i_epoch, epochs[i_epoch], duration, rebase, z.copy(),
                                    self._van_loan_matrix(R, S, k) * tau)
                step.exp = next(exps)
                z = step.exp @ z

            step.z_out = z.copy()
            tape.steps.append(step)

        # the entry distribution propagated to the start time with the plain generator
        alpha_ext = np.zeros((k + 1) * n)
        alpha_ext[:n] = self.state_space.alpha
        for epoch in epochs:
            if epoch.start_time >= start_time:
                break

            self.state_space.update_epoch(epoch)
            alpha_ext[:n] = self._propagate_plain(alpha_ext[:n], min(epoch.end_time, start_time) - epoch.start_time,
                                                  use_action)

        if tape is not None:
            tape.alpha_ext, tape.z, tape.lamb_end, tape.factor = alpha_ext, z, lamb, factorial(k) * lamb ** k

        if suffixes:
            return np.array([1.0] + [
                factorial(i) * lamb ** i * float(alpha_ext[:n] @ z[(k - i) * n:(k - i + 1) * n]) for i in range(1, k + 1)
            ])

        return np.array([factorial(k) * lamb ** k * float(alpha_ext @ z)])

    def _dense_expm_is_faster(
            self,
            d_action: int,
            d_dense: int,
            count: int,
            steps: List[Tuple[Epoch, float]],
            n_dense: int = None,
            taped: bool = False
    ) -> bool:
        """
        Whether ``count`` dense exponentials of size ``d_dense`` per step are cheaper than one sparse action of an
        operator of dimension ``d_action`` per step, over the steps ``(epoch, tau)``, or ``n_dense`` times ``count``
        dense exponentials when they are reused across steps, by ``_MOMENT_COST`` with the step norms of the rate
        matrix ``S`` of each epoch. Leaves the state space in the epoch it found.

        :param d_action: The dimension of the operator of the action.
        :param d_dense: The size of each dense exponential.
        :param count: The number of dense exponentials per step.
        :param steps: The epochs and the times of the actions.
        :param n_dense: The number of steps with their own dense exponentials, by default all.
        :param taped: Whether the pass is recorded for its gradient.
        :return: Whether to form the dense exponentials.
        """
        steps = [(epoch, tau) for epoch, tau in steps if tau > 0]
        if not steps:
            return False

        bound = _MOMENT_COST.bound(d_action, d_dense, count, len(steps), len(steps) if n_dense is None else n_dense,
                                   taped=taped)
        if bound == np.inf:
            return False

        # the fixed costs of the actions alone may exceed the dense exponentials, without reading a generator
        if bound < 0:
            return True

        # the summed step norm, reading each epoch's generator once
        durations: Dict[Epoch, float] = {}
        for epoch, tau in steps:
            durations[epoch] = durations.get(epoch, 0.0) + tau

        current = self.state_space.epoch
        norm = 0.0
        for epoch, duration in durations.items():
            self.state_space.update_epoch(epoch)
            norm += float(abs(self.state_space.S).sum(axis=0).max()) * duration
        self.state_space.update_epoch(current)

        return bool(norm > bound)

    @staticmethod
    def _closed_form_steps(epochs: List[Epoch], start_time: float) -> List[Tuple[Epoch, float]]:
        """
        The epochs of ``_get_epochs_until_unbounded`` and the times the closed form spends in them under an
        exponential, up to the later of ``start_time`` and the start of the last epoch.

        :param epochs: The epochs.
        :param start_time: The non-negative start time.
        :return: The epochs and the times.
        """
        horizon = max(start_time, epochs[-1].start_time)

        return [(epoch, min(epoch.end_time, horizon) - epoch.start_time) for epoch in epochs]

    def _grid_steps(self, start_time: float, end_times: np.ndarray) -> Tuple[List[Tuple[Epoch, float]], int]:
        """
        The actions and the number of distinct dense exponentials of a propagation from ``start_time`` through the
        sorted finite ``end_times`` after it, as ``_accumulate_windowed`` and ``_mean_occupation_grid`` take them: one
        action per step of a run of equal steps (``_StepRuns``) and per remainder to the end of an epoch, and one dense
        exponential per epoch and step length.

        :param start_time: The non-negative start time.
        :param end_times: The sorted finite end times.
        :return: The epochs and the times of the actions, and the number of dense exponentials.
        """
        times = end_times[end_times > start_time]
        steps, dense, x, i = [], set(), start_time, 0
        for epoch in self.demography.epochs:
            if i == len(times):
                break
            if epoch.end_time <= x:
                continue

            j = int(np.searchsorted(times, epoch.end_time, side='right'))
            runs, x = _StepRuns.split(x, times[i:j])
            if j < len(times):
                runs.append((epoch.end_time - x, 1))
                x = epoch.end_time
            steps += [(epoch, step) for step, count in runs if step > 0 for _ in range(count)]
            dense.update((epoch.start_time, step) for step, _ in runs if step > 0)
            i = j

        return steps, len(dense)

    def _flattening_applies(self, k: int, rewards: Sequence[Reward] = None) -> bool:
        """
        Whether the block-counting state space can be flattened onto the lineage-counting state space for this
        moment of the standard coalescent on a single population and a single locus: the first moment of any reward,
        and the second moment of two rewards that are each an unfolded or folded SFS bin with polymorphic blocks.
        Takes precedence over the closed form and the batched occupation.

        :param k: The order of the moment.
        :param rewards: The rewards, required for the second moment.
        :return: Whether to flatten.
        """
        if not (
                Settings.flatten_block_counting and
                isinstance(self.state_space, BlockCountingStateSpace) and
                isinstance(self.state_space.model, StandardCoalescent) and
                self.lineage_config.n_pops == 1 and
                self.locus_config.n == 1
        ):
            return False

        if k == 1:
            return True

        return k == 2 and rewards is not None and len(rewards) == 2 and all(
            self._sfs_reward_block_sizes(r, self.lineage_config.n) is not None for r in rewards
        )

    def _transient_block(self, idx_t: np.ndarray, sparse: bool = False) -> 'sp.spmatrix | np.ndarray':
        """
        The transient block ``S[idx_t, idx_t]`` of the rate matrix, as a dense array or, with ``sparse=True``, a
        sparse CSC matrix.
        """
        S = self.state_space.S
        if sp.issparse(S):
            sub = S[idx_t][:, idx_t]
            return sub.tocsc() if sparse else np.asarray(sub.todense())
        sub = np.asarray(S)[np.ix_(idx_t, idx_t)]
        return sp.csc_matrix(sub) if sparse else sub

    def _dense_rate_matrix(self) -> np.ndarray:
        """
        The full rate matrix as a dense array (densifying if it is stored sparse). Used by the dense moment paths,
        which are only taken for state spaces small enough that a dense matrix is cheap.
        """
        S = self.state_space.S
        return np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)

    def _epoch_csr(self, i_epoch: int) -> sp.csr_matrix:
        """
        The rate matrix of epoch ``i_epoch`` of ``_get_epochs_until_unbounded`` as a CSR matrix, the state space being
        set to that epoch. A rate matrix stored as CSR is returned as is, since ``StateSpace.update_epoch`` may rescale
        it in place. A conversion is memoized per state space, since every bin, reward ordering and order of a
        closed-form moment reads it.

        :param i_epoch: The epoch number.
        :return: The rate matrix.
        """
        ss = self.state_space
        if sp.issparse(ss.S) and ss.S.format == 'csr':
            return ss.S

        memo = self._state_space_memo('_epoch_csr_cache')

        if id(ss) not in memo or memo[id(ss)][0] is not ss:
            memo[id(ss)] = (ss, {})

        csr = memo[id(ss)][1]
        if i_epoch not in csr:
            S = ss.S
            csr[i_epoch] = S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S))

        return csr[i_epoch]

    def _mean_occupation_grid(self, end_times: Sequence[float], start_time: float = None) -> np.ndarray:
        """
        Expected time spent in each state of ``E`` over ``[start_time, t]`` for each end time ``t``, threaded across
        epochs with the augmented generator ``[[S, I], [0, 0]]`` of ``PhaseTypeDistribution.moment``, for the batched
        mean accumulation of a spectrum. Within an epoch, the end times are taken as runs of equal steps
        (``_StepRuns``), with one dense exponential per step length or one interval action per run. A positive start
        time subtracts the occupation up to it.

        :param end_times: Times at which to evaluate the occupation. A NaN end time gives a NaN occupation.
        :param start_time: Time from which to accumulate. By default, the start time of the distribution.
        :return: Array of shape ``(len(end_times), n_states)``.
        :raises ValueError: If an end time or the start time is negative.
        """
        end_times = np.asarray(end_times, dtype=float)

        undefined = np.isnan(end_times)
        if undefined.any():
            out = np.full((len(end_times), self.state_space.k), np.nan)
            if not undefined.all():
                out[~undefined] = self._mean_occupation_grid(end_times[~undefined], start_time=start_time)
            return out

        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        if start_time is None:
            start_time = self._tree_height.start_time

        _validate_start_time(start_time)

        if start_time > 0:
            return (
                    self._mean_occupation_grid(np.maximum(end_times, start_time), start_time=0.0) -
                    self._mean_occupation_grid([start_time], start_time=0.0)
            )

        # infinite end times take the occupation until absorption, in closed form when absorption is certain in the
        # last epoch, where the absorbing states carry no occupation
        infinite = np.isinf(end_times)
        if infinite.any():
            occupation = self._occupation_times() if Settings.closed_form_last_epoch else None
            if occupation is not None:
                out = np.zeros((len(end_times), self.state_space.k))
                out[np.ix_(infinite, occupation[1])] = occupation[0]
                if not infinite.all():
                    out[~infinite] = self._mean_occupation_grid(end_times[~infinite], start_time=0.0)
                return out

            end_times = np.where(infinite, self._get_time_to_absorption(), end_times)

        order = np.argsort(end_times)
        t_sorted = end_times[order]

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)
        self.state_space.update_epoch(epoch)
        n = self.state_space.k

        # the occupation integral is read off the augmented generator ``[[S, I], [0, 0]]``, by its (transposed)
        # matrix-exponential action or its dense exponentials
        use_action = not self._dense_expm_is_faster(2 * n, 2 * n, 1, *self._grid_steps(0.0, t_sorted))

        p = np.concatenate([np.asarray(self.state_space.alpha, dtype=float), np.zeros(n)])
        x, i = 0.0, 0
        out = np.zeros((len(t_sorted), n))

        while True:
            self._check_numerical_stability(self.state_space.S, i_epoch)
            S = self.state_space.S

            if use_action:
                augT = sp.bmat([
                    [sp.csc_matrix(S), sp.identity(n, format='csc')],
                    [None, sp.csc_matrix((n, n))],
                ], format='csc').T.tocsc()

                def advance(v: np.ndarray, tau: float) -> np.ndarray:
                    """``v`` advanced by ``tau``."""
                    return Backend.expm_multiply(augT * tau, v, action=spla.expm_multiply, blocks=2)
            else:
                aug = np.zeros((2 * n, 2 * n))
                aug[:n, :n] = np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)
                aug[:n, n:] = np.eye(n)
                propagators = _Propagators(lambda tau: self._block_expm((aug * tau)[None], n, [False, True])[0])

                def advance(v: np.ndarray, tau: float) -> np.ndarray:
                    """``v`` advanced by ``tau``."""
                    return v @ propagators(tau)

            # the end times up to the end of the epoch, as runs of equal steps
            j = int(np.searchsorted(t_sorted, epoch.end_time, side='right'))
            runs, x_end = _StepRuns.split(x, t_sorted[i:j])

            for step, count in runs:
                if step <= 0:
                    out[i:i + count] = p[n:]
                elif use_action and count > 1:
                    P = Backend.expm_multiply_interval(
                        augT, p, step * count, count, interval=_SCIPY_EXPM.compute_action_interval, advance=advance
                    )
                    out[i:i + count], p = P[:, n:], P[-1]
                else:
                    for c in range(i, i + count):
                        p = advance(p, step)
                        out[c] = p[n:]
                i += count

            if i == len(t_sorted):
                break

            if epoch.end_time > x_end:
                p = advance(p, epoch.end_time - x_end)
            x = epoch.end_time
            i_epoch, epoch = next(epochs)
            self.state_space.update_epoch(epoch)

        return out[np.argsort(order)]

    def _occupation_times(
            self,
            cap: float = None,
            tape: List['_OccupationStep'] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Expected occupation times of the transient states until absorption (the vector ``m`` of
        ``PhaseTypeDistribution.moment``). Finite epochs use the augmented generator ``[[T_i, I], [0, 0]]`` on the
        transient block, and the final epoch contributes ``p (-T)^{-1}``. With ``cap`` the accumulation stops at that
        time, and the epoch containing it is treated as finite, so a windowed mean subtracts the capped occupation.

        :param cap: If given, accumulate occupation only up to this absolute time instead of to absorption.
        :param tape: If given, receives one ``_OccupationStep`` per epoch, the record ``MeanGradient`` differentiates.
            The sparse actions are then applied by ``_TapedAction``.
        :return: ``(occupation, idx_t)`` with the occupation times over the transient states ``idx_t`` of the final
            epoch, or ``None`` if absorption is not almost sure (callers then fall back to per-bin evaluation).
        :raises ModelError: if the occupation times propagated to an epoch are not finite.
        """
        if not self._absorption_certain_in_last_epoch():
            return None

        epochs = self._get_epochs_until_unbounded()

        self.state_space.update_epoch(epochs[-1])
        absorbing = self.state_space.absorbing
        # only the states that can carry mass; see ``_alpha_support``
        idx_t = np.where(~absorbing & self._alpha_support())[0]
        nt = len(idx_t)
        sparse = self._solve_sparse(nt)

        p = np.asarray(self.state_space.alpha)[idx_t].astype(float)
        m = np.zeros(nt)

        self._logger.debug(
            "occupation times (batched mean%s): %s factorization, n_t=%d, %d finite epoch(s)",
            f", capped at {cap:.3g}" if cap is not None else "",
            "sparse" if sparse else "dense", nt, len(epochs) - 1
        )

        # Each epoch contributes its occupation over ``[epoch.start_time, upper]``. Uncapped, the final (unbounded)
        # epoch's ``upper`` is infinite and its contribution is the closed form ``p (-T)^{-1}``; every other epoch is a
        # finite Van Loan block. The within-epoch occupation integral ``A = int_0^tau exp(S t) dt`` is read off the
        # augmented generator ``[[S, I], [0, 0]]``, robust even when ``S`` is singular (e.g. a migration barrier),
        # unlike ``(exp(S tau) - I) S^-1``. Only the row-action ``[p, 0] exp(aug tau) = [p exp(S tau), p A]`` is needed
        # (the propagated entry distribution and the occupation increment ``p A`` at once), so for large state spaces
        # apply the sparse matrix-exponential action instead of forming the dense ``2 nt x 2 nt`` exponential. The
        # generators and steps do not depend on ``p``, so they are gathered first and the dense exponentials evaluated
        # in stacks.
        steps = []
        for i_epoch, epoch in enumerate(epochs):
            self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, i_epoch)

            upper = epoch.end_time if cap is None else min(epoch.end_time, cap)
            S = self._transient_block(idx_t, sparse=sparse)

            if np.isinf(upper):
                steps.append((epoch, np.inf, S, False))
                break

            tau = upper - epoch.start_time

            use_action = not self._dense_expm_is_faster(
                2 * nt, 2 * nt, 1, [(epoch, tau)], taped=tape is not None
            )
            steps.append((epoch, tau, S, use_action))

            if cap is not None and upper >= cap:
                break

        def augmented(S, tau) -> np.ndarray:
            aug = np.zeros((2 * nt, 2 * nt))
            aug[:nt, :nt] = S.toarray() if sp.issparse(S) else S
            aug[:nt, nt:] = np.eye(nt)
            return aug * tau

        exps = self._block_expms([
            lambda S=S, tau=tau: augmented(S, tau) for _, tau, S, use_action in steps
            if np.isfinite(tau) and not use_action
        ], nt, [False, True])

        for i_epoch, (epoch, tau, S, use_action) in enumerate(steps):
            if not (np.isfinite(p).all() and np.isfinite(m).all()):
                raise ModelError(
                    f"Non-finite occupation times at the start of epoch {i_epoch}. This is likely due to an "
                    "ill-conditioned rate matrix. Use less extreme population sizes or growth rates."
                )

            if np.isinf(tau):
                # final unbounded epoch, no cap: occupation to absorption = p (-T)^{-1}, i.e. solve (-T)^T x = p
                solve = self._lu_solver(-S.T, sparse)
                if tape is not None:
                    tape.append(_OccupationStep(epoch, idx_t, np.inf, S, np.concatenate([p, m])))
                    tape[-1].solve = solve
                m += solve(p)
                break

            if use_action:
                aug = sp.bmat([
                    [sp.csc_matrix(S), sp.identity(nt, format='csc')],
                    [None, sp.csc_matrix((nt, nt))]
                ], format='csc')
                # [p, 0] exp(aug tau) = (exp((aug tau)^T) [p; 0])^T, so apply the action to the transposed generator
                if tape is None:
                    w = Backend.expm_multiply(
                        (aug * tau).T.tocsc(), np.concatenate([p, np.zeros(nt)]), action=spla.expm_multiply, blocks=2
                    )
                else:
                    action = _TapedAction((aug * tau).T.tocsr(), blocks=2)
                    step = _OccupationStep(epoch, idx_t, tau, S, np.concatenate([p, m]), action=action)
                    w, step.checkpoints = action.apply(np.concatenate([p, np.zeros(nt)]))
                    tape.append(step)
                m += w[nt:]
                p = w[:nt]
            else:
                exp_aug = next(exps)
                if tape is not None:
                    tape.append(_OccupationStep(epoch, idx_t, tau, S, np.concatenate([p, m]), exp_aug=exp_aug))
                m += p @ exp_aug[:nt, nt:]
                p = p @ exp_aug[:nt, :nt]

        return m, idx_t

    @staticmethod
    def _forward_operator(S: sp.csr_matrix, R: np.ndarray, rewarded: np.ndarray, cu: float, cw: float) -> sp.csr_matrix:
        """
        The transpose of the generator of the forward pass of ``_bin_second_moments`` for the column action, of
        ``2 m + 1`` blocks of the size of ``S``: ``S`` on the diagonal blocks of ``p`` and each ``u_a``, ``diag(r_a) /
        cu`` from ``p`` to ``u_a`` and ``diag(c) / cw`` from ``u_a`` to ``w_a``, with ``c`` the indicator of the states
        where some reward is non-zero, so that the vector is stationary once the rewards are collected.

        :param S: The rate matrix.
        :param R: The reward vectors ``r_a`` as columns.
        :param rewarded: The indicator ``c``.
        :param cu: The factor ``u_a`` is stored against.
        :param cw: The factor of ``w_a`` relative to ``u_a``.
        :return: The transposed generator.
        """
        n, m = R.shape
        St = S.T.tocsr()
        states = np.arange(n)
        rewarded = np.flatnonzero(rewarded)
        bins, bins_rewarded = np.repeat(np.arange(m), n), np.repeat(np.arange(m), rewarded.size)

        coupling = sp.csr_matrix((
            np.concatenate([R.T.ravel() / cu, np.full(m * rewarded.size, 1.0 / cw)]),
            (np.concatenate([(1 + bins) * n + np.tile(states, m), (1 + m + bins_rewarded) * n + np.tile(rewarded, m)]),
             np.concatenate([np.tile(states, m), (1 + bins_rewarded) * n + np.tile(rewarded, m)]))
        ), shape=((2 * m + 1) * n,) * 2)

        return (sp.block_diag([St] * (m + 1) + [sp.csr_matrix((m * n, m * n))], format='csr') + coupling).tocsr()

    @staticmethod
    def _forward_van_loan(S: np.ndarray, r: np.ndarray, rewarded: np.ndarray, cu: float, cw: float) -> np.ndarray:
        """
        The dense generator of the forward pass of ``_bin_second_moments`` restricted to one bin, the blocks of ``p``,
        ``u_a`` and ``w_a`` of ``_forward_operator`` untransposed.

        :param S: The rate matrix.
        :param r: The reward vector ``r_a``.
        :param rewarded: The indicator ``c`` of ``_forward_operator``.
        :param cu: The factor ``u_a`` is stored against.
        :param cw: The factor of ``w_a`` relative to ``u_a``.
        :return: The matrix of three blocks in each dimension.
        """
        n = S.shape[0]
        V = np.zeros((3 * n, 3 * n))
        V[:n, :n] = S
        V[n:2 * n, n:2 * n] = S
        V[:n, n:2 * n] = np.diag(r / cu)
        V[n:2 * n, 2 * n:] = np.diag(rewarded / cw)

        return V

    def _bin_second_moments_flattened(
            self,
            rewards: Sequence[Reward],
            start_time: float = None,
            end_time: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The means and the cross-moments of ``_bin_second_moments`` for SFS bin rewards flattened onto the
        lineage-counting state space, from the occupation and the cross-moments of ``_level_cross_moments``, the
        latter summed over the block sizes of each bin.

        :param rewards: The bin rewards, each flattened as in ``_flattening_applies``.
        :param start_time: The start time. By default, that of the distribution.
        :param end_time: The end time. By default, that of the distribution, or absorption.
        :return: The means and the cross-moments.
        """
        n = self.lineage_config.n
        start, end = self._resolve_window(start_time, end_time)
        self._logger.debug("second moments of %d SFS bins flattened onto the lineage-counting state space", len(rewards))

        # the block sizes counted by each bin
        F = np.zeros((len(rewards), n - 1))
        for a, r in enumerate(rewards):
            F[a, np.asarray(self._sfs_reward_block_sizes(r, n)) - 1] = 1.0

        occupation, cross = (x[0] for x in self._tree_height._level_cross_moments((end,), start))
        mean = occupation @ np.column_stack([self._flattened_weights(r) for r in rewards])[:n - 1]

        return mean, F @ (cross + cross.T) @ F.T

    @BlasThreadLimit()
    def _bin_second_moments(
            self,
            rewards: Sequence[Reward],
            start_time: float = None,
            end_time: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The means ``E[R_a]`` and the second cross-moments ``E[R_a R_b]`` of the rewards over the accumulation window of
        the distribution, by the forward pass of ``PhaseTypeDistribution.moment`` shared by all rewards. The finite
        epochs take the stacked sparse action, or one dense exponential per reward where ``_dense_expm_is_faster``, and
        the unbounded last epoch is closed with one factorization of ``-T``.

        :param rewards: The rewards.
        :param start_time: The start time. By default, that of the distribution.
        :param end_time: The end time. By default, that of the distribution, or absorption.
        :return: The means, of shape ``(m,)``, and the cross-moments, of shape ``(m, m)``, for ``m`` rewards.
        :raises ValueError: If a reward is not accumulable, see ``Reward._check_accumulable``.
        :raises ModelError: If a moment is not a number.
        """
        if rewards and all(self._flattening_applies(2, (r, r)) for r in rewards):
            return self._bin_second_moments_flattened(rewards, start_time, end_time)

        ss = self.state_space
        start, end = self._resolve_window(start_time, end_time)
        closed = bool(np.isinf(end) and Settings.closed_form_last_epoch and self._absorption_certain_in_last_epoch(2))

        if closed:
            epochs = self._get_epochs_until_unbounded(2)
            self._check_demography_conditioning(epochs)
            finite = list(enumerate(epochs[:-1]))
            steps = self._closed_form_steps(epochs, start)
        else:
            if np.isinf(end):
                end = self._get_time_to_absorption()
            finite = []
            for i_epoch, epoch in enumerate(self.demography.epochs):
                finite.append((i_epoch, epoch))
                if epoch.end_time >= end:
                    break
            ss.update_epoch(finite[0][1])
            steps = [(epoch, min(epoch.end_time, end) - epoch.start_time) for _, epoch in finite]

        n = ss.k
        vectors = [np.asarray(r._get(ss), dtype=float) for r in rewards]
        Reward._check_accumulable(ss, rewards, vectors)
        R = np.column_stack(vectors)
        m = R.shape[1]
        use_action = not self._dense_expm_is_faster((2 * m + 1) * n, 3 * n, m, steps)
        scale = self._reward_scale(list(R.T))
        rewarded = (R != 0).any(axis=1).astype(float)

        self._logger.debug(
            "second moments of %d rewards over [%.3g, %.3g]: forward pass by the %s, %d finite epoch(s)%s", m, start,
            end, "stacked sparse action" if use_action else "dense exponential per reward", len(finite),
            ", closed last epoch" if closed else ""
        )

        # x = [p, u_1, ..., u_m, w_1, ..., w_m], with u_a stored divided by cu and w_a by cu * cw
        x = np.zeros((2 * m + 1) * n)
        x[:n] = ss.alpha
        cu = cw = 1.0

        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            for i_epoch, epoch in finite:
                ss.update_epoch(epoch)
                self._check_numerical_stability(ss.S, i_epoch)
                t0, t1 = epoch.start_time, epoch.end_time if closed else min(epoch.end_time, end)

                if t0 < start:
                    x[:n] = self._propagate_plain(x[:n], min(t1, start) - t0, use_action)
                    t0 = start

                if t1 <= t0:
                    continue

                # balance each epoch on its own rates, an exact diagonal similarity (see ``_rebase``)
                cu_new = self._get_regularization_factor(ss.S, t1 - t0, scale)
                cw_new = self._get_regularization_factor(ss.S, t1 - t0)
                x[n:(m + 1) * n] *= cu / cu_new
                x[(m + 1) * n:] *= cu * cw / (cu_new * cw_new)
                cu, cw = cu_new, cw_new

                if use_action:
                    A = self._forward_operator(sp.csr_matrix(ss.S), R, rewarded, cu, cw) * (t1 - t0)
                    x = Backend.expm_multiply(A, x, blocks=2 * m + 1, depth=3)
                    continue

                S = self._dense_rate_matrix()
                exps = self._block_expms([
                    lambda a=a: self._forward_van_loan(S, R[:, a], rewarded, cu, cw) * (t1 - t0) for a in range(m)
                ], n, [False, False, True])
                p, U, W = x[:n], x[n:(m + 1) * n].reshape(m, n), x[(m + 1) * n:].reshape(m, n)
                y = np.empty_like(x)
                for a, E in enumerate(exps):
                    if a == 0:
                        y[:n] = p @ E[:n, :n]
                    y[(1 + a) * n:(2 + a) * n] = p @ E[:n, n:2 * n] + U[a] @ E[n:2 * n, n:2 * n]
                    y[(1 + m + a) * n:(2 + m + a) * n] = p @ E[:n, 2 * n:] + U[a] @ E[n:2 * n, 2 * n:] + W[a]
                x = y

            p = x[:n]
            U = x[n:(m + 1) * n].reshape(m, n) * cu
            W = x[(m + 1) * n:].reshape(m, n) * (cu * cw)
            mean = U.sum(axis=1)
            ordered = W @ R  # [a, b]: reward a collected before reward b

            if closed:
                # w_a(inf) r_b = w_a(t_L) r_b + (u_a(t_L) + m_L * r_a) (-T)^{-1} r_b with m_L = p(t_L) (-T)^{-1}
                last = epochs[-1]
                ss.update_epoch(last)
                self._check_numerical_stability(ss.S, len(epochs) - 1)
                if start > last.start_time:
                    p = self._propagate_plain(p, start - last.start_time, use_action)

                idx_t = np.where(~ss.absorbing & self._alpha_support(2))[0]
                sparse = self._solve_sparse(len(idx_t))
                solve = self._last_epoch_solver(id(ss), len(epochs), idx_t.astype(np.intp).tobytes(), sparse)
                R_t = R[idx_t]
                m_last = solve.T(p[idx_t])
                mean = mean + m_last @ R_t
                ordered = ordered + (U[:, idx_t] + m_last * R_t.T) @ solve(R_t)

        if np.isnan(mean).any() or np.isnan(ordered).any():
            raise ModelError(
                "NaN value encountered when computing moment. This is likely due to an ill-conditioned rate matrix."
            )

        return mean, ordered + ordered.T
