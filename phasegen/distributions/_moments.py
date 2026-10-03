"""
Moment evaluation engine: the Van Loan / closed-form / matrix-exponential machinery shared by every
phase-type distribution, mixed into :class:`~phasegen.distributions.phase_type.PhaseTypeDistribution`.
"""

import itertools
import logging
from collections import deque
from ..caching import cache
from math import comb, factorial
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, TYPE_CHECKING
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csg
from ..coalescent_models import StandardCoalescent
from ..demography import Epoch
from ..errors import ModelError
from ..expm import Backend
from ..rewards import Reward, CustomReward, UnfoldedSFSReward, FoldedSFSReward, UnitReward, CombinedReward
from ..settings import Settings
from ..state_space import BlockCountingStateSpace

from ._common import _make_hashable, _validate_order, _validate_reward_count, _validate_start_time

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

#: Summed 1-norm of the steps ``S tau`` per cubed Van Loan dimension above which the closed form takes the dense
#: exponential in its finite epochs and its propagation to the start time, and below which the sparse action. The
#: action takes a number of matrix-vector products proportional to the step norm, the dense exponential a cost cubic
#: in the dimension. Measured crossover about 1e-5 to 1e-4 on block-counting spaces of 297 to 627 states, with the
#: dense exponential 8 to 30 times faster on lineage-counting chains of 260 to 600 states at 2e-4 to 2e-3.
#: ``_occupation_times`` applies it to each epoch on its own, with the dimension of its augmented generator.
_DENSE_EXPM_STEP_NORM = 5e-5

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
    def _van_loan_matrix(R, S, k: int = 1, sparse: bool = False, heads: int = 1) -> 'sp.spmatrix | np.ndarray':
        """
        The Van Loan matrix of ``PhaseTypeDistribution.moment``, assembled directly as sparse CSR when ``sparse``.
        With several ``heads`` (sparse only), the first block row and column are repeated once per head, each head
        coupled to the shared second block by its own reward.

        :param R: List of reward vectors, one per head followed by the rewards 2, ..., k.
        :param S: Intensity matrix (dense or sparse, matching ``sparse``).
        :param k: The order of the moment.
        :param sparse: Whether to build a sparse matrix.
        :param heads: The number of first blocks.
        :return: Van Loan matrix of :math:`(k + h) \times (k + h)` blocks for :math:`h` heads.
        """
        if sparse:
            blocks = [[None] * (k + heads) for _ in range(k + heads)]
            for i in range(k + heads):
                blocks[i][i] = S
                if heads - 1 <= i < k + heads - 1:
                    blocks[i][i + 1] = sp.diags(R[i])
            for a in range(heads - 1):
                blocks[a][heads] = sp.diags(R[a])
            return sp.bmat(blocks, format='csr')

        O = np.zeros_like(S)
        return np.block([
            [S if i == j else np.diag(R[i]) if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)
        ])

    @staticmethod
    def _van_loan_expm(A: np.ndarray, k: int, n: int) -> np.ndarray:
        """
        The dense exponential of a Van Loan matrix of :meth:`_van_loan_matrix`, with its strictly lower blocks set to
        their exact value of zero. The rebases between balancing factors multiply these blocks by powers of the
        factor ratio, so the rounding the exponential leaves in them would otherwise reach the moment.

        :param A: Van Loan matrix of ``(k + 1) x (k + 1)`` blocks of size ``n``, times the step.
        :param k: The order of the moment.
        :param n: The number of states, the size of one block.
        :return: The exponential of ``A``.
        """
        E = expm(A)
        for i in range(1, k + 1):
            E[i * n:(i + 1) * n, :i * n] = 0.0

        return E

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
        Distinct from :attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`, which
        governs the matrix-exponential action, though ``_accumulate_closed_form`` and ``_occupation_times`` let this
        one decide both, their exponentials following the sparsity their factorization already takes, except on the
        stiff generators of ``_DENSE_EXPM_STEP_NORM``.

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
        :return: Callable ``b -> x`` solving ``A x = b``.
        """
        if sparse:
            if isinstance(perm, str):
                perm = MomentEvaluator._block_triangular_order(A)
            if perm is None:
                return spla.splu(sp.csc_matrix(A)).solve

            inv = np.empty_like(perm)
            inv[perm] = np.arange(perm.size)
            lu = spla.splu(sp.csr_matrix(A)[perm][:, perm].tocsc(), permc_spec='NATURAL')
            return lambda b: lu.solve(np.asarray(b)[perm])[inv]

        lu = sla.lu_factor(A)
        return lambda b: sla.lu_solve(lu, b)

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

        - Matrix exponentials are formed densely while :math:`(k + 1)|E|` is below
          :attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`, and are otherwise
          applied to vectors as sparse actions (Al-Mohy and Higham, 2011). In the closed form below, the sparse action
          is taken from :attr:`Settings.closed_form_sparse_min_states
          <phasegen.settings.Settings.closed_form_sparse_min_states>` transient states on, except where the rates are
          stiff enough for the dense exponential to be faster.
        - :math:`\mathbf{U}` is never formed. A single LU factorization of :math:`-\mathbf{T}` serves all solves. It is
          sparse from :attr:`Settings.closed_form_sparse_min_states
          <phasegen.settings.Settings.closed_form_sparse_min_states>` transient states on, with the states ordered by
          the strongly connected components of the transition graph so that the factors stay nearly triangular.
        - The closed form requires :attr:`Settings.closed_form_last_epoch
          <phasegen.settings.Settings.closed_form_last_epoch>`, accumulation until absorption and certain absorption
          from every transient state of the last epoch that can carry mass. Otherwise the last epoch is integrated up to
          :attr:`TreeHeightDistribution.t_max <phasegen.distributions.TreeHeightDistribution.t_max>`.
        - Spectra share one computation across bins. The expected occupation times :math:`\mathbf{m}` of the
          transient states, which equal :math:`\boldsymbol{\alpha}_T \mathbf{U}` in a single epoch, give every bin
          mean as :math:`\mathbf{m}\, \mathbf{r}_j`, and in a single epoch the two-point occupation
          :math:`\operatorname{diag}(\mathbf{m})\, \mathbf{U}` gives every covariance at once. For one population
          under the :class:`~phasegen.coalescent_models.StandardCoalescent`, the mean SFS is computed on the smaller
          lineage-counting state space (:attr:`Settings.flatten_block_counting
          <phasegen.settings.Settings.flatten_block_counting>`).

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
        :raises ValueError: If ``k`` is not a non-negative integer, if the start time is negative, exceeds the
            end time, or lies beyond the time of almost sure absorption, if the population sizes and migration rates
            are too far apart for a reliable evaluation, or if the moment is not a number.
        """
        k = _validate_order(k)
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

    def _advance_action(self, w: np.ndarray, tau: float, lamb: float, op: Tuple, k: int) -> Tuple[np.ndarray, float]:
        """
        Advance the forward extended vector by ``tau`` within the current epoch by the sparse matrix-exponential
        action. Each step is balanced on the factor of the epoch capped at the reward scale times
        ``tau / _MIN_SCALED_STEP``, since block ``j`` of the vector gains the ``j``-th power of the scaled step, and
        floored at ``_MIN_BALANCE`` times the factor.

        :param w: Extended row vector of ``(k + 1)`` blocks, stored against ``lamb`` and rebased in place.
        :param tau: The step, non-positive for none.
        :param lamb: The factor ``w`` is stored against.
        :param op: The operator of :meth:`_action_operator` for the current epoch.
        :param k: The order of the moment.
        :return: The advanced vector and the factor it is stored against.
        """
        if tau <= 0:
            return w, lamb

        Vt, gen, rew, factor, scale = op
        if Settings.regularize:
            lamb_step = min(factor, max(scale * tau / _MIN_SCALED_STEP, _MIN_BALANCE * factor))
            lamb = self._rebase_forward(w, lamb, lamb_step, k, self.state_space.k)

        A = sp.csr_matrix((tau * gen + (tau / lamb) * rew, Vt.indices, Vt.indptr), shape=Vt.shape)

        return Backend.expm_multiply(A, w), lamb

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
        """
        Evaluate the kth (non-central) moment at different end times using the lineage counting state space.

        :param k: The order of the moment.
        :param end_times: Sequence of end times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time from which to start accumulation.
        :return: The moment accumulated at the specified times or time.
        :raises ValueError: If the state space is not a :class:`~phasegen.state_space.BlockCountingStateSpace`, or if
            k is not 1, or if there are multiple populations or loci or if the coalescent model is not the standard
            coalescent.
        """

        if not isinstance(self.state_space, BlockCountingStateSpace):
            raise ValueError("Flattened accumulation is only supported for BlockCountingStateSpace.")

        if k != 1:
            raise ValueError("Flattened accumulation is only supported for k = 1.")

        if self.lineage_config.n_pops != 1 or self.locus_config.n != 1:
            raise ValueError("Flattened accumulation is only supported for a single population and a single locus.")

        if not isinstance(self.state_space.model, StandardCoalescent):
            raise ValueError("Flattened accumulation is only supported for standard coalescent.")

        weights = self._flattened_weights(rewards[0] if rewards else self.reward)

        # Create a custom reward that returns the weights.
        weighted_reward = CustomReward(lambda _: weights)

        return self._tree_height._accumulate(
            k=k, end_times=end_times, rewards=(weighted_reward,), start_time=start_time
        )

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
        :param end_times: Sequence of ends times or end time when to evaluate the moment. A NaN end time gives a NaN
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
        if self._flattening_applies(k):
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
        ``e`` in the last block at each end time. The exponential is dense below :attr:`Settings.expm_action_min_dim
        <phasegen.settings.Settings.expm_action_min_dim>` and a sparse action above. For ``k >= 2`` the difference of
        two moments accumulated from 0 would omit the cross terms.

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

        use_action = (k + 1) * n >= Settings.expm_action_min_dim

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

        def advance(w, tau, lamb) -> Tuple[np.ndarray, float]:
            if use_action:
                return self._advance_action(w, tau, lamb, op, k)

            return w @ self._van_loan_expm(op * tau / lamb, k, n), lamb

        moments = np.zeros_like(t_sorted, dtype=float)

        # The Van Loan exponential is evaluated over the absorption time, which scales with Ne (the doubling search
        # in ``_get_absorption_time`` deliberately spans many orders of magnitude). For a large time the dense
        # ``expm`` can transiently over/underflow inside scipy's scaling-squaring on some BLAS builds, even though
        # the regularized result (corrected by ``lamb ** k``) is finite. The benign intermediate over/divide/invalid
        # is silenced here and the *output* is checked for finiteness below, so a genuine blow-up still surfaces.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            for i, u in enumerate(t_sorted):
                if u <= start_time:
                    # an empty (or reversed) window accumulates no reward
                    continue

                # advance through whole epochs between u_prev and u
                while u > epoch.end_time:
                    w, lamb = advance(w, epoch.end_time - u_prev, lamb)
                    u_prev = epoch.end_time
                    i_epoch, epoch = next(epochs)
                    self.state_space.update_epoch(epoch)

                    if use_action:
                        op = self._action_operator(rewards, k, i_epoch)
                    else:
                        # balance each epoch on its own rates (see ``_rebase``)
                        lamb_new = self._balance(epoch, start_time, t_sorted[-1], scale)
                        lamb = self._rebase_forward(w, lamb, lamb_new, k, n)
                        op = self._dense_van_loan(R, lamb, k, i_epoch)

                # remaining time in the current epoch
                w, lamb = advance(w, u - u_prev, lamb)
                moments[i] = factorial(k) * lamb ** k * float(w @ e_ext)
                u_prev = u

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
        w, prev = np.asarray(th.state_space.alpha, dtype=float), None

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
                if prev is not None:
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
        after deserialization is not. A payload that stored the memo in another form starts afresh.

        :param name: Attribute name of the memo.
        :return: The memo.
        """
        memo = self.__dict__.get(name)

        if not isinstance(memo, dict):
            memo = self.__dict__[name] = {}

        return memo

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
            heads: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Raw moment of a single reward ordering from ``start_time`` to absorption with the closed-form last epoch of
        ``PhaseTypeDistribution.moment``: the backward recursion with one LU of ``-T`` (``_lu_solver``), then the
        finite epochs from ``start_time`` on applied backwards to the extended vector, and last the entry distribution
        propagated to ``start_time``.

        With ``heads``, each head takes the place of the first reward in turn. The blocks of the extended vector after
        the first do not depend on the first reward, so one extended vector holds a first block per head and the
        remaining blocks once, and a single LU and, on the sparse action, a single action per epoch serve every head.

        :param k: The order of the moment, at least 1.
        :param rewards: Sequence of k rewards (a single ordering).
        :param start_time: The non-negative start time.
        :param heads: Rewards each substituted for the first of ``rewards``. By default, the first of ``rewards``.
        :return: The kth moment accumulated until absorption, one per head.
        """
        heads = tuple(rewards[:1]) if heads is None else tuple(heads)
        m = len(heads)

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
        # finite-epoch Van Loan exponentials take the sparse action with the sparse LU, except on the stiff generators
        # of ``_dense_expm_is_faster``.
        sparse = self._solve_sparse(len(idx_t))
        use_action = sparse and not self._dense_expm_is_faster(k, epochs, start_time)

        # the dense exponential of the stacked Van Loan matrix grows with the cube of the number of heads, so there each
        # head is evaluated on its own
        if m > 1 and not use_action:
            return np.concatenate([
                self._accumulate_closed_form(k, (h,) + tuple(rewards[1:]), start_time) for h in heads
            ])

        # transient sub-generator and its (sparse or dense) factorization, reused across the back-substitution
        T = self._transient_block(idx_t, sparse=sparse)
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
        solve = self._lu_solver(-T, sparse)

        # the j-th block of the extended vector is stored divided by lamb ** (k - j), which balances the Van Loan
        # exponentials of the finite epochs, so the moment is multiplied by lamb ** k at the end. The heads share one
        # factor, whose reward scale spans every head.
        r_all = [np.asarray(r._get(self.state_space), dtype=float) for r in rewards]
        r_heads = [np.asarray(h._get(self.state_space), dtype=float) for h in heads]
        scale = self._reward_scale(r_heads + r_all[1:])
        lamb = self._get_regularization_factor(self.state_space.S, scale=scale)

        # reward diagonals restricted to the transient states (the off-diagonal Van Loan reward blocks are diagonal)
        r_t = [r[idx_t] for r in r_all]

        nu = [None] * (k + 1)
        nu[k] = e[idx_t]
        for j in range(k - 1, 0, -1):
            nu[j] = solve(r_t[j] * nu[j + 1]) / lamb

        # the extended vector holds the first block of each head, then the blocks 1, ..., k shared by the heads
        z = np.zeros((m + k) * n)
        for a, r_h in enumerate(r_heads):
            z[a * n + idx_t] = solve(r_h[idx_t] * nu[1]) / lamb
        for j in range(1, k + 1):
            z[(m - 1 + j) * n + idx_t] = nu[j]
        z[(m - 1 + k) * n + idx_a] = e[idx_a]

        # --- preceding finite epochs, backward, via the (sparse or dense) full Van Loan matrix exponential ---
        for i_epoch, epoch in reversed(list(enumerate(epochs[:-1]))):
            if epoch.end_time <= start_time:
                break

            self.state_space.update_epoch(epoch)

            # balance each epoch on its own rates. ``V * tau`` carries the epoch's generator on the diagonal and its
            # reward blocks divided by the factor, so a factor drawn from one epoch leaves the reward blocks of an
            # epoch with a different rate scale far from one, and the scaling-and-squaring of the exponential loses
            # their cancellation. Rebasing the stored vector is the exact diagonal similarity that permits it.
            lamb_new = self._balance(epoch, start_time, np.inf, scale)
            z[:(m - 1) * n] *= (lamb / lamb_new) ** k
            lamb = self._rebase(z[(m - 1) * n:], lamb, lamb_new, k, n)

            self._check_numerical_stability(self.state_space.S, i_epoch)
            tau = (epoch.end_time - max(epoch.start_time, start_time)) / lamb

            if use_action:
                r_vecs = [np.asarray(r._get(self.state_space), dtype=float) for r in heads + tuple(rewards[1:])]
                V = self._van_loan_matrix(r_vecs, self._epoch_csr(i_epoch) * lamb, k, sparse=True, heads=m)
                z = Backend.expm_multiply(V * tau, z)
            else:
                R = [r._get(self.state_space) for r in heads + tuple(rewards[1:])]
                V = self._van_loan_matrix(R, self._dense_rate_matrix() * lamb, k)
                z = self._van_loan_expm(V * tau, k, n) @ z

        # the entry distribution propagated to the start time with the plain generator
        alpha_ext = np.zeros((k + 1) * n)
        alpha_ext[:n] = self.state_space.alpha
        for epoch in epochs:
            if epoch.start_time >= start_time:
                break

            self.state_space.update_epoch(epoch)
            alpha_ext[:n] = self._propagate_plain(alpha_ext[:n], min(epoch.end_time, start_time) - epoch.start_time,
                                                  use_action)

        return factorial(k) * lamb ** k * np.array([
            float(alpha_ext @ np.concatenate([z[a * n:(a + 1) * n], z[m * n:]])) for a in range(m)
        ])

    def _dense_expm_is_faster(self, k: int, epochs: List[Epoch], start_time: float) -> bool:
        """
        Whether the dense exponentials of ``_accumulate_closed_form``, those of the finite epochs and of the propagation
        to ``start_time``, are cheaper than the sparse action: below :attr:`Settings.expm_action_min_dim
        <phasegen.settings.Settings.expm_action_min_dim>` in the Van Loan dimension, where the summed 1-norm of the
        steps ``S tau`` exceeds ``_DENSE_EXPM_STEP_NORM`` times the cube of the dimension. Leaves the state space in
        the last epoch.

        :param k: The order of the moment.
        :param epochs: The epochs of ``_get_epochs_until_unbounded``.
        :param start_time: The non-negative start time.
        :return: Whether to form the dense exponentials.
        """
        dim = (k + 1) * self.state_space.k
        if dim >= Settings.expm_action_min_dim:
            return False

        # the exponentials span the time up to the later of the start time and the start of the last epoch
        horizon, norm = max(start_time, epochs[-1].start_time), 0.0
        for epoch in epochs:
            tau = min(epoch.end_time, horizon) - epoch.start_time
            if tau > 0:
                self.state_space.update_epoch(epoch)
                norm += float(abs(self.state_space.S).sum(axis=0).max()) * tau

        self.state_space.update_epoch(epochs[-1])

        return norm > _DENSE_EXPM_STEP_NORM * dim ** 3

    def _flattening_applies(self, k: int) -> bool:
        """
        Whether the block-counting state space can be flattened onto the lineage-counting state space for this
        moment: the first moment of the standard coalescent on a single population and a single locus. Takes
        precedence over the closed form and the batched occupation.
        """
        return (
                Settings.flatten_block_counting and
                k == 1 and
                isinstance(self.state_space, BlockCountingStateSpace) and
                isinstance(self.state_space.model, StandardCoalescent) and
                self.lineage_config.n_pops == 1 and
                self.locus_config.n == 1
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
        mean accumulation of a spectrum. A positive start time subtracts the occupation up to it.

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

        # the occupation integral is read off the augmented generator ``[[S, I], [0, 0]]``; for large state spaces
        # apply its (transposed) matrix-exponential action instead of forming the dense 2n x 2n exponential
        use_action = 2 * n >= Settings.expm_action_min_dim

        def advance(p, m, tau) -> 'Tuple[np.ndarray, np.ndarray]':
            if tau <= 0:
                return p, m
            S = self.state_space.S
            if use_action:
                aug = sp.bmat([
                    [sp.csc_matrix(S), sp.identity(n, format='csc')],
                    [None, sp.csc_matrix((n, n))],
                ], format='csc')
                w = spla.expm_multiply((aug * tau).T.tocsc(), np.concatenate([p, m]))
            else:
                aug = np.zeros((2 * n, 2 * n))
                aug[:n, :n] = np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)
                aug[:n, n:] = np.eye(n)
                w = np.concatenate([p, m]) @ expm(aug * tau)
            return w[:n], w[n:]

        p = np.asarray(self.state_space.alpha, dtype=float)
        m = np.zeros(n)
        u_prev = 0.0
        out = np.zeros((len(t_sorted), n))

        for idx, u in enumerate(t_sorted):
            while u > epoch.end_time:
                self._check_numerical_stability(self.state_space.S, i_epoch)
                p, m = advance(p, m, epoch.end_time - u_prev)
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, i_epoch)
            p, m = advance(p, m, u - u_prev)
            out[idx] = m
            u_prev = u

        return out[np.argsort(order)]

    def _occupation_times(self, cap: float = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Expected occupation times of the transient states until absorption (the vector ``m`` of
        ``PhaseTypeDistribution.moment``). Finite epochs use the augmented generator ``[[T_i, I], [0, 0]]`` on the
        transient block, and the final epoch contributes ``p (-T)^{-1}``. With ``cap`` the accumulation stops at that
        time, and the epoch containing it is treated as finite, so a windowed mean subtracts the capped occupation.

        :param cap: If given, accumulate occupation only up to this absolute time instead of to absorption.
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
        # apply the sparse matrix-exponential action instead of forming the dense ``2 nt x 2 nt`` exponential.
        for i_epoch, epoch in enumerate(epochs):
            self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, i_epoch)

            if not (np.isfinite(p).all() and np.isfinite(m).all()):
                raise ModelError(
                    f"Non-finite occupation times at the start of epoch {i_epoch}. This is likely due to an "
                    "ill-conditioned rate matrix. Use less extreme population sizes or growth rates."
                )

            upper = epoch.end_time if cap is None else min(epoch.end_time, cap)

            if np.isinf(upper):
                # final unbounded epoch, no cap: occupation to absorption = p (-T)^{-1}, i.e. solve (-T)^T x = p
                neg_t = -self._transient_block(idx_t, sparse=sparse)
                m += self._lu_solver(neg_t.T, sparse)(p)
                break

            S = self._transient_block(idx_t, sparse=sparse)
            tau = upper - epoch.start_time

            # the dense exponential where the step is stiff enough to be the cheaper one, see ``_DENSE_EXPM_STEP_NORM``
            use_action = sparse and not (
                    2 * nt < Settings.expm_action_min_dim and
                    float(abs(S).sum(axis=0).max()) * tau > _DENSE_EXPM_STEP_NORM * (2 * nt) ** 3
            )

            if use_action:
                aug = sp.bmat([
                    [sp.csc_matrix(S), sp.identity(nt, format='csc')],
                    [None, sp.csc_matrix((nt, nt))]
                ], format='csc')
                # [p, 0] exp(aug tau) = (exp((aug tau)^T) [p; 0])^T, so apply the action to the transposed generator
                w = spla.expm_multiply((aug * tau).T.tocsc(), np.concatenate([p, np.zeros(nt)]))
                m += w[nt:]
                p = w[:nt]
            else:
                aug = np.zeros((2 * nt, 2 * nt))
                aug[:nt, :nt] = S.toarray() if sp.issparse(S) else S
                aug[:nt, nt:] = np.eye(nt)
                exp_aug = expm(aug * tau)
                m += p @ exp_aug[:nt, nt:]
                p = p @ exp_aug[:nt, :nt]

            if cap is not None and upper >= cap:
                break

        return m, idx_t

    def _two_point_occupation(self) -> Optional[Tuple[np.ndarray, 'Callable', np.ndarray]]:
        """
        Factors of the two-point occupation matrix ``K = diag(m) (-T)^{-1}`` of ``PhaseTypeDistribution.moment``,
        defined for a single epoch without an accumulation window, so that ``R^T K R = (m * R)^T solve(R)`` for a
        stack of reward columns ``R``. ``K`` is never formed, and the LU of ``-T`` is sparse from
        :attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`
        transient states on. Other cases return ``None`` and callers evaluate per pair.

        :return: ``(m, solve, idx_t)`` with ``m = alpha (-T)^{-1}`` and ``solve`` applying ``(-T)^{-1}``, over the
            transient states ``idx_t``, or ``None`` when not applicable (caller falls back).
        """
        if not (Settings.closed_form_last_epoch and self._tree_height.end_time is None):
            return None

        if self._tree_height.start_time > 0:
            # the two-point occupation is a double integral over ``s < u``; the windowed (start_time > 0) version is
            # not the full one minus a box, so the batched covariance cannot subtract it the way the mean does. Fall
            # back to the per-pair path, which accumulates each pair over ``[start_time, absorption]`` directly.
            self._logger.debug(
                "two-point occupation: start_time=%.3g > 0; using per-pair covariance (windowed two-point occupation "
                "not batched)", self._tree_height.start_time
            )
            return None

        epochs = self._get_epochs_until_unbounded(2)

        # only the single-epoch closed form is used; the multi-epoch ODE is stiffness-fragile (see docstring)
        if len(epochs) > 1:
            self._logger.debug(
                "two-point occupation: %d epochs; using per-pair matrix-exponential (multi-epoch closed form "
                "disabled)", len(epochs)
            )
            return None

        if not self._absorption_certain_in_last_epoch(2):
            return None

        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, 0)
        absorbing = self.state_space.absorbing
        # only the states that can carry mass; see ``_alpha_support``
        idx_t = np.where(~absorbing & self._alpha_support(2))[0]

        sparse = self._solve_sparse(len(idx_t))
        neg_t = -self._transient_block(idx_t, sparse=sparse)
        m = self._lu_solver(neg_t.T, sparse)(np.asarray(self.state_space.alpha, dtype=float)[idx_t])

        self._logger.debug(
            "two-point occupation: single-epoch closed form diag(m)(-T)^-1, %s LU (n_t=%d)",
            'sparse' if sparse else 'dense', len(idx_t)
        )

        return m, self._lu_solver(neg_t, sparse), idx_t
