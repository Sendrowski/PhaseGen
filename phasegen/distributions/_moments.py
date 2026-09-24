"""
Moment evaluation engine: the Van Loan / closed-form / matrix-exponential machinery shared by every
phase-type distribution, mixed into :class:`~phasegen.distributions.phase_type.PhaseTypeDistribution`.
"""

import itertools
import logging
from collections import deque
from ..caching import cache
from math import comb, factorial
from typing import Dict, List, Tuple, Collection, Iterable, Optional, Sequence, TYPE_CHECKING
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csg
from ..coalescent_models import StandardCoalescent
from ..demography import Epoch
from ..expm import Backend
from ..rewards import Reward, CustomReward, UnfoldedSFSReward, FoldedSFSReward, UnitReward, CombinedReward
from ..settings import Settings
from ..state_space import BlockCountingStateSpace

from ._common import _make_hashable

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


class MomentEvaluator:
    """Moment-evaluation methods of :class:`~phasegen.distributions.PhaseTypeDistribution`, described in its
    ``moment``."""

    # attributes provided by the host PhaseTypeDistribution this mixin is mixed into
    state_space: 'StateSpace'
    tree_height: 'TreeHeightDistribution'
    demography: 'Demography'
    reward: Reward
    lineage_config: 'LineageConfig'
    locus_config: 'LocusConfig'
    _logger: logging.Logger
    _absorption_certain_cache: Dict[int, bool]
    _alpha_support_cache: Dict[int, np.ndarray]
    _epochs_cache: Optional[List[Epoch]]

    @staticmethod
    def _van_loan_matrix(R, S, k: int = 1, sparse: bool = False) -> 'sp.spmatrix | np.ndarray':
        """
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
        one decide both, their Van Loan exponential following the sparsity their factorization already takes.

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
          applied to vectors as sparse actions (Al-Mohy and Higham, 2011).
        - :math:`\mathbf{U}` is never formed. A single LU factorization of :math:`-\mathbf{T}` serves all solves. It is
          sparse from :attr:`Settings.closed_form_sparse_min_states
          <phasegen.settings.Settings.closed_form_sparse_min_states>` transient states on, with the states ordered by
          the strongly connected components of the transition graph so that the factors stay nearly triangular.
        - The closed form requires :attr:`Settings.closed_form_last_epoch
          <phasegen.settings.Settings.closed_form_last_epoch>`, accumulation until absorption from a zero start time,
          and certain absorption from every transient state of the last epoch that can carry mass. Otherwise the last
          epoch is integrated up to :attr:`TreeHeightDistribution.t_max
          <phasegen.distributions.TreeHeightDistribution.t_max>`.
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
        :raises ValueError: If the start time is negative, exceeds the end time, or lies beyond the time of almost
            sure absorption, if the population sizes and migration rates are too far apart for a reliable
            evaluation, or if the moment is not a number.
        """
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            # an infinite end time accumulates until absorption
            end_time = np.inf if self.tree_height.end_time is None else self.tree_height.end_time

        if start_time < 0:
            raise ValueError("Start time must be greater than or equal to 0.")

        if start_time > 0 and np.isinf(end_time):
            t_absorption = self._get_time_to_absorption()

            if start_time > t_absorption:
                raise ValueError(
                    f"The window start time ({start_time:.1f}) lies beyond the time of almost sure absorption "
                    f"({t_absorption:.1f}), so the accumulation window is empty."
                )

        if end_time < start_time:
            raise ValueError("End time must be greater than equal start time.")

        if start_time > 0 and int(k) == 1:
            # the mean is additive in time, so the windowed mean is the difference of the two cumulative means
            m_start, m_end = MomentEvaluator.accumulate(
                self,
                k=k,
                end_times=[start_time, end_time],
                rewards=rewards,
                center=center,
                permute=permute,
                start_time=0.0
            )

            m = float(m_end - m_start)
        elif start_time > 0:
            # for k >= 2 the windowed moment ``E[(Y_b - Y_a)^k]`` is NOT the difference of the cumulative-from-0
            # moments ``m_b - m_a`` (that omits the cross terms); accumulate it directly over the window by
            # propagating the entry distribution to ``start_time`` and running the Van Loan accumulation from there
            m = float(MomentEvaluator.accumulate(
                self,
                k=k,
                end_times=[end_time],
                rewards=rewards,
                center=center,
                permute=permute,
                start_time=start_time
            )[0])
        else:
            m = float(MomentEvaluator.accumulate(
                self,
                k=k,
                end_times=[end_time],
                rewards=rewards,
                center=center,
                permute=permute,
                start_time=0.0
            )[0])

        if np.isnan(m):
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return m

    @staticmethod
    def _get_regularization_factor(S: np.ndarray, duration: float = np.inf) -> float:
        """
        The balancing factor of the Van Loan matrix, the reciprocal geometric mean of the positive rates of ``S``
        capped at ``duration``, or 1 when ``Settings.regularize`` is disabled. Scaling ``S`` by it and the step by its
        inverse divides the reward blocks by the factor, which the callers undo by multiplying the moment by its
        ``k``-th power. The cap keeps the scaled step at or above one: the block of order ``j`` of the exponential
        scales as the ``j``-th power of the scaled step, and below one the higher orders fall under the resolution of
        double precision.

        :param S: Intensity matrix.
        :param duration: Length of the epoch within the accumulation window.
        :return: Regularization factor.
        """
        if not Settings.regularize:
            return 1.0

        # obtain positive rates (for a sparse matrix, the positive stored entries)
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        # rewards in the Van Loan matrix are of order 1
        factor = 10 ** - np.log10(rates).mean()

        return min(factor, duration) if duration > 0 else factor

    def _balance(self, epoch: 'Epoch', start: float, end: float) -> float:
        """
        The regularization factor of the current rate matrix for the part of ``epoch`` within ``[start, end]``.

        :param epoch: The epoch the state space is set to.
        :param start: Start of the accumulation window.
        :param end: End of the accumulation window.
        :return: Regularization factor.
        """
        duration = min(epoch.end_time, end) - max(epoch.start_time, start)

        return self._get_regularization_factor(self.state_space.S, duration)

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
        Rebase the forward extended vector ``w = alpha_ext Q`` from one balancing factor onto another, in place, the
        row counterpart of :meth:`_rebase_propagator`. Block ``j`` of ``w`` is row block 0 of ``Q``, so it carries the
        factor to the power ``-j`` and is multiplied by ``(lamb / lamb_new) ** j``. An exact diagonal similarity.

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

    @staticmethod
    def _rebase_propagator(Q: np.ndarray, lamb: float, lamb_new: float, k: int, n: int) -> float:
        """
        Rebase the extended propagator from one balancing factor onto another, in place, the matrix counterpart of
        :meth:`_rebase`.

        Block ``(i, j)`` of ``Q`` carries the factor to the power ``i - j``, so changing the factor multiplies that
        block by ``(lamb / lamb_new) ** (j - i)``. Like :meth:`_rebase` this is an exact diagonal similarity.

        :param Q: Extended propagator of ``(k + 1) x (k + 1)`` blocks of size ``n``, modified in place.
        :param lamb: The factor the blocks are currently stored against.
        :param lamb_new: The factor to store them against.
        :param k: The order of the moment.
        :param n: The number of states, the size of one block.
        :return: ``lamb_new``, the factor now in force.
        """
        if lamb_new == lamb:
            return lamb

        ratio = lamb / lamb_new
        for i in range(k + 1):
            for j in range(k + 1):
                if i != j:
                    Q[i * n:(i + 1) * n, j * n:(j + 1) * n] *= ratio ** (j - i)

        return lamb_new

    def _check_demography_conditioning(self) -> None:
        """
        Fail fast when the population sizes and migration rates of the first epoch span more than double precision,
        which makes both the absorption-time search and the closed-form solve unreliable. Keyed on the demography, not
        the rate matrix, whose range multiple-merger models widen legitimately.

        :raises ValueError: if the population sizes and migration rates differ by a factor of more than ``1e16``.
        """
        epoch = self.demography.get_epoch(0)

        # coalescence rates scale as 1 / pop_size, migration enters at its own rate
        scales = [1 / v for v in epoch.pop_sizes.values() if v > 0]
        scales += [v for v in epoch.migration_rates.values() if v > 0]
        ratio = max(scales) / min(scales) if scales else 1

        if ratio > 1e16:
            raise ValueError(
                "The demography is too ill-conditioned to reliably compute the time of almost sure absorption: its "
                f"population sizes and migration rates differ by a factor of {ratio:.1e}. Use less extreme "
                "parameters, or set the end time manually (see ``Coalescent.end_time``)."
            )

    def _check_numerical_stability(self, S: np.ndarray, epoch: int) -> None:
        """
        Warn about potential numerical instability with very small or very large rates, once per epoch.

        :param S: (Regularized) intensity matrix.
        :param epoch: Epoch number.
        """
        warned = self.__dict__.setdefault('_stability_warned', set())
        if epoch in warned:
            return

        # positive (off-diagonal) rates; for a sparse matrix these are the positive stored entries
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        if rates.min() / rates.max() < 1e-10:
            warned.add(epoch)
            self._logger.warning(
                f"Intensity matrix in epoch {epoch} contains rates that differ by more than 10 orders of magnitude: "
                f"min: {rates.min()}, max: {rates.max()}. "
                f"This may lead to numerical instability, despite matrix regularization."
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
        """
        k = int(k)

        if start_time is None:
            start_time = self.tree_height.start_time

        if rewards is None:
            rewards = [self.reward] * k

        if k != len(rewards):
            raise ValueError(f"Number of specified rewards for moment of order {k} must be {k}.")

        if k == 0:
            return np.ones_like(list(end_times))

        # center moments around the mean
        if center and k > 1:
            self._logger.debug("accumulate (k=%d): centering (subtracting lower-order moment products)", k)

            components = []

            # first order moments
            means = [
                MomentEvaluator.accumulate(
                    self,
                    k=1,
                    rewards=(rewards[i],),
                    end_times=end_times,
                    start_time=start_time
                ) for i in range(k)
            ]

            for i in range(k + 1):
                # iterate over all possible subsets of rewards of size i
                for indices in itertools.combinations(range(k), i):
                    # joint moment
                    mu_i = MomentEvaluator.accumulate(
                        self,
                        k=i,
                        rewards=tuple(rewards[j] for j in indices),
                        end_times=end_times,
                        center=False,
                        permute=permute,
                        start_time=start_time
                    )

                    # product of means of remaining rewards
                    mu1 = np.prod([means[j] for j in range(k) if j not in indices], axis=0)

                    components += [(-1) ** (k - i) * mu_i * mu1]

            return np.sum(components, axis=0)

        if permute:
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
            rewards: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times using the lineage counting state space.

        :param k: The order of the moment.
        :param end_times: Sequence of end times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
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

        reward = rewards[0] if rewards else self.reward
        n = self.lineage_config.n

        # Prefer the closed-form Kingman block-size weights, which never build the (p(n)-state) block-counting space.
        weights = self._flattened_sfs_weights(reward, n)
        if weights is not None:
            self._logger.debug(
                "flattening block-counting onto the lineage-counting state space (%d states) via the closed-form "
                "Kingman block-size weights", self.tree_height.state_space.k
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
                len(self.state_space.states), self.tree_height.state_space.k
            )

        # Create a custom reward that returns the weights.
        weighted_reward = CustomReward(lambda _: weights)

        return self.tree_height._accumulate(k=k, end_times=end_times, rewards=(weighted_reward,))

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
        :param end_times: Sequence of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time from which to start accumulation. When positive, delegates to
            ``_accumulate_windowed``, which accumulates the reward over the window ``[start_time, t]`` directly.
            By default, ``0`` (accumulation from the origin).
        :return: The moment accumulated at the specified times or time.
        """
        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        elif len(rewards) != k:
            raise ValueError(f"Number of rewards must be {k}.")

        end_times = np.array(end_times, dtype=float)

        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        # flattening takes precedence over the closed form (it shrinks the state space, which dominates the cost) and
        # re-enters this method on the lineage-counting state space, where the flattened reward is checked
        if start_time <= 0 and self._flattening_applies(k):
            self._logger.debug("accumulate (k=%d): flattened block-counting", k)
            return self._accumulate_flattened(k, end_times, rewards)

        Reward._check_accumulable(self.state_space, rewards)

        # windowed accumulation: propagate the entry distribution to the window start, then run the Van Loan
        # accumulation over ``[start_time, t]`` (the correct k >= 2 windowed moment, not m_end - m_start)
        if start_time > 0:
            end_times = np.where(np.isinf(end_times), self._get_time_to_absorption(), end_times)
            return self._accumulate_windowed(k, float(start_time), end_times, rewards)

        # infinite end times accumulate until absorption, in closed form when absorption is certain in the last
        # epoch and otherwise over the estimated absorption time
        infinite = np.isinf(end_times)
        if infinite.any():
            if Settings.closed_form_last_epoch and self._absorption_certain_in_last_epoch():
                self._logger.debug("accumulate (k=%d): closed-form last epoch", k)
                moments = np.empty(end_times.shape)
                moments[infinite] = self._accumulate_closed_form(k, rewards)
                if not infinite.all():
                    moments[~infinite] = self._accumulate(k, tuple(end_times[~infinite]), rewards)
                return moments

            end_times = np.where(infinite, self._get_time_to_absorption(), end_times)

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(end_times)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get state space for the first epoch
        self.state_space.update_epoch(epoch)

        # number of states
        n_states = self.state_space.k

        # for large (sparse) state spaces, compute the moment via the action of the matrix exponential on a vector
        # (threading through the epochs) instead of forming the dense Van Loan propagator
        if (k + 1) * n_states >= Settings.expm_action_min_dim:
            self._logger.debug(
                "accumulate (k=%d): sparse matrix-exponential action (Van Loan dim %d >= %d)",
                k, (k + 1) * n_states, Settings.expm_action_min_dim
            )
            return self._accumulate_action(k, end_times, t_sorted, rewards)

        self._logger.debug("accumulate (k=%d): dense Van Loan matrix exponential (dim %d)", k, (k + 1) * n_states)

        # initialize block matrix holding (rewarded) moments
        Q = np.eye(n_states * (k + 1))
        u_prev = 0

        # initialize probabilities
        moments = np.zeros_like(t_sorted, dtype=float)

        # regularization parameter
        lamb = self._balance(epoch, 0.0, t_sorted[-1])

        # regularized intensity matrix
        S = self._dense_rate_matrix() * lamb

        # check numerical stability
        self._check_numerical_stability(S, 0)

        # get reward matrix
        R = [r._get(state_space=self.state_space) for r in rewards]

        # get Van Loan matrix
        V = self._van_loan_matrix(R, S, k)

        # The Van Loan exponential is evaluated over the absorption time, which scales with Ne (the doubling search
        # in ``_get_absorption_time`` deliberately spans many orders of magnitude). For a large time the dense
        # ``expm`` can transiently over/underflow inside scipy's scaling-squaring on some BLAS builds, even though
        # the regularized result (corrected by ``lamb ** k``) is finite. The benign intermediate over/divide/invalid
        # is silenced here and the *output* is checked for finiteness below, so a genuine blow-up still surfaces.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            # iterate through sorted values
            for i, u in enumerate(t_sorted):

                # iterate over epochs between u_prev and u
                while u > epoch.end_time:
                    # update transition matrix with remaining time in current epoch
                    Q @= expm(V * (epoch.end_time - u_prev) / lamb)

                    # fetch and update for next epoch
                    u_prev = epoch.end_time
                    i_epoch, epoch = next(epochs)
                    self.state_space.update_epoch(epoch)

                    # balance each epoch on its own rates, rebasing the propagator accordingly (see ``_rebase``)
                    lamb = self._rebase_propagator(Q, lamb, self._balance(epoch, 0.0, t_sorted[-1]), k, n_states)

                    # compute Van Loan matrix for next epoch using regularized intensity matrix
                    S = self._dense_rate_matrix() * lamb
                    self._check_numerical_stability(S, i_epoch)
                    V = self._van_loan_matrix(R, S, k)

                # update with remaining time in current epoch
                Q @= expm(V * (u - u_prev) / lamb)

                alpha = self.state_space.alpha
                e = self.state_space.e
                moments[i] = factorial(k) * lamb ** k * alpha @ Q[:n_states, -n_states:] @ e

                u_prev = u

        # sort probabilities back to original order (inverse of the sorting permutation)
        moments = moments[np.argsort(np.argsort(end_times))]

        # the suppressed intermediate over/underflow must not have corrupted the (finite) result
        if not np.isfinite(moments).all():
            self._logger.warning(
                "Non-finite values encountered when computing moments. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def _accumulate_action(
            self,
            k: int,
            end_times: np.ndarray,
            t_sorted: np.ndarray,
            rewards: Sequence[Reward]
    ) -> np.ndarray:
        """
        Sparse-action variant of ``_accumulate``: threads the row vector holding ``alpha`` in its first block through
        the epochs by the action of the transposed Van Loan matrix and reads off its product with ``e`` in the last
        block at each end time.

        :param k: The order of the moment.
        :param end_times: The (unsorted) end times, used to restore the original order.
        :param t_sorted: The sorted end times.
        :param rewards: Sequence of k rewards.
        :return: The moment accumulated at the specified times.
        """
        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)
        self.state_space.update_epoch(epoch)

        n = self.state_space.k
        lamb = self._balance(epoch, 0.0, t_sorted[-1])

        def transposed_van_loan() -> 'sp.spmatrix':
            """Transposed sparse Van Loan matrix for the current epoch (transposed for the left vector action)."""
            S = self.state_space.S * lamb
            self._check_numerical_stability(S, i_epoch)
            r_vecs = [np.asarray(r._get(state_space=self.state_space), dtype=float) for r in rewards]
            return self._van_loan_matrix(r_vecs, sp.csr_matrix(S), k, sparse=True).T.tocsr()

        Vt = transposed_van_loan()

        # w = alpha_ext (alpha in the first block); e_ext = e in the last block, so w @ Q @ e_ext = alpha @ Q[:n,-n:] @ e
        w = np.zeros((k + 1) * n)
        w[:n] = self.state_space.alpha
        e_ext = np.zeros((k + 1) * n)
        e_ext[-n:] = self.state_space.e

        moments = np.zeros_like(t_sorted, dtype=float)
        u_prev = 0.0

        for i, u in enumerate(t_sorted):

            # advance through whole epochs between u_prev and u
            while u > epoch.end_time:
                w = Backend.expm_multiply(Vt * ((epoch.end_time - u_prev) / lamb), w)
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

                # balance each epoch on its own rates (see ``_rebase``)
                lamb = self._rebase_forward(w, lamb, self._balance(epoch, 0.0, t_sorted[-1]), k, n)
                Vt = transposed_van_loan()

            # remaining time in the current epoch
            w = Backend.expm_multiply(Vt * ((u - u_prev) / lamb), w)
            moments[i] = factorial(k) * lamb ** k * float(w @ e_ext)
            u_prev = u

        moments = moments[np.argsort(np.argsort(end_times))]

        if np.isnan(moments).any():
            self._logger.warning(
                "NaN values encountered when computing moments via the matrix-exponential action. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

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

    def _accumulate_windowed(
            self,
            k: int,
            start_time: float,
            end_times: np.ndarray,
            rewards: Sequence[Reward]
    ) -> np.ndarray:
        """
        Raw moment of a single reward ordering over the window ``[start_time, t]`` for each ``t`` in ``end_times``:
        propagates ``alpha`` to ``start_time`` with the plain generator, then runs the Van Loan accumulation from
        there. For ``k >= 2`` the difference of two moments accumulated from 0 would omit the cross terms.

        :param k: The order of the moment.
        :param start_time: The (positive) window start time.
        :param end_times: The window end times.
        :param rewards: Sequence of k rewards (a single ordering).
        :return: The windowed moment accumulated over ``[start_time, t]`` for each ``t`` in ``end_times``.
        """
        end_times = np.asarray(end_times, dtype=float)
        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

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
            "accumulate (k=%d): windowed from t=%.3g via the propagated entry distribution (%s Van Loan)",
            k, start_time, "sparse action" if use_action else "dense"
        )

        # rewards are epoch-invariant (they depend on the states, not the rates), matching the cumulative paths;
        # only the intensity matrix (and hence the Van Loan matrix) is refreshed per epoch
        lamb = self._balance(epoch, start_time, t_sorted[-1])
        moments = np.zeros_like(t_sorted, dtype=float)

        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            if use_action:
                def transposed_van_loan() -> 'sp.spmatrix':
                    """Transposed sparse Van Loan matrix for the current epoch (for the left vector action)."""
                    S = self.state_space.S * lamb
                    self._check_numerical_stability(S, i_epoch)
                    r_vecs = [np.asarray(r._get(state_space=self.state_space), dtype=float) for r in rewards]
                    return self._van_loan_matrix(r_vecs, sp.csr_matrix(S), k, sparse=True).T.tocsr()

                Vt = transposed_van_loan()
                # w = alpha_start in the first block; e_ext = e in the last block, so w @ Q @ e_ext = alpha_start @ Q[:n,-n:] @ e
                w = np.zeros((k + 1) * n)
                w[:n] = alpha_start
                e_ext = np.zeros((k + 1) * n)
                e_ext[-n:] = self.state_space.e

                for i, u in enumerate(t_sorted):
                    if u <= start_time:
                        # an empty (or reversed) window accumulates no reward
                        moments[i] = 0.0
                        continue
                    while u > epoch.end_time:
                        w = Backend.expm_multiply(Vt * ((epoch.end_time - u_prev) / lamb), w)
                        u_prev = epoch.end_time
                        i_epoch, epoch = next(epochs)
                        self.state_space.update_epoch(epoch)

                        # balance each epoch on its own rates (see ``_rebase``)
                        lamb = self._rebase_forward(w, lamb, self._balance(epoch, start_time, t_sorted[-1]), k, n)
                        Vt = transposed_van_loan()
                    w = Backend.expm_multiply(Vt * ((u - u_prev) / lamb), w)
                    moments[i] = factorial(k) * lamb ** k * float(w @ e_ext)
                    u_prev = u
            else:
                S = self._dense_rate_matrix() * lamb
                self._check_numerical_stability(S, i_epoch)
                R = [r._get(state_space=self.state_space) for r in rewards]
                V = self._van_loan_matrix(R, S, k)
                Q = np.eye(n * (k + 1))
                e = np.asarray(self.state_space.e)

                for i, u in enumerate(t_sorted):
                    if u <= start_time:
                        # an empty (or reversed) window accumulates no reward
                        moments[i] = 0.0
                        continue
                    while u > epoch.end_time:
                        Q @= expm(V * (epoch.end_time - u_prev) / lamb)
                        u_prev = epoch.end_time
                        i_epoch, epoch = next(epochs)
                        self.state_space.update_epoch(epoch)

                        # balance each epoch on its own rates (see ``_rebase``)
                        lamb = self._rebase_propagator(Q, lamb, self._balance(epoch, start_time, t_sorted[-1]), k, n)
                        S = self._dense_rate_matrix() * lamb
                        self._check_numerical_stability(S, i_epoch)
                        V = self._van_loan_matrix(R, S, k)
                    Q @= expm(V * (u - u_prev) / lamb)
                    moments[i] = factorial(k) * lamb ** k * alpha_start @ Q[:n, -n:] @ e
                    u_prev = u

        # restore the original (unsorted) order
        moments = moments[np.argsort(np.argsort(end_times))]

        if not np.isfinite(moments).all():
            self._logger.warning(
                "Non-finite values encountered when computing windowed moments. "
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
        if self.tree_height.end_time is None:
            return self.tree_height.t_max

        return self.tree_height._get_absorption_time()

    def _get_epochs_until_unbounded(self) -> List[Epoch]:
        """
        Materialize the demographic epochs up to and including the one taken to hold until absorption. The iteration
        stops at an epoch with an infinite end time, or at the first epoch beginning at or after the time of almost
        sure absorption (``TreeHeightDistribution.t_max`` without an accumulation window), whose rates are then
        extended over the remaining time.

        :return: List of epochs, the last of which is unbounded.
        """
        # the epochs depend only on the demography and the absorption time, both fixed for the distribution, while
        # the closed form queries them once per moment and an SFS evaluates many bins
        if getattr(self, '_epochs_cache', None) is not None:
            return self._epochs_cache

        epochs, t_absorption, survival, scale, extra = [], None, 0.0, 1.0, 0

        for epoch in self.demography.epochs:

            if epoch.end_time == np.inf:
                epochs.append(epoch)
                break

            # the bound costs an absorption-time search, so it is evaluated only once a finite epoch requires it
            if t_absorption is None:
                t_absorption = self._get_time_to_absorption()
                survival = self.tree_height._survival(t_absorption)
                scale = self.tree_height._get_absorption_scale()

            # an epoch reached after absorption is almost sure stands in for every epoch after it, so it may only be
            # held where the reward that substitution misplaces is negligible. The count bounds the search, a
            # demography having infinitely many epochs by design.
            if (epoch.start_time >= t_absorption and extra < self.tree_height._max_extension_epochs
                    and not self.tree_height._extension_is_negligible(epoch, survival, scale)):
                epochs.append(epoch)
                extra += 1
                continue

            if epoch.start_time >= t_absorption:
                epochs.append(Epoch(
                    start_time=epoch.start_time,
                    end_time=np.inf,
                    pop_sizes=epoch.pop_sizes,
                    migration_rates=epoch.migration_rates
                ))
                break

            epochs.append(epoch)

        self._epochs_cache = epochs

        return epochs

    def _absorption_certain_in_last_epoch(self) -> bool:
        """
        Whether every transient state that can carry mass (see ``_alpha_support``) can reach an absorbing state in the
        final epoch, so that ``-T`` restricted to those states is non-singular and the closed form applies. When
        ``False``, for example for a migration barrier in the last epoch, absorption may still happen in earlier
        epochs, and callers use the matrix exponential up to the absorption-time estimate.

        :return: Whether absorption is certain from every transient state of the last epoch that can carry mass.
        """
        # the result depends only on the (fixed) last-epoch structure, so memoize it: the closed form queries this
        # once per moment, and an SFS/jSFS evaluates many bins, so recomputing the reachability each time dominated.
        # One host serves several state spaces (the lineage-counting one for a tree height, the block-counting one
        # for a spectrum), so the memo is per state space.
        cache = self.__dict__.setdefault('_absorption_certain_cache', {})
        key = id(self.state_space)

        if key in cache:
            return cache[key]

        support = self._alpha_support()

        self.state_space.update_epoch(self._get_epochs_until_unbounded()[-1])
        absorbing, reach = self._reaches_absorption()

        # only the states that can carry mass matter. A state the initial vector never reaches, such as a deme
        # declared with no samples and no migration into it, has no bearing on whether absorption is certain.
        cache[key] = bool(reach[support & ~absorbing].all())
        return cache[key]

    def _alpha_support(self) -> np.ndarray:
        """
        The states that can carry probability mass: the forward closure of the initial support under each epoch's
        transitions in turn. Memoized per state space, one host serving several of them.

        :return: Boolean mask over the states of the current state space.
        """
        ss = self.state_space
        cache = self.__dict__.setdefault('_alpha_support_cache', {})
        key = id(ss)

        if key in cache:
            return cache[key]

        support = np.asarray(ss.alpha) > 0

        for epoch in self._get_epochs_until_unbounded():
            ss.update_epoch(epoch)
            S = ss.S
            adj = (S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S)))
            adj = (adj != 0).T.tocsr()

            while True:
                nxt = support | (adj @ support > 0)
                if np.array_equal(nxt, support):
                    break
                support = nxt

        cache[key] = support
        return support

    def _reaches_absorption(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward reachability over the rate graph of the current epoch: a state reaches absorption if it is absorbing
        or has a positive rate to a state that does, propagated with a sparse adjacency matrix. Used by
        ``_absorption_certain_in_last_epoch`` and ``_assert_absorbs``.

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

    def _assert_absorbs(self, w: np.ndarray) -> None:
        """
        Raise if the demography can never absorb, for example for an isolated deme or one-way migration in the final
        epoch. Mass of the propagated distribution ``w`` on states that cannot reach absorption in the final epoch
        (``state_space`` updated to it) is permanent. Called by the absorption-time search.

        :param w: State distribution at a large time in the final, unbounded epoch.
        :raises ValueError: if a non-negligible fraction of the mass can never reach a common ancestor.
        """
        _, reach = self._reaches_absorption()
        stuck = float(np.asarray(w)[~reach].sum())

        if stuck > 1e-8:
            raise ValueError(
                f"The demography does not absorb: a fraction {stuck:.2e} of the probability mass remains on "
                "states that can never reach a common ancestor, so there is no almost-sure absorption time. "
                "This typically means a deme is isolated or migration is one-way/blocked in the final "
                "(unbounded) epoch, leaving lineages that can never coalesce. Check the migration structure "
                "of the last epoch."
            )

    def _accumulate_closed_form(self, k: int, rewards: Sequence[Reward]) -> float:
        """
        Raw moment of a single reward ordering to absorption with the closed-form last epoch of
        ``PhaseTypeDistribution.moment``: the backward recursion with one LU of ``-T`` (``_lu_solver``), then the
        finite epochs applied backwards to the extended vector.

        :param k: The order of the moment.
        :param rewards: Sequence of k rewards (a single ordering).
        :return: The kth moment accumulated until absorption.
        """
        self._check_demography_conditioning()

        epochs = self._get_epochs_until_unbounded()
        n = self.state_space.k

        # --- final, unbounded epoch: limit vector z ---
        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, len(epochs) - 1)
        absorbing = self.state_space.absorbing

        # restrict to the states that can carry mass: a transient state outside the support contributes nothing and
        # makes ``-T`` singular when it cannot reach absorption. Mass leaves the support nowhere, the support being
        # a forward closure, so the restricted solve is exact on it.
        support = self._alpha_support()
        idx_t = np.where(~absorbing & support)[0]
        idx_a = np.where(absorbing)[0]
        e = np.asarray(self.state_space.e)

        # The closed form factors the transient sub-generator ``T`` (size = number of transient states), whose
        # dense-LU vs sparse-LU crossover sits at :attr:`closed_form_sparse_min_states` transient states. This is a different
        # quantity from the Van Loan dimension that governs the matrix-exponential path (:attr:`expm_action_min_dim`):
        # the LU only ever sees ``T``, independent of the moment order, so the threshold is on ``len(idx_t)`` alone.
        use_action = self._solve_sparse(len(idx_t))

        # transient sub-generator and its (sparse or dense) factorization, reused across the back-substitution
        T = self._transient_block(idx_t, sparse=use_action)
        if use_action:
            self._logger.debug(
                "closed form (k=%d): sparse LU (splu) of T (n_t=%d >= %d), %d finite epoch(s)",
                k, len(idx_t), Settings.closed_form_sparse_min_states, len(epochs) - 1
            )
        else:
            self._logger.debug(
                "closed form (k=%d): dense LU of T (n_t=%d), %d finite epoch(s)", k, len(idx_t), len(epochs) - 1
            )
        solve = self._lu_solver(-T, use_action)

        # the j-th block of the extended vector is stored divided by lamb ** (k - j), which balances the Van Loan
        # exponentials of the finite epochs, so the moment is multiplied by lamb ** k at the end
        lamb = self._get_regularization_factor(self.state_space.S)

        # reward diagonals restricted to the transient states (the off-diagonal Van Loan reward blocks are diagonal)
        r_t = [np.asarray(r._get(self.state_space), dtype=float)[idx_t] for r in rewards]

        nu = [None] * (k + 1)
        nu[k] = e[idx_t]
        for j in range(k - 1, -1, -1):
            nu[j] = solve(r_t[j] * nu[j + 1]) / lamb

        z = np.zeros((k + 1) * n)
        for j in range(k + 1):
            z[j * n + idx_t] = nu[j]
        z[k * n + idx_a] = e[idx_a]

        # --- preceding finite epochs, backward, via the (sparse or dense) full Van Loan matrix exponential ---
        for i_epoch, epoch in reversed(list(enumerate(epochs[:-1]))):
            self.state_space.update_epoch(epoch)

            # balance each epoch on its own rates. ``V * tau`` carries the epoch's generator on the diagonal and its
            # reward blocks divided by the factor, so a factor drawn from one epoch leaves the reward blocks of an
            # epoch with a different rate scale far from one, and the scaling-and-squaring of the exponential loses
            # their cancellation. Rebasing the stored vector is the exact diagonal similarity that permits it.
            lamb = self._rebase(z, lamb, self._balance(epoch, 0.0, np.inf), k, n)

            S = self.state_space.S * lamb
            self._check_numerical_stability(S, i_epoch)
            tau = (epoch.end_time - epoch.start_time) / lamb

            if use_action:
                r_vecs = [np.asarray(r._get(self.state_space), dtype=float) for r in rewards]
                S_csr = S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S))
                V = self._van_loan_matrix(r_vecs, S_csr, k, sparse=True)
                z = Backend.expm_multiply(V * tau, z)
            else:
                S_dense = np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)
                R = [r._get(self.state_space) for r in rewards]
                V = self._van_loan_matrix(R, S_dense, k)
                z = expm(V * tau) @ z

        alpha_ext = np.zeros((k + 1) * n)
        alpha_ext[:n] = self.state_space.alpha

        return factorial(k) * lamb ** k * float(alpha_ext @ z)

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

    def _mean_occupation_grid(self, end_times: Sequence[float], start_time: float = None) -> np.ndarray:
        """
        Expected time spent in each state of ``E`` over ``[start_time, t]`` for each end time ``t``, threaded across
        epochs with the augmented generator ``[[S, I], [0, 0]]`` of ``PhaseTypeDistribution.moment``, for the batched
        mean accumulation of a spectrum. A positive start time subtracts the occupation up to it.

        :param end_times: Times at which to evaluate the occupation.
        :param start_time: Time from which to accumulate. By default, the start time of the distribution.
        :return: Array of shape ``(len(end_times), n_states)``.
        """
        end_times = np.asarray(end_times, dtype=float)
        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        if start_time is None:
            start_time = self.tree_height.start_time

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
        """
        if not self._absorption_certain_in_last_epoch():
            return None

        epochs = self._get_epochs_until_unbounded()

        self.state_space.update_epoch(epochs[-1])
        absorbing = self.state_space.absorbing
        # only the states that can carry mass; see ``_alpha_support``
        idx_t = np.where(~absorbing & self._alpha_support())[0]
        nt = len(idx_t)
        use_action = self._solve_sparse(nt)

        p = np.asarray(self.state_space.alpha)[idx_t].astype(float)
        m = np.zeros(nt)

        self._logger.debug(
            "occupation times (batched mean%s): %s factorization, n_t=%d, %d finite epoch(s)",
            f", capped at {cap:.3g}" if cap is not None else "",
            "sparse" if use_action else "dense", nt, len(epochs) - 1
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

            upper = epoch.end_time if cap is None else min(epoch.end_time, cap)

            if np.isinf(upper):
                # final unbounded epoch, no cap: occupation to absorption = p (-T)^{-1}, i.e. solve (-T)^T x = p
                neg_t = -self._transient_block(idx_t, sparse=use_action)
                m += self._lu_solver(neg_t.T, use_action)(p)
                break

            S = self._transient_block(idx_t, sparse=use_action)
            tau = upper - epoch.start_time
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
                aug[:nt, :nt] = S
                aug[:nt, nt:] = np.eye(nt)
                exp_aug = expm(aug * tau)
                m += p @ exp_aug[:nt, nt:]
                p = p @ exp_aug[:nt, :nt]

            if cap is not None and upper >= cap:
                break

        return m, idx_t

    def _two_point_occupation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Dense two-point occupation matrix ``K = diag(m) (-T)^{-1}`` of ``PhaseTypeDistribution.moment``, defined for a
        single epoch without an accumulation window. Other cases return ``None`` and callers evaluate per pair.

        :return: ``(K, idx_t)`` over the transient states, or ``None`` when not applicable (caller falls back).
        """
        if not (Settings.closed_form_last_epoch and self.tree_height.end_time is None):
            return None

        if self.tree_height.start_time > 0:
            # the two-point occupation is a double integral over ``s < u``; the windowed (start_time > 0) version is
            # not the full one minus a box, so the batched covariance cannot subtract it the way the mean does. Fall
            # back to the per-pair path, which accumulates each pair over ``[start_time, absorption]`` directly.
            self._logger.debug(
                "two-point occupation: start_time=%.3g > 0; using per-pair covariance (windowed two-point occupation "
                "not batched)", self.tree_height.start_time
            )
            return None

        epochs = self._get_epochs_until_unbounded()

        # only the single-epoch closed form is used; the multi-epoch ODE is stiffness-fragile (see docstring)
        if len(epochs) > 1:
            self._logger.debug(
                "two-point occupation: %d epochs; using per-pair matrix-exponential (multi-epoch closed form "
                "disabled)", len(epochs)
            )
            return None

        if not self._absorption_certain_in_last_epoch():
            return None

        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, 0)
        absorbing = self.state_space.absorbing
        # only the states that can carry mass; see ``_alpha_support``
        idx_t = np.where(~absorbing & self._alpha_support())[0]

        neg_t_inv = sla.inv(-self._transient_block(idx_t))
        m = np.asarray(self.state_space.alpha)[idx_t].astype(float) @ neg_t_inv

        self._logger.debug("two-point occupation: single-epoch closed form diag(m)(-T)^-1 (n_t=%d)", len(idx_t))

        return np.diag(m) @ neg_t_inv, idx_t
