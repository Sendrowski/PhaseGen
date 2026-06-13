"""Phase-type distribution (moment engine) and the tree-height distribution."""

import itertools
import logging
from ..caching import cached_property, cache
from math import factorial
from typing import List, Tuple, Collection, Iterable, Optional, Sequence, Union
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
from ..coalescent_models import StandardCoalescent
from ..demography import Demography
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, CustomReward
from ..settings import Settings
from ..spectrum import SFS
from ..state_space import BlockCountingStateSpace, LineageCountingStateSpace, StateSpace

from ._common import _make_hashable
from .base import DensityAwareDistribution, MarginalDemeDistributions, MarginalLocusDistributions, MomentAwareDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')


class PhaseTypeDistribution(MomentAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process.
    """

    def __init__(
            self,
            state_space: StateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography = None,
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward. By default, the tree height reward.
        """
        if demography is None:
            demography = Demography()

        if reward is None:
            reward = TreeHeightReward()

        super().__init__()

        #: Population configuration
        self.lineage_config: LineageConfig = state_space.lineage_config

        #: Locus configuration
        self.locus_config: LocusConfig = state_space.locus_config

        #: Reward
        self.reward: Reward = reward

        #: State space
        self.state_space: StateSpace = state_space

        #: Demography
        self.demography: Demography = demography

        #: Tree height distribution
        self.tree_height: TreeHeightDistribution = tree_height

    @staticmethod
    def _get_van_loan_matrix(R: List[np.ndarray], S: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Get the block matrix for the given reward matrices and transition matrix.

        :param R: List of length k of reward matrices
        :param S: Intensity matrix
        :param k: The order of the moment.
        :return: Van Loan matrix which is a block matrix of size (k + 1) * (k + 1)
        """
        # matrix of zeros
        O = np.zeros_like(S)

        # create compound matrix
        return np.block([[S if i == j else R[i] if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])

    @staticmethod
    def _get_van_loan_matrix_sparse(R: List[np.ndarray], S: 'sp.spmatrix', k: int = 1) -> 'sp.spmatrix':
        """
        Sparse, block-bidiagonal Van Loan matrix: the (sparse) intensity matrix ``S`` on the diagonal and the
        (diagonal) reward matrices on the super-diagonal. Built directly as a sparse matrix to avoid materializing
        the dense ``(k + 1) * n`` block matrix.

        :param R: List of length k of reward vectors (the diagonals of the reward matrices).
        :param S: Sparse intensity matrix.
        :param k: The order of the moment.
        :return: Sparse Van Loan matrix of size ``(k + 1) * (k + 1)`` blocks.
        """
        blocks = [[None] * (k + 1) for _ in range(k + 1)]
        for i in range(k + 1):
            blocks[i][i] = S
            if i < k:
                blocks[i][i + 1] = sp.diags(R[i])

        return sp.bmat(blocks, format='csr')

    @cached_property
    def mean(self) -> float | SFS:
        """
        First moment / mean.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> float | SFS:
        """
        Second central moment / variance.
        """
        return self.moment(k=2, center=True)

    @cached_property
    def std(self) -> float | SFS:
        """
        Standard deviation.
        """
        return self.var ** 0.5

    @cached_property
    def m2(self) -> float | SFS:
        """
        Second (non-central) moment.
        """
        return self.moment(k=2, center=False)

    @cached_property
    def demes(self) -> MarginalDemeDistributions:
        """
        Marginal distributions over each deme.
        """
        return MarginalDemeDistributions(self)

    @cached_property
    def loci(self) -> MarginalLocusDistributions:
        """
        Marginal distributions over each locus.
        """
        return MarginalLocusDistributions(self)

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
        """
        Get the kth (non-central) (cross-)moment.

        :param k: The order of the moment.
        :param rewards: Iterable of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The kth moment
        """
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            # evaluate the moment to absorption: signal the closed-form path with an infinite end time when it
            # applies (no explicit end time, accumulation from 0, and absorption certain in the last epoch), but not
            # when flattening applies (which takes precedence and delegates to the smaller lineage-counting space),
            # otherwise use the estimated absorption time
            if (
                    Settings.closed_form_last_epoch and
                    not self._flattening_applies(k) and
                    start_time == 0 and
                    self.tree_height.end_time is None and
                    self._absorption_certain_in_last_epoch()
            ):
                end_time = np.inf
            else:
                end_time = self.tree_height.t_max

        if start_time > 0:
            m_start, m_end = PhaseTypeDistribution.accumulate(
                self,
                k=k,
                end_times=[start_time, end_time],
                rewards=rewards,
                center=center,
                permute=permute
            )

            m = float(m_end - m_start)
        else:
            m = float(PhaseTypeDistribution.accumulate(
                self,
                k=k,
                end_times=[end_time],
                rewards=rewards,
                center=center,
                permute=permute
            )[0])

        if np.isnan(m):
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return m

    @staticmethod
    def _get_regularization_factor(S: np.ndarray) -> float:
        """
        Get the regularization factor for the given intensity matrix. We
        multiply the intensity matrix by this factor to improve numerical
        stability when computing the matrix exponential of the Van Loan matrix.
        If regularization is disabled, this factor is 1.

        :param S: Intensity matrix.
        :return: Regularization factor.
        """
        if not Settings.regularize:
            return 1.0

        # obtain positive rates (for a sparse matrix, the positive stored entries)
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        # rewards in the Van Loan matrix are of order 1
        return 10 ** - np.log10(rates).mean()

    def _check_demography_conditioning(self):
        """
        Fail fast on extreme demographies whose population sizes or migration rates differ by more than ~double
        precision. Such demographies make the moment computation numerically unreliable, whether via the
        matrix-exponential absorption-time estimate (where scipy's ``expm`` one-norm power iteration becomes
        intermittently prohibitively slow) or the closed-form transient solve (where the rate matrix is
        ill-conditioned). Detected up front from the demography (not the rate matrix, whose range can also be
        widened by the coalescent model, e.g. multiple-merger models).

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

    def _check_numerical_stability(self, S: np.ndarray, epoch: int):
        """
        Warn about potential numerical instability with very small or very large rates.

        :param S: (Regularized) intensity matrix.
        :param epoch: Epoch number.
        """
        # positive (off-diagonal) rates; for a sparse matrix these are the positive stored entries
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        if rates.min() / rates.max() < 1e-10:
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
            permute: bool = True
    ) -> np.ndarray:
        """
        Evaluate the kth moment at different end times.

        :param k: The order of the moment.
        :param end_times: List of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The moment accumulated at the specified times or time.
        """
        k = int(k)

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
                PhaseTypeDistribution.accumulate(
                    self,
                    k=1,
                    rewards=(rewards[i],),
                    end_times=end_times
                ) for i in range(k)
            ]

            for i in range(k + 1):
                # iterate over all possible subsets of rewards of size i
                for indices in itertools.combinations(range(k), i):
                    # joint moment
                    mu_i = PhaseTypeDistribution.accumulate(
                        self,
                        k=i,
                        rewards=tuple(rewards[j] for j in indices),
                        end_times=end_times,
                        center=False,
                        permute=permute
                    )

                    # product of means of remaining rewards
                    mu1 = np.prod([means[j] for j in range(k) if j not in indices], axis=0)

                    components += [(-1) ** (k - i) * mu_i * mu1]

            return np.sum(components, axis=0)

        if permute:
            # get all possible permutations of rewards
            permutations = list(itertools.permutations(rewards))

            # compute average over all permutations
            return np.sum([self._accumulate(k, tuple(end_times), r) for r in permutations], axis=0) / len(permutations)

        return self._accumulate(k, tuple(end_times), rewards)

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
        :raises ValueError: If the state space is not a BlockCountingStateSpace, or if k is not 1, or if there are
            multiple populations or loci or if the coalescent model is not the standard coalescent.
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
        r = reward._get(self.state_space)

        probs = self.state_space._state_probs

        # sum up weights for each state based on the number of lineages
        n = self.lineage_config.n
        weights = np.zeros(n)
        for i, s in enumerate(self.state_space.states):
            weights[n - s.lineages.sum()] += probs[i] * r[i]

        # Create a custom reward that returns the weights.
        weighted_reward = CustomReward(lambda _: weights)

        self._logger.debug(
            "flattening block-counting state space (%d states) onto the lineage-counting state space (%d states)",
            len(self.state_space.states), self.tree_height.state_space.k
        )

        return self.tree_height._accumulate(k=k, end_times=end_times, rewards=(weighted_reward,))

    @_make_hashable
    @cache
    def _accumulate(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times.

        :param k: The order of the moment.
        :param end_times: Sequence of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :return: The moment accumulated at the specified times or time.
        """
        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        elif len(rewards) != k:
            raise ValueError(f"Number of rewards must be {k}.")

        end_times = np.array(end_times)

        # flattening takes precedence over the closed form (it shrinks the state space, which dominates the cost)
        if self._flattening_applies(k):
            self._logger.debug("accumulate (k=%d): flattened block-counting", k)
            return self._accumulate_flattened(k, end_times, rewards)

        # closed-form evaluation of the moment to absorption (signalled by an infinite end time): the final
        # unbounded epoch is solved directly instead of exponentiating over the estimated absorption time
        if Settings.closed_form_last_epoch and end_times.size == 1 and np.isinf(end_times.flat[0]):
            self._logger.debug("accumulate (k=%d): closed-form last epoch", k)
            return np.array([self._accumulate_closed_form(k, rewards)])

        # check for negative values
        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

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
        lamb = self._get_regularization_factor(self.state_space.S)

        # regularized intensity matrix
        S = self._dense_rate_matrix() * lamb

        # check numerical stability
        self._check_numerical_stability(S, 0)

        # get reward matrix
        R = [np.diag(r._get(state_space=self.state_space)) for r in rewards]

        # get Van Loan matrix
        V = self._get_van_loan_matrix(S=S, R=R, k=k)

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

                    # compute Van Loan matrix for next epoch using regularized intensity matrix
                    S = self._dense_rate_matrix() * lamb
                    self._check_numerical_stability(S, 0)
                    V = self._get_van_loan_matrix(S=S, R=R, k=k)

                # update with remaining time in current epoch
                Q @= expm(V * (u - u_prev) / lamb)

                alpha = self.state_space.alpha
                e = self.state_space.e
                moments[i] = factorial(k) * lamb ** k * alpha @ Q[:n_states, -n_states:] @ e

                u_prev = u

        # sort probabilities back to original order
        moments = moments[np.argsort(end_times)]

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
        Sparse-action variant of :meth:`_accumulate` for large state spaces. Instead of forming the dense Van Loan
        propagator ``Q = prod_i exp(V_i tau_i)`` and reading off ``alpha @ Q[:n, -n:] @ e``, this threads the vector
        ``w = alpha_ext`` through the epochs via the action of the matrix exponential on the (sparse) Van Loan
        matrix (``scipy.sparse.linalg.expm_multiply``), reading off ``w @ e_ext`` at each end time. This is exact
        (a product applied to a vector is a sequence of matrix-vector actions) and exploits the rate matrix sparsity.

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
        lamb = self._get_regularization_factor(self.state_space.S)

        def transposed_van_loan() -> 'sp.spmatrix':
            """Transposed sparse Van Loan matrix for the current epoch (transposed for the left vector action)."""
            S = self.state_space.S * lamb
            self._check_numerical_stability(S, i_epoch)
            r_vecs = [np.asarray(r._get(state_space=self.state_space), dtype=float) for r in rewards]
            return self._get_van_loan_matrix_sparse(R=r_vecs, S=sp.csr_matrix(S), k=k).T.tocsr()

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
                Vt = transposed_van_loan()

            # remaining time in the current epoch
            w = Backend.expm_multiply(Vt * ((u - u_prev) / lamb), w)
            moments[i] = factorial(k) * lamb ** k * float(w @ e_ext)
            u_prev = u

        moments = moments[np.argsort(end_times)]

        if np.isnan(moments).any():
            self._logger.warning(
                "NaN values encountered when computing moments via the matrix-exponential action. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def _get_epochs_until_unbounded(self) -> List['Epoch']:
        """
        Materialize the demographic epochs up to and including the final, unbounded epoch (``end_time == inf``).

        :return: List of epochs, the last of which is unbounded.
        """
        epochs = []
        for epoch in self.demography.epochs:
            epochs.append(epoch)
            if epoch.end_time == np.inf:
                break
        return epochs

    def _absorption_certain_in_last_epoch(self) -> bool:
        """
        Structural check on the final (unbounded) epoch: whether every transient state can reach an absorbing
        state, i.e. the transient sub-generator is non-singular and the moment-to-absorption can be evaluated in
        closed form. When this is ``False`` (e.g. disconnected demes or a migration barrier in the last epoch) the
        moment may still be finite if absorption occurs in earlier epochs, so callers fall back to the
        matrix-exponential path rather than relying on the closed form.

        :return: Whether absorption is certain from every transient state of the last epoch.
        """
        # the result depends only on the (fixed) last-epoch structure, so memoize it: the closed form queries this
        # once per moment, and an SFS/jSFS evaluates many bins, so recomputing the reachability each time dominated.
        if getattr(self, '_absorption_certain_cache', None) is not None:
            return self._absorption_certain_cache

        self.state_space.update_epoch(self._get_epochs_until_unbounded()[-1])
        absorbing, reach = self._reaches_absorption()

        self._absorption_certain_cache = bool(reach[~absorbing].all())
        return self._absorption_certain_cache

    def _reaches_absorption(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward reachability over the *current* epoch's rate graph: which states can reach an absorbing state.
        A state can reach absorption iff it is absorbing or has an outgoing edge (rate ``S[i, j] > 0``) to a state
        that can; this is propagated backwards with a sparse adjacency, so each pass is O(nnz). Used both to decide
        whether the closed form applies (:meth:`_absorption_certain_in_last_epoch`) and to guard against demographies
        that never absorb (:meth:`_get_absorption_time`).

        :return: ``(absorbing, reach)`` boolean masks over the states; ``reach`` includes the absorbing states.
        """
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])

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

    def _assert_absorbs(self, T: np.ndarray):
        """
        Raise if the demography can never absorb. Distinguishes a *structural* barrier (an isolated deme or a
        one-way/blocked migration in the final, unbounded epoch, leaving lineages that can never coalesce) from a
        merely slow or numerically imprecise computation. ``T`` is the transition matrix integrated to a large time,
        so ``alpha @ T`` is the occupation distribution there and its support is exactly the mass still in play; the
        final epoch's rate graph (``state_space`` is expected to be updated to it) tells us which states can
        structurally reach absorption. Residual mass parked on states that cannot is permanent. Shared by the
        absorption-time and quantile searches, both of which otherwise silently run to their iteration ceiling.

        :param T: Transition matrix integrated from time 0 to a large time in the final, unbounded epoch.
        :raises ValueError: if a non-negligible fraction of the mass can never reach a common ancestor.
        """
        _, reach = self._reaches_absorption()
        stuck = float((self.state_space.alpha @ T)[~reach].sum())

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
        Evaluate the kth (non-central) moment accumulated until absorption, evaluating the final unbounded epoch in
        closed form. The final epoch's contribution to ``t -> inf`` is the limit ``z = lim_t exp(V t) e_ext`` of the
        Van Loan propagator, whose transient part is the back-substitution ``nu_j = (-T)^{-1} R_j nu_{j+1}`` (with
        ``nu_k`` the exit vector) and whose absorbing part is the exit vector in the last block. The preceding finite
        epochs are applied to ``z`` via the (well-conditioned, finite-interval) matrix exponential of the full Van
        Loan matrix. This is for a single reward ordering; permutation averaging is handled by :meth:`accumulate`.

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
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]
        idx_a = np.where(absorbing)[0]
        e = np.asarray(self.state_space.e)

        # The closed form factors the transient sub-generator ``T`` (size = number of transient states), whose
        # dense-LU vs sparse-LU crossover sits at ``Settings.closed_form_sparse_min_states`` transient states. This is a different
        # quantity from the Van Loan dimension that governs the matrix-exponential path (:attr:`expm_action_min_dim`):
        # the LU only ever sees ``T``, independent of the moment order, so the threshold is on ``len(idx_t)`` alone.
        use_action = len(idx_t) >= Settings.closed_form_sparse_min_states

        # transient sub-generator and its (sparse or dense) factorization, reused across the back-substitution
        T = self._transient_block(idx_t, sparse=use_action)
        if use_action:
            self._logger.debug(
                "closed form (k=%d): sparse LU (splu) of T (n_t=%d >= %d), %d finite epoch(s)",
                k, len(idx_t), Settings.closed_form_sparse_min_states, len(epochs) - 1
            )
            solver = spla.splu(sp.csc_matrix(-T))
            solve = solver.solve
        else:
            self._logger.debug(
                "closed form (k=%d): dense LU of T (n_t=%d), %d finite epoch(s)", k, len(idx_t), len(epochs) - 1
            )
            lu = sla.lu_factor(-T)
            solve = lambda b: sla.lu_solve(lu, b)

        # reward diagonals restricted to the transient states (the off-diagonal Van Loan reward blocks are diagonal)
        r_t = [np.asarray(r._get(self.state_space), dtype=float)[idx_t] for r in rewards]

        nu = [None] * (k + 1)
        nu[k] = e[idx_t]
        for j in range(k - 1, -1, -1):
            nu[j] = solve(r_t[j] * nu[j + 1])

        z = np.zeros((k + 1) * n)
        for j in range(k + 1):
            z[j * n + idx_t] = nu[j]
        z[k * n + idx_a] = e[idx_a]

        # --- preceding finite epochs, backward, via the (sparse or dense) full Van Loan matrix exponential ---
        for i_epoch, epoch in reversed(list(enumerate(epochs[:-1]))):
            self.state_space.update_epoch(epoch)
            S = self.state_space.S
            self._check_numerical_stability(S, i_epoch)
            tau = epoch.end_time - epoch.start_time

            if use_action:
                r_vecs = [np.asarray(r._get(self.state_space), dtype=float) for r in rewards]
                S_csr = S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S))
                V = self._get_van_loan_matrix_sparse(R=r_vecs, S=S_csr, k=k)
                z = Backend.expm_multiply(V * tau, z)
            else:
                S_dense = np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)
                R = [np.diag(r._get(self.state_space)) for r in rewards]
                V = self._get_van_loan_matrix(S=S_dense, R=R, k=k)
                z = expm(V * tau) @ z

        alpha_ext = np.zeros((k + 1) * n)
        alpha_ext[:n] = self.state_space.alpha

        return factorial(k) * float(alpha_ext @ z)

    def _flattening_applies(self, k: int) -> bool:
        """
        Whether the block-counting state space can be flattened to the (much smaller) lineage-counting state space
        for this moment: the first moment of the standard coalescent on a single population and a single locus. When
        it applies it takes precedence over the closed form / batched occupation, because reducing the state space
        (e.g. thousands of block states to ``n`` lineage states) dominates the per-solve cost.
        """
        return (
                Settings.flatten_block_counting and
                k == 1 and
                isinstance(self.state_space, BlockCountingStateSpace) and
                isinstance(self.state_space.model, StandardCoalescent) and
                self.lineage_config.n_pops == 1 and
                self.locus_config.n == 1
        )

    def _transient_block(self, idx_t: np.ndarray, sparse: bool = False):
        """
        The transient sub-generator ``T = S[idx_t, idx_t]`` extracted from the (dense or sparse) rate matrix,
        returned as a dense array (default) or a sparse CSC matrix (``sparse=True``, for the large-state-space LU /
        exp-action paths, which never materialise the dense block).
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

    def _occupation_times(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Expected total time spent in each transient state until absorption. This is the bin-independent quantity
        shared by every bin of a *mean* spectrum: the mean of a reward ``r`` is simply ``occupation . r``, so a whole
        SFS / joint SFS mean is one contraction ``occupation @ R`` over the stacked bin rewards instead of a separate
        solve per bin. Finite epochs contribute ``p_i (exp(S_i tau_i) - I) S_i^{-1}`` (entered with distribution
        ``p_i``); the final unbounded epoch contributes ``p (-T)^{-1}``.

        :return: ``(occupation, idx_t)`` with the occupation times over the transient states ``idx_t`` of the final
            epoch, or ``None`` if absorption is not almost sure (callers then fall back to per-bin evaluation).
        """
        if not self._absorption_certain_in_last_epoch():
            return None

        epochs = self._get_epochs_until_unbounded()

        self.state_space.update_epoch(epochs[-1])
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]
        nt = len(idx_t)
        use_action = nt >= Settings.closed_form_sparse_min_states

        p = np.asarray(self.state_space.alpha)[idx_t].astype(float)
        m = np.zeros(nt)

        self._logger.debug(
            "occupation times (batched mean): %s factorization, n_t=%d, %d finite epoch(s)",
            "sparse" if use_action else "dense", nt, len(epochs) - 1
        )

        # finite epochs: accumulate the within-epoch occupation and propagate the entry distribution. The occupation
        # integral ``A = int_0^tau exp(S t) dt`` is read off the augmented (Van Loan) generator ``[[S, I], [0, 0]]``,
        # which is robust even when the finite-epoch block ``S`` is singular (e.g. a migration barrier), unlike
        # ``(exp(S tau) - I) S^-1``. Only the row-action ``[p, 0] exp(aug tau) = [p exp(S tau), p A]`` is needed (the
        # propagated entry distribution and the occupation increment ``p A`` at once), so for large state spaces apply
        # the sparse matrix-exponential action instead of forming the dense ``2 nt x 2 nt`` exponential.
        for epoch in epochs[:-1]:
            self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, 0)
            S = self._transient_block(idx_t, sparse=use_action)
            tau = epoch.end_time - epoch.start_time
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

        # final unbounded epoch: occupation = p (-T)^{-1}, i.e. solve (-T)^T x = p
        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, len(epochs) - 1)
        neg_t = -self._transient_block(idx_t, sparse=use_action)
        if use_action:
            m += spla.splu(sp.csc_matrix(neg_t.T)).solve(p)
        else:
            m += sla.solve(neg_t.T, p)

        return m, idx_t

    def _two_point_occupation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Two-point occupation matrix ``K_{a,b} = int_{s<u} P(X_s = a, X_u = b) ds du`` — the bin-independent quantity
        shared by every *pair* of a second-moment spectrum: the uncentered cross-moment of two rewards ``r, r'`` is
        ``r^T (K + K^T) r'``, so the whole 2-SFS covariance is a single contraction over the stacked bin rewards.

        Restricted to a **single (unbounded) epoch**, where it is the exact closed form ``K = diag(m) (-T)^{-1}``
        (``m = alpha (-T)^{-1}`` the occupation times) and needs no numerical integration. The multi-epoch version
        requires integrating the ``O(n_states^2)`` matrix ODE ``dJ/du = J S + diag(f(u))``, whose explicit
        integrator degenerates (very many tiny steps) on stiff demographies, so it is deliberately not used: the
        caller falls back to the per-pair matrix-exponential path instead.

        :return: ``(K, idx_t)`` over the transient states, or ``None`` when not applicable (caller falls back).
        """
        if not (Settings.closed_form_last_epoch and self.tree_height.end_time is None):
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
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]

        neg_t_inv = sla.inv(-self._transient_block(idx_t))
        m = np.asarray(self.state_space.alpha)[idx_t].astype(float) @ neg_t_inv

        self._logger.debug("two-point occupation: single-epoch closed form diag(m)(-T)^-1 (n_t=%d)", len(idx_t))

        return np.diag(m) @ neg_t_inv, idx_t

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate samples from the mean reward distribution by simulating trajectories.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
        """
        if rewards is None:
            rewards = [self.reward]

        n_rewards = len(rewards)
        samples = np.zeros((n_samples, n_rewards))
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        alpha = self.state_space.alpha
        states_visited = np.zeros_like(alpha)
        R = np.array([r._get(self.state_space) for r in rewards])

        # iterate over samples
        for i in tqdm(range(n_samples), disable=not Settings.use_pbar):
            mass = np.zeros(n_rewards)
            t = 0
            rate = 0
            state = np.random.choice(len(alpha), p=alpha)
            epochs = self.demography.epochs
            traj_probs = [] if record_visits else None

            try:
                # find first non-zero rate epoch
                while rate == 0:
                    epoch = next(epochs)
                    self.state_space.update_epoch(epoch)
                    rate = -self.state_space.S[state, state]

                # sample next time step
                dt = np.random.exponential(1 / rate)

                # iterate over transitions
                while True:

                    # iterate over epochs
                    while t + dt >= epoch.end_time:
                        # reward until epoch boundary
                        mass += R[:, state] * (epoch.end_time - t)
                        dt -= (epoch.end_time - t)
                        t = epoch.end_time

                        # advance epoch
                        epoch = next(epochs)
                        self.state_space.update_epoch(epoch)

                        new_rate = -self.state_space.S[state, state]
                        if new_rate == 0:
                            t = epoch.end_time
                            continue

                        # rescale remaining time
                        dt *= rate / new_rate
                        rate = new_rate

                    # step completes in current epoch
                    mass += R[:, state] * dt
                    t += dt

                    # sample next state
                    probs = self.state_space.S[state].copy()
                    probs[state] = 0
                    state = np.random.choice(len(probs), p=probs / rate)

                    states_visited[state] += 1

                    if absorbing[state]:
                        raise StopIteration

                    rate = -self.state_space.S[state, state]

                    # if rate is zero, we skip to the next epoch
                    if rate == 0:
                        t = epoch.end_time
                        continue

                    # sample next time step
                    dt = np.random.exponential(1 / rate)

            except StopIteration:
                pass

            samples[i] = mass

        # normalize states visited
        states_visited /= n_samples

        return (samples, states_visited) if record_visits else samples

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of (non-central) moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and the
            99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        k = int(k)

        from ..visualization import Visualization

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = f"Moment accumulation ({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})"

        y = self.accumulate(k, end_times, rewards, center, permute)

        Visualization.plot(
            ax=ax,
            x=end_times,
            y=y,
            xlabel='t',
            ylabel='moment',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process that allows the computation of the
    density function. This is currently only possible with default rewards.
    """
    #: Maximum number of epochs to consider when determining time to almost sure absorption.
    max_epochs: int = 10000

    #: Maximum number of time we double the end time when determining time to almost sure absorption.
    max_iter: int = 20

    #: Probability of almost sure absorption.
    p_absorption: float = 1 - 1e-15

    def __init__(
            self,
            state_space: LineageCountingStateSpace,
            demography: Demography = None,
            start_time: float = 0,
            end_time: float = None
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param demography: The demography.
        :param start_time: Time when to start accumulating moments.
        :param end_time: Time when to end accumulation of moments. By default, the time until almost sure absorption.
        """
        if start_time < 0:
            raise ValueError("Start time must be greater than or equal to 0.")

        if end_time is not None and end_time < 0:
            raise ValueError("End time must be greater than or equal to 0.")

        if end_time is not None and end_time < start_time:
            raise ValueError("End time must be greater than equal start time.")

        super().__init__(
            state_space=state_space,
            tree_height=self,
            demography=demography,
            reward=TreeHeightReward()
        )

        #: State space
        self.state_space: LineageCountingStateSpace = state_space

        #: Start time
        self.start_time: float = start_time

        #: End time
        self.end_time: float | None = end_time

    def cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not isinstance(self.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        # assume scalar if not array
        if not isinstance(t, Iterable):
            return self.cdf(np.array([t]))[0]

        # check for negative values
        if np.any(t < 0):
            raise ValueError("Negative values are not allowed.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(t).astype(float)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get the transition matrix for the first epoch
        self.state_space.update_epoch(epoch)

        # initialize transition matrix
        T = np.eye(self.state_space.k)
        u_prev = 0

        # initialize probabilities
        probs = np.zeros_like(t_sorted)

        # take reward vector as exit vector
        e = self.reward._get(self.state_space)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                self._check_numerical_stability(self.state_space.S, i_epoch)

                # update transition matrix with remaining time in current epoch
                T @= expm(self._dense_rate_matrix() * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

            self._check_numerical_stability(self.state_space.S, i_epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(self._dense_rate_matrix() * (u - u_prev))

            probs[i] = 1 - self.state_space.alpha @ T @ e

            u_prev = u

        # sort probabilities back to original order
        probs = probs[np.argsort(t)]

        if np.isnan(probs).any():
            self._logger.critical(
                "NaN values in CDF. This is likely due to an ill-conditioned rate matrix."
            )

        return probs

    def _update(
            self,
            u: float,
            u_prev: float,
            T: np.ndarray,
            epoch: 'Epoch'
    ) -> Tuple[float, np.ndarray, 'Epoch']:
        """
        Update transition matrix and time.

        :param u: Time to update to.
        :param u_prev: Previous time.
        :param T: Transition matrix.
        :param epoch: Current epoch.
        :return: Updated time, transition matrix, and epoch.
        """
        self.state_space.update_epoch(epoch)

        while u > epoch.end_time:

            # update transition matrix with remaining time in current epoch
            tau = epoch.end_time - u_prev
            T = T @ expm(self._dense_rate_matrix() * tau)
            u_prev = epoch.end_time

            # fetch and update for next epoch
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)
        else:
            # update transition matrix
            T = T @ expm(self._dense_rate_matrix() * (u - u_prev))

        return u, T, epoch

    @cached_property
    def _e(self) -> np.ndarray:
        """
        Exit vector.
        """
        return self.reward._get(self.state_space)

    def _cum(self, T: np.ndarray) -> float:
        """
        Get cumulative probability for given transition matrix.

        :param T: Transition matrix.
        :return: Cumulative probability.
        """
        return float(1 - self.state_space.alpha @ T @ self._e)

    @cache
    def quantile(
            self,
            q: float,
            expansion_factor: float = 2,
            precision: float = 1e-5,
            max_iter: int = 1000
    ):
        """
        Find the specified quantile of a CDF using an adaptive bisection method.

        :param q: The desired quantile (between 0 and 1).
        :param expansion_factor: Factor by which to multiply the upper bound that does not yet contain the quantile.
        :param precision: Precision for quantile, i.e. ``F(b) - F(a) < precision``.
        :param max_iter: Maximum number of iterations.
        :return: The quantile.
        """
        if q < 0 or q > 1:
            raise ValueError("Specified quantile must be between 0 and 1.")

        if expansion_factor <= 1:
            raise ValueError("Expansion factor must be greater than 1.")

        # finite upper bound for the search: the time of almost-sure absorption (any quantile q < 1 lies below it).
        # This also guards against a demography that never absorbs — ``_get_absorption_time`` raises in that case —
        # and keeps the expansion below from doubling ``b`` to an overflow-inducing ceiling. A user-supplied end
        # time bounds the (necessarily proper) distribution instead.
        b_max = self.end_time if self.end_time is not None else self._get_absorption_time()

        # initialize bounds
        a, b = 0, 1

        T_a = np.eye(self.state_space.k)
        epoch_a, epoch_b = self.demography.get_epoch(0), self.demography.get_epoch(0)
        b, T_b, epoch_b = self._update(min(b, b_max), a, T_a, epoch_b)

        i = 0

        # expand the upper bound until its CDF reaches q (bounded by the absorption time, so it always terminates)
        while self._cum(T_b) < q and b < b_max and i < max_iter:
            b, T_b, epoch_b = self._update(min(b * expansion_factor, b_max), b, T_b, epoch_b)

            i += 1

        # use bisection method within the determined bounds
        while self._cum(T_b) - self._cum(T_a) > precision and i < max_iter:
            m, T_m, epoch_m = self._update((a + b) / 2, a, T_a, epoch_a)

            if self._cum(T_m) < q:
                a, T_a, epoch_a = m, T_m, epoch_m
            else:
                b, T_b, epoch_b = m, T_m, epoch_m

            i += 1

        # warn if maximum number of iterations reached
        if i - 1 == max_iter:
            raise RuntimeError("Maximum number of iterations reached when determining quantile.")

        return (a + b) / 2

    def pdf(self, t: float | Sequence[float], dx: float = None) -> float | np.ndarray:
        """
        Density function. We use numerical differentiation of the CDF to calculate the density. This provides good
        results as the CDF is exact and continuous.

        :param t: Value or values to evaluate the density function at.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Density
        """
        if dx is None:
            dx = self.quantile(0.99) / 1e10

        if isinstance(t, Iterable):
            t = np.array(t)

        # determine (non-negative) evaluation points
        x1 = np.max([t - dx / 2, np.zeros_like(t)], axis=0)
        x2 = x1 + dx

        return (self.cdf(x2) - self.cdf(x1)) / dx

    @cached_property
    def t_max(self) -> float:
        """
        Time until which computations are performed. This is either the end time specified when initializing
        the distribution or the time until almost sure absorption.
        """
        if self.end_time is not None:
            return self.end_time

        t_abs = self._get_absorption_time()

        if t_abs < self.start_time:
            raise ValueError(
                f"Determined time of almost sure absorption ({t_abs:.1f}) "
                f"is smaller than start time ({self.start_time:.1f}). "
                "The start time may be too large or the demography not well-defined."
            )

        return t_abs

    def _get_absorption_time(self) -> float:
        """
        Get a time estimate for when we have reached absorption almost surely.
        We base this computation on the transition matrix rather than the moments, because here
        we have a good idea about how likely absorption is, and can warn the user if necessary.
        Stopping the computation when no more rewards are accumulated is not a good idea, as this
        can happen before almost sure absorption (exponential runaway growth, temporary isolation in different demes).
        """
        i = 0
        T = np.eye(self.state_space.k)
        epoch = self.demography.get_epoch(0)

        self._check_demography_conditioning()

        t = 2 ** int(np.log2(np.mean(list(epoch.pop_sizes.values()))))
        expansion_factor = 2

        t, T, epoch = self._update(t, 0, T, epoch)
        p = self._cum(T)

        # multiple time by expansion_factor until we reach p_absorption
        while p < self.p_absorption and i < self.max_iter:
            t, T, epoch = self._update(t * expansion_factor, t, T, epoch)
            p = self._cum(T)

            if np.isnan(p):
                self._logger.critical(
                    "Could not reliably find time of almost sure absorption "
                    "as probability of absorption is NaN. "
                    "This is likely due to an ill-conditioned rate matrix. "
                    f"Using time {t:.1f}. "
                )

            i += 1

        # if absorption was not reached, fail loudly for a demography that *never* absorbs rather than returning the
        # doubling ceiling (see :meth:`_assert_absorbs`).
        if p < self.p_absorption and not np.isnan(p):
            self._assert_absorbs(T)

        if i - 1 == self.max_iter:
            self._logger.warning(
                "Could not reliably find time of almost sure absorption after maximum number of iterations. "
                f"Using time {t:.1f} with probability of absorption 1 - {1 - p:.1e}. "
                "This could be due to numerical imprecision, unreachable states or very large or small "
                "absorption times. You can set the end time manually (see `Coalescent.end_time`) or increase "
                "the maximum number of iterations (`TreeHeightDistribution.max_iter`)."
            )

        return t

    def _empirical_cdf(self, n_samples: int, reward: Reward = None, t: float | Sequence[float] = None) -> np.ndarray:
        """
        Generate an empirical cumulative distribution function (CDF) by sampling from the distribution.

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :return: Sorted array of sampled total rewards.
        """
        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        samples = self._sample(n_samples, reward).reshape(n_samples)

        x = np.sort(samples)
        y = np.arange(1, n_samples + 1) / n_samples

        if x.ndim == 1:
            return np.interp(t, x, y)

    def _plot_empirical_cdf(
            self,
            n_samples: int = 1000,
            reward: Reward = None,
            t: float | Sequence[float] = None,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Empirical CDF'
    ) -> 'plt.Axes':
        """
        Plot the empirical cumulative distribution function (CDF).

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        y = self._empirical_cdf(n_samples, reward, t)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=y,
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

