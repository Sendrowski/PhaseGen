"""
The backward passes that give the exact derivatives of moments, as described in
:meth:`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>`: that of the
occupation-time recursion for first moments, and that of the closed form of higher cross-moments.
"""

import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _exact_1_norm, _exact_inf_norm, _fragment_3_1

from ..demography import Epoch
from ..expm import Backend, _ACTION_STEP_NORM, _GrowthBound

if TYPE_CHECKING:
    from ..coalescent_models import CoalescentModel
    from ..locus import LocusConfig
    from ..state_space import StateSpace
    from .phase_type import DemographyGradient, PhaseTypeDistribution

#: Unit roundoff targeted by the truncated Taylor action, as in :func:`scipy.sparse.linalg.expm_multiply`.
_TOL = 2.0 ** -53

#: Number of Taylor terms whose outer products are contracted in one batch of the reverse pass.
_CHUNK = 64

#: Relative tolerance of the reconstruction of a generator from the basis of :class:`_GeneratorBasis`.
_BASIS_RTOL = 1e-10

#: Largest step of the differences of the generator over a parameter of the coalescent model it is not affine in,
#: relative to the magnitude of the parameter (at least 1).
_PARAMETER_STEP = 1e-6

#: Step of the exact difference of the generator over a parameter it is affine in, relative to the magnitude of the
#: parameter (at least 1).
_AFFINE_STEP = 1.0

#: Smallest relative step tried, below which a parameter counts as not variable.
_MIN_PARAMETER_STEP = 1e-14

#: Number of steps a parameter must be able to move towards each side of a difference.
_STEP_MARGIN = 1000

#: Bytes of the contractions of the generator derivatives kept per state space for the epochs, by their rates, most
#: recently used, beyond those of the gradient being computed.
_CONTRACTION_BYTES = 256 * 2 ** 20



def _rate_parameters(model: 'CoalescentModel', locus_config: 'LocusConfig') -> Dict[str, float]:
    """
    The parameters the generator depends on besides the demography: those of the coalescent model, such as ``alpha``
    of a Beta coalescent, and the recombination rate of two loci.

    :param model: The coalescent model.
    :param locus_config: The locus configuration.
    :return: The parameters by name.
    """
    return model._parameters() | ({'recombination_rate': float(locus_config.recombination_rate)}
                                  if locus_config.n == 2 else {})


class _TapedAction:
    r"""
    The action :math:`\mathbf{y} = e^{\mathbf{M}} \mathbf{x}` of a sparse square matrix :math:`\mathbf{M}` on a vector
    :math:`\mathbf{x}`, by the truncated Taylor algorithm of Al-Mohy and Higham (2011, Algorithm 3.2), with the degree,
    the number of steps and the diagonal shift :math:`\mu` chosen as by :func:`scipy.sparse.linalg.expm_multiply`,
    recording the input of every step for the reverse pass
    :meth:`_TapedAction.reverse() <phasegen.distributions._gradient._TapedAction.reverse>`, which recomputes the Taylor
    terms one step at a time. Above a 1-norm of ``_ACTION_STEP_NORM`` the steps end once the vector is stationary over
    the remaining ones, tested as in :meth:`Backend.expm_multiply() <phasegen.expm.Backend.expm_multiply>`.
    """

    def __init__(self, M: sp.csr_matrix, blocks: int = 1) -> None:
        """
        Choose the degree and the number of steps of the action.

        :param M: The sparse matrix to exponentiate.
        :param blocks: The number of equal diagonal blocks of ``M`` outside of which it is strictly block triangular,
            see :meth:`Backend.expm_multiply() <phasegen.expm.Backend.expm_multiply>`.
        """
        n = M.shape[0]

        #: Diagonal shift.
        self.mu: float = float(M.diagonal().sum()) / n

        #: Shifted matrix :math:`\mathbf{M} - \mu \mathbf{I}`, whose sparsity pattern carries the gradient.
        self.B: sp.csr_matrix = (M - self.mu * sp.identity(n, format='csr')).tocsr()

        norm = _exact_1_norm(self.B)
        if norm == 0:
            self.degree, self.steps = 0, 1
        else:
            self.degree, self.steps = _fragment_3_1(LazyOperatorNormInfo(self.B, A_1_norm=norm, ell=2), 1, _TOL, ell=2)

        self._Bt = self.B.T.tocsr()

        #: The matrix, its entrywise absolute value, the bound on the growth of its exponential and the steps after
        #: which :meth:`_TapedAction.apply() <phasegen.distributions._gradient._TapedAction.apply>` tests stationarity,
        #: at the fractions of :meth:`Backend._step_fractions() <phasegen.expm.Backend._step_fractions>` when the norm
        #: of the matrix exceeds ``_ACTION_STEP_NORM``.
        self._M, self._abs_M, self._growth, self._checks = M, None, None, set()
        norm = _exact_1_norm(M)
        if np.isfinite(norm) and norm > _ACTION_STEP_NORM:
            self._abs_M = abs(M)
            self._growth = _GrowthBound(M, self._abs_M, blocks)
            self._checks = {int(np.ceil(f * self.steps)) for f in Backend._step_fractions(norm)} - {self.steps}

    def apply(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, int]]]:
        """
        Apply the action.

        :param x: The vector.
        :return: :math:`e^{\\mathbf{M}} \\mathbf{x}`, and the input and number of Taylor terms of every step, from which
            :meth:`_TapedAction.reverse() <phasegen.distributions._gradient._TapedAction.reverse>` recomputes the terms
            one step at a time.
        """
        eta = np.exp(self.mu / self.steps)
        checkpoints, F = [], x
        for i in range(1, self.steps + 1):
            terms = self._terms(F)
            checkpoints.append((F, len(terms)))
            F = eta * np.sum(terms, axis=0)
            f = (self.steps - i) / self.steps
            if i in self._checks and Backend._stationary(self._M, self._abs_M, F, f, self._growth(f)):
                break

        return F, checkpoints

    def _terms(self, b: np.ndarray, n_terms: int = None) -> List[np.ndarray]:
        """
        The Taylor terms of one step, truncated as by :func:`scipy.sparse.linalg.expm_multiply` or after ``n_terms``.

        :param b: The input of the step.
        :param n_terms: The number of terms, by default the truncation criterion decides.
        :return: The terms, the input first.
        """
        terms, F = [b], b
        c1 = _exact_inf_norm(b)
        for j in range(self.degree if n_terms is None else n_terms - 1):
            b = self.B.dot(b) / (self.steps * (j + 1))
            terms.append(b)
            if n_terms is None:
                c2 = _exact_inf_norm(b)
                F = F + b
                if c1 + c2 <= _TOL * _exact_inf_norm(F):
                    break
                c1 = c2

        return terms

    def reverse(
            self,
            lam: np.ndarray,
            checkpoints: List[Tuple[np.ndarray, int]],
            rows: np.ndarray,
            cols: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reverse pass of :meth:`_TapedAction.apply() <phasegen.distributions._gradient._TapedAction.apply>` for a scalar
        :math:`L` with :math:`\\partial L / \\partial \\mathbf{y} = \\lambda`, holding the shift, the degree and the
        number of steps fixed.

        :param lam: The adjoint of the result.
        :param checkpoints: The step inputs and term counts recorded by
            :meth:`_TapedAction.apply() <phasegen.distributions._gradient._TapedAction.apply>`.
        :param rows: The rows of the entries of :math:`\\mathbf{M}` to differentiate by.
        :param cols: Their columns.
        :return: :math:`\\partial L / \\partial \\mathbf{x}`, and :math:`\\partial L / \\partial M_{rc}` over the entries
            ``(rows, cols)``.
        """
        eta = np.exp(self.mu / self.steps)
        grad = np.zeros(len(rows))
        for b, n_terms in reversed(checkpoints):
            terms = self._terms(b, n_terms)
            g = eta * lam
            adjoint = g
            left, right = [], []
            for j in range(len(terms) - 2, -1, -1):
                c = 1.0 / (self.steps * (j + 1))
                left.append(c * adjoint)
                right.append(terms[j])
                adjoint = g + c * self._Bt.dot(adjoint)
            lam = adjoint

            for k in range(0, len(left), _CHUNK):
                L = np.stack(left[k:k + _CHUNK], axis=1)
                R = np.stack(right[k:k + _CHUNK], axis=1)
                grad += np.einsum('ik,ik->i', L[rows], R[cols])

        return lam, grad


class _OccupationStep:
    """
    One epoch of the occupation-time recursion of ``MomentEvaluator._occupation_times``, as recorded for
    ``MeanGradient``.
    """

    def __init__(
            self,
            epoch: Epoch,
            idx_t: np.ndarray,
            tau: float,
            T: 'np.ndarray | sp.spmatrix',
            w: np.ndarray,
            exp_aug: np.ndarray = None,
            action: _TapedAction = None
    ) -> None:
        """
        Record an epoch.

        :param epoch: The epoch.
        :param idx_t: The transient states carried.
        :param tau: The duration, infinite for the last epoch.
        :param T: The transient block of the generator.
        :param w: The state distribution and the accumulated occupation at the start of the epoch.
        :param exp_aug: The dense exponential of the augmented generator, on the dense path.
        :param action: The action of the transposed augmented generator, on the sparse path.
        """
        self.epoch, self.idx_t, self.tau, self.T, self.w = epoch, idx_t, tau, T, w
        self.exp_aug, self.action = exp_aug, action

        #: Step inputs and term counts of the action.
        self.checkpoints: Optional[List[Tuple[np.ndarray, int]]] = None

        #: The solver of ``(-T)^T`` of the last epoch, whose attribute ``T`` solves ``-T``.
        self.solve: Optional[Callable] = None

    def augmented(self) -> sp.csr_matrix:
        """
        The augmented generator :math:`[[\\mathbf{T}, \\mathbf{I}], [\\mathbf{0}, \\mathbf{0}]]`, unscaled.
        """
        n = len(self.idx_t)
        return sp.bmat([[sp.csr_matrix(self.T), sp.identity(n)], [None, sp.csr_matrix((n, n))]], format='csr')


class _VanLoanStep:
    """
    One finite epoch of ``MomentEvaluator._accumulate_closed_form``, as recorded for ``CrossMomentGradient``. The
    extended column vector entering the epoch is multiplied entrywise by ``rebase`` into ``z_in`` and then mapped by
    the exponential of the Van Loan matrix ``A`` into ``z_out``.
    """

    def __init__(self, index: int, epoch: Epoch, duration: float, rebase: np.ndarray, z_in: np.ndarray,
                 A: 'np.ndarray | sp.spmatrix') -> None:
        """
        Record an epoch.

        :param index: The position of the epoch in the epochs of the moment.
        :param epoch: The epoch.
        :param duration: Its duration, by which the generator blocks of ``A`` are scaled.
        :param rebase: The rebasing onto the balancing factor of the epoch, as a vector of factors.
        :param z_in: The rebased extended vector.
        :param A: The Van Loan matrix times its step, linear in the duration.
        """
        self.index, self.epoch, self.duration, self.rebase, self.z_in, self.A = index, epoch, duration, rebase, z_in, A

        #: The dense exponential of ``A``, on the dense path.
        self.exp: Optional[np.ndarray] = None

        #: The action of ``A``, on the sparse path, and its step inputs and term counts.
        self.action: Optional[_TapedAction] = None
        self.checkpoints: Optional[List[Tuple[np.ndarray, int]]] = None

        #: The extended vector leaving the epoch.
        self.z_out: Optional[np.ndarray] = None


class _ClosedFormTape:
    """
    The record of ``MomentEvaluator._accumulate_closed_form`` for one ordering of the rewards from time zero, which
    ``CrossMomentGradient`` differentiates.
    """

    def __init__(self) -> None:
        """
        Start an empty record.
        """
        #: The epochs of the moment, the last of which lasts until absorption.
        self.epochs: List[Epoch] = []

        #: The transient states of the last epoch that can carry mass, and the solver of ``-T`` on them, whose
        #: attribute ``T`` solves the transposed system.
        self.idx_t: Optional[np.ndarray] = None
        self.solve: Optional[Callable] = None

        #: The balancing factor of the last epoch.
        self.lamb: float = 1.0

        #: The reward vectors over all states, in order.
        self.rewards: List[np.ndarray] = []

        #: The blocks ``0, ..., k - 1`` of the extended vector of the last epoch on its transient states.
        self.blocks: List[np.ndarray] = []

        #: The finite epochs, latest first, as they are applied.
        self.steps: List[_VanLoanStep] = []

        #: The extended initial distribution, the extended vector at time zero, the balancing factor its blocks are
        #: stored against and the factor of the moment, which is ``factor * alpha_ext @ z``.
        self.alpha_ext: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.lamb_end: float = 1.0
        self.factor: float = 1.0


class _GeneratorBasis:
    r"""
    The generator as a linear function of the inverse coalescent time scales and the migration rates,

    .. math::
        \mathbf{S} = \mathbf{K} + \sum_d \frac{\mathbf{C}_d}{s(N_d)} + \sum_{a \neq b} m_{ab} \mathbf{M}_{ab},

    where :math:`s(N)` is the time scale of the coalescent model at population size :math:`N`, :math:`\mathbf{C}_d`
    the coalescence part of deme :math:`d` at unit inverse time scale, :math:`\mathbf{M}_{ab}` the migration part
    from deme :math:`a` to deme :math:`b` at unit rate and :math:`\mathbf{K}` the part that depends on neither, such
    as recombination.
    """

    def __init__(self, state_space: 'StateSpace') -> None:
        """
        Build the basis from generators of the state space at unit population sizes.

        :param state_space: The state space, returned to its epoch afterwards.
        """
        self.state_space = state_space
        self.pops: List[str] = list(state_space.lineage_config.pop_names)
        self.pairs: List[Tuple[str, str]] = [(a, b) for a in self.pops for b in self.pops if a != b]

        epoch = state_space.epoch
        model = state_space.model

        def generator(sizes: Dict[str, float], rates: Dict[Tuple[str, str], float]) -> sp.csr_matrix:
            state_space.update_epoch(Epoch(pop_sizes=sizes, migration_rates=rates))
            return sp.csr_matrix(state_space.S, dtype=float, copy=True)

        try:
            unit = {p: 1.0 for p in self.pops}
            S0 = generator(unit, {})
            c1, c2 = 1 / model._get_timescale(1.0), 1 / model._get_timescale(2.0)

            #: Coalescence part of each deme at unit inverse time scale.
            self.C: Dict[str, sp.csr_matrix] = {
                p: ((S0 - generator(unit | {p: 2.0}, {})) / (c1 - c2)).tocsr() for p in self.pops
            }

            #: Migration part of each ordered pair at unit rate.
            self.M: Dict[Tuple[str, str], sp.csr_matrix] = {
                pair: (generator(unit, {pair: 1.0}) - S0).tocsr() for pair in self.pairs
            }

            #: Part depending on neither the population sizes nor the migration rates.
            self.K: sp.csr_matrix = (S0 - sum((C * c1 for C in self.C.values()), sp.csr_matrix(S0.shape))).tocsr()

            # the generator is affine in the inverse time scales and migration rates for every coalescent model, which
            # an epoch away from the unit values checks
            sizes = {p: 0.7 + 0.37 * i for i, p in enumerate(self.pops)}
            rates = {pair: 0.3 + 0.21 * j for j, pair in enumerate(self.pairs)}
            self._check(sizes, rates, generator(sizes, rates))
        finally:
            state_space.update_epoch(epoch)

        self._transient: Dict[bytes, Tuple[Dict[str, sp.csr_matrix], Dict[Tuple[str, str], sp.csr_matrix],
                                           np.ndarray, np.ndarray]] = {}

        #: Finite-difference stencils of the generator over the parameters of :func:`_rate_parameters`, as bases with
        #: weights, by parameter, ``None`` for a parameter that cannot be varied.
        self._stencils: Dict[str, Optional[List[Tuple['_GeneratorBasis', float]]]] = {}

        #: The contractions of :meth:`contraction`, by set of states.
        self._contractions: Dict[bytes, '_Contraction'] = {}

        #: The bytes of the epochs cached by the contractions, by contraction and rates, least recently used first.
        self._cached: 'OrderedDict[Tuple[_Contraction, tuple], int]' = OrderedDict()
        self._bytes = 0

    def contraction(self, idx: np.ndarray) -> '_Contraction':
        """
        The contraction of the basis over a set of states, memoized.

        :param idx: The states.
        :return: The contraction.
        """
        key = np.asarray(idx).tobytes()
        if key not in self._contractions:
            self._contractions[key] = _Contraction(self, idx)

        return self._contractions[key]

    def touch(self, contraction: '_Contraction', key: tuple, keep: int) -> None:
        """
        Mark an epoch cached by a contraction as most recently used, evicting the least recently used epochs of all
        contractions beyond :data:`_CONTRACTION_BYTES`.

        :param contraction: The contraction.
        :param key: The rates of the epoch.
        :param keep: The number of most recently used epochs kept regardless, those of the gradient being computed.
        """
        if (contraction, key) not in self._cached:
            self._cached[contraction, key] = contraction.size(key)
            self._bytes += self._cached[contraction, key]
        self._cached.move_to_end((contraction, key))

        while self._bytes > _CONTRACTION_BYTES and len(self._cached) > keep:
            (oldest, old), size = self._cached.popitem(last=False)
            del oldest._epochs[old]
            self._bytes -= size

    def _generator(self, sizes: Dict[str, float], rates: Dict[Tuple[str, str], float]) -> sp.csr_matrix:
        """
        The generator rebuilt from the basis.

        :param sizes: The population sizes by deme.
        :param rates: The migration rates by ordered pair, zero where absent.
        :return: The generator.
        """
        model = self.state_space.model
        S = self.K + sum((self.C[p] / model._get_timescale(sizes[p]) for p in self.pops), sp.csr_matrix(self.K.shape))
        for pair in self.pairs:
            S = S + rates.get(pair, 0.0) * self.M[pair]

        return S.tocsr()

    def _varied(self, name: str, value: float) -> '_GeneratorBasis':
        """
        The basis of the state space with one parameter of :func:`_rate_parameters` changed.

        :param name: The parameter.
        :param value: Its value.
        :return: The basis.
        :raises ValueError: If the value is invalid.
        """
        ss = self.state_space
        if name == 'recombination_rate':
            if value < 0:
                raise ValueError('The recombination rate must not be negative.')
            locus_config = copy.copy(ss.locus_config)
            locus_config.recombination_rate = value
            return _GeneratorBasis(ss._with(locus_config=locus_config))

        return _GeneratorBasis(ss._with(model=ss.model._with(**{name: value})))

    def _valid(self, name: str, value: float) -> bool:
        """
        Whether a parameter of :func:`_rate_parameters` may take a value.

        :param name: The parameter.
        :param value: The value.
        :return: Whether it may.
        """
        if name == 'recombination_rate':
            return value >= 0

        try:
            self.state_space.model._with(**{name: value})
        except (ValueError, ZeroDivisionError, TypeError):
            return False

        return True

    def _stencil(self, name: str) -> Optional[List[Tuple['_GeneratorBasis', float]]]:
        """
        The bases and weights of a difference of the generator over a parameter, memoized. A parameter the generator
        is affine in (the recombination rate and the parameters in ``_affine_parameters`` of the model) takes an exact
        two-point difference with a large step towards valid values. Any other takes a second-order central
        difference where it may vary by a thousand steps either way and a one-sided one towards valid values otherwise.

        :param name: The parameter.
        :return: The bases with their weights, or ``None`` if the parameter cannot be varied.
        """
        if name not in self._stencils:
            model = self.state_space.model
            x = _rate_parameters(model, self.state_space.locus_config)[name]
            scale = max(abs(x), 1.0)
            stencil = None

            if name == 'recombination_rate' or name in model._affine_parameters:
                for h in (_AFFINE_STEP * scale, -_AFFINE_STEP * scale):
                    if stencil is None and self._valid(name, x + h):
                        stencil = [(self._varied(name, x + h), 1 / h), (self, -1 / h)]
            else:
                h = _PARAMETER_STEP * scale
                while stencil is None and h > _MIN_PARAMETER_STEP * scale:
                    if self._valid(name, x - _STEP_MARGIN * h) and self._valid(name, x + _STEP_MARGIN * h):
                        stencil = [(self._varied(name, x + h), 1 / (2 * h)), (self._varied(name, x - h), -1 / (2 * h))]
                    h /= 10

                for sign in (1, -1):
                    h = sign * _PARAMETER_STEP * scale
                    while stencil is None and abs(h) > _MIN_PARAMETER_STEP * scale:
                        if self._valid(name, x + 2 * _STEP_MARGIN * h):
                            stencil = [(self, -3 / (2 * h)), (self._varied(name, x + h), 4 / (2 * h)),
                                       (self._varied(name, x + 2 * h), -1 / (2 * h))]
                        h /= 10

            self._stencils[name] = stencil

        return self._stencils[name]

    @property
    def parameter_names(self) -> List[str]:
        """
        The parameters of :func:`_rate_parameters` the generator is differentiated by, those that can be varied.
        """
        return [name for name in _rate_parameters(self.state_space.model, self.state_space.locus_config)
                if self._stencil(name) is not None]

    def parameter_derivatives(self, epoch: Epoch, idx_t: np.ndarray) -> Dict[str, sp.csr_matrix]:
        r"""
        The derivatives :math:`\partial \mathbf{T} / \partial \theta` of the transient block of the generator of an
        epoch by the parameters :math:`\theta` of :func:`_rate_parameters`, by differences of the bases, which are
        exact for the parameters the generator is affine in.

        :param epoch: The epoch.
        :param idx_t: The transient states.
        :return: The derivatives by parameter, omitting a parameter that cannot be varied.
        """
        out = {}
        for name in _rate_parameters(self.state_space.model, self.state_space.locus_config):
            stencil = self._stencil(name)
            if stencil is not None:
                dS = sum((w * basis._generator(epoch.pop_sizes, epoch.migration_rates) for basis, w in stencil),
                         sp.csr_matrix(self.K.shape))
                out[name] = dS.tocsr()[idx_t][:, idx_t].tocsr()

        return out

    def transient(self, idx_t: np.ndarray) -> Tuple[
        Dict[str, sp.csr_matrix], Dict[Tuple[str, str], sp.csr_matrix], np.ndarray, np.ndarray]:
        """
        The coalescence and migration parts restricted to transient states, and the union of their sparsity patterns
        with those of the parameter stencils, memoized per set of states.

        :param idx_t: The transient states.
        :return: The coalescence parts by deme, the migration parts by ordered pair, and the rows and columns of the
            pattern.
        """
        key = np.asarray(idx_t).tobytes()
        if key not in self._transient:
            C = {p: self.C[p][idx_t][:, idx_t].tocsr() for p in self.pops}
            M = {pair: self.M[pair][idx_t][:, idx_t].tocsr() for pair in self.pairs}
            parts = [self.K] + list(self.C.values()) + list(self.M.values())
            for name in _rate_parameters(self.state_space.model, self.state_space.locus_config):
                for basis, _ in self._stencil(name) or []:
                    if basis is not self:
                        parts += [basis.K] + list(basis.C.values()) + list(basis.M.values())
            union = sum((abs(X) for X in parts), sp.csr_matrix(self.K.shape))[idx_t][:, idx_t]
            coo = sp.coo_matrix(union)
            self._transient[key] = (C, M, coo.row, coo.col)

        return self._transient[key]

    def coefficients(self, epoch: Epoch) -> Dict[str, float]:
        r"""
        The factors :math:`-s'(N_d) / s(N_d)^2` of the derivatives :math:`\partial \mathbf{S} / \partial N_d =
        -\mathbf{C}_d\, s'(N_d) / s(N_d)^2` in an epoch, where the derivative :math:`s'` of the time scale is obtained
        by a complex step. The derivatives by the migration rates are :math:`\partial \mathbf{S} / \partial m_{ab} =
        \mathbf{M}_{ab}`.

        :param epoch: The epoch.
        :return: The factors by deme.
        """
        model = self.state_space.model
        out = {}
        for p in self.pops:
            N = epoch.pop_sizes[p]
            h = 1e-30 * max(N, 1.0)
            out[p] = -(np.imag(model._get_timescale(N + 1j * h)) / h) / model._get_timescale(N) ** 2

        return out

    def _check(self, sizes: Dict[str, float], rates: Dict[Tuple[str, str], float], S: sp.csr_matrix) -> None:
        """
        Raise if the basis does not reproduce a generator.

        :param sizes: The population sizes of the generator.
        :param rates: Its migration rates.
        :param S: The generator.
        :raises ValueError: If the generator is not reproduced.
        """
        diff = abs(self._generator(sizes, rates) - S).max()
        if diff > _BASIS_RTOL * max(abs(S).max(), 1e-300):
            raise ValueError(
                f"The generator is not affine in the inverse time scales and migration rates (deviation {diff:.3g})."
            )


class MeanGradient:
    """
    Values and derivatives of the first moments of rewards, by a forward pass through the occupation-time recursion
    that records each epoch and a backward pass per reward, as described in :meth:`PhaseTypeDistribution.gradient()
    <phasegen.distributions.PhaseTypeDistribution.gradient>`.
    """

    def __init__(self, distribution: 'PhaseTypeDistribution') -> None:
        """
        Configure the gradient.

        :param distribution: The distribution whose state space and demography define the occupation times.
        """
        self.distribution = distribution

    def compute_columns(self, rewards: np.ndarray) -> Optional[List[Tuple[float, 'DemographyGradient']]]:
        """
        The means and gradients of several rewards, sharing the forward pass, the factorization of the last epoch and
        its occupation times.

        :param rewards: The reward vectors over the states of the state space, one per column.
        :return: The mean and gradient of each column, or ``None`` where the occupation times are not defined in closed
            form.
        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        from .phase_type import DemographyGradient

        dist = self.distribution
        tape: List[_OccupationStep] = []
        occupation = dist._occupation_times(tape=tape)
        if occupation is None:
            return None

        m, idx_t = occupation
        nt = len(idx_t)
        basis = _generator_basis(dist.state_space)
        con = basis.contraction(idx_t)
        names = basis.parameter_names

        # last epoch: d(p (-T)^{-1} r) = q dT u with q = p (-T)^{-1} and u = (-T)^{-1} r
        last = tape[-1]
        q = last.solve(last.w[:nt])

        # the augmented generator [[T, I], [0, 0]] applied to a vector, and its dense form scaled by the step
        def augmented(step: _OccupationStep, v: np.ndarray) -> np.ndarray:
            return np.concatenate([step.T @ v[:nt] + v[nt:], np.zeros(nt)])

        dense = {e: step.augmented().toarray() * step.tau for e, step in enumerate(tape[:-1]) if step.action is None}

        out = []
        for r in np.asarray(rewards, dtype=float)[idx_t].T:
            grad = DemographyGradient([step.epoch for step in tape], basis.pops, basis.pairs, parameters=names)

            u = last.solve.T(r)
            con.add(grad, len(tape) - 1, q[con.rows] * u[con.cols], 1.0)

            v = np.concatenate([u, r])

            # the last augmented generator annihilates its adjoint vector on the transient states
            V_next_v = np.concatenate([last.T @ u + r, np.zeros(nt)])

            # the generator of a finite epoch enters its exponent multiplied by its duration
            for e in range(len(tape) - 2, -1, -1):
                step = tape[e]
                w_next = tape[e + 1].w
                grad.start_times[e + 1] += float(w_next @ augmented(step, v) - w_next @ V_next_v)

                if step.action is None:
                    G = sla.expm_frechet(dense[e].T, np.outer(step.w, v), compute_expm=False)[:nt, :nt]
                    con.add(grad, e, G[con.rows, con.cols], step.tau)
                    v = step.exp_aug @ v
                else:
                    # the action applies exp(A^T) to [p, 0], so dL/dA_{rc} = dL/dM_{cr}, taken over the entries every
                    # generator of the space can take, also those at a rate of zero in this epoch
                    v_in, gM = step.action.reverse(v, step.checkpoints, con.cols, con.rows)
                    con.add(grad, e, gM, step.tau)
                    # the action is applied to [p, 0], and the accumulated occupation is carried past it
                    v = np.concatenate([v_in[:nt], v[nt:]])

                V_next_v = augmented(step, v)

            grad.parameters = {name: float(value) for name, value in grad.parameters.items()}
            out.append((float(m @ r), grad))

        return out


class _Contraction:
    """
    The contraction of a derivative by the entries of a generator with the derivatives of the generator by the
    population sizes, the migration rates and the parameters of the coalescent model, over the entries of a set of
    states that the generators can take.
    """

    def __init__(self, basis: _GeneratorBasis, idx: np.ndarray) -> None:
        """
        Restrict the basis to a set of states.

        :param basis: The generator basis.
        :param idx: The states.
        """
        self.basis, self.idx = basis, idx
        C, M, self.rows, self.cols = basis.transient(idx)

        #: The coalescence and migration parts at the entries.
        self.C = {p: np.asarray(X[self.rows, self.cols]).ravel() for p, X in C.items()}
        self.M = {pair: np.asarray(X[self.rows, self.cols]).ravel() for pair, X in M.items()}

        #: The factors of the coalescence parts and the parameter derivatives at the entries, by the rates of the
        #: recent epochs, evicted by the basis.
        self._epochs: Dict[tuple, Tuple[Dict[str, float], Dict[str, np.ndarray]]] = {}

    def add(self, grad: 'DemographyGradient', e: int, g: np.ndarray, scale: float) -> None:
        """
        Add a derivative by the entries of the generator of an epoch to the gradient.

        :param grad: The gradient.
        :param e: The position of the epoch.
        :param g: The derivative by the entries ``(rows, cols)``.
        :param scale: The factor with which the generator enters.
        """
        epoch = grad.epochs[e]
        key = (tuple(sorted(epoch.pop_sizes.items())), tuple(sorted(epoch.migration_rates.items())))
        if key not in self._epochs:
            self._epochs[key] = (self.basis.coefficients(epoch), {
                name: np.asarray(dS[self.rows, self.cols]).ravel()
                for name, dS in self.basis.parameter_derivatives(epoch, self.idx).items()
            })
        self.basis.touch(self, key, len(grad.epochs))
        coefficients, derivatives = self._epochs[key]

        for i, pop in enumerate(grad.pops):
            grad.pop_sizes[e, i] += scale * coefficients[pop] * float(self.C[pop] @ g)
        for j, pair in enumerate(grad.pairs):
            grad.migration_rates[e, j] += scale * float(self.M[pair] @ g)
        for name, dS in derivatives.items():
            grad.parameters[name] += scale * float(dS @ g)

    def size(self, key: tuple) -> int:
        """
        The bytes of a cached epoch: its parameter derivatives and about a kilobyte for its dictionaries.

        :param key: The rates of the epoch.
        :return: The bytes.
        """
        return 1024 + sum(dS.nbytes for dS in self._epochs[key][1].values())


class CrossMomentGradient:
    r"""
    Values and derivatives of raw cross-moments of :math:`k \geq 2` rewards accumulated until absorption, from the
    closed form of ``MomentEvaluator._accumulate_closed_form`` of one ordering, by a backward pass through its record,
    as described in :meth:`PhaseTypeDistribution.gradient()
    <phasegen.distributions.PhaseTypeDistribution.gradient>`. Block :math:`k - i` of its extended vector holds the
    moment of the last :math:`i` rewards divided by :math:`i!\, \lambda^i`, with :math:`\lambda` the balancing factor
    of its blocks, so the backward pass also differentiates these.
    """

    def __init__(self, distribution: 'PhaseTypeDistribution') -> None:
        """
        Configure the gradient.

        :param distribution: The distribution whose state space and demography define the moment.
        """
        self.distribution = distribution

    def compute(self, heads: np.ndarray, tail: Sequence[np.ndarray],
                seeds: Dict[int, np.ndarray] = None) -> Optional[List[Optional[Tuple[float, 'DemographyGradient']]]]:
        """
        For each column of ``heads``, the moment of the ordering of that column followed by the rewards of ``tail``,
        plus the moments of the last ``i`` rewards of ``tail`` weighted by ``seeds[i]``, and its gradient.

        :param heads: The first reward vectors over the states, one per column.
        :param tail: The following reward vectors.
        :param seeds: The weights of the moments of the last ``i`` rewards of the tail, one per column, by ``i``.
        :return: The weighted moment and its gradient of each column, ``None`` for a column without weight, or
            ``None`` where absorption is not certain in the last epoch.
        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        from math import factorial
        from ..rewards import CustomReward
        from .phase_type import DemographyGradient

        dist = self.distribution
        k = 1 + len(tail)
        if not dist._absorption_certain_in_last_epoch(k):
            return None

        heads = np.asarray(heads, dtype=float)
        seeds = {i: np.asarray(w, dtype=float) for i, w in (seeds or {}).items()}
        n = dist.state_space.k
        basis = _generator_basis(dist.state_space)
        names = basis.parameter_names
        full = basis.contraction(np.arange(n))
        rows = np.concatenate([j * n + full.rows for j in range(k + 1)])
        cols = np.concatenate([j * n + full.cols for j in range(k + 1)])
        tail = tuple(CustomReward(lambda _, r=np.asarray(r, dtype=float): r) for r in tail)
        last = None

        out, done = [], {}
        for c, head in enumerate(heads.T):
            weights = {i: float(w[c]) for i, w in seeds.items() if w[c] != 0}
            if not head.any() and not weights:
                out.append(None)
                continue

            # columns with the same head and weights share their pass
            key = (head.tobytes(), tuple(sorted(weights.items())))
            if key in done:
                out.append(done[key])
                continue

            tape = _ClosedFormTape()
            value = float(dist._accumulate_closed_form(k, (CustomReward(lambda _, r=head: r),) + tail, tape=tape)[0])
            grad = DemographyGradient(tape.epochs, basis.pops, basis.pairs, parameters=names)
            last = last or basis.contraction(tape.idx_t)

            # the adjoint of the extended vector at time zero, on block 0 for the moment of the whole ordering and on
            # block k - i for that of the last i rewards
            y = tape.factor * tape.alpha_ext
            for i, w in weights.items():
                scale = w * factorial(i) * tape.lamb_end ** i
                y[(k - i) * n:(k - i + 1) * n] += scale * tape.alpha_ext[:n]
                value += scale * float(tape.alpha_ext[:n] @ tape.z[(k - i) * n:(k - i + 1) * n])

            # the finite epochs, earliest first, carrying the adjoint row vector forward
            for step in reversed(tape.steps):
                if step.action is None:
                    G = sla.expm_frechet(step.A.T, np.outer(y, step.z_in), compute_expm=False)
                    g = sum(G[j * n:(j + 1) * n, j * n:(j + 1) * n] for j in range(k + 1))[full.rows, full.cols]
                    y_in = y @ step.exp
                else:
                    y_in, gA = step.action.reverse(y, step.checkpoints, rows, cols)
                    g = gA.reshape(k + 1, -1).sum(axis=0)

                # the generator enters every diagonal block multiplied by the duration, and the boundaries of the epoch
                # move its duration
                full.add(grad, step.index, g, step.duration)
                dt = float(y @ (step.A @ step.z_out)) / step.duration
                grad.start_times[step.index + 1] += dt
                if step.index > 0:
                    grad.start_times[step.index] -= dt

                y = y_in * step.rebase

            # the last epoch: block j is (-T)^{-1} (r_j * block j + 1) / lamb, the last block the ones vector
            solve = tape.solve.T
            a = y[tape.idx_t]
            g = np.zeros(len(last.rows))
            for j in range(k):
                q = solve(a)
                g += q[last.rows] * tape.blocks[j][last.cols]
                if j + 1 < k:
                    a = y[(j + 1) * n + tape.idx_t] + q / tape.lamb * tape.rewards[j][tape.idx_t]
            last.add(grad, len(tape.epochs) - 1, g, 1.0)

            grad.parameters = {name: float(v) for name, v in grad.parameters.items()}
            done[key] = (value, grad)
            out.append(done[key])

        return out


def _group_terms(
        terms: Sequence[Tuple[np.ndarray, Tuple[np.ndarray, ...]]]
) -> Tuple[List[Tuple[Tuple[np.ndarray, ...], np.ndarray, Dict[int, np.ndarray]]], Optional[np.ndarray]]:
    """
    Group weighted ordered raw moments of ``PhaseTypeDistribution._moment_terms`` into backward passes. An ordering is
    linear in its first reward, so the orderings that share their following rewards take one pass with their first
    rewards summed. An ordering that ends the following rewards of a longer group is read from the trailing blocks of
    that group's pass, and the remaining means take one pass of ``MeanGradient``.

    :param terms: The weights, one per column, with the reward vectors of each ordering.
    :return: The following rewards, the first rewards weighted per column and the weights of the trailing moments of
        each group, and the weighted rewards of the remaining means, ``None`` if there are none.
    """
    keys = [tuple(v.tobytes() for v in vectors) for _, vectors in terms]
    tails = {}
    for (_, vectors), key in zip(terms, keys):
        if len(vectors) > 1:
            tails.setdefault(key[1:], vectors[1:])

    # the longest groups first, which can hold the most trailing moments
    hosts = sorted(tails, key=len, reverse=True)
    groups = {tail: [tails[tail], None, {}] for tail in tails}
    means = None
    for (w, vectors), key in zip(terms, keys):
        host = next((t for t in hosts if len(t) >= len(key) and t[len(t) - len(key):] == key), None)
        if host is not None:
            seeds = groups[host][2]
            seeds[len(key)] = seeds.get(len(key), 0) + w
        elif len(key) > 1:
            group = groups[key[1:]]
            group[1] = np.outer(vectors[0], w) + (0 if group[1] is None else group[1])
        else:
            means = np.outer(vectors[0], w) + (0 if means is None else means)

    n_columns = len(terms[0][0]) if terms else 0
    out = []
    for tail, heads, seeds in groups.values():
        if heads is not None or seeds:
            out.append((tail, np.zeros((len(tail[0]), n_columns)) if heads is None else heads, seeds))

    return out, means


def _passes(
        distribution: 'PhaseTypeDistribution',
        terms: Sequence[Tuple[np.ndarray, Tuple[np.ndarray, ...]]],
        n_columns: int
) -> Optional[List[List[Optional[Tuple[float, 'DemographyGradient']]]]]:
    """
    The backward passes of :func:`_group_terms` over weighted ordered raw moments.

    :param distribution: The distribution whose state space the reward vectors are defined on.
    :param terms: The weights, one per column, with the reward vectors of each ordering.
    :param n_columns: The number of columns.
    :return: Per pass, the weighted moment and its gradient of each column, ``None`` for a column without weight, or
        ``None`` where absorption is not certain in the last epoch.
    """
    groups, means = _group_terms(terms)

    outs = [CrossMomentGradient(distribution).compute(heads, tail, seeds) for tail, heads, seeds in groups]
    if means is not None:
        used = list(np.flatnonzero(np.abs(means).sum(axis=0)))
        out = MeanGradient(distribution).compute_columns(means[:, used])
        outs.append(None if out is None else [out[used.index(c)] if c in used else None for c in range(n_columns)])

    return None if any(o is None for o in outs) else outs


def _entries_gradients(
        distribution: 'PhaseTypeDistribution',
        entries: Sequence[Sequence[Tuple[float, Tuple[np.ndarray, ...]]]]
) -> Optional[List[Tuple[float, 'DemographyGradient']]]:
    """
    The weighted sums of ordered raw moments of ``PhaseTypeDistribution._moment_terms`` of several entries and their
    gradients, by the passes of :func:`_passes` over all entries at once.

    :param distribution: The distribution whose state space the reward vectors are defined on.
    :param entries: The coefficients with the reward vectors of each ordering, per entry.
    :return: The sum and its gradient per entry, or ``None`` where absorption is not certain in the last epoch.
    """
    from .phase_type import DemographyGradient

    n = len(entries)
    terms = [(np.eye(n)[c] * coefficient, vectors) for c, entry in enumerate(entries) for coefficient, vectors in entry]
    outs = _passes(distribution, terms, n)
    if outs is None:
        return None

    k = max(len(vectors) for _, vectors in terms)
    results = []
    for c in range(n):
        column = [o[c] for o in outs if o[c] is not None]
        results.append((sum(v for v, _ in column), DemographyGradient._sum([g for _, g in column])) if column else
                       (0.0, _zero_gradient(distribution, k)))

    return results


def _zero_gradient(distribution: 'PhaseTypeDistribution', k: int) -> 'DemographyGradient':
    """
    The gradient of a moment of order ``k`` with zero weight.

    :param distribution: The distribution.
    :param k: The order, which fixes the epochs.
    :return: The gradient.
    """
    from .phase_type import DemographyGradient

    basis = _generator_basis(distribution.state_space)
    names = basis.parameter_names

    return DemographyGradient(distribution._get_epochs_until_unbounded(k), basis.pops, basis.pairs, parameters=names)


def _generator_basis(state_space: 'StateSpace') -> _GeneratorBasis:
    """
    The generator basis of a state space, kept on the state space, which :class:`~phasegen.inference.Inference`
    shares across evaluations, as is the error of a space without one.

    :param state_space: The state space.
    :return: The basis.
    :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
    """
    if state_space._generator_basis is None:
        try:
            state_space._generator_basis = _GeneratorBasis(state_space)
        except ValueError as e:
            state_space._generator_basis = e

    if isinstance(state_space._generator_basis, ValueError):
        raise state_space._generator_basis

    return state_space._generator_basis
