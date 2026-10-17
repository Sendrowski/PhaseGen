"""
The backward pass of the occupation-time recursion that gives the exact derivatives of first moments, as described in
:meth:`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>`.
"""

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _exact_1_norm, _exact_inf_norm, _fragment_3_1

from .._blas import BlasThreadLimit
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

#: Largest step of the differences of the generator over a parameter of the coalescent model or the recombination
#: rate, relative to the magnitude of the parameter (at least 1).
_PARAMETER_STEP = 1e-6

#: Smallest relative step tried, below which a parameter counts as not variable.
_MIN_PARAMETER_STEP = 1e-14

#: Number of steps a parameter must be able to move towards each side of a difference.
_STEP_MARGIN = 1000



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
    the number of steps and the diagonal shift :math:`\mu` chosen as by
    :func:`scipy.sparse.linalg.expm_multiply`, recording the input of every step for the reverse pass :meth:`reverse`,
    which recomputes the Taylor terms one step at a time. Above a 1-norm of ``_ACTION_STEP_NORM`` the steps end once
    the vector is stationary over the remaining ones, tested as in :meth:`Backend.expm_multiply()
    <phasegen.expm.Backend.expm_multiply>`.
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
        #: which :meth:`apply` tests stationarity, at the fractions of :meth:`Backend._step_fractions()
        #: <phasegen.expm.Backend._step_fractions>` when the norm of the matrix exceeds ``_ACTION_STEP_NORM``.
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
        :return: :math:`e^{\\mathbf{M}} \\mathbf{x}`, and the input and number of Taylor terms of every step, from
            which :meth:`reverse` recomputes the terms one step at a time.
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
        Reverse pass of :meth:`apply` for a scalar :math:`L` with :math:`\\partial L / \\partial \\mathbf{y} = \\lambda`,
        holding the shift, the degree and the number of steps fixed.

        :param lam: The adjoint of the result.
        :param checkpoints: The step inputs and term counts recorded by :meth:`apply`.
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
            action: _TapedAction = None,
            sparse: bool = False
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
        :param sparse: Whether the solve of the last epoch is sparse.
        """
        self.epoch, self.idx_t, self.tau, self.T, self.w = epoch, idx_t, tau, T, w
        self.exp_aug, self.action, self.sparse = exp_aug, action, sparse

        #: Step inputs and term counts of the action.
        self.checkpoints: Optional[List[Tuple[np.ndarray, int]]] = None

    def augmented(self) -> sp.csr_matrix:
        """
        The augmented generator :math:`[[\\mathbf{T}, \\mathbf{I}], [\\mathbf{0}, \\mathbf{0}]]`, unscaled.
        """
        n = len(self.idx_t)
        return sp.bmat([[sp.csr_matrix(self.T), sp.identity(n)], [None, sp.csr_matrix((n, n))]], format='csr')


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
        The bases and weights of a second-order difference of the generator over a parameter, memoized. A model
        parameter takes a central difference where it may vary by a thousand steps either way and a one-sided one
        towards valid values otherwise. The recombination rate, on which the generator is linear, takes the full step.

        :param name: The parameter.
        :return: The bases with their weights, or ``None`` if the parameter cannot be varied.
        """
        if name not in self._stencils:
            x = _rate_parameters(self.state_space.model, self.state_space.locus_config)[name]
            scale = max(abs(x), 1.0)
            stencil = None

            h = _PARAMETER_STEP * scale
            if name == 'recombination_rate':
                stencil = ([(self._varied(name, x + h), 1 / (2 * h)), (self._varied(name, x - h), -1 / (2 * h))]
                           if x >= h else [(self, -3 / (2 * h)), (self._varied(name, x + h), 4 / (2 * h)),
                                           (self._varied(name, x + 2 * h), -1 / (2 * h))])

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

    def parameter_derivatives(self, epoch: Epoch, idx_t: np.ndarray) -> Dict[str, sp.csr_matrix]:
        r"""
        The derivatives :math:`\partial \mathbf{T} / \partial \theta` of the transient block of the generator of an
        epoch by the parameters :math:`\theta` of :func:`_rate_parameters`, by second-order differences of the bases,
        which are exact for the recombination rate, on which the generator depends linearly.

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

    @property
    def _log(self) -> logging.Logger:
        """
        The logger.
        """
        return logging.getLogger(f"phasegen.{type(self).__name__}")

    @BlasThreadLimit()
    def compute(self, reward: np.ndarray) -> Optional[Tuple[float, 'DemographyGradient']]:
        """
        The mean and its gradient.

        :param reward: The reward vector over the states of the state space.
        :return: The mean and its gradient, or ``None`` where the occupation times are not defined in closed form.
        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        out = self.compute_columns(np.asarray(reward, dtype=float)[:, None])

        return None if out is None else out[0]

    @BlasThreadLimit()
    def compute_columns(self, rewards: np.ndarray) -> Optional[List[Tuple[float, 'DemographyGradient']]]:
        """
        The means and gradients of several rewards, sharing the forward pass, the factorization of the last epoch and
        the adjoint of the initial distribution.

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
        basis = self._basis()
        C, M, rows, cols = basis.transient(idx_t)

        # last epoch: d(p (-T)^{-1} r) = q dT u with q = p (-T)^{-1} and u = (-T)^{-1} r
        last = tape[-1]
        neg_t = -last.T
        solve = dist._lu_solver(neg_t, last.sparse)
        q = dist._lu_solver(neg_t.T, last.sparse)(last.w[:nt])

        # the factors of the coalescence parts and the derivatives of the generator by the model parameters and the
        # recombination rate, per epoch
        coefficients = [basis.coefficients(step.epoch) for step in tape]
        dT = [basis.parameter_derivatives(step.epoch, idx_t) for step in tape]
        names = list(dT[0])

        out = []
        for r in np.asarray(rewards, dtype=float)[idx_t].T:
            grad = DemographyGradient([step.epoch for step in tape], basis.pops, basis.pairs, parameters=names)

            # G is the derivative with respect to the transient block of the generator of epoch e, which enters the
            # exponent of a finite epoch multiplied by its duration, passed as the scale
            def accumulate(e: int, G: 'sp.spmatrix | np.ndarray', scale: float) -> None:
                G = sp.csr_matrix(G)
                for i, pop in enumerate(basis.pops):
                    grad.pop_sizes[e, i] += scale * coefficients[e][pop] * float(C[pop].multiply(G).sum())
                for j, pair in enumerate(basis.pairs):
                    grad.migration_rates[e, j] += scale * float(M[pair].multiply(G).sum())
                for name in names:
                    grad.parameters[name] += scale * float(dT[e][name].multiply(G).sum())

            u = solve(r)
            accumulate(len(tape) - 1,
                       sp.csr_matrix((q[rows] * u[cols], (rows, cols)), shape=(nt, nt)) if last.sparse
                       else np.outer(q, u), 1.0)

            v = np.concatenate([u, r])

            # the last augmented generator annihilates its adjoint vector on the transient states
            V_next_v = np.concatenate([last.T @ u + r, np.zeros(nt)])

            for e in range(len(tape) - 2, -1, -1):
                step = tape[e]
                V = step.augmented()
                w_next = tape[e + 1].w
                grad.start_times[e + 1] += float(w_next @ (V @ v) - w_next @ V_next_v)

                if step.action is None:
                    A = V.toarray() * step.tau
                    G = sla.expm_frechet(A.T, np.outer(step.w, v), compute_expm=False)[:nt, :nt]
                    accumulate(e, G, step.tau)
                    v = step.exp_aug @ v
                else:
                    # the action applies exp(A^T) to [p, 0], so dL/dA_{rc} = dL/dM_{cr}, taken over the entries every
                    # generator of the space can take, also those at a rate of zero in this epoch
                    v_in, gM = step.action.reverse(v, step.checkpoints, cols, rows)
                    GA = sp.csr_matrix((gM, (rows, cols)), shape=(nt, nt))
                    accumulate(e, GA, step.tau)
                    # the action is applied to [p, 0], and the accumulated occupation is carried past it
                    v = np.concatenate([v_in[:nt], v[nt:]])

                V_next_v = V @ v

            grad.parameters = {name: float(value) for name, value in grad.parameters.items()}
            out.append((float(m @ r), grad))

        return out

    def _basis(self) -> _GeneratorBasis:
        """
        The generator basis of the state space, kept on the state space, which :class:`~phasegen.inference.Inference`
        shares across evaluations, as is the error of a space without one.

        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        ss = self.distribution.state_space
        if ss._generator_basis is None:
            try:
                ss._generator_basis = _GeneratorBasis(ss)
            except ValueError as e:
                ss._generator_basis = e

        if isinstance(ss._generator_basis, ValueError):
            raise ss._generator_basis

        return ss._generator_basis
