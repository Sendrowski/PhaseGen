"""
Exact gradients of first moments with respect to the population sizes, migration rates and epoch start times of a
demography, by the adjoint of the occupation-time recursion of
:meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
"""

import copy
import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _exact_1_norm, _exact_inf_norm, _fragment_3_1

from ..demography import Epoch

if TYPE_CHECKING:
    from ..state_space import StateSpace
    from .phase_type import PhaseTypeDistribution

#: Unit roundoff targeted by the truncated Taylor action, as in :func:`scipy.sparse.linalg.expm_multiply`.
_TOL = 2.0 ** -53

#: Number of Taylor terms whose outer products are contracted in one batch of the reverse pass.
_CHUNK = 64

#: Relative tolerance of the reconstruction of a generator from the basis of :class:`_GeneratorBasis`.
_BASIS_RTOL = 1e-10

#: Step of the central differences of the generator over a parameter of the coalescent model or the recombination
#: rate, relative to the magnitude of the parameter (at least 1).
_PARAMETER_STEP = 1e-6


class _TapedAction:
    r"""
    The action :math:`\mathbf{y} = e^{\mathbf{M}} \mathbf{x}` of a sparse square matrix :math:`\mathbf{M}` on a vector
    :math:`\mathbf{x}`, by the truncated Taylor algorithm of Al-Mohy and Higham (2011, Algorithm 3.2), with the degree,
    the number of steps and the diagonal shift :math:`\mu` chosen as by
    :func:`scipy.sparse.linalg.expm_multiply`, recording the input of every step for the reverse pass :meth:`reverse`,
    which recomputes the Taylor terms one step at a time.
    """

    def __init__(self, M: sp.csr_matrix) -> None:
        """
        Choose the degree and the number of steps of the action.

        :param M: The sparse matrix to exponentiate.
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

    def apply(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, int]]]:
        """
        Apply the action.

        :param x: The vector.
        :return: :math:`e^{\\mathbf{M}} \\mathbf{x}`, and the input and number of Taylor terms of every step, from
            which :meth:`reverse` recomputes the terms one step at a time.
        """
        eta = np.exp(self.mu / self.steps)
        checkpoints, F = [], x
        for _ in range(self.steps):
            terms = self._terms(F)
            checkpoints.append((F, len(terms)))
            F = eta * np.sum(terms, axis=0)

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
    :class:`MeanGradient`.
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

        #: The bases at a parameter shifted down and up, with the shifted values, by parameter.
        self._shifted: Dict[str, Optional[Tuple['_GeneratorBasis', '_GeneratorBasis', float, float]]] = {}

    @staticmethod
    def parameters(model: Any, locus_config: Any) -> Dict[str, float]:
        """
        The continuous parameters of a coalescent model, such as ``alpha`` of a Beta coalescent or ``psi`` and ``c`` of
        a Dirac coalescent, and the recombination rate of two loci.

        :param model: The coalescent model.
        :param locus_config: The locus configuration.
        :return: The parameters by name.
        """
        out = {}
        for name in inspect.signature(type(model).__init__).parameters:
            value = getattr(model, name, None)
            if name not in ('self', 'scale_time') and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name] = float(value)

        if locus_config.n == 2:
            out['recombination_rate'] = float(locus_config.recombination_rate)

        return out

    def _shift(self, name: str, value: float) -> '_GeneratorBasis':
        """
        The basis of a copy of the state space with one parameter changed.

        :param name: The parameter.
        :param value: Its value.
        :return: The basis.
        """
        ss = self.state_space
        clone = copy.copy(ss)
        for key in ('S', 'transition', '_generator_basis'):
            clone.__dict__.pop(key, None)
        clone._previous_S, clone._cache = {}, {}

        if name == 'recombination_rate':
            clone.locus_config = copy.copy(ss.locus_config)
            clone.locus_config.recombination_rate = value
        else:
            model = ss.model
            kwargs = {k: getattr(model, k) for k in inspect.signature(type(model).__init__).parameters
                      if k != 'self' and hasattr(model, k)}
            clone.model = type(model)(**(kwargs | {name: value}))

        return _GeneratorBasis(clone)

    def _shifted_bases(self, name: str) -> Optional[Tuple['_GeneratorBasis', '_GeneratorBasis', float, float]]:
        """
        The bases at a parameter shifted down and up by a central step, one-sided where a shifted value is invalid,
        memoized.

        :param name: The parameter.
        :return: The bases and the shifted values, or ``None`` if neither side can be shifted.
        """
        if name not in self._shifted:
            x = self.parameters(self.state_space.model, self.state_space.locus_config)[name]
            h = _PARAMETER_STEP * max(abs(x), 1.0)
            sides = []
            for y in (x - h, x + h):
                try:
                    sides.append((self._shift(name, y), y))
                except (ValueError, ZeroDivisionError):
                    sides.append((self, x))

            (down, x_down), (up, x_up) = sides
            self._shifted[name] = None if x_up == x_down else (down, up, x_down, x_up)

        return self._shifted[name]

    def parameter_derivatives(self, epoch: Epoch, idx_t: np.ndarray) -> Dict[str, sp.csr_matrix]:
        r"""
        The derivatives :math:`\partial \mathbf{T} / \partial \theta` of the transient block of the generator of an
        epoch by the parameters :math:`\theta` of :meth:`parameters`, by central differences of the bases, which are
        exact for the recombination rate, on which the generator depends linearly.

        :param epoch: The epoch.
        :param idx_t: The transient states.
        :return: The derivatives by parameter, omitting a parameter that cannot be shifted.
        """
        out = {}
        for name in self.parameters(self.state_space.model, self.state_space.locus_config):
            shifted = self._shifted_bases(name)
            if shifted is None:
                continue

            down, up, x_down, x_up = shifted
            generators = []
            for basis in (down, up):
                model = basis.state_space.model
                S = basis.K + sum((basis.C[p] / model._get_timescale(epoch.pop_sizes[p]) for p in basis.pops),
                                  sp.csr_matrix(basis.K.shape))
                for pair in basis.pairs:
                    S = S + epoch.migration_rates.get(pair, 0.0) * basis.M[pair]
                generators.append(S)

            out[name] = ((generators[1] - generators[0]) / (x_up - x_down)).tocsr()[idx_t][:, idx_t].tocsr()

        return out

    def transient(self, idx_t: np.ndarray) -> Tuple[
        Dict[str, sp.csr_matrix], Dict[Tuple[str, str], sp.csr_matrix], np.ndarray, np.ndarray]:
        """
        The coalescence and migration parts restricted to transient states, and the union of the sparsity patterns of
        the restricted parts and :attr:`K`, which holds every entry a generator of the space can take, whatever its
        rates, memoized per set of states.

        :param idx_t: The transient states.
        :return: The coalescence parts by deme, the migration parts by ordered pair, and the rows and columns of the
            pattern.
        """
        key = np.asarray(idx_t).tobytes()
        if key not in self._transient:
            C = {p: self.C[p][idx_t][:, idx_t].tocsr() for p in self.pops}
            M = {pair: self.M[pair][idx_t][:, idx_t].tocsr() for pair in self.pairs}
            union = abs(self.K[idx_t][:, idx_t])
            for X in list(C.values()) + list(M.values()):
                union = union + abs(X)
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
        model = self.state_space.model
        rebuilt = self.K + sum((self.C[p] / model._get_timescale(sizes[p]) for p in self.pops), sp.csr_matrix(S.shape))
        for pair in self.pairs:
            rebuilt = rebuilt + rates.get(pair, 0.0) * self.M[pair]

        diff = abs(rebuilt - S).max()
        if diff > _BASIS_RTOL * max(abs(S).max(), 1e-300):
            raise ValueError(
                f"The generator is not affine in the inverse time scales and migration rates (deviation {diff:.3g})."
            )


class DemographyGradient:
    r"""
    Derivatives of a first moment :math:`\mu` with respect to the population sizes :math:`N_{d,e}` of the demes
    :math:`d`, the migration rates :math:`m_{ab,e}` from deme :math:`a` to deme :math:`b` and the start times
    :math:`t_e` of the epochs :math:`e` the moment was evaluated over, as returned by
    :meth:`PhaseTypeDistribution.gradient() <phasegen.distributions.PhaseTypeDistribution.gradient>`. For a spectrum,
    each array carries the shape of the spectrum as leading axes, one derivative per bin.
    """

    def __init__(self, epochs: Sequence[Epoch], pops: Sequence[str], pairs: Sequence[Tuple[str, str]],
                 shape: Tuple[int, ...] = (), parameters: Sequence[str] = ()) -> None:
        """
        Initialize a zero gradient.

        :param epochs: The epochs.
        :param pops: The demes.
        :param pairs: The ordered pairs of demes.
        :param shape: The shape of the moment, empty for a scalar.
        :param parameters: The parameters of the coalescent model and the recombination rate differentiated by.
        """
        #: The epochs, the last of which extends until absorption.
        self.epochs: List[Epoch] = list(epochs)

        #: The demes.
        self.pops: List[str] = list(pops)

        #: The ordered pairs of demes.
        self.pairs: List[Tuple[str, str]] = list(pairs)

        #: Derivative :math:`\partial \mu / \partial N_{d,e}`, of shape ``shape + (n_epochs, n_demes)``.
        self.pop_sizes: np.ndarray = np.zeros(tuple(shape) + (len(self.epochs), len(self.pops)))

        #: Derivative :math:`\partial \mu / \partial m_{ab,e}`, of shape ``shape + (n_epochs, n_pairs)``, in the
        #: order of :attr:`pairs`.
        self.migration_rates: np.ndarray = np.zeros(tuple(shape) + (len(self.epochs), len(self.pairs)))

        #: Derivative :math:`\partial \mu / \partial t_e`, of shape ``shape + (n_epochs,)``. The first epoch starts at
        #: zero, so its entry is zero.
        self.start_times: np.ndarray = np.zeros(tuple(shape) + (len(self.epochs),))

        #: Derivatives by the continuous parameters of the coalescent model, such as ``alpha`` of a Beta coalescent or
        #: ``psi`` and ``c`` of a Dirac coalescent, and by the ``recombination_rate`` of two loci, each of shape
        #: ``shape``.
        self.parameters: Dict[str, np.ndarray] = {name: np.zeros(tuple(shape)) for name in parameters}

    @classmethod
    def _stack(cls, gradients: Sequence['DemographyGradient'], positions: Sequence[Tuple[int, ...]],
               shape: Tuple[int, ...]) -> 'DemographyGradient':
        """
        The gradient of a spectrum from the gradients of its entries.

        :param gradients: The gradients of the entries, over the same epochs.
        :param positions: The positions of the entries in the spectrum.
        :param shape: The shape of the spectrum.
        :return: The gradient, zero at the other positions.
        """
        first = gradients[0]
        out = cls(first.epochs, first.pops, first.pairs, shape, list(first.parameters))
        for g, pos in zip(gradients, positions):
            out.pop_sizes[pos] = g.pop_sizes
            out.migration_rates[pos] = g.migration_rates
            out.start_times[pos] = g.start_times
            for name, value in g.parameters.items():
                out.parameters[name][pos] = value

        return out

    def __iadd__(self, other: 'DemographyGradient') -> 'DemographyGradient':
        """
        Add a gradient over the same epochs.

        :param other: The gradient.
        :return: This gradient.
        :raises ValueError: If the epochs or demes differ.
        """
        if ([e.index for e in other.epochs] != [e.index for e in self.epochs] or other.pops != self.pops or
                other.pairs != self.pairs):
            raise ValueError("The gradients are over different epochs or demes.")

        self.pop_sizes += other.pop_sizes
        self.migration_rates += other.migration_rates
        self.start_times += other.start_times
        for name, value in other.parameters.items():
            self.parameters[name] = self.parameters.get(name, 0.0) + value

        return self

    @staticmethod
    def quantities(epochs: Sequence[Epoch], pops: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """
        The population sizes, migration rates and start times of epochs, in the layout of :meth:`flat`.

        :param epochs: The epochs.
        :param pops: The demes.
        :param pairs: The ordered pairs of demes.
        :return: The quantities.
        """
        return np.concatenate([
            [e.pop_sizes[p] for e in epochs for p in pops],
            [e.migration_rates.get(pair, 0.0) for e in epochs for pair in pairs],
            [e.start_time for e in epochs]
        ])

    def flat(self) -> np.ndarray:
        """
        The gradient as one vector, in the layout of :meth:`quantities`.
        """
        return np.concatenate([self.pop_sizes.ravel(), self.migration_rates.ravel(), self.start_times])


class MeanGradient:
    r"""
    Value and gradient of the mean :math:`\mathbb{E}[R] = \mathbf{m}\, \mathbf{r}` of an accumulated reward with
    reward vector :math:`\mathbf{r}`, where :math:`\mathbf{m}` holds the expected occupation times of the states
    until absorption, with respect to the demographic quantities of :class:`DemographyGradient`.

    The occupation is threaded through the finite epochs as the row vector :math:`\mathbf{w}_e = [\mathbf{p}_e,
    \mathbf{m}_e]` of the state distribution and the accumulated occupation, :math:`\mathbf{w}_{e+1} = \mathbf{w}_e
    e^{\mathbf{A}_e}` with :math:`\mathbf{A}_e = \tau_e [[\mathbf{T}_e, \mathbf{I}], [\mathbf{0}, \mathbf{0}]]`, where
    :math:`\mathbf{T}_e` is the transient block of the generator of epoch :math:`e` and :math:`\tau_e` its duration.
    The last epoch :math:`E` contributes :math:`\mathbf{p}_E (-\mathbf{T}_E)^{-1}`. With the adjoint vectors
    :math:`\mathbf{v}_E = [(-\mathbf{T}_E)^{-1} \mathbf{r}; \mathbf{r}]` and :math:`\mathbf{v}_e = e^{\mathbf{A}_e}
    \mathbf{v}_{e+1}`,

    .. math::
        \frac{\partial\, \mathbb{E}[R]}{\partial \theta} =
        \sum_{e < E} \tau_e \Big\langle \frac{\partial \mathbf{T}_e}{\partial \theta},\,
        L\big(\mathbf{A}_e^\top, \mathbf{w}_e^\top \mathbf{v}_{e+1}^\top\big)_{11} \Big\rangle
        + \mathbf{p}_E (-\mathbf{T}_E)^{-1} \frac{\partial \mathbf{T}_E}{\partial \theta} (-\mathbf{T}_E)^{-1} \mathbf{r},

    where :math:`\theta` is a population size or a migration rate, :math:`L(\mathbf{A}, \mathbf{E})` the Fréchet
    derivative of the matrix exponential, :math:`[\cdot]_{11}` its upper left block and :math:`\langle \cdot, \cdot
    \rangle` the entrywise inner product. On the sparse path, the Fréchet derivative is replaced by the reverse pass of
    the truncated Taylor action. The start time :math:`t_{e+1}` of epoch :math:`e + 1` enters as :math:`\mathbf{w}_{e+1} (\mathbf{V}_e - \mathbf{V}_{e+1}) \mathbf{v}_{e+1}`,
    :math:`\mathbf{V}_e = \mathbf{A}_e / \tau_e`.

    The following example evaluates the gradient of the expected tree height.

    ::

        coal = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))
        th = coal.tree_height

        value, grad = MeanGradient(th).compute(th.reward._get(th.state_space))
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

    def compute(self, reward: np.ndarray) -> Optional[Tuple[float, DemographyGradient]]:
        """
        The mean and its gradient.

        :param reward: The reward vector over the states of the state space.
        :return: The mean and its gradient, or ``None`` where the occupation times are not defined in closed form.
        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        out = self.compute_columns(np.asarray(reward, dtype=float)[:, None])

        return None if out is None else out[0]

    def compute_columns(self, rewards: np.ndarray) -> Optional[List[Tuple[float, DemographyGradient]]]:
        """
        The means and gradients of several rewards, sharing the forward pass, the factorization of the last epoch and
        the adjoint of the initial distribution.

        :param rewards: The reward vectors over the states of the state space, one per column.
        :return: The mean and gradient of each column, or ``None`` where the occupation times are not defined in closed
            form.
        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
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

        # the derivatives of the generator by the model parameters and the recombination rate, per epoch
        dT = [basis.parameter_derivatives(step.epoch, idx_t) for step in tape]
        names = list(dT[0])

        out = []
        for r in np.asarray(rewards, dtype=float)[idx_t].T:
            grad = DemographyGradient([step.epoch for step in tape], basis.pops, basis.pairs, parameters=names)

            # G is the derivative with respect to the transient block of the generator of epoch e, which enters the
            # exponent of a finite epoch multiplied by its duration, passed as the scale
            def accumulate(e: int, step: _OccupationStep, G: 'sp.spmatrix | np.ndarray', scale: float) -> None:
                coefficients = basis.coefficients(step.epoch)
                G = sp.csr_matrix(G)
                for i, pop in enumerate(basis.pops):
                    grad.pop_sizes[e, i] += scale * coefficients[pop] * float(C[pop].multiply(G).sum())
                for j, pair in enumerate(basis.pairs):
                    grad.migration_rates[e, j] += scale * float(M[pair].multiply(G).sum())
                for name in names:
                    grad.parameters[name] += scale * float(dT[e][name].multiply(G).sum())

            u = solve(r)
            accumulate(len(tape) - 1, last,
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
                    accumulate(e, step, G, step.tau)
                    v = step.exp_aug @ v
                else:
                    # the action applies exp(A^T) to [p, 0], so dL/dA_{rc} = dL/dM_{cr}, taken over the entries every
                    # generator of the space can take, also those at a rate of zero in this epoch
                    v_in, gM = step.action.reverse(v, step.checkpoints, cols, rows)
                    GA = sp.csr_matrix((gM, (rows, cols)), shape=(nt, nt))
                    accumulate(e, step, GA, step.tau)
                    v = self._adjoint_input(v_in, v, nt)

                V_next_v = V @ v

            out.append((float(m @ r), grad))

        return out

    def _basis(self) -> _GeneratorBasis:
        """
        The generator basis of the state space, kept on the state space, which :class:`~phasegen.inference.Inference`
        shares across evaluations, as is the error of a space without one.

        :raises ValueError: If the generator is not affine in the inverse time scales and migration rates.
        """
        ss = self.distribution.state_space
        basis = ss.__dict__.get('_generator_basis')
        if not isinstance(basis, (_GeneratorBasis, ValueError)):
            try:
                basis = _GeneratorBasis(ss)
            except ValueError as e:
                basis = e
            ss.__dict__['_generator_basis'] = basis

        if isinstance(basis, ValueError):
            raise basis

        return basis

    @staticmethod
    def _adjoint_input(v_in: np.ndarray, v_out: np.ndarray, nt: int) -> np.ndarray:
        """
        The adjoint at the start of an epoch on the sparse path, where the action is applied to ``[p, 0]`` and the
        accumulated occupation is carried past it.

        :param v_in: The adjoint of the input of the action.
        :param v_out: The adjoint at the end of the epoch.
        :param nt: The number of transient states.
        :return: The adjoint of ``[p, m]`` at the start of the epoch.
        """
        return np.concatenate([v_in[:nt], v_out[nt:]])
