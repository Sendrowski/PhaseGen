"""
Exact Jacobians of statistics of a parametrized coalescent with respect to its parameters.
"""

import copy
import itertools
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._blas import BlasThreadLimit
from .distributions import Coalescent, DemographyGradient
from .distributions._common import _sqrt_spectrum, _validate_order
from .distributions._gradient import CrossMomentGradient, MeanGradient, _generator_basis, _group_terms, \
    _rate_parameters
from .spectrum import AbstractSpectrum

#: The distributions of a :class:`~phasegen.distributions.Coalescent` whose moments the gradient supports.
_DISTRIBUTIONS = ('sfs', 'fsfs', 'jsfs', 'sfs2', 'tree_height', 'total_branch_length')

#: Relative step of the central differences of the statistic over the entries of a moment. The statistic is evaluated on
#: moments it is handed, a smooth function of them, so that a small step leaves a truncation error of order its square.
_MOMENT_STEP = 1e-5

#: Relative step of the differences of the quantities of the coalescent over a parameter.
_PARAMETER_STEP = 1e-6

#: Relative tolerance between a moment and its value recomputed by the backward pass.
_VALUE_RTOL = 1e-9

#: Relative step of the differences of :meth:`Gradient._differences` by default, that of
#: :func:`scipy.optimize.approx_derivative` for central differences.
_DEFAULT_STEP = np.finfo(float).eps ** (1 / 3)

#: The errors by which a function differenced by :meth:`Gradient._stencil` signals that it is invalid at a point.
_INVALID_ERRORS = (ValueError, ZeroDivisionError)


class _Unsupported(BaseException):
    """
    The statistic reads a quantity the gradient does not support. A :class:`BaseException`, so that a statistic
    catching :class:`Exception` does not absorb it.
    """
    pass


class _Read(NamedTuple):
    """
    A moment the statistic reads: the distribution whose state space carries it, its kind, ``'mean'`` for the first
    moment, ``'moment'`` for a higher one and ``'cov'`` for the covariance of the bins of a spectrum, the arguments of
    that kind, the reward of the mean (``None`` for the reward of the distribution) or the order, rewards, centering
    and permutation of a higher moment, and its value.
    """
    dist: Any
    kind: str
    args: Any
    value: Any


#: The orders and centering of the higher moments a distribution serves as attributes.
_MOMENTS = {'var': (2, True), 'm2': (2, False), 'm3': (3, False), 'm4': (4, False)}


class _DistributionRecorder:
    """
    Stand-in for a distribution of a coalescent that serves its moments, recording each read: the mean, the variance,
    standard deviation and raw moments up to the fourth, and the covariance and correlation of the bins of a spectrum,
    and for the two-locus spectrum its mean.
    """

    def __init__(self, distribution: Any, recorder: '_CoalescentRecorder', name: str) -> None:
        """
        Wrap a distribution.

        :param distribution: The distribution.
        :param recorder: The recorder of the coalescent.
        :param name: The attribute name of the distribution.
        """
        self._distribution, self._recorder, self._name = distribution, recorder, name

    def __getattr__(self, attr: str) -> Any:
        """
        Serve a moment.

        :param attr: The attribute.
        :return: The moment, possibly overridden, or a function of one.
        :raises _Unsupported: For any other attribute.
        """
        dist = self._distribution
        spectrum = hasattr(dist, '_cov_entries')

        if attr == 'mean':
            return self._recorder.serve(lambda: _Read(dist, 'mean', None, dist.mean))

        if attr in _MOMENTS and dist._differentiable(_MOMENTS[attr][0]):
            k, center = _MOMENTS[attr]
            return self._recorder.serve(lambda: _Read(dist, 'moment', (k, None, center, True), getattr(dist, attr)))

        if attr == 'std' and dist._differentiable(2):
            var = self.var
            return _sqrt_spectrum(var) if isinstance(var, AbstractSpectrum) else max(float(var), 0.0) ** 0.5

        if attr == 'cov' and spectrum:
            return self._recorder.serve(lambda: _Read(dist, 'cov', None, dist.cov))

        if attr == 'corr' and spectrum:
            return dist._corr_from(self.cov)

        self._recorder.unsupported = True
        raise _Unsupported(f"{self._name}.{attr}")

    def moment(self, k: int = 1, rewards: Sequence = None, start_time: float = None, end_time: float = None,
               center: bool = True, permute: bool = True) -> Any:
        """
        Serve a moment, see :meth:`_CoalescentRecorder.order()
        <phasegen.gradient._CoalescentRecorder.order>`.
        """
        dist = self._distribution
        k = self._recorder.order(k, rewards, start_time, end_time, f"{self._name}.moment")

        if not dist._differentiable(k):
            self._recorder.unsupported = True
            raise _Unsupported(f"{self._name}.moment of order {k}")

        if k == 1:
            return self._recorder.serve(lambda: _Read(
                dist, 'mean', None if rewards is None else rewards[0],
                dist.mean if rewards is None else dist.moment(1, rewards=rewards)
            ))

        return self._recorder.serve(lambda: _Read(
            dist, 'moment', (k, None if rewards is None else tuple(rewards), center, permute),
            dist.moment(k, rewards=rewards, center=center, permute=permute)
        ))


class _CoalescentRecorder:
    """
    Stand-in for a :class:`~phasegen.distributions.Coalescent` passed to a statistic, admitting only the moments of
    :data:`_DISTRIBUTIONS` that :class:`_DistributionRecorder` serves and :meth:`Coalescent.moment()
    <phasegen.distributions.Coalescent.moment>`. The reads of one evaluation are numbered in order, so that a moment
    can be overridden in the next evaluation.
    """

    def __init__(self, coal: Coalescent) -> None:
        """
        Wrap a coalescent.

        :param coal: The coalescent.
        """
        self._coal = coal

        #: The reads of the current evaluation, in order.
        self.reads: List[_Read] = []

        #: The reads of the first evaluation, in order.
        self._first: List[_Read] = []

        #: Moments served in place of the computed ones, by the number of the read.
        self.overrides: Dict[int, Any] = {}

        #: Whether the statistic read an unsupported quantity.
        self.unsupported: bool = False

    def serve(self, make: Callable[[], _Read]) -> Any:
        """
        Record a read and serve its moment.

        :param make: Returns the read, called at the first evaluation only.
        :return: The moment, possibly overridden.
        """
        i = len(self.reads)
        if i == len(self._first):
            self._first.append(make())

        self.reads.append(self._first[i])

        return self.overrides.get(i, self._first[i].value)

    def order(self, k: Any, rewards: Optional[Sequence], start_time: Optional[float], end_time: Optional[float],
              name: str) -> int:
        """
        The order of a moment read, normalised as by :meth:`PhaseTypeDistribution.moment()
        <phasegen.distributions.PhaseTypeDistribution.moment>`, after validating the arguments of the read.

        :param k: The order, at least one, an integer or a float of integral value.
        :param rewards: ``None``, or one reward per order.
        :param start_time: ``None`` or zero.
        :param end_time: ``None`` or infinite.
        :param name: The quantity, for the signal.
        :return: The order.
        :raises _Unsupported: For an order below one, a number of rewards other than the order or a time window.
        """
        try:
            order = _validate_order(k)
        except (TypeError, ValueError):
            order = 0

        if order < 1 or (rewards is not None and len(rewards) != order) or start_time not in (None, 0) or \
                end_time not in (None, np.inf):
            self.unsupported = True
            raise _Unsupported(f"{name} with k={k}, a time window or a number of rewards other than k")

        return order

    def moment(self, k: int = 1, rewards: Sequence = None, start_time: float = None, end_time: float = None,
               center: bool = True, permute: bool = True) -> Any:
        """
        Serve a moment of :meth:`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, see
        :meth:`_CoalescentRecorder.order() <phasegen.gradient._CoalescentRecorder.order>`.
        """
        coal = self._coal
        k = self.order(k, rewards, start_time, end_time, 'moment')

        # the distribution carries the reward of a first moment as its own
        if k == 1:
            return self.serve(lambda: _Read(coal._get_dist(1, rewards), 'mean', None, coal.moment(1, rewards=rewards)))

        return self.serve(lambda: _Read(
            coal._get_dist(k, rewards), 'moment', (k, None if rewards is None else tuple(rewards), center, permute),
            coal.moment(k, rewards=rewards, center=center, permute=permute)
        ))

    def __getattr__(self, attr: str) -> Any:
        """
        Wrap a supported distribution.

        :param attr: The attribute.
        :return: The wrapped distribution.
        :raises _Unsupported: For any other attribute.
        """
        if attr not in _DISTRIBUTIONS:
            self.unsupported = True
            raise _Unsupported(attr)

        return _DistributionRecorder(getattr(self._coal, attr), self, attr)


class _Host(NamedTuple):
    """
    The derivatives of the statistic by the quantities of the epochs of one state space: the epochs, demes, pairs and
    parameters of the quantities, and the derivatives of each entry of the statistic in the layout of
    :meth:`DemographyGradient.flat() <phasegen.distributions.DemographyGradient.flat>`.
    """
    indices: List[int]
    pops: List[str]
    pairs: List[Tuple[str, str]]
    names: List[str]
    flat: np.ndarray


class Gradient:
    r"""
    Exact Jacobian of a statistic of a parametrized coalescent with respect to its parameters,

    .. math::
        J_{j i} = \frac{\partial s_j\big(\mathrm{coal}(\theta)\big)}{\partial \theta_i},

    where :math:`s` is the statistic, a function of the coalescent returning a number or an array, such as the
    negative log-likelihood of :class:`~phasegen.norms.PoissonLikelihood` of an observed SFS given the mean of
    :attr:`Coalescent.sfs <phasegen.distributions.Coalescent.sfs>`, or that mean normalized to sum to one with
    :meth:`Spectrum.normalize() <sfsutils.spectrum.Spectrum.normalize>`. :math:`s_j` is its :math:`j`-th element, a
    single one for a number, :math:`\theta_i` the :math:`i`-th parameter, :math:`\mathrm{coal}(\theta)` the coalescent
    returned by the ``coal`` callback and :math:`J_{j i}` the entry of the Jacobian in row :math:`j` and column
    :math:`i`. The statistic may read the moments of :attr:`Coalescent.sfs <phasegen.distributions.Coalescent.sfs>`,
    :attr:`Coalescent.fsfs <phasegen.distributions.Coalescent.fsfs>`, :attr:`Coalescent.jsfs
    <phasegen.distributions.Coalescent.jsfs>`, :attr:`Coalescent.tree_height
    <phasegen.distributions.Coalescent.tree_height>` and :attr:`Coalescent.total_branch_length
    <phasegen.distributions.Coalescent.total_branch_length>`: the mean, the variance and standard deviation, the raw
    moments ``m2`` to ``m4``, the covariance and correlation of the bins of a spectrum, and their moments of any order,
    as from :meth:`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, each with one reward per order and
    no time window, and the mean of :attr:`Coalescent.sfs2 <phasegen.distributions.Coalescent.sfs2>`. The loss of an :class:`~phasegen.inference.Inference` is such a statistic. As a function of the parameters, the
    statistic is the composition

    .. math::
        s_j(\theta) = s_j\big(\mu(q(\theta))\big),

    where :math:`q` are the quantities that define the coalescent: the population size of each deme and the migration
    rate between each pair of demes in each epoch, the start time of each epoch, the parameters of the coalescent
    model and the recombination rate. The :math:`\mu` are the entries of all moments the statistic reads, such as the
    bins of a mean SFS. By the chain rule,

    .. math::
        J_{j i} = \sum_l \frac{\partial s_j}{\partial \mu_l} \sum_k \frac{\partial \mu_l}{\partial q_k}
        \frac{\partial q_k}{\partial \theta_i},

    where :math:`l` runs over the entries :math:`\mu_l` of the moments and :math:`k` over the quantities :math:`q_k`.
    The factor :math:`\partial \mu_l / \partial q_k` is exact, from :meth:`PhaseTypeDistribution.gradient()
    <phasegen.distributions.PhaseTypeDistribution.gradient>`. The other two are central differences of a function
    :math:`f` of one variable :math:`x` with step :math:`h`,

    .. math::
        f'(x) \approx \frac{f(x + h) - f(x - h)}{2 h},

    which are one-sided of second order at a bound or next to a value at which :math:`f` is invalid. For
    :math:`\partial s_j / \partial \mu_l`, the statistic is evaluated on the computed moments with one entry shifted,
    which needs no further moments. For :math:`\partial q_k / \partial \theta_i`, the callback builds the coalescent at
    shifted parameters, and only its quantities :math:`q_k` are read. The Jacobian costs a few evaluations of the
    moments, whatever the number of parameters. A parameter that changes the coalescent otherwise is differentiated by
    such a difference of the statistic, and a statistic without an exact gradient by such differences altogether, as
    noted in the log.

    ::

        gradient = pg.Gradient(
            coal=lambda t, Ne: pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})),
            statistic=lambda coal: coal.sfs.mean.normalize().polymorphic
        )

        value, jacobian = gradient.compute(dict(t=0.3, Ne=0.5))
    """

    def __init__(self, coal: Callable[..., Coalescent], statistic: Callable[[Coalescent], Any]) -> None:
        """
        Configure the Jacobian.

        :param coal: Callback returning the coalescent for keyword parameters.
        :param statistic: Function of the coalescent returning a number, an array or a spectrum.
        """
        #: Callback returning the coalescent for keyword parameters.
        self.coal: Callable[..., Coalescent] = coal

        #: Function of the coalescent returning the statistic.
        self.statistic: Callable[[Coalescent], Any] = statistic

    @property
    def _log(self) -> logging.Logger:
        """
        The logger.
        """
        return logging.getLogger(f"phasegen.{type(self).__name__}")

    def _entries(self, coal: Any) -> np.ndarray:
        """
        The entries of the statistic.

        :param coal: The coalescent or its recorder.
        :return: The entries, the data of a spectrum.
        """
        value = self.statistic(coal)

        return np.asarray(getattr(value, 'data', value), dtype=float)

    @BlasThreadLimit()
    def compute(
            self,
            params: Dict[str, float],
            bounds: Dict[str, Tuple[float, float]] = None,
            exact: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The statistic and its Jacobian at given parameters.

        :param params: The parameters.
        :param bounds: Bounds of the parameters, which the differences taken through the callback respect. By default,
            the parameters are unbounded. A parameter whose bounds coincide has a derivative of zero.
        :param exact: Whether to try the exact Jacobian, which a statistic without one replaces by differences.
        :return: The entries of the statistic, and the Jacobian of shape ``entries.shape + (len(params),)``, with the
            parameters in the order of ``params``.
        """
        bounds = bounds or {}
        out = self._exact(params, bounds) if exact else None

        if out is not None:
            return out

        if exact:
            self._log.info('The statistic has no exact gradient, using finite differences.')

        x = np.array(list(params.values()), dtype=float)
        value = self._entries(self.coal(**params))
        jacobian = self._differences(lambda y: self._entries(self.coal(**dict(zip(params, y)))).ravel(), x,
                                     [bounds.get(key, (-np.inf, np.inf)) for key in params])

        return value, jacobian.reshape(value.shape + (len(params),))

    @staticmethod
    def _differences(f: Callable[[np.ndarray], Any], x: np.ndarray, bounds: Sequence[Tuple[float, float]],
                     steps: np.ndarray = None, f0: Any = None) -> np.ndarray:
        """
        The Jacobian of a function by the differences of :meth:`Gradient._stencil` over each parameter, zero for a
        parameter whose bounds coincide or at which ``f`` is invalid on both sides.

        :param f: The function of the parameters, returning a number or a vector, or ``None`` where it is invalid.
        :param x: The parameters.
        :param bounds: Their bounds.
        :param steps: The step of each parameter, by default :data:`_DEFAULT_STEP` times its magnitude (at least 1).
        :param f0: The value at ``x``, evaluated if not given.
        :return: The Jacobian, of shape ``(len(x),)`` for a number and ``(len(f(x)), len(x))`` for a vector.
        :raises ValueError: The last error of ``f``, or a :class:`ZeroDivisionError`, if it raises on both sides of a
            parameter.
        """
        value = np.asarray(f(x) if f0 is None else f0, dtype=float)
        steps = _DEFAULT_STEP * np.maximum(np.abs(x), 1) if steps is None else steps
        jacobian = np.zeros(value.shape + (len(x),))

        for i, (lo, hi) in enumerate(bounds):
            if lo == hi:
                continue

            def g(y: float) -> Any:
                z = x.copy()
                z[i] = y
                return f(z)

            stencil = Gradient._stencil(g, x[i], lo, hi, steps[i], value)
            if stencil is not None:
                values, coefficients, dx = stencil
                jacobian[..., i] = sum(c * np.asarray(v, dtype=float) for c, v in zip(coefficients, values)) / dx

        return jacobian

    @staticmethod
    def _stencil(f: Callable[[float], Any], x: float, lo: float, hi: float, h: float,
                 f0: Any) -> Optional[Tuple[List[Any], Tuple[int, ...], float]]:
        r"""
        A difference of a function of one parameter,

        .. math::
            f'(x) \approx \frac{1}{d} \sum_k c_k f(x_k),

        central over :math:`x_k = x \pm h` with :math:`c = (-1, 1)`, or one-sided of second order over
        :math:`x, x + s, x + 2s` with :math:`c = (-3, 4, -1)` where a side of the central difference leaves the bounds
        :math:`[l, u]` or ``f`` is invalid there, toward the other side first and with :math:`|s| \leq h` within the
        bounds. Here :math:`d` is the distance between the outermost points. ``f`` is invalid where it returns ``None``
        or raises one of :data:`_INVALID_ERRORS`.

        :param f: The function.
        :param x: The parameter :math:`x`.
        :param lo: The lower bound :math:`l`.
        :param hi: The upper bound :math:`u`.
        :param h: The step :math:`h`.
        :param f0: The value of ``f`` at ``x``.
        :return: The values :math:`f(x_k)`, the coefficients :math:`c_k` and :math:`d`, or ``None`` if ``f`` is invalid
            on both sides without raising.
        :raises ValueError: The last error of ``f``, or a :class:`ZeroDivisionError`, if it is invalid on both sides and
            raised.
        """
        cache, error = {}, None

        def value(y: float) -> Any:
            nonlocal error
            if y not in cache:
                try:
                    cache[y] = f(y)
                except _INVALID_ERRORS as e:
                    cache[y], error = None, e
            return cache[y]

        if lo <= x - h and x + h <= hi:
            up, down = value(x + h), value(x - h)
            if down is not None and up is not None:
                return [down, up], (-1, 1), (x + h) - (x - h)
            sides = (1, -1) if down is None else (-1, 1)
        else:
            sides = (1, -1) if hi - x >= x - lo else (-1, 1)

        for sign in sides:
            s = sign * min(h, (hi - x if sign > 0 else x - lo) / 2)
            if x + s == x:
                continue
            f1, f2 = value(x + s), value(x + 2 * s)
            if f1 is not None and f2 is not None:
                return [f0, f1, f2], (-3, 4, -1), (x + 2 * s) - x

        if error is not None:
            raise error

        return None

    def supported(self, params: Dict[str, float]) -> bool:
        """
        Whether the statistic reads only supported moments, evaluated at some parameters. A model that cannot be
        evaluated there leaves the question open, which is answered in favour of the exact Jacobian.

        :param params: The parameters.
        :return: Whether the statistic is supported.
        """
        try:
            return self._record(self.coal(**params)) is not None
        except Exception:
            return True

    def _record(self, coal: Coalescent) -> Optional[Tuple[np.ndarray, _CoalescentRecorder]]:
        """
        Evaluate the statistic through a recorder.

        :param coal: The coalescent.
        :return: The entries and the recorder, or ``None`` if the statistic reads an unsupported quantity, a windowed
            moment, or no moment.
        """
        recorder = _CoalescentRecorder(coal)

        # a statistic from another language, such as R, raises its own error in place of the signal, which the flag of
        # the recorder identifies
        try:
            value = self._entries(recorder)
        except BaseException as e:
            if isinstance(e, _Unsupported) or recorder.unsupported:
                self._log.debug("The statistic reads %s, which the exact gradient does not support", e)
                return None
            raise

        if recorder.unsupported or not recorder.reads:
            return None

        if any(read.dist._resolve_window() != (0.0, np.inf) for read in recorder.reads):
            return None

        return value, recorder

    def _exact(self, params: Dict[str, float],
               bounds: Dict[str, Tuple[float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        The statistic and its exact Jacobian.

        :param params: The parameters.
        :param bounds: The bounds of the parameters.
        :return: The entries and the Jacobian, or ``None`` where the statistic reads an unsupported quantity, a moment
            is windowed, an entry is not finite, the last epoch has no closed form, the generator is not affine in the
            rates, or the value of the backward pass differs from the moments.
        """
        coal = self.coal(**params)
        record = self._record(coal)
        if record is None:
            return None

        value, recorder = record
        if not np.isfinite(value).all():
            return None

        n_entries = value.size

        # Each entry of a moment is a weighted sum of ordered raw moments, which are linear in their first reward. The
        # orderings, weighted by the derivatives of each entry of the statistic, are grouped per host into backward
        # passes by _group_terms.
        hosts: Dict[int, list] = {}
        contraction, magnitude = np.zeros(n_entries), np.zeros(n_entries)
        for i, read in enumerate(list(recorder.reads)):
            host, entries = self._read_entries(read)
            try:
                D = self._moment_derivatives(recorder, i, read.value, [entry[0] for entry in entries])
            except BaseException as e:
                if isinstance(e, _Unsupported) or recorder.unsupported:
                    self._log.debug("The statistic reads %s at a perturbed moment, which the exact gradient does not "
                                    "support", e)
                    return None
                raise

            terms = hosts.setdefault(id(host), [host, []])[1]
            for l, (_, entry_terms, check, scale) in enumerate(entries):
                contraction += D[:, l] * check
                magnitude += np.abs(D[:, l]) * scale
                terms.extend((coefficient * D[:, l], vectors) for coefficient, vectors in entry_terms)

        adjoint = np.zeros(n_entries)
        derivatives = []
        for host, terms in hosts.values():
            try:
                _generator_basis(host.state_space)
            except ValueError as e:
                self._log.info("%s Using finite differences.", e)
                return None

            groups, means = _group_terms(terms)
            outs = [CrossMomentGradient(host).compute(heads, tail, seeds) for tail, heads, seeds in groups]
            if means is not None:
                used = np.flatnonzero(np.abs(means).sum(axis=0))
                out = MeanGradient(host).compute_columns(means[:, used])
                outs.append(None if out is None else [out[list(used).index(j)] if j in used else None
                                                      for j in range(n_entries)])

            for out in outs:
                if out is None:
                    return None

                results = [(j, r) for j, r in enumerate(out) if r is not None]
                if not results:
                    continue

                g = results[0][1][1]
                flat = np.zeros((n_entries, len(g.flat())))
                for j, (v, grad) in results:
                    adjoint[j] += v
                    flat[j] = grad.flat()

                derivatives.append(_Host([e.index for e in g.epochs], g.pops, g.pairs, list(g.parameters), flat))

        # the contraction of the derivatives of the statistic with the moments, against its magnitude, which
        # cancellation leaves far above the contraction for a statistic invariant to the scale of the moments
        if np.any(np.abs(adjoint - contraction) > _VALUE_RTOL * np.maximum(magnitude, 1e-300)):
            self._log.warning("The value of the backward pass differs from the moments, using finite differences.")
            return None

        jacobian = self._chain(params, bounds, coal, derivatives, value.shape)

        return value, jacobian

    @staticmethod
    def _read_entries(read: _Read) -> Tuple[Any, List[Tuple[Tuple[int, ...], list, float, float]]]:
        """
        The entries of a moment read, each with the ordered raw moments whose derivatives give its derivatives, see
        :meth:`PhaseTypeDistribution._moment_entries()
        <phasegen.distributions.PhaseTypeDistribution._moment_entries>`.

        :param read: The read.
        :return: The distribution whose state space carries the reward vectors, and the position, terms, weighted
            value and sum of the moduli of the weighted values of each entry.
        """
        if read.kind == 'mean':
            return read.dist._mean_entries(read.args, read.value)

        if read.kind == 'cov':
            return read.dist, read.dist._cov_entries()

        k, rewards, center, permute = read.args

        return read.dist, read.dist._moment_entries(k, rewards, center, permute, value=read.value)

    def _moment_derivatives(self, recorder: _CoalescentRecorder, i: int, moment: Any,
                            positions: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Central differences of the entries of the statistic over the entries of a moment.

        :param recorder: The recorder of the coalescent.
        :param i: The number of the read.
        :param moment: The moment.
        :param positions: The positions of its entries.
        :return: The derivatives, of shape ``(n_entries_statistic, len(positions))``.
        :raises _Unsupported: If the statistic reads other moments at a perturbed moment.
        """
        n_reads = len(recorder.reads)

        # an entry of zero takes its step from the largest entry
        scale = max(abs(self._entry(moment, pos)) for pos in positions)
        out = []
        for pos in positions:
            x = self._entry(moment, pos)
            h = _MOMENT_STEP * max(abs(x), 1e-3 * scale, 1e-300)
            values = []
            for sign in (1, -1):
                recorder.overrides[i] = self._with_entry(moment, pos, x + sign * h)
                recorder.reads = []
                values.append(self._entries(recorder).ravel())

                # the numbering of the reads holds only for a statistic that reads the same moments in each evaluation
                if len(recorder.reads) != n_reads:
                    raise _Unsupported("a number of moments that depends on their values")
            out.append((values[0] - values[1]) / (2 * h))

        recorder.overrides.pop(i, None)

        return np.array(out).T

    @staticmethod
    def _entry(moment: Any, pos: Tuple[int, ...]) -> float:
        """
        An entry of a moment.

        :param moment: The moment, a scalar or a spectrum.
        :param pos: The position, an empty tuple for a scalar.
        :return: The entry.
        """
        return float(moment) if pos == () else float(getattr(moment, 'data', moment)[pos])

    @staticmethod
    def _with_entry(moment: Any, pos: Tuple[int, ...], x: float) -> Any:
        """
        A copy of a moment with one entry replaced.

        :param moment: The moment.
        :param pos: The position.
        :param x: The entry.
        :return: The copy.
        """
        if pos == ():
            return x

        out = copy.deepcopy(moment)
        getattr(out, 'data', out)[pos] = x

        return out

    def _chain(self, params: Dict[str, float], bounds: Dict[str, Tuple[float, float]], coal: Coalescent,
               hosts: List[_Host], shape: Tuple[int, ...]) -> np.ndarray:
        """
        The Jacobian over the parameters from the derivatives over the quantities of the coalescent.

        :param params: The parameters.
        :param bounds: Their bounds.
        :param coal: The coalescent at the parameters.
        :param hosts: The derivatives over the quantities of each host.
        :param shape: The shape of the statistic.
        :return: The Jacobian, of shape ``shape + (len(params),)``.
        """
        jacobian = np.zeros((int(np.prod(shape)), len(params)))

        # the quantities of the moments, located by epoch index in the coalescent the callback builds
        q0 = [self._quantities(coal, h) for h in hosts]
        names = sorted(set(itertools.chain.from_iterable(h.names for h in hosts)))

        for i, (key, x) in enumerate(params.items()):
            lo, hi = bounds.get(key, (-np.inf, np.inf))
            if lo == hi:
                continue

            coals, coefficients, dx = self._stencil(lambda y: self.coal(**(params | {key: y})), x, lo, hi,
                                                    _PARAMETER_STEP * max(abs(x), 1.0), coal)

            if all(self._same_but_rates(coal, c, names) for c in coals):
                try:
                    q = [[self._quantities(c, host) for host in hosts] for c in coals]
                except (IndexError, KeyError):
                    q = None

                # a perturbation that moves an epoch across an index changes the quantities by far more than the step
                if q is not None and all(np.all(np.abs(qi[k] - q0[k]) <= 1e-3 * (np.abs(q0[k]) + 1))
                                         for qi in q for k in range(len(hosts))):
                    jacobian[:, i] = sum(host.flat @ sum(c * qi[k] for c, qi in zip(coefficients, q))
                                         for k, host in enumerate(hosts)) / dx
                    continue

            jacobian[:, i] = sum(c * self._entries(ci).ravel() for c, ci in zip(coefficients, coals)) / dx

        return jacobian.reshape(tuple(shape) + (len(params),))

    @staticmethod
    def _quantities(coal: Coalescent, host: _Host) -> np.ndarray:
        """
        The quantities of a coalescent over the epochs and parameters of a host.

        :param coal: The coalescent.
        :param host: The host.
        :return: The quantities.
        :raises IndexError: If the demography has fewer epochs.
        """
        epochs = list(itertools.islice(coal.demography.epochs, max(host.indices) + 1))

        return DemographyGradient.quantities([epochs[k] for k in host.indices], host.pops, host.pairs,
                                             _rate_parameters(coal.model, coal.locus_config), host.names)

    @staticmethod
    def _same_but_rates(a: Coalescent, b: Coalescent, names: List[str]) -> bool:
        """
        Whether two coalescents differ at most in their demography and in the named parameters of the coalescent
        model and the recombination rate.

        :param a: The first coalescent.
        :param b: The second coalescent.
        :param names: The parameters that may differ.
        :return: Whether they do.
        """
        if not all(getattr(a, k) == getattr(b, k) for k in ('lineage_config', 'start_time', 'end_time')):
            return False

        if not (a.lineage_distribution == b.lineage_distribution and a.locus_distribution == b.locus_distribution):
            return False

        if (a.locus_config.n, a.locus_config.n_unlinked) != (b.locus_config.n, b.locus_config.n_unlinked):
            return False

        theta_a = _rate_parameters(a.model, a.locus_config)
        theta_b = _rate_parameters(b.model, b.locus_config)

        if any(theta_a[name] != theta_b.get(name) and name not in names for name in theta_a):
            return False

        if a.model == b.model:
            return True

        # a model that differs is described by its parameters only if it can be rebuilt from them
        try:
            return type(a.model) is type(b.model) and a.model._with(**b.model._parameters()) == b.model
        except (ValueError, TypeError, ZeroDivisionError):
            return False
