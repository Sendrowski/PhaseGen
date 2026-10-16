"""
Gradient of an :class:`~phasegen.inference.Inference` loss with respect to its parameters, by the adjoint of the
first moments the loss reads.
"""

import copy
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .distributions import Coalescent
from .distributions._gradient import DemographyGradient, MeanGradient, _GeneratorBasis

#: The distributions of a :class:`~phasegen.distributions.Coalescent` whose first moments the gradient supports.
_DISTRIBUTIONS = ('sfs', 'fsfs', 'jsfs', 'tree_height', 'total_branch_length')

#: Relative step of the Richardson-extrapolated central differences of the loss over the entries of a mean, whose
#: truncation error is of fourth order in the step.
_LOSS_STEP = 1e-3

#: Relative step of the central differences of the demographic quantities over a parameter.
_PARAM_STEP = 1e-6

#: Relative tolerance between a mean and its value recomputed by the adjoint.
_VALUE_RTOL = 1e-9


class _Unsupported(BaseException):
    """
    The loss reads a quantity the gradient does not support. A :class:`BaseException`, so that a loss catching
    :class:`Exception` does not absorb it.
    """
    pass


class _Read:
    """
    A first moment the loss reads: the distribution whose state space carries it, the reward multiplied with the
    rewards of its entries (``None`` for the reward of the distribution), and its value.
    """

    def __init__(self, dist: Any, reward: Any, value: Any) -> None:
        """
        Record a read.

        :param dist: The distribution.
        :param reward: The reward, or ``None``.
        :param value: The value of the moment.
        """
        self.dist, self.reward, self.value = dist, reward, value


class _DistributionRecorder:
    """
    Stand-in for a distribution of a coalescent that serves its mean and its first moments, recording each read.
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
        Serve the mean.

        :param attr: The attribute.
        :return: The mean, possibly overridden.
        :raises _Unsupported: For any attribute other than the mean.
        """
        dist = self._distribution

        if attr == 'mean':
            return self._recorder.serve(lambda: (dist, None, dist.mean))

        self._recorder.unsupported = True
        raise _Unsupported(f"{self._name}.{attr}")

    def moment(self, k: int = 1, rewards: Any = None, start_time: float = None, end_time: float = None,
               center: bool = True, permute: bool = True) -> Any:
        """
        Serve a first moment.

        :param k: The order, which must be one.
        :param rewards: One reward or ``None``.
        :param start_time: Must be ``None``.
        :param end_time: Must be ``None``.
        :param center: Ignored, a first moment being its own mean.
        :param permute: Ignored.
        :return: The moment, possibly overridden.
        :raises _Unsupported: For another order or a time window.
        """
        reward = self._recorder.reward(k, rewards, start_time, end_time, f"{self._name}.moment")
        dist = self._distribution

        return self._recorder.serve(lambda: (dist, reward, dist.moment(1, rewards=rewards)))


class _CoalescentRecorder:
    """
    Stand-in for a :class:`~phasegen.distributions.Coalescent` passed to a loss, admitting only first moments: the
    means and first moments of :data:`_DISTRIBUTIONS` and :meth:`Coalescent.moment` of order one. The reads of one
    evaluation are numbered in order, so that a moment can be overridden in the next evaluation.
    """

    def __init__(self, coal: Coalescent) -> None:
        """
        Wrap a coalescent.

        :param coal: The coalescent.
        """
        self._coal = coal

        #: The reads of the current evaluation, in order.
        self.reads: List[_Read] = []

        #: The reads of the first evaluation, by number.
        self._first: Dict[int, _Read] = {}

        #: Moments served in place of the computed ones, by the number of the read.
        self.overrides: Dict[int, Any] = {}

        #: Whether the loss read an unsupported quantity.
        self.unsupported: bool = False

    def serve(self, make: Callable[[], Tuple[Any, Any, Any]]) -> Any:
        """
        Record a read and serve its moment.

        :param make: Returns the distribution, the reward and the value of the moment, called at the first evaluation
            only.
        :return: The moment, possibly overridden.
        """
        i = len(self.reads)
        if i not in self._first:
            self._first[i] = _Read(*make())

        self.reads.append(self._first[i])

        return self.overrides.get(i, self._first[i].value)

    def reward(self, k: Any, rewards: Any, start_time: float, end_time: float, name: str) -> Any:
        """
        The reward of a first moment, validating its arguments.

        :param k: The order.
        :param rewards: One reward or ``None``.
        :param start_time: The start time.
        :param end_time: The end time.
        :param name: The quantity, for the signal.
        :return: The reward, or ``None`` for that of the distribution.
        :raises _Unsupported: For another order, more rewards or a time window.
        """
        if k != 1 or (rewards is not None and len(rewards) != 1) or start_time is not None or end_time is not None:
            self.unsupported = True
            raise _Unsupported(f"{name} with k={k}, a time window or several rewards")

        return None if rewards is None else rewards[0]

    def moment(self, k: int = 1, rewards: Any = None, start_time: float = None, end_time: float = None,
               center: bool = True, permute: bool = True) -> Any:
        """
        Serve a first moment of :meth:`Coalescent.moment`.

        :param k: The order, which must be one.
        :param rewards: One reward or ``None``.
        :param start_time: Must be ``None``.
        :param end_time: Must be ``None``.
        :param center: Ignored, a first moment being its own mean.
        :param permute: Ignored.
        :return: The moment, possibly overridden.
        :raises _Unsupported: For another order or a time window.
        """
        self.reward(k, rewards, start_time, end_time, 'moment')
        coal = self._coal

        # the distribution carries the reward of the moment as its own
        return self.serve(lambda: (coal._get_dist(1, rewards), None, coal.moment(1, rewards=rewards)))

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


class LossGradient:
    r"""
    Value and gradient of a loss :math:`L(\boldsymbol{\mu}(\theta), y)` over the parameters :math:`\theta` of a
    coalescent callback, for a loss that reads only the entries :math:`\mu_j` of first moments of the coalescent (see
    :class:`_CoalescentRecorder`) and the observation :math:`y`. By the chain rule,

    .. math::
        \frac{\partial L}{\partial \theta_i} = \sum_j \frac{\partial L}{\partial \mu_j}
        \sum_k \frac{\partial \mu_j}{\partial q_k} \frac{\partial q_k}{\partial \theta_i},

    where :math:`q_k` are the population sizes, migration rates and start times of the epochs. The derivative
    :math:`\partial L / \partial \mu_j` is a central difference of the loss, which needs no model evaluation, the
    contraction :math:`\sum_j (\partial L / \partial \mu_j)\, \partial \mu_j / \partial q_k` is one adjoint pass of
    :class:`~phasegen.distributions._gradient.MeanGradient` per distribution carrying the occupation times, and
    :math:`\partial q_k / \partial \theta_i` is a central difference of the demography the callback builds. The
    quantities :math:`q_k` include the parameters of the coalescent model and the recombination rate. A parameter that
    changes the coalescent otherwise is differentiated by a central difference of the loss.
    """

    def __init__(self, get_dist: Callable[..., Coalescent], loss: Callable[[Coalescent, Any], float],
                 observation: Any) -> None:
        """
        Configure the gradient.

        :param get_dist: Callback returning the coalescent for keyword parameters.
        :param loss: The loss.
        :param observation: The observation passed to the loss.
        """
        self.get_dist, self.loss, self.observation = get_dist, loss, observation

    @property
    def _log(self) -> logging.Logger:
        """
        The logger.
        """
        return logging.getLogger(f"phasegen.{type(self).__name__}")

    def supported(self, params: Dict[str, float]) -> bool:
        """
        Whether the loss reads only supported means, evaluated at some parameters. A model that cannot be evaluated
        there leaves the question open, which is answered in favour of the exact gradient.

        :param params: The parameters.
        :return: Whether the loss is supported.
        """
        try:
            return self._record(self.get_dist(**params)) is not None
        except Exception:
            return True

    def _record(self, coal: Coalescent) -> Optional[Tuple[float, _CoalescentRecorder]]:
        """
        Evaluate the loss through a recorder.

        :param coal: The coalescent.
        :return: The loss and the recorder, or ``None`` if the loss reads an unsupported quantity, a windowed mean, or
            no mean.
        """
        recorder = _CoalescentRecorder(coal)

        # a loss from another language, such as R, raises its own error in place of the signal, which the flag of the
        # recorder identifies
        try:
            value = float(self.loss(recorder, self.observation))
        except BaseException as e:
            if isinstance(e, _Unsupported) or recorder.unsupported:
                self._log.debug("Loss reads %s, which the exact gradient does not support", e)
                return None
            raise

        if recorder.unsupported or not recorder.reads:
            return None

        if any(read.dist._resolve_window() != (0.0, np.inf) for read in recorder.reads):
            return None

        return value, recorder

    def compute(self, params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Optional[
        Tuple[float, np.ndarray]]:
        """
        The loss and its gradient.

        :param params: The parameters.
        :param bounds: The bounds of the parameters, which the differences of the callback respect.
        :return: The loss and its gradient over the parameters in the order of ``params``, or ``None`` where the loss
            reads a quantity other than a supported mean, a supported mean is windowed, the loss is not finite, or the
            occupation times are not defined in closed form.
        """
        coal = self.get_dist(**params)
        record = self._record(coal)
        if record is None:
            return None

        value, recorder = record
        if not np.isfinite(value):
            return None

        # the reward of every moment read, weighted by the derivatives of the loss, summed per host distribution,
        # since the moment is linear in the reward and one adjoint pass serves every distribution on a host
        hosts: Dict[int, list] = {}
        for i, read in enumerate(list(recorder.reads)):
            host, R, positions = read.dist._mean_rewards(read.reward)
            try:
                dmu = self._loss_derivatives(recorder, i, read.value, positions)
            except BaseException as e:
                if isinstance(e, _Unsupported) or recorder.unsupported:
                    self._log.debug("Loss reads %s at a perturbed moment, which the exact gradient does not support", e)
                    return None
                raise

            terms = np.array([d * self._entry(read.value, pos) for d, pos in zip(dmu, positions)])
            entry = hosts.setdefault(id(host), [host, 0.0, np.zeros(0)])
            entry[1] = entry[1] + R @ dmu
            entry[2] = np.concatenate([entry[2], terms])

        grad_q = None
        for host, reward, terms in hosts.values():
            out = MeanGradient(host).compute(reward)
            if out is None:
                return None

            # the contraction of the loss derivatives with the means, against its magnitude, which cancellation leaves
            # far above the contraction for a loss invariant to the scale of the mean
            adjoint_value, g = out
            if abs(adjoint_value - terms.sum()) > _VALUE_RTOL * np.abs(terms).sum():
                self._log.warning("The adjoint value %g differs from %g, using finite differences",
                                  adjoint_value, terms.sum())
                return None

            if grad_q is None:
                grad_q = g
            else:
                grad_q += g

        return value, self._chain(params, bounds, coal, grad_q)

    def _loss_derivatives(self, recorder: _CoalescentRecorder, i: int, mean: Any,
                          positions: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Richardson-extrapolated central differences of the loss over the entries of a moment.

        :param recorder: The recorder of the coalescent.
        :param i: The number of the read.
        :param mean: The moment.
        :param positions: The positions of the entries.
        :return: The derivatives.
        :raises _Unsupported: If the loss reads other moments at a perturbed moment.
        """
        n_reads = len(recorder.reads)
        # an entry of zero takes its step from the largest entry
        scale = max(abs(self._entry(mean, pos)) for pos in positions)
        out = np.zeros(len(positions))
        for j, pos in enumerate(positions):
            x = self._entry(mean, pos)
            h = _LOSS_STEP * max(abs(x), 1e-3 * scale, 1e-300)
            diff = []
            for step in (h, h / 2):
                values = []
                for sign in (1, -1):
                    recorder.overrides[i] = self._with_entry(mean, pos, x + sign * step)
                    recorder.reads = []
                    values.append(float(self.loss(recorder, self.observation)))

                    # the numbering of the reads holds only for a loss that reads the same moments in each evaluation
                    if len(recorder.reads) != n_reads:
                        raise _Unsupported("a number of moments that depends on their values")
                diff.append((values[0] - values[1]) / (2 * step))
            out[j] = (4 * diff[1] - diff[0]) / 3

        recorder.overrides.pop(i, None)

        return out

    @staticmethod
    def _entry(mean: Any, pos: Tuple[int, ...]) -> float:
        """
        An entry of a mean.

        :param mean: The mean, a scalar or a spectrum.
        :param pos: The position, an empty tuple for a scalar.
        :return: The entry.
        """
        return float(mean) if pos == () else float(mean.data[pos])

    @staticmethod
    def _with_entry(mean: Any, pos: Tuple[int, ...], x: float) -> Any:
        """
        A copy of a mean with one entry replaced.

        :param mean: The mean.
        :param pos: The position.
        :param x: The entry.
        :return: The copy.
        """
        if pos == ():
            return x

        out = copy.deepcopy(mean)
        out.data[pos] = x

        return out

    def _chain(self, params: Dict[str, float], bounds: Dict[str, Tuple[float, float]], coal: Coalescent,
               grad_q: DemographyGradient) -> np.ndarray:
        """
        The gradient over the parameters from the gradient over the demographic quantities.

        :param params: The parameters.
        :param bounds: Their bounds.
        :param coal: The coalescent at the parameters.
        :param grad_q: The gradient over the demographic quantities.
        :return: The gradient.
        """
        indices = [e.index for e in grad_q.epochs]
        flat = grad_q.flat()
        out = np.zeros(len(params))

        # the epochs of the moments, located by index in the demography the callback builds
        q0 = DemographyGradient.quantities(grad_q.epochs, grad_q.pops, grad_q.pairs)
        aligned = np.allclose(self._quantities(coal, indices, grad_q), q0, rtol=1e-12, atol=0)

        for i, (key, x) in enumerate(params.items()):
            lo, hi = bounds.get(key, (-np.inf, np.inf))
            h = _PARAM_STEP * max(abs(x), 1.0)
            up, down = min(x + h, hi), max(x - h, lo)

            coals = [self.get_dist(**(params | {key: y})) for y in (up, down)]
            if aligned and all(self._same_but_rates(coal, c, grad_q) for c in coals):
                try:
                    q = [self._quantities(c, indices, grad_q) for c in coals]
                except (IndexError, KeyError):
                    q = None

                # a perturbation that moves an epoch across an index changes the quantities by far more than the step
                if q is not None and all(np.all(np.abs(qi - q0) <= 1e-3 * (np.abs(q0) + 1)) for qi in q):
                    theta = [_GeneratorBasis.parameters(c.model, c.locus_config) for c in coals]
                    rates = sum(float(g) * (theta[0][name] - theta[1][name]) for name, g in grad_q.parameters.items())
                    out[i] = (float(flat @ (q[0] - q[1])) + rates) / (up - down)
                    continue

            losses = [float(self.loss(c, self.observation)) for c in coals]
            out[i] = (losses[0] - losses[1]) / (up - down)

        return out

    @staticmethod
    def _quantities(coal: Coalescent, indices: List[int], grad_q: DemographyGradient) -> np.ndarray:
        """
        The demographic quantities of a coalescent over the epochs of a gradient.

        :param coal: The coalescent.
        :param indices: The indices of the epochs.
        :param grad_q: The gradient.
        :return: The quantities.
        :raises IndexError: If the demography has fewer epochs.
        """
        epochs = list(itertools.islice(coal.demography.epochs, max(indices) + 1))
        return DemographyGradient.quantities([epochs[k] for k in indices], grad_q.pops, grad_q.pairs)

    @staticmethod
    def _same_but_rates(a: Coalescent, b: Coalescent, grad_q: DemographyGradient) -> bool:
        """
        Whether two coalescents differ at most in their demography and in the parameters of the coalescent model and
        the recombination rate that a gradient holds.

        :param a: The first coalescent.
        :param b: The second coalescent.
        :param grad_q: The gradient.
        :return: Whether they do.
        """
        if not all(getattr(a, k) == getattr(b, k) for k in ('lineage_config', 'start_time', 'end_time')):
            return False

        if type(a.model) is not type(b.model) or getattr(a.model, 'scale_time', None) != getattr(
                b.model, 'scale_time', None):
            return False

        if (a.locus_config.n, a.locus_config.n_unlinked) != (b.locus_config.n, b.locus_config.n_unlinked):
            return False

        theta_a = _GeneratorBasis.parameters(a.model, a.locus_config)
        theta_b = _GeneratorBasis.parameters(b.model, b.locus_config)

        return all(theta_a[name] == theta_b[name] or name in grad_q.parameters for name in theta_a)
