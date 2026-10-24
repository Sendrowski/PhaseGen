"""
Inference class for gradient-based parameter inference with respect to a specified loss function,
summary statistics, and a :class:`~phasegen.distributions.Coalescent` distribution.
"""

import copy
import hashlib
import logging
from collections import defaultdict
from .caching import cached_property
from typing import Dict, Tuple, Callable, Any, List, Literal, Iterable, Optional, TYPE_CHECKING

import dill
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from .demography import Demography
from ._blas import BlasThreadLimit
from .gradient import Gradient
from .distributions import Coalescent
from .distributions.coalescent import _StateSpaces
from .errors import ModelError
from .serialization import Serializable
from .settings import Settings
from .state_space import StateSpace
from .utils import parallelize

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .visualization import _CurveData

logger = logging.getLogger('phasegen')

#: Finite penalty substituted for a non-finite loss in a gradient-free optimization, and in a gradient-based one at its
#: first evaluation, large enough that any genuine loss wins the minimisation.
_LOSS_PENALTY = 1e100

#: Margin of the penalty of a gradient-based optimization above the loss of its current iterate, relative to the
#: magnitude of that loss (at least 1).
_PENALTY_MARGIN = 1e-6

#: Step of the differences of the loss over a parameter in a gradient-based optimization without the exact gradient,
#: relative to its magnitude (at least 1), or to its value for a parameter on a logarithmic scale.
_DIFFERENCE_STEP = 1e-6

#: Runs whose loss exceeds the best by at most this fraction of its magnitude (at least 1) count as reaching the optimum,
#: about a thousandfold the relative tolerance on the loss at which L-BFGS-B stops by default.
_RESTART_LOSS_RTOL = 1e-6

#: Estimates of runs reaching the optimum that differ by more than this fraction, or by more than this in the logarithm
#: for a parameter on a logarithmic scale, are reported as disagreeing.
_RESTART_PARAM_RTOL = 1e-2

#: The methods of `scipy.optimize.minimize` that honour bounds and need no analytic Jacobian.
_BOUNDED_METHODS = ('nelder-mead', 'powell', 'l-bfgs-b', 'tnc', 'slsqp', 'cobyla', 'cobyqa')

#: The methods of :data:`_BOUNDED_METHODS` that use a gradient.
_GRADIENT_METHODS = ('l-bfgs-b', 'tnc', 'slsqp')

#: Consecutive evaluations of an optimization without an exact gradient after which it takes finite differences only.
_MAX_GRADIENT_FAILURES = 3

#: Most parameters whose bootstrap distributions are drawn in a single column.
_BOOTSTRAP_COLUMN_MAX = 4

#: Height of the figure of bootstrap distributions drawn as subplots, relative to the default figure height.
_BOOTSTRAP_HEIGHT_SCALE = 0.75

#: Step of the central differences of a trajectory over a parameter, relative to its magnitude (at least 1).
_TRAJECTORY_STEP = 1e-6

#: Eigenvalues of the Fisher information below this fraction of the largest count as no curvature.
_CURVATURE_RTOL = 1e-10

#: Step of the central differences of the gradient that give the Fisher information, relative to the magnitude of
#: each parameter, large against the error of the gradient.
_HESSIAN_STEP = 1e-4

#: Spawn key of the generator that draws the resampled observations of the Godambe information from the entropy of the
#: inference, of length two so that it differs from the indices of ``create_run`` and ``create_bootstrap``.
_GODAMBE_SPAWN_KEY = (2 ** 32 - 1, 0)


class Inference(Serializable):
    r"""
    Gradient-based parameter inference with respect to a specified loss function,
    summary statistics, and a :class:`~phasegen.distributions.Coalescent` distribution.
    The optimization minimises the loss over the parameters :math:`\theta` (the entries of ``x0``,
    constrained to ``bounds``),

    .. math::
        \hat{\theta} = \arg\min_{\theta} L\big(\mathrm{coal}(\theta),\, y\big),

    where :math:`L` is the user-supplied ``loss`` function (any scalar objective, not necessarily a likelihood),
    :math:`\mathrm{coal}(\theta)` the coalescent distribution returned by the ``coal`` callback, and :math:`y` the
    observation. The minimisation is performed with a gradient-based scipy optimizer (L-BFGS-B by default),
    restarted from several initial points.

    .. versionadded:: 2.2.0
        The gradient of the loss is computed exactly where the loss reads supported moments (see
        :class:`~phasegen.gradient.Gradient`), at a cost largely independent of the number of parameters. This
        considerably speeds up inference, especially with many parameters.

    The following example infers the population size of a single deme from an observed site-frequency spectrum.

    ::

        inf = pg.Inference(
            bounds={'Ne': (0.1, 10)}, observation=pg.SFS([0, 8, 4, 3, 2, 0]),
            coal=lambda Ne: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
            loss=lambda coal, obs: pg.PoissonLikelihood().compute(obs.polymorphic, coal.sfs.mean.polymorphic)
        )
        inf.run()
    """
    #: Default options passed to the optimization algorithm.
    #: See the options of :ref:`L-BFGS-B <scipy:optimize.minimize-lbfgsb>`.
    default_opts = dict()

    #: Static for backward compatibility with serialized objects that lack the attribute.
    _entropy: int | None = None

    #: Static for backward compatibility with serialized objects whose initial guess was not materialized.
    _x0: Dict[str, float] | None = None

    #: Whether this object is a bootstrap replicate created by :meth:`create_bootstrap`, whose run records a replicate
    #: without an estimate with NaN parameters, as :meth:`bootstrap` does. Static for backward compatibility with
    #: serialized objects that lack the attribute.
    _is_bootstrap: bool = False

    #: How the gradient of the loss is obtained. Static for backward compatibility with serialized objects that lack the
    #: attribute.
    gradient: Literal['exact', 'finite_differences'] = 'exact'

    #: The parameters optimized on a logarithmic scale. Static for backward compatibility with serialized objects that
    #: lack the attribute.
    log_scale: Tuple[str, ...] = ()

    #: The penalty added to the loss. Static for backward compatibility with serialized objects that lack the
    #: attribute.
    penalty: Callable[..., float] | None = None

    #: The key of the inputs the Fisher information was last computed from, holding the ``loss`` and ``coal`` callables
    #: themselves, and the information. It is not serialized. Static for backward compatibility with serialized objects
    #: that lack the attribute.
    _fisher_information: Tuple[Tuple, pd.DataFrame] | None = None

    #: The key of the inputs the covariance of the score was last computed from, and the covariance. Static for
    #: backward compatibility with serialized objects that lack the attribute.
    _score_covariance_cache: Tuple[Tuple, pd.DataFrame] | None = None

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]],
            coal: Callable[..., Coalescent],
            loss: Callable[[Coalescent, Any], float],
            x0: Dict[str, float] = None,
            observation: Any = None,
            resample: Callable[[Any, np.random.Generator], Any] = None,
            n_runs: int = 10,
            n_bootstraps: int = 100,
            do_bootstrap: bool = False,
            parallelize: bool = False,
            pbar: bool = True,
            seed: int = None,
            cache: bool = True,
            opts: Dict = None,
            method_mle: str = 'L-BFGS-B',
            gradient: Literal['exact', 'finite_differences'] = 'exact',
            log_scale: bool | Iterable[str] = False,
            penalty: Callable[..., float] = None
    ) -> None:
        """
        Configure the inference.

        :param bounds: Dictionary of tuples representing the bounds for each
            parameter in x0. A parameter whose bounds coincide is held fixed.
        :param coal: Callback returning the configured coalescent distribution on which
            the inference is based. The parameters specified in ``x0`` and ``bounds``
            are passed as keyword arguments.
        :param loss: The loss function, evaluating :math:`L(\\theta)`. This function must return a single numerical
            value that is to be minimized. It receives as first argument the coalescent
            distribution returned by the ``coal`` callback, and as second argument the
            observation passed to the ``observation`` argument (if any). A typical choice aggregates a
            :class:`~phasegen.norms.Norm` or :class:`~phasegen.norms.Likelihood` over observed and modelled
            summary statistics (e.g. the :class:`~phasegen.norms.PoissonLikelihood` for site-frequency-spectrum counts).
        :param x0: Dictionary of initial numeric guesses for parameters to optimize.
        :param observation: The observed summary statistic the inference is based on.
            This is passed as second argument to the ``loss`` function. It is required for
            automatic bootstrapping.
        :param resample: Callback that resamples the observation. This is
            required for automatic bootstrapping. The resample function must accept
            the observation as first argument and a random number generator as second
            argument, and must return a resampled observation.
        :param n_runs: Number of independent optimization runs.
        :param n_bootstraps: Number of bootstrap replicates.
        :param do_bootstrap: Whether to perform automatic bootstrapping.
        :param parallelize: Whether to run the optimization runs and the bootstrap replicates in parallel, one worker
            process per CPU core. ``Settings.parallelize = False`` overrides it. The callbacks are serialized together
            with the variables they reference, so lambdas and functions defined in scripts or notebooks may be passed.
            On macOS the workers start by ``spawn``, which re-imports the calling script, so its top-level code must be
            guarded by ``if __name__ == '__main__':``. For a fast model, starting the workers can cost more than the
            parallel runs save. To distribute the work across jobs or machines, :meth:`Inference.create_run()
            <phasegen.inference.Inference.create_run>` and :meth:`Inference.create_bootstrap()
            <phasegen.inference.Inference.create_bootstrap>` create :class:`Inference` objects that run independently
            and are merged afterwards.
        :param pbar: Whether to show a progress bar.
        :param seed: Seed for the random number generator.
        :param cache: Whether to reuse the state spaces of :attr:`Coalescent.state_spaces
            <phasegen.distributions.Coalescent.state_spaces>` across optimization iterations when they are equivalent,
            so that only their rate matrices are rebuilt.
            This speeds up optimizations over demographic parameters such as population sizes or migration rates.
        :param opts: Additional options passed to the optimization algorithm.
            See the options of :ref:`L-BFGS-B <scipy:optimize.minimize-lbfgsb>`.
        :param method_mle: Method to use for optimization, a method of :func:`scipy.optimize.minimize` that honours
            bounds without an analytic Jacobian: :ref:`Nelder-Mead <scipy:optimize.minimize-neldermead>`,
            :ref:`Powell <scipy:optimize.minimize-powell>`, :ref:`L-BFGS-B <scipy:optimize.minimize-lbfgsb>`,
            :ref:`TNC <scipy:optimize.minimize-tnc>`, :ref:`SLSQP <scipy:optimize.minimize-slsqp>`,
            :ref:`COBYLA <scipy:optimize.minimize-cobyla>` or :ref:`COBYQA <scipy:optimize.minimize-cobyqa>`.
        :param gradient: How the gradient-based methods (L-BFGS-B, TNC and SLSQP) obtain the gradient of the loss. With
            ``'exact'``, a loss that reads only the moments listed at :class:`~phasegen.gradient.Gradient` is
            differentiated exactly (:meth:`PhaseTypeDistribution.gradient()
            <phasegen.distributions.PhaseTypeDistribution.gradient>`), at a cost of a few moment evaluations whatever
            the number of parameters. Other losses, and every loss with ``'finite_differences'``, are differentiated by
            central differences.
        :param log_scale: The name or names of the parameters optimized as their logarithms, ``True`` for all of
            them, which suits population sizes and rates spanning orders of magnitude and often reduces the number of
            iterations considerably. Their sampled start points are log-uniform within the bounds. The estimates,
            bounds and bootstraps stay on the original scale.
        :param penalty: A penalty :math:`P(\\theta)` added to the loss, which receives the parameters as keyword
            arguments and returns a number, such as a smoothness penalty on the population sizes of successive epochs.
            The optimization then minimises :math:`L + P`, and :attr:`Inference.loss_inferred
            <phasegen.inference.Inference.loss_inferred>` includes it. Its gradient is a central difference with the
            steps of the loss, which needs no moments, so the loss keeps its exact gradient.
        :raises ValueError: If a lower bound exceeds its upper bound, ``x0`` lies outside the bounds or does not specify
            exactly the parameters in ``bounds``, ``method_mle`` is not one of the supported methods, ``gradient`` is not
            one of the supported values, ``n_runs`` or ``n_bootstraps`` is less than 1, or a parameter on a logarithmic
            scale is unknown or has a lower bound that is not positive.
        """
        if do_bootstrap and (observation is None or resample is None):
            raise ValueError('Observation and resample arguments must be provided for automatic bootstrapping.')

        if int(n_runs) < 1 or int(n_bootstraps) < 1:
            raise ValueError(f'n_runs and n_bootstraps must be at least 1, got {n_runs} and {n_bootstraps}.')

        reversed_bounds = [key for key, (lower, upper) in bounds.items() if not lower <= upper]
        if reversed_bounds:
            raise ValueError(f'The lower bound exceeds the upper bound for parameters {reversed_bounds}.')

        try:
            opt.show_options(solver='minimize', method=method_mle, disp=False)
        except ValueError:
            raise ValueError(f'Unknown optimization method {method_mle!r}, see scipy.optimize.minimize.') from None

        if gradient not in ('exact', 'finite_differences'):
            raise ValueError(f"gradient must be 'exact' or 'finite_differences', got {gradient!r}.")

        log_names = tuple(bounds) if log_scale is True else (
            (log_scale,) if isinstance(log_scale, str) else tuple(log_scale or ()))
        invalid = [name for name in log_names if name not in bounds or not bounds[name][0] > 0]
        if invalid:
            raise ValueError(f'Parameters on a logarithmic scale must be in bounds with a positive lower bound, got '
                             f'{invalid}.')

        if method_mle.lower() not in _BOUNDED_METHODS:
            raise ValueError(
                f'Optimization method {method_mle!r} does not honour bounds without an analytic Jacobian. Use one '
                f'of {list(_BOUNDED_METHODS)}.'
            )

        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Dictionary of tuples representing the bounds for each parameter in x0.
        self.bounds: Dict[str, Tuple[float, float]] = bounds

        #: Callback returning the configured coalescent distribution.
        self.coal: Callable[..., Coalescent] = coal

        #: Loss function.
        self.loss: Callable[[Coalescent, Any], float] = loss

        #: The observed summary statistic the inference is based on.
        self.observation: Any = observation

        #: Callback that is used to resample the observation.
        self.resample: Callable[[Any, np.random.Generator], Any] = resample

        #: Number of optimization runs.
        self.n_runs: int = int(n_runs)

        #: Number of bootstrap replicates.
        self.n_bootstraps: int = int(n_bootstraps)

        #: Whether to perform automatic bootstrapping.
        self.do_bootstrap: bool = do_bootstrap

        #: Whether to parallelize the computations.
        self.parallelize: bool = parallelize

        #: Whether to show a progress bar.
        self.pbar: bool = pbar

        #: Seed for the random number generator.
        self.seed: int | None = None if seed is None else int(seed)

        #: Entropy of the random number generator, from which the generators of created runs and bootstraps derive.
        self._entropy: int = np.random.SeedSequence(self.seed).entropy

        #: Random number generator.
        self._rng: np.random.Generator = np.random.default_rng(np.random.SeedSequence(self._entropy))

        #: The parameters optimized on a logarithmic scale.
        self.log_scale: Tuple[str, ...] = log_names

        #: Dictionary of initial numeric guesses for parameters to optimize, sampled within the bounds if not given.
        self._x0: Dict[str, float] = self._sample() if x0 is None else x0

        #: Whether to cache the state spaces
        self.cache: bool = cache

        #: Optimization options
        self.opts: Dict = self.default_opts | (opts or {})

        #: Optimization method
        self.method_mle: str = method_mle

        #: How the gradient of the loss is obtained.
        self.gradient: Literal['exact', 'finite_differences'] = gradient

        #: The penalty added to the loss.
        self.penalty: Callable[..., float] | None = penalty

        #: Optimization result
        self.result: OptimizeResult | None = None

        #: Inferred parameters.
        self.params_inferred: Dict = {}

        #: Loss of the best optimization run
        self.loss_inferred: float | None = None

        #: Coalescent distribution of best run
        self.dist_inferred: Coalescent | None = None

        #: Bootstrap parameters
        self.bootstraps: pd.DataFrame = pd.DataFrame(columns=list(self.bounds.keys()) + ['loss', 'result'])

        #: Initial optimization runs
        self.runs: pd.DataFrame = self.bootstraps.copy()

        # an explicit start point outside the box wastes its run, so reject it here rather than at ``create_run``.
        # A sampled x0 is drawn inside the bounds by construction, so this only validates what the caller passed.
        if self._x0 is not None:
            self._check_x0_within_bounds()

    def _check_x0_within_bounds(self) -> None:
        """
        Check if the initial parameters are within the specified bounds.
        """
        if not all([self.bounds[key][0] <= value <= self.bounds[key][1] for key, value in self.x0.items()]):
            raise ValueError('Initial parameters must be within the specified bounds.')

    @property
    def param_names(self) -> List[str]:
        """
        Get the names of the parameters.
        """
        return list(self.bounds.keys())

    @property
    def x0(self) -> Dict[str, float]:
        """
        Initial parameters, sampled within the bounds when none were given.
        """
        if self._x0 is None:
            self._x0 = self._sample()

        # x0 must cover every bounds parameter: `_sample()`-generated runs always span all bounds keys, so a partial
        # x0 would make the first run optimize a lower-dimensional subspace than the rest (a ragged run set that
        # crashes or mislabels params). Fail early rather than silently drop the missing dimensions.
        missing = [key for key in self.bounds.keys() if key not in self._x0]
        if missing:
            raise ValueError(f"x0 must specify every parameter in bounds; missing: {missing}.")

        unknown = [key for key in self._x0 if key not in self.bounds]
        if unknown:
            raise ValueError(f"x0 must specify only parameters in bounds, got unknown parameters {unknown}.")

        # canonicalize to `bounds` key order so that every run's `result.x` (ordered by the passed x0's keys) lines
        # up with `self.x0.keys()` and the DataFrame columns; `_sample()`-generated runs are already in bounds order
        return {key: self._x0[key] for key in self.bounds.keys()}

    def __getstate__(self) -> dict:
        """
        Get the state of the object for serialization.

        The ``coal``, ``loss``, ``resample`` and ``penalty`` callables are serialized with ``dill`` (they are typically
        lambdas/closures that the standard pickler and ``copy.deepcopy`` cannot handle reliably, especially
        once they have themselves been restored from a previous ``dill`` round-trip). They are dumped
        directly from ``self`` without a prior deep copy. Only the remaining state is deep-copied, so the live object
        is left untouched. The cached Fisher information and score covariance, whose keys hold the callables, are left
        out.

        The dump is recursive, so that the module-level names a callable references (the package alias, the
        observation, helper functions) travel with it. A restored callable therefore evaluates against the
        namespace it was written with, in a worker process started by ``spawn`` and in a later session alike.

        :return: State of the object.
        """
        callables = ['coal', 'loss', 'resample', 'penalty']

        state = copy.deepcopy({
            key: value for key, value in self.__dict__.items() if key not in callables + ['_fisher_information', '_score_covariance_cache']
        })

        # a plain dict, as jsonpickle encodes the instance dictionary of an OptimizeResult as an additional item
        if state['result'] is not None:
            state['result'] = dict(state['result'])

        for key in callables:
            state[f'{key}_pickled'] = dill.dumps(getattr(self, key), recurse=True)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore the state of the object from a serialized state.

        :param state: State of the object.
        """
        self.__dict__.update(state)

        if '_state_spaces' in self.__dict__:
            self._state_spaces = _StateSpaces._rename_deprecated(self._state_spaces)

        # jsonpickle restores the instance dictionary of an encoded OptimizeResult as an item named '__dict__'
        if self.result is not None:
            self.result = OptimizeResult({k: v for k, v in dict(self.result).items() if k != '__dict__'})

        for key in ['coal', 'loss', 'resample', 'penalty']:
            if f'{key}_pickled' in state:
                setattr(self, key, dill.loads(state[f'{key}_pickled']))
                self.__dict__.pop(f'{key}_pickled')

    def get_coal(self, **kwargs) -> Coalescent:
        """
        Get the (possibly cached) coalescent distribution.

        :param kwargs: Keyword arguments passed to the callback specified as ``coal``.
        :return: Coalescent distribution.
        """
        coal = self.coal(**kwargs)

        # if state space caching is enabled, replace each state space by the cached one if it matches, set to the
        # first epoch of this coalescent as a freshly built state space is
        if self.cache:

            try:
                spaces = self._state_spaces
            except ModelError:
                # the model is invalid at x0, so the state spaces of this coalescent are the ones reused
                spaces = dict(coal.state_spaces)

                if Settings.cache:
                    self.__dict__['_state_spaces'] = spaces

            for name, cached in spaces.items():
                if coal.state_spaces[name] == cached:
                    cached.update_epoch(coal.demography.get_epoch(0))
                    coal.__dict__.setdefault('_built_state_spaces', {})[name] = cached

        return coal

    @cached_property
    def _state_spaces(self) -> Dict[str, StateSpace]:
        """
        The state spaces of the coalescent at ``x0``, reused across loss evaluations when caching is enabled. Only the
        rate matrices are recomputed per evaluation. If the model is invalid at ``x0``, :meth:`get_coal` sets them to
        those of the first coalescent it builds.
        """
        return dict(self.coal(**self.x0).state_spaces)

    @staticmethod
    def _get_loss_function(
            observation: Any,
            x0: Dict[str, float],
            pbar: tqdm | None,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float],
            logger: logging.Logger = logger,
            smooth: bool = False,
            penalty: Callable[..., float] = None
    ) -> '_PenalizedLoss':
        """
        Get the loss function that accepts a list as an argument, with a finite penalty where the model raises or the
        loss is not finite.

        :param observation: Observation.
        :param x0: Initial parameters.
        :param pbar: Progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param logger: Logger.
        :param smooth: Whether the penalty lies just above the loss of the current iterate, for a gradient-based
            optimizer.
        :param penalty: The penalty added to the loss, see :class:`Inference`.
        :return: Loss function.
        """
        return _PenalizedLoss(observation, x0, pbar, get_dist, get_loss, logger, smooth, penalty)

    @staticmethod
    def _penalty_gradient(
            penalty: Callable[..., float],
            params: Dict[str, float],
            bounds: Dict[str, Tuple[float, float]],
            log_scale: Iterable[str] = ()
    ) -> np.ndarray:
        """
        The gradient of a penalty by the differences of :meth:`Gradient._stencil()
        <phasegen.gradient.Gradient._stencil>`, with the steps of the loss.

        :param penalty: The penalty, see :class:`Inference`.
        :param params: The parameters.
        :param bounds: Their bounds.
        :param log_scale: The parameters optimized as their logarithms, whose steps are relative to their values.
        :return: The gradient, in the order of ``params``.
        """
        names = list(params)
        x = np.array(list(params.values()), dtype=float)
        log = np.array([name in log_scale for name in names], dtype=bool)
        steps = _DIFFERENCE_STEP * np.where(log, np.abs(x), np.maximum(np.abs(x), 1))

        return Gradient._differences(lambda y: penalty(**dict(zip(names, y))), x, [bounds[name] for name in names],
                                     steps)

    @staticmethod
    def _report(params: Dict[str, float], loss: float, pbar: tqdm | None, logger: logging.Logger) -> None:
        """
        Log an evaluation of the loss and show it on the progress bar.

        :param params: The parameters.
        :param loss: The loss.
        :param pbar: Progress bar.
        :param logger: Logger.
        """
        data = params | {'loss': loss}

        logger.debug(f"Current iteration: ({', '.join([f'{k}={v:.4f}' for k, v in data.items()])})")

        if pbar is not None:
            pbar.update()
            pbar.set_postfix(data)

    def _announce_gradient(self, x0: Dict[str, float], observation: Any) -> None:
        """
        Log how the optimizations take the gradient of the loss, at INFO level where it is exact and as a warning where
        the exact gradient was asked for but the loss is not supported at ``x0`` and ``observation``. Each optimization
        decides for itself at its own start point and observation.

        :param x0: The parameters at which the loss is probed.
        :param observation: The observation.
        """
        if self.gradient != 'exact' or self.method_mle.lower() not in _GRADIENT_METHODS:
            return

        reason = Gradient(self.get_coal, lambda coal: self.loss(coal, observation))._unsupported_reason(x0)
        if reason is None:
            self._logger.info('Using the exact gradient of the loss.')
        else:
            self._logger.warning('The loss has no exact gradient. %s Using finite differences.', reason)

    @staticmethod
    def _get_loss_and_gradient(
            loss: '_PenalizedLoss',
            x0: Dict[str, float],
            bounds: Dict[str, Tuple[float, float]],
            pbar: tqdm | None,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float],
            observation: Any,
            logger: logging.Logger = logger,
            exact: bool = True,
            log_scale: Iterable[str] = (),
            penalty: Callable[..., float] = None
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """
        The loss with its gradient, exact where ``exact`` is set and :class:`~phasegen.gradient.Gradient` supports the
        loss, and otherwise the finite differences described at :class:`Inference` with steps of
        :data:`_DIFFERENCE_STEP`. A penalized point has a zero gradient once the current iterate is valid, and otherwise
        the central differences of the penalized loss, which point from an invalid start toward the valid region.

        :param loss: The loss function of
            :meth:`Inference._get_loss_function() <phasegen.inference.Inference._get_loss_function>`.
        :param x0: Initial parameters, for their names.
        :param bounds: The bounds of the parameters.
        :param pbar: Progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param observation: Observation.
        :param logger: Logger.
        :param exact: Whether to try the exact gradient.
        :param log_scale: The parameters optimized as their logarithms.
        :param penalty: The penalty added to the loss, see :class:`Inference`, which ``loss`` includes.
        :return: Function returning the loss and its gradient.
        """
        gradient = Gradient(get_dist, lambda coal: get_loss(coal, observation))
        bounds_list = [bounds[k] for k in x0.keys()]
        log = np.array([key in log_scale for key in x0.keys()])

        # consecutive evaluations without an exact gradient, after which the remaining ones skip the attempt
        failures = 0 if exact else _MAX_GRADIENT_FAILURES

        def loss_and_gradient(params: np.ndarray) -> Tuple[float, np.ndarray]:
            nonlocal failures
            params_dict = dict(zip(x0.keys(), params))
            out = None
            if failures < _MAX_GRADIENT_FAILURES:
                try:
                    out = gradient._exact(params_dict, bounds)
                except Exception as e:
                    logger.debug('Exact gradient unavailable at %s (%s), using finite differences', params_dict, e)

            if out is not None and penalty is not None:
                out = (out[0] + penalty(**params_dict),
                       out[1] + Inference._penalty_gradient(penalty, params_dict, bounds, log_scale))

            if out is not None and np.isfinite(out[0]) and np.isfinite(out[1]).all():
                failures = 0
                loss.record(float(out[0]))
                Inference._report(params_dict, float(out[0]), pbar, logger)
                return float(out[0]), out[1]

            value = loss(params)

            if loss.invalid and loss.current < _LOSS_PENALTY:
                return value, np.zeros(len(params))

            failures += 1
            if failures == _MAX_GRADIENT_FAILURES:
                logger.warning(
                    'The exact gradient was unavailable at %d consecutive evaluations, the last at %s. Using finite '
                    'differences for the rest of this optimization.', failures, params_dict
                )

            steps = _DIFFERENCE_STEP * np.where(log, np.abs(params), np.maximum(np.abs(params), 1))

            return value, Gradient._differences(loss if loss.invalid else loss.evaluate, params, bounds_list, steps,
                                                value)

        return loss_and_gradient

    @staticmethod
    def _optimize(
            observation: Any,
            x0: Dict[str, float],
            bounds: Dict[str, Tuple[float, float]],
            show_pbar: bool,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float],
            opts: dict = None,
            method_mle: str = 'L-BFGS-B',
            logger: logging.Logger = logger,
            gradient: Literal['exact', 'finite_differences'] = 'exact',
            log_scale: Iterable[str] = (),
            penalty: Callable[..., float] = None
    ) -> OptimizeResult:
        """
        Perform the optimization over the parameters whose bounds differ, holding the others fixed.

        :param observation: Observation.
        :param x0: Initial parameters.
        :param bounds: Bounds for the parameters.
        :param show_pbar: Whether to show a progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param opts: Additional options passed to the optimization algorithm.
        :param method_mle: Method of `scipy.optimize.minimize` used for the optimization.
        :param logger: Logger.
        :param gradient: How the gradient of the loss is obtained, see :class:`Inference`.
        :param log_scale: The parameters optimized as their logarithms.
        :param penalty: The penalty added to the loss, see :class:`Inference`.
        :return: Result of the optimization procedure. The estimates ``x`` and the gradient of the loss ``jac``, zero for
            a fixed parameter, are on the original scale. With a parameter on a logarithmic scale or a fixed one,
            ``hess_inv`` is omitted, and the other solver diagnostics refer to the free parameters and the logarithms.
        """
        bounds_dict = {key: bounds[key] for key in x0.keys()}
        lower, upper = np.array(list(bounds_dict.values()), dtype=float).reshape(-1, 2).T
        free = lower < upper
        log = np.array([key in log_scale for key in x0.keys()])
        start = np.array(list(x0.values()), dtype=float)

        pbar = tqdm(desc='Optimizing loss function') if show_pbar else None

        # a gradient-based optimizer backtracks from a penalty just above its iterate, which a gradient-free one need not
        smooth = method_mle.lower() in _GRADIENT_METHODS
        penalized = loss = Inference._get_loss_function(
            observation=observation,
            x0=x0,
            pbar=pbar,
            get_dist=get_dist,
            get_loss=get_loss,
            logger=logger,
            smooth=smooth,
            penalty=penalty
        )

        if smooth:
            exact = gradient == 'exact' and Gradient(get_dist, lambda coal: get_loss(coal, observation)).supported(x0)
            loss = Inference._get_loss_and_gradient(loss, x0, bounds_dict, pbar, get_dist, get_loss, observation, logger,
                                                    exact, log_scale, penalty)

        # the optimizer sees the free parameters, with y = log(x) for those on a logarithmic scale, and dL/dy = x dL/dx,
        # with x clipped to its bounds
        def to_x(y: np.ndarray) -> np.ndarray:
            x = start.copy()
            x[free] = np.where(log[free], np.clip(np.exp(np.where(log[free], y, 0)), lower[free], upper[free]), y)
            return x

        def fun(y: np.ndarray) -> Any:
            x = to_x(y)
            if not smooth:
                return loss(x)
            value, grad = loss(x)
            return value, (grad * np.where(log, x, 1))[free]

        if free.any():
            with BlasThreadLimit():
                result: OptimizeResult = opt.minimize(
                    fun=fun,
                    x0=np.where(log, np.log(np.where(log, start, 1)), start)[free],
                    method=method_mle,
                    bounds=[(np.log(lo), np.log(hi)) if is_log else (lo, hi)
                            for lo, hi, is_log in zip(lower[free], upper[free], log[free])],
                    jac=smooth or None,
                    options=opts,
                    callback=penalized.update
                )
            result.x = to_x(result.x)

            if isinstance(result.get('jac'), np.ndarray) and result.jac.shape == (free.sum(),):
                jac = np.zeros(len(start))
                jac[free] = result.jac / np.where(log[free], result.x[free], 1)
                result.jac = jac

            if log.any() or not free.all():
                result.pop('hess_inv', None)
        else:
            result = OptimizeResult(x=start, fun=penalized(start), jac=np.zeros(len(start)), nit=0, nfev=1,
                                    success=True, message='All parameters are fixed by their bounds.')

        if show_pbar:
            pbar.close()

        return result

    def _run(self) -> OptimizeResult:
        """
        Execute the main optimization.

        :returns: Result of the optimization procedure.
        """
        observation = self.observation
        bounds = self.bounds
        get_dist = self.get_coal
        get_loss = self.loss
        opts = self.opts
        method_mle = self.method_mle
        gradient = self.gradient
        log_scale = self.log_scale
        penalty = self.penalty
        logger = self._logger

        self._announce_gradient(self.x0, observation)

        self._logger.debug(f"Using '{method_mle}' optimizer.")

        def run_sample(x0: Dict[str, float]) -> OptimizeResult:
            """
            Run a single bootstrap sample.

            :param x0: Initial parameters.
            :return: Bootstrap sample.
            """
            # isolate a single run's failure so one bad start point (an ill-conditioned demography, a raising
            # model evaluation) does not abort the whole multi-start; the non-converged sentinel is dropped by the
            # finite-loss filter in _run
            try:
                return self._optimize(
                    observation=observation,
                    x0=x0,
                    bounds=bounds,
                    show_pbar=False,
                    get_dist=get_dist,
                    get_loss=get_loss,
                    opts=opts,
                    method_mle=method_mle,
                    logger=logger,
                    gradient=gradient,
                    log_scale=log_scale,
                    penalty=penalty
                )
            except Exception as e:
                logger.warning(f'Optimization run from x0={x0} failed and was skipped: {e}')
                return OptimizeResult(x=np.array(list(x0.values()), dtype=float), fun=np.inf, success=False,
                                      message=str(e))

        results = parallelize(
            func=run_sample,
            data=[self.x0] + [self._sample() for _ in range(self.n_runs - 1)],
            parallelize=self.parallelize,
            pbar=self.pbar,
            desc='Optimizing',
            dtype=object
        )

        n_success = sum([result.success for result in results])

        if n_success < self.n_runs:
            self._logger.warning(
                f'Only {n_success} out of {self.n_runs} optimization runs converged.'
            )

        # get the best result, ignoring runs whose loss is non-finite (a NaN loss would otherwise never be
        # displaced by `min`, since both `x < NaN` and `NaN < x` are False)
        finite = [result for result in results if np.isfinite(result.fun)]

        # a bootstrap replicate whose optimization raised in every run has no estimate, as in bootstrap()
        if not finite and self._is_bootstrap:
            self._logger.warning('Bootstrap replicate failed and its parameters are NaN: %s', results[0].message)

            self.result = OptimizeResult(x=np.full(len(self.x0), np.nan), fun=np.inf, success=False,
                                         message=results[0].message)
            self.params_inferred = dict(zip(self.x0.keys(), self.result.x))
            self.loss_inferred = self.result.fun
            self.dist_inferred = None
            self.runs = pd.DataFrame(
                [list(result.x) + [result.fun, str(result)] for result in results],
                columns=list(self.x0.keys()) + ['loss', 'result']
            )

            return self.result

        if not finite:
            raise RuntimeError(
                'None of the optimization runs returned a finite loss. The loss function raised or returned a '
                'non-finite value at every evaluated point; the preceding warnings name the parameters and the '
                'underlying error.'
            )

        self.result = min(finite, key=lambda result: result.fun)

        # every evaluation hit the penalty of the loss wrapper, so the reported estimate is the start point
        if self.result.fun >= _LOSS_PENALTY:
            self._logger.warning(
                'The loss was invalid at every evaluated point, so the reported parameters are the start point '
                'rather than an estimate. The preceding warnings name the parameters and the underlying error.'
            )

        self._check_agreement(results)

        # fetch optimized params
        self.params_inferred = dict(zip(list(self.x0.keys()), self.result.x))

        self._logger.info(
            f'Inferred parameters: ({", ".join([f"{k}={v:.4f}" for k, v in self.params_inferred.items()])})'
        )

        # loss of best run
        self.loss_inferred = self.result.fun

        self.runs = pd.DataFrame(
            [list(result.x) + [result.fun, str(result)] for result in results],
            columns=list(self.x0.keys()) + ['loss', 'result']
        )

        # coalescent distribution of best run
        self.dist_inferred = self.get_coal(**self.params_inferred)

        # return the result of the optimization
        return self.result

    def _check_agreement(self, results: List[OptimizeResult]) -> None:
        """
        Warn when runs reaching the loss of the best run within :data:`_RESTART_LOSS_RTOL` end at estimates that differ
        from it by more than :data:`_RESTART_PARAM_RTOL`.

        :param results: The results of the runs, among them the best run
            :attr:`Inference.result <phasegen.inference.Inference.result>`.
        """
        best = self.result
        if not best.fun < _LOSS_PENALTY:
            return

        log = np.array([name in self.log_scale for name in self.param_names])
        tolerance = _RESTART_LOSS_RTOL * max(abs(best.fun), 1)
        differ = np.zeros(len(log), dtype=bool)
        n_ties = 0

        for result in results:
            if result is best or not result.fun - best.fun <= tolerance or not np.isfinite(result.x).all():
                continue

            n_ties += 1
            a, b = np.asarray(result.x, dtype=float), np.asarray(best.x, dtype=float)
            differ |= np.where(
                log,
                np.abs(np.log(np.where(log, a, 1)) - np.log(np.where(log, b, 1))) > _RESTART_PARAM_RTOL,
                np.abs(a - b) > _RESTART_PARAM_RTOL * np.maximum(np.abs(a), np.abs(b))
            )

        if differ.any():
            self._logger.warning(
                '%d other runs reach the loss of the best run within a relative %.0e but differ from its estimate by '
                'more than a relative %.0e in %s, so the loss may be multimodal or these parameters not identifiable.',
                n_ties, _RESTART_LOSS_RTOL, _RESTART_PARAM_RTOL, ', '.join(np.array(self.param_names)[differ])
            )

    def _sample(self) -> Dict[str, float]:
        """
        Sample initial parameters uniformly within the bounds, log-uniformly for the parameters on a logarithmic scale.

        :return: Sampled parameters.
        :raises ValueError: If a bound is not finite.
        """
        unbounded = [key for key, bounds in self.bounds.items() if not np.all(np.isfinite(bounds))]

        if unbounded:
            raise ValueError(
                f'Start points cannot be sampled for parameters with a non-finite bound: {unbounded}. Pass x0 and '
                f'use a single run, or give finite bounds.'
            )

        return {
            key: float(np.clip(np.exp(self._rng.uniform(np.log(lower), np.log(upper))), lower, upper))
            if key in self.log_scale else self._rng.uniform(lower, upper)
            for key, (lower, upper) in self.bounds.items()
        }

    def run(self) -> None:
        """
        Minimize the loss from ``n_runs`` start points, store the best estimate in :attr:`Inference.params_inferred
        <phasegen.inference.Inference.params_inferred>` and every run in :attr:`Inference.runs
        <phasegen.inference.Inference.runs>`, and bootstrap when ``do_bootstrap`` is set.
        """
        self._run()

        if self.do_bootstrap:
            self.bootstrap()

    def bootstrap(self) -> None:
        """
        Perform bootstrapping to estimate parameter uncertainty. For each of :attr:`n_bootstraps` replicates the
        observation :math:`y` is resampled to :math:`y^{*}` via the :paramref:`~phasegen.inference.Inference.resample`
        callback and the inference is rerun, yielding :math:`\\hat{\\theta}^{*} = \\arg\\min_{\\theta}
        L(\\mathrm{coal}(\\theta),\\, y^{*})`. The spread of the replicate estimates :math:`\\{\\hat{\\theta}^{*}_b\\}`
        estimates the sampling distribution of :math:`\\hat{\\theta}`. Each replicate starts from
        :math:`\\hat{\\theta}`, and the replicates are stored in :attr:`Inference.bootstraps
        <phasegen.inference.Inference.bootstraps>`.

        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        """
        if not self.params_inferred:
            raise RuntimeError('The main optimization must be run first (call run()).')

        x0 = self.params_inferred
        bounds = self.bounds
        get_dist = self.get_coal
        get_loss = self.loss
        opts = self.opts
        method_mle = self.method_mle
        gradient = self.gradient
        log_scale = self.log_scale
        penalty = self.penalty
        logger = self._logger

        self._announce_gradient(x0, self.observation)

        def run_sample(observation: Any) -> OptimizeResult:
            """
            Run a single bootstrap sample.

            :param observation: Observation.
            :return: Bootstrap sample, with NaN parameters and an infinite loss if the optimization raised.
            """
            try:
                return Inference._optimize(
                    observation=observation,
                    x0=x0,
                    bounds=bounds,
                    show_pbar=False,
                    get_dist=get_dist,
                    get_loss=get_loss,
                    opts=opts,
                    method_mle=method_mle,
                    logger=logger,
                    gradient=gradient,
                    log_scale=log_scale,
                    penalty=penalty
                )
            except Exception as e:
                logger.warning('Bootstrap replicate failed and its parameters are NaN: %s', e)
                return OptimizeResult(x=np.full(len(x0), np.nan), fun=np.inf, success=False, message=str(e))

        results = parallelize(
            func=run_sample,
            data=[self.resample(self.observation, self._rng) for _ in range(self.n_bootstraps)],
            parallelize=self.parallelize,
            pbar=self.pbar,
            desc='Bootstrapping',
            dtype=object
        )

        # a replicate whose every evaluation hit the penalty of the loss wrapper has no estimate
        n_penalty = 0
        for result in results:
            if _LOSS_PENALTY <= result.fun < np.inf:
                self._invalidate(result)
                n_penalty += 1

        if n_penalty > 0:
            self._logger.warning(
                'The loss was invalid at every evaluated point of %d out of %d bootstrap replicates, so their '
                'parameters are NaN. The preceding warnings name the parameters and the underlying error.',
                n_penalty, self.n_bootstraps
            )

        # count successful optimizations
        n_success = sum([result.success for result in results])

        if n_success < self.n_bootstraps:
            self._logger.warning(
                f'{n_success} out of {self.n_bootstraps} bootstrap replicates converged.'
            )

        # store bootstrapped parameters
        self.bootstraps = pd.DataFrame(
            [list(result.x) + [result.fun, str(result)] for result in results],
            columns=list(self.x0.keys()) + ['loss', 'result']
        )

        # log mean and std of bootstrapped parameters
        self._logger.info(
            f'Bootstrapped parameters: '
            f'mean: ({", ".join([f"{k}={v:.4f}" for k, v in self.bootstraps[self.param_names].mean().items()])})'
            + (
                f', std: ({", ".join([f"{k}={v:.4f}" for k, v in self.bootstraps[self.param_names].std().items()])})'
                if self.n_bootstraps > 1 else ''
            )
        )

    @staticmethod
    def _invalidate(result: OptimizeResult) -> OptimizeResult:
        """
        Mark a result whose loss was invalid at every evaluated point as not converged, with NaN parameters.

        :param result: Result of the optimization procedure, modified in place.
        :return: The same result.
        """
        result.success = False
        result.message = 'The loss was invalid at every evaluated point.'
        result.x = np.full(len(result.x), np.nan)

        return result

    @property
    def _bootstrap_values(self) -> np.ndarray:
        """
        Bootstrapped parameter values, of shape ``(n_bootstraps, n_params)``, with columns in the order of
        :attr:`param_names`.
        """
        return self.bootstraps[self.param_names].to_numpy(dtype=float)

    @property
    def _bootstrap_demographies(self) -> List[Demography]:
        """
        The demography of each bootstrap replicate with an estimate.

        :return: One demography per row of :attr:`_bootstrap_values` without NaN.
        """
        return [
            self.get_coal(**dict(zip(self.param_names, row))).demography
            for row in self._bootstrap_values if not np.isnan(row).any()
        ]

    def fisher_information(self) -> pd.DataFrame:
        r"""
        The observed information at the estimate, the Hessian of the loss,

        .. math::
            I_{ij} = \frac{\partial^2 L\big(\mathrm{coal}(\theta),\, y\big)}{\partial \theta_i \, \partial \theta_j}
            \bigg|_{\theta = \hat{\theta}},

        where :math:`L` is the loss, :math:`\mathrm{coal}(\theta)` the coalescent returned by the ``coal`` callback at
        the parameters :math:`\theta`, :math:`y` the observation and :math:`\hat{\theta}` the estimate
        :attr:`Inference.params_inferred <phasegen.inference.Inference.params_inferred>`. Where :math:`L` is a negative
        log-likelihood, :math:`I` is the observed Fisher information. The Hessian is a central difference of the
        gradient of :meth:`Gradient.compute() <phasegen.gradient.Gradient.compute>`, which is exact where
        :attr:`Inference.gradient <phasegen.inference.Inference.gradient>` asks for it and the loss supports it. With a
        penalty :math:`P(\theta)` configured as :attr:`Inference.penalty <phasegen.inference.Inference.penalty>`, it is
        the Hessian of :math:`L + P`, with the gradient of :math:`P` taken by central differences. The differences stay
        within the bounds, and a parameter whose bounds coincide has no curvature.

        :return: The information, with the parameter names as index and columns.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        """
        if not self.params_inferred:
            raise RuntimeError('The main optimization must be run first (call run()).')

        names = self.param_names
        x = np.array([self.params_inferred[name] for name in names], dtype=float)
        key = (tuple(x), self._observation_key(), self.loss, self.coal, tuple(self.bounds[n] for n in names),
               self.gradient, self.penalty)

        if self._fisher_information is not None and self._fisher_information[0] == key:
            return self._fisher_information[1]

        bounds = [self.bounds[name] for name in names]
        gradient = Gradient(self.get_coal, lambda coal: self.loss(coal, self.observation))

        def grad(params: np.ndarray) -> np.ndarray:
            g = gradient.compute(dict(zip(names, params)), self.bounds, exact=self.gradient == 'exact')[1]
            if self.penalty is None:
                return g
            return g + Inference._penalty_gradient(self.penalty, dict(zip(names, params)), self.bounds)

        hessian = Gradient._differences(grad, x, bounds, _HESSIAN_STEP * np.where(x != 0, np.abs(x), 1))
        info = pd.DataFrame((hessian + hessian.T) / 2, index=names, columns=names)
        self._fisher_information = (key, info)

        return info

    def _observation_key(self) -> Any:
        """
        A key of the observation that changes when its data do, or its identity where it has no numeric data.
        """
        try:
            return hashlib.sha1(np.ascontiguousarray(
                np.asarray(getattr(self.observation, 'data', self.observation), dtype=float)
            ).tobytes()).hexdigest()
        except (TypeError, ValueError):
            return id(self.observation)

    def _score_covariance(self, n: int) -> pd.DataFrame:
        r"""
        The covariance of the score over resampled observations,

        .. math::
            J = \frac{1}{n - 1} \sum_{b=1}^{n} (g_b - \bar{g})(g_b - \bar{g})^\top, \qquad
            g_b = \nabla_\theta L\big(\mathrm{coal}(\theta),\, y^{*}_b\big) \big|_{\theta = \hat{\theta}},

        where :math:`y^{*}_b` are :math:`n` observations drawn with the
        :paramref:`~phasegen.inference.Inference.resample` callback from a generator derived from :attr:`Inference.seed
        <phasegen.inference.Inference.seed>`, :math:`\bar{g}` is the mean of the scores :math:`g_b` and the gradients
        are those of :meth:`Gradient.compute() <phasegen.gradient.Gradient.compute>`.

        :param n: The number of resampled observations, at least 2.
        :return: The covariance, with the parameter names as index and columns.
        :raises ValueError: If :paramref:`~phasegen.inference.Inference.resample` is ``None``, ``n`` is less than 2 or
            fewer than two scores are finite.
        """
        if self.resample is None:
            raise ValueError("The Godambe covariance needs the resample callback to draw observations.")

        n = int(n)
        if n < 2:
            raise ValueError(f'n must be at least 2, got {n}.')

        names = self.param_names
        x = {name: float(self.params_inferred[name]) for name in names}
        entropy = self.seed if self._entropy is None else self._entropy
        key = (tuple(x.values()), self._observation_key(), self.resample, self.loss, self.coal,
               tuple(self.bounds[name] for name in names), self.gradient, n, entropy)

        if self._score_covariance_cache is not None and self._score_covariance_cache[0] == key:
            return self._score_covariance_cache[1]

        rng = np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=_GODAMBE_SPAWN_KEY))
        exact = self.gradient == 'exact' and Gradient(
            self.get_coal, lambda coal: self.loss(coal, self.observation)).supported(x)

        # the moments and their derivatives are those of the estimate for every resampled observation
        observations = [self.resample(self.observation, rng) for _ in range(n)]
        losses = [lambda coal, y=y: self.loss(coal, y) for y in observations]
        scores = Gradient(self.get_coal, losses[0])._scores(x, self.bounds, losses) if exact else None

        if scores is None:
            scores = [Gradient(self.get_coal, loss).compute(x, self.bounds, exact=exact)[1] for loss in losses]

        scores = np.array(scores, dtype=float)

        finite = np.isfinite(scores).all(axis=1)
        if finite.sum() < 2:
            raise ValueError(f'Fewer than two of the {n} scores of the resampled observations are finite.')

        if not finite.all():
            self._logger.warning('The score is not finite for %d of %d resampled observations, which are left out.',
                                 n - finite.sum(), n)

        # centred, as the mean score is not zero where the estimate lies on a bound or short of the optimum
        cov = pd.DataFrame(np.atleast_2d(np.cov(scores[finite], rowvar=False)), index=names, columns=names)
        self._score_covariance_cache = (key, cov)

        return cov

    def covariance(self, method: Literal['fisher', 'godambe'] = 'fisher', n: int = 100) -> pd.DataFrame:
        r"""
        The asymptotic covariance of the estimate :math:`\hat{\theta}`. With ``'fisher'``, it is the inverse
        :math:`H^{-1}` of the observed information :math:`H` of :meth:`Inference.fisher_information()
        <phasegen.inference.Inference.fisher_information>`, which assumes that the loss :math:`L` is the negative
        log-likelihood of independent observations, such as the bins of an SFS of unlinked loci. With ``'godambe'``,
        it is the sandwich estimate of the inverse Godambe information
        (`Coffman et al. 2016 <https://doi.org/10.1093/molbev/msv255>`__),

        .. math::
            \Sigma = H^{-1} J H^{-1},

        where :math:`J` is the covariance of the score :math:`\nabla_\theta L\big(\mathrm{coal}(\theta),\, y^{*}\big)`
        at :math:`\hat{\theta}` over :math:`n` observations :math:`y^{*}` drawn with the
        :paramref:`~phasegen.inference.Inference.resample` callback, without refitting. It holds for a misspecified or
        composite likelihood, such as that of an SFS of linked sites, when the callback reproduces the sampling of the
        observation, for instance by a block bootstrap over the genome. The draws derive from :attr:`Inference.seed
        <phasegen.inference.Inference.seed>`, and :math:`J` is cached.

        :math:`H^{-1}` is taken over the directions in which :math:`H` is positive. A parameter that loads on a
        direction of zero or negative curvature, such as one the loss does not depend on, has an undefined variance,
        reported as NaN. The covariance describes the curvature of the loss at the estimate only, so it neither reflects
        other optima nor the uncertainty of parameters on a bound, both of which :meth:`Inference.bootstrap()
        <phasegen.inference.Inference.bootstrap>` captures. A warning is logged where :math:`H` is not positive definite
        or a parameter lies on a bound.

        ::

            inf.run()
            inf.covariance(method='godambe', n=200)

        :param method: ``'fisher'`` for :math:`H^{-1}`, ``'godambe'`` for :math:`H^{-1} J H^{-1}`.
        :param n: The number of resampled observations :math:`n` of the Godambe covariance, at least 2. The standard
            errors have a relative Monte-Carlo error of about :math:`1 / \sqrt{2 n}`.
        :return: The covariance, with the parameter names as index and columns.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``method`` is unknown, or for ``'godambe'``, if
            :paramref:`~phasegen.inference.Inference.resample` is ``None`` or ``n`` is less than 2.
        """
        if method not in ('fisher', 'godambe'):
            raise ValueError(f"method must be 'fisher' or 'godambe', got {method!r}.")

        info = self.fisher_information()
        w, V = np.linalg.eigh(info.to_numpy())
        positive = w > _CURVATURE_RTOL * max(np.abs(w).max(), 1e-300)

        if not positive.all():
            self._logger.warning('The Fisher information is not positive definite, so the estimate is not a strict '
                                 'local minimum of the loss or some parameters are not identifiable.')

        on_bound = [name for name in self.param_names if self.params_inferred[name] in self.bounds[name]]
        if on_bound:
            self._logger.warning('The parameters %s lie on a bound, where the asymptotic covariance does not describe '
                                 'their uncertainty.', ', '.join(on_bound))

        cov = (V[:, positive] / w[positive]) @ V[:, positive].T
        if method == 'godambe':
            cov = cov @ self._score_covariance(n).to_numpy() @ cov

        undefined = (V[:, ~positive] ** 2).sum(axis=1) > _CURVATURE_RTOL
        cov[undefined, :] = np.nan
        cov[:, undefined] = np.nan

        return pd.DataFrame(cov, index=info.index, columns=info.columns)

    def standard_errors(self, method: Literal['fisher', 'godambe'] = 'fisher', n: int = 100) -> pd.Series:
        r"""
        The asymptotic standard errors of the estimate, :math:`\sqrt{\Sigma_{ii}}`, the square roots of the diagonal
        of the covariance :math:`\Sigma` of :meth:`Inference.covariance() <phasegen.inference.Inference.covariance>`,
        NaN where it is undefined.

        ::

            inf.run()
            pd.DataFrame({
                'estimate': inf.params_inferred,
                'standard error (Fisher)': inf.standard_errors(),
                'standard error (Godambe)': inf.standard_errors(method='godambe')
            })

        :param method: ``'fisher'`` for the inverse observed information, ``'godambe'`` for the sandwich estimate,
            see :meth:`Inference.covariance() <phasegen.inference.Inference.covariance>`.
        :param n: The number of resampled observations of the Godambe covariance.
        :return: The standard errors, indexed by parameter name.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``method`` is unknown, or for ``'godambe'``, if
            :paramref:`~phasegen.inference.Inference.resample` is ``None`` or ``n`` is less than 2.
        """
        return pd.Series(np.sqrt(np.diag(self.covariance(method, n).to_numpy())), index=self.param_names)

    @property
    def _n_free(self) -> int:
        """
        The number of free parameters, those whose bounds differ.
        """
        return sum(lower < upper for lower, upper in self.bounds.values())

    def _check_likelihood(self) -> None:
        """
        Check that the loss of the estimate can be read as a negative log-likelihood.

        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If a penalty is added to the loss.
        """
        if self.loss_inferred is None:
            raise RuntimeError('The main optimization must be run first (call run()).')

        if self.penalty is not None:
            raise ValueError('The loss includes a penalty, so it is not a negative log-likelihood.')

    def aic(self) -> float:
        r"""
        The Akaike information criterion of the estimate,

        .. math::
            \mathrm{AIC} = 2 k + 2 \hat{L},

        where :math:`k` is the number of free parameters, those whose bounds differ, and :math:`\hat{L}` the loss of
        the estimate, :attr:`Inference.loss_inferred <phasegen.inference.Inference.loss_inferred>`, read as a
        negative log-likelihood. Of models fitted to the same observation with the same likelihood, the one with the
        lowest criterion is preferred.

        ::

            inf.run()
            inf.aic()

        :return: The criterion.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If a penalty is added to the loss.
        """
        self._check_likelihood()

        return 2 * self._n_free + 2 * float(self.loss_inferred)

    def likelihood_ratio_test(self, null: 'Inference') -> pd.Series:
        r"""
        The likelihood-ratio test of a null model nested in this one, fitted to the same observation with the same
        likelihood. With :math:`\hat{L}_0` and :math:`\hat{L}_1` the losses of the null model and of this one, read
        as negative log-likelihoods, the statistic

        .. math::
            D = 2 \big(\hat{L}_0 - \hat{L}_1\big)

        is asymptotically :math:`\chi^2_d`-distributed under the null model, with :math:`d = k_1 - k_0` the
        difference of their numbers of free parameters, and the p-value is :math:`P(\chi^2_d \geq D)`. Where the null
        model fixes a parameter on a bound of this one, such as a migration rate of zero, the distribution is a mixture
        of :math:`\chi^2` distributions with fewer degrees of freedom, so the p-value is conservative. For linked sites,
        whose likelihood is composite, the statistic is not :math:`\chi^2`-distributed.

        ::

            alt.run()
            null.run()
            alt.likelihood_ratio_test(null)

        :param null: The null model, an inference that has been run.
        :return: The statistic :math:`D` (``statistic``), the degrees of freedom :math:`d` (``df``) and the p-value
            (``p-value``).
        :raises RuntimeError: If either inference has not been run.
        :raises ValueError: If either loss includes a penalty, the observations differ or the null model has no fewer
            free parameters than this one.
        """
        from scipy.stats import chi2

        self._check_likelihood()
        null._check_likelihood()

        if null._observation_key() != self._observation_key():
            raise ValueError('The null model was fitted to a different observation.')

        df = self._n_free - null._n_free
        if df < 1:
            raise ValueError(f'The null model must have fewer free parameters than this one, got {null._n_free} and '
                             f'{self._n_free}.')

        statistic = 2 * (float(null.loss_inferred) - float(self.loss_inferred))

        if statistic < -2 * _RESTART_LOSS_RTOL * max(abs(float(self.loss_inferred)), 1):
            self._logger.warning(
                'The loss of this model exceeds that of the null model by %.4g, so the models are not nested or this '
                'optimization did not reach its optimum.', -statistic / 2
            )

        return pd.Series({'statistic': statistic, 'df': df, 'p-value': chi2.sf(max(statistic, 0), df)})

    def _trajectory_times(self, t: np.ndarray = None) -> np.ndarray:
        """
        The times at which trajectories are evaluated.

        :param t: Times, by default
            :attr:`Settings.plot_inference_n_grid <phasegen.settings.Settings.plot_inference_n_grid>` points up to the
            :attr:`Settings.plot_inference_quantile <phasegen.settings.Settings.plot_inference_quantile>` quantile of
            the inferred tree height, or up to the end time of a windowed coalescent.
        :return: The times.
        :raises RuntimeError: If the main optimization has not been run.
        """
        if self.dist_inferred is None:
            raise RuntimeError('The main optimization must be run first (call run()).')

        if t is not None:
            return np.asarray(t, dtype=float)

        tree_height = self.dist_inferred.tree_height

        # a windowed coalescent has no tree-height quantile, so its end time bounds the plot
        t_end = tree_height.t_max if tree_height._windowed else tree_height.quantile(Settings.plot_inference_quantile)

        return np.linspace(0, t_end, Settings.plot_inference_n_grid)

    def _trajectory_band(
            self,
            t: np.ndarray,
            kind: Literal['all', 'pop_sizes', 'migration'],
            level: float,
            method: Literal['fisher', 'godambe'] = 'fisher',
            n: int = 100
    ) -> Tuple['_CurveData', np.ndarray, np.ndarray, np.ndarray]:
        """
        The trajectories of the estimate with their standard errors and confidence limits, see
        :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>`.

        :param t: Times.
        :param kind: The trajectories.
        :param level: The confidence level.
        :param method: The covariance, see :meth:`Inference.covariance() <phasegen.inference.Inference.covariance>`.
        :param n: The number of resampled observations of the Godambe covariance.
        :return: The trajectories, and the standard errors, lower and upper limits, each of shape
            ``(n_trajectories, n_times)``.
        """
        from scipy.stats import norm

        names = self.param_names
        cov = self.covariance(method, n).to_numpy()
        estimate = self.dist_inferred.demography._plot_data(t, kind)
        y = np.asarray(estimate.y, dtype=float)
        positive = y > 0

        # derivatives of the logarithms of positive trajectories, and of the values of the others, by central
        # differences of the demographies the callback builds, which need no moments
        jacobian = np.zeros((len(names),) + y.shape)
        for i, name in enumerate(names):
            lo, hi = self.bounds[name]
            if lo == hi:
                continue

            x = self.params_inferred[name]
            h = _TRAJECTORY_STEP * max(abs(x), 1.0)
            up, down = min(x + h, hi), max(x - h, lo)
            curves = [np.asarray(self.get_coal(**(self.params_inferred | {name: v})).demography._plot_data(t, kind).y,
                                 dtype=float) for v in (up, down)]

            with np.errstate(divide='ignore', invalid='ignore'):
                jacobian[i] = np.where(positive, np.log(curves[0]) - np.log(curves[1]), curves[0] - curves[1])
            jacobian[i] /= up - down

        # a parameter without a defined variance leaves undefined the trajectories that depend on it
        undefined = np.isnan(np.diag(cov))
        cov = np.nan_to_num(cov)
        variance = np.einsum('i...,ij,j...->...', jacobian, cov, jacobian)
        variance[np.any(jacobian[undefined] != 0, axis=0)] = np.nan
        se = np.sqrt(np.maximum(variance, 0))

        z = norm.ppf(0.5 + level / 2)
        lower = np.where(positive, y * np.exp(-z * se), np.maximum(y - z * se, 0))
        upper = np.where(positive, y * np.exp(z * se), y + z * se)
        se = np.where(positive, y * se, se)

        return estimate, se, lower, upper

    def trajectory_intervals(
            self,
            t: np.ndarray = None,
            kind: Literal['all', 'pop_sizes', 'migration'] = 'pop_sizes',
            level: float = 0.95,
            method: Literal['fisher', 'godambe'] = 'fisher',
            n: int = 100
    ) -> pd.DataFrame:
        r"""
        Approximate confidence intervals of the population size and migration rate trajectories of the estimate, from
        the covariance :math:`\Sigma` of :meth:`Inference.covariance() <phasegen.inference.Inference.covariance>` by
        the delta method. The estimated parameters are taken as approximately normal, and their covariance is
        propagated through the demographic function :math:`f` that maps parameters to trajectories.

        Since population sizes and migration rates are positive and act multiplicatively, their logs are closer to
        normal, so the propagation is done on the log scale. A positive trajectory value :math:`y = f(\hat\theta)`,
        such as the population size of a deme at time :math:`t`, has the standard error

        .. math::
            \mathrm{se}(\log y) = \sqrt{g^\top \Sigma\, g}, \qquad g = \frac{\nabla f(\hat\theta)}{f(\hat\theta)},

        where :math:`\hat\theta` are the estimated parameters and the gradient is a central difference of the
        demography the :paramref:`~phasegen.inference.Inference.coal` callback builds. Exponentiating the normal
        interval of :math:`\log y` gives the lower and upper confidence limits :math:`y\, e^{\pm z\, \mathrm{se}(\log
        y)}`, with :math:`z` the normal quantile of the level, which stay positive. A trajectory that is zero, such as
        an absent migration rate, takes limits on the linear scale, clipped at zero. The intervals are local, as the
        covariance is.

        ::

            inf.run()
            inf.trajectory_intervals(kind='pop_sizes')

        :param t: Times. By default, :attr:`Settings.plot_inference_n_grid
            <phasegen.settings.Settings.plot_inference_n_grid>` points up to the
            :attr:`Settings.plot_inference_quantile <phasegen.settings.Settings.plot_inference_quantile>` quantile of
            the inferred tree height, or up to the end time of a windowed coalescent.
        :param kind: The trajectories, ``'pop_sizes'``, ``'migration'`` or ``'all'``.
        :param level: The confidence level.
        :param method: The covariance, ``'fisher'`` or ``'godambe'``, see :meth:`Inference.covariance()
            <phasegen.inference.Inference.covariance>`.
        :param n: The number of resampled observations of the Godambe covariance.
        :return: One row per time and trajectory, with the columns ``time``, ``trajectory``, ``estimate``,
            ``standard error``, ``lower`` and ``upper``.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``method`` is unknown, or for ``'godambe'``, if
            :paramref:`~phasegen.inference.Inference.resample` is ``None`` or ``n`` is less than 2.
        """
        t = self._trajectory_times(t)
        estimate, se, lower, upper = self._trajectory_band(t, kind, level, method, n)

        return pd.DataFrame({
            'time': np.tile(t, len(estimate.labels)),
            'trajectory': np.repeat(estimate.labels, len(t)),
            'estimate': np.asarray(estimate.y, dtype=float).ravel(),
            'standard error': se.ravel(),
            'lower': lower.ravel(),
            'upper': upper.ravel()
        })

    def _plot_demography_data(
            self,
            t: np.ndarray = None,
            kind: Literal['all', 'pop_sizes', 'migration'] = 'all',
            interval: Literal['bootstrap', 'fisher', 'godambe'] | None = 'bootstrap'
    ) -> Tuple['_CurveData', List['_CurveData'], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Trajectories of the inferred demography with their uncertainty, as drawn by
        :meth:`Inference.plot_demography() <phasegen.inference.Inference.plot_demography>`,
        :meth:`Inference.plot_pop_sizes() <phasegen.inference.Inference.plot_pop_sizes>` and
        :meth:`Inference.plot_migration() <phasegen.inference.Inference.plot_migration>`.

        :param t: Times at which to evaluate the trajectories, see
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>`.
        :param kind: The trajectories to include, ``'pop_sizes'``, ``'migration'`` or ``'all'``.
        :param interval: ``'bootstrap'`` for the trajectories of the bootstrap replicates, ``'fisher'`` or ``'godambe'``
            for the 95% intervals of
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` with that
            covariance, or ``None``.
        :return: The inferred trajectories, those of each bootstrap replicate, and the lower and upper limits of the
            intervals, ``None`` without them.
        :raises RuntimeError: If the main optimization has not been run.
        :raises ValueError: If ``interval`` is unknown.
        """
        if interval not in ('bootstrap', 'fisher', 'godambe', None):
            raise ValueError(f"interval must be 'bootstrap', 'fisher', 'godambe' or None, got {interval!r}.")

        t = self._trajectory_times(t)

        if interval in ('fisher', 'godambe'):
            inferred, _, lower, upper = self._trajectory_band(t, kind, 0.95, interval)
            return inferred, [], (lower, upper)

        bootstraps = self._bootstrap_demographies if interval == 'bootstrap' else []

        return self.dist_inferred.demography._plot_data(t, kind), [d._plot_data(t, kind) for d in bootstraps], None

    def plot_bootstraps(
            self,
            title: str | List[str] = None,
            show: bool = True,
            file: str = None,
            subplots: bool = True,
            kind: Literal['hist', 'kde'] = 'hist',
            ax: 'plt.Axes' | List['plt.Axes'] = None,
            kwargs: dict = None,
            truth: Dict[str, float] = None
    ) -> 'plt.Axes' | List['plt.Axes']:
        """
        Plot bootstrapped parameters.

        :param title: Title or list of titles.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param subplots: Whether to plot subplots.
        :param kind: Kind of plot. Either 'hist' or 'kde'.
        :param ax: Axes or list of axes.
        :param kwargs: Additional keyword arguments passed to the pandas plot function.
        :param truth: True values of some or all parameters, keyed by name, drawn as dashed vertical lines.
        :return: Axes or list of axes.
        :raises RuntimeError: If no bootstraps are available, or if ``kind`` is ``'kde'`` and a parameter has fewer
            than two finite replicates.
        :raises ValueError: If ``subplots`` is true and fewer axes than parameters are given, or if ``truth`` names
            an unknown parameter.
        """
        from .visualization import Visualization

        import matplotlib

        if kwargs is None:
            kwargs = {}

        if self.bootstraps.empty:
            raise RuntimeError('No bootstraps available.')

        if truth is None:
            truth = {}

        unknown = set(truth) - set(self.param_names)
        if unknown:
            raise ValueError(f"truth names unknown parameters: {', '.join(sorted(unknown))}.")

        if kind == 'kde':
            n_finite = np.isfinite(self.bootstraps[self.param_names].to_numpy(dtype=float)).sum(axis=0)
            few = [name for name, k in zip(self.param_names, n_finite) if k < 2]
            if few:
                raise RuntimeError(f"A kernel density estimate needs at least two finite replicates per parameter, "
                                   f"which {', '.join(few)} lack.")

        if kind == 'hist':
            kwargs = {'bins': 20} | kwargs

        if subplots:
            import matplotlib.pyplot as plt

            # one axis per parameter, so that each histogram is binned over the range of its own parameter, in one
            # column for a few parameters and in three otherwise
            n = len(self.param_names)
            if ax is None:
                ncols = 1 if n <= _BOOTSTRAP_COLUMN_MAX else 3
                width, height = plt.rcParams['figure.figsize']
                fig, ax = plt.subplots(int(np.ceil(n / ncols)), ncols, squeeze=False,
                                       figsize=(width, _BOOTSTRAP_HEIGHT_SCALE * height))
                for extra in ax.ravel()[n:]:
                    extra.set_visible(False)
            elif np.ravel(ax).size < n:
                raise ValueError(f"Plotting {n} parameters as subplots needs {n} axes, got {np.ravel(ax).size}.")
            ax = np.ravel(ax)[:n]
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for i, name in enumerate(self.param_names):
                self.bootstraps[name].plot(ax=ax[i], kind=kind, legend=True,
                                           **({'color': colors[i % len(colors)]} | kwargs))
                ax[i].set_ylabel('')

                if name in truth:
                    ax[i].axvline(truth[name], color='black', linestyle='--')

                if isinstance(title, list):
                    ax[i].set_title(title[i])

            ax[0].figure.supylabel('frequency' if kind == 'hist' else 'density')

            if not isinstance(title, list):
                ax[0].figure.suptitle('Marginal distributions' if title is None else title)
        else:
            ax = self.bootstraps[self.param_names].plot(
                ax=ax,
                kind=kind,
                title='Marginal distributions' if title is None else title,
                **kwargs
            )

            # in the colour pandas gives each parameter
            colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

            for i, name in enumerate(self.param_names):
                if name in truth:
                    ax.axvline(truth[name], color=colors[i % len(colors)], linestyle='--')

        # make layout tight
        np.ravel(ax)[0].figure.tight_layout()

        return Visualization.show_and_save(ax, show=show, file=file)

    def plot_demography(
            self,
            t: np.ndarray = None,
            interval: Literal['bootstrap', 'fisher', 'godambe'] | None = 'bootstrap',
            show: bool = True,
            file: str = None,
            kwargs: dict = None,
            ax: Optional['plt.Axes'] = None,
            truth: Demography = None,
            include_bootstraps: bool = None
    ) -> 'plt.Axes':
        """
        Plot inferred demography.

        :param t: Time points. By default, :attr:`Settings.plot_inference_n_grid
            <phasegen.settings.Settings.plot_inference_n_grid>` points up to the
            :attr:`Settings.plot_inference_quantile <phasegen.settings.Settings.plot_inference_quantile>` quantile of
            the inferred tree height, or up to the end time of a windowed coalescent.
        :param interval: The uncertainty drawn with the estimate: ``'bootstrap'`` for the trajectory of each
            bootstrap replicate, ``'fisher'`` or ``'godambe'`` for the 95% intervals of
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` with that
            covariance, or ``None``.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: Axes to plot on.
        :param truth: The true demography, whose trajectories are drawn dashed in the colour of the inferred ones with
            the same label.
        :param include_bootstraps: Deprecated, use ``interval``. ``True`` selects ``'bootstrap'`` and ``False``
            ``None``.
        :return: Axes.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``interval`` is unknown, for ``'godambe'`` if
            :paramref:`~phasegen.inference.Inference.resample` is ``None``, or if ``truth`` has a trajectory the
            inferred demography lacks.
        """
        return self._plot_demography(
            t=t,
            show=show,
            interval=interval,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='all',
            truth=truth,
            include_bootstraps=include_bootstraps
        )

    def plot_pop_sizes(
            self,
            t: np.ndarray = None,
            show: bool = True,
            interval: Literal['bootstrap', 'fisher', 'godambe'] | None = 'bootstrap',
            file: str = None,
            kwargs: dict = None,
            ax: Optional['plt.Axes'] = None,
            truth: Demography = None,
            include_bootstraps: bool = None
    ) -> 'plt.Axes':
        """
        Plot inferred population sizes.

        :param t: Time points. By default, :attr:`Settings.plot_inference_n_grid
            <phasegen.settings.Settings.plot_inference_n_grid>` points up to the
            :attr:`Settings.plot_inference_quantile <phasegen.settings.Settings.plot_inference_quantile>` quantile of
            the inferred tree height, or up to the end time of a windowed coalescent.
        :param show: Whether to show the plot.
        :param interval: The uncertainty drawn with the estimate: ``'bootstrap'`` for the trajectory of each
            bootstrap replicate, ``'fisher'`` or ``'godambe'`` for the 95% intervals of
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` with that
            covariance, or ``None``.
        :param file: File to save the plot to.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: Axes to plot on.
        :param truth: The true demography, whose trajectories are drawn dashed in the colour of the inferred ones with
            the same label.
        :param include_bootstraps: Deprecated, use ``interval``. ``True`` selects ``'bootstrap'`` and ``False``
            ``None``.
        :return: Axes.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``interval`` is unknown, for ``'godambe'`` if
            :paramref:`~phasegen.inference.Inference.resample` is ``None``, or if ``truth`` has a trajectory the
            inferred demography lacks.
        """
        return self._plot_demography(
            t=t,
            show=show,
            interval=interval,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='pop_sizes',
            truth=truth,
            include_bootstraps=include_bootstraps
        )

    def plot_migration(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            interval: Literal['bootstrap', 'fisher', 'godambe'] | None = 'bootstrap',
            kwargs: dict = None,
            ax: Optional['plt.Axes'] = None,
            truth: Demography = None,
            include_bootstraps: bool = None
    ) -> 'plt.Axes':
        """
        Plot inferred migration rates.

        :param t: Time points. By default, :attr:`Settings.plot_inference_n_grid
            <phasegen.settings.Settings.plot_inference_n_grid>` points up to the
            :attr:`Settings.plot_inference_quantile <phasegen.settings.Settings.plot_inference_quantile>` quantile of
            the inferred tree height, or up to the end time of a windowed coalescent.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param interval: The uncertainty drawn with the estimate: ``'bootstrap'`` for the trajectory of each
            bootstrap replicate, ``'fisher'`` or ``'godambe'`` for the 95% intervals of
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` with that
            covariance, or ``None``.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: Axes to plot on.
        :param truth: The true demography, whose trajectories are drawn dashed in the colour of the inferred ones with
            the same label.
        :param include_bootstraps: Deprecated, use ``interval``. ``True`` selects ``'bootstrap'`` and ``False``
            ``None``.
        :return: Axes.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``interval`` is unknown, for ``'godambe'`` if
            :paramref:`~phasegen.inference.Inference.resample` is ``None``, or if ``truth`` has a trajectory the
            inferred demography lacks.
        """
        return self._plot_demography(
            t=t,
            show=show,
            interval=interval,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='migration',
            truth=truth,
            include_bootstraps=include_bootstraps
        )

    def _deprecated_interval(
            self,
            interval: Literal['bootstrap', 'fisher', 'godambe'] | bool | None,
            include_bootstraps: bool | None
    ) -> Literal['bootstrap', 'fisher', 'godambe'] | None:
        """
        Map the deprecated ``include_bootstraps``, passed by keyword or as a boolean in the position of ``interval``,
        to ``interval``, logging a warning.

        :param interval: The interval, or a boolean passed in its position.
        :param include_bootstraps: Whether to draw the bootstrap replicates, ``None`` if not given.
        :return: The interval.
        """
        if isinstance(interval, bool):
            interval, include_bootstraps = 'bootstrap', interval

        if include_bootstraps is None:
            return interval

        self._logger.warning("include_bootstraps is deprecated, use interval='bootstrap' or interval=None.")

        return 'bootstrap' if include_bootstraps else None

    def _plot_demography(
            self,
            t: np.ndarray,
            show: bool,
            interval: Literal['bootstrap', 'fisher', 'godambe'] | None,
            ax: Optional['plt.Axes'],
            kind: Literal['pop_sizes', 'migration', 'all'],
            file: str = None,
            kwargs: dict = None,
            truth: Demography = None,
            include_bootstraps: bool = None
    ) -> 'plt.Axes':
        """
        Plot the trajectories of :meth:`_plot_demography_data`.

        :param t: Time points, ``None`` for the default of :meth:`_plot_demography_data`.
        :param show: Whether to show the plot.
        :param interval: The uncertainty drawn with the estimate: ``'bootstrap'`` for the trajectory of each
            bootstrap replicate, ``'fisher'`` or ``'godambe'`` for the 95% intervals of
            :meth:`Inference.trajectory_intervals() <phasegen.inference.Inference.trajectory_intervals>` with that
            covariance, or ``None``.
        :param ax: Axes to plot on.
        :param kind: The trajectories to include.
        :param file: File to save the plot to.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param truth: The true demography, drawn dashed.
        :param include_bootstraps: Deprecated form of ``interval``.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from .visualization import Visualization

        if kwargs is None:
            kwargs = {}

        interval = self._deprecated_interval(interval, include_bootstraps)
        inferred, bootstraps, band = self._plot_demography_data(t, kind, interval)
        true = None if truth is None else truth._plot_data(inferred.x, kind)

        if true is not None and set(true.labels) - set(inferred.labels):
            raise ValueError(f"truth has trajectories the inferred demography lacks: "
                             f"{', '.join(sorted(set(true.labels) - set(inferred.labels)))}.")

        if ax is None:
            ax = plt.subplots()[1]

        Visualization.plot_rates(ax=ax, data=inferred, show=False, kwargs=kwargs)

        # each bootstrap trajectory and interval in the colour of its series, without a legend entry of its own
        colors = [line.get_color() for line in ax.lines[-len(inferred.labels):]]

        if band is not None:
            for lower, upper, color in zip(*band, colors):
                ax.fill_between(inferred.x, lower, upper, step='post', color=color, alpha=0.25, linewidth=0)

        for data in bootstraps:
            for y, color in zip(data.y, colors):
                style = {'color': color, 'alpha': 0.3} | kwargs
                ax.plot(data.x, y, drawstyle='steps-post', label='_nolegend_', **style)

        if true is not None and true.labels:
            for label, y in zip(true.labels, true.y):
                ax.plot(true.x, y, drawstyle='steps-post', linestyle='--',
                        color=colors[inferred.labels.index(label)], label=f'{label} (true)')

            ax.legend()

        return Visualization.show_and_save(ax, show=show, file=file)

    def _spawn(self, index: int | None) -> 'Inference':
        """
        Copy this Inference object with an independent random number generator.

        :param index: Index of the copy. Copies with distinct indices have independent generators that are
            reproducible from :attr:`seed`, or from the entropy drawn at construction if no seed was given. ``None``
            draws fresh entropy.
        :return: Inference object.
        """
        other = copy.deepcopy(self)
        other.__dict__.pop('_state_spaces', None)
        other.__dict__.pop('_is_bootstrap', None)

        # the spawned object performs a single optimization whose result is merged back with ``add_run``, which reads
        # only the main result; bootstrapping it would repeat ``n_bootstraps`` fits per job and discard every one
        other.do_bootstrap = False

        # the copy starts unfitted, so merging it back before it has run is rejected
        other.result = None
        other.params_inferred = {}
        other.loss_inferred = None
        other.dist_inferred = None
        other.bootstraps = self.bootstraps.iloc[0:0].copy()
        other.runs = self.runs.iloc[0:0].copy()

        if index is None:
            sequence = np.random.SeedSequence()
        else:
            # the entropy of a seeded generator is its seed
            entropy = self.seed if self._entropy is None else self._entropy
            sequence = np.random.SeedSequence(entropy, spawn_key=(int(index),))

        other.seed = int(sequence.generate_state(1)[0])
        other._entropy = other.seed
        other._rng = np.random.default_rng(np.random.SeedSequence(other._entropy))

        return other

    def create_run(self, x0: Dict[str, float] = None, index: int = None) -> 'Inference':
        """
        Create a new Inference object which performs a single optimization run independently. This is useful when
        parallelizing runs on a cluster. You can add performed runs by using the :meth:`add_run` method.

        :param x0: Initial parameters. By default, they are sampled within the bounds.
        :param index: Index of the run, such as a cluster job index. Runs with distinct indices sample independent
            start points, reproducibly from :attr:`seed` and across reloads of a saved Inference object. By default,
            each call draws fresh entropy.
        :return: Inference object.
        """
        other = self._spawn(index)
        other._x0 = other._sample() if x0 is None else x0
        other._check_x0_within_bounds()
        other.n_runs = 1

        return other

    def add_run(self, inference: 'Inference') -> None:
        """
        Append the run of another Inference object to :attr:`Inference.runs <phasegen.inference.Inference.runs>`,
        and adopt its estimate when its loss is lower than the current one.

        :param inference: Inference object.
        :raises RuntimeError: If ``inference`` has not been run.
        """
        if inference.loss_inferred is None:
            raise RuntimeError('The provided Inference object must be run first (call run()).')

        # add the loss of the new run to the list of losses
        self.runs.loc[len(self.runs)] = (
                inference.params_inferred | dict(loss=inference.loss_inferred, result=str(inference.result))
        )

        # update the result if the loss of the new run is lower
        if self.loss_inferred is None or inference.loss_inferred < self.loss_inferred:
            self.result = inference.result
            self.params_inferred = inference.params_inferred
            self.loss_inferred = inference.loss_inferred
            self.dist_inferred = inference.dist_inferred

    def add_runs(self, inferences: Iterable['Inference']) -> None:
        """
        Merge the main optimization results from an iterable of Inference objects by calling
        :meth:`Inference.add_run() <phasegen.inference.Inference.add_run>` on each.

        :param inferences: Iterable of Inference objects.
        """
        for inference in inferences:
            self.add_run(inference)

    def create_bootstrap(self, n_runs: int = 1, index: int = None) -> 'Inference':
        """
        Resample the observation and return a new Inference object with the resampled observation, whose optimization
        starts from the estimate :attr:`params_inferred` as in :meth:`bootstrap`.
        This is useful when parallelizing bootstraps on a cluster. You can add performed bootstraps
        by using the :meth:`add_bootstrap` method. A replicate whose optimization raises in every run is recorded with
        NaN parameters and an infinite loss, as in :meth:`bootstrap`.

        :param n_runs: Number of optimization runs. The first run starts from the estimate and any further runs from
            start points sampled within the bounds.
        :param index: Index of the bootstrap replicate, such as a cluster job index. Replicates with distinct indices
            resample independently, reproducibly from :attr:`seed` and across reloads of a saved Inference object. By
            default, each call draws fresh entropy.
        :return: Inference object with the resampled observation.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        :raises ValueError: If ``n_runs`` is less than 1.
        """
        if not self.params_inferred:
            raise RuntimeError('The main optimization must be run first (call run()).')

        if int(n_runs) < 1:
            raise ValueError(f'n_runs must be at least 1, got {n_runs}.')

        other = self._spawn(index)
        other._x0 = dict(self.params_inferred)
        other.observation = self.resample(self.observation, other._rng)
        other.n_runs = n_runs
        other._is_bootstrap = True

        return other

    def add_bootstrap(self, bootstrap: 'Inference' | Dict[str, float]) -> None:
        """
        Add main optimization result from another Inference object as a bootstrap to the current Inference object.

        :param bootstrap: Either an Inference object or a dictionary of inferred parameters. A dictionary is added
            with a missing loss and result. An Inference object whose loss was invalid at every evaluated point, or
            whose optimization raised in every run, is added with NaN parameters, as in :meth:`bootstrap`.
        :raises RuntimeError: If the provided Inference object has not been run yet.
        :raises ValueError: If the dictionary keys differ from the parameter names.
        """
        if isinstance(bootstrap, Inference):
            if bootstrap.loss_inferred is None:
                raise RuntimeError('The provided Inference object must be run first (call run()).')

            params, result = bootstrap.params_inferred, bootstrap.result

            # a replicate whose every evaluation hit the penalty of the loss wrapper has no estimate, as in bootstrap()
            if _LOSS_PENALTY <= bootstrap.loss_inferred < np.inf:
                self._logger.warning(
                    'The loss of the bootstrap replicate was invalid at every evaluated point, so its parameters '
                    'are NaN.'
                )
                params = dict.fromkeys(params, np.nan)
                result = self._invalidate(OptimizeResult(result))

            row = params | dict(loss=bootstrap.loss_inferred, result=str(result))
        else:
            if set(bootstrap.keys()) != set(self.param_names):
                raise ValueError(f'Bootstrap parameters {list(bootstrap.keys())} must match {self.param_names}.')

            row = dict(bootstrap) | dict(loss=np.nan, result=None)

        self.bootstraps.loc[len(self.bootstraps)] = row

    def add_bootstraps(self, data: Iterable['Inference'] | Iterable[Dict[str, float]]) -> None:
        """
        Add bootstraps from an iterable of Inference objects or dictionaries of inferred parameters by calling
        :meth:`Inference.add_bootstrap() <phasegen.inference.Inference.add_bootstrap>` on each.

        :param data: Iterable of Inference objects or dictionaries of inferred parameters.
        """
        for d in data:
            self.add_bootstrap(d)


class _PenalizedLoss:
    """
    The loss of a parameter vector, with a finite penalty where the model raises or the loss is not finite.
    """

    def __init__(
            self,
            observation: Any,
            x0: Dict[str, float],
            pbar: tqdm | None,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float],
            logger: logging.Logger,
            smooth: bool = False,
            penalty: Callable[..., float] = None
    ) -> None:
        """
        Configure the loss.

        :param observation: Observation.
        :param x0: Initial parameters, for their names.
        :param pbar: Progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param logger: Logger.
        :param smooth: Whether the penalty lies just above the loss of the current iterate, rather than at
            :data:`_LOSS_PENALTY`, for a gradient-based optimizer.
        :param penalty: The penalty added to the loss, see :class:`Inference`.
        """
        self.observation = observation
        self.names = list(x0.keys())
        self.pbar = pbar
        self.get_dist = get_dist
        self.get_loss = get_loss
        self.logger = logger
        self.smooth = smooth
        self.regularization = penalty

        #: The loss of the current iterate, that of the first evaluation until
        #: :meth:`_PenalizedLoss.update() <phasegen.inference._PenalizedLoss.update>` reports an iterate.
        self.current: float | None = None

        #: Whether the last evaluation was penalized.
        self.invalid: bool = False

    @property
    def penalty(self) -> float:
        """
        The penalty of an invalid point.
        """
        if not self.smooth or self.current is None:
            return _LOSS_PENALTY

        # just above the current iterate, so that a line search rejects the point and bisects back toward the iterate,
        # where a cliff collapses its interpolated step and stops the optimization early
        return self.current + _PENALTY_MARGIN * max(abs(self.current), 1)

    def update(self, intermediate_result: OptimizeResult) -> None:
        """
        Record the loss of an iterate, as a callback of :func:`scipy.optimize.minimize`. Methods passing only the
        parameters leave the loss of the first evaluation in place.

        :param intermediate_result: The iterate.
        """
        if isinstance(intermediate_result, OptimizeResult) and np.isfinite(intermediate_result.get('fun', np.nan)):
            self.current = float(intermediate_result.fun)

    def evaluate(self, params: list) -> float | None:
        """
        Evaluate the loss with the penalty of :class:`Inference` but without that of an invalid point.

        :param params: The parameters, in the order of ``x0``.
        :return: The loss, or ``None`` where the model raises or the loss is not finite.
        """
        params_dict = dict(zip(self.names, params))

        # a model the parameters make invalid, such as a demography that cannot absorb on a bound of zero
        # migration, counts as a non-finite loss, so the optimizer steps away rather than the run being lost
        try:
            loss = self.get_loss(self.get_dist(**params_dict), self.observation)
        except (ModelError, np.linalg.LinAlgError) as e:
            self.logger.warning('The model raised "%s" for %s; substituting a finite penalty', e, params_dict)
            return None

        if self.regularization is not None and np.ndim(loss) == 0:
            loss = loss + self.regularization(**params_dict)

        # a non-finite loss (NaN or +/-inf) fed to the optimizer poisons its gradient and steps it to invalid
        # parameters, so it is penalized as well
        if np.ndim(loss) != 0 or not np.isfinite(loss):
            self.logger.warning('Loss function returned invalid value "%s" for %s; substituting a finite penalty',
                                loss, params_dict)
            return None

        return float(loss)

    def __call__(self, params: list) -> float:
        """
        Evaluate the loss.

        :param params: The parameters, in the order of ``x0``.
        :return: The loss, or the penalty.
        """
        loss = self.evaluate(params)
        self.invalid = loss is None

        if self.invalid:
            loss = self.penalty
            self.current = loss if self.current is None else self.current
        else:
            self.record(loss)

        Inference._report(dict(zip(self.names, params)), loss, self.pbar, self.logger)

        return loss

    def record(self, value: float) -> None:
        """
        Record a valid evaluation, taken as the current iterate if no evaluation preceded it.

        :param value: The loss.
        """
        if self.current is None:
            self.current = value


class WeightedLoss:  # pragma: no cover
    r"""
    Combination of loss components normalized by their running averages. For components :math:`c` with values
    :math:`L_c`, weights :math:`w_c` and running averages :math:`\bar{L}_c` over the most recent values passed to
    :meth:`WeightedLoss.compute() <phasegen.inference.WeightedLoss.compute>`, the combined loss is

    .. math::

        \sum_c L_c\, \frac{w_c / \bar{L}_c}{\sum_{c'} w_{c'} / \bar{L}_{c'}},

    so that each component contributes in proportion to its weight irrespective of its scale.

    The following example combines two loss components, weighting the second twice as much as the first.

    ::

        loss = pg.inference.WeightedLoss({'sfs': 1, 'fst': 2})

        total = loss.compute({'sfs': 0.3, 'fst': 0.01})
    """

    def __init__(self, weights: Dict[str, float], n_max: int | None = 100) -> None:
        """
        Initialize the class with the provided parameters.

        :param weights: Dictionary of weights :math:`w_c` for each component of the loss function.
        :param n_max: Maximum number of recent values in the running averages. Use ``None`` to consider all values.
        """
        #: Weights for each component of the loss function.
        self.weights: Dict[str, float] = weights

        #: Maximum recent values to consider for the average.
        self.n_max: int = n_max if n_max is not None else 10 ** 21

        #: Keys of the weights.
        self.keys: List[str] = list(weights.keys())

        #: Cached values.
        self.cache: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))

        #: Logger instance.
        self._logger = logger.getChild(self.__class__.__name__)

    @property
    def average(self) -> Dict[str, float]:
        r"""
        Running average :math:`\bar{L}_c` of each loss component over its most recent ``n_max`` values, keyed by
        component.
        """
        return {key: np.mean(self.cache[key][-self.n_max:]) for key in self.keys}

    def compute(self, loss: Dict[str, float]) -> float:
        """
        Compute the weighted loss.

        :param loss: Dictionary of loss values for each component of the loss function.
        :return: Weighted loss.
        """
        for key in self.keys:
            self.cache[key] = np.append(self.cache[key], loss[key])

        avg = self.average
        avg = np.array([avg[key] for key in self.keys])
        weights = np.array([self.weights[key] for key in self.keys])

        adjusted = weights / avg
        adjusted /= np.sum(adjusted)
        adjusted_dict = {key: value for key, value in zip(self.keys, adjusted)}

        self._logger.debug(f'loss: {loss}, weights: {adjusted_dict}')

        return np.sum([loss[key] * adjusted_dict[key] for key in self.keys])
