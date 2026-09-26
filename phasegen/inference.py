"""
Inference class for gradient-based parameter inference with respect to a specified loss function,
summary statistics, and a :class:`~phasegen.distributions.Coalescent` distribution.
"""

import copy
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
from .distributions import Coalescent
from .serialization import Serializable
from .settings import Settings
from .state_space import StateSpace
from .utils import parallelize

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .visualization import _CurveData

logger = logging.getLogger('phasegen')

#: Finite penalty substituted for a non-finite loss, large enough that any genuine loss wins the minimisation.
_LOSS_PENALTY = 1e100


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
    """
    #: Default options passed to the optimization algorithm.
    #: See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    default_opts = dict()

    #: Static for backward compatibility with serialized objects that lack the attribute.
    _entropy: int | None = None

    #: Static for backward compatibility with serialized objects whose initial guess was not materialized.
    _x0: Dict[str, float] | None = None

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
            method_mle: str = 'L-BFGS-B'
    ) -> None:
        """
        Initialize the class with the provided parameters.

        :param bounds: Dictionary of tuples representing the bounds for each
            parameter in x0.
        :param coal: Callback returning the configured coalescent distribution on which
            the inference is based on. The parameters specified in ``x0`` and ``bounds``
            are passed as keyword arguments.
        :param loss: The loss function, evaluating :math:`L(\\theta)`. This function must return a single numerical
            value that is to be minimized. It receives as first argument the coalescent
            distribution returned by the ``coal`` callback, and as second argument the
            observation passed to the ``observation`` argument (if any). A typical choice aggregates a
            :class:`~phasegen.norms.Norm` or :class:`~phasegen.norms.Likelihood` over observed and modelled
            summary statistics (e.g. the :class:`~phasegen.norms.PoissonLikelihood` for site-frequency-spectrum counts).
        :param x0: Dictionary of initial numeric guesses for parameters to optimize.
        :param observation: The observed summary statistic the inference is based on.
            This is passed as second argument to the ``loss`` function, and is only required
            if you want to use automatic bootstrapping.
        :param resample: Callback that resamples the observation. This is
            required for automatic bootstrapping. The resample function must accept
            the observation as first argument and a random number generator as second
            argument, and must return a resampled observation.
        :param n_runs: Number of independent optimization runs.
        :param n_bootstraps: Number of bootstrap replicates.
        :param do_bootstrap: Whether to perform automatic bootstrapping.
        :param parallelize: Whether to parallelize the computations across available CPU cores.
            ``Settings.parallelize = False`` overrides it.

            .. note:: Parallelization across multiple CPU cores is not always faster than single-threaded execution.
                It can also lead to hanging processes due to pickling issues, depending on how the
                provided callback function is defined. For more scalable parallelization, consider using the
                :meth:`create_run` and :meth:`create_bootstrap` methods to create new :class:`Inference` objects that can be
                run independently, and whose results can be merged subsequently.
        :param pbar: Whether to show a progress bar.
        :param seed: Seed for the random number generator.
        :param cache: Whether to reuse the state spaces of :attr:`Coalescent.state_spaces
            <phasegen.distributions.Coalescent.state_spaces>` across optimization iterations when they are equivalent,
            so that only their rate matrices are rebuilt.
            This speeds up optimizations over demographic parameters such as population sizes or migration rates.
        :param opts: Additional options passed to the optimization algorithm.
            See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
        :param method_mle: Method to use for optimization. See `scipy.optimize.minimize` for available methods.
        """
        if do_bootstrap and (observation is None or resample is None):
            raise ValueError('Observation and resample arguments must be provided for automatic bootstrapping.')

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

        #: Dictionary of initial numeric guesses for parameters to optimize, sampled within the bounds if not given.
        self._x0: Dict[str, float] = self._sample() if x0 is None else x0

        #: Whether to cache the state spaces
        self.cache: bool = cache

        #: Optimization options
        self.opts: Dict = self.default_opts | (opts or {})

        #: Optimization method
        self.method_mle: str = method_mle

        #: Optimization result
        self.result: OptimizeResult | None = None

        #: Inferred parameters.
        self.params_inferred: Dict = {}

        #: Loss of the best optimization run
        self.loss_inferred: float | None = None

        # losses for the `n_runs` independent optimization runs
        self.loss_runs: np.ndarray = np.array([])

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

        # canonicalize to `bounds` key order so that every run's `result.x` (ordered by the passed x0's keys) lines
        # up with `self.x0.keys()` and the DataFrame columns; `_sample()`-generated runs are already in bounds order
        return {key: self._x0[key] for key in self.bounds.keys()}

    def __getstate__(self) -> dict:
        """
        Get the state of the object for serialization.

        The ``coal``, ``loss`` and ``resample`` callables are serialized with ``dill`` (they are typically
        lambdas/closures that the standard pickler and ``copy.deepcopy`` cannot handle reliably, especially
        once they have themselves been restored from a previous ``dill`` round-trip). They are dumped
        directly from ``self`` without a prior deep copy. Only the remaining state is deep-copied, so the live object
        is left untouched.

        The dump is recursive, so that the module-level names a callable references (the package alias, the
        observation, helper functions) travel with it. A restored callable therefore evaluates against the
        namespace it was written with, in a worker process started by ``spawn`` and in a later session alike.

        :return: State of the object.
        """
        callables = ['coal', 'loss', 'resample']

        state = copy.deepcopy({key: value for key, value in self.__dict__.items() if key not in callables})

        # a plain dict, as jsonpickle encodes the instance dictionary of an OptimizeResult as an additional item
        if state['result'] is not None:
            state['result'] = dict(state['result'])

        for key in callables:
            state[f'{key}_pickled'] = dill.dumps(self.__dict__[key], recurse=True)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore the state of the object from a serialized state.

        :param state: State of the object.
        """
        self.__dict__.update(state)

        if self.result is not None:
            self.result = OptimizeResult(self.result)

        for key in ['coal', 'loss', 'resample']:
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

            for name, cached in self._state_spaces.items():
                if getattr(coal, name) == cached:
                    cached.update_epoch(coal.demography.get_epoch(0))
                    coal.__dict__[name] = cached

        return coal

    @cached_property
    def _state_spaces(self) -> Dict[str, StateSpace]:
        """
        The state spaces of the coalescent at ``x0``, reused across loss evaluations when caching is enabled. Only the
        rate matrices are recomputed per evaluation.
        """
        return self.coal(**self.x0).state_spaces

    @staticmethod
    def _get_loss_function(
            observation: Any,
            x0: Dict[str, float],
            pbar: tqdm | None,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float],
            logger: logging.Logger = logger
    ) -> Callable[[list], float]:
        """
        Get the loss function that accepts a list as an argument.

        :param observation: Observation.
        :param x0: Initial parameters.
        :param pbar: Progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param logger: Logger.
        :return: Loss function.
        """

        def loss(params: list) -> float:
            """
            Loss function that accepts a list as an argument.

            :param params: List of parameters to optimize.
            :return: Value of the loss function.
            """
            # convert the list of parameters back into a dictionary
            params_dict = dict(zip(x0.keys(), params))

            # a model the parameters make invalid, such as a demography that cannot absorb on a bound of zero
            # migration, counts as a non-finite loss, so the optimizer steps away rather than the run being lost
            try:
                loss = get_loss(get_dist(**params_dict), observation)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
                logger.warning('The model raised "%s" for %s; substituting a large finite penalty', e, params_dict)
                loss = np.nan

            # a non-finite loss (NaN or +/-inf) fed to the optimizer poisons its finite-difference gradient and
            # steps it to invalid parameters; substitute a large finite penalty so it stays in a valid region
            if np.ndim(loss) != 0 or not np.isfinite(loss):
                logger.warning(f'Loss function returned invalid value "{loss}" for {params_dict}; '
                               f'substituting a large finite penalty')
                loss = _LOSS_PENALTY

            loss = float(loss)

            data = params_dict | {'loss': loss}

            logger.debug(f"Current iteration: ({', '.join([f'{k}={v:.4f}' for k, v in data.items()])})")

            if pbar is not None:
                pbar.update()
                pbar.set_postfix(data)

            return loss

        return loss

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
            logger: logging.Logger = logger
    ) -> OptimizeResult:
        """
        Perform the optimization.

        :param observation: Observation.
        :param x0: Initial parameters.
        :param bounds: Bounds for the parameters.
        :param show_pbar: Whether to show a progress bar.
        :param get_dist: Callback returning the configured coalescent distribution.
        :param get_loss: Loss function.
        :param opts: Additional options passed to the optimization algorithm.
        :param logger: Logger.
        :return: Result of the optimization procedure.
        """
        # convert dictionaries to lists
        bounds = [bounds[key] for key in x0.keys()]

        pbar = tqdm(desc='Optimizing loss function') if show_pbar else None

        # get the loss function
        loss = Inference._get_loss_function(
            observation=observation,
            x0=x0,
            pbar=pbar,
            get_dist=get_dist,
            get_loss=get_loss,
            logger=logger
        )

        # perform the optimization
        result: OptimizeResult = opt.minimize(
            fun=loss,
            x0=np.array(list(x0.values())),
            method=method_mle,
            bounds=bounds,
            options=opts
        )

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
        logger = self._logger

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
                    logger=logger
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

    def _sample(self) -> Dict[str, float]:
        """
        Sample initial parameters by using the provided bounds.

        :return: Sampled parameters.
        """
        return {key: self._rng.uniform(*bounds) for key, bounds in self.bounds.items()}

    def run(self) -> None:
        """
        Execute the optimization.
        """
        self._run()

        if self.do_bootstrap:
            self.bootstrap()

    def bootstrap(self) -> None:
        """
        Perform bootstrapping to estimate parameter uncertainty. For each of :attr:`n_bootstraps` replicates the
        observation :math:`y` is resampled to :math:`y^{*}` via the ``resample`` callback and the inference is rerun,
        yielding :math:`\\hat{\\theta}^{*} = \\arg\\min_{\\theta} L(\\mathrm{coal}(\\theta),\\, y^{*})`. The spread of the
        replicate estimates :math:`\\{\\hat{\\theta}^{*}_b\\}` estimates the sampling distribution of :math:`\\hat{\\theta}`.
        Each replicate starts from :math:`\\hat{\\theta}`, and the replicates are stored in
        :attr:`Inference.bootstraps <phasegen.inference.Inference.bootstraps>`.

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
        logger = self._logger

        def run_sample(observation: Any) -> OptimizeResult:
            """
            Run a single bootstrap sample.

            :param observation: Observation.
            :return: Bootstrap sample.
            """
            # run the optimization
            return Inference._optimize(
                observation=observation,
                x0=x0,
                bounds=bounds,
                show_pbar=False,
                get_dist=get_dist,
                get_loss=get_loss,
                opts=opts,
                method_mle=method_mle,
                logger=logger
            )

        results = parallelize(
            func=run_sample,
            data=[self.resample(self.observation, self._rng) for _ in range(self.n_bootstraps)],
            parallelize=self.parallelize,
            pbar=self.pbar,
            desc='Bootstrapping',
            dtype=object
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
        The demography of each bootstrap replicate.

        :return: One demography per row of :attr:`_bootstrap_values`.
        """
        return [self.get_coal(**dict(zip(self.param_names, row))).demography for row in self._bootstrap_values]

    def _plot_demography_data(
            self,
            t: np.ndarray = None,
            kind: Literal['all', 'pop_sizes', 'migration'] = 'all',
            include_bootstraps: bool = True
    ) -> Tuple['_CurveData', List['_CurveData']]:
        """
        Trajectories of the inferred demography and of the demography of each bootstrap replicate, as drawn by
        :meth:`plot_demography`, :meth:`plot_pop_sizes` and :meth:`plot_migration`.

        :param t: Times at which to evaluate the trajectories. By default, :attr:`Settings.plot_inference_n_grid`
            points up to the :attr:`Settings.plot_inference_quantile` quantile of the inferred tree height, or up to
            the end time of a windowed coalescent.
        :param kind: The trajectories to include, ``'pop_sizes'``, ``'migration'`` or ``'all'``.
        :param include_bootstraps: Whether to include the bootstrap replicates.
        :return: The inferred trajectories, and the trajectories of each bootstrap replicate.
        :raises RuntimeError: If the main optimization has not been run.
        """
        if self.dist_inferred is None:
            raise RuntimeError('The main optimization must be run first (call run()).')

        if t is None:
            tree_height = self.dist_inferred.tree_height

            # a windowed coalescent has no tree-height quantile, so its end time bounds the plot
            if tree_height._windowed:
                t_end = tree_height.t_max
            else:
                t_end = tree_height.quantile(Settings.plot_inference_quantile)

            t = np.linspace(0, t_end, Settings.plot_inference_n_grid)

        bootstraps = self._bootstrap_demographies if include_bootstraps else []

        return self.dist_inferred.demography._plot_data(t, kind), [d._plot_data(t, kind) for d in bootstraps]

    def plot_bootstraps(
            self,
            title: str | List[str] = None,
            show: bool = True,
            file: str = None,
            subplots: bool = True,
            kind: Literal['hist', 'kde'] = 'hist',
            ax: 'plt.Axes' | List['plt.Axes'] = None,
            kwargs: dict = None
    ) -> 'plt.Axes' | List['plt.Axes']:
        """
        Plot bootstrapped parameters.

        :param title: Title or list of titles.
        :param show: Whether to show the plot.
        :param file: File to save the plot.
        :param subplots: Whether to plot subplots.
        :param kind: Kind of plot. Either 'hist' or 'kde'.
        :param ax: Axes or list of axes.
        :param kwargs: Additional keyword arguments passed to the pandas plot function.
        :return: Axes or list of axes.
        """
        from .visualization import Visualization
        import matplotlib.pyplot as plt

        if kwargs is None:
            kwargs = {}

        if self.bootstraps is None:
            raise RuntimeError('No bootstraps available.')

        if kind == 'hist':
            kwargs = {'bins': 20} | kwargs

        # avoid empty plots
        # plt.close()

        ax = self.bootstraps[self.param_names].plot(
            ax=ax,
            kind=kind,
            title='Marginal distributions' if title is None else title,
            subplots=subplots,
            **kwargs
        )

        # make layout tight
        plt.tight_layout()

        Visualization.show_and_save(show=show, file=file)

        return ax

    def plot_demography(
            self,
            t: np.ndarray = None,
            include_bootstraps: bool = True,
            show: bool = True,
            file: str = None,
            kwargs: dict = None,
            ax: List['plt.Axes'] | None = None
    ) -> List['plt.Axes']:
        """
        Plot inferred demography.

        :param t: Time points. By default, 100 time points are used that extend
            from 0 to the 99th percentile of the tree height distribution.
        :param include_bootstraps: Whether to include bootstraps.
        :param show: Whether to show the plot.
        :param file: File to save the plot.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: List of axes to plot on.
        :return: List of axes.
        """
        return self._plot_demography(
            t=t,
            show=show,
            include_bootstraps=include_bootstraps,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='all'
        )

    def plot_pop_sizes(
            self,
            t: np.ndarray = None,
            show: bool = True,
            include_bootstraps: bool = True,
            file: str = None,
            kwargs: dict = None,
            ax: Optional['plt.Axes'] = None
    ) -> 'plt.Axes':
        """
        Plot inferred population sizes.

        :param t: Time points. By default, 100 time points are used that extend
            from 0 to the 99th percentile of the tree height distribution.
        :param show: Whether to show the plot.
        :param include_bootstraps: Whether to include bootstraps.
        :param file: File to save the plot.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: List of axes to plot on.
        :return: Axes.
        """
        return self._plot_demography(
            t=t,
            show=show,
            include_bootstraps=include_bootstraps,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='pop_sizes'
        )

    def plot_migration(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            include_bootstraps: bool = True,
            kwargs: dict = None,
            ax: Optional['plt.Axes'] = None
    ) -> 'plt.Axes':
        """
        Plot inferred migration rates.

        :param t: Time points. By default, 100 time points are used that extend
            from 0 to the 99th percentile of the tree height distribution.
        :param show: Whether to show the plot.
        :param file: File to save the plot.
        :param include_bootstraps: Whether to include bootstraps.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :param ax: List of axes to plot on.
        :return: Axes.
        """
        return self._plot_demography(
            t=t,
            show=show,
            include_bootstraps=include_bootstraps,
            file=file,
            kwargs=kwargs,
            ax=ax,
            kind='migration'
        )

    def _plot_demography(
            self,
            t: np.ndarray,
            show: bool,
            include_bootstraps: bool,
            ax: Optional['plt.Axes'],
            kind: Literal['pop_sizes', 'migration', 'all'],
            file: str = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the trajectories of :meth:`_plot_demography_data`.

        :param t: Time points, ``None`` for the default of :meth:`_plot_demography_data`.
        :param show: Whether to show the plot.
        :param include_bootstraps: Whether to include bootstraps.
        :param ax: Axes to plot on.
        :param kind: The trajectories to include.
        :param file: File to save the plot.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from .visualization import Visualization

        if kwargs is None:
            kwargs = {}

        inferred, bootstraps = self._plot_demography_data(t, kind, include_bootstraps)

        if ax is None:
            plt.close()
            ax = plt.gca()

        Visualization.plot_rates(ax=ax, data=inferred, show=False, kwargs=kwargs)

        # each bootstrap trajectory in the colour of its series, without a legend entry of its own
        colors = [line.get_color() for line in ax.lines[-len(inferred.labels):]]

        for data in bootstraps:
            for y, color in zip(data.y, colors):
                style = {'color': color, 'alpha': 0.3} | kwargs
                ax.plot(data.x, y, drawstyle='steps-post', label='_nolegend_', **style)

        Visualization.show_and_save(show=show, file=file)

        return ax

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

        # the spawned object performs a single optimization whose result is merged back with ``add_run``, which reads
        # only the main result; bootstrapping it would repeat ``n_bootstraps`` fits per job and discard every one
        other.do_bootstrap = False

        # the copy starts unfitted, so merging it back before it has run is rejected
        other.result = None
        other.params_inferred = {}
        other.loss_inferred = None
        other.loss_runs = np.array([])
        other.dist_inferred = None
        other.bootstraps = self.bootstraps.iloc[0:0].copy()
        other.runs = self.runs.iloc[0:0].copy()

        if index is None:
            sequence = np.random.SeedSequence()
        else:
            sequence = np.random.SeedSequence(self._entropy, spawn_key=(int(index),))

        other.seed = int(sequence.generate_state(1)[0])
        other._entropy = other.seed
        other._rng = np.random.default_rng(np.random.SeedSequence(other._entropy))

        return other

    def create_run(self, x0: Dict[str, float] = None, index: int = None) -> 'Inference':
        """
        Create a new Inference object which can be run independently. This is useful when parallelizing runs on a
        cluster. You can add performed runs by using the :meth:`add_run` method.

        :param x0: Initial parameters. By default, they are sampled within the bounds.
        :param index: Index of the run, such as a cluster job index. Runs with distinct indices sample independent
            start points, reproducibly from :attr:`seed` and across reloads of a saved Inference object. By default,
            each call draws fresh entropy.
        :return: Inference object.
        """
        other = self._spawn(index)
        other._x0 = other._sample() if x0 is None else x0
        other._check_x0_within_bounds()

        return other

    def add_run(self, inference: 'Inference') -> None:
        """
        Merge the main optimization result from another Inference object into the current Inference object. We only
        store the result of the run with the lowest loss.

        :param inference: Inference object.
        :raises RuntimeError: If the main optimization has not been run yet.
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
        :meth:`Inference.add_run() <phasegen.Inference.add_run>` on each.

        :param inferences: Iterable of Inference objects.
        """
        for inference in inferences:
            self.add_run(inference)

    def create_bootstrap(self, n_runs: int = 1, index: int = None) -> 'Inference':
        """
        Resample the observation and return a new Inference object with the resampled observation, whose optimization
        starts from the estimate :attr:`params_inferred` as in :meth:`bootstrap`.
        This is useful when parallelizing bootstraps on a cluster. You can add performed bootstraps
        by using the :meth:`add_bootstrap` method.

        :param n_runs: Number of optimization runs. The first run starts from the estimate and any further runs from
            start points sampled within the bounds.
        :param index: Index of the bootstrap replicate, such as a cluster job index. Replicates with distinct indices
            resample independently, reproducibly from :attr:`seed` and across reloads of a saved Inference object. By
            default, each call draws fresh entropy.
        :return: Inference object with the resampled observation.
        :raises RuntimeError: If :meth:`Inference.run() <phasegen.inference.Inference.run>` has not been called.
        """
        if not self.params_inferred:
            raise RuntimeError('The main optimization must be run first (call run()).')

        other = self._spawn(index)
        other._x0 = dict(self.params_inferred)
        other.observation = self.resample(self.observation, other._rng)
        other.n_runs = n_runs

        return other

    def add_bootstrap(self, bootstrap: 'Inference' | Dict[str, float]) -> None:
        """
        Add main optimization result from another Inference object as a bootstrap to the current Inference object.

        :param bootstrap: Either an Inference object or a dictionary of inferred parameters. A dictionary is added
            with a missing loss and result.
        :raises RuntimeError: If the provided Inference object has not been run yet.
        :raises ValueError: If the dictionary keys differ from the parameter names.
        """
        if isinstance(bootstrap, Inference):
            if bootstrap.loss_inferred is None:
                raise RuntimeError('The provided Inference object must be run first (call run()).')

            row = bootstrap.params_inferred | dict(loss=bootstrap.loss_inferred, result=str(bootstrap.result))
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


class WeightedLoss:  # pragma: no cover
    r"""
    Combination of loss components normalized by their running averages. For components :math:`c` with values
    :math:`L_c`, weights :math:`w_c` and running averages :math:`\bar{L}_c` over the most recent values passed to
    :meth:`WeightedLoss.compute() <phasegen.inference.WeightedLoss.compute>`, the combined loss is

    .. math::

        \sum_c L_c\, \frac{w_c / \bar{L}_c}{\sum_{c'} w_{c'} / \bar{L}_{c'}},

    so that each component contributes in proportion to its weight irrespective of its scale.
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
        """
        Average of the cached values.
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
