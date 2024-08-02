"""
Inference module.
"""

import copy
import logging
from functools import cached_property
from typing import Dict, Tuple, Callable, Any, List, Literal, Iterable, Optional

import dill
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from .demography import Demography
from .distributions import Coalescent
from .serialization import Serializable
from .state_space import BlockCountingStateSpace, LineageCountingStateSpace
from .utils import parallelize

logger = logging.getLogger('phasegen')


class Inference(Serializable):
    """
    Gradient-based parameter inference with respect to a specified loss function,
    summary statistics, and a :class:`~phasegen.distributions.Coalescent` distribution.
    The optimization is performed via the BFGS algorithm from scipy.

    .. note::
        TODO there are problems when pickling this object if is has already been unpickled previously.
    """
    #: Default options passed to the optimization algorithm.
    #: See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    default_opts = dict()

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
            parallelize: bool = True,
            pbar: bool = True,
            seed: int = None,
            cache: bool = True,
            opts: Dict = None
    ):
        """
        Initialize the class with the provided parameters.

        :param bounds: Dictionary of tuples representing the bounds for each
            parameter in x0.
        :param coal: Callback returning the configured coalescent distribution on which
            the inference is based on. The parameters specified in ``x0`` and ``bounds``
            are passed as keyword arguments.
        :param loss: The loss function. This function must return a single numerical
            value that is to be minimized. It receives as first argument the coalescent
            distribution returned by the ``dist`` callback, and as second argument the
            observation passed to the ``observation`` argument (if any).
        :param x0: Dictionary of initial numeric guesses for parameters to optimize.
        :param observation: The observed summary statistic the inference is based on.
            This is passed as second argument to the ``loss`` function, and is only required
            if you want to use automatic bootstrapping.
        :param resample: Callback that is used to resample the observation. This is
            required for automatic bootstrapping. The resample function must accept
            the observation as first argument and a random number generator as second
            argument, and must return a resampled observation.
        :param n_runs: Number of independent optimization runs.
        :param n_bootstraps: Number of bootstrap replicates.
        :param do_bootstrap: Whether to perform automatic bootstrapping.
        :param parallelize: Whether to parallelize the computations.
        :param pbar: Whether to show a progress bar.
        :param seed: Seed for the random number generator.
        :param cache: Whether to cache the state spaces across the given optimization iterations given
            that they are equivalent. The can significantly speed up the optimization as we do not
            require to recompute the complete state spaces for each iteration. This only leads to
            performance improvements if optimizing demographic parameters such as population sizes
            or migration rates.
        :param opts: Additional options passed to the optimization algorithm.
            See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
        """
        if do_bootstrap and (observation is None or resample is None):
            raise ValueError('Observation and resample arguments must be provided for automatic bootstrapping.')

        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Dictionary of initial numeric guesses for parameters to optimize.
        self._x0: Dict[str, float] | None = x0

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

        #: Random number generator.
        self._rng: np.random.Generator = np.random.default_rng(seed)

        #: Whether to cache the state spaces
        self.cache: bool = cache

        if opts is None:
            #: Optimization options
            self.opts = self.default_opts
        else:
            #: Optimization options
            self.opts = self.default_opts | opts

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
        self.bootstraps: pd.DataFrame = pd.DataFrame(columns=list(self.bounds.keys()))

    def _check_x0_within_bounds(self):
        """
        Check if the initial parameters are within the specified bounds.
        """
        if not all([self.bounds[key][0] <= value <= self.bounds[key][1] for key, value in self.x0.items()]):
            raise ValueError('Initial parameters must be within the specified bounds.')

    @cached_property
    def x0(self) -> Dict[str, float]:
        """
        Initial parameters.
        """
        return self._x0 if self._x0 is not None else self._sample()

    def __getstate__(self) -> dict:
        """
        Get the state of the object for serialization.

        :return: State of the object.
        """
        state = copy.deepcopy(self.__dict__)

        for key in ['coal', 'loss', 'resample']:
            state[f'{key}_pickled'] = dill.dumps(state[key])
            state.pop(key)

        return state

    def __setstate__(self, state: dict):
        """
        Restore the state of the object from a serialized state.

        :param state: State of the object.
        """
        self.__dict__.update(state)

        for key in ['coal', 'loss', 'resample']:
            setattr(self, key, dill.loads(state[f'{key}_pickled']))
            self.__dict__.pop(f'{key}_pickled')

    def get_coal(self, **kwargs) -> Coalescent:
        """
        Get the (possibly cached) coalescent distribution.

        :param kwargs: Keyword arguments passed to the callback specified as ``dist`.
        :return: Coalescent distribution.
        """
        coal = self.coal(**kwargs)

        # disable parallelization to avoid problematic double parallelization
        coal.parallelize = False

        # if state space caching is enabled, replace by cached state space if possible
        if self.cache:

            if coal.lineage_counting_state_space == self._lineage_counting_state_space:
                coal.__dict__['lineage_counting_state_space'] = self._lineage_counting_state_space

            if coal.block_counting_state_space == self._block_counting_state_space:
                coal.__dict__['block_counting_state_space'] = self._block_counting_state_space

        return coal

    @cached_property
    def _lineage_counting_state_space(self) -> LineageCountingStateSpace:
        """
        Lineage-counting state space which only keeps track of the number of lineages present.
        """
        return self.coal(**self.x0).lineage_counting_state_space

    @cached_property
    def _block_counting_state_space(self) -> BlockCountingStateSpace:
        """
        Block-counting state space which keeps track for the number of lineages that subtend `i` lineages.
        """
        return self.coal(**self.x0).block_counting_state_space

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

            # get the coalescent distribution
            dist = get_dist(**params_dict)

            # return the value of the loss function
            loss = get_loss(dist, observation)

            data = params_dict | {'loss': loss}

            logger.debug(f"Current iteration: ({', '.join([f'{k}={v:.4f}' for k, v in data.items()])})")

            if not np.isscalar(loss) or np.isnan(loss):
                logger.warning(f'Loss function returned invalid value "{loss}" for {params_dict}')

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
            method='L-BFGS-B',
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

        def run_sample(x0: Dict[str, float]) -> OptimizeResult:
            """
            Run a single bootstrap sample.

            :param x0: Initial parameters.
            :return: Bootstrap sample.
            """
            # perform the optimization
            return self._optimize(
                observation=observation,
                x0=x0,
                bounds=bounds,
                show_pbar=False,
                get_dist=get_dist,
                get_loss=get_loss,
                opts=opts,
                logger=self._logger
            )

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

        # get the best result
        self.result = min(results, key=lambda result: result.fun)

        # fetch optimized params
        self.params_inferred = dict(zip(list(self.x0.keys()), self.result.x))

        self._logger.info(
            f'Inferred parameters: ({", ".join([f"{k}={v:.4f}" for k, v in self.params_inferred.items()])})'
        )

        # loss of best run
        self.loss_inferred = self.result.fun

        # loss for all runs
        self.loss_runs = np.array([result.fun for result in results])

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

    def run(self):
        """
        Execute the optimization.
        """
        self._run()

        if self.do_bootstrap:
            self.bootstrap()

    def bootstrap(self):
        """
        Perform bootstrapping.

        :return: Bootstrap replicates.
        """
        if self.params_inferred is None:
            raise RuntimeError('The main optimization must be run first (call the `run` method).')

        x0 = self.params_inferred
        bounds = self.bounds
        get_dist = self.get_coal
        get_loss = self.loss
        opts = self.opts

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
                logger=self._logger
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
                f'Only {n_success} out of {self.n_bootstraps} bootstrap replicates converged.'
            )

        # store bootstrapped parameters
        self.bootstraps = pd.DataFrame([result.x for result in results], columns=list(self.x0.keys()))

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
        plt.close()

        ax = self.bootstraps.plot(
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
            kind='pop_size'
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
            kind: Literal['pop_size', 'migration', 'all'],
            file: str = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot inferred population sizes, migration rates, or both.

        :param t: Time points. By default, 100 time points are used that extend
            from 0 to the 99th percentile of the tree height distribution.
        :param show: Whether to show the plot.
        :param include_bootstraps: Whether to include bootstraps.
        :param ax: Axes to plot on.
        :param file: File to save the plot.
        :param kwargs: Additional keyword arguments passed to the plot function.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from .visualization import Visualization

        if kwargs is None:
            kwargs = {}

        if self.dist_inferred is None:
            raise RuntimeError('The main optimization must be run first (call the `run` method).')

        if t is None:
            t = np.linspace(0, self.dist_inferred.tree_height.quantile(0.99), 100)

        # mapping of kind to plot function
        funcs = dict(
            all='plot',
            pop_size='plot_pop_sizes',
            migration='plot_migration'
        )

        if ax is None:
            plt.close()
            ax = plt.gca()

        def plot(d: Demography, kwargs2: dict) -> 'plt.Axes':
            """
            Plot inferred demography.

            :param d: Demography.
            :param kwargs2: Additional keyword arguments passed to the plot function.
            :return: Axes.
            """
            getattr(d, funcs[kind])(
                t=t,
                ax=ax,
                show=False,
                kwargs=kwargs2 | kwargs
            )

            return ax

        plot(self.dist_inferred.demography, {'color': 'C0'})

        # plot bootstrapped demography
        if include_bootstraps:
            for i, row in self.bootstraps.iterrows():
                plot(self.get_coal(**row.to_dict()).demography, {'color': 'C0', 'alpha': 0.3})

        Visualization.show_and_save(show=show, file=file)

        return ax

    def create_run(self, x0: Dict[str, float] = None) -> 'Inference':
        """
        Create a new Inference object which can be run independently. This is useful when parallelizing runs on a
        cluster. You can add performed runs by using the `add_run` method.

        :param x0: Initial parameters.
        :return: Inference object.
        """
        other = copy.deepcopy(self)

        other._x0 = x0
        other._check_x0_within_bounds()

        # generate a new random seed if seeded
        if other.seed is not None:
            other.seed = self._rng.integers(0, 2 ** 32 - 1)
            other._rng = np.random.default_rng(other.seed)

        return other

    def add_run(self, inference: 'Inference'):
        """
        Merge the main optimization result from another Inference object into the current Inference object. We only
        store the result of the run with the lowest loss.

        :param inference: Inference object.
        :raises RuntimeError: If the main optimization has not been run yet.
        """
        if inference.loss_inferred is None:
            raise RuntimeError('The provided Inference object must be run first (call the `run` method).')

        # add the loss of the new run to the list of losses
        self.loss_runs = np.append(self.loss_runs, inference.loss_runs)

        # update the result if the loss of the new run is lower
        if self.loss_inferred is None or inference.loss_inferred < self.loss_inferred:
            self.result = inference.result
            self.params_inferred = inference.params_inferred
            self.loss_inferred = inference.loss_inferred
            self.dist_inferred = inference.dist_inferred

    def add_runs(self, inferences: Iterable['Inference']):
        """
        Merge the main optimization results from an iterable of Inference objects with the current Inference object. We
        only store the result of the run with the lowest loss.

        :param inferences: Iterable of Inference objects.
        """
        for inference in inferences:
            self.add_run(inference)

    def create_bootstrap(self, n_runs: int = 1) -> 'Inference':
        """
        Resample the observation and return a new Inference object with the resampled observation.
        This is useful when parallelizing bootstraps on a cluster. You can add performed bootstraps
        by using the `add_bootstrap` method.

        :return: Resampled observation.
        """
        other = copy.deepcopy(self)

        other.observation = self.resample(other.observation, self._rng)
        other.n_runs = n_runs

        return other

    def add_bootstrap(self, data: 'Inference' | Dict[str, float]):
        """
        Add main optimization result from another Inference object as a bootstrap to the current Inference object.

        :param data: Either an Inference object or a dictionary of inferred parameters.
        :raises RuntimeError: If the main optimization has not been run yet.
        """
        if isinstance(data, dict):
            # add bootstrap parameters
            self.bootstraps.loc[len(self.bootstraps)] = data

        elif isinstance(data, Inference):

            if data.loss_inferred is None:
                raise RuntimeError('The provided Inference object must be run first (call the `run` method).')

            self.add_bootstrap(data.params_inferred)

        else:
            raise ValueError('Invalid data type.')

    def add_bootstraps(self, data: Iterable['Inference'] | Iterable[Dict[str, float]]):
        """
        Add bootstraps from an iterable of Inference objects.

        :param data: Iterable of Inference objects or dictionaries of inferred parameters.
        """
        for d in data:
            self.add_bootstrap(d)
