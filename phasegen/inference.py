import logging
from functools import cached_property
from typing import Dict, Tuple, Callable, Any

import dill
import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from .distributions import Coalescent
from .serialization import Serializable
from .state_space import BlockCountingStateSpace, DefaultStateSpace
from .utils import parallelize

logger = logging.getLogger('phasegen')


class Inference(Serializable):
    """
    The Inference class is designed to perform arbitrary inference using the
    provided loss function, through coalescent simulation based on phase-type
    theory. The optimization is performed via the BFGS algorithm from scipy.

    TODO run main inference several times and take the best result
    """
    #: Default options passed to the optimization algorithm.
    #: See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    default_opts = dict()

    def __init__(
            self,
            x0: Dict[str, float | int | None],
            bounds: Dict[str, Tuple[float | int | None, float | int | None]],
            dist: Callable[..., Coalescent],
            loss: Callable[[Coalescent, Any], float],
            observation: Any = None,
            resample: Callable[[Any], Any] = None,
            n_bootstrap: int = 100,
            do_bootstrap: bool = False,
            parallelize: bool = True,
            pbar: bool = True,
            cache: bool = True,
            opts: Dict = None
    ):
        """
        Initialize the class with the provided parameters.

        :param x0: Dictionary of initial numeric guesses for parameters to optimize.
        :param bounds: Dictionary of tuples representing the bounds for each
            parameter in x0.
        :param dist: Callback returning the configured coalescent distribution on which
            the inference is based on. The parameters specified in ``x0`` and ``bounds``
            are passed as keyword arguments.
        :param loss: The loss function. This function must return a single numerical
            value that is to be minimized. It receives as first argument the coalescent
            distribution returned by the ``dist`` callback, and as second argument the
            observation passed to the ``observation`` argument (if any).
        :param observation: The observation. This is passed as second argument to the
            ``loss`` function, and is only required if you want to use automatic
            bootstrapping.
        :param resample: Callback that is used to resample the observation. This is
            required for automatic bootstrapping.
        :param n_bootstrap: Number of bootstrap replicates.
        :param do_bootstrap: Whether to perform automatic bootstrapping.
        :param parallelize: Whether to parallelize the simulations.
        :param pbar: Whether to show a progress bar.
        :param cache: Whether to cache the state spaces across the given optimization iterations given
            that they are equivalent. The can significantly speed up the optimization as we do not
            require to recompute the complete state spaces for each iteration. This only leads to
            performance improvements if optimizing demographic parameters such as population sizes
            or migration rates.
        :param opts: Additional options passed to the optimization algorithm.
            See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
        """
        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Dictionary of initial numeric guesses for parameters to optimize.
        self.x0: Dict[str, float | int | None] = x0

        #: Dictionary of tuples representing the bounds for each parameter in x0.
        self.bounds: Dict[str, Tuple[float | int | None, float | int | None]] = bounds

        #: Callback returning the configured coalescent distribution.
        self.dist: Callable[..., Coalescent] = dist

        #: Loss function.
        self.loss: Callable[[Coalescent, Any], float] = loss

        #: The observation.
        self.observation: Any = observation

        #: Callback that is used to resample the observation.
        self.resample: Callable[[Any], Any] = resample

        #: Number of bootstrap replicates.
        self.n_bootstrap: int = n_bootstrap

        #: Whether to perform automatic bootstrapping.
        self.do_bootstrap: bool = do_bootstrap

        #: Whether to parallelize the simulations.
        self.parallelize: bool = parallelize

        #: Whether to show a progress bar.
        self.pbar: bool = pbar

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

        #: Coalescent distribution of best run
        self.dist_inferred: Coalescent | None = None

        #: Bootstrap replicates
        self.bootstraps: np.ndarray | None = None

    def __getstate__(self) -> dict:
        """
        Get the state of the object for serialization.

        :return: State of the object.
        """
        state = self.__dict__.copy()

        for key in ['dist', 'loss', 'resample']:
            state[f'{key}_pickled'] = dill.dumps(state[key])
            state.pop(key)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Set the state of the object from deserialization.

        :param state: State of the object.
        """
        self.__dict__.update(state)

        for key in ['dist', 'loss', 'resample']:
            setattr(self, key, dill.loads(state[f'{key}_pickled']))
            del self.__dict__[f'{key}_pickled']

    def get_dist(self, **kwargs) -> Coalescent:
        """
        Get the (possibly cached) coalescent distribution.

        TODO test state space caching by comparing cached und uncached results.

        :param kwargs: Keyword arguments passed to the callback specified as ``dist``.
        :return: Coalescent distribution.
        """
        dist = self.dist(**kwargs)

        # if state space caching is enabled, replace by cached state space if possible
        if self.cache:

            if dist.default_state_space == self.default_state_space:
                dist.__dict__['default_state_space'] = self.default_state_space

            if dist.block_counting_state_space == self.block_counting_state_space:
                dist.__dict__['block_counting_state_space'] = self.block_counting_state_space

        return dist

    @cached_property
    def default_state_space(self) -> DefaultStateSpace:
        """
        Default state space.

        :return: Default state space.
        """
        return self.dist(**self.x0).default_state_space

    @cached_property
    def block_counting_state_space(self) -> BlockCountingStateSpace:
        """
        Block counting state space.

        :return: Block counting state space.
        """
        return self.dist(**self.x0).block_counting_state_space

    @staticmethod
    def _get_loss_function(
            observation: Any,
            x0: Dict[str, float],
            pbar: tqdm | None,
            get_dist: Callable[..., Coalescent],
            get_loss: Callable[[Coalescent, Any], float]
    ) -> Callable[[list], float]:
        """
        Get the loss function that accepts a list as an argument.

        :param observation: Observation.
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

            if pbar is not None:
                pbar.update()
                pbar.set_postfix(**{'loss': loss} | params_dict)

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
            opts: Dict = None
    ) -> OptimizeResult:
        """
        Perform the optimization.

        :param loss: Loss function.
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
            get_loss=get_loss
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
        # perform the optimization
        self.result: OptimizeResult = self._optimize(
            observation=self.observation,
            x0=self.x0,
            bounds=self.bounds,
            show_pbar=self.pbar,
            get_dist=self.get_dist,
            get_loss=self.loss,
            opts=self.opts
        )

        # fetch optimized params
        self.params_inferred = dict(zip(list(self.x0.keys()), self.result.x))

        # loss of best run
        self.loss_inferred = self.result.fun

        # coalescent distribution of best run
        self.dist_inferred = self.get_dist(**self.params_inferred)

        # return the result of the optimization
        return self.result

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
        get_dist = self.get_dist
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
                opts=opts
            )

        results = parallelize(
            func=run_sample,
            data=[self.resample(self.observation) for _ in range(self.n_bootstrap)],
            parallelize=self.parallelize,
            pbar=self.pbar,
            desc='Bootstrapping'
        )

        pass
