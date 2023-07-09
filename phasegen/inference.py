from typing import Dict, Tuple, Callable

import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from .distributions import Coalescent


class Inference:
    """
    The Inference class is designed to perform arbitrary inference using the
    provided loss function, through coalescent simulation based on phase-type
    theory. The optimization is performed via the BFGS algorithm from scipy.

    :Example:

    >>> def loss_fn(a=0, b=0):
    ...     return (a - 2) ** 2 + (b - 3) ** 2
    ...
    >>> x0 = {'a': 0, 'b': 0}
    >>> bounds = {'a': (None, None), 'b': (None, None)}
    >>> inf = Inference(x0, bounds, loss_fn, None)
    >>> result = inf.run()
    """

    def __init__(
            self,
            x0: Dict[str, float | int | None],
            bounds: Dict[str, Tuple[float | int | None, float | int | None]],
            dist: Callable[..., Coalescent],
            loss: Callable[[Coalescent], float],
            parallelize: bool = True,
            pbar: bool = True
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
            value that is to be minimized. It receives as argument the coalescent
            distribution returned by the callback specified as ``dist``.
        :param parallelize: Whether to parallelize the simulations.
        :param pbar: Whether to show a progress bar.
        """
        #: Dictionary of initial numeric guesses for parameters to optimize.
        self.x0: Dict[str, float | int | None] = x0

        #: Dictionary of tuples representing the bounds for each parameter in x0.
        self.bounds: Dict[str, Tuple[float | int | None, float | int | None]] = bounds

        #: Callback returning the configured coalescent distribution.
        self.dist: Callable[..., Coalescent] = dist

        #: Loss function.
        self.loss: Callable[[Coalescent], float] = loss

        #: Whether to parallelize the simulations.
        self.parallelize: bool = parallelize

        #: Whether to show a progress bar.
        self.pbar: bool = pbar

        #: Optimization result
        self.result: OptimizeResult | None = None

        #: Inferred parameters.
        self.params_inferred: Dict = {}

        #: Loss of the best optimization run
        self.loss_inferred: float | None = None

        #: Coalescent distribution of best run
        self.dist_inferred: Coalescent | None = None

        #: Progress bar
        self.tqdm = tqdm(desc='Optimizing loss function')

    def run(self) -> OptimizeResult:
        """
        Execute the optimization.

        :returns: Result of the optimization procedure.
        :rtype: scipy.optimize.OptimizeResult
        """

        # convert dictionaries to lists
        x0 = np.array(list(self.x0.values()))
        bounds = [self.bounds[key] for key in self.x0.keys()]

        def loss(params: list) -> float:
            """
            Loss function that accepts a list as an argument.

            :param params: List of parameters to optimize.
            :return: Value of the loss function.
            """
            # convert the list of parameters back into a dictionary
            params_dict = dict(zip(self.x0.keys(), params))

            # update progress bar
            if self.pbar:
                self.tqdm.update()

            # return the value of the loss function
            return self.loss(**params_dict)

        # perform the optimization
        self.result: OptimizeResult = opt.minimize(
            fun=loss,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        # fetch optimized params
        self.params_inferred = dict(zip(list(self.x0.keys()), self.result.x))

        # loss of best run
        self.loss_inferred = self.result.x

        # coalescent distribution of best run
        self.dist_inferred = self.dist(**self.params_inferred)

        # return the result of the optimization
        return self.result
