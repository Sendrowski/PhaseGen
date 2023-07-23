import logging
from abc import abstractmethod
from collections import defaultdict
from itertools import islice
from typing import List, Callable, Dict, Iterable, Tuple, Generator, cast, Sized

import numpy as np
from matplotlib import pyplot as plt

from .visualization import Visualization

logger = logging.getLogger('phasegen')


class Demography:
    """
    Base class for demographic scenarios.
    """

    #: Population sizes.
    _pop_sizes: Dict[str, Iterable[float]]

    #: Times at which the population sizes change.
    _times: Iterable[float]

    #: Number of epochs.
    n_epochs: int | None

    #: Number of populations / demes.
    n_pops: int

    #: Population names.
    pop_names: List[str]

    def __init__(self):
        """
        Initialize the demography.
        """
        #: The logger instance
        self.logger = logger.getChild(self.__class__.__name__)

    def plot(
            self,
            show: bool = True,
            file: str = None,
            t_max: float = None,
    ) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :return: Axes object.
        """
        pass

    @property
    def times(self) -> Generator[float, None, None]:
        """
        Get a generator for the times at which the population sizes change for any population.

        :return: Times at which the population sizes change.
        """
        return (t for t in self._times)

    @property
    def pop_sizes(self) -> Dict[str, Generator[float, None, None]]:
        """
        Get a generator for the population sizes per population and epoch.

        :return: Population sizes per population.
        """
        return dict((p, (n for n in self._pop_sizes[p])) for p in self.pop_names)

    @abstractmethod
    def get_rate(self, t: float) -> Dict[str, float]:
        """
        Get the coalescence rate.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        pass

    @abstractmethod
    def get_cum_rate(self, t: float) -> Dict[str, float]:
        """
        Get the cumulative coalescence rate.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        pass


class TimeHomogeneousDemography(Demography):
    """
    Demographic scenario where the population size is constant.
    """
    #: Number of epochs.
    n_epochs: int = 1

    def __init__(
            self,
            pop_size: float | Dict[str, float] | List[float] = 1
    ):
        """
        Create a constant demographic scenario.

        :param pop_size: A single population size, or a list of population sizes for various population, or
            a dictionary mapping population names to population sizes.
        """
        super().__init__()

        if isinstance(pop_size, dict):
            self.pop_size = dict(pop_size)

        elif isinstance(pop_size, list):
            self.pop_size = {i: v for i, v in enumerate(pop_size)}
        else:
            # assume a single population
            self.pop_size = {'pop_0': float(pop_size)}

        # check that all population sizes are positive
        if (np.array(list(self.pop_size.values())) <= 0).any():
            raise ValueError('All population size must be greater than zero.')

        #: Population sizes.
        self._pop_sizes: Dict[str, np.ndarray] = {i: np.array([v]) for i, v in self.pop_size.items()}

        #: Times at which the population sizes change.
        self._times: np.ndarray = np.array([0])

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_size)

        #: Population names.
        self.pop_names: List[str] = list(self.pop_size.keys())

    def plot(
            self,
            show: bool = True,
            file: str = None,
            t_max: float = None,
    ) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :return: Axes object.
        """
        return Visualization.plot_pop_sizes(
            times={p: np.array([0, 1]) for p in self.pop_names},
            pop_sizes={p: np.array([self.pop_size[p], self.pop_size[p]]) for p in self.pop_names},
            t_max=t_max,
            show=show,
            file=file
        )

    def get_rate(self, t: float) -> Dict[str, float]:
        """
        Get the coalescence rate per population.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        return {p: 1 / self.pop_size[p] for p in self.pop_names}

    def get_cum_rate(self, t: float) -> Dict[str, float]:
        """
        Get the cumulative coalescence rate per population.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        return {p: t / self.pop_size[p] for p in self.pop_names}


class PiecewiseTimeHomogeneousDemography(Demography):
    """
    Piecewise constant demographic scenario.
    """

    def __init__(
            self,
            pop_sizes: List[float] | Dict[str, Iterable[float]],
            times: List[float] | Dict[str, Iterable[float]]
    ):
        """
        The population sizes and times these changes occur backwards in time.

        :param pop_sizes: List of population sizes if there is only one population, or a dictionary mapping
            population names to lists of population sizes.
        :param times: List of times at which the population sizes change if there is only one population, or a
            dictionary mapping population names to lists of times at which the population sizes change. Time starts
            at zero and increases backwards in time.
        """
        super().__init__()

        # convert to dictionary
        if not isinstance(pop_sizes, dict):
            # assume a single population
            pop_sizes = {'pop_0': pop_sizes}

        # convert to dictionary
        if not isinstance(times, dict):
            # assume a single population
            times = {'pop_0': times}

        # check that the population names match
        if pop_sizes.keys() != times.keys():
            raise ValueError('Population names must match between pop_sizes and times.')

        # whether all population sizes and times are lists
        is_list = True

        #: Population names.
        self.pop_names: List[str] = list(pop_sizes.keys())

        for p in self.pop_names:

            if isinstance(pop_sizes[p], (list, np.ndarray)) and isinstance(times[p], (list, np.ndarray)):
                # check that the number of population sizes and times match
                if len(cast(Sized, pop_sizes[p])) != len(cast(Sized, times[p])):
                    raise ValueError(f'Number of population sizes and times do not match for population {p}.')

                # check that all population sizes are positive
                if (np.array(pop_sizes[p]) <= 0).any():
                    raise ValueError('All population sizes must be greater than zero.')

                # check that all times are positive
                if (np.array(times[p]) < 0).any():
                    raise ValueError('All times must be greater than or equal to zero.')
            else:
                is_list = False

        # merge population sizes and times so that we have single list of times and the
        # corresponding population sizes at each time
        # If we were given generators, we assume this is already the case
        if is_list:

            # replace the population sizes and times
            times, pop_sizes = self.flatten(
                times=times,
                pop_sizes=pop_sizes
            )

            # number of epochs
            n_epochs = len(times)
        else:

            # replace population times by the times of the first population
            times = times[self.pop_names[0]]

            # we don't know the number of epochs
            n_epochs = None

        #: Times at which the population sizes change.
        self._times: Iterable[float] = times

        #: Population sizes at each time.
        self._pop_sizes: Dict[str, Iterable[float]] = pop_sizes

        #: Number of epochs.
        self.n_epochs: int | None = n_epochs

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_names)

    @staticmethod
    def flatten(
            times: Dict[str, List[float]],
            pop_sizes: Dict[str, List[float]]

    ) -> (List[float], Dict[str, List[float]]):
        """
        Flatten population sizes and times into a list of times and a list of population sizes.

        :param pop_sizes: Dictionary mapping population names to lists of population sizes.
        :param times: Dictionary mapping population names to lists of times.
        :return: List of times and list of population sizes.
        """
        # get all unique times
        times_all = np.sort(np.unique(np.array([i for s in times.values() for i in s], dtype=float)))

        # flattened list of population names
        new_pop_sizes = defaultdict(list)

        # loop over all times
        for t in times_all:

            # for each population
            for pop, time in times.items():

                # if the time is in this population's times
                if t in time:
                    # Get the index of this time
                    index = time.index(t)
                    # Add the population size at this index to the new sizes
                    new_pop_sizes[pop].append(pop_sizes[pop][index])

                # if the time is not in this population's times, carry the last size forward
                elif new_pop_sizes[pop]:
                    new_pop_sizes[pop].append(new_pop_sizes[pop][-1])

        return times_all, dict(new_pop_sizes)

    def plot(
            self,
            show: bool = True,
            file: str = None,
            t_max: float = None,
            max_epochs: int = 100,
    ) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :param max_epochs: Maximum number of epochs to plot.
        :return: Axes object.
        """
        # get times and population sizes for a maximum of max_epochs epochs
        times = dict((p, list(islice(self.times, max_epochs))) for p in self.pop_names)
        pop_sizes = dict((p, list(islice(self.pop_sizes[p], max_epochs))) for p in self.pop_names)

        return Visualization.plot_pop_sizes(
            times=times,
            pop_sizes=pop_sizes,
            t_max=t_max,
            show=show,
            file=file
        )

    def get_rate(self, t: float) -> Dict[str, float]:
        """
        Get the coalescence rate per population.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        # obtain index of previous epoch
        i = np.sum(np.array(self._times) <= t) - 1

        # return rate
        return {p: 1 / next(islice(self.pop_sizes[p], i, None)) for p in self.pop_names}

    def get_cum_rate(self, t: float) -> Dict[str, float]:
        """
        Get the cumulative coalescence rate per population.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        # obtain index of previous epoch
        i = np.sum(np.array(self._times) <= t) - 1

        if i <= 0:
            return {p: t / next(self.pop_sizes[p]) for p in self.pop_names}

        rate = dict()
        for p in self.pop_names:
            # get population sizes up to i + 1
            pop_sizes = np.array(list(islice(self.pop_sizes[p], i + 1)))
            time = next(islice(self.times, i))

            rate[p] = np.dot(self._times[:i], 1 / pop_sizes[:i]) + (t - time) / pop_sizes[i]

        return rate


class ContinuousDemography(PiecewiseTimeHomogeneousDemography):
    """
    Continuous demographic scenario (which is discretized dynamically based on the
    increase/decrease in population size).
    """

    def __init__(
            self,
            trajectory: Callable[[float], Dict[str, float] | float],
            min_size: float = 1e-3,
            start_size: float = 1,
            max_growth: float = 1
    ):
        """
        Create a continuous demographic scenario.

        :param trajectory: Function that returns the population size at a given time. Either a single value or a
            dictionary with population names as keys.
        :param min_size: Minimum discretization interval
        :param start_size: Population size at the start of new epochs.
        :param max_growth: Maximum absolute growth rate compared to the previous population size.
        """
        # initialize logger
        Demography.__init__(self)

        #: Population size trajectory function
        self.trajectory = trajectory

        #: Minimum discretization interval
        self.min_size = min_size

        #: Step size at the start of new epochs
        self.start_size = start_size

        #: Maximum absolute growth rate compared to the previous population size
        self.max_growth = max_growth

        #: Population names
        self.pop_names = list(self.trajectory(0).keys())

        super().__init__(
            pop_sizes=self.pop_sizes,
            times=dict((p, self.times) for p in self.pop_names)
        )

    @property
    def times(self) -> Iterable[float]:
        """
        Get the times at which the population sizes change.

        :return: Times.
        """
        return map(lambda x: x[1], self.generate_epochs())

    @property
    def pop_sizes(self) -> Dict[str, Iterable[float]]:
        """
        Get the population sizes.

        :return: Population sizes.
        """
        pop_sizes: Dict[str, Iterable[float]] = {}

        # obtain epoch population sizes generator
        for pop in self.pop_names:
            pop_sizes[pop] = map(lambda x, p=pop: x[0][p], self.generate_epochs())

        return pop_sizes

    def generate_epochs(self) -> Iterable[Tuple[Dict[str, float], float]]:
        # initialize step size
        # decrease by factor 1/2 until we have a step size smaller than min_size or the growth rate is smaller than
        # max_growth
        step_size = self.start_size

        # initialize the previous population size
        prev_pop_prev = np.array(list(self.trajectory(0).values()))
        time = 0

        # yield the initial population size and time
        yield self.trajectory(0), 0

        while True:
            # get the new population size at the current time
            pop_size_curr = np.array(list(self.trajectory(time + step_size).values()))

            # check if the population size is infinite
            if np.isinf(pop_size_curr).any():
                self.logger.warning(f'Population size is {pop_size_curr} at time {time + step_size}. Stopping.')
                break

            # check if the population size is NaN
            if np.isnan(pop_size_curr).any():
                raise ValueError(f'Population size is {pop_size_curr} at time {time + step_size}. Stopping.')

            # calculate the growth rate
            growth_rate = np.abs(pop_size_curr - prev_pop_prev).max()

            # check the constraints
            if step_size < self.min_size or growth_rate < self.max_growth:
                # update the previous population size and time
                prev_pop_prev = pop_size_curr
                time += step_size

                # reset the step size
                step_size = self.start_size

                # get the population size at the endpoints
                a = self.trajectory(time)
                b = self.trajectory(time + step_size)

                # yield the current population size and time
                midpoint = dict((p, ((a[p] + b[p]) / 2)) for p in self.pop_names)

                # yield the current population size and time
                yield midpoint, time

            # decrease the step size
            step_size /= 2


class ExponentialDemography(ContinuousDemography):
    """
    Demographic scenario where the population size grows exponentially.
    """

    def __init__(
            self,
            growth_rate: float | Dict[str, float],
            N0: float | Dict[str, float] = 1,
            min_size: float = 1e-3,
            start_size: float = 1,
            max_growth: float = 1
    ):
        """
        :param growth_rate: Exponential growth rate so that at time ``t`` in the past we have
            ``N0 * exp(- growth_rate * t)``. Either a single value or a dictionary with population names as keys.
        :param N0: Initial population size (only used if growth_rate is specified). Either a single value or a
            dictionary with population names as keys.
        :param min_size: Minimum discretization interval
        :param start_size: Population size at the start of new epochs.
        :param max_growth: Maximum absolute growth rate compared to the previous population size.
        """
        # wrap in dictionary
        if isinstance(growth_rate, (float, int)):
            self.growth_rate: Dict[str, float] = {'pop_0': growth_rate}
        else:
            self.growth_rate: Dict[str, float] = growth_rate

        # wrap in dictionary
        if isinstance(N0, (float, int)):
            self.N0: Dict[str, float] = {'pop_0': N0}
        else:
            self.N0: Dict[str, float] = N0

        # check that the population names match
        if growth_rate.keys() != self.N0.keys():
            raise ValueError('Population names must match between growth_rate and N0.')

        # check that all Ne are positive
        if (np.array(list(self.N0.values())) <= 0).any():
            raise ValueError('All initial population sizes must be greater than zero.')

        def trajectory(t: float) -> Dict[str, float]:
            """
            Exponential trajectory.

            :param t: Time.
            :return: Population sizes at time ``t``.
            """
            N0 = np.array(list(self.N0.values()))
            growth_rate = np.array(list(self.growth_rate.values()))

            return dict(zip((self.N0.keys()), N0 * np.exp(- growth_rate * t)))

        super().__init__(
            trajectory=trajectory,
            min_size=min_size,
            start_size=start_size,
            max_growth=max_growth
        )
