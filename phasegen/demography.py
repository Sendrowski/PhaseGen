import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from itertools import islice
from typing import List, Callable, Dict, Iterable, Tuple, Any, Iterator

import numpy as np
from matplotlib import pyplot as plt

from .visualization import Visualization

logger = logging.getLogger('phasegen')


class Demography(ABC):
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

    #: Migration rates.
    _migration_rates: Dict[Tuple[str, str], float]

    def __init__(self):
        """
        Initialize the demography.
        """
        #: The logger instance
        self.logger = logger.getChild(self.__class__.__name__)

    @staticmethod
    def _plot_rates(
            times_it: Iterator[float],
            rates_it: Dict[Any, Iterator[float]],
            show: bool = True,
            file: str = None,
            t_max: float = 10,
            max_epochs: int = 100,
            title: str = None,
            ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Plot the migration rates over time.

        :param times_it: Iterator over times.
        :param rates_it: Iterator over rates.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :param max_epochs: Maximum number of epochs to plot.
        :param title: Title of the plot.
        :param ax: Axes object to plot to.
        :return: Axes object.
        """
        # get times until t_max or max_epochs
        times = []
        for _ in range(max_epochs):
            try:
                t = next(times_it)
                if t > t_max:
                    break
                times.append(t)
            except StopIteration:
                break

        # get rates for times
        rates = {k: [next(v) for _ in times] for k, v in rates_it.items()}

        if len(times) != max_epochs:
            # add t_max as last entry
            times.append(t_max)
            for k, v in rates.items():
                v.append(v[-1])

        return Visualization.plot_rates(
            times=times,
            rates={str(k): np.array(v) for k, v in rates.items()},
            show=show,
            file=file,
            title=title,
            ax=ax
        )

    def plot_pop_sizes(
            self,
            show: bool = True,
            file: str = None,
            t_max: float = 10,
            max_epochs: int = 100,
            title: str = 'Population size trajectory',
            ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot.
        :param t_max: Maximum time to plot.
        :param max_epochs: Maximum number of epochs to plot.
        :param title: Title of the plot.
        :param ax: Axes object to plot to.
        :return: Axes object.
        """
        return self._plot_rates(
            times_it=self.times,
            rates_it=self.pop_sizes,
            show=show,
            file=file,
            t_max=t_max,
            max_epochs=max_epochs,
            title=title,
            ax=ax
        )

    def plot_migration(
            self,
            show: bool = True,
            file: str = None,
            t_max: float = 10,
            max_epochs: int = 100,
            title: str = 'Migration rate trajectory',
            ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Plot the migration over time.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :param max_epochs: Maximum number of epochs to plot.
        :param title: Title of the plot.
        :param ax: Axes object to plot to.
        :return: Axes object.
        """
        return self._plot_rates(
            times_it=self.times,
            rates_it={f"{k[0]}->{k[1]}": v for k, v in self.migration_rates.items()},
            show=show,
            file=file,
            t_max=t_max,
            max_epochs=max_epochs,
            title=title,
            ax=ax
        )

    def plot(self, show: bool = True, file: str = None, t_max: float = 10, max_epochs: int = 100) -> List[plt.Axes]:
        """
        Plot the demographic scenario.

        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param t_max: Maximum time to plot.
        :param max_epochs: Maximum number of epochs to plot.
        :return: Axes objects
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        self.plot_pop_sizes(show=False, t_max=t_max, max_epochs=max_epochs, ax=axes[0])
        self.plot_migration(show=show, file=file, t_max=t_max, max_epochs=max_epochs, ax=axes[1])

        return axes

    @staticmethod
    def exponential_growth(
            x0: float | np.ndarray | Dict[Any, float],
            growth_rate: float | np.ndarray | Dict[Any, float],
    ) -> Callable[[float], float | np.ndarray | Dict[Any, float]]:
        """
        Exponential growth trajectory.

        :param x0: Initial rate. A single value, a numpy array or a dictionary mapping keys to values.
        :param growth_rate: Exponential growth rate. A single value, a numpy array or a dictionary mapping keys to
            values.
        :return: Function that returns the rates at a given time.
        """
        if isinstance(x0, dict) and isinstance(growth_rate, dict):
            return lambda t: dict((k, x0[k] * np.exp(- growth_rate[k] * t)) for k in x0)

        return lambda t: x0 * np.exp(- growth_rate * t)

    @property
    def times(self) -> Iterator[float]:
        """
        Get a generator for the times at which the population sizes change for any population.

        :return: Times at which the population sizes change.
        """
        return (t for t in self._times)

    @property
    def pop_sizes(self) -> Dict[str, Iterator[float]]:
        """
        Get a generator for the population sizes per population and epoch.

        :return: Population sizes per population.
        """
        return dict((p, (n for n in self._pop_sizes[p])) for p in self.pop_names)

    @property
    def migration_rates(self) -> Dict[Tuple[str, str], Iterator[float]]:
        """
        Get array representation of the migration matrix.

        :return: Migration matrix.
        """
        return dict((p, (n for n in self._migration_rates[p])) for p in self._migration_rates)

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

    @abstractmethod
    def to_msprime(self):
        """
        Convert the demographic scenario to an msprime demographic model.

        :return: msprime demographic model.
        """
        pass


class PiecewiseTimeHomogeneousDemography(Demography):
    """
    Piecewise constant demographic scenario.
    """

    def __init__(
            self,
            pop_sizes: Dict[str, Dict[float, float]] | List[Dict[float, float]] | Dict[float, float] = [{0: 1}],
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] | Dict[float, np.ndarray] | None = None
    ):
        """
        Create a piecewise time-homogeneous demographic scenario.

        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name, or a list of dictionaries of the form ``{time1: size1, time2: size2}`` ordered
            by population index, or a single dictionary of the form ``{time1: size1, time2: size2}`` for a single
            population. Note that the first time must always be 0, and that population sizes must always be positive.
        :param migration_rates: Migration matrix. Use ``None`` for no migration.
            A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2: rate2}}`` where ``m_ij`` is the
            migration rate (backward in time) from population ``pop_i`` to population ``pop_j`` at time ``time1`` etc.
            Alternatively, a dictionary of 2-dimensional numpy arrays where the rows correspond to the source
            population and the columns to the destination. Note that migration rates for which the source and
            destination population are the same are ignored and that the first time must always be 0.
        """
        super().__init__()

        # raise error if pop_sizes is neither a list nor a dictionary
        if not isinstance(pop_sizes, (list, dict)):
            raise ValueError('Population sizes must be a list or a dictionary.')

        # if we have a list of population sizes, assume that the population names are pop_0, pop_1, ...
        if isinstance(pop_sizes, list):
            pop_sizes = {f'pop_{i}': v for i, v in enumerate(pop_sizes)}

        # if pop size dict is not a dictionary of dictionaries, wrap it in a dictionary
        if not isinstance(next(iter(pop_sizes.values())), dict):
            pop_sizes = {'pop_0': pop_sizes}

        # make sure population sizes are positive
        for p, sizes in pop_sizes.items():
            if any(s <= 0 for s in sizes.values()):
                raise ValueError(f'Population sizes must be positive at all times.')

        #: Population names.
        self.pop_names: List[str] = list(pop_sizes.keys())

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_names)

        # initialize zero migration rates if None is given
        if migration_rates is None:
            migration_rates = {(p, q): {0: 0} for p in self.pop_names for q in self.pop_names}
        elif not isinstance(migration_rates, dict):
            raise ValueError('Migration rates must be a dictionary.')

        # convert migration rates to dictionary of dictionaries
        if len(migration_rates) and isinstance(next(iter(migration_rates.values())), (np.ndarray, list)):

            # raise error if shape is different from (n_pops, n_pops)
            if np.array(next(iter(migration_rates.values()))).shape != (self.n_pops, self.n_pops):
                raise ValueError(f'Migration matrices must be of shape (n_pops, n_pops) = '
                                 f'({self.n_pops}, {self.n_pops}) to coincide with population '
                                 'size configuration')

            rates_new = defaultdict(lambda: {})

            for time, rate_matrix in migration_rates.items():
                for p in self.pop_names:
                    for q in self.pop_names:
                        if p != q:
                            rates_new[(p, q)][time] = rate_matrix[self.pop_names.index(p)][self.pop_names.index(q)]

            migration_rates = rates_new

        # fill non-existing and diagonal migration rates with zero
        for p in self.pop_names:
            for q in self.pop_names:
                if p == q or (p != q and (p, q) not in migration_rates):
                    migration_rates[(p, q)] = {0: 0}

        # flatten the population sizes and migration rates
        times: np.ndarray[float]
        rates: Dict[Any, np.ndarray[float]]
        times, rates = self._flatten(pop_sizes | migration_rates)

        # check that all times are non-negative
        if np.any(np.array(times) < 0):
            raise ValueError('All times must not be negative.')

        # check that all migration rates are non-negative
        if np.any(np.array([rates[p] for p in migration_rates]) < 0):
            raise ValueError('Migration rates must not be negative at all times.')

        # check that all population sizes are positive
        if np.any(np.array([rates[p] for p in self.pop_names]) <= 0):
            raise ValueError('Population sizes must be positive at all times.')

        #: Times at which the population sizes change.
        self._times: Iterable[float] = times

        #: Population sizes at each time.
        self._pop_sizes: Dict[str, Iterable[float]] = {p: rates[p] for p in self.pop_names}

        #: Migration rates at each time.
        self._migration_rates: Dict[Tuple[str, str], Iterable[float]] = {p: rates[p] for p in migration_rates}

        #: Number of epochs.
        self.n_epochs: int | None = len(self._times)

    @staticmethod
    def _flatten(
            rates: Dict[Any, Dict[float, float]]
    ) -> (np.ndarray[float], Dict[Any, np.ndarray[float]]):
        """
        Flatten rates into a list of times and a list of rates.

        :param rates: Dictionary mapping key to dictionary mapping times to rates.
        :return: List of times and dictionary mapping key to list of rates at each time.
        """
        # get all unique times
        times_all = list(np.sort(np.unique(np.array([i for s in rates.values() for i in s], dtype=float))))

        # flattened list of migration rates
        new_rates = defaultdict(lambda: np.zeros((len(times_all))))

        # loop over all times
        for i, t in enumerate(times_all):

            # for each key
            for key, r in rates.items():

                # if the time is in this population's times
                if t in r:
                    # add the rate at this index to the new rates
                    new_rates[key][i] = r[t]

                # if the time is not present, carry the last rate forward
                else:
                    new_rates[key][i] = new_rates[key][i - 1]

        return times_all, dict(new_rates)

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

    def to_msprime(
            self,
            max_epochs: int = 1000
    ) -> 'msprime.Demography':
        """
        Convert to an msprime demography object.

        :param max_epochs: Maximum number of epochs to use. This is necessary when the number of epochs is infinite.
        :return: msprime demography object.
        """
        import msprime as ms

        # create demography object
        d: ms.Demography = ms.Demography(
            populations=[ms.Population(name=pop, initial_size=next(self.pop_sizes[pop])) for pop in self.pop_names],
            migration_matrix=np.array([[next(self.migration_rates[(p, q)]) for q in self.pop_names]
                                       for p in self.pop_names])
        )

        # iterate over populations
        for pop in self.pop_names:
            # add population size changes
            for time, size in zip(islice(self.times, 1, max_epochs + 1),
                                  islice(self.pop_sizes[pop], 1, max_epochs + 1)):
                # noinspection all
                d.add_population_parameters_change(
                    time=time,
                    initial_size=size,
                    population=pop
                )

        # iterate over migration rates
        for (p, q), rates in self.migration_rates.items():

            if p != q:
                # add migration rate changes
                for time, rate in zip(islice(self.times, 1, max_epochs + 1), islice(rates, 1, max_epochs + 1)):
                    # noinspection all
                    d.add_migration_rate_change(
                        time=time,
                        rate=rate,
                        source=p,
                        dest=q
                    )

        # sort events by time
        d.sort_events()

        return d


class TimeHomogeneousDemography(PiecewiseTimeHomogeneousDemography):
    """
    Demographic scenario that is constant over time.
    """

    def __init__(
            self,
            pop_sizes: float | Dict[str, float] | List[float] = 1,
            migration_rates: Dict[Tuple[str, str], float] | np.ndarray | None = None
    ):
        """
        Create a time-homogeneous demographic scenario.

        :param pop_sizes: A single population size if there is only one population, or a list of population sizes
            for various populations, or a dictionary mapping population names to population sizes like
            ``{'pop_0': 1, 'pop_1': 2}``.
        :param migration_rates: Migration rates. Use ``None`` for no migration.
            Either a dictionary of the form ``{(pop_i, pop_j): m_ij}`` where ``m_ij`` is the migration rate from
            population ``pop_i`` to population ``pop_j`` backward in time or a 2-dimensional numpy array where the
            rows correspond to the source population and the columns to the destination. Note that migration rates for
            which the source and destination population are the same are ignored.
        """
        # wrap pop_sizes in dictionaries
        if isinstance(pop_sizes, (list, np.ndarray)):
            pop_sizes = [{0: p} for p in pop_sizes]

        elif isinstance(pop_sizes, dict):
            pop_sizes = {p: {0: s} for p, s in pop_sizes.items()}

        else:
            pop_sizes = {0: pop_sizes}

        # wrap migration_rates in dictionary if it is a dictionary
        if isinstance(migration_rates, dict):
            migration_rates = {(p, q): {0: m} for (p, q), m in migration_rates.items()}

        # wrap migration_rates in dictionary if it is a numpy array or list
        if isinstance(migration_rates, (np.ndarray, list)):
            migration_rates = {0: np.array(migration_rates)}

        super().__init__(
            pop_sizes=pop_sizes,
            migration_rates=migration_rates
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

    def __eq__(self, other: Any) -> bool:
        """
        Check if two demographic scenarios are equal.

        TODO no longer used

        :param other: Any other object.
        :return: Whether the two demographic scenarios are equal.
        """
        raise NotImplementedError

        if not isinstance(other, TimeHomogeneousDemography):
            return False

        if self.n_pops != other.n_pops:
            return False

        if self.pop_names != other.pop_names:
            return False

        if not all(self.pop_size[p] == other.pop_size[p] for p in self.pop_names):
            return False

        if not all(self._migration_rates[pair] == other._migration_rates[pair] for pair in self._migration_rates):
            return False

        return True


class DiscretizedDemography(PiecewiseTimeHomogeneousDemography):
    """
    Discretized demographic scenario.
    """

    def __init__(
            self,
            pop_sizes: Callable[[float], Dict[str, float]],
            migration_rates: Callable[[float], Dict[Tuple[str, str], float]] = None,
            min_size: float = 1e-3,
            start_size: float = 1,
            max_growth: float = 1
    ):
        """
        Create a continuous demographic scenario. By default, the discretization intervals are determined dynamically
        based on the increase/decrease in the population size / migration rates. Set ``min_size`` equal to
        ``start_size`` to have a fixed discretization interval.

        :param pop_sizes: Function that returns the population size at a given time. A dictionary of population sizes
            at time t indexed by population name.
        :param migration_rates: Function that returns a dictionary of the form ``{(pop_i, pop_j): m_ij}`` given time t,
            where ``m_ij`` is the migration rate from population ``pop_i`` to population ``pop_j``.
        :param min_size: Minimum discretization interval size.
        :param start_size: Interval size at the start of new epochs.
        :param max_growth: Maximum absolute growth rate compared to the previous population size.
        """
        if start_size < min_size:
            raise ValueError('Start size must be larger than minimum size.')

        if min_size <= 0:
            raise ValueError('Minimum size must be positive.')

        # initialize logger
        Demography.__init__(self)

        #: Function that returns the population size at a given time.
        self._pop_sizes: Callable[[float], Dict[str, float]] = pop_sizes

        #: Function that returns the migration rate at a given time.
        self._migration_rates: Callable[[float], Dict[Tuple[str, str], float]] = migration_rates

        #: Minimum discretization interval
        self.min_size: float = min_size

        #: Step size at the start of new epochs
        self.start_size: float = start_size

        #: Maximum absolute growth rate compared to the previous population size
        self.max_growth: float = max_growth

        #: Population names
        self.pop_names: List[str] = list(self._pop_sizes(0).keys())

        #: Number of populations / demes
        self.n_pops: int = len(self.pop_names)

        #: Number of epochs
        self.n_epochs: int | None = None

    @property
    def times(self) -> Iterator[float]:
        """
        Get the times at which the population sizes change.

        :return: Times.
        """
        return map(lambda x: x[0], self._generate_epochs())

    @property
    def pop_sizes(self) -> Dict[str, Iterator[float]]:
        """
        Get the population sizes.

        :return: Population sizes.
        """
        pop_sizes: Dict[str, Iterable[float]] = {}

        # obtain epoch population sizes generator
        for pop in self.pop_names:
            pop_sizes[pop] = map(lambda x, p=pop: x[1][p], self._generate_epochs())

        return pop_sizes

    @property
    def migration_rates(self) -> Dict[Tuple[str, str], Iterator[float]]:
        """
        Get the migration rates.

        :return: Migration rates.
        """
        migration_rates = {}

        # obtain epoch migration rates generator
        for p in self.pop_names:
            for q in self.pop_names:
                migration_rates[(p, q)] = map(lambda x, y=(p, q): x[1][y] if y in x[1] else 0, self._generate_epochs())

        return migration_rates

    def _generate_epochs(self) -> Iterator[Tuple[float, Dict[Any, float]]]:
        """
        Generate epochs.

        :return: Iterator over epochs, return tuple of time and rates at time.
        """
        # initialize step size
        # decrease by factor 1/2 until we have a step size smaller than min_size or the growth rate is smaller than
        # max_growth
        step_size = self.start_size

        def rates(t: float) -> Dict[str, float]:
            """
            Population size and growth rates at time ``t``.

            :param t: Time.
            :return: Population size and growth rates.
            """
            return self._pop_sizes(t) | self._migration_rates(t)

        # initialize the previous population size
        x0 = rates(0)
        rates_prev = np.array(list(x0.values()))
        time = 0

        # yield the initial population size and time
        yield 0, rates(0)

        while True:
            # get the new population size at the current time
            rates_curr = np.array(list(rates(time + step_size).values()))

            # check if the population size is infinite
            if np.isinf(rates_curr).any():
                self.logger.warning(f'Rates are {rates_curr} at time {time + step_size}. Stopping.')
                break

            # check if the population size is NaN
            if np.isnan(rates_curr).any():
                raise ValueError(f'Rates are is {rates_curr} at time {time + step_size}. Stopping.')

            # calculate the growth rate
            growth_rate = np.abs(rates_curr - rates_prev).max()

            # check the constraints
            if step_size <= self.min_size or growth_rate <= self.max_growth:
                # update the previous population size and time
                rates_prev = rates_curr
                time += step_size

                # get the population size at the endpoints
                a = rates(time)
                b = rates(time + step_size)

                # yield the current population size and time
                midpoint = dict((p, ((a[p] + b[p]) / 2)) for p in x0)

                # reset the step size
                step_size = self.start_size

                # yield the current population size and time
                yield time, midpoint

            else:
                # decrease the step size
                step_size /= 2
