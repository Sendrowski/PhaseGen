import itertools
import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List, Callable, Dict, Iterable, Tuple, Any, Iterator, Collection

import numpy as np
from matplotlib import pyplot as plt

from .visualization import Visualization

logger = logging.getLogger('phasegen')


class Epoch:
    """
    Epoch of a demographic scenario with constant population sizes and migration rates.
    """

    #: Start time of the epoch.
    start_time: float

    #: End time of the epoch.
    end_time: float

    #: Population sizes.
    pop_sizes: Dict[str, float]

    #: Migration rates.
    migration_rates: Dict[Tuple[str, str], float]

    def __init__(
            self,
            start_time: float = 0,
            end_time: float = np.inf,
            pop_sizes: Dict[str, float] = {'pop_0': 1},
            migration_rates: Dict[Tuple[str, str], float] = {}
    ):
        """
        Initialize the epoch.

        :param start_time: Start time of the epoch.
        :param end_time: End time of the epoch.
        :param pop_sizes: Population sizes.
        :param migration_rates: Migration rates.
        """
        #: Start time of the epoch.
        self.start_time: float = start_time

        #: End time of the epoch.
        self.end_time: float = end_time

        #: Population sizes.
        self.pop_sizes: Dict[str, float] = pop_sizes.copy()

        #: Population names.
        self.pop_names: List[str] = sorted(list(self.pop_sizes.keys()))

        #: Number of populations.
        self.n_pops: int = len(self.pop_names)

        migration_rates = migration_rates.copy()

        # fill non-existing migration rates with zero
        for p in self.pop_sizes:
            for q in self.pop_sizes:
                if p != q and (p, q) not in migration_rates:
                    migration_rates[(p, q)] = 0

        #: Migration rates.
        self.migration_rates: Dict[Tuple[str, str], float] = migration_rates


class DemographicEvent(ABC):
    """
    Base class for demographic events.
    """
    #: Start time of the event.
    start_time: float

    #: End time of the event.
    end_time: float

    #: Population names.
    pop_names: List[str]

    @abstractmethod
    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        """
        pass

    @staticmethod
    def _flatten(
            rates: Dict[Any, Dict[float, float]]
    ) -> (np.ndarray[float], Dict[float, Dict[Any, float]]):
        """
        Flatten rates into a list of times and a list of rates.

        :param rates: Dictionary mapping key to dictionary mapping times to rates.
        :return: Array of times and dictionary mapping key to dictionary mapping population to rate.
        """
        # get all unique times
        times_all = np.sort(np.unique(np.array([i for s in rates.values() for i in s], dtype=float)))

        # flattened list of migration rates
        new_rates: Dict[float, Dict[Any, float]] = defaultdict(lambda: {})

        # loop over all times
        for t in times_all:

            # for each key
            for key, r in rates.items():

                # if the time is in this population's times
                if t in r:
                    # add rate
                    new_rates[t][key] = r[t]

        return times_all, dict(new_rates)


class DiscreteDemographicEvent(DemographicEvent, ABC):
    """
    Base class for discrete demographic events.
    """
    #: Time at which the events occur in ascending order.
    times: np.ndarray[float]

    def _broadcast(self, epoch: Epoch):
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        # times which are within the time interval
        times: np.ndarray[float] = self.times[(
                (epoch.start_time < self.times) &
                (self.times <= epoch.end_time) &
                (self.times > 0)
        )]

        if len(times):
            epoch.end_time = times[0]


class DiscreteRateChanges(DiscreteDemographicEvent):
    """
    Demographic event for discrete changes in population sizes and migration rates.
    """

    def __init__(
            self,
            pop_sizes: Dict[str, Dict[float, float]] = {},
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] = {}
    ):
        """
        Initialize the population size change.

        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name, or a list of dictionaries of the form ``{time1: size1, time2: size2}`` ordered
            by population index, or a single dictionary of the form ``{time1: size1, time2: size2}`` for a single
            population.
        :param migration_rates: Migration rates. A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2:
            rate2}}`` of migration from population ``pop_i`` to population ``pop_j`` at time ``time1`` etc.
        """
        # make sure population sizes are positive
        for p, sizes in pop_sizes.items():
            if any(s <= 0 for s in sizes.values()):
                raise ValueError(f'Population sizes must be positive at all times.')

        # initialize zero migration rates if None is given
        if migration_rates is None:
            migration_rates = {(p, q): {0: 0} for p in pop_sizes for q in pop_sizes}
        elif not isinstance(migration_rates, dict):
            raise ValueError('Migration rates must be a dictionary.')

        #: Population names.
        self.pop_names: List[str] = sorted(list(set(pop_sizes.keys()).union(
            {p for k in migration_rates for p in k})))

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_names)

        migration_rates = migration_rates.copy()

        # fill non-existing and diagonal migration rates with zero
        for p in self.pop_names:
            for q in self.pop_names:
                if p == q or (p != q and (p, q) not in migration_rates):
                    migration_rates[(p, q)] = {0: 0}

        # flatten the population sizes and migration rates
        times: np.ndarray[float]
        rates: Dict[float, Dict[Any, float]]
        times, rates = self._flatten(pop_sizes | migration_rates)

        # check that all times are non-negative
        if np.any(np.array(times) < 0):
            raise ValueError('All times must not be negative.')

        # check that all migration rates are non-negative
        if np.any(np.array([rates[k][t] for k in rates for t in migration_rates if t in rates[k]]) < 0):
            raise ValueError('Migration rates must not be negative at all times.')

        # check that all population sizes are positive
        if np.any(np.array([rates[k][t] for k in rates for t in pop_sizes if t in rates[k]]) <= 0):
            raise ValueError('Population sizes must be positive at all times.')

        #: Times at which the population size changes occur.
        self.times: np.ndarray[float] = times

        #: Population sizes.
        self.pop_sizes: Dict[float, Dict[str, float]] = {
            t: {x: pops[x] for x in self.pop_names if x in pops if x in pops} for t, pops in rates.items()
        }

        #: Migration rates at each time.
        self.migration_rates: Dict[float, Dict[Tuple[str, str], float]] = {
            t: {(p, q): rates[t][(p, q)] for p in self.pop_names for q in self.pop_names if (p, q) in rates[t]}
            for t in rates
        }

        #: Start time of the event.
        self.start_time: float = self.times[0]

        #: End time of the event.
        self.end_time: float = self.times[-1]

    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        """
        for t in self.times[(epoch.start_time <= self.times) & (self.times < epoch.end_time)]:
            epoch.pop_sizes |= self.pop_sizes[t]
            epoch.migration_rates |= self.migration_rates[t]


class PopSizeChanges(DiscreteRateChanges):
    """
    Demographic event for changes in population size.
    """

    def __init__(self, pop_sizes: Dict[str, Dict[float, float]]):
        """
        Initialize the population size change.

        :param pop_sizes: Population sizes. A dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``.
        """
        super().__init__(pop_sizes=pop_sizes)


class PopSizeChange(PopSizeChanges):
    """
    Demographic event for a single change in population size.
    """

    def __init__(self, pop: str, time: float, size: float):
        """
        Initialize the population size change.

        :param pop: Population name.
        :param time: Time at which the population size changes.
        :param size: Population size.
        """
        super().__init__({pop: {time: size}})


class MigrationRateChanges(DiscreteRateChanges):
    """
    Demographic event for changes in migration rates.
    """

    def __init__(self, migration_rates: Dict[Tuple[str, str], Dict[float, float]]):
        """
        Initialize the migration rate change.

        :param migration_rates: Migration rates. A dictionary of the form
            ``{(pop_i, pop_j): {time1: rate1, time2: rate2}}`` of migration from population ``pop_i`` to population
            ``pop_j`` at time ``time1`` etc.
        """
        super().__init__(migration_rates=migration_rates)


class MigrationRateChange(MigrationRateChanges):
    """
    Demographic event for a single change in migration rate.
    """

    def __init__(self, pop1: str, pop2: str, time: float, rate: float):
        """
        Initialize the migration rate change.

        :param pop1: Source population name.
        :param pop2: Destination population name.
        :param time: Time at which the migration rate changes.
        :param rate: Migration rate.
        """
        super().__init__({(pop1, pop2): {time: rate}})


class DiscretizedDemographicEvent(DemographicEvent, ABC):
    """
    Base class for discretized demographic events.
    """
    pass


class DiscretizedRateChange(DiscretizedDemographicEvent):
    """
    Demographic event for discretized rate changes of a single population or migration rate.
    """

    def __init__(
            self,
            trajectory: Callable[[float], float],
            start_time: float,
            end_time: float = np.inf,
            pop: str | None = None,
            source_pop: str | None = None,
            dest_pop: str | None = None,
            step_size: float = 0.1
    ):
        """
        Initialize the population size change.

        :param trajectory: Trajectory function taking the time as argument and returning the rate.
        :param start_time: Start time of the event.
        :param end_time: End time of the event.
        :param pop: Population name or None if no population size changes.
        :param source_pop: Source population name or None if no migration rate changes.
        :param dest_pop: Destination population name or None if no migration rate changes.
        :param step_size: Step size used for the discretization.
        """
        if pop is None and (source_pop is None or dest_pop is None):
            raise ValueError('Either pop or source_pop and dest_pop must be specified.')

        #: Population name.
        self.pop: str | None = pop

        #: Population names.
        self.pop_names: List[str] = sorted(list(p for p in {pop, source_pop, dest_pop} if p is not None))

        #: Start time of the event.
        self.start_time: float = start_time

        #: End time of the event.
        self.end_time: float = end_time

        #: Trajectory function.
        self.trajectory: Callable[[float], float] = trajectory

        #: Step size used for the discretization.
        self.step_size: float = step_size

        #: Source population name.
        self.source_pop: str | None = source_pop

        #: Destination population name.
        self.dest_pop: str | None = dest_pop

    def _broadcast(self, epoch: Epoch):
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        # return if there is no overlap
        if epoch.end_time < self.start_time or epoch.start_time > self.end_time:
            return

        # if this event starts after the epoch, we take the start time
        if self.start_time > epoch.start_time:
            epoch.end_time = self.start_time
        else:
            n_steps = np.ceil((epoch.start_time - self.start_time + 1e-10) / self.step_size)
            epoch.end_time = self.start_time + n_steps * self.step_size

    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        """
        # if epoch is contained in the event
        if self.start_time <= epoch.start_time and epoch.end_time < self.end_time:

            rate_start = self.trajectory(epoch.start_time)
            rate_end = self.trajectory(epoch.end_time)
            rate = (rate_start + rate_end) / 2

            if self.pop is None:
                epoch.migration_rates[(self.source_pop, self.dest_pop)] = rate
            else:
                epoch.pop_sizes[self.pop] = rate


class DiscretizedRateChanges(DiscretizedDemographicEvent):
    """
    Demographic event for discretized rate changes of multiple populations or migration rates.
    """

    def __init__(
            self,
            trajectory: Dict[Any, Callable[[float], float]],
            start_time: Dict[Any, float] | float,
            end_time: Dict[Any, float] | float = np.inf,
            step_size: float = 0.1
    ):
        """
        Initialize the population size change.

        :param trajectory: Trajectory functions taking the time as argument and returning the rate.
        :param start_time: Start times of the events. A single value or a dictionary mapping keys to values.
        :param end_time: End times of the events.
        :param step_size: Step size used for the discretization.
        """
        #: Discretized rate change events.
        self.events = {}
        for k in trajectory:
            self.events[k] = DiscretizedRateChange(
                trajectory=trajectory[k],
                start_time=start_time[k] if isinstance(start_time, dict) else start_time,
                end_time=end_time[k] if isinstance(end_time, dict) else end_time,
                pop=k if isinstance(k, str) else None,
                source_pop=k[0] if isinstance(k, tuple) else None,
                dest_pop=k[1] if isinstance(k, tuple) else None,
                step_size=step_size
            )

        #: Population names.
        self.pop_names: List[str] = sorted(list(set([p for e in self.events.values() for p in e.pop_names])))

        #: Start time of the event.
        self.start_time: float = min([e.start_time for e in self.events.values()])

        #: End time of the event.
        self.end_time: float = max([e.end_time for e in self.events.values()])

    def _broadcast(self, epoch: Epoch):
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        for e in self.events.values():
            e._broadcast(epoch)

    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        :return: Epoch.
        """
        for e in self.events.values():
            e._apply(epoch)


class ExponentialRateChanges(DiscretizedRateChanges):
    """
    Demographic event for exponential rate changes of multiple populations or migration rates.
    """

    def __init__(
            self,
            initial_rate: Dict[Any, float],
            growth_rate: Dict[Any, float] | float,
            start_time: Dict[Any, float] | float,
            end_time: Dict[Any, float] | float = np.inf,
            step_size: float = 0.1
    ):
        """
        Initialize the exponential growth.

        :param initial_rate: Initial rates. A dictionary mapping keys to values. Keys are either population names or
            tuples of population names for population sizes and migration rates, respectively.
        :param growth_rate: Exponential growth rates. A single value or a dictionary mapping keys to values.
        :param start_time: Start times of the growth. A single value or a dictionary mapping keys to values.
        :param end_time: End times of the growth.
        :param step_size: Step size used for the discretization.
        """

        def get_trajectory(k: Any) -> Callable[[float], float]:
            """
            Get the trajectory function for the given key.

            :param k: Key.
            :return: Trajectory function.
            """
            g = growth_rate[k] if isinstance(growth_rate, dict) else growth_rate
            t0 = start_time[k] if isinstance(start_time, dict) else start_time
            x0 = initial_rate[k] if isinstance(initial_rate, dict) else initial_rate

            # return lambda and bind g, t0 and x0 into it
            # noinspection all
            return lambda t, g=g, t0=t0, x0=x0: x0 * np.exp(- g * (t - t0))

        super().__init__(
            trajectory={k: get_trajectory(k) for k in initial_rate},
            start_time=start_time,
            end_time=end_time,
            step_size=step_size
        )


class ExponentialPopSizeChanges(ExponentialRateChanges):
    """
    Demographic event for exponential population size changes of multiple populations.
    """

    def __init__(
            self,
            initial_size: Dict[str, float],
            growth_rate: Dict[str, float] | float,
            start_time: Dict[str, float] | float,
            end_time: Dict[str, float] | float = np.inf,
            step_size: float = 0.1
    ):
        """
        Initialize the exponential growth.

        :param initial_size: Initial population sizes. A dictionary mapping population names to sizes.
        :param growth_rate: Exponential growth rates. A single value or a dictionary mapping keys to values.
        :param start_time: Start times of the growth. A single value or a dictionary mapping keys to values.
        :param end_time: End times of the growth.
        :param step_size: Step size used for the discretization.
        """
        super().__init__(
            initial_rate=initial_size,
            growth_rate=growth_rate,
            start_time=start_time,
            end_time=end_time,
            step_size=step_size
        )


class Demography:
    """
    Class storing full demographic information.
    """
    #: Population names.
    pop_names: List[str]

    #: Number of populations.
    n_pops: int

    def __init__(
            self,
            events: List[DemographicEvent] = [],
            pop_sizes: Dict[str, Dict[float, float]] = {},
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] = {},
            max_size: float = 1e3
    ):
        """
        Initialize the demography.

        :param events: List of demographic events.
        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name, or a list of dictionaries of the form ``{time1: size1, time2: size2}`` ordered
            by population index, or a single dictionary of the form ``{time1: size1, time2: size2}`` for a single
            population.
        :param migration_rates: Migration rates. A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2:
            rate2}}`` of migration from population ``pop_i`` to population ``pop_j`` at time ``time1`` etc.
        :param max_size: Maximum size of the discretized epoch.
        """
        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Maximum size of the discretized epoch.
        self.max_size: float = max_size

        # add population size and migration rate changes if specified
        if len(pop_sizes) or len(migration_rates):
            events.append(DiscreteRateChanges(pop_sizes=pop_sizes, migration_rates=migration_rates))

        #: Array of demographic events.
        self.events: np.ndarray = np.array(events)

        #: Population names.
        self._prepare_events()

    def _prepare_events(self):
        """
        Sort events by start time and determine population names and number of populations.
        """
        # sort events by start time
        self.events = np.array(sorted(self.events, key=lambda e: e.start_time))

        # determine population names
        self.pop_names = sorted(list(set([p for e in self.events for p in e.pop_names])))

        # determine number of populations
        self.n_pops = len(self.pop_names)

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

        self._prepare_events()

        first_epoch = next(self.epochs)

        # create demography object
        d: ms.Demography = ms.Demography(
            populations=[ms.Population(name=pop, initial_size=first_epoch.pop_sizes[pop]) for pop in self.pop_names],
            migration_matrix=np.array([[first_epoch.migration_rates[(p, q)] for q in self.pop_names]
                                       for p in self.pop_names])
        )

        for epoch in itertools.islice(self.epochs, 1, max_epochs + 1):
            # iterate over populations
            for pop in self.pop_names:
                # add population size changes
                # noinspection PyTypeChecker
                d.add_population_parameters_change(
                    time=epoch.start_time,
                    initial_size=epoch.pop_sizes[pop],
                    population=pop
                )

            # iterate over migration rates
            for (p, q) in itertools.product(self.pop_names, repeat=2):

                if p != q:
                    # noinspection all
                    d.add_migration_rate_change(
                        time=epoch.start_time,
                        rate=epoch.migration_rates[(p, q)],
                        source=p,
                        dest=q
                    )

        # sort events by time
        d.sort_events()

        return d

    @property
    def epochs(self) -> Iterator[Epoch]:
        """
        Get a generator for the epochs.
        """
        self._prepare_events()

        prev = Epoch(
            start_time=0,
            end_time=0,
            pop_sizes={p: 1 for p in self.pop_names},
            migration_rates={k: 0 for k in itertools.product(self.pop_names, repeat=2)}
        )

        while True:

            # potential next epoch
            epoch = Epoch(
                start_time=prev.end_time,
                end_time=prev.end_time + self.max_size,
                pop_sizes=prev.pop_sizes,
                migration_rates=prev.migration_rates
            )

            # iterate over continuous events that occur before or at start time
            for e in self.events:
                # adjust end time
                e._broadcast(epoch)

            # apply the events to the epoch
            [e._apply(epoch) for e in self.events]

            yield epoch
            prev = epoch

    def get_epochs(self, t: float | List[float]) -> Epoch | np.ndarray:
        """
        Get the epoch at the given times.

        :param t: Time or times.
        :return: Epoch or array of epochs.
        """
        if not isinstance(t, Iterable):
            return self.get_epochs([t])[0]

        # sort times in ascending order
        t_sorted: Collection[float] = np.sort(t)

        # get epoch iterator
        iterator: Iterator[Epoch] = self.epochs

        # get first epoch
        epoch = next(iterator)

        # initialize array of epochs
        epochs = np.zeros_like(t_sorted, dtype=Epoch)

        for i, time in enumerate(t_sorted):
            # wind forward until we reach the epoch enclosing the current time
            while not epoch.start_time <= time < epoch.end_time:
                epoch = next(iterator)

            # add epoch to array
            epochs[i] = epoch

        # sort back to original order
        return np.array(epochs[np.argsort(t)])

    def plot_pop_sizes(
            self,
            t: np.ndarray = np.linspace(0, 10, 1000),
            show: bool = True,
            file: str = None,
            title: str = 'Population size trajectory',
            ylabel: str = '$N_e(t)$',
            ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Plot the population size over time.

        :param t: Times at which to plot the population sizes.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot.
        :param title: Title of the plot.
        :param ylabel: Label of the y-axis.
        :param ax: Axes object to plot to.
        :return: Axes object.
        """

        return Visualization.plot_rates(
            times=list(t),
            rates=dict(zip(
                self.pop_names,
                np.array([[e.pop_sizes[p] for p in self.pop_names] for e in self.get_epochs(t)]).T
            )),
            show=show,
            file=file,
            title=title,
            ylabel=ylabel,
            ax=ax
        )

    def plot_migration(
            self,
            t: np.ndarray = np.linspace(0, 10, 100),
            show: bool = True,
            file: str = None,
            title: str = 'Migration rate trajectory',
            ylabel: str = '$m_{ij}(t)$',
            ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        Plot the migration over time.

        :param t: Times at which to plot the migration rates.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot.
        :param ylabel: Label of the y-axis.
        :param ax: Axes object to plot to.
        :return: Axes object.
        """
        return Visualization.plot_rates(
            times=list(t),
            rates=dict(zip(
                [f"{k[0]}->{k[1]}" for k in itertools.product(self.pop_names, repeat=2)],
                np.array([[e.migration_rates[k] for k in itertools.product(self.pop_names, repeat=2)]
                          for e in self.get_epochs(t)]).T
            )),
            show=show,
            file=file,
            title=title,
            ylabel=ylabel,
            ax=ax
        )

    def plot(
            self,
            t: np.ndarray = np.linspace(0, 10, 100),
            show: bool = True,
            file: str = None
    ) -> List[plt.Axes]:
        """
        Plot the demographic scenario.

        :param t: Times at which to plot the population sizes and migration rates.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :return: Axes objects
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        self.plot_pop_sizes(t=t, show=False, file=file, ax=axes[0])
        self.plot_migration(t=t, show=show, file=file, ax=axes[1])

        return axes
