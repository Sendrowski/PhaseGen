"""
Demographic events and demography class.
"""

import itertools
import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from functools import cached_property
from typing import List, Callable, Dict, Iterable, Tuple, Any, Iterator, Sequence

import numpy as np

logger = logging.getLogger('phasegen')


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
            events: List['DemographicEvent'] = None,
            pop_sizes: Dict[str, Dict[float, float]] | Dict[str, float] | float = None,
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] | Dict[Tuple[str, str], float] = None,
            warn_n_epochs: int = 20
    ):
        """
        Initialize the demography.

        :param events: List of demographic events.
        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name and time at which the population size changes, or a dictionary of the form
            ``{pop_i: size}`` if the population size is constant, or a single float if there is only one population
            and the population size is constant.
        :param migration_rates: Migration rates. A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2:
            rate2}}`` of migration from population ``pop_i`` to population ``pop_j`` at time ``time1`` etc. or
            alternatively a dictionary of the form ``{(pop_i, pop_j): rate}`` if the migration rate is constant over
            time.
        :param warn_n_epochs: Threshold for the number of epochs considered after which a warning is issued.
        """
        if events is None:
            events = []

        if pop_sizes is None:
            pop_sizes = {}

        # wrap population size in dictionary if it is a single float
        elif isinstance(pop_sizes, (float, int)):
            pop_sizes = {'pop_0': {0: pop_sizes}}

        # wrap population size in dictionary if only one time per population is given
        elif isinstance(pop_sizes, dict) and isinstance(list(pop_sizes.values())[0], (float, int)):
            pop_sizes = {p: {0: s} for p, s in pop_sizes.items()}

        if migration_rates is None:
            migration_rates = {}

        # wrap migration rate in dictionary if only one time per migration pair is given
        elif isinstance(migration_rates, dict) and isinstance(list(migration_rates.values())[0], (float, int)):
            migration_rates = {(p, q): {0: r} for (p, q), r in migration_rates.items()}

        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Threshold for the number of epochs considered after which a warning is issued.
        self.warn_n_epochs: int = int(warn_n_epochs)

        #: Whether a warning about the number of epochs has been already issued.
        self._issued_warning = False

        #: Array of demographic events.
        self.events: List[DemographicEvent] = list(events)

        # add population size and migration rate changes if specified
        if len(pop_sizes) or len(migration_rates):
            self.events += [DiscreteRateChanges(pop_sizes=pop_sizes, migration_rates=migration_rates)]

        # prepare events
        self._prepare_events()

        # issue warning if multiple populations are specified but no migration rates are given
        if self.n_pops > 1 and migration_rates == {} and len(events) == 0:
            self._logger.warning(
                'Multiple populations are specified, but no migration rates were given so far. '
                'Initializing with zero migration rates between all populations. '
                'Note that this may lead to infinite coalescence times if not changed later.'
            )

    def _prepare_events(self):
        """
        Sort events by start time and determine population names and number of populations.
        """
        # sort events by start time
        self.events = sorted(self.events, key=lambda e: e.start_time)

        # determine population names
        self.pop_names = sorted(list(set([p for e in self.events for p in e.pop_names])))

        # determine number of populations
        self.n_pops = len(self.pop_names)

    def to_msprime(
            self,
            max_epochs: int = 1000
    ) -> 'msprime.Demography':
        """
        Convert to an Msprime demography object.

        :param max_epochs: Maximum number of epochs to use. Note that the number of epochs may be infinite.
        :return: msprime demography object.
        :raise ImportError: If Msprime is not installed.
        """
        try:
            import msprime as ms
        except ImportError:
            raise ImportError('Msprime must be installed to use this method.')

        self._prepare_events()

        first_epoch = next(self.epochs)

        # create demography object
        d: ms.Demography = ms.Demography(
            populations=[ms.Population(name=pop, initial_size=first_epoch.pop_sizes[pop]) for pop in self.pop_names],
            migration_matrix=np.array([[first_epoch.migration_rates[(p, q)] for q in self.pop_names]
                                       for p in self.pop_names])
        )

        for epoch in itertools.islice(self.epochs, 1, int(max_epochs) + 1):
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

    def _to_demes(self) -> 'demes.Graph':
        """
        Convert to demes object (see https://tskit.dev/msprime/docs/stable/api.html#msprime.Demography.to_demes).
        TODO: msprime raises an error when converting to demes (migration[0]: invalid migration)

        :return: Demes object.
        :raise ImportError: If msprime is not installed.
        """
        self.to_msprime().to_demes()

    @property
    def epochs(self) -> Iterator['Epoch']:
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

        i = 0
        while True:

            # issue warning if number of epochs exceeds threshold
            if i == self.warn_n_epochs and not self._issued_warning:
                self._logger.warning(
                    f'Number of epochs considered exceeds {self.warn_n_epochs}. '
                    'Note that the runtime increases linearly with the number of epochs.'
                )
                self._issued_warning = True

            # potential next epoch
            epoch = Epoch(
                start_time=prev.end_time,
                end_time=np.inf,
                pop_sizes=prev.pop_sizes,
                migration_rates=prev.migration_rates
            )

            # broadcast events
            for e in self.events:
                # adjust end time
                e._broadcast(epoch)

            # apply the events to the epoch
            [e._apply(epoch) for e in self.events]

            yield epoch
            prev = epoch

            if epoch.end_time == np.inf:
                break

            i += 1

    def has_n_epochs(self, n: int) -> bool:
        """
        Check whether the demography has at least `n` epochs.

        :param n: Number of epochs.
        :return: Whether the demography has at least `n` epochs.
        """
        # get epoch iterator
        epochs = self.epochs

        for _ in range(int(n)):
            try:
                next(epochs)
            except StopIteration:
                return False

        return True

    def get_epochs(self, t: Iterable[float]) -> Sequence['Epoch']:
        """
        Get the epochs at the given times.

        :param t: Times.
        :return: Array of epochs.
        """
        t = list(t)

        # sort times in ascending order
        t_sorted: Sequence[float] = np.sort(t)

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

    def get_epoch(self, t: float = 0) -> 'Epoch':
        """
        Get the epoch at the given time.

        :param t: Time.
        :return: Epoch.
        """
        return self.get_epochs([t])[0]

    def add_events(self, events: List['DemographicEvent']):
        """
        Add demographic events.

        :param events: List of demographic events.
        """
        self.events += events

        self._prepare_events()

    def add_event(self, event: 'DemographicEvent'):
        """
        Add a demographic event.

        :param event: Demographic event.
        """
        self.add_events([event])

    def plot_pop_sizes(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            title: str = 'Population size trajectory',
            ylabel: str = '$N_e$',
            ax: 'plt.Axes' = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the population size over time.

        :param t: Times at which to plot the population sizes. By default, we use 1000 time points between
            time 0 and 10.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot.
        :param title: Title of the plot.
        :param ylabel: Label of the y-axis.
        :param ax: Axes object to plot to.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        from .visualization import Visualization

        if t is None:
            t = np.linspace(0, 10, 1000)

        if kwargs is None:
            kwargs = {}

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
            kwargs=kwargs,
            ax=ax
        )

    def plot_migration(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            title: str = 'Migration rate trajectory',
            ylabel: str = '$m_{ij}$',
            ax: 'plt.Axes' = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the migration over time.

        :param t: Times at which to plot the migration rates. By default, we use 1000 time points between time 0 and 10.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot.
        :param ylabel: Label of the y-axis.
        :param ax: Axes object to plot to.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        from .visualization import Visualization

        if t is None:
            t = np.linspace(0, 10, 1000)

        if kwargs is None:
            kwargs = {}

        # get all pairs of populations
        pops = [(p, q) for p in self.pop_names for q in self.pop_names if p != q]

        return Visualization.plot_rates(
            times=list(t),
            rates=dict(zip(
                [f"{p[0]}->{p[1]}" for p in pops],
                np.array([[e.migration_rates[p] for p in pops]
                          for e in self.get_epochs(t)]).T
            )),
            show=show,
            file=file,
            title=title,
            ylabel=ylabel,
            kwargs=kwargs,
            ax=ax
        )

    def plot(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            ylabel: str = '$N_e, m_{ij}$',
            ax: 'plt.Axes' = None,
            title: str = 'Demography',
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the demographic scenario.

        :param t: Times at which to plot the population sizes and migration rates. By default, we use 1000 time points
            between time 0 and 10.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param ylabel: Label of the y-axis.
        :param ax: Axes object to plot to.
        :param title: Title of the plot.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        from matplotlib import pyplot as plt

        if t is None:
            t = np.linspace(0, 10, 1000)

        if kwargs is None:
            kwargs = {}

        if ax is None:
            _, ax = plt.subplots()

        self.plot_pop_sizes(t=t, show=False, ax=ax, title=title, ylabel=ylabel, kwargs=kwargs)
        self.plot_migration(t=t, show=show, file=file, ax=ax, title=title, ylabel=ylabel, kwargs=kwargs)

        return ax


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
            pop_sizes: Dict[str, float] = None,
            migration_rates: Dict[Tuple[str, str], float] = None
    ):
        """
        Initialize the epoch.

        :param start_time: Start time of the epoch.
        :param end_time: End time of the epoch.
        :param pop_sizes: Population sizes. By default, we have ``{'pop_0': 1}`.
        :param migration_rates: Migration rates. By default, we have zero migration rates between all populations.
        """
        if pop_sizes is None:
            pop_sizes = {'pop_0': 1}

        if migration_rates is None:
            migration_rates = {}

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

    @cached_property
    def tau(self) -> float:
        """
        Time interval of the epoch.
        """
        return self.end_time - self.start_time

    def __eq__(self, other):
        """
        Compare epochs using their hash.

        :param other: The other epoch.
        :return: Whether the epochs are equal.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """
        Hash the epoch. Note that we do not include the start and end time, since they are not relevant for the
        state space created from the epoch.

        :return: Hash of the epoch.
        """
        return hash((
            tuple(self.pop_sizes.items()),
            tuple(self.migration_rates.items())
        ))

    def __str__(self):
        """
        String representation of the epoch.

        :return: String representation.
        """
        string = (
            f"Epoch(start_time={self.start_time:.4g}, "
            f"end_time={self.end_time:.4g}, "
            f"pop_sizes=({', '.join([f'{p}={s:.4g}' for p, s in self.pop_sizes.items()])})"
        )

        if self.n_pops > 1:
            string += (
                f", migration_rates=({', '.join([f'{p}->{q}={r:.4g}' for (p, q), r in self.migration_rates.items()])})"
            )

        return string

    def to_string(self):
        """
        Alias for :meth:`__str__`.

        :return: String representation.
        """
        return str(self)


class DemographicEvent(ABC):
    """
    Base class for (discrete) demographic events.
    """
    #: Start time of the event.
    start_time: float

    #: Population names.
    pop_names: List[str]

    @abstractmethod
    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        """
        pass

    @abstractmethod
    def _broadcast(self, epoch: Epoch):
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        pass

    @staticmethod
    def _flatten(
            rates: Dict[Any, Dict[float, float]]
    ) -> (np.ndarray, Dict[float, Dict[Any, float]]):
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
    times: np.ndarray

    def _broadcast(self, epoch: Epoch):
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        # times which are within the time interval
        times: np.ndarray = self.times[(
                (epoch.start_time < self.times) &
                (self.times <= epoch.end_time) &
                (self.times > 0)
        )]

        # if there are times within the interval
        # set the end time to the most recent time
        if len(times):
            epoch.end_time = times[0]


class DiscreteRateChanges(DiscreteDemographicEvent):
    """
    Demographic event for discrete changes in population sizes and migration rates.
    """

    def __init__(
            self,
            pop_sizes: Dict[str, Dict[float, float]] = None,
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] = None
    ):
        """
        Initialize the population size change.

        :param pop_sizes: Population sizes. Either a dictionary of the form `{pop_i: {time1: size1, time2: size2}}`,
            indexed by population name, or a list of dictionaries of the form `{time1: size1, time2: size2}` ordered
            by population index, or a single dictionary of the form `{time1: size1, time2: size2}` for a single
            population.
        :param migration_rates: Migration rates. A dictionary of the form `{(pop_i, pop_j): {time1: rate1, time2:
            rate2}}` of migration from population `pop_i` to population `pop_j` at time `time1` etc.
        """
        if pop_sizes is None:
            pop_sizes = {}

        if migration_rates is None:
            migration_rates = {}

        if not isinstance(pop_sizes, dict):
            raise ValueError('Population sizes must be a dictionary.')

        if not isinstance(migration_rates, dict):
            raise ValueError('Migration rates must be a dictionary.')

        if len(pop_sizes) == 0 and len(migration_rates) == 0:
            raise ValueError('Either one population size or migration rate must be specified.')

        # make sure population sizes are positive
        for p, sizes in pop_sizes.items():
            if any(s <= 0 for s in sizes.values()):
                raise ValueError(f'Population sizes must be positive at all times.')

        # initialize zero migration rates if None is given
        if migration_rates is None:
            migration_rates = {}
        elif not isinstance(migration_rates, dict):
            raise ValueError('Migration rates must be a dictionary.')

        #: Population names.
        self.pop_names: List[str] = sorted(list(set(pop_sizes.keys()).union(
            {p for k in migration_rates for p in k})))

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_names)

        # flatten the population sizes and migration rates
        times: np.ndarray
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
        self.times: np.ndarray = times

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

        :param pop_sizes: Population sizes. A dictionary of the form `{pop_i: {time1: size1, time2: size2}}`.
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

    def __init__(self, rates: Dict[Tuple[str, str], Dict[float, float]]):
        """
        Initialize the (backwards-time) migration rate change.

        :param rates: Migration rates. A dictionary of the form
            `{(pop_i, pop_j): {time1: rate1, time2: rate2}}` of migration from population `pop_i` to population
            `pop_j` at time `time1` etc.
        """
        super().__init__(migration_rates=rates)


class MigrationRateChange(MigrationRateChanges):
    """
    Demographic event for a single change in migration rate.
    """

    def __init__(self, source: str, dest: str, time: float, rate: float):
        """
        Initialize the (backwards-time) migration rate change.

        :param source: Source population name.
        :param dest: Destination population name.
        :param time: Time at which the migration rate changes.
        :param rate: Migration rate.
        """
        super().__init__({(source, dest): {time: rate}})


class SymmetricMigrationRateChanges(MigrationRateChanges):
    """
    Demographic event for changes in symmetric migration rates.
    """

    def __init__(self, pops: Iterable[str], rate: Dict[float, float] | float):
        """
        Initialize the (backwards-time) migration rate change.

        :param pops: Population names across which the migration rates change uniformly.
        :param rate: Migration rates. A dictionary of the form `{time1: rate1, time2: rate2}` of migration
            from population `pop_i` to population `pop_j` at time `time1` etc. or alternatively a single float
            if the migration rate is constant over time.
        """
        if isinstance(rate, (float, int)):
            rate = {0: rate}

        rate = {(p, q): rate for p in pops for q in pops if p != q}

        super().__init__(rates=rate)


class PopulationSplit(DiscreteDemographicEvent):
    """
    Demographic event for a population split (forward in time).
    This corresponds to population merger backwards in time.
    Since ``phasegen`` does not support deterministic lineage movement due to its inherent structure,
    we can model a population split by specifying a large unidirectional migration rate from the derived
    to the ancestral population.
    """

    def __init__(
            self,
            time: float,
            derived: str | List[str],
            ancestral: str,
            multiplier: float = 100
    ):
        """
        Initialize the population split.

        :param time: Time of the split.
        :param derived: Derived populations from which all lineages move to the ancestral population.
        :param ancestral: Ancestral population to which all lineages move.
        :param multiplier: Migration rate multiplier. The migration rate from the derived to the ancestral population is
            set to the population size of the derived population times this multiplier. This value should be chosen
            large enough to ensure that the lineages move to the ancestral population *fast enough*.
        """
        if isinstance(derived, str):
            derived = [derived]

        #: Time of the split.
        self.start_time: float = time

        #: Times at which the event occurs.
        self.times: np.ndarray = np.array([time])

        #: Population names.
        self.pop_names: List[str] = sorted(derived + [ancestral])

        #: Derived populations.
        self.derived: List[str] = derived

        #: Ancestral population.
        self.ancestral: str = ancestral

        #: Migration rate multiplier.
        self.multiplier: float = multiplier

    def _apply(self, epoch: Epoch):
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        """
        # if epoch is contained in the event
        if epoch.start_time <= self.start_time < epoch.end_time:
            # specify high migration rate from derived to ancestral population
            for p in self.derived:
                epoch.migration_rates[(self.ancestral, p)] = epoch.pop_sizes[p] * self.multiplier

            # set all derived population sizes to zero
            # for p in self.derived:
            #    epoch.pop_sizes[p] = 0

            # set all migration rates to the derived populations to zero
            for p in self.derived:
                for q in epoch.pop_names:
                    epoch.migration_rates[(p, q)] = 0


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
            source: str | None = None,
            dest: str | None = None,
            step_size: float = 0.1
    ):
        """
        Initialize the population size change.

        :param trajectory: Trajectory function taking the time as argument and returning the rate.
        :param start_time: Start time of the event.
        :param end_time: End time of the event.
        :param pop: Population name or None if no population size changes.
        :param source: Source population name or None if no migration rate changes.
        :param dest: Destination population name or None if no migration rate changes.
        :param step_size: Step size used for the discretization.
        """
        if pop is None and (source is None or dest is None):
            raise ValueError('Either pop or source_pop and dest_pop must be specified.')

        #: Population name.
        self.pop: str | None = pop

        #: Population names.
        self.pop_names: List[str] = sorted(list(p for p in {pop, source, dest} if p is not None))

        #: Start time of the event.
        self.start_time: float = start_time

        #: End time of the event.
        self.end_time: float = end_time

        #: Trajectory function.
        self.trajectory: Callable[[float], float] = trajectory

        #: Step size used for the discretization.
        self.step_size: float = step_size

        #: Source population name.
        self.source_pop: str | None = source

        #: Destination population name.
        self.dest_pop: str | None = dest

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
                source=k[0] if isinstance(k, tuple) else None,
                dest=k[1] if isinstance(k, tuple) else None,
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
