"""
Demographic events and demography class.
"""

import itertools
import re
import logging
import numbers
from abc import abstractmethod, ABC
from collections import defaultdict
from .caching import cached_property
from typing import List, Callable, Dict, Iterable, Tuple, Any, Iterator, Literal, Sequence, TYPE_CHECKING

import dill
import numpy as np

from .coalescent_models import CoalescentModel, StandardCoalescent
from .settings import Settings

if TYPE_CHECKING:
    import msprime
    from matplotlib import pyplot as plt
    from .visualization import _CurveData

logger = logging.getLogger('phasegen')


class Demography:
    r"""
    Class storing full demographic information: piecewise-constant population sizes :math:`N(t)` and backward-in-time
    migration rates :math:`m_{ij}(t)`, resolved into a sequence of epochs on which both are constant. Within an epoch
    the coalescent generator :math:`\mathbf{S}` is therefore constant, and consecutive epochs differ only in
    :math:`N(t)` and :math:`m_{ij}(t)` (see :class:`~phasegen.demography.Epoch`).
    """
    #: Population names.
    pop_names: List[str]

    #: Number of populations.
    n_pops: int

    #: Whether the warning about events setting the same rate in one epoch was issued. Static for backward
    #: compatibility.
    _issued_overlap_warning: bool = False

    #: Coalescent model whose pairwise coalescence rate sets the drain rate of a population split. ``None`` stands
    #: for the standard coalescent. Static for backward compatibility.
    _model: CoalescentModel | None = None

    def __init__(
            self,
            events: List['DemographicEvent'] = None,
            pop_sizes: Dict[str, Dict[float, float]] | Dict[str, float] | Dict[float, float] | float = None,
            migration_rates: Dict[Tuple[str, str], Dict[float, float]] | Dict[Tuple[str, str], float] = None,
            warn_n_epochs: int = 20
    ) -> None:
        """
        Initialize the demography.

        :param events: List of demographic events.
        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name and time at which the population size changes, or a dictionary of the form
            ``{pop_i: size}`` if the population size is constant, or a single float if there is only one population
            and the population size is constant, or a dictionary of the form ``{time1: size1, time2: size2}`` for a
            single population.
        :param migration_rates: Migration rates. A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2:
            rate2}}`` of migration from population ``pop_i`` to population ``pop_j`` backwards in time from time
            ``time1`` etc., or alternatively a dictionary of the form ``{(pop_i, pop_j): rate}`` if the migration
            rate is constant over time.
        :param warn_n_epochs: Threshold for the number of epochs considered after which a warning is issued.
        """
        if events is None:
            events = []

        if pop_sizes is None:
            pop_sizes = {}

        # assuming a single population with constant size if a float is given
        elif isinstance(pop_sizes, numbers.Real):
            pop_sizes = {'pop_0': {0: pop_sizes}}

        # assuming a single population if only a dictionary of time to size is given
        elif isinstance(pop_sizes, dict) and pop_sizes and isinstance(list(pop_sizes.keys())[0], numbers.Real):
            pop_sizes = {'pop_0': pop_sizes}

        # assuming constant population sizes if only a dictionary of population to size is given
        elif isinstance(pop_sizes, dict) and pop_sizes and isinstance(list(pop_sizes.values())[0], numbers.Real):
            pop_sizes = {p: {0: s} for p, s in pop_sizes.items()}

        if migration_rates is None:
            migration_rates = {}

        # wrap migration rate in dictionary if only one time per migration pair is given
        elif isinstance(migration_rates, dict) and migration_rates and isinstance(list(migration_rates.values())[0], numbers.Real):
            migration_rates = {(p, q): {0: r} for (p, q), r in migration_rates.items()}

        #: The logger instance
        self._logger = logger.getChild(self.__class__.__name__)

        #: Threshold for the number of epochs considered after which a warning is issued.
        self.warn_n_epochs: int = int(warn_n_epochs)

        #: Whether a warning about the number of epochs has been already issued.
        self._issued_warning = False

        #: Whether the warning about events setting the same rate in one epoch was issued.
        self._issued_overlap_warning = False

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

    def _prepare_events(self) -> None:
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

        :param max_epochs: Maximum number of epoch changes to use, a warning being logged if the demography has more. Note
            that the number of epochs may be infinite.
        :return: msprime demography object.
        :raise ImportError: If Msprime is not installed.
        """
        try:
            import msprime as ms
        except ImportError:
            raise ImportError('Msprime must be installed to use this method.')

        self._prepare_events()

        first_epoch = next(self.epochs)

        names = self._msprime_names

        # create demography object
        d: ms.Demography = ms.Demography(
            populations=[ms.Population(name=names[pop], initial_size=first_epoch.pop_sizes[pop])
                         for pop in self.pop_names],
            migration_matrix=np.array([[first_epoch.migration_rates[(p, q)] for q in self.pop_names]
                                       for p in self.pop_names])
        )

        # iterate over epochs
        epoch = first_epoch
        for epoch in itertools.islice(self.epochs, 1, int(max_epochs) + 1):

            # iterate over populations
            for pop in self.pop_names:
                # add population size changes
                # noinspection PyTypeChecker
                d.add_population_parameters_change(
                    time=epoch.start_time,
                    initial_size=epoch.pop_sizes[pop],
                    population=names[pop]
                )

            # iterate over migration rates
            for (p, q) in itertools.product(self.pop_names, repeat=2):

                if p != q:
                    # noinspection all
                    d.add_migration_rate_change(
                        time=epoch.start_time,
                        rate=epoch.migration_rates[(p, q)],
                        source=names[p],
                        dest=names[q]
                    )

        if self.has_n_epochs(int(max_epochs) + 2):
            self._logger.warning(
                "The demography has more than %d epochs, so the msprime demography keeps the rates of the last of "
                "them from %g on. Pass a larger max_epochs or a coarser discretization.",
                int(max_epochs) + 1, epoch.start_time
            )

        # sort events by time
        d.sort_events()

        return d

    @property
    def _msprime_names(self) -> Dict[str, str]:
        """
        The msprime name of each population. msprime requires Python identifiers, so any other character is replaced
        by an underscore, a name not starting with a letter or an underscore is prefixed with ``pop_``, and a clash is
        resolved by a numeric suffix. Identifiers are kept as they are.

        :return: The msprime name by population name.
        """
        names, taken = {}, {p for p in self.pop_names if p.isidentifier()}

        for pop in self.pop_names:
            if pop.isidentifier():
                names[pop] = pop
                continue

            name = re.sub(r'\W', '_', pop)
            if not name.isidentifier():
                name = f'pop_{name}'

            candidate, i = name, 1
            while candidate in taken:
                candidate, i = f'{name}_{i}', i + 1

            names[pop] = candidate
            taken.add(candidate)

        return names

    def _to_demes(self) -> 'demes.Graph':
        """
        Convert to demes object (see https://tskit.dev/msprime/docs/stable/api.html#msprime.Demography.to_demes).
        demes accepts migration rates of at most 1, so a larger rate is rejected by msprime's conversion.

        :return: Demes object.
        :raise ImportError: If msprime is not installed.
        :raise ValueError: If a migration rate exceeds 1 (``migration[0]: invalid migration``).
        """
        return self.to_msprime().to_demes()

    @property
    def epochs(self) -> Iterator['Epoch']:
        """
        Get a generator for the epochs. An epoch is built once and kept, so later iterations and lookups reuse it.
        """
        self._prepare_events()

        epochs, i = self._built_epochs(), 0

        while True:
            if i == len(epochs):
                epochs.append(self._build_epoch(epochs[-1] if epochs else None, i))

            yield epochs[i]

            if epochs[i].end_time == np.inf:
                return

            i += 1

    def _built_epochs(self) -> List['Epoch']:
        """
        The epochs built so far, discarded when the events change.

        :return: The epochs, in order.
        """
        key = (id(self._model),) + tuple(map(id, self.events))

        if self.__dict__.get('_epoch_key') != key:
            self.__dict__['_epoch_key'] = key
            self.__dict__['_epoch_cache'] = []

        return self.__dict__['_epoch_cache']

    def _build_epoch(self, prev: 'Epoch | None', i: int) -> 'Epoch':
        """
        Build the epoch following ``prev``, the first one when ``prev`` is ``None``.

        :param prev: The previous epoch.
        :param i: The index of the epoch.
        :return: The epoch.
        """
        if prev is None:
            prev = Epoch(
                start_time=0,
                end_time=0,
                pop_sizes={p: 1 for p in self.pop_names},
                migration_rates={k: 0 for k in itertools.product(self.pop_names, repeat=2)}
            )

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

        # apply the rate changes, then isolate the derived population of every split before any split sets its
        # drain rate, so that the epoch does not depend on the order of events sharing a start time
        splits = [e for e in self.events if isinstance(e, PopulationSplit)]

        # a rate set by two events in one epoch keeps the value of the later one in the list, which is sorted by
        # start time, so the precedence depends on how the changes are grouped into events
        setters = {}
        for e in self.events:
            if not isinstance(e, PopulationSplit):
                for key in e._apply(epoch):
                    setters.setdefault(key, []).append(e)

        clashes = [k for k, events in setters.items() if len(events) > 1]
        if clashes and not self._issued_overlap_warning:
            self._logger.warning(
                "Several events set %s in the epoch starting at %g. The event starting last takes precedence, so "
                "combine the changes into one event to make the precedence explicit.",
                ', '.join(map(str, clashes)), epoch.start_time
            )
            self._issued_overlap_warning = True

        for e in splits:
            e._isolate(epoch)

        model = StandardCoalescent() if self._model is None else self._model

        for e in splits:
            e._apply(epoch, model)

        epoch.index = i

        return epoch

    def __getstate__(self) -> dict:
        """
        The state for serialization, without the epochs built so far.

        :return: State.
        """
        state = self.__dict__.copy()
        state.pop('_epoch_cache', None)
        state.pop('_epoch_key', None)
        state.pop('_epoch_index', None)

        return state

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
        :raises ValueError: If a time is negative or NaN, or infinite on a demography with infinitely many epochs.
        """
        t = np.asarray(list(t), dtype=float)

        if np.isnan(t).any() or (t < 0).any():
            raise ValueError(f'Epochs are defined at non-negative times, got {t[np.isnan(t) | (t < 0)][0]}.')

        if np.isinf(t).any() and not self._has_finitely_many_epochs:
            raise ValueError('The demography has infinitely many epochs, so there is no epoch at infinity.')

        # build the epochs up to the latest time, then look each time up among their start times
        t_max = t.max(initial=0.0)
        epochs = self._built_epochs()

        if not epochs or epochs[-1].end_time <= t_max:
            for epoch in self.epochs:
                if epoch.end_time > t_max:
                    break

            epochs = self._built_epochs()

        # the start times and the object array of the built epochs, rebuilt when epochs are added or discarded
        index = self.__dict__.get('_epoch_index')
        if index is None or index[0] is not epochs or len(index[1]) != len(epochs):
            index = (epochs, np.array([e.start_time for e in epochs]), np.array(epochs, dtype=object))
            self.__dict__['_epoch_index'] = index

        return index[2][np.searchsorted(index[1], t, side='right') - 1]

    def get_epoch(self, t: float = 0) -> 'Epoch':
        """
        Get the epoch at the given time.

        :param t: Time.
        :return: Epoch.
        :raises ValueError: If ``t`` is negative or NaN, or infinite on a demography with infinitely many epochs.
        """
        return self.get_epochs([t])[0]

    @property
    def _has_finitely_many_epochs(self) -> bool:
        """
        Whether the demography has finitely many epochs, which holds unless a discretized event has no end.

        :return: Whether the number of epochs is finite.
        """
        return all(e.end_time < np.inf for e in self.events if isinstance(e, DiscretizedDemographicEvent))

    def add_events(self, events: List['DemographicEvent']) -> None:
        """
        Add demographic events.

        :param events: List of demographic events.
        """
        self.events += events

        self._prepare_events()

    def add_event(self, event: 'DemographicEvent') -> None:
        """
        Add a demographic event.

        :param event: Demographic event.
        """
        self.add_events([event])

    def _plot_data(self, t: np.ndarray = None, kind: Literal['all', 'pop_sizes', 'migration'] = 'all') -> '_CurveData':
        """
        Trajectories of the population sizes and migration rates, as drawn by :meth:`plot`, :meth:`plot_pop_sizes`
        and :meth:`plot_migration`.

        :param t: Times at which to evaluate the trajectories. By default, :attr:`Settings.plot_demography_n_grid`
            points from 0 to :attr:`Settings.plot_demography_end_time`.
        :param kind: The trajectories to include: ``'pop_sizes'`` (one per population, named after it),
            ``'migration'`` (one per ordered pair of distinct populations, named ``'<pop>-><pop>'``), or ``'all'``.
        :return: The trajectories, one row of values per name.
        :raises ValueError: If ``kind`` is unknown.
        """
        from .visualization import _CurveData

        labels = dict(
            all=('Demography', '$N_e, m_{ij}$'),
            pop_sizes=('Population size trajectory', '$N_e$'),
            migration=('Migration rate trajectory', '$m_{ij}$')
        )

        if kind not in labels:
            raise ValueError(f"Unknown kind {kind!r}, must be one of {list(labels)}.")

        if t is None:
            t = np.linspace(0, Settings.plot_demography_end_time, Settings.plot_demography_n_grid)

        t = np.asarray(t, dtype=float)
        epochs = self.get_epochs(t)
        pairs = [(p, q) for p in self.pop_names for q in self.pop_names if p != q]
        names, values = [], []

        if kind in ('all', 'pop_sizes'):
            names += list(self.pop_names)
            values += [[e.pop_sizes[p] for e in epochs] for p in self.pop_names]

        if kind in ('all', 'migration'):
            names += [f"{p}->{q}" for p, q in pairs]
            values += [[e.migration_rates[pair] for e in epochs] for pair in pairs]

        title, ylabel = labels[kind]

        return _CurveData(
            x=t,
            y=np.array(values, dtype=float).reshape(len(names), len(t)),
            labels=names,
            xlabel='t',
            ylabel=ylabel,
            title=title
        )

    def _plot(self, kind: str, t: np.ndarray, show: bool, file: str, title: str, ylabel: str, ax: 'plt.Axes',
              kwargs: dict) -> 'plt.Axes':
        """
        Plot the trajectories of :meth:`_plot_data`.

        :param kind: The trajectories to include.
        :param t: Times at which to evaluate the trajectories, ``None`` for the default times.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot, ``None`` for the default title.
        :param ylabel: Label of the y-axis, ``None`` for the default label.
        :param ax: Axes object to plot to.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        from .visualization import Visualization

        return Visualization.plot_rates(ax=ax, data=self._plot_data(t, kind), file=file, show=show, title=title,
                                        ylabel=ylabel, kwargs=kwargs)

    def plot_pop_sizes(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            title: str = None,
            ylabel: str = None,
            ax: 'plt.Axes' = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the population size over time.

        :param t: Times at which to plot the population sizes. By default,
            :attr:`~phasegen.settings.Settings.plot_demography_n_grid` points from 0 to
            :attr:`~phasegen.settings.Settings.plot_demography_end_time`.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot, ``None`` for the default title.
        :param ylabel: Label of the y-axis, ``None`` for the default label.
        :param ax: Axes object to plot to.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        return self._plot('pop_sizes', t, show, file, title, ylabel, ax, kwargs)

    def plot_migration(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            title: str = None,
            ylabel: str = None,
            ax: 'plt.Axes' = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the migration rates over time.

        :param t: Times at which to plot the migration rates. By default,
            :attr:`~phasegen.settings.Settings.plot_demography_n_grid` points from 0 to
            :attr:`~phasegen.settings.Settings.plot_demography_end_time`.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Title of the plot, ``None`` for the default title.
        :param ylabel: Label of the y-axis, ``None`` for the default label.
        :param ax: Axes object to plot to.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        return self._plot('migration', t, show, file, title, ylabel, ax, kwargs)

    def plot(
            self,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            ylabel: str = None,
            ax: 'plt.Axes' = None,
            title: str = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot the population sizes and migration rates over time.

        :param t: Times at which to plot the trajectories. By default,
            :attr:`~phasegen.settings.Settings.plot_demography_n_grid` points from 0 to
            :attr:`~phasegen.settings.Settings.plot_demography_end_time`.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param ylabel: Label of the y-axis, ``None`` for the default label.
        :param ax: Axes object to plot to.
        :param title: Title of the plot, ``None`` for the default title.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: Axes object.
        """
        return self._plot('all', t, show, file, title, ylabel, ax, kwargs)

class Epoch:
    r"""
    Epoch of a demographic scenario with constant population sizes :math:`N` and migration rates :math:`m_{ij}`. As
    both are constant over the epoch, the coalescent generator :math:`\mathbf{S}` is constant here, and coalescence
    rates scale inversely with :math:`N`: under the standard (Kingman) coalescent a state with :math:`i` lineages
    coalesces at rate :math:`\binom{i}{2}/N`.
    """

    #: Start time of the epoch.
    start_time: float

    #: End time of the epoch.
    end_time: float

    #: Population sizes.
    pop_sizes: Dict[str, float]

    #: Migration rates.
    migration_rates: Dict[Tuple[str, str], float]

    #: Position of the epoch in its demography, counted from 0. It does not enter the equality of epochs.
    index: int = 0

    def __init__(
            self,
            start_time: float = 0,
            end_time: float = np.inf,
            pop_sizes: Dict[str, float] = None,
            migration_rates: Dict[Tuple[str, str], float] = None
    ) -> None:
        """
        Initialize the epoch.

        :param start_time: Start time of the epoch.
        :param end_time: End time of the epoch.
        :param pop_sizes: Population sizes. By default, we have ``{'pop_0': 1}``.
        :param migration_rates: Migration rates of the form ``{(pop_i, pop_j): rate}``, where ``rate`` is the
            backward-in-time rate at which a lineage moves from population ``pop_i`` to population ``pop_j``. By
            default, we have zero migration rates between all populations.
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
        r"""
        Time interval of the epoch, :math:`\tau = t_{\mathrm{end}} - t_{\mathrm{start}}`.
        """
        return self.end_time - self.start_time

    def __eq__(self, other) -> bool:
        """
        Compare epochs using their hash.

        :param other: The other epoch.
        :return: Whether the epochs are equal.
        """
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """
        Hash the epoch. Note that we do not include the start and end time, since they are not relevant for the
        state space created from the epoch.

        :return: Hash of the epoch.
        """
        return hash((
            tuple(self.pop_sizes.items()),
            tuple(self.migration_rates.items())
        ))

    def __str__(self) -> str:
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


class DemographicEvent(ABC):
    """
    Base class for (discrete) demographic events.
    """
    #: Start time of the event.
    start_time: float

    #: Population names.
    pop_names: List[str]

    @abstractmethod
    def _apply(self, epoch: Epoch) -> set:
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        :return: The keys of the rates set, population names for sizes and ``(source, dest)`` pairs for migration.
        """
        pass

    @abstractmethod
    def _broadcast(self, epoch: Epoch) -> None:
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

    def _broadcast(self, epoch: Epoch) -> None:
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
    ) -> None:
        """
        Initialize the population size change.

        :param pop_sizes: Population sizes, a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}`` indexed
            by population name. :class:`~phasegen.demography.Demography` also accepts the single-population and
            constant-size forms and normalises them to this one.
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

        for key in migration_rates:
            if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(p, str) for p in key)):
                raise ValueError(f'Migration rates must be keyed by (source, destination) pairs of population names, '
                                 f'got {key!r}.')

            if key[0] == key[1]:
                raise ValueError(f'Migration rates must be between distinct populations, got {key!r}.')

        #: Population names.
        self.pop_names: List[str] = sorted(list(set(pop_sizes.keys()).union(
            {p for k in migration_rates for p in k})))

        #: Number of populations / demes.
        self.n_pops: int = len(self.pop_names)

        # flatten the population sizes and migration rates
        times: np.ndarray
        rates: Dict[float, Dict[Any, float]]
        times, rates = self._flatten(pop_sizes | migration_rates)

        # the negated comparisons also reject NaN
        if np.any(~(np.array(times, dtype=float) >= 0)):
            raise ValueError('All times must not be negative.')

        migration = np.array([rates[k][t] for k in rates for t in migration_rates if t in rates[k]], dtype=float)
        if np.any(~((migration >= 0) & (migration < np.inf))):
            raise ValueError('Migration rates must be finite and non-negative at all times.')

        sizes = np.array([rates[k][t] for k in rates for t in pop_sizes if t in rates[k]], dtype=float)
        if np.any(~((sizes > 0) & (sizes < np.inf))):
            raise ValueError('Population sizes must be finite and positive at all times.')

        #: Times at which the population size changes occur.
        self.times: np.ndarray = times

        #: Population sizes.
        self.pop_sizes: Dict[float, Dict[str, float]] = {
            t: {x: pops[x] for x in self.pop_names if x in pops} for t, pops in rates.items()
        }

        #: Migration rates at each time.
        self.migration_rates: Dict[float, Dict[Tuple[str, str], float]] = {
            t: {(p, q): rates[t][(p, q)] for p in self.pop_names for q in self.pop_names if (p, q) in rates[t]}
            for t in rates
        }

        #: Start time of the event.
        self.start_time: float = self.times[0]

    def _apply(self, epoch: Epoch) -> set:
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        :return: The keys of the rates set.
        """
        keys = set()

        for t in self.times[(epoch.start_time <= self.times) & (self.times < epoch.end_time)]:
            epoch.pop_sizes |= self.pop_sizes[t]
            epoch.migration_rates |= self.migration_rates[t]
            keys |= set(self.pop_sizes[t]) | set(self.migration_rates[t])

        return keys


class PopSizeChanges(DiscreteRateChanges):
    """
    Demographic event for changes in population size.
    """

    def __init__(self, pop_sizes: Dict[str, Dict[float, float]]) -> None:
        """
        Initialize the population size change.

        :param pop_sizes: Population sizes. A dictionary of the form `{pop_i: {time1: size1, time2: size2}}`.
        """
        super().__init__(pop_sizes=pop_sizes)


class PopSizeChange(PopSizeChanges):
    """
    Demographic event for a single change in population size.
    """

    def __init__(self, pop: str, time: float, size: float) -> None:
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

    def __init__(self, rates: Dict[Tuple[str, str], Dict[float, float]]) -> None:
        """
        Initialize the (backwards-time) migration rate change.

        :param rates: Migration rates. A dictionary of the form
            `{(pop_i, pop_j): {time1: rate1, time2: rate2}}` of migration from population `pop_i` to population
            `pop_j` backwards in time with `rate1` from time `time1` etc.
        """
        super().__init__(migration_rates=rates)


class MigrationRateChange(MigrationRateChanges):
    """
    Demographic event for a single change in migration rate.
    """

    def __init__(self, source: str, dest: str, time: float, rate: float) -> None:
        """
        Initialize the (backwards-time) migration rate change.

        :param source: Source population name.
        :param dest: Destination population name.
        :param time: Time at which the migration rate changes.
        :param rate: Migration rate from the source to the destination population backwards in time.
        """
        super().__init__({(source, dest): {time: rate}})


class SymmetricMigrationRateChanges(MigrationRateChanges):
    """
    Demographic event for changes in symmetric migration rates.
    """

    def __init__(self, pops: Iterable[str], rate: Dict[float, float] | float) -> None:
        """
        Initialize the (backwards-time) migration rate change.

        :param pops: Population names across which the migration rates change uniformly.
        :param rate: Migration rates. A dictionary of the form `{time1: rate1, time2: rate2}` of migration
            from population `pop_i` to population `pop_j` at time `time1` etc. or alternatively a single float
            if the migration rate is constant over time.
        """
        if isinstance(rate, numbers.Real):
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
    ) -> None:
        r"""
        Initialize the population split.

        :param time: Time of the split.
        :param derived: Derived populations from which all lineages move to the ancestral population.
        :param ancestral: Ancestral population to which all lineages move.
        :param multiplier: Migration rate multiplier. The migration rate from the derived to the ancestral population
            is set to :math:`m = c \max_i \lambda_{2,2} / \tau(N_i)`, the multiplier :math:`c` times the fastest
            pairwise coalescence rate of the epoch, where :math:`\lambda_{2,2}` is the rate at which two lineages
            merge under the coalescent model (1 for the standard and the Beta coalescent, :math:`1 + c_D \psi^2` for
            the Dirac coalescent with parameters :math:`\psi` and :math:`c_D`), and :math:`\tau(N_i)` is the
            model's time scale for population size :math:`N_i` (:math:`N_i` for the standard coalescent). The rate is
            set in the epoch of the split and again in every later epoch. A lineage thus leaves a derived population
            after a mean time :math:`1 / m`, a fraction :math:`1 / c` of the mean time to coalescence of a pair in the
            population that coalesces fastest. With more than two lineages in a population the first coalescence
            comes sooner, so the fraction of coalescent time the drain displaces is larger than :math:`1 / c`.
        :raises ValueError: If the time is negative, the ancestral population is among the derived ones, or the
            multiplier is not positive and finite.
        """
        if isinstance(derived, str):
            derived = [derived]

        if not time >= 0:
            raise ValueError(f'The split time must be non-negative, got {time}.')

        if ancestral in derived:
            raise ValueError(f'The ancestral population {ancestral!r} must not be among the derived populations.')

        if not 0 < multiplier < np.inf:
            raise ValueError(f'The migration rate multiplier must be positive and finite, got {multiplier}.')

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

    def _isolate(self, epoch: Epoch) -> None:
        """
        Switch off all migration into and out of the derived populations if the split falls into the epoch, so that,
        backward in time, no lineage enters a drained derived population. :class:`Demography` isolates the derived
        populations of all splits before it applies any of them.

        :param epoch: Epoch.
        """
        if epoch.start_time <= self.start_time < epoch.end_time:
            for p in self.derived:
                for q in epoch.pop_names:
                    if q != p:
                        epoch.migration_rates[(p, q)] = 0
                        epoch.migration_rates[(q, p)] = 0

    def _apply(self, epoch: Epoch, model: CoalescentModel) -> set:
        """
        Set the drain rate from each derived population to the ancestral population in every epoch from the split
        onwards, so that a later split isolating the ancestral population does not strand lineages still in a derived
        one. The rate uses the population sizes of the epoch, so it must be applied after the rate changes and after
        :meth:`PopulationSplit._isolate() <phasegen.demography.PopulationSplit._isolate>` of every split.

        :param epoch: Epoch.
        :param model: Coalescent model, whose pairwise coalescence rate and time scale set the drain rate.
        :return: The keys of the drain rates set.
        """
        if self.start_time >= epoch.end_time:
            return set()

        # the drain rate is a multiple of the fastest pairwise coalescence rate of the epoch, so that the lineages
        # leave the derived populations before any coalescence the split displaces
        timescale = min(model._get_timescale(N) for N in epoch.pop_sizes.values())
        rate = self.multiplier * model._get_rate(b=2, k=2) / timescale

        for p in self.derived:
            epoch.migration_rates[(p, self.ancestral)] = rate

        return {(p, self.ancestral) for p in self.derived}


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
    ) -> None:
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

        if pop is None and source == dest:
            raise ValueError(f'Migration must be between distinct populations, got source and destination {source!r}.')

        if not step_size > 0:
            raise ValueError(f'The step size must be positive, got {step_size}.')

        if not 0 <= start_time <= end_time:
            raise ValueError(f'The times must satisfy 0 <= start_time <= end_time, got {start_time} and {end_time}.')

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

    def __getstate__(self) -> dict:
        """
        The state for serialization, with the trajectory dumped by ``dill``, which also restores a lambda or a closure.

        :return: State.
        """
        state = self.__dict__.copy()
        state['trajectory'] = dill.dumps(self.trajectory, recurse=True)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore the state from a serialized state.

        :param state: State.
        """
        self.__dict__.update(state)

        if isinstance(self.__dict__.get('trajectory'), bytes):
            self.trajectory = dill.loads(self.trajectory)

    def _broadcast(self, epoch: Epoch) -> None:
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        # return if there is no overlap
        if epoch.end_time < self.start_time or epoch.start_time >= self.end_time:
            return

        # if this event starts after the epoch, we take the start time
        if self.start_time > epoch.start_time:
            epoch.end_time = self.start_time
        else:
            # only lower the end time, and never overshoot the last (possibly partial) step or the event's end time
            n_steps = np.ceil((epoch.start_time - self.start_time + 1e-10) / self.step_size)

            # the offset absorbs rounding only at small times, so step on until the epoch has positive length
            while self.start_time + n_steps * self.step_size <= epoch.start_time:
                n_steps += 1

            epoch.end_time = min(epoch.end_time, self.start_time + n_steps * self.step_size, self.end_time)

    def _apply(self, epoch: Epoch) -> set:
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        :return: The keys of the rates set.
        """
        # if epoch is contained in the event, up to and including a trailing partial interval ending at self.end_time
        if self.start_time <= epoch.start_time and epoch.end_time <= self.end_time:

            rate_start = self.trajectory(epoch.start_time)
            rate_end = self.trajectory(epoch.end_time)
            rate = (rate_start + rate_end) / 2

            if self.pop is None:
                if not 0 <= rate < np.inf:
                    raise ValueError(f'The migration rate trajectory from {self.source_pop} to {self.dest_pop} gives '
                                     f'{rate} on [{epoch.start_time:g}, {epoch.end_time:g}), which is not finite and '
                                     f'non-negative.')

                epoch.migration_rates[(self.source_pop, self.dest_pop)] = rate

                return {(self.source_pop, self.dest_pop)}

            # a decaying trajectory may underflow to zero far out, which a simulation accepts and the exact
            # computation rejects where it reaches that epoch
            if not 0 <= rate < np.inf:
                raise ValueError(f'The population size trajectory of {self.pop} gives {rate} on '
                                 f'[{epoch.start_time:g}, {epoch.end_time:g}), which is negative or not finite.')

            epoch.pop_sizes[self.pop] = rate

            return {self.pop}

        return set()


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
    ) -> None:
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

    def _broadcast(self, epoch: Epoch) -> None:
        """
        Adjust the end time of the epoch to the next time at which the rate changes due to this event.

        :param epoch: Epoch.
        """
        for e in self.events.values():
            e._broadcast(epoch)

    def _apply(self, epoch: Epoch) -> set:
        """
        Apply the demographic event to the given epoch if applicable.

        :param epoch: Epoch.
        :return: The keys of the rates set.
        """
        return set().union(*(e._apply(epoch) for e in self.events.values()))


class _ExponentialTrajectory:
    r"""
    The exponential trajectory :math:`x(t) = x_0 \exp\!\big(-g\,(t - t_0)\big)` as a callable object, which
    serializes with its parameters where a closure would not.
    """

    def __init__(self, x0: float, g: float, t0: float) -> None:
        """
        :param x0: Value at the start time.
        :param g: Growth rate.
        :param t0: Start time.
        """
        self.x0: float = x0
        self.g: float = g
        self.t0: float = t0

    def __call__(self, t: float) -> float:
        """
        :param t: Time.
        :return: The value at ``t``.
        """
        return self.x0 * np.exp(-self.g * (t - self.t0))


class ExponentialRateChanges(DiscretizedRateChanges):
    r"""
    Demographic event for exponential rate changes of multiple populations or migration rates. Each rate follows the
    trajectory :math:`x(t) = x_0 \exp\!\big(-g\,(t - t_0)\big)`, with initial value :math:`x_0` at start time
    :math:`t_0` and growth rate :math:`g`, discretized into piecewise-constant steps (see
    :class:`~phasegen.demography.DiscretizedRateChanges`).
    """

    def __init__(
            self,
            initial_rate: Dict[Any, float],
            growth_rate: Dict[Any, float] | float,
            start_time: Dict[Any, float] | float,
            end_time: Dict[Any, float] | float = np.inf,
            step_size: float = 0.1
    ) -> None:
        """
        Initialize the exponential growth.

        :param initial_rate: Initial rates. A dictionary mapping keys to values. Keys are either population names or
            tuples of population names for population sizes and migration rates, respectively.
        :param growth_rate: Exponential growth rates. A single value or a dictionary mapping keys to values.
        :param start_time: Start times of the growth. A single value or a dictionary mapping keys to values.
        :param end_time: End times of the growth.
        :param step_size: Step size used for the discretization.
        """

        def get_trajectory(k: Any) -> '_ExponentialTrajectory':
            """
            Get the trajectory for the given key.

            :param k: Key.
            :return: Trajectory, a callable of the time.
            """
            g = growth_rate[k] if isinstance(growth_rate, dict) else growth_rate
            t0 = start_time[k] if isinstance(start_time, dict) else start_time
            x0 = initial_rate[k] if isinstance(initial_rate, dict) else initial_rate

            return _ExponentialTrajectory(x0=x0, g=g, t0=t0)

        super().__init__(
            trajectory={k: get_trajectory(k) for k in initial_rate},
            start_time=start_time,
            end_time=end_time,
            step_size=step_size
        )


class ExponentialPopSizeChanges(ExponentialRateChanges):
    r"""
    Demographic event for exponential population size changes of multiple populations, following
    :math:`N(t) = N_0 \exp\!\big(-g\,(t - t_0)\big)` (see :class:`~phasegen.demography.ExponentialRateChanges`).
    """

    def __init__(
            self,
            initial_size: Dict[str, float],
            growth_rate: Dict[str, float] | float,
            start_time: Dict[str, float] | float,
            end_time: Dict[str, float] | float = np.inf,
            step_size: float = 0.1
    ) -> None:
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
