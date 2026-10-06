"""
Initial distribution over lineage or locus configurations.
"""

from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from .lineage import LineageConfig
from .locus import LocusConfig


class InitialDistribution:
    r"""
    Mixture of starting configurations of the coalescent, for example over the number of lineages sampled in each
    deme, given as :class:`~phasegen.lineage.LineageConfig` components, or over the number of lineages whose loci are
    initially unlinked, given as :class:`~phasegen.locus.LocusConfig` components. With :math:`w_i > 0` the weight and
    :math:`\boldsymbol{\alpha}_i` the initial vector of component :math:`i`, the process starts from

    .. math::
        \boldsymbol{\alpha} = \sum_i \frac{w_i}{\sum_j w_j} \boldsymbol{\alpha}_i,

    so every raw moment, transform, density and distribution function is the weighted sum of those of the
    components, while central moments and quantiles are those of the mixture. It is passed as ``n`` or ``loci`` to
    :class:`~phasegen.distributions.Coalescent`. All components share one state space: lineage configurations have
    the same populations and total number of lineages, and, for the joint SFS and :math:`F_{ST}`, the same number per
    population, while locus configurations have the same number of loci and recombination rate.

    The following example samples three of the four lineages in ``pop_1`` with probability 0.75.

    ::

        init = pg.InitialDistribution([(1, {'pop_0': 3, 'pop_1': 1}), (3, {'pop_0': 1, 'pop_1': 3})])

        coal = pg.Coalescent(n=init, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 0.5},
            migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        ))
    """

    def __init__(
            self,
            components: Iterable[Tuple[float, LineageConfig | LocusConfig | int | Dict[str, int] | List[int]]]
    ) -> None:
        """
        Initialize the initial distribution.

        :param components: Pairs ``(weight, config)`` of a positive weight, normalized over the components, and a
            configuration, either a :class:`~phasegen.lineage.LineageConfig` or anything it accepts, or a
            :class:`~phasegen.locus.LocusConfig`.
        :raises TypeError: If a component is not a pair, a weight is not a number, or lineage and locus
            configurations are mixed.
        :raises ValueError: If there are no components, a weight is not positive and finite, or the configurations do
            not share one state space.
        """
        components = list(components)

        if not components:
            raise ValueError("An initial distribution requires at least one component.")

        if any(not isinstance(c, Sequence) or isinstance(c, str) or len(c) != 2 for c in components):
            raise TypeError("The components of an initial distribution must be (weight, config) pairs.")

        try:
            weights = np.array([float(w) for w, _ in components])
        except (TypeError, ValueError):
            raise TypeError(f"The weights must be numbers, got {[w for w, _ in components]}.") from None

        if not np.all(np.isfinite(weights) & (weights > 0)):
            raise ValueError(f"The weights must be positive and finite, got {list(weights)}.")

        configs = [c if isinstance(c, (LineageConfig, LocusConfig)) else LineageConfig(c) for _, c in components]

        if len({type(c) for c in configs}) > 1:
            raise TypeError("An initial distribution holds either lineage or locus configurations, not both.")

        if isinstance(configs[0], LineageConfig):
            shared = [(c.pop_names, c.n) for c in configs]
            what = "the same populations, in the same order, and the same total number of lineages"
        else:
            shared = [(c.n, c.recombination_rate) for c in configs]
            what = "the same number of loci and recombination rate"

        if any(s != shared[0] for s in shared):
            raise ValueError(f"The components of an initial distribution must share one state space, with {what}.")

        #: Weights of the components, normalized to sum to one.
        self.weights: np.ndarray = weights / weights.sum()

        #: Configurations of the components.
        self.configs: List[LineageConfig] | List[LocusConfig] = configs

    def __iter__(self) -> Iterator[Tuple[float, LineageConfig | LocusConfig]]:
        """
        Iterate over the components.

        :return: Iterator over the pairs ``(weight, config)`` with normalized weights.
        """
        return iter(zip(self.weights, self.configs))

    def __eq__(self, other) -> bool:
        """
        Check if two initial distributions are equal.

        :param other: Other initial distribution
        :return: Whether both have the same weights and configurations, in the same order
        """
        return (
                isinstance(other, InitialDistribution)
                and np.array_equal(self.weights, other.weights)
                and type(self.configs[0]) is type(other.configs[0])
                and self.configs == other.configs
        )

    def _map(self, func: Callable[[LineageConfig | LocusConfig], LineageConfig | LocusConfig]) -> 'InitialDistribution':
        """
        Apply a function to every configuration, keeping the weights.

        :param func: Function mapping a configuration to a new one.
        :return: The new initial distribution.
        """
        other = InitialDistribution([(1, func(c)) for c in self.configs])
        other.weights = self.weights.copy()

        return other

    @staticmethod
    def _split(
            config: 'LineageConfig | LocusConfig | InitialDistribution'
    ) -> 'Tuple[LineageConfig | LocusConfig, InitialDistribution | None]':
        """
        Split a configuration argument into the configuration of the first component, which defines the state space,
        and the initial distribution, ``None`` for a single configuration.

        :param config: A configuration or an initial distribution.
        :return: The reference configuration and the initial distribution.
        """
        if isinstance(config, InitialDistribution):
            return config.configs[0], config

        return config, None
