"""
Initial distribution over lineage or locus configurations.
"""

from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from .lineage import LineageConfig
from .locus import LocusConfig


class InitialDistribution:
    r"""
    Weighted mixture of lineage or locus configurations from which the coalescent starts. Component :math:`i` with
    weight :math:`w_i` has the initial vector :math:`\boldsymbol{\alpha}_i`, and the process starts from

    .. math::
        \boldsymbol{\alpha} = \sum_i \frac{w_i}{\sum_j w_j} \boldsymbol{\alpha}_i,

    so every moment, transform, density and distribution function is the correspondingly weighted sum over the
    components. An initial distribution of lineage configurations is passed as ``n`` and one of locus configurations
    as ``loci`` to :class:`~phasegen.distributions.Coalescent` and
    :class:`~phasegen.distributions.MsprimeCoalescent`, the latter drawing the starting configuration of each
    replicate from the normalized weights.

    All components share one state space: lineage configurations have the same populations, in the same order, and
    the same total number of lineages, and locus configurations have the same number of loci and recombination rate.
    The joint block-counting state space, which underlies the joint SFS, further depends on the number of lineages
    per population, which then must agree between the components.
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
