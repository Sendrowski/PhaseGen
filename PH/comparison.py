from typing import List

import numpy as np

from .distributions import PiecewiseConstantDemography, VariablePopSizeCoalescent, MsprimeCoalescent, \
    ConstantPopSizeCoalescent


class Comparison:
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List,
            times: np.ndarray | List,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            alpha: np.ndarray | List = None
    ):
        self.msprime = MsprimeCoalescent(
            n=n,
            pop_sizes=pop_sizes,
            times=times,
            num_replicates=num_replicates,
            n_threads=n_threads,
            parallelize=parallelize
        )

        self.ph = VariablePopSizeCoalescent(
            n=n,
            demography=PiecewiseConstantDemography(
                pop_sizes=pop_sizes,
                times=times
            ),
            alpha=alpha
        )

        self.ph_const = ConstantPopSizeCoalescent(
            n=n,
            Ne=float(np.mean(pop_sizes)),
            alpha=alpha
        )

    def to_file(self, file: str) -> None:
        """
        Save object to file.
        :param file:
        :return:
        """
        pass
