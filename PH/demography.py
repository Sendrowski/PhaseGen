from typing import List

import numpy as np


class Demography:
    pass


class PiecewiseConstantDemography(Demography):
    """
    Demographic scenario where containing a number
    of instantaneous population size changes.
    """

    def __init__(self, pop_sizes: np.ndarray | List, times: np.ndarray | List):
        """
        The population sizes and times these changes occur backwards in time.
        We need to start with a population size at time 0 but this time
        can be omitted in which case len(pop_size) == len(times) + 1.
        :param pop_sizes:
        :param times:
        """
        if len(pop_sizes) == 0:
            raise Exception('At least one population size must be provided')

        # add 0 if no times are specified
        if len(times) < len(pop_sizes):
            times = [0] + list(times)

        if len(times) != len(pop_sizes):
            raise Exception('The specified number of times population size change occurs'
                            'and the number of population sizes does not match.')

        self.pop_sizes: np.ndarray = np.array(pop_sizes)
        self.times: np.ndarray = np.array(times)
