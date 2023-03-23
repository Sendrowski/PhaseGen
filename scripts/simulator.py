import msprime as ms
import tskit
from typing import Generator, List
from PH import *
import JSON


class Simulator:
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """
    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List,
            times: np.ndarray | List,
            num_replicates: int,
            alpha: np.ndarray | List = None
    ):
        self.n = n
        self.pop_sizes = pop_sizes
        self.times = times
        self.num_replicates = num_replicates

        if alpha is None:
            self.alpha = np.eye(1, n - 1, 0)
        else:
            self.alpha = alpha

        self.msprime = {}
        self.ph = {}

    def simulate(self) -> None:
        """
        Simluate moment using both phase-type theory and msprime.
        :return:
        """
        self.simulate_ph()
        self.simulate_msprime()

    def simulate_msprime(self) -> None:
        """
        Simulate moments using msprime.
        :return:
        """
        # configure demography
        d = ms.Demography()
        d.add_population(initial_size=self.pop_sizes[0])

        for i in range(1, len(self.pop_sizes)):
            d.add_population_parameters_change(time=self.times[i], initial_size=self.pop_sizes[i])

        # simulate trees
        g: Generator = ms.sim_ancestry(
            samples=self.n,
            num_replicates=self.num_replicates,
            demography=d,
            model=ms.StandardCoalescent(),
            ploidy=1
        )

        ts: tskit.TreeSequence
        heights = np.zeros(self.num_replicates)
        total_branch_lengths = np.zeros(self.num_replicates)
        for i, ts in enumerate(g):
            t: tskit.Tree = ts.first()
            total_branch_lengths[i] = t.total_branch_length
            heights[i] = t.time(t.root)

        self.msprime = dict(
            # get moments of tree height
            height=dict(
                mu=np.mean(heights),
                var=np.var(heights)
            ),
            # get moments of branch length
            total_branch_length=dict(
                mu=np.mean(total_branch_lengths),
                var=np.var(total_branch_lengths)
            )
        )

    def simulate_ph(self) -> None:
        """
        Simulate moments using phase-type theory.
        :return:
        """
        cd = VariablePopulationSizeCoalescentDistribution(
            model=StandardCoalescent(),
            n=self.n,
            alpha=self.alpha,
            demography=PiecewiseConstantDemography(pop_sizes=self.pop_sizes, times=self.times)
        )

        height = dict(
            mu=cd.mean,
            var=cd.var
        )

        cd = cd.set_reward(rewards.TotalBranchLength())

        total_branch_length = dict(
            mu=cd.mean,
            var=cd.var
        )

        self.ph = dict(
            height=height,
            total_branch_length=total_branch_length
        )

    def to_file(self, file: str) -> None:
        """
        Save object to file.
        :param file:
        :return:
        """
        JSON.save(self.__dict__, file)
