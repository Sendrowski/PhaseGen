"""
Settings for the PhaseGen application.
"""


class Settings:
    #: Whether to flatten the block-counting state space when possible.
    #: In certain cases, this can be achieved by computing block probabilities
    #: and adjusting the rewards of the lineage-counting state space accordingly.
    #: This can substantially speed up computations.
    flatten_block_counting: bool = True

    #: Whether to show a progress bar for certain operations such as state space generation.
    use_pbar: bool = False

    #: Whether to parallelize phase-type computations across multiple CPU cores.
    #: This may improve performance in some cases, but can also be detrimental due to
    #: inter-process data copying and can lead to hanging processes.
    parallelize: bool = False

    #: Whether to regularize the intensity matrix for numerical stability.
    regularize: bool = True

    #: Whether to cache the rate matrix for different epochs which increases performance.
    cache_epochs: bool = True
