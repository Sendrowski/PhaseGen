"""
Settings for the PhaseGen application.
"""
from contextlib import contextmanager


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

    @staticmethod
    @contextmanager
    def set_pbar(enabled: bool = True):
        """
        Context manager to temporarily enable or disable the progress bar.
        """
        prev = Settings.use_pbar
        Settings.use_pbar = enabled
        try:
            yield
        finally:
            Settings.use_pbar = prev