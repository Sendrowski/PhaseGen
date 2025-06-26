"""
Settings for the PhaseGen application.
"""


class Settings:
    #: Whether to flatten the block-counting state space when possible.
    #: In certain cases, this can be achieved by computing block probabilities
    #: and adjusting the rewards of the lineage-counting state space accordingly.
    #: This can substantially speed up computations.
    flatten_block_counting: bool = True
