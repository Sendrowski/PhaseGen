"""
Site-frequency spectra and their higher-dimensional generalizations (the joint multi-population SFS and the
two-locus SFS).

The container classes are defined in :mod:`sfsutils` and re-exported here so that ``phasegen`` code, and jsonpickle
fixtures serialized against this module path, can reach them.
"""

import logging

# noinspection PyUnresolvedReferences
from sfsutils import AbstractSpectrum, Spectrum, Spectra, TwoSFS, TwoLocusSFS, JointSFS  # noqa: F401

logger = logging.getLogger('phasegen').getChild('spectrum')


class SFS(Spectrum):
    """
    A site-frequency spectrum.
    """
    pass
