"""
Site-frequency spectra and their higher-dimensional generalizations (the joint multi-population SFS and the
two-locus SFS).

The container classes now live in :mod:`sfsutils` and are re-exported here so that ``phasegen`` code, and jsonpickle
fixtures serialized against this module path, can reach them.
"""

import logging

# noinspection PyUnresolvedReferences
from sfsutils import Spectrum, Spectra, TwoSFS, TwoLocusSFS, JointSFS  # noqa: F401

logger = logging.getLogger('phasegen').getChild('spectrum')


class SFS(Spectrum):
    """
    A site-frequency spectrum.
    """
    pass


#: Deserialization alias so jsonpickle can resolve ``phasegen.spectrum.SFS2`` in fixtures serialized before the class
#: was renamed to :class:`TwoSFS` and moved to :mod:`sfsutils`. Not part of the public API.
SFS2 = TwoSFS
