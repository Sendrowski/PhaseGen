.. _modules.spectrum:

SFS classes
-----------

``phasegen`` expresses its site-frequency spectra through the container classes of the standalone :mod:`sfsutils` package,
on which ``phasegen`` depends and which it re-exports, so they remain importable directly from ``phasegen`` (e.g.
``from phasegen import SFS, TwoSFS, TwoLocusSFS, JointSFS, Spectra``). Their full API reference is hosted in the
:mod:`sfsutils` documentation:

- :class:`~sfsutils.spectrum.Spectrum` — a single site-frequency spectrum, with folding, polarising, resampling, and plotting. ``phasegen`` returns single-population spectra as ``SFS``, a thin subclass with identical behaviour.
- :class:`~sfsutils.spectrum.Spectra` — a named collection of spectra supporting grouped operations and joint visualisation.
- :class:`~sfsutils.spectrum.TwoSFS` — the two-dimensional second-moment (2-SFS) of a single population's site-frequency spectrum.
- :class:`~sfsutils.spectrum.JointSFS` — the joint site-frequency spectrum across populations.
- :class:`~sfsutils.spectrum.TwoLocusSFS` — the two-locus site-frequency spectrum under recombination.
