"""
Root conftest, imported before the ``testing`` package pulls in numpy -- which is the only point at which the thread
limits below can still take effect, since BLAS reads them when its shared library loads.

Under ``pytest -n auto`` each xdist worker is its own process, and each would otherwise let numpy/BLAS and numba open
a thread pool sized to the whole machine: ten workers times ~3 threads on ten cores is a threefold oversubscription,
and the suite spends its time in contention rather than in work. Pinning every worker to one thread cut the suite from
1290s to 293s.

This is *almost* always invisible: single- vs multi-threaded BLAS reductions differ only in summation order, i.e. at
the last bit. But it is not strictly "no change to what is computed". An ill-conditioned inversion sitting exactly on
a stability threshold can be tipped across it by that last bit: the Fourier-cosine SFS-bin pdf of an extreme
demography (e.g. ``2_epoch_rapid_decline_n_5``, bin 3) has a marginally non-monotone cosine CDF, and the last bit
decides whether the monotonicity clamp fires -- a smooth pdf (total variation ~0.002) versus a spurious spike from the
clamp kink (~0.28). So a tolerance is really tuned against whatever thread count evaluated it; ``tune_dist_tols`` runs
unpinned and will disagree with the pinned suite on such a leaf unless it is pinned the same way. Acceptable for the
speed it buys, but a known caveat -- not ideal for a test harness. Set the variables yourself to override
(``setdefault``).
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
