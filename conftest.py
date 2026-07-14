"""
Root conftest, imported before the ``testing`` package pulls in numpy -- which is the only point at which the thread
limits below can still take effect, since BLAS reads them when its shared library loads.

Under ``pytest -n auto`` each xdist worker is its own process, and each would otherwise let numpy/BLAS and numba open
a thread pool sized to the whole machine: ten workers times ~3 threads on ten cores is a threefold oversubscription,
and the suite spends its time in contention rather than in work. Pinning every worker to one thread cut the suite from
1290s to 293s with no change to what is computed. Set the variables yourself to override (``setdefault``).
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
