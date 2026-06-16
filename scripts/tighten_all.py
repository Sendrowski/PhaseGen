"""Tighten-only re-tune of many configs in parallel (never loosens a tolerance; clamps each matched leaf to
``min(current, 1.5x observed)``). Writes each config YAML in place; sync to the fixtures afterwards with the
``update_tolerances`` snakemake rule. Usage: ``python scripts/tighten_all.py <config> [<config> ...]``."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _init_worker():
    for v in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'


def _tighten(name: str):
    import warnings
    warnings.filterwarnings('ignore')
    import importlib.util
    spec = importlib.util.spec_from_file_location("tdt", os.path.join(os.path.dirname(__file__), "tune_dist_tols.py"))
    tdt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdt)
    try:
        n = tdt.retune(name, tighten_only=True)
        return name, n, None
    except Exception as e:  # keep the batch going; report the failure
        return name, 0, repr(e)


if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if not a.startswith('--')]
    jobs = int(os.environ.get('TIGHTEN_JOBS', os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        for name, n, err in ex.map(_tighten, names):
            print(f"{'ERR ' if err else ''}tightened {n:>3} in {name}" + (f"  :: {err}" if err else ""), flush=True)
