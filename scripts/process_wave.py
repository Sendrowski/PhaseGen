"""
Drive the full dist-stat pipeline for a wave of (already augmented) scenario configs:
  regenerate fixture -> tune tolerances to ~1.5x observed -> sync into fixture -> verify (assertions pass).

    python scripts/process_wave.py [--regen] [--jobs N] [--tune-jobs N] <config_name> ...

``--regen`` forces fixture re-simulation (needed when dist stats were just added); omit it to only re-tune/sync/verify
against an existing fixture. ``--jobs`` is the snakemake parallelism for the (regen) simulation phase; ``--tune-jobs``
the number of configs tuned/verified concurrently (the tune+verify of different configs are independent and CPU-bound,
so they run across cores -- defaults to the core count). Prints a per-config PASS/FAIL summary at the end."""
import concurrent.futures as cf
import os
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

SNAKE = "/Users/janek/miniforge3/bin/snakemake"
SER = "results/comparisons/serialized"


def snake(targets, jobs=1):
    # --nolock so a wave can run concurrently with another snakemake invocation in the same workdir (the targets
    # are disjoint per config, so there is no actual file contention)
    cmd = [SNAKE, "--use-conda", "--nolock", "--scheduler", "greedy", "-j", str(jobs), "-f", *targets]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def _init_worker():
    """Pin each tune worker to a single thread so N concurrent configs do not oversubscribe the cores (each config's
    analytic comparison is single-process; numba/BLAS would otherwise each spin up their own thread pool)."""
    for var in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, '1')


def _tune_one(n: str) -> tuple:
    """Tune tolerances -> sync into the fixture -> verify, for one config. Runs in a worker process (the configs are
    independent and CPU-bound, so the wave's tune/verify phase parallelises across cores)."""
    sys.path.insert(0, "scripts")
    import importlib
    tune = importlib.import_module("tune_dist_tols")
    from phasegen.comparison import Comparison
    try:
        k = tune.retune(n)
    except Exception as e:
        return n, f"TUNE-ERR {type(e).__name__}: {e}"
    # sync yaml tolerances into the fixture (cheap re-embed, no re-simulation)
    subprocess.run(["rm", "-f", f"{SER}/.{n}.tolerances_synced"])
    snake([f"{SER}/.{n}.tolerances_synced"], 1)
    # verify
    try:
        c = Comparison.from_file(f"{SER}/{n}.json")
        c.do_assertion = True
        c.visualize = False
        c.compare(n)
        return n, f"PASS ({c.n_assertions} asserts, tuned {k})"
    except AssertionError as e:
        return n, f"FAIL {str(e)[:80]}"
    except Exception as e:
        return n, f"VERIFY-ERR {type(e).__name__}: {str(e)[:80]}"


def main(argv):
    regen = "--regen" in argv
    argv = [a for a in argv if a != "--regen"]
    jobs = 4
    tune_jobs = None
    for flag in ("--jobs", "--tune-jobs"):
        if flag in argv:
            i = argv.index(flag); val = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
            if flag == "--jobs":
                jobs = val
            else:
                tune_jobs = val
    names = argv
    if tune_jobs is None:
        tune_jobs = min(len(names), os.cpu_count() or 4)

    if regen:
        print(f"[regen] {len(names)} fixtures (jobs={jobs}) ...", flush=True)
        rc = snake([f"{SER}/{n}.json" for n in names], jobs)
        print(f"[regen] snakemake rc={rc}", flush=True)

    # tune + verify each config in parallel across cores (independent, CPU-bound). NB: do NOT globally disable logging
    # -- the tuner reads the comparison's INFO log lines to learn the observed diffs (per-process, so no cross-talk).
    print(f"[tune] {len(names)} configs (tune_jobs={tune_jobs}) ...", flush=True)
    results = {}
    with cf.ProcessPoolExecutor(max_workers=tune_jobs, initializer=_init_worker) as ex:
        for n, r in ex.map(_tune_one, names):
            results[n] = r
            print(f"  {n}: {r}", flush=True)

    print("\n===== WAVE SUMMARY =====")
    for n, r in results.items():
        print(f"  {'OK ' if r.startswith('PASS') else 'XX '} {n}: {r}")
    bad = [n for n, r in results.items() if not r.startswith('PASS')]
    print(f"{len(names) - len(bad)}/{len(names)} passed" + (f"; FAILED: {bad}" if bad else ""))


if __name__ == '__main__':
    sys.path.insert(0, "scripts")
    main(sys.argv[1:])
